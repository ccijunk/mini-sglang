"""T4c · LV1-b 引擎验证（t4_mini_sglang.md §4）：E-B1..B4。

运行（C5 契约命令，容器内，GPU 必要）：
docker run --rm --gpus all --network none -v "$PWD":/w -w /w/mini-sglang \
  -e PYTHONPATH=/w/mini-sglang/python \
  -v "$PWD/.cache":/opt/cache -e TVM_FFI_CACHE_DIR=/opt/cache/tvm-ffi \
  -e FLASHINFER_WORKSPACE_BASE=/opt/cache/flashinfer \
  nano-stack/engine:sglang python3 -m pytest tests -q

E-B2 参考实现 = tests/reference_model.py（T3 原样移植：跨引擎同一 naive GQA fp32
参考 = 一致性对照锚）。噪声对策同 T3 教训：bf16/批量形状噪声只可能翻转 near-tie 的
argmax，decisive 步（top1-top2 gap ≥ 5e-2）强制与 fp32 参考一致（teacher-forced）。
"""

import pytest
import torch

from minisgl.core import SamplingParams
from minisgl.message import UserMsg
from minisgl.models.nano_model import NanoForCausalLM

from reference_model import CFG
from reference_model import forward as ref_forward

GAP = 5e-2  # decisive 步阈值：低于此值视为 bf16 噪声区，不判 argmax


def _encode(sched, text: str) -> torch.Tensor:
    return sched.tokenizer.encode(text, return_tensors="pt").view(-1).to(torch.int32)


def run_prompts(sched, prompts: list[str], max_tokens: int = 16) -> list[list[int]]:
    """经真实 prefill/decode 调度循环批量跑 prompts（greedy），按 uid 回收 token 流。"""
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)
    sched.inbox += [
        UserMsg(uid=i, input_ids=_encode(sched, p), sampling_params=sp)
        for i, p in enumerate(prompts)
    ]
    out: list[list[int]] = [[] for _ in prompts]
    finished: set[int] = set()
    # mirror Scheduler.run_forever's DISABLE_OVERLAP branch (scheduler.py:121-125):
    # normal_loop does not wrap forward in the engine stream itself
    with torch.inference_mode(), sched.engine_stream_ctx:
        sched.engine.stream.wait_stream(sched.stream)
        while len(finished) < len(prompts):
            sched.normal_loop()
            for msg in sched.results:
                if msg.uid not in finished:
                    out[msg.uid].append(msg.next_token)
                    if msg.finished:
                        finished.add(msg.uid)
            sched.results.clear()
    assert all(len(t) == max_tokens for t in out), "every request must reach max_tokens (no EOS)"
    return out


def _ref_top1_gap(fp32, ctx):
    logits = ref_forward(fp32, ctx, CFG)
    top2 = torch.topk(logits, 2)
    return top2.indices[0].item(), (top2.values[0] - top2.values[1]).item()


def _teacher_force(fp32, ctx: list[int], stream: list[int], checked: list[int]):
    """从公共前缀 ctx 出发，逐步校验 stream 中每个 decisive 步的 argmax。"""
    ctx = list(ctx)
    for tid in stream:
        top1, gap = _ref_top1_gap(fp32, ctx)
        if gap > GAP:
            assert top1 == tid
            checked[0] += 1
        ctx.append(tid)


def test_e_b1_weight_load_bitwise(scheduler, c1_weights):
    _, bf16 = c1_weights
    params = scheduler.engine.model.state_dict()  # BaseOP tree protocol (not nn.Module)
    H, KVH, D, I = CFG["n_heads"], CFG["n_kv_heads"], CFG["head_dim"], CFG["intermediate_size"]
    for key, w in bf16.items():
        if "q_proj" in key:
            pname, sl = key.replace("q_proj", "qkv_proj"), slice(0, H * D)
        elif "k_proj" in key:
            pname, sl = key.replace("k_proj", "qkv_proj"), slice(H * D, (H + KVH) * D)
        elif "v_proj" in key:
            pname, sl = key.replace("v_proj", "qkv_proj"), slice((H + KVH) * D, (H + 2 * KVH) * D)
        elif "gate_proj" in key:
            pname, sl = key.replace("gate_proj", "gate_up_proj"), slice(0, I)
        elif "up_proj" in key:
            pname, sl = key.replace("up_proj", "gate_up_proj"), slice(I, 2 * I)
        else:
            pname, sl = key, None
        assert pname in params, f"missing param {pname}"
        p = params[pname].detach().cpu()
        assert torch.equal(p[sl] if sl is not None else p, w), key


def test_e_b2_greedy_batch_serial_reference(scheduler, c1_weights):
    fp32, _ = c1_weights
    prompts = ["!#%&", "ABCDEFGH", "0123456789()", "*+-/:;<=>?@"]  # 全部落在 42 字符表内
    batch = run_prompts(scheduler, prompts, max_tokens=16)
    serial = []
    for p in prompts:
        serial.append(run_prompts(scheduler, [p], max_tokens=16)[0])

    checked, total, diverged = [0], 0, 0
    for p, bt, st in zip(prompts, batch, serial):
        ctx = scheduler.tokenizer.encode(p)
        for i, (x, y) in enumerate(zip(bt, st)):
            total += 1
            top1, gap = _ref_top1_gap(fp32, ctx)
            if x == y:
                if gap > GAP:  # decisive 步：与 fp32 参考强制一致
                    assert top1 == x
                    checked[0] += 1
                ctx.append(x)
            else:  # 分叉只允许在 near-tie 步
                diverged += 1
                assert gap < GAP, f"batch/serial diverged at decisive step {i} of {p!r}"
                _teacher_force(fp32, ctx, bt[i:], checked)
                _teacher_force(fp32, list(ctx), st[i:], checked)
                break
    assert checked[0] / total > 0.8, f"decisive coverage too low: {checked[0]}/{total}, diverged={diverged}"


def test_e_b3_kv_accounting(scheduler):
    eng = scheduler.engine
    kv_org = NanoForCausalLM.kv_org
    assert kv_org.pools == 1 and kv_org.layout == "paged_kt_vt"
    # C3 账面：2(K+V) × n_kv × head_dim × 2B = 256 B/token/层（C1 尺寸）
    ns = type("Cfg", (), {"num_kv_heads": CFG["n_kv_heads"], "head_dim": CFG["head_dim"], "dtype": torch.bfloat16})
    assert kv_org.per_token_bytes(ns()) == 2 * 2 * 32 * torch.bfloat16.itemsize == 256
    buf = eng.kv_cache._kv_buffer  # (2, n_layers, num_pages(+1 dummy), page_size, kvh, head_dim)
    assert buf.dtype == torch.bfloat16
    two, L, P, ps, KVH, D = buf.shape
    assert (two, ps, KVH, D) == (2, 1, CFG["n_kv_heads"], CFG["head_dim"])
    assert L == CFG["n_layers"]
    # 全 buffer 账面（含 engine.py:59 的 +1 dummy page）：K+V 双侧 == per_token × 层 × 页
    assert buf.numel() * buf.element_size() == 256 * L * P


def test_e_b4_prefix_cache(scheduler):
    X = "ABCDEFGH" * 32          # 256 共享前缀字符
    p_a, p_b = X + "0123456789", X + "*+-/:;<=>?@"  # tails 首字符不同（0 vs *）
    p_c = X + "!#%&"            # 从未跑过：探针（只 match 不 insert）

    out_a_cold = run_prompts(scheduler, [p_a], max_tokens=1)[0]
    out_b_cold = run_prompts(scheduler, [p_b], max_tokens=1)[0]

    # pA 完成后：pC（仅共享 X）的 radix 命中长度 == 256
    from minisgl.core import SamplingParams
    from minisgl.scheduler.utils import PendingReq

    probe = PendingReq(
        uid=999,
        input_ids=_encode(scheduler, p_c),
        sampling_params=SamplingParams(max_tokens=1),
    )
    result = scheduler.cache_manager.match_req(probe)
    assert result.cuda_handle.cached_len == 256, "radix match must hit the shared 256-char prefix"

    # 缓存路径输出 == 非缓存路径（greedy 确定性：逐 token 相等）
    out_a_warm = run_prompts(scheduler, [p_a], max_tokens=1)[0]
    out_b_warm = run_prompts(scheduler, [p_b], max_tokens=1)[0]
    assert out_a_warm == out_a_cold and out_b_warm == out_b_cold
    scheduler.cache_manager.check_integrity()
