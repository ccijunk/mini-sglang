"""T4d · C2 server 契约验证（E-C1/E-C2/E-C3，t4 §3；契约文本 = doc/contracts/server_api.md v1.2）。

真实多进程链路：`python -m minisgl` 子进程（uvicorn frontend + tokenizer/detokenizer 进程 +
scheduler 进程，ZMQ ipc 互联），HTTP 断言走容器 loopback。判据对照契约，不对照实现。

启动参数约束（t4 §6.12/§6.13）：
- sdpa 后端：fi prefill 模板拒绝 head_dim=32
- --cuda-graph-max-bs 0：graph.py:64-65 `<1 -> []` -> eager（sdpa 的 tolist()/int()
  CPU 读取不 capture-safe，T3 enforce_eager 同构）
- --num-pages 200000：与 T4c conftest 同参——固定页预算，不依赖 memory_ratio 启发式
- fixture 模型目录名 = "nano-model"：model_name = basename(model_path)（C2 §models）
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest

from c1_fixture import write_model_dir

MAX_RUNNING = 8
STARTUP_TIMEOUT = 240  # 冷启动含 flashinfer JIT（.cache 单例挂载后走缓存）

# C2 §metrics v1.2：3 counter + 2 histogram + 3 gauge，基础标签 engine + model
CONTRACT_METRICS = [
    "nano_request_total",
    "nano_request_prompt_tokens",
    "nano_request_completion_tokens",
    "nano_ttft_seconds",
    "nano_tpot_seconds",
    "nano_queue_depth",
    "nano_running_batch",
    "nano_kv_cache_utilization",
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _request(method: str, url: str, body: dict | None = None, timeout: float = 180):
    """返回 (status, parsed)；非 JSON 响应（/metrics）原样返回文本。"""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, raw = resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:  # 4xx/5xx body 仍是契约错误体
        status, raw = e.code, e.read().decode()
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


def _fail_with_log(log_path: str, msg: str):
    with open(log_path, encoding="utf-8", errors="replace") as f:
        tail = f.read()[-4000:]
    pytest.fail(f"{msg}\n---- server log tail ----\n{tail}")


@pytest.fixture(scope="session")
def base_url(tmp_path_factory):
    # LV2 (C5 R5): assert Job targets the Service — skip the local server spawn.
    # The Job is applied together with the engine Deployment, so poll /health
    # instead of racing the engine's first ready probe (backoffLimit 0: one
    # connection-refused would fail the whole assertion run).
    remote = os.environ.get("NANO_BASE_URL")
    if remote:
        base = remote.rstrip("/")
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while True:
            try:
                status, body = _request("GET", f"{base}/health", timeout=2)
                if status == 200 and body == {"status": "ok"}:
                    break
            except (urllib.error.URLError, TimeoutError, OSError):
                pass  # engine pod not ready yet
            if time.monotonic() >= deadline:
                pytest.fail(f"remote engine not healthy within {STARTUP_TIMEOUT}s: {base}")
            time.sleep(1)
        yield base
        return

    model_dir = str(tmp_path_factory.mktemp("t4d") / "nano-model")
    write_model_dir(model_dir)  # C1 fixture（T4c 单一事实源，幂等）
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    log_dir = os.environ.get("T4D_LOG_DIR", os.path.dirname(model_dir))
    log_path = os.path.join(log_dir, "server.log")
    env = dict(os.environ, PYTHONFAULTHANDLER="1", TOKENIZERS_PARALLELISM="false")
    cmd = [
        sys.executable, "-m", "minisgl",
        "--model-path", model_dir,
        "--host", "127.0.0.1", "--port", str(port),
        "--attention-backend", "sdpa",
        "--cuda-graph-max-bs", "0",
        "--num-pages", "200000",
        "--max-running-requests", str(MAX_RUNNING),
    ]
    log = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, env=env,  # setsid -> killpg 覆盖全部子进程
    )
    try:
        healthy = False
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                log.flush()
                _fail_with_log(log_path, f"server exited early (rc={proc.returncode})")
            try:
                status, body = _request("GET", f"{base}/health", timeout=2)
                if status == 200 and body == {"status": "ok"}:
                    healthy = True
                    break
            except (urllib.error.URLError, TimeoutError, OSError):
                pass  # not up yet
            time.sleep(1)
        if not healthy:
            _fail_with_log(log_path, f"server not healthy within {STARTUP_TIMEOUT}s")
        yield base
    finally:
        rc = proc.poll()
        if rc is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        else:
            # server died mid-session — surface its log instead of masking the cause
            log.flush()
            _fail_with_log(log_path, f"server exited during session (rc={rc})")
        log.close()


# ---------------------------------------------------------------------------
# E-C1: C2 合规矩阵（端点包络 + 错误语义）
# ---------------------------------------------------------------------------


def test_c2_health_and_models(base_url):
    status, body = _request("GET", f"{base_url}/health")
    assert status == 200 and body == {"status": "ok"}

    status, body = _request("GET", f"{base_url}/v1/models")
    assert status == 200
    assert body["object"] == "list"
    card = body["data"][0]
    assert card["id"] == "nano-model"
    assert card["object"] == "model"
    assert card["owned_by"] == "nano-ai-infra-stack"


def test_c2_completion_envelope(base_url):
    prompt = "!!ABC0123"  # 全部落在词表 chr(33..74)
    body = {"model": "nano-model", "prompt": prompt, "max_tokens": 8, "temperature": 0.0}
    status, resp = _request("POST", f"{base_url}/v1/completions", body)
    assert status == 200
    assert resp["id"].startswith("cmpl-")
    assert resp["object"] == "text_completion"
    assert isinstance(resp["created"], int)
    assert resp["model"] == "nano-model"
    choice = resp["choices"][0]
    assert choice["index"] == 0
    assert len(choice["text"]) == 8  # char-level：1 token == 1 char
    # char 词表无 EOS（t4 §6.4）-> "length" 是唯一可达 finish 分支
    assert choice["finish_reason"] == "length"
    assert resp["usage"] == {
        "prompt_tokens": len(prompt),
        "completion_tokens": 8,
        "total_tokens": len(prompt) + 8,
    }
    # greedy（argmax 路径，与 RNG 无关）：同请求两次 -> 输出 bitwise 一致
    _, resp2 = _request("POST", f"{base_url}/v1/completions", body)
    assert resp2["choices"][0]["text"] == choice["text"]


def test_c2_error_matrix(base_url):
    # 404 model_not_found + 错误体四字段形状
    status, err = _request("POST", f"{base_url}/v1/completions",
                           {"model": "nope", "prompt": "!"})
    assert status == 404
    assert err["error"]["code"] == "model_not_found"
    assert set(err["error"]) == {"message", "type", "param", "code"}

    # 400 未知字段（extra="forbid"，C2 子集语义）
    status, _ = _request("POST", f"{base_url}/v1/completions",
                         {"model": "nano-model", "prompt": "!", "n": 2})
    assert status == 400

    # 400 字段约束（max_tokens ge=1）
    status, _ = _request("POST", f"{base_url}/v1/completions",
                         {"model": "nano-model", "prompt": "!", "max_tokens": 0})
    assert status == 400

    # 400 词表外字符（vocab = chr(33..74)；'a' = chr(97)）
    status, err = _request("POST", f"{base_url}/v1/completions",
                           {"model": "nano-model", "prompt": "abc", "max_tokens": 1})
    assert status == 400 and err["error"]["param"] == "prompt"

    # 422 context_length_exceeded：max_position_embeddings=1024（C1 CFG）
    status, err = _request("POST", f"{base_url}/v1/completions",
                           {"model": "nano-model", "prompt": "!" * 1000, "max_tokens": 256})
    assert status == 422
    assert err["error"]["code"] == "context_length_exceeded"

    # 边界：768 + 256 == 1024 恰好放行（严格大于才触发 422）
    status, resp = _request("POST", f"{base_url}/v1/completions",
                            {"model": "nano-model", "prompt": "!" * 768,
                             "max_tokens": 256, "temperature": 0.0})
    assert status == 200
    assert resp["usage"]["total_tokens"] == 1024


def test_c2_503_overload(base_url):
    body = {"model": "nano-model", "prompt": "!" * 16, "max_tokens": 256, "temperature": 0.0}

    def fire(_):
        return _request("POST", f"{base_url}/v1/completions", body)

    with ThreadPoolExecutor(max_workers=9) as pool:
        results = list(pool.map(fire, range(9)))

    statuses = [s for s, _ in results]
    # 8 准入（max_running_req=8）+ 1 门口拒绝（C2 §503 admission control）
    assert statuses.count(503) == 1, statuses
    assert statuses.count(200) == 8, statuses
    _, err = next(r for r in results if r[0] == 503)
    assert err["error"]["code"] == "engine_overloaded"
    assert err["error"]["type"] == "server_error"


# ---------------------------------------------------------------------------
# E-D1: 并发吞吐 > 串行（PLAN 验证②）—— batch decode 摊销 forward 次数
# ---------------------------------------------------------------------------


def test_d1_concurrent_throughput(base_url):
    body = {"model": "nano-model", "prompt": "!", "max_tokens": 64, "temperature": 0.0}
    fire = lambda: _request("POST", f"{base_url}/v1/completions", body)

    t0 = time.monotonic()
    for _ in range(8):
        status, _ = fire()
        assert status == 200
    t_serial = time.monotonic() - t0

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: fire(), range(8)))
    t_parallel = time.monotonic() - t0
    assert all(s == 200 for s, _ in results)

    # 8 requests in one decode batch = 64 forward steps vs 512 serial steps
    assert t_parallel < t_serial, (t_serial, t_parallel)


# ---------------------------------------------------------------------------
# E-C2: seed 进程级准入 -> flashinfer 采样 bitwise 复现（temperature=1.0 真采样路径）
# ---------------------------------------------------------------------------


def test_c2_seed_determinism(base_url):
    body = {"model": "nano-model", "prompt": "!#$%&'()*+,-./", "max_tokens": 32,
            "temperature": 1.0, "seed": 42}
    _, r1 = _request("POST", f"{base_url}/v1/completions", body)
    _, r2 = _request("POST", f"{base_url}/v1/completions", body)
    assert r1["usage"]["completion_tokens"] == 32
    assert r1["choices"][0]["text"] == r2["choices"][0]["text"], \
        "seeded sampling must be bitwise reproducible (E-C2)"


# ---------------------------------------------------------------------------
# E-C3: /metrics — 指标名/标签为契约，数值语义实证
# ---------------------------------------------------------------------------


def test_c3_metrics(base_url):
    # 先打一笔流量，保证 counter 非零（不依赖测试顺序）
    status, _ = _request("POST", f"{base_url}/v1/completions",
                         {"model": "nano-model", "prompt": "!",
                          "max_tokens": 4, "temperature": 0.0})
    assert status == 200

    status, text = _request("GET", f"{base_url}/metrics")
    assert status == 200
    lines = text.splitlines()
    for name in CONTRACT_METRICS:
        assert any(l.startswith(name) for l in lines), f"missing metric family {name}"

    # C2 基础标签 engine + model 必须出现在序列上
    assert 'engine="mini-sglang",model="nano-model"' in text

    # request counter 记账（status="ok" 序列 >= 1）
    ok_line = next(l for l in lines
                   if l.startswith("nano_request_total{") and 'status="ok"' in l)
    assert float(ok_line.rsplit("}", 1)[1].strip()) >= 1

    # histogram 观测落地
    assert any(l.startswith("nano_ttft_seconds_count") for l in lines)
    assert any(l.startswith("nano_tpot_seconds_count") for l in lines)

    # gauge：请求结束后 radix 缓存持页（evictable）-> 占用率 > 0（T3/T5 语义）
    kv_line = next(l for l in lines if l.startswith("nano_kv_cache_utilization{"))
    kv = float(kv_line.rsplit("}", 1)[1].strip())
    assert 0.0 < kv <= 1.0, kv
