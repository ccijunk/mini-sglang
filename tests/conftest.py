"""T4c · pytest fixtures。C1 fixture 生成器在 c1_fixture.py（T3 原样移植，单一事实源）。

Engine 驱动：minisgl 基座内建 offline_mode（scheduler/io.py 早退分支，
offline_receive_msg/offline_send_result 两钩子）——TestScheduler 子类注入内存 IO，
scheduler/engine/radix 全部走真实路径（C5 R3 挂载 live 代码）。
"""

import pytest
import torch

from c1_fixture import gen_weights, write_model_dir


@pytest.fixture(scope="session")
def c1_weights():
    return gen_weights()


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    return write_model_dir(str(tmp_path_factory.mktemp("nano_model")))


@pytest.fixture(scope="session")
def scheduler(model_dir):
    from minisgl.distributed import DistributedInfo
    from minisgl.scheduler.config import SchedulerConfig
    from minisgl.scheduler.scheduler import Scheduler

    class TestScheduler(Scheduler):
        # offline hooks (base raises NotImplementedError — see scheduler/io.py)
        def offline_receive_msg(self, blocking: bool = False):
            msgs, self.inbox = self.inbox, []
            return msgs

        def offline_send_result(self, reply):
            self.results.extend(reply)

    cfg = SchedulerConfig(
        model_path=model_dir,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        # fi's prefill templates reject head_dim=32 — flashinfer fa2 batch prefill
        # requires NUM_MMA_D_VO = head_dim/16 >= 4, else IsInvalid() raises
        # "Invalid configuration" at the first prefill plan (flashinfer 0.7.0.post1
        # prefill.cuh:417,2899; verified in stack issue #23, t4 design §6 item 15).
        # sdpa = reference backend
        attention_backend="sdpa",
        cuda_graph_bs=[],  # eager: determinism over speed (T3 enforce_eager analog)
        # GPU shared with the T3 nano-vllm pod (VLLM::EngineCore ~3.4GB of 4GB):
        # memory_ratio heuristic is unreliable in this fragmented environment —
        # pin a fixed page budget instead (200k pages * 512B/token/layer == ~100MB)
        num_page_override=200_000,
        max_running_req=8,
        offline_mode=True,
    )
    sched = TestScheduler(cfg)
    sched.inbox = []
    sched.results = []
    yield sched
    sched.shutdown()
