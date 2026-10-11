from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .utils import deserialize_type, serialize_type


@dataclass
class BaseFrontendMsg:
    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    data: List[BaseFrontendMsg]


@dataclass
class UserReply(BaseFrontendMsg):
    uid: int
    incremental_output: str
    finished: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class GaugeReply(BaseFrontendMsg):
    """Frontend-side gauge snapshot, forwarded by the tokenizer process (t4 §3 T4d)."""

    queue_depth: int
    running_batch: int
    kv_utilization: float
