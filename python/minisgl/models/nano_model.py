"""NanoForCausalLM (T4c): C1 weight contract -> minisgl primitives.

Structurally identical to Qwen3ForCausalLM (RopeAttn has_qk_norm + GatedMLP +
tie_word_embeddings) because C1 keys are HF Qwen3 style — the difference is the
C3 kv_org declaration on the class (qwen3.py is kept unmodified as a reference).
Weight naming: layer attrs mirror the C1 table (qkv_proj/gate_up_proj consume
weight.py merge groups; q_norm/k_norm per-head RMSNorm).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as NanoMLP
from .utils import RopeAttn as NanoAttn

if TYPE_CHECKING:
    from .config import ModelConfig


@dataclass(frozen=True)
class KVOrg:
    """C3 KVOrganization engine-side declaration (pools / per_token_bytes / layout)."""

    pools: int
    per_token_bytes: Callable[..., int]
    layout: str


class NanoDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = NanoAttn(config, layer_id, has_qk_norm=True)
        self.mlp = NanoMLP(config)
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class NanoModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [NanoDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class NanoForCausalLM(BaseLLMModel):
    # C3: single GQA pool, token-granular pages (page_size=1), K/V bf16
    kv_org = KVOrg(
        pools=1,
        per_token_bytes=lambda cfg: 2 * cfg.num_kv_heads * cfg.head_dim * cfg.dtype.itemsize,
        layout="paged_kt_vt",
    )

    def __init__(self, config: ModelConfig):
        self.model = NanoModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["NanoForCausalLM", "KVOrg"]
