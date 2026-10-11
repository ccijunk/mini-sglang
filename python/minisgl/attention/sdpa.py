from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .fa import FAMetadata, FlashAttentionBackend

if TYPE_CHECKING:
    from minisgl.core import Batch
    from minisgl.models import ModelConfig


class SdpaBackend(FlashAttentionBackend):
    """Pure-PyTorch SDPA attention backend (t4_mini_sglang.md §6.12).

    Inherits the fa backend's metadata pipeline verbatim (FAMetadata is
    backend-agnostic); only the kernel call is swapped. Exists because the
    flashinfer prefill templates reject head_dim=32 (prefill.cuh:4783
    "Invalid configuration" — no matching template for this MMA layout on
    sm86), while the nano C1 model has head_dim 32. Engine B's mandate is
    cross-engine correctness, not performance; fi remains the backend for
    real workloads (head_dim >= 64).
    """

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        metadata = batch.attn_metadata
        assert isinstance(metadata, FAMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        return _sdpa_impl(
            q=q,
            k_cache=self.kvcache.k_cache(layer_id),
            v_cache=self.kvcache.v_cache(layer_id),
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            softmax_scale=self.scale,
        )


def _sdpa_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    # q: (total_q, H, D) ragged; k/v cache: (P, page_size, KVH, D) token-major
    P, ps, KVH, D = k_cache.shape
    assert ps == 1, "SdpaBackend assumes token-granular pages (page_size=1)"
    H = q.shape[1]
    g = H // KVH  # GQA group size
    qlens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
    outs: list[torch.Tensor] = []
    offset = 0
    for i, qlen in enumerate(qlens):
        qi = q[offset : offset + qlen]  # (qlen, H, D)
        offset += qlen
        klen = int(cache_seqlens[i])
        slots = page_table[i, :klen].long()  # page_size=1: page index == token slot
        ki = k_cache[slots, 0].repeat_interleave(g, dim=1)  # (klen, H, D)
        vi = v_cache[slots, 0].repeat_interleave(g, dim=1)
        # bottom-right causal alignment: query row j (global pos klen-qlen+j)
        # attends keys [0, klen-qlen+j]; for decode (qlen=1) the mask is all-true
        mask = torch.arange(klen, device=q.device)[None, :] <= (
            klen - qlen + torch.arange(qlen, device=q.device)[:, None]
        )
        out = F.scaled_dot_product_attention(
            qi.transpose(0, 1).unsqueeze(0),
            ki.transpose(0, 1).unsqueeze(0),
            vi.transpose(0, 1).unsqueeze(0),
            attn_mask=mask.unsqueeze(0),
            scale=softmax_scale,
        )
        outs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0)
