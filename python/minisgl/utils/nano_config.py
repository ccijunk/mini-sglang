"""Nano sidecar config shim (T4c): NanoConfig field names -> HF-style attrs.

`ModelConfig.from_hf` is getattr-driven (models/config.py), so a SimpleNamespace
carrying HF attribute names is sufficient — no PretrainedConfig subclass needed.
Seam: utils/hf.cached_load_hf_config dispatches on config.json model_type == "nano"
(single choke point covering EngineConfig.hf_config AND models.weight.load_weight).
"""

import json
import os
from types import SimpleNamespace

# NanoConfig field (C1 sidecar) -> HF attribute name. Keys absent from the C1
# sidecar (hidden_act) get a declared default; the nano model uses SiLU gating.
_FIELD_MAP = {
    "n_layers": "num_hidden_layers",
    "n_heads": "num_attention_heads",
    "n_kv_heads": "num_key_value_heads",
}


def is_nano_model(model_path: str) -> bool:
    """True iff model_path/config.json declares model_type == "nano"."""
    try:
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return raw.get("model_type") == "nano"


def load_nano_hf_config(model_path: str) -> SimpleNamespace:
    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    assert raw.get("model_type") == "nano", "not a nano sidecar config"
    kwargs = {dest: raw[src] for src, dest in _FIELD_MAP.items() if src in raw}
    # C1 sidecar has no hidden_act; the nano model is gated-SiLU (declared, t4 §6)
    return SimpleNamespace(
        architectures=["NanoForCausalLM"],
        model_type="nano",
        rope_scaling=None,
        hidden_act="silu",
        # C1 sidecar has no dtype; fixture weights are saved bf16 (c1_fixture) —
        # args.py "auto" resolution needs the string (declared, t4 §6)
        dtype=raw.get("dtype", "bfloat16"),
        tie_word_embeddings=raw.get("tie_word_embeddings", True),
        num_local_experts=0,
        **kwargs,
        **{
            k: raw[k]
            for k in (
                "vocab_size",
                "hidden_size",
                "head_dim",
                "intermediate_size",
                "rms_norm_eps",
                "rope_theta",
                "max_position_embeddings",
            )
        },
    )
