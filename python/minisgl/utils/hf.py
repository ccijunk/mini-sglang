import functools
import json
import os
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.asyncio import tqdm
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    from .nano_config import is_nano_model

    if is_nano_model(model_path):  # nano sidecar: C1 vocab.json itos (T4c)
        from ..tokenizer.nano_tokenizer import CharTokenizer

        return CharTokenizer.from_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Some Mistral models store chat_template in a separate JSON file
    if not getattr(tokenizer, "chat_template", None):
        try:
            path = hf_hub_download(repo_id=model_path, filename="chat_template.json")
            with open(path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = json.load(f)["chat_template"]
        except Exception:
            pass
    return tokenizer


@functools.cache
def _load_nano_config(model_path: str) -> Any:
    from .nano_config import load_nano_hf_config

    return load_nano_hf_config(model_path)


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    from .nano_config import is_nano_model

    if is_nano_model(model_path):  # nano sidecar: NanoConfig field names (T4c)
        return _load_nano_config(model_path)
    return AutoConfig.from_pretrained(model_path)


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    config = _load_hf_config(model_path)
    if not hasattr(config, "to_dict"):  # nano shim: SimpleNamespace, already HF-attr-shaped
        return config
    return type(config)(**config.to_dict())


def download_hf_weight(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    try:
        return snapshot_download(
            model_path,
            allow_patterns=["*.safetensors"],
            tqdm_class=DisabledTqdm,
        )
    except Exception as e:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )