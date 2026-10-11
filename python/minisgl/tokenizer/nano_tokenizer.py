"""CharTokenizer (T4c): C1 sidecar vocab.json (itos) -> PreTrainedTokenizerBase.

nano-model same format: vocab.json = {"itos": [char, ...]}, char-level, no
BOS/EOS/PAD special tokens — finish_reason "stop" is unreachable (t4 §6.4).
"""

import json
import os
from typing import List

from transformers import PreTrainedTokenizer


class CharTokenizer(PreTrainedTokenizer):
    # PreTrainedTokenizer (slow) implements the encode/decode pipeline via the
    # _tokenize/_convert hooks below; PreTrainedTokenizerBase is an ABC whose
    # encode_plus raises NotImplementedError.
    def __init__(self, itos: List[str], **kwargs):
        self.itos = list(itos)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        super().__init__(**kwargs)

    @classmethod
    def from_model_path(cls, model_path: str) -> "CharTokenizer":
        with open(os.path.join(model_path, "vocab.json"), encoding="utf-8") as f:
            return cls(json.load(f)["itos"])

    # ---- transformers slow-tokenizer protocol ----
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        if token not in self.stoi:
            raise ValueError(f"character {token!r} not in vocab (size {len(self.itos)})")
        return self.stoi[token]

    def _convert_id_to_token(self, index: int) -> str:
        return self.itos[index]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        # char-level: no separator between tokens (default " ".join would corrupt)
        return "".join(tokens)

    def get_vocab(self) -> dict:
        return dict(self.stoi)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def eos_token_id(self):
        return None  # no EOS in the char vocab (t4 §6.4)

    @property
    def bos_token_id(self):
        return None

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None):
        path = os.path.join(save_directory, "vocab.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f)
        return (path,)
