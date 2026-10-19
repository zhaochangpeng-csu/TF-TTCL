"""Frozen Qwen3 text embeddings for the direct-transfer baseline.

Uses the model-card last-token pooling recipe on CPU so the VLM has exclusive
use of the selected GPU. No hash or lexical substitute is permitted.
"""
from __future__ import annotations

import math
from pathlib import Path


class Qwen3TextEmbedder:
    def __init__(self, model_path: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not Path(model_path).is_dir():
            raise FileNotFoundError(model_path)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                                        padding_side="left")
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to("cpu")
        self.model_path = model_path

    def encode(self, text: str, *, query: bool = False) -> tuple[float, ...]:
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        if query:
            text = ("Instruct: Given a question, retrieve useful reasoning rules "
                    "for answering it\nQuery:" + text)
        inputs = self.tokenizer([text], padding=True, truncation=True,
                                max_length=2048, return_tensors="pt")
        with self.torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            last = hidden[0, int(inputs["attention_mask"][0].sum().item()) - 1]
            vector = self.torch.nn.functional.normalize(last.float(), p=2, dim=0)
        result = tuple(float(value) for value in vector.tolist())
        if not result or any(not math.isfinite(value) for value in result):
            raise ValueError("Nonfinite Qwen3 embedding")
        return result


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embedding dimension mismatch")
    return sum(a * b for a, b in zip(left, right))
