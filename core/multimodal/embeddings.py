"""Embedding interfaces used by evidence memory.

The deterministic hash backend is intended for tests and pipeline smoke runs.
Research runs must use a backend whose name does not start with ``hash-``.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Protocol, Sequence

import httpx
import numpy as np

from .client import image_to_data_url


class MultimodalEmbedder(Protocol):
    name: str

    def encode_text(self, text: str) -> np.ndarray: ...

    def encode_image(self, image_paths: Sequence[str]) -> np.ndarray: ...

    def encode_joint(self, text: str, image_paths: Sequence[str]) -> np.ndarray: ...


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


class HashMultimodalEmbedder:
    """Dependency-free deterministic test backend, never a paper baseline."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions
        self.name = f"hash-{dimensions}"

    def _features(self, values: Sequence[bytes]) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for value in values:
            digest = hashlib.sha256(value).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        return _normalize(vector)

    def encode_text(self, text: str) -> np.ndarray:
        tokens = re.findall(r"\w+|[^\w\s]", text.lower())
        return self._features([token.encode("utf-8") for token in tokens] or [b""])

    def encode_image(self, image_paths: Sequence[str]) -> np.ndarray:
        blocks = []
        for path in image_paths:
            payload = Path(path).read_bytes()
            blocks.extend(payload[i : i + 1024] for i in range(0, len(payload), 1024))
        return self._features(blocks or [b""])

    def encode_joint(self, text: str, image_paths: Sequence[str]) -> np.ndarray:
        return _normalize(self.encode_text(text) + self.encode_image(image_paths))


class VLLMMultimodalEmbedder:
    """Qwen3-VL-Embedding client backed by vLLM's OpenAI embedding route.

    Recent vLLM versions accept chat-style multimodal content at
    ``/v1/embeddings``. Keeping the model in a separate server prevents the
    generator and retriever from repeatedly swapping weights.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "EMPTY",
        timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.name = model_name
        self.api_key = api_key
        self.client = httpx.Client(timeout=timeout)

    def _encode(self, text: str, image_paths: Sequence[str]) -> np.ndarray:
        content: list[Dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": image_to_data_url(path)}}
            for path in image_paths
        ]
        content.append({"type": "text", "text": text})
        response = self.client.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.name,
                "input": [{"role": "user", "content": content}],
                "encoding_format": "float",
            },
        )
        response.raise_for_status()
        payload = response.json()
        vector = np.asarray(payload["data"][0]["embedding"], dtype=np.float32)
        return _normalize(vector)

    def encode_text(self, text: str) -> np.ndarray:
        return self._encode(f"Represent this question for retrieval:\n{text}", ())

    def encode_image(self, image_paths: Sequence[str]) -> np.ndarray:
        return self._encode("Represent the visual evidence for retrieval.", image_paths)

    def encode_joint(self, text: str, image_paths: Sequence[str]) -> np.ndarray:
        return self._encode(
            f"Represent this visual question and its evidence for retrieval:\n{text}",
            image_paths,
        )
