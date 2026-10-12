"""Immutable response traces for exact paired inference comparisons.

The evidence arm replays the budget arm's inputs and responses. Its token and
model latency totals describe the shared computation, not a second expenditure.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .client import VLMBackend
from .image_ops import sha256_file
from .schema import GenerationResponse, SamplingConfig


class ResponseTraceBackend:
    def __init__(self, backend: VLMBackend, directory: Path, namespace: dict[str, Any],
                 mode: str) -> None:
        if mode not in {"record", "replay"}:
            raise ValueError("Response trace mode must be record or replay")
        self.backend = backend
        self.model_name = backend.model_name
        self.directory = directory
        self.namespace = namespace
        self.mode = mode
        self.executed_calls = 0
        self.replayed_calls = 0
        if mode == "record":
            directory.mkdir(parents=True, exist_ok=True)
        elif not directory.is_dir():
            raise FileNotFoundError(f"Missing recorded response trace: {directory}")

    @property
    def trace_stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "executed_calls_this_process": self.executed_calls,
            "replayed_calls_this_process": self.replayed_calls,
            "cost_from_shared_trace": self.mode == "replay",
        }

    def generate(self, system_prompt: str, user_text: str, image_paths: Sequence[str],
                 sampling: SamplingConfig) -> GenerationResponse:
        sampling_payload = asdict(sampling)
        # Preserve old immutable trace keys for unconstrained calls.
        if sampling_payload.get("output_schema") is None:
            sampling_payload.pop("output_schema", None)
        request = {
            "namespace": self.namespace,
            "model": self.model_name,
            "system_prompt": system_prompt,
            "user_text": user_text,
            "image_hashes": [sha256_file(path) for path in image_paths],
            "sampling": sampling_payload,
        }
        encoded = json.dumps(request, sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        path = self.directory / f"{key}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["request"] != request:
                raise ValueError("Response trace request mismatch")
            self.replayed_calls += 1
            return GenerationResponse(**payload["response"])
        if self.mode == "replay":
            raise RuntimeError(
                f"Unrecorded request {key}; refusing to execute a model call in replay mode"
            )
        response = self.backend.generate(system_prompt, user_text, image_paths, sampling)
        # Exclusive creation refuses accidental overwrite of an immutable response.
        with path.open("x", encoding="utf-8") as handle:
            json.dump({"request": request, "response": asdict(response)}, handle,
                      ensure_ascii=False)
        self.executed_calls += 1
        return response
