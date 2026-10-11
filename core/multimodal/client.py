"""OpenAI-compatible multimodal VLM client."""

from __future__ import annotations

import base64
import json
import mimetypes
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from openai import OpenAI

from .schema import (
    ContrastiveEvidenceVerdict,
    EvidenceClaim,
    EvidenceVerdict,
    GenerationResponse,
    SamplingConfig,
)


class VLMBackend(Protocol):
    model_name: str

    def generate(
        self,
        system_prompt: str,
        user_text: str,
        image_paths: Sequence[str],
        sampling: SamplingConfig,
    ) -> GenerationResponse: ...


def image_to_data_url(path: str) -> str:
    source = Path(path)
    mime = mimetypes.guess_type(source.name)[0] or "image/png"
    payload = base64.b64encode(source.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def build_multimodal_content(text: str, image_paths: Sequence[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image_to_data_url(path)}}
        for path in image_paths
    ]
    content.append({"type": "text", "text": text})
    return content


class OpenAIVLMClient:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "EMPTY",
        timeout: float = 300.0,
    ) -> None:
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def generate(
        self,
        system_prompt: str,
        user_text: str,
        image_paths: Sequence[str],
        sampling: SamplingConfig,
    ) -> GenerationResponse:
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": build_multimodal_content(user_text, image_paths),
            }
        )
        request: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": sampling.temperature,
            "max_tokens": sampling.max_tokens,
            "stream": False,
        }
        if sampling.top_p is not None:
            request["top_p"] = sampling.top_p
        if sampling.seed is not None:
            request["seed"] = sampling.seed
        if sampling.request_logprobs:
            request.update({"logprobs": True, "top_logprobs": 1})

        started = time.monotonic()
        response = self.client.chat.completions.create(**request)
        latency = time.monotonic() - started
        choice = response.choices[0]
        token_logprobs: List[float] = []
        if sampling.request_logprobs and choice.logprobs and choice.logprobs.content:
            token_logprobs = [item.logprob for item in choice.logprobs.content]
        usage = response.usage
        return GenerationResponse(
            content=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            logprobs=token_logprobs,
            latency_seconds=latency,
        )


def parse_json_object(text: str) -> Mapping[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object in model response")
    return json.loads(cleaned[start : end + 1])


def parse_evidence_verdict(text: str) -> EvidenceVerdict:
    allowed = {"entailed", "contradicted", "insufficient"}
    try:
        payload = parse_json_object(text)
        status = str(payload.get("answer_status", "insufficient")).lower()
        status = status if status in allowed else "insufficient"
        claims = []
        for item in payload.get("claims", []):
            claim_status = str(item.get("status", "insufficient")).lower()
            if claim_status not in allowed:
                claim_status = "insufficient"
            claims.append(
                EvidenceClaim(
                    text=str(item.get("text", "")),
                    status=claim_status,  # type: ignore[arg-type]
                    evidence=str(item.get("evidence", "")),
                )
            )
        return EvidenceVerdict(
            answer_status=status,  # type: ignore[arg-type]
            claims=claims,
            rationale=str(payload.get("rationale", "")),
        )
    except (ValueError, TypeError, json.JSONDecodeError):
        return EvidenceVerdict(
            answer_status="insufficient", parse_error=True, rationale=text[:500]
        )


def parse_contrastive_verdict(text: str) -> ContrastiveEvidenceVerdict:
    allowed_statuses = {"entailed", "contradicted", "insufficient"}
    winner_aliases = {
        "response_x": "candidate_1",
        "response_y": "candidate_2",
        "candidate_1": "candidate_1",
        "candidate_2": "candidate_2",
        "undecidable": "undecidable",
    }
    try:
        payload = parse_json_object(text)
        raw_winner = str(payload.get("winner", "undecidable")).strip().lower()
        winner = winner_aliases.get(raw_winner, "undecidable")
        first = str(
            payload.get("response_x_status", payload.get("candidate_1_status", "insufficient"))
        ).strip().lower()
        second = str(
            payload.get("response_y_status", payload.get("candidate_2_status", "insufficient"))
        ).strip().lower()
        evidence_payload = payload.get("decisive_evidence", [])
        counterevidence_payload = payload.get("counterevidence", [])
        malformed = (
            raw_winner not in winner_aliases
            or first not in allowed_statuses
            or second not in allowed_statuses
            or not isinstance(evidence_payload, list)
            or not isinstance(counterevidence_payload, list)
        )
        evidence = [
            str(item).strip()
            for item in evidence_payload
            if str(item).strip()
        ] if isinstance(evidence_payload, list) else []
        counterevidence = [
            str(item).strip()
            for item in counterevidence_payload
            if str(item).strip()
        ] if isinstance(counterevidence_payload, list) else []
        return ContrastiveEvidenceVerdict(
            winner=winner,  # type: ignore[arg-type]
            candidate_1_status=first if first in allowed_statuses else "insufficient",  # type: ignore[arg-type]
            candidate_2_status=second if second in allowed_statuses else "insufficient",  # type: ignore[arg-type]
            decisive_evidence=evidence,
            counterevidence=counterevidence,
            rationale=str(payload.get("rationale", "")),
            parse_error=malformed,
        )
    except (ValueError, TypeError, json.JSONDecodeError):
        return ContrastiveEvidenceVerdict(
            winner="undecidable",
            candidate_1_status="insufficient",
            candidate_2_status="insufficient",
            rationale=text[:500],
            parse_error=True,
        )
