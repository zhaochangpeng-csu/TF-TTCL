"""Dataset-aware final-answer extraction and equivalence."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping, Optional, Sequence


_FINAL_PATTERNS = (
    re.compile(r"(?:final answer|answer)\s*[:：]\s*(.+)$", re.I | re.M),
    re.compile(r"\\boxed\{([^{}]+)\}"),
)


def extract_final_answer(text: str) -> str:
    text = text.strip()
    for pattern in _FINAL_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return str(matches[-1]).strip()
    return text.splitlines()[-1].strip() if text else ""


_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def extract_mathvista_answer(text: str, metadata: Mapping[str, Any] | None = None) -> str:
    """Deterministically extract MathVista's short answer from a response.

    The official benchmark uses a separate LLM for general answer extraction.
    EviMem experiments avoid that external teacher dependency by enforcing a
    final-answer line and applying a transparent answer-type-aware extractor.
    """
    metadata = metadata or {}
    raw = unicodedata.normalize("NFKC", extract_final_answer(text)).strip()
    choices: Sequence[str] = [str(item) for item in metadata.get("choices", [])]
    question_type = str(metadata.get("question_type", "")).lower()
    answer_type = str(metadata.get("answer_type", "")).lower()

    if question_type == "multi_choice" and choices:
        # Only treat a leading standalone letter as an option marker. An
        # unanchored search incorrectly maps ordinary answers ending in A-E
        # (for example, "Common water flea") to an option letter.
        option_match = re.match(
            r"^\s*(?:option\s*|answer\s*)?\(?([A-Za-z])\)?(?=$|[).:\-\s])",
            raw,
            flags=re.I,
        )
        if option_match:
            index = ord(option_match.group(1).upper()) - ord("A")
            if 0 <= index < len(choices):
                return choices[index]
        lowered = raw.lower().strip(" .")
        exact = next((choice for choice in choices if choice.lower() == lowered), None)
        if exact is not None:
            return exact
        contained = [choice for choice in choices if choice.lower() in lowered]
        if len(contained) == 1:
            return contained[0]

    if answer_type == "list":
        lists = re.findall(r"\[[^\[\]]*\]", raw)
        return lists[-1] if lists else raw

    if answer_type in {"integer", "float"}:
        numbers = _NUMBER_PATTERN.findall(raw.replace(",", ""))
        if numbers:
            value = float(numbers[-1])
            if answer_type == "integer":
                return str(int(value))
            try:
                precision = int(float(metadata.get("precision", 2)))
            except (TypeError, ValueError):
                precision = 2
            return str(round(value, precision))
    return raw


def normalize_answer(
    text: str,
    dataset: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> str:
    if dataset.lower() == "mathvista":
        text = extract_mathvista_answer(text, metadata)
    answer = unicodedata.normalize("NFKC", extract_final_answer(text)).strip().lower()
    answer = answer.replace("’", "'")
    answer = re.sub(r"^[\s\"']+|[\s\"']+$", "", answer)
    answer = re.sub(r"\s+", " ", answer)
    if dataset.lower() in {"textvqa", "docvqa"}:
        answer = re.sub(r"[^\w\s.-]", "", answer)
        answer = re.sub(r"\b(a|an|the)\b", " ", answer)
        answer = re.sub(r"\s+", " ", answer).strip()
    elif dataset.lower() in {"scienceqa", "scienceqa_img"}:
        match = re.match(
            r"^(?:option\s*)?\(?([a-e])\)?(?:\s*[).:\-]\s*|\s+).+$",
            answer,
        )
        if not match:
            match = re.match(r"^(?:option\s*)?\(?([a-e])\)?[.)]?$", answer)
        if match:
            answer = match.group(1)
    elif dataset.lower() == "mathvista":
        answer = answer.replace(",", "")
        answer = re.sub(r"\s*(?:degrees?|°|%|dollars?)$", "", answer)
    return answer


def answers_match(left: str, right: str, dataset: str = "") -> bool:
    a = normalize_answer(left, dataset)
    b = normalize_answer(right, dataset)
    if not a or not b:
        return False
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) <= 1e-6
    except ValueError:
        return False
