"""Dataset-aware final-answer extraction and equivalence."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


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


def normalize_answer(text: str, dataset: str = "") -> str:
    answer = unicodedata.normalize("NFKC", extract_final_answer(text)).strip().lower()
    answer = answer.replace("’", "'")
    answer = re.sub(r"^[\s\"']+|[\s\"']+$", "", answer)
    answer = re.sub(r"\s+", " ", answer)
    if dataset.lower() in {"textvqa", "docvqa"}:
        answer = re.sub(r"[^\w\s.-]", "", answer)
        answer = re.sub(r"\b(a|an|the)\b", " ", answer)
        answer = re.sub(r"\s+", " ", answer).strip()
    elif dataset.lower() in {"scienceqa", "scienceqa_img"}:
        match = re.search(r"(?:option\s*)?\(?([a-e])\)?$", answer)
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
