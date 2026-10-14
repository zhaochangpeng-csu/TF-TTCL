"""Conservative, label-free rule-form check for a future transfer protocol.

This check detects obvious example-specific statements; it is not a semantic
entailment verifier. It does not modify the frozen transfer v1 run.
"""
from __future__ import annotations

import re
from typing import Any


_PROCEDURAL_START = re.compile(
    r"^(?:check|inspect|identify|compare|read|count|locate|calculate|verify|"
    r"determine|use|consider|look for|if|when|to)\b", re.I
)
_ASSERTION_START = re.compile(
    r"^(?:the|this|a|an) (?:image|picture|map|diagram|highlighted|punnett square|table)\b", re.I
)
_CONDITIONAL = re.compile(r"\b(?:if|when|unless|otherwise)\b", re.I)
_WORD = re.compile(r"[a-z0-9]+", re.I)


def literal_mention(text: str, phrase: str | None) -> bool:
    parts = _WORD.findall((phrase or "").lower())
    if not parts:
        return False
    pattern = r"(?<![a-z0-9])" + r"\W+".join(map(re.escape, parts)) + r"(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


def assess_rule(entry: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic rejection reasons using only the source experience."""
    positive = str(entry.get("positive_rule", "")).strip()
    negative = str(entry.get("negative_rule", "")).strip()
    reasons = []
    if not positive or not _PROCEDURAL_START.search(positive):
        reasons.append("no_procedural_opening")
    if _ASSERTION_START.search(positive):
        reasons.append("source_observation_as_rule")
    source_negative = entry.get("negative_answer_text")
    if (literal_mention(positive, source_negative)
            and not _CONDITIONAL.search(positive)):
        reasons.append("negative_answer_asserted_in_positive_rule")
    if len(positive) > 400 or len(negative) > 400:
        reasons.append("too_long")
    return {"accepted": not reasons, "reasons": reasons}
