"""Answer-blind observations and small, auditable shared-evidence operators.

This is a domain-limited hypothesis probe, not a general VQA solver. No model
generated Python, candidate verdict, or confidence is executed or trusted.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Sequence


MAGNET_FACT_SYSTEM = """Perform literal text transcription, not question answering.
Inspect the FIRST image only. List visible standalone single uppercase letters
with their center positions. Do not infer missing letters, letters from colors,
or an expected number of letters. Ignore words and numbers. A uniformly blank
image contains no letters: return {"items":[]}.
Return ONLY a JSON object with key items, an array. Each item has EXACTLY keys
text (the verbatim single uppercase letter), x, and y (numeric center position,
normalized from 0 to 1000). Origin is top left; x increases to the right and y
downward. Include only letters actually visible. No expected letter identities,
question, answer options, or conclusions are provided."""

THERMAL_FACT_SYSTEM = """Transcribe a visible temperature table, without answering
any question. No question or answer options are supplied. Inspect FIRST image
only. Read EVERY data row in image order from a table whose columns explicitly
name initial and final temperatures in degrees Celsius. Do not infer missing
numbers from text or knowledge. If headers, unit, or any required number are
not readable, return {"visible":false,"unit":null,"rows":[]}.
Otherwise return JSON only with keys visible (true), unit ("C"), and rows
(a list with EVERY row, each containing only numeric initial and final keys).
Do not discuss heat flow, candidate answers, conclusions, or confidence."""

SHARED_PROMPTS = (MAGNET_FACT_SYSTEM, THERMAL_FACT_SYSTEM)
VERSION = "shared_facts_v2_literal_positions"


def route_operator(question: str, choices: Sequence[str]) -> str | None:
    """Fixed label-free routing; exclude text-only and unsupported tasks."""
    query = question.rsplit("\n\nQuestion:\n", 1)[-1].lower()
    normalized = [re.sub(r"\s+", " ", str(c).strip().lower()) for c in choices]
    if len(normalized) != 2 or len(set(normalized)) != 2:
        return None
    if set(normalized) == {"attract", "repel"} and "magnet" in query:
        return "facing_poles"
    if "thermal energy" in query and "transferred" in query:
        if all(_flow_direction(c) is not None for c in normalized):
            if len({_flow_direction(c) for c in normalized}) == 2:
                return "temperature_delta"
    return None


def _flow_direction(choice: str) -> str | None:
    # Only support the published two-entity flow template; do not generalize
    # to ambiguous descriptions or multiple simultaneous source/sink pairs.
    parts = re.split(r"\s*(?:(?:\.\s*){2,}|…+)\s*", choice.strip().lower())
    if len(parts) != 2:
        return None
    if parts == ["the surroundings", "each vial"]:
        return "into_vials"
    if parts == ["each vial", "the surroundings"]:
        return "out_of_vials"
    return None


def parse_observation(text: str, operator: str) -> dict[str, Any]:
    """Strict schema, finite numbers and no answer-bearing extra fields."""
    failure = {"parse_error": True, "visible": False, "facts": None}
    try:
        cleaned = text.strip()
        if cleaned.startswith("```json\n") and cleaned.endswith("```"):
            cleaned = cleaned[8:-3].strip()
        raw = json.loads(cleaned)
        if not isinstance(raw, dict):
            return failure
        if operator == "facing_poles":
            if set(raw) != {"items"} or not isinstance(raw["items"], list) or len(raw["items"]) > 64:
                return failure
            items = raw["items"]
            for item in items:
                if not isinstance(item, dict) or set(item) != {"text", "x", "y"}:
                    return failure
                if not isinstance(item["text"], str) or not re.fullmatch(r"[A-Z]", item["text"]):
                    return failure
                if any(type(item[key]) not in (int, float) or not math.isfinite(item[key])
                       or not 0 <= item[key] <= 1000 for key in ("x", "y")):
                    return failure
            ordered = sorted(items, key=lambda item: item["x"])
            visible = (len(ordered) == 4 and all(item["text"] in ("N", "S") for item in ordered)
                and all(ordered[i + 1]["x"] - ordered[i]["x"] >= 10 for i in range(3))
                and max(item["y"] for item in ordered) - min(item["y"] for item in ordered) <= 150)
            labels = [item["text"] for item in ordered]
            facts = {"ordered_labels": labels, "facing_poles": labels[1:3]} if visible else None
        elif operator == "temperature_delta":
            if set(raw) != {"visible", "unit", "rows"} or type(raw.get("visible")) is not bool:
                return failure
            visible = raw["visible"]
            rows = raw["rows"]
            if not isinstance(rows, list):
                return failure
            if not raw["visible"]:
                if rows or raw["unit"] is not None:
                    return failure
                facts = None
            else:
                if raw["unit"] != "C" or not 1 <= len(rows) <= 16:
                    return failure
                for row in rows:
                    if not isinstance(row, dict) or set(row) != {"initial", "final"}:
                        return failure
                    if any(type(v) not in (int, float) or not math.isfinite(v)
                           or not -273.15 <= v <= 10000 for v in row.values()):
                        return failure
                facts = {"unit": "C", "rows": rows}
        else:
            return failure
        return {"parse_error": False, "visible": bool(visible), "facts": facts}
    except (ValueError, TypeError, KeyError):
        return failure


def execute_operator(operator: str, facts: dict[str, Any],
                     choices: Sequence[str]) -> dict[str, Any]:
    """Pure deterministic reasoning, returning the calculation for audit."""
    if operator == "facing_poles":
        left, right = facts["facing_poles"]
        relation = "repel" if left == right else "attract"
        directions = [str(c).strip().lower() for c in choices]
        calculation = {"left_facing": left, "right_facing": right,
                       "same_pole": left == right, "relation": relation}
    elif operator == "temperature_delta":
        deltas = [row["final"] - row["initial"] for row in facts["rows"]]
        relation = ("into_vials" if all(d > 0 for d in deltas) else
                    "out_of_vials" if all(d < 0 for d in deltas) else None)
        directions = [_flow_direction(c) for c in choices]
        calculation = {"temperature_deltas": deltas, "heat_flow": relation}
    else:
        raise ValueError(f"Unsupported operator: {operator}")
    matches = [i for i, direction in enumerate(directions) if direction == relation]
    answer = chr(97 + matches[0]) if relation is not None and len(matches) == 1 else None
    return {"answer": answer, "calculation": calculation}


def assess_certificate(operator: str, observations: list[dict[str, Any]],
                       choices: Sequence[str], candidate_answers: Sequence[str]) -> dict[str, Any]:
    """Real/fidelity stable and blank/mismatch non-reproducing, or abstain.

    Equal facts on a mismatched image fail closed even if coincidental. A
    malformed counterfactual never counts as absent evidence.
    """
    result: dict[str, Any] = {"facts_stable": False, "counterfactual_drop": False,
                             "gate_passed": False, "computed_answer": None,
                             "calculation": {}, "fallback_reason": "malformed_observation"}
    if len(observations) not in (3, 4) or any(o.get("parse_error") for o in observations):
        return result
    first, second, *counterfactuals = observations
    stable = bool(first["visible"] and second["visible"]
                  and first["facts"] == second["facts"])
    result["facts_stable"] = stable
    if not stable:
        result["fallback_reason"] = "missing_or_unstable_real_facts"
        return result
    # Blank must contain NO observation; mismatch may expose other facts, not
    # reproduce the original. These checks concern extraction, not QA answers.
    dropped = (not counterfactuals[0]["visible"] and
               all(not o["visible"] or o["facts"] != first["facts"]
                   for o in counterfactuals[1:]))
    result["counterfactual_drop"] = dropped
    calculated = execute_operator(operator, first["facts"], choices)
    result.update(computed_answer=calculated["answer"], calculation=calculated["calculation"])
    if not dropped:
        result["fallback_reason"] = "counterfactual_reproduces_facts"
    elif calculated["answer"] is None:
        result["fallback_reason"] = "nonunique_or_mixed_calculation"
    elif calculated["answer"] not in candidate_answers:
        result["fallback_reason"] = "computed_answer_not_proposed"
    else:
        result.update(gate_passed=True, fallback_reason=None)
    return result
