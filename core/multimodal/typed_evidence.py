"""Bounded observations plus explicit reusable operators, version 3.

JSON constraints establish syntax, not truth. Overflow, missing operands,
ambiguous programs and counterfactual observations all fail closed.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Sequence

VERSION = "typed_v3_bounded_relations_counts_tables"


def obj(properties: dict) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


COORD = {"type": "integer", "minimum": 0, "maximum": 1000}
LETTER_SCHEMA = obj({"overflow": {"type": "boolean"}, "items": {
    "type": "array", "maxItems": 8,
    "items": obj({"text": {"type": "string", "pattern": "^[A-Za-z]$"},
                  "x": COORD, "y": COORD})}})
TABLE_SCHEMA = obj({"overflow": {"type": "boolean"}, "columns": {
    "type": "array", "maxItems": 4, "items": {"type": "string", "maxLength": 48}},
    "rows": {"type": "array", "maxItems": 8, "items": obj({
        "label": {"type": "string", "maxLength": 64}, "values": {
            "type": "array", "maxItems": 4,
            "items": {"type": ["number", "null"]}}})}})
OBJECT_SCHEMA = obj({"overflow": {"type": "boolean"}, "objects": {
    "type": "array", "maxItems": 16, "items": obj({
        "color": {"type": "string", "maxLength": 16},
        "shape": {"type": "string", "maxLength": 16},
        "size": {"type": "string", "maxLength": 16}, "x": COORD, "y": COORD})}})

LETTER_PROMPT = """Transcribe standalone visible single letters, uppercase or
lowercase, with center coordinates x/y normalized to integers 0..1000 (top-left
origin). Ignore words and numbers. No expected letters, question, answers, or
conclusions are given. Inspect only the FIRST image. Return the constrained
JSON with overflow and items (each with text,x,y). If there are more than eight
letters, or a required letter cannot be read, set overflow=true. Never guess
missing letters or derive letters from colors. A blank image returns
overflow=false and items=[]. Include at most eight letters, never an arbitrary
truncated subset presented as complete. No interpretation or confidence."""
TABLE_PROMPT = """Transcribe a visible numeric table in the FIRST image. No
question, options or candidate answers are provided. Return constrained JSON:
overflow, columns (all numeric column headers verbatim, including units), rows
(each with label verbatim and values in column order). Do not infer missing
values or interpret trends. Use null for unreadable numeric cells. Maximum
four numeric columns and eight rows; if more, or the table cannot be faithfully
read, set overflow=true. Do not truncate a table as if complete. If no numeric
table is visible, return overflow=false, columns=[], rows=[]. No conclusions."""
OBJECT_PROMPT = """List separate simple geometric objects visible in the FIRST
image. No question, filter criteria, answer choices or candidates are provided.
Return constrained JSON with overflow and objects. Each object has color,
shape, size (literal visible attributes; empty string when unknown), and integer
center x,y in 0..1000 from top left. Use singular shape names. Do not count
labels, arrows or decorations as objects. Do not infer occluded objects.
Maximum sixteen objects; if more, or boundaries are ambiguous, set
overflow=true, not a complete-looking subset. Blank image: overflow=false,
objects=[]. Do not compute counts, filtered sets, conclusions or confidence."""
SPECS = {"letters": (LETTER_PROMPT, LETTER_SCHEMA, 512),
         "table": (TABLE_PROMPT, TABLE_SCHEMA, 1024),
         "objects": (OBJECT_PROMPT, OBJECT_SCHEMA, 1536)}

SHAPES = {"objects": None, "balls": "sphere", "spheres": "sphere", "cubes": "cube",
          "circles": "circle", "squares": "square", "triangles": "triangle",
          "cylinders": "cylinder", "cones": "cone"}
COLORS = {"red", "orange", "yellow", "green", "blue", "purple", "pink", "black", "white", "brown", "gray"}


def compile_program(question: str, choices: Sequence[str]) -> dict | None:
    """Label-free, bounded lexical compilation; no model-generated code."""
    query = question.rsplit("\n\nQuestion:\n", 1)[-1].strip().lower()
    cs = [c.strip().lower() for c in choices]
    if len(set(cs)) != len(cs) or not 2 <= len(cs) <= 6:
        return None
    if set(cs) == {"attract", "repel"} and "magnet" in query:
        return {"operator": "facing_poles", "modality": "letters"}
    if "thermal energy" in query and "transferred" in query and len(cs) == 2:
        parts = [re.split(r"\s*(?:(?:\.\s*){2,}|…+)\s*", c) for c in cs]
        if all(len(p) == 2 for p in parts) and parts[0] == list(reversed(parts[1])):
            entities = set(parts[0])
            if "the surroundings" in entities and len(entities) == 2:
                body = next(e for e in entities if e != "the surroundings")
                if re.fullmatch(r"(?:each|the) [a-z ]+", body):
                    return {"operator": "temperature_delta", "modality": "table", "body": body}
    extreme = re.fullmatch(r"which .+ has (?:the )?(highest|lowest|greatest|smallest|largest) ([a-z ]+)\?", query)
    if extreme:
        return {"operator": "table_extreme", "modality": "table", "metric": extreme[2].strip(),
                "direction": "min" if extreme[1] in {"lowest", "smallest"} else "max"}
    count = re.fullmatch(r"how many ([a-z ]+) (?:are (?:shown|visible|there)(?: in (?:the|this) (?:image|picture))?|do you see)\?", query)
    if count and all(re.fullmatch(r"\d{1,3}", c) for c in cs):
        words = count[1].split()
        if words[-1] not in SHAPES:
            return None
        filters = {}
        if SHAPES[words[-1]]:
            filters["shape"] = SHAPES[words[-1]]
        for word in words[:-1]:
            key = "color" if word in COLORS else "size" if word in {"small", "large"} else None
            if key is None or key in filters:
                return None
            filters[key] = word
        return {"operator": "object_count", "modality": "objects", "filters": filters}
    return None


def _unique_json(text: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate key")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _schema_valid(value: Any, schema: dict) -> bool:
    kind = schema["type"]
    if isinstance(kind, list):
        return any(_schema_valid(value, {**schema, "type": k}) for k in kind)
    if kind == "null":
        return value is None
    if kind == "boolean":
        return type(value) is bool
    if kind == "object":
        return isinstance(value, dict) and set(value) == set(schema["properties"]) and all(
            _schema_valid(value[k], s) for k, s in schema["properties"].items())
    if kind == "array":
        return isinstance(value, list) and len(value) <= schema["maxItems"] and all(
            _schema_valid(v, schema["items"]) for v in value)
    if kind == "string":
        return isinstance(value, str) and len(value) <= schema.get("maxLength", 10000) and (
            "pattern" not in schema or re.fullmatch(schema["pattern"], value) is not None)
    if kind in {"integer", "number"}:
        return (type(value) in ((int,) if kind == "integer" else (int, float))
                and math.isfinite(value) and schema.get("minimum", -1e300) <= value <= schema.get("maximum", 1e300))
    return False


def _facing_facts(items: list[dict]) -> dict | None:
    if len(items) != 4 or any(item["text"] not in {"N", "S"} for item in items):
        return None
    # Farthest endpoints define a direction-independent, collinear axis.
    pairs = [(sum((a[k] - b[k]) ** 2 for k in ("x", "y")), a, b)
             for i, a in enumerate(items) for b in items[i + 1:]]
    distance2, origin, end = max(pairs, key=lambda pair: pair[0])
    if distance2 == 0:
        return None
    length = math.sqrt(distance2)
    ux, uy = (end["x"] - origin["x"]) / length, (end["y"] - origin["y"]) / length
    projected = []
    for item in items:
        dx, dy = item["x"] - origin["x"], item["y"] - origin["y"]
        if abs(-uy * dx + ux * dy) > 75:
            return None
        projected.append((ux * dx + uy * dy, item["text"]))
    projected.sort()
    if any(projected[i + 1][0] - projected[i][0] < 10 for i in range(3)):
        return None
    labels = [p[1] for p in projected]
    if set(labels[:2]) != {"N", "S"} or set(labels[2:]) != {"N", "S"}:
        return None
    return {"facing_poles": sorted(labels[1:3]), "outer_poles": sorted([labels[0], labels[3]])}


def parse_observation(text: str, program: dict) -> dict:
    result = {"parse_error": True, "complete": False, "visible": False,
              "facts": None, "literal_empty": False}
    try:
        raw = _unique_json(text)
        modality = program["modality"]
        if not _schema_valid(raw, SPECS[modality][1]):
            return result
        result.update(parse_error=False, complete=not raw["overflow"])
        if raw["overflow"]:
            return result
        if modality == "letters":
            facts = _facing_facts(raw["items"])
            result["literal_empty"] = not raw["items"]
        elif modality == "table":
            columns, rows = raw["columns"], raw["rows"]
            result["literal_empty"] = not columns and not rows
            valid = (bool(columns) and bool(rows) and len(set(columns)) == len(columns)
                and len({r["label"] for r in rows}) == len(rows)
                and all(r["label"].strip() and len(r["values"]) == len(columns)
                        and all(v is not None for v in r["values"]) for r in rows))
            if not result["literal_empty"] and not valid:
                result["complete"] = False
            facts = {"columns": columns, "rows": rows} if valid else None
        else:
            objects = raw["objects"]
            result["literal_empty"] = not objects
            if len({(o["x"], o["y"]) for o in objects}) != len(objects):
                result["complete"] = False
            # Count filters cannot treat an unknown attribute as a non-match.
            if any(not o[k].strip() for o in objects for k in program["filters"]):
                result["complete"] = False
            facts = [{k: o[k].strip().lower() for k in ("color", "shape", "size")}
                     for o in objects] if objects else None
            if facts:
                facts.sort(key=lambda o: json.dumps(o, sort_keys=True))
        result.update(visible=facts is not None, facts=facts)
        return result
    except (ValueError, TypeError, KeyError, OverflowError):
        return result


def count_filtered(objects: list[dict], filters: dict) -> int:
    return sum(all(o[k] == value for k, value in filters.items()) for o in objects)


def unique_extreme(values: Sequence[float], direction: str) -> int | None:
    best = (min if direction == "min" else max)(values)
    indices = [i for i, value in enumerate(values) if value == best]
    return indices[0] if len(indices) == 1 else None


def execute_program(program: dict, facts: Any, choices: Sequence[str]) -> dict:
    operator = program["operator"]
    result = {"answer": None, "calculation": {}}
    expected = None
    cs = [c.strip().lower() for c in choices]
    if operator == "facing_poles":
        poles = facts["facing_poles"]
        expected = "repel" if poles[0] == poles[1] else "attract"
        result["calculation"] = {"facing_poles": poles, "equal": poles[0] == poles[1], "relation": expected}
    elif operator == "object_count":
        n = count_filtered(facts, program["filters"])
        expected = str(n)
        result["calculation"] = {"filters": program["filters"], "count": n}
    elif operator == "temperature_delta":
        columns = [c.lower() for c in facts["columns"]]
        initial = [i for i, c in enumerate(columns) if "initial" in c and "temperature" in c and ("°c" in c or "celsius" in c)]
        final = [i for i, c in enumerate(columns) if "final" in c and "temperature" in c and ("°c" in c or "celsius" in c)]
        if len(initial) != 1 or len(final) != 1:
            return result
        deltas = [r["values"][final[0]] - r["values"][initial[0]] for r in facts["rows"]]
        direction = "in" if all(d > 0 for d in deltas) else "out" if all(d < 0 for d in deltas) else None
        desired = ["the surroundings", program["body"]] if direction == "in" else [program["body"], "the surroundings"] if direction == "out" else None
        matched = [i for i, c in enumerate(cs) if re.split(r"\s*(?:(?:\.\s*){2,}|…+)\s*", c) == desired]
        result["calculation"] = {"deltas": deltas, "direction": direction}
        if len(matched) == 1:
            result["answer"] = chr(97 + matched[0])
        return result
    else:
        metric = program["metric"].split()
        indices = [i for i, c in enumerate(facts["columns"])
                   if all(w in re.findall(r"[a-z]+", c.lower()) for w in metric)]
        if len(indices) != 1:
            return result
        values = [r["values"][indices[0]] for r in facts["rows"]]
        selected = unique_extreme(values, program["direction"])
        result["calculation"] = {"column": facts["columns"][indices[0]], "values": values,
                                 "direction": program["direction"], "row_index": selected}
        if selected is not None:
            expected = facts["rows"][selected]["label"].strip().lower()
    matches = [i for i, c in enumerate(cs) if c == expected]
    if expected is not None and len(matches) == 1:
        result["answer"] = chr(97 + matches[0])
    return result


def assess_certificate(program: dict, observations: list[dict], choices: Sequence[str],
                       candidate_answers: Sequence[str]) -> dict:
    result = {"facts_stable": False, "counterfactual_drop": False, "gate_passed": False,
              "computed_answer": None, "calculation": {}, "fallback_reason": "malformed_observation"}
    if len(observations) not in (3, 4):
        return result
    first, second, *counterfactuals = observations
    result["facts_stable"] = bool(all(not o["parse_error"] and o["complete"] and o["visible"]
                                     for o in (first, second)) and first["facts"] == second["facts"])
    if any(o["parse_error"] for o in observations):
        return result
    if any(not o["complete"] for o in observations):
        result["fallback_reason"] = "incomplete_or_overflow_observation"
        return result
    if not result["facts_stable"]:
        result["fallback_reason"] = "missing_or_unstable_real_facts"
        return result
    dropped = counterfactuals[0]["literal_empty"] and all(
        not o["visible"] or o["facts"] != first["facts"] for o in counterfactuals[1:])
    result["counterfactual_drop"] = bool(dropped)
    calculated = execute_program(program, first["facts"], choices)
    result.update(computed_answer=calculated["answer"], calculation=calculated["calculation"])
    if not dropped:
        result["fallback_reason"] = "counterfactual_reproduces_or_hallucinates_facts"
    elif calculated["answer"] is None:
        result["fallback_reason"] = "ambiguous_execution"
    elif calculated["answer"] not in candidate_answers:
        result["fallback_reason"] = "computed_answer_not_proposed"
    else:
        result.update(gate_passed=True, fallback_reason=None)
    return result
