"""Local dataset adapters with a hard label/inference boundary."""

from __future__ import annotations

import base64
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from .schema import EvaluationTarget, MultimodalSample


_LABEL_KEYS = {
    "answer",
    "answers",
    "label",
    "labels",
    "output",
    "gold",
    "annotation",
    "solution",
    "lecture",
    "rationale",
    "explanation",
    "response",
}


@dataclass(frozen=True)
class LoadedExample:
    sample: MultimodalSample
    target: Optional[EvaluationTarget]


def _read_records(path: Path) -> List[Mapping[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    for key in ("data", "examples", "questions"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _first(record: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _resolve_images(value: Any, image_root: Path) -> List[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    resolved = []
    for item in values:
        if not isinstance(item, str):
            raise TypeError("Local adapters require image paths, not in-memory image objects")
        path = Path(item)
        if not path.is_absolute():
            path = image_root / path
        resolved.append(str(path.resolve()))
    return resolved


def _extract_answers(record: Mapping[str, Any], choices: Sequence[str]) -> List[str]:
    raw = _first(record, ("answers", "answer", "label", "output", "gold"))
    if raw is None:
        return []
    values = raw if isinstance(raw, list) else [raw]
    answers: List[str] = []
    for value in values:
        if isinstance(value, int) and choices and 0 <= value < len(choices):
            answers.extend([chr(65 + value), str(choices[value])])
        else:
            answers.append(str(value))
    return answers


def load_local_dataset(
    path: str,
    dataset: str,
    image_root: Optional[str] = None,
    limit: Optional[int] = None,
    order: str = "forward",
    seed: int = 0,
) -> List[LoadedExample]:
    source = Path(path)
    root = Path(image_root) if image_root else source.parent
    records = _read_records(source)
    if order == "reverse":
        records.reverse()
    elif order == "random":
        random.Random(seed).shuffle(records)
    elif order != "forward":
        raise ValueError("order must be forward, reverse, or random")
    if limit is not None:
        records = records[:limit]

    examples: List[LoadedExample] = []
    for index, record in enumerate(records):
        sample_id = str(_first(record, ("id", "pid", "question_id", "questionId"), index))
        question = str(_first(record, ("question", "query", "instruction", "text"), ""))
        choices = [str(item) for item in _first(record, ("choices", "options"), [])]
        image_value = _first(record, ("image_paths", "image_path", "image", "image_id"))
        images = _resolve_images(image_value, root)
        metadata = {
            key: value
            for key, value in record.items()
            if key not in _LABEL_KEYS
            and key not in {"image_paths", "image_path", "image", "image_id", "question", "query", "instruction", "text", "choices", "options"}
        }
        sample = MultimodalSample(
            sample_id=sample_id,
            dataset=dataset,
            question=question,
            image_paths=tuple(images),
            choices=tuple(choices),
            metadata=metadata,
        )
        answers = _extract_answers(record, choices)
        target_metadata = {
            key: record[key]
            for key in ("answer_type", "precision", "question_type", "choices", "options")
            if key in record
        }
        target = EvaluationTarget(sample_id, tuple(answers), target_metadata) if answers else None
        examples.append(LoadedExample(sample, target))
    return examples
