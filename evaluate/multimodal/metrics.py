"""Label-isolated metrics for ScienceQA, MathVista, TextVQA and DocVQA."""

from __future__ import annotations

import math
import random
import re
import string
import ast
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

from core.multimodal.answers import answers_match, normalize_answer


def accuracy(prediction: str, targets: Sequence[str], dataset: str) -> float:
    return float(any(answers_match(prediction, target, dataset) for target in targets))


_ARTICLES = {"a", "an", "the"}
_NUMBER_WORDS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}


def _vqa_normalize(text: str) -> str:
    text = text.lower().replace("\n", " ").replace("\t", " ").strip()
    text = re.sub(r"(?<!\d)[.,](?!\d)", " ", text)
    text = re.sub(r"[;\/\[\]\"{}()=+\\_<>@`?!]", " ", text)
    words = []
    for word in text.split():
        word = _NUMBER_WORDS.get(word, word)
        if word not in _ARTICLES:
            words.append(word)
    return " ".join(words)


def vqa_accuracy(prediction: str, targets: Sequence[str]) -> float:
    normalized = _vqa_normalize(prediction)
    normalized_targets = [_vqa_normalize(target) for target in targets]
    if not normalized_targets:
        return 0.0
    per_annotator = []
    for index in range(len(normalized_targets)):
        others = normalized_targets[:index] + normalized_targets[index + 1 :]
        matches = sum(normalized == target for target in others)
        per_annotator.append(min(1.0, matches / 3.0))
    return sum(per_annotator) / len(per_annotator)


def _levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (char_left != char_right),
                )
            )
        previous = current
    return previous[-1]


def anls(prediction: str, targets: Sequence[str], threshold: float = 0.5) -> float:
    prediction = normalize_answer(prediction, "docvqa")
    scores = []
    for target in targets:
        normalized_target = normalize_answer(target, "docvqa")
        denominator = max(len(prediction), len(normalized_target), 1)
        similarity = 1.0 - _levenshtein(prediction, normalized_target) / denominator
        scores.append(similarity if similarity >= threshold else 0.0)
    return max(scores, default=0.0)


def _mathvista_normalize(value: str, metadata: Mapping[str, Any]) -> str:
    answer_type = str(metadata.get("answer_type", "")).lower()
    cleaned = normalize_answer(value, "mathvista")
    try:
        if answer_type == "integer":
            return str(int(round(float(cleaned))))
        if answer_type == "float":
            precision = int(metadata.get("precision", 2))
            return str(round(float(cleaned), precision))
        if answer_type == "list":
            return str(ast.literal_eval(cleaned))
    except (ValueError, TypeError, SyntaxError):
        return cleaned
    return cleaned


def score_prediction(
    dataset: str,
    prediction: str,
    targets: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> float:
    key = dataset.lower()
    if key == "textvqa":
        return vqa_accuracy(prediction, targets)
    if key == "docvqa":
        return anls(prediction, targets)
    if key == "mathvista":
        meta = metadata or {}
        normalized = _mathvista_normalize(prediction, meta)
        return float(any(normalized == _mathvista_normalize(target, meta) for target in targets))
    return accuracy(prediction, targets, key)


def bootstrap_mean_ci(
    values: Sequence[float], samples: int = 10000, seed: int = 0
) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples)
    )
    low = means[int(0.025 * (samples - 1))]
    high = means[int(0.975 * (samples - 1))]
    return sum(values) / len(values), low, high
