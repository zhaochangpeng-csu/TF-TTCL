"""Faithful TF-TTCL candidate partitioning, independent of the old noisy pilot.

Only generated-token log probabilities from each candidate's actual context
are accepted. Missing or incomplete probabilities are a hard protocol error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .schema import GenerationResponse


class LogprobUnavailable(ValueError):
    """The model service cannot support the original generation-PPL rule."""


def strict_generation_ppl(response: GenerationResponse) -> float:
    """Equation (5): exp(-mean generated-token logprob), without imputation."""
    n = response.completion_tokens
    values = response.logprobs
    if n <= 0 or len(values) != n:
        raise LogprobUnavailable(f"Expected {n} generated token logprobs, received {len(values)}")
    if any(not math.isfinite(value) or value > 1e-6 for value in values):
        raise LogprobUnavailable("Generated token logprobs are nonfinite or positive")
    try:
        ppl = math.exp(-sum(values) / n)
    except OverflowError as error:
        raise LogprobUnavailable("Generation PPL overflowed") from error
    if not math.isfinite(ppl):
        raise LogprobUnavailable("Generation PPL is not finite")
    return ppl


@dataclass(frozen=True)
class ScoredCandidate:
    role: str
    answer: str
    response: GenerationResponse
    ppl: float

    @classmethod
    def from_response(cls, role: str, answer: str,
                      response: GenerationResponse) -> "ScoredCandidate":
        if not answer:
            raise ValueError("Candidate answer must be parsed before selection")
        return cls(role, answer, response, strict_generation_ppl(response))


@dataclass(frozen=True)
class DirectSelection:
    positive: ScoredCandidate
    negative: ScoredCandidate | None
    positive_roles: tuple[str, ...]
    negative_roles: tuple[str, ...]
    reason: str
    summarize_positive: bool
    summarize_negative: bool
    similarity_ranking: tuple[tuple[str, float], ...] = ()


def _validate(candidates: Sequence[ScoredCandidate]) -> None:
    if len(candidates) != 5 or [c.role for c in candidates] != [
            "teacher", "student_1", "student_2", "student_3", "student_4"]:
        raise ValueError("Expected Teacher followed by four Students")
    if any(not c.answer or not math.isfinite(c.ppl) or c.ppl < 1 for c in candidates):
        raise ValueError("All candidates require finite generation PPL and parsed answers")


def select_closed(candidates: Sequence[ScoredCandidate]) -> DirectSelection:
    """ScienceQA closed-task cases from the paper and original selection code."""
    _validate(candidates)
    groups: dict[str, list[ScoredCandidate]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.answer, []).append(candidate)
    if len(groups) == 5:
        return DirectSelection(candidates[0], None, (), tuple(c.role for c in candidates),
                               "all_distinct_teacher_no_summary", False, False)
    if len(groups) == 1:
        best = min(candidates, key=lambda c: c.ppl)
        return DirectSelection(best, None, tuple(c.role for c in candidates), (),
                               "all_same_positive_rule", True, False)
    largest = max(len(group) for group in groups.values())
    tied = [group for group in groups.values() if len(group) == largest]
    winning_group = min(tied, key=lambda group: min(c.ppl for c in group))
    positive = min(winning_group, key=lambda c: c.ppl)
    negative_group = [c for c in candidates if c.answer != positive.answer]
    negative = min(negative_group, key=lambda c: c.ppl)
    return DirectSelection(positive, negative,
                           tuple(c.role for c in winning_group),
                           tuple(c.role for c in negative_group),
                           "largest_cluster" if len(tied) == 1 else "tied_cluster_min_ppl",
                           True, True)


def select_open(candidates: Sequence[ScoredCandidate],
                student_similarities: Sequence[float]) -> DirectSelection:
    """Original open-task branch: Teacher plus top half of four Students."""
    _validate(candidates)
    if len(student_similarities) != 4 or any(not math.isfinite(x) for x in student_similarities):
        raise ValueError("Four finite Teacher-to-Student semantic similarities required")
    ranking = sorted(zip(candidates[1:], student_similarities), key=lambda item: -item[1])
    positive_group = [candidates[0], *(candidate for candidate, _ in ranking[:2])]
    negative_group = [candidate for candidate, _ in ranking[2:]]
    return DirectSelection(min(positive_group, key=lambda c: c.ppl),
                           min(negative_group, key=lambda c: c.ppl),
                           tuple(c.role for c in positive_group),
                           tuple(c.role for c in negative_group),
                           "teacher_similarity_top_half", True, True,
                           tuple((c.role, float(score)) for c, score in ranking))
