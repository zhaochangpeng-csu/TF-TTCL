"""Label-free policies for a separately frozen noisy-experience transfer pilot.

These are pseudo-labels, NOT correctness judgments. Noisy/no-memory/random use
the same majority pair. Evidence gates affect writes, not current selection.
Actual reasoning collection, rule distillation and online evaluation must be
supplied by the pilot runner; this module alone is not a memory experiment.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from .schema import Candidate


@dataclass(frozen=True)
class PseudoPair:
    positive: Candidate
    negative: Candidate | None
    consensus: float
    writable: bool
    reason: str


def majority_pair(candidates: Sequence[Candidate]) -> PseudoPair:
    """CRT-style grouping, with explicit deterministic no-logprob fallback.

    When available, highest mean logprob (= lowest mean NLL/PPL) selects the
    representative and resolves group ties. Without it, trajectory order wins.
    An all-distinct multi-candidate set produces no experience. A different
    answer is merely a pseudo-negative; disagreement does not establish error.
    """
    if not candidates or any(not c.normalized_answer for c in candidates):
        raise ValueError("Require nonempty parsed candidate answers")
    groups = {}
    for candidate in candidates:
        groups.setdefault(candidate.normalized_answer, []).append(candidate)
    counts = Counter({answer: len(group) for answer, group in groups.items()})

    def representative(group):
        if all(c.response.mean_logprob is not None for c in group):
            return max(group, key=lambda c: c.response.mean_logprob)
        return group[0]

    winners = [answer for answer, n in counts.items() if n == max(counts.values())]
    winner_reps = [representative(groups[answer]) for answer in winners]
    positive = representative(winner_reps)
    others = [c for c in candidates if c.normalized_answer != positive.normalized_answer]
    negative = representative(others) if others else None
    all_distinct = len(candidates) > 1 and len(groups) == len(candidates)
    return PseudoPair(positive, negative, counts[positive.normalized_answer] / len(candidates),
                      not all_distinct,
                      "all_distinct_skip" if all_distinct else
                      ("consensus_pseudo_pair" if len(winners) == 1 else "tie_pseudo_pair_not_certified"))


def answer_content(candidate: Candidate, choices: Sequence[str]) -> str:
    """Persist semantic option text; the letter's meaning changes across items."""
    answer = candidate.normalized_answer.strip().lower()
    if len(answer) == 1 and "a" <= answer <= "z" and choices:
        index = ord(answer) - ord("a")
        if index >= len(choices):
            raise ValueError("Option letter outside current choices")
        return str(choices[index])
    return candidate.normalized_answer


@dataclass(frozen=True)
class WriteDecision:
    write: bool
    trust: float
    reason: str


def write_policy(policy: str, pair: PseudoPair, *, evidence_passed: bool = False,
                 evidence_answer: str | None = None, rule_valid: bool = True) -> WriteDecision:
    if policy not in {"none", "noisy", "gated_weighted", "random"}:
        raise ValueError("Unknown transfer policy")
    if policy == "none":
        return WriteDecision(False, 0.0, "no_memory_paid_distillation_control")
    if not pair.writable or not rule_valid:
        return WriteDecision(False, 0.0, "invalid_rule_or_all_distinct")
    if policy in {"noisy", "random"}:
        return WriteDecision(True, 1.0, "uncertified_pseudo_pair")
    if not evidence_passed or evidence_answer != pair.positive.normalized_answer:
        return WriteDecision(False, 0.0, "evidence_does_not_support_same_pseudo_positive")
    # Consensus is a heuristic ranking weight, not a calibrated probability.
    if not math.isfinite(pair.consensus) or not 0 < pair.consensus <= 1:
        raise ValueError("Invalid consensus weight")
    return WriteDecision(True, pair.consensus, "evidence_gated_consensus_weight_not_probability")


def deterministic_random_past(items, sample_index: int, seed: int, top_k: int):
    """Random retrieval control with exactly the same noisy write pool.

    Entries carry `sample_index` and `experience_id`. No current/future entry
    can be retrieved, even if supplied by a caller. Stable hash ordering makes
    restart independent of Python RNG state and iteration order.
    """
    if top_k < 0:
        raise ValueError("top_k must be nonnegative")
    eligible = [item for item in items if item.sample_index < sample_index]
    return sorted(eligible, key=lambda item: (
        hashlib.sha256(f"{seed}:{sample_index}:{item.experience_id}".encode()).hexdigest(),
        item.experience_id))[:top_k]
