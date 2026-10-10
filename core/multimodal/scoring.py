"""Calibration-free evidence scoring and Borda aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .answers import answers_match
from .schema import Candidate, ComponentScores, EvidenceVerdict, SelectionDecision


@dataclass
class _AnswerGroup:
    key: str
    members: List[Candidate]

    @property
    def representative(self) -> Candidate:
        with_logprobs = [m for m in self.members if m.response.mean_logprob is not None]
        if with_logprobs:
            return max(with_logprobs, key=lambda item: item.response.mean_logprob or float("-inf"))
        return self.members[0]


def _group_candidates(candidates: Sequence[Candidate], dataset: str) -> List[_AnswerGroup]:
    groups: List[_AnswerGroup] = []
    for candidate in candidates:
        matched = next(
            (g for g in groups if answers_match(candidate.normalized_answer, g.key, dataset)),
            None,
        )
        if matched is None:
            groups.append(_AnswerGroup(candidate.normalized_answer, [candidate]))
        else:
            matched.members.append(candidate)
    return groups


def _competition_ranks(values: Mapping[str, float], reverse: bool = True) -> Dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=reverse)
    ranks: Dict[str, float] = {}
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for key, _ in ordered[cursor:end]:
            ranks[key] = average_rank
        cursor = end
    return ranks


def select_with_evidence(
    candidates: Sequence[Candidate],
    counterfactual_answers: Sequence[str],
    verdicts: Mapping[str, EvidenceVerdict],
    dataset: str,
    enabled_components: Sequence[str] = ("invariance", "sensitivity", "evidence"),
    use_logprob_tiebreak: bool = True,
) -> SelectionDecision:
    if not candidates:
        raise ValueError("At least one candidate is required")

    groups = _group_candidates(candidates, dataset)
    total = len(candidates)
    cf_total = max(1, len(counterfactual_answers))
    scores: Dict[str, ComponentScores] = {}
    counterfactual_supports: Dict[str, float] = {}

    for group in groups:
        true_support = len(group.members) / total
        cf_support = sum(
            answers_match(group.key, answer, dataset) for answer in counterfactual_answers
        ) / cf_total
        verdict = verdicts.get(group.key)
        evidence = verdict.score if verdict else 0.5
        counterfactual_supports[group.key] = cf_support
        scores[group.key] = ComponentScores(
            invariance=true_support,
            sensitivity=max(0.0, true_support - cf_support),
            evidence=evidence,
            mean_logprob=group.representative.response.mean_logprob,
        )

    valid_components = {"invariance", "sensitivity", "evidence"}
    components = tuple(enabled_components)
    if not components or any(name not in valid_components for name in components):
        raise ValueError(f"enabled_components must be a non-empty subset of {sorted(valid_components)}")
    component_ranks = {
        name: _competition_ranks({key: getattr(value, name) for key, value in scores.items()})
        for name in components
    }
    n_groups = len(groups)
    borda = {
        key: sum(n_groups - component_ranks[name][key] for name in components)
        for key in scores
    }
    wins = {
        key: sum(component_ranks[name][key] == 1.0 for name in components)
        for key in scores
    }

    def ordering(group: _AnswerGroup) -> Tuple[float, int, float, int]:
        mean_logprob = scores[group.key].mean_logprob
        return (
            borda[group.key],
            wins[group.key],
            mean_logprob if use_logprob_tiebreak and mean_logprob is not None else float("-inf"),
            -candidates.index(group.representative),
        )

    best = max(groups, key=ordering)
    required_wins = len(components) // 2 + 1
    verifier = verdicts.get(best.key)
    evidence_gate = (
        "evidence" not in components
        or (
            verifier is not None
            and not verifier.parse_error
            and verifier.answer_status == "entailed"
            and bool(verifier.claims)
            and all(claim.status == "entailed" for claim in verifier.claims)
        )
    )
    sensitivity_gate = "sensitivity" not in components or scores[best.key].sensitivity > 0.0
    eligible = wins[best.key] >= required_wins and evidence_gate and sensitivity_gate

    negative_groups = [
        group
        for group in groups
        if group.key != best.key
        and (
            scores[group.key].evidence < 0.34
            or counterfactual_supports[group.key] > scores[group.key].invariance
        )
    ]
    negative = min(negative_groups, key=ordering) if negative_groups else None

    return SelectionDecision(
        answer=best.key,
        representative=best.representative,
        scores=scores[best.key],
        borda_score=borda[best.key],
        component_wins=wins[best.key],
        eligible_for_memory=eligible,
        negative_answer=negative.key if negative else None,
        negative_representative=negative.representative if negative else None,
        reason=(
            f"Borda={borda[best.key]:.3f}; component_wins={wins[best.key]}/{len(components)}; "
            f"evidence_gate={evidence_gate}; sensitivity_gate={sensitivity_gate}; "
            f"memory={'accepted' if eligible else 'quarantined'}"
        ),
    )
