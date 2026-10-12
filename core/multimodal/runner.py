"""Single-pass EviMem-TTCL experiment runner."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import time
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .answers import normalize_answer
from .client import (
    VLMBackend,
    parse_contrastive_verdict,
    parse_evidence_certificate,
    parse_evidence_verdict,
    parse_candidate_grounding,
    parse_question_grounding,
    parse_json_object,
)
from .image_ops import (
    make_blank_views,
    make_fidelity_views,
    make_grid_intervention_views,
    sha256_file,
)
from .memory import EvidenceMemory, RetrievalBundle
from .prompts import (
    ANSWER_SYSTEM,
    CANDIDATE_LOCALIZE_SYSTEM,
    CERTIFICATE_VERIFY_SYSTEM,
    CONTRASTIVE_VERIFY_SYSTEM,
    DISTILL_SYSTEM,
    PARAPHRASE_SYSTEM,
    QUESTION_LOCALIZE_SYSTEM,
    VERIFY_SYSTEM,
    format_memory_context,
    format_question,
)
from .schema import (
    Candidate,
    ComponentScores,
    EvidenceVerdict,
    Experience,
    GenerationResponse,
    MultimodalSample,
    SamplingConfig,
    SelectionDecision,
)
from .scoring import select_with_evidence
from .shared_evidence import (
    MAGNET_FACT_SYSTEM, THERMAL_FACT_SYSTEM, SHARED_PROMPTS, VERSION as SHARED_VERSION,
    assess_certificate, parse_observation, route_operator,
)


LOGGER = logging.getLogger(__name__)
METHODS = {
    "zero_shot",
    "cot",
    "self_consistency",
    "budget_matched",
    "contrastive_budget",
    "contrastive_evidence",
    "certificate_budget",
    "certificate_evidence",
    "causal_budget",
    "causal_evidence",
    "shared_budget",
    "shared_evidence",
    "direct_ttcl",
    "evidence_only",
    "evimem",
}

_QUESTION_MARKER = "\n\nQuestion:\n"
_LOGIC_TOKENS = {
    "after",
    "before",
    "different",
    "except",
    "false",
    "least",
    "less",
    "more",
    "most",
    "never",
    "no",
    "not",
    "same",
    "true",
    "without",
}
_QUESTION_STARTERS = {
    "can",
    "could",
    "describe",
    "does",
    "how",
    "is",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "would",
}


def _split_context_and_query(question: str) -> Tuple[str, str]:
    """Keep benchmark context immutable while augmenting only the query clause."""
    if _QUESTION_MARKER in question:
        context, query = question.rsplit(_QUESTION_MARKER, 1)
        return context + _QUESTION_MARKER, query
    return "", question


def _protected_query_features(text: str) -> Tuple[Tuple[str, ...], ...]:
    normalized = text.strip()
    words = re.findall(r"[A-Za-z]+", normalized.lower())
    logic = tuple(sorted(word for word in words if word in _LOGIC_TOKENS))
    numbers = tuple(sorted(re.findall(r"[-+]?\d+(?:\.\d+)?%?", normalized)))
    quoted = tuple(
        sorted(match[1].strip().lower() for match in re.findall(r"(['\"])(.+?)\1", normalized))
    )
    proper = []
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9'-]*\b", normalized):
        if match.group(0).lower() in _QUESTION_STARTERS:
            continue
        proper.append(match.group(0).lower())
    return numbers, logic, quoted, tuple(sorted(proper))


def _semantic_variant_is_safe(original: str, variant: str) -> bool:
    cleaned = variant.strip()
    if not cleaned or "final answer" in cleaned.lower():
        return False
    if len(cleaned) < max(4, int(0.35 * len(original.strip()))):
        return False
    original_features = _protected_query_features(original)
    variant_features = _protected_query_features(cleaned)
    if original_features[:3] != variant_features[:3]:
        return False
    variant_words = set(re.findall(r"[A-Za-z0-9'-]+", cleaned.lower()))
    return all(name in variant_words for name in original_features[3])


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _scientific_config_hash(config: Mapping[str, Any]) -> str:
    """Hash scientific settings while ignoring the operational resume toggle."""
    normalized = json.loads(_stable_json(config))
    experiment = normalized.get("experiment")
    if isinstance(experiment, dict):
        experiment.pop("resume", None)
    return _sha256_text(_stable_json(normalized))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _majority_decision(candidates: Sequence[Candidate]) -> SelectionDecision:
    counts = Counter(item.normalized_answer for item in candidates)
    first_index = {answer: next(i for i, c in enumerate(candidates) if c.normalized_answer == answer) for answer in counts}
    best_answer = max(counts, key=lambda answer: (counts[answer], -first_index[answer]))
    best_items = [item for item in candidates if item.normalized_answer == best_answer]
    representative = max(
        best_items,
        key=lambda item: item.response.mean_logprob
        if item.response.mean_logprob is not None
        else float("-inf"),
    )
    ordered = counts.most_common()
    unique_majority = len(ordered) == 1 or ordered[0][1] > ordered[1][1]
    negative_answer = ordered[-1][0] if len(ordered) > 1 else None
    negative = next(
        (item for item in candidates if item.normalized_answer == negative_answer), None
    )
    return SelectionDecision(
        answer=best_answer,
        representative=representative,
        scores=ComponentScores(counts[best_answer] / len(candidates), 0.0, 0.5, representative.response.mean_logprob),
        borda_score=float(counts[best_answer]),
        component_wins=1 if unique_majority else 0,
        eligible_for_memory=unique_majority,
        negative_answer=negative_answer,
        negative_representative=negative,
        reason=f"majority={counts[best_answer]}/{len(candidates)}; unique={unique_majority}",
    )


def _greedy_decision(candidates: Sequence[Candidate]) -> SelectionDecision:
    if not candidates:
        raise ValueError("At least one candidate is required")
    representative = next(
        (item for item in candidates if item.trajectory == "original_greedy"),
        candidates[0],
    )
    support = sum(
        item.normalized_answer == representative.normalized_answer for item in candidates
    ) / len(candidates)
    return SelectionDecision(
        answer=representative.normalized_answer,
        representative=representative,
        scores=ComponentScores(support, 0.0, 0.5, representative.response.mean_logprob),
        borda_score=support,
        component_wins=0,
        eligible_for_memory=False,
        reason="certificate=fallback_original_greedy",
    )


class EviMemRunner:
    def __init__(
        self,
        model: VLMBackend,
        memory: EvidenceMemory,
        output_dir: Path,
        method: str = "evimem",
        seed: int = 0,
        max_tokens: int = 256,
        request_logprobs: bool = False,
        resume: bool = True,
        config_snapshot: Optional[Mapping[str, Any]] = None,
        ablation: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; expected one of {sorted(METHODS)}")
        self.model = model
        self.memory = memory
        self.output_dir = output_dir
        self.method = method
        self.seed = seed
        self.max_tokens = max_tokens
        self.request_logprobs = request_logprobs
        self.resume = resume
        self.config_snapshot = dict(config_snapshot or {})
        self.ablation = {
            "text_variants": True,
            "image_variant": True,
            "joint_variant": True,
            "invariance": True,
            "sensitivity": True,
            "evidence": True,
            "logprob_tiebreak": True,
            "rule_memory": True,
            "episode_memory": True,
            "positive_memory": True,
            "negative_memory": True,
            "memory_gate": True,
            "retrieval_modalities": ["text", "image", "joint"],
            **dict(ablation or {}),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_cache_dir = self.output_dir / "image_cache"
        self.predictions_path = self.output_dir / "predictions.jsonl"
        self.memory_path = self.output_dir / "memory.jsonl"
        self.manifest_path = self.output_dir / "manifest.json"
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency = 0.0
        self._prompt_hash = _sha256_text(
            "\n".join(
                [
                    ANSWER_SYSTEM,
                    PARAPHRASE_SYSTEM,
                    VERIFY_SYSTEM,
                    CONTRASTIVE_VERIFY_SYSTEM,
                    CERTIFICATE_VERIFY_SYSTEM,
                    QUESTION_LOCALIZE_SYSTEM,
                    CANDIDATE_LOCALIZE_SYSTEM,
                    DISTILL_SYSTEM,
                ]
            )
        )
        if self.method in {"shared_budget", "shared_evidence"}:
            self._prompt_hash = _sha256_text(
                self._prompt_hash + "\n" + "\n".join(SHARED_PROMPTS)
                + "\n" + sha256_file(str(Path(__file__).with_name("shared_evidence.py")))
            )

    def _sampling(self, temperature: float, offset: int, max_tokens: Optional[int] = None) -> SamplingConfig:
        return SamplingConfig(
            temperature=temperature,
            top_p=0.9 if temperature else None,
            max_tokens=max_tokens or self.max_tokens,
            seed=self.seed + offset,
            request_logprobs=self.request_logprobs,
        )

    def _generate(
        self,
        system: str,
        text: str,
        images: Sequence[str],
        sampling: SamplingConfig,
    ) -> GenerationResponse:
        response = self.model.generate(system, text, images, sampling)
        self.total_prompt_tokens += response.prompt_tokens
        self.total_completion_tokens += response.completion_tokens
        self.total_latency += response.latency_seconds
        return response

    def _paraphrases(self, question: str, sample_index: int) -> List[str]:
        context, query = _split_context_and_query(question)
        response = self._generate(
            PARAPHRASE_SYSTEM,
            query,
            (),
            self._sampling(0.0, sample_index * 100 + 1, max_tokens=256),
        )
        try:
            payload = parse_json_object(response.content)
            proposed = [
                str(item).strip()
                for item in payload.get("questions", [])
                if str(item).strip()
            ]
        except (ValueError, TypeError, json.JSONDecodeError):
            proposed = []
        variants = [
            item if _semantic_variant_is_safe(query, item) else query
            for item in proposed[:2]
        ]
        while len(variants) < 2:
            variants.append(query)
        return [context + item for item in variants[:2]]

    def _candidate(
        self,
        sample: MultimodalSample,
        trajectory: str,
        question: str,
        images: Sequence[str],
        system: str,
        sampling: SamplingConfig,
        memory_conditioned: bool = False,
    ) -> Candidate:
        answer_metadata = {**sample.metadata, "choices": list(sample.choices)}
        response = self._generate(
            system,
            format_question(question, list(sample.choices), answer_metadata),
            images,
            sampling,
        )
        return Candidate(
            candidate_id=f"{sample.sample_id}:{trajectory}",
            trajectory=trajectory,
            response=response,
            normalized_answer=normalize_answer(response.content, sample.dataset, answer_metadata),
            question_variant=question,
            image_paths=tuple(images),
            memory_conditioned=memory_conditioned,
        )

    def _generate_candidates(
        self,
        sample: MultimodalSample,
        sample_index: int,
        memory_context: str,
        fidelity_images: Sequence[str],
    ) -> List[Candidate]:
        question = sample.question
        variants = (
            self._paraphrases(question, sample_index)
            if self.ablation["text_variants"] or self.ablation["joint_variant"]
            else [question, question]
        )
        clean_system = ANSWER_SYSTEM
        memory_system = ANSWER_SYSTEM + ("\n\n" + memory_context if memory_context else "")
        specs = [
            ("original_greedy", question, sample.image_paths, 0.0, False),
            ("original_sampled", question, sample.image_paths, 0.7, False),
        ]
        if self.ablation["text_variants"]:
            specs.extend(
                [
                    ("text_variant_1", variants[0], sample.image_paths, 0.0, True),
                    ("text_variant_2", variants[1], sample.image_paths, 0.0, True),
                ]
            )
        if self.ablation["image_variant"]:
            specs.append(("image_variant", question, fidelity_images or sample.image_paths, 0.0, False))
        if self.ablation["joint_variant"]:
            specs.append(("joint_variant", variants[0], fidelity_images or sample.image_paths, 0.0, True))
        return [
            self._candidate(
                sample,
                name,
                variant,
                images,
                memory_system if use_memory else clean_system,
                self._sampling(temperature, sample_index * 100 + 10 + offset),
                memory_conditioned=bool(use_memory and memory_context),
            )
            for offset, (name, variant, images, temperature, use_memory) in enumerate(specs)
        ]

    def _counterfactual_answers(
        self,
        sample: MultimodalSample,
        sample_index: int,
        blank_images: Sequence[str],
        mismatch_images: Sequence[str],
    ) -> List[str]:
        results = []
        answer_metadata = {**sample.metadata, "choices": list(sample.choices)}
        probes = [("blank", blank_images)]
        if mismatch_images:
            probes.append(("mismatch", mismatch_images))
        for offset, (name, images) in enumerate(probes):
            response = self._generate(
                ANSWER_SYSTEM,
                format_question(sample.question, list(sample.choices), answer_metadata),
                images,
                self._sampling(0.0, sample_index * 100 + 30 + offset),
            )
            results.append(normalize_answer(response.content, sample.dataset, answer_metadata))
        return results

    def _verify_top_answers(
        self,
        sample: MultimodalSample,
        candidates: Sequence[Candidate],
        sample_index: int,
    ) -> Dict[str, EvidenceVerdict]:
        counts = Counter(candidate.normalized_answer for candidate in candidates)
        first = {answer: next(i for i, c in enumerate(candidates) if c.normalized_answer == answer) for answer in counts}
        top_answers = sorted(counts, key=lambda answer: (-counts[answer], first[answer]))[:2]
        verdicts: Dict[str, EvidenceVerdict] = {}
        answer_metadata = {**sample.metadata, "choices": list(sample.choices)}
        for offset, answer in enumerate(top_answers):
            representative = next(c for c in candidates if c.normalized_answer == answer)
            prompt = (
                f"Question:\n{format_question(sample.question, list(sample.choices), answer_metadata)}\n\n"
                f"Proposed answer:\n{representative.response.content}"
            )
            response = self._generate(
                VERIFY_SYSTEM,
                prompt,
                sample.image_paths,
                self._sampling(0.0, sample_index * 100 + 40 + offset, max_tokens=384),
            )
            verdicts[answer] = parse_evidence_verdict(response.content)
        return verdicts

    def _contrastive_adjudication(
        self,
        sample: MultimodalSample,
        candidates: Sequence[Candidate],
        sample_index: int,
        fidelity_images: Sequence[str],
    ) -> Tuple[SelectionDecision, List[Dict[str, Any]]]:
        majority = _majority_decision(candidates)
        groups: List[Tuple[str, List[Candidate]]] = []
        for candidate in candidates:
            group = next(
                (item for item in groups if item[0] == candidate.normalized_answer),
                None,
            )
            if group is None:
                groups.append((candidate.normalized_answer, [candidate]))
            else:
                group[1].append(candidate)
        if len(groups) < 2:
            majority.eligible_for_memory = False
            majority.reason += "; contrastive=unanimous_no_negative"
            return majority, []

        groups.sort(key=lambda item: (-len(item[1]), candidates.index(item[1][0])))
        first_answer = groups[0][0]
        second_answer = groups[1][0]
        answer_metadata = {**sample.metadata, "choices": list(sample.choices)}
        question = format_question(sample.question, list(sample.choices), answer_metadata)

        def display_answer(answer: str) -> str:
            if len(answer) == 1 and "a" <= answer <= "z":
                choice_index = ord(answer) - ord("a")
                if choice_index < len(sample.choices):
                    return f"option {answer.upper()} — {sample.choices[choice_index]}"
            return answer

        views = [
            ("original", sample.image_paths, first_answer, second_answer),
            (
                "fidelity_swapped",
                fidelity_images or sample.image_paths,
                second_answer,
                first_answer,
            ),
        ]
        resolved: List[str] = []
        records: List[Dict[str, Any]] = []
        loser_contradicted = False
        valid_views = True
        for offset, (view, images, candidate_1, candidate_2) in enumerate(views):
            prompt = (
                f"Question and options:\n{question}\n\n"
                f"RESPONSE_X proposed answer: {display_answer(candidate_1)}\n"
                f"RESPONSE_Y proposed answer: {display_answer(candidate_2)}"
            )
            response = self._generate(
                CONTRASTIVE_VERIFY_SYSTEM,
                prompt,
                images,
                self._sampling(0.0, sample_index * 100 + 60 + offset, max_tokens=384),
            )
            verdict = parse_contrastive_verdict(response.content)
            slot_to_answer = {
                "candidate_1": candidate_1,
                "candidate_2": candidate_2,
            }
            statuses = {
                "candidate_1": verdict.candidate_1_status,
                "candidate_2": verdict.candidate_2_status,
            }
            entailed_slots = [slot for slot, status in statuses.items() if status == "entailed"]
            status_winner = entailed_slots[0] if len(entailed_slots) == 1 else None
            loser_slot = (
                "candidate_2" if status_winner == "candidate_1" else "candidate_1"
                if status_winner == "candidate_2" else None
            )
            loser_status = statuses.get(loser_slot, "insufficient")
            declared_winner_compatible = verdict.winner in {
                status_winner,
                "undecidable",
            }
            view_valid = (
                not verdict.parse_error
                and status_winner in slot_to_answer
                and declared_winner_compatible
                and loser_status in {"contradicted", "insufficient"}
                and bool(verdict.decisive_evidence)
            )
            if view_valid:
                resolved.append(slot_to_answer[status_winner])
                loser_contradicted = loser_contradicted or loser_status == "contradicted"
            else:
                valid_views = False
            records.append(
                {
                    "view": view,
                    "candidate_1": candidate_1,
                    "candidate_2": candidate_2,
                    "raw_response": response.content,
                    "verdict": asdict(verdict),
                    "status_winner": status_winner,
                    "resolved_answer": slot_to_answer.get(status_winner),
                    "valid": view_valid,
                }
            )

        stable = (
            valid_views
            and len(resolved) == 2
            and resolved[0] == resolved[1]
        )
        if not stable:
            majority.eligible_for_memory = False
            majority.reason += "; contrastive=fallback_unstable"
            return majority, records

        winner = resolved[0]
        winner_group = next(item for item in groups if item[0] == winner)
        loser_answer = second_answer if winner == first_answer else first_answer
        loser_group = next(item for item in groups if item[0] == loser_answer)
        support = len(winner_group[1]) / len(candidates)
        return (
            SelectionDecision(
                answer=winner,
                representative=winner_group[1][0],
                scores=ComponentScores(support, 1.0, 1.0),
                borda_score=2.0,
                component_wins=3 if loser_contradicted else 2,
                eligible_for_memory=loser_contradicted,
                negative_answer=loser_answer,
                negative_representative=loser_group[1][0],
                reason=(
                    "contrastive=stable_two_view; evidence_gate=True; "
                    + ("memory=accepted" if loser_contradicted else "memory=rejected_no_contradiction")
                ),
            ),
            records,
        )

    def _certificate_adjudication(
        self,
        sample: MultimodalSample,
        candidates: Sequence[Candidate],
        sample_index: int,
        fidelity_images: Sequence[str],
        blank_images: Sequence[str],
        mismatch_images: Sequence[str],
    ) -> Tuple[SelectionDecision, Dict[str, Any]]:
        greedy = _greedy_decision(candidates)
        greedy_answer = greedy.answer
        alternatives = [
            answer
            for answer, _ in Counter(
                item.normalized_answer
                for item in candidates
                if item.normalized_answer != greedy_answer
            ).most_common()
        ]
        audit: Dict[str, Any] = {
            "greedy_answer": greedy_answer,
            "alternative_answer": alternatives[0] if alternatives else None,
            "real_stable": False,
            "counterfactual_drop": False,
            "gate_passed": False,
            "changed_from_greedy": False,
            "fallback_reason": "no_alternative" if not alternatives else "unverified",
            "views": [],
        }
        if not alternatives:
            return greedy, audit

        alternative = alternatives[0]
        answer_metadata = {**sample.metadata, "choices": list(sample.choices)}
        question = format_question(sample.question, list(sample.choices), answer_metadata)

        def display_answer(answer: str) -> str:
            if len(answer) == 1 and "a" <= answer <= "z":
                choice_index = ord(answer) - ord("a")
                if choice_index < len(sample.choices):
                    return f"option {answer.upper()} — {sample.choices[choice_index]}"
            return answer

        views: List[Tuple[str, Sequence[str], str, str, str]] = [
            ("original", sample.image_paths, greedy_answer, alternative, "real"),
            (
                "fidelity_swapped",
                fidelity_images or sample.image_paths,
                alternative,
                greedy_answer,
                "real",
            ),
            ("blank", blank_images, greedy_answer, alternative, "counterfactual"),
        ]
        if mismatch_images:
            views.append(
                ("mismatch_swapped", mismatch_images, alternative, greedy_answer, "counterfactual")
            )

        real_winners: List[str] = []
        counterfactual_records: List[Dict[str, Any]] = []
        for offset, (view, images, candidate_1, candidate_2, view_kind) in enumerate(views):
            prompt = (
                f"Question and options:\n{question}\n\n"
                f"RESPONSE_X proposed answer: {display_answer(candidate_1)}\n"
                f"RESPONSE_Y proposed answer: {display_answer(candidate_2)}"
            )
            response = self._generate(
                CERTIFICATE_VERIFY_SYSTEM,
                prompt,
                images,
                self._sampling(0.0, sample_index * 100 + 70 + offset, max_tokens=384),
            )
            verdict = parse_evidence_certificate(response.content)
            slot_to_answer = {"candidate_1": candidate_1, "candidate_2": candidate_2}
            statuses = {
                "candidate_1": verdict.candidate_1_status,
                "candidate_2": verdict.candidate_2_status,
            }
            visual_support = {
                "candidate_1": verdict.candidate_1_visual_support,
                "candidate_2": verdict.candidate_2_visual_support,
            }
            entailed = [slot for slot, status in statuses.items() if status == "entailed"]
            status_winner = entailed[0] if len(entailed) == 1 else None
            resolved_answer = slot_to_answer.get(status_winner)
            winner_has_atom = bool(
                status_winner
                and any(atom.supports == status_winner for atom in verdict.evidence_atoms)
            )
            real_valid = bool(
                view_kind == "real"
                and not verdict.parse_error
                and verdict.winner == status_winner
                and status_winner in slot_to_answer
                and visual_support.get(status_winner) == "present"
                and winner_has_atom
            )
            record = {
                "view": view,
                "view_kind": view_kind,
                "candidate_1": candidate_1,
                "candidate_2": candidate_2,
                "image_paths": list(images),
                "image_hashes": [sha256_file(path) for path in images],
                "raw_response": response.content,
                "verdict": asdict(verdict),
                "status_winner": status_winner,
                "resolved_answer": resolved_answer,
                "real_valid": real_valid,
            }
            if view_kind == "real" and real_valid and resolved_answer is not None:
                real_winners.append(resolved_answer)
            if view_kind == "counterfactual":
                counterfactual_records.append(record)
            audit["views"].append(record)

        real_stable = len(real_winners) == 2 and real_winners[0] == real_winners[1]
        winner = real_winners[0] if real_stable else None
        counterfactual_drop = bool(winner and counterfactual_records)
        if winner:
            for record in counterfactual_records:
                verdict = record["verdict"]
                winner_slot = (
                    "candidate_1" if record["candidate_1"] == winner else "candidate_2"
                )
                winner_status = verdict[f"{winner_slot}_status"]
                winner_support = verdict[f"{winner_slot}_visual_support"]
                winner_atoms = [
                    atom
                    for atom in verdict["evidence_atoms"]
                    if atom["supports"] == winner_slot
                ]
                dropped = (
                    not verdict["parse_error"]
                    and winner_status != "entailed"
                    and winner_support in {"absent", "conflicting"}
                    and not winner_atoms
                )
                record["winner_visual_support_dropped"] = dropped
                counterfactual_drop = counterfactual_drop and dropped

        gate_passed = bool(real_stable and counterfactual_drop and winner)
        audit.update(
            {
                "real_stable": real_stable,
                "counterfactual_drop": counterfactual_drop,
                "gate_passed": gate_passed,
                "changed_from_greedy": bool(gate_passed and winner != greedy_answer),
                "fallback_reason": None if gate_passed else (
                    "unstable_real_views" if not real_stable else "counterfactual_support_did_not_drop"
                ),
            }
        )
        if not gate_passed or winner is None:
            greedy.reason += f"; certificate={audit['fallback_reason']}"
            return greedy, audit

        representative = next(
            item for item in candidates if item.normalized_answer == winner
        )
        support = sum(item.normalized_answer == winner for item in candidates) / len(candidates)
        decision = SelectionDecision(
            answer=winner,
            representative=representative,
            scores=ComponentScores(support, 1.0, 1.0, representative.response.mean_logprob),
            borda_score=2.0,
            component_wins=2,
            eligible_for_memory=False,
            negative_answer=(alternative if winner == greedy_answer else greedy_answer),
            negative_representative=next(
                (
                    item
                    for item in candidates
                    if item.normalized_answer
                    == (alternative if winner == greedy_answer else greedy_answer)
                ),
                None,
            ),
            reason=(
                "certificate=accepted_real_stable_counterfactual_drop; "
                f"changed_from_greedy={winner != greedy_answer}; memory=disabled"
            ),
        )
        return decision, audit

    def _shared_adjudication(
        self, sample: MultimodalSample, candidates: Sequence[Candidate],
        sample_index: int, fidelity_images: Sequence[str],
        blank_images: Sequence[str], mismatch_images: Sequence[str],
    ) -> Tuple[SelectionDecision, Dict[str, Any]]:
        greedy = _greedy_decision(candidates)
        operator = route_operator(sample.question, sample.choices)
        audit: Dict[str, Any] = {
            "version": SHARED_VERSION, "operator": operator,
            "greedy_answer": greedy.answer,
            "candidate_disagreement": len({c.normalized_answer for c in candidates}) > 1,
            "facts_stable": False, "counterfactual_drop": False,
            "gate_passed": False, "changed_from_greedy": False,
            "fallback_reason": "unsupported_task", "observations": [],
        }
        if operator is None or len(sample.image_paths) != 1:
            if operator is not None:
                audit["fallback_reason"] = "requires_single_image"
            return greedy, audit
        system = MAGNET_FACT_SYSTEM if operator == "facing_poles" else THERMAL_FACT_SYSTEM
        specs = [("original", sample.image_paths),
                 ("fidelity", fidelity_images or sample.image_paths),
                 ("blank", blank_images)]
        if mismatch_images:
            specs.append(("history_mismatch", mismatch_images))
        parsed = []
        for offset, (view, images) in enumerate(specs):
            response = self._generate(
                system, "Transcribe only the visible observations in the specified JSON schema.",
                images[:1], self._sampling(0.0, sample_index * 100 + 86 + offset, max_tokens=256),
            )
            observation = parse_observation(response.content, operator)
            parsed.append(observation)
            audit["observations"].append({
                "view": view, "raw_response": response.content, "parsed": observation,
                "image_hashes": [sha256_file(p) for p in images[:1]],
                "question_exposed": False, "choices_exposed": False,
                "candidate_answers_exposed": False,
            })
        audit.update(assess_certificate(
            operator, parsed, sample.choices, [c.normalized_answer for c in candidates],
        ))
        if not audit["gate_passed"]:
            greedy.reason += f"; shared={audit['fallback_reason']}"
            return greedy, audit
        winner = audit["computed_answer"]
        audit["changed_from_greedy"] = winner != greedy.answer
        representative = next(c for c in candidates if c.normalized_answer == winner)
        support = sum(c.normalized_answer == winner for c in candidates) / len(candidates)
        return SelectionDecision(
            answer=winner, representative=representative,
            scores=ComponentScores(support, 1.0, 1.0), borda_score=2.0,
            component_wins=2, eligible_for_memory=False,
            reason="shared=stable_observations_executable_reasoning; memory=disabled",
        ), audit

    def _causal_adjudication(
        self,
        sample: MultimodalSample,
        candidates: Sequence[Candidate],
        sample_index: int,
        fidelity_images: Sequence[str],
    ) -> Tuple[SelectionDecision, Dict[str, Any]]:
        """Conservatively override greedy only after a symmetric causal crossover.

        Localization is only a proposal. Selection is determined by executable
        focus/drop interventions and fails closed whenever the two answers share
        a causal footprint or the answer-blind question region is ambiguous.
        """
        greedy = _greedy_decision(candidates)
        greedy_answer = greedy.answer
        alternatives = [
            answer
            for answer, _ in Counter(
                item.normalized_answer
                for item in candidates
                if item.normalized_answer != greedy_answer
            ).most_common()
        ]
        audit: Dict[str, Any] = {
            "version": "ccic_v1_grid4_strict",
            "greedy_answer": greedy_answer,
            "alternative_answer": alternatives[0] if alternatives else None,
            "localization_stable": False,
            "candidate_regions_disjoint": False,
            "question_region_alternative_exclusive": False,
            "causal_crossover": False,
            "gate_passed": False,
            "changed_from_greedy": False,
            "fallback_reason": "no_alternative" if not alternatives else "unverified",
            "localizations": [],
            "interventions": [],
        }
        if not alternatives or not sample.image_paths:
            if alternatives and not sample.image_paths:
                audit["fallback_reason"] = "no_image"
            return greedy, audit

        alternative = alternatives[0]
        answer_metadata = {**sample.metadata, "choices": list(sample.choices)}
        question = format_question(sample.question, list(sample.choices), answer_metadata)

        def display_answer(answer: str) -> str:
            if len(answer) == 1 and "a" <= answer <= "z":
                choice_index = ord(answer) - ord("a")
                if choice_index < len(sample.choices):
                    return f"option {answer.upper()} — {sample.choices[choice_index]}"
            return answer

        question_specs = [
            ("original_question_only", sample.image_paths),
            ("fidelity_question_only", fidelity_images or sample.image_paths),
        ]
        question_proposals: List[set[str]] = []
        localization_valid = True
        for offset, (view, images) in enumerate(question_specs):
            response = self._generate(
                QUESTION_LOCALIZE_SYSTEM,
                sample.question,
                images,
                self._sampling(0.0, sample_index * 100 + 78 + offset, max_tokens=128),
            )
            proposal = parse_question_grounding(response.content)
            question_proposals.append(set(proposal.cells))
            localization_valid = localization_valid and not proposal.parse_error
            audit["localizations"].append(
                {
                    "kind": "question_only",
                    "view": view,
                    "candidate_answers_exposed": False,
                    "image_paths": list(images),
                    "image_hashes": [sha256_file(path) for path in images],
                    "raw_response": response.content,
                    "proposal": asdict(proposal),
                }
            )

        candidate_specs = [
            ("original", sample.image_paths, greedy_answer, alternative),
            (
                "fidelity_swapped",
                fidelity_images or sample.image_paths,
                alternative,
                greedy_answer,
            ),
        ]
        candidate_proposals: List[Dict[str, set[str]]] = []
        for offset, (view, images, candidate_1, candidate_2) in enumerate(candidate_specs):
            prompt = (
                f"Question and options:\n{question}\n\n"
                f"RESPONSE_X proposed answer: {display_answer(candidate_1)}\n"
                f"RESPONSE_Y proposed answer: {display_answer(candidate_2)}"
            )
            response = self._generate(
                CANDIDATE_LOCALIZE_SYSTEM,
                prompt,
                images,
                self._sampling(0.0, sample_index * 100 + 80 + offset, max_tokens=256),
            )
            proposal = parse_candidate_grounding(response.content)
            by_answer = {
                candidate_1: set(proposal.candidate_1_cells),
                candidate_2: set(proposal.candidate_2_cells),
            }
            localization_valid = localization_valid and not proposal.parse_error
            candidate_proposals.append(by_answer)
            audit["localizations"].append(
                {
                    "kind": "candidate_conditional",
                    "view": view,
                    "candidate_1": candidate_1,
                    "candidate_2": candidate_2,
                    "image_paths": list(images),
                    "image_hashes": [sha256_file(path) for path in images],
                    "raw_response": response.content,
                    "proposal": asdict(proposal),
                }
            )

        stable = bool(
            localization_valid
            and len(question_proposals) == 2
            and len(candidate_proposals) == 2
            and question_proposals[0] == question_proposals[1]
            and candidate_proposals[0][greedy_answer]
            == candidate_proposals[1][greedy_answer]
            and candidate_proposals[0][alternative]
            == candidate_proposals[1][alternative]
        )
        audit["localization_stable"] = stable
        if not stable:
            audit["fallback_reason"] = "unstable_or_malformed_localization"
            greedy.reason += "; causal=unstable_or_malformed_localization"
            return greedy, audit

        question_cells = question_proposals[0]
        greedy_cells = candidate_proposals[0][greedy_answer]
        alternative_cells = candidate_proposals[0][alternative]
        regions_disjoint = greedy_cells.isdisjoint(alternative_cells)
        alternative_exclusive = bool(
            question_cells & alternative_cells
            and not question_cells & greedy_cells
        )
        audit.update(
            {
                "question_cells": sorted(question_cells),
                "greedy_cells": sorted(greedy_cells),
                "alternative_cells": sorted(alternative_cells),
                "candidate_regions_disjoint": regions_disjoint,
                "question_region_alternative_exclusive": alternative_exclusive,
            }
        )
        if not regions_disjoint:
            audit["fallback_reason"] = "overlapping_candidate_regions"
            greedy.reason += "; causal=overlapping_candidate_regions"
            return greedy, audit
        if not alternative_exclusive:
            audit["fallback_reason"] = "question_region_not_alternative_exclusive"
            greedy.reason += "; causal=question_region_not_alternative_exclusive"
            return greedy, audit

        alternative_focus, alternative_drop = make_grid_intervention_views(
            sample.image_paths,
            sorted(alternative_cells),
            self.image_cache_dir,
            sample.sample_id,
            "alternative",
        )
        greedy_focus, greedy_drop = make_grid_intervention_views(
            sample.image_paths,
            sorted(greedy_cells),
            self.image_cache_dir,
            sample.sample_id,
            "greedy",
        )
        intervention_specs = [
            ("alternative_focus", alternative_focus, alternative),
            ("alternative_drop", alternative_drop, greedy_answer),
            ("greedy_focus", greedy_focus, greedy_answer),
            ("greedy_drop", greedy_drop, alternative),
        ]
        observed: Dict[str, str] = {}
        for offset, (name, images, expected) in enumerate(intervention_specs):
            candidate = self._candidate(
                sample,
                f"causal_{name}",
                sample.question,
                images,
                ANSWER_SYSTEM,
                self._sampling(0.0, sample_index * 100 + 82 + offset),
            )
            observed[name] = candidate.normalized_answer
            audit["interventions"].append(
                {
                    "view": name,
                    "expected_answer": expected,
                    "observed_answer": candidate.normalized_answer,
                    "image_paths": list(images),
                    "image_hashes": [sha256_file(path) for path in images],
                    "raw_response": candidate.response.content,
                }
            )

        causal_crossover = all(
            observed[name] == expected
            for name, _, expected in intervention_specs
        )
        audit.update(
            {
                "causal_crossover": causal_crossover,
                "gate_passed": causal_crossover,
                "changed_from_greedy": causal_crossover,
                "fallback_reason": None if causal_crossover else "incomplete_causal_crossover",
            }
        )
        if not causal_crossover:
            greedy.reason += "; causal=incomplete_causal_crossover"
            return greedy, audit

        representative = next(
            item for item in candidates if item.normalized_answer == alternative
        )
        support = sum(
            item.normalized_answer == alternative for item in candidates
        ) / len(candidates)
        decision = SelectionDecision(
            answer=alternative,
            representative=representative,
            scores=ComponentScores(support, 1.0, 1.0, representative.response.mean_logprob),
            borda_score=2.0,
            component_wins=2,
            eligible_for_memory=False,
            negative_answer=greedy_answer,
            negative_representative=greedy.representative,
            reason=(
                "causal=ccic_strict_symmetric_crossover; changed_from_greedy=True; "
                "memory=disabled"
            ),
        )
        return decision, audit

    def _distill(
        self,
        sample: MultimodalSample,
        decision: SelectionDecision,
        sample_index: int,
    ) -> Tuple[str, str, str]:
        negative = (
            decision.negative_representative.response.content
            if decision.negative_representative
            else "No reliable negative candidate was identified."
        )
        answer_metadata = {**sample.metadata, "choices": list(sample.choices)}
        prompt = (
            f"Question:\n{format_question(sample.question, list(sample.choices), answer_metadata)}\n\n"
            f"Evidence-supported candidate:\n{decision.representative.response.content}\n\n"
            f"Unsupported or weaker candidate:\n{negative}"
        )
        response = self._generate(
            DISTILL_SYSTEM,
            prompt,
            sample.image_paths,
            self._sampling(0.0, sample_index * 100 + 50, max_tokens=384),
        )
        try:
            payload = parse_json_object(response.content)
            positive_rule = str(payload.get("positive_rule", "")).strip()
            negative_rule = str(payload.get("negative_rule", "")).strip()
            if decision.negative_representative is None:
                negative_rule = ""
            return positive_rule, negative_rule, str(payload.get("evidence_summary", "")).strip()
        except (ValueError, TypeError, json.JSONDecodeError):
            return response.content.strip(), "", ""

    def _memory_context(self, sample: MultimodalSample, sample_index: int) -> Tuple[str, RetrievalBundle]:
        if self.method not in {"evimem", "direct_ttcl"}:
            empty = RetrievalBundle((), ())
            return "", empty
        bundle = self.memory.retrieve(
            sample,
            sample_index,
            top_rules=8,
            modalities=tuple(self.ablation["retrieval_modalities"]),
        )
        rules = [
            ("Use: " if item.rule_type == "positive" else "Avoid: ") + item.content
            for item in bundle.rules
            if self.ablation["rule_memory"]
            and (item.rule_type != "positive" or self.ablation["positive_memory"])
            and (item.rule_type != "negative" or self.ablation["negative_memory"])
        ]
        rules = rules[:4]
        episodes = [] if self.method == "direct_ttcl" or not self.ablation["episode_memory"] else [
            f"Question pattern: {item.experience.question}; evidence: {item.experience.evidence_summary}"
            for item in bundle.episodes
        ]
        return format_memory_context(rules, episodes), bundle

    def _build_experience(
        self,
        sample: MultimodalSample,
        sample_index: int,
        decision: SelectionDecision,
        verdicts: Mapping[str, EvidenceVerdict],
        positive_rule: str,
        negative_rule: str,
        evidence_summary: str,
    ) -> Experience:
        verifier = verdicts.get(decision.answer)
        trust = (
            decision.scores.invariance
            + decision.scores.sensitivity
            + decision.scores.evidence
        ) / 3.0 * (decision.component_wins / 3.0)
        return Experience(
            experience_id=f"exp_{sample_index:06d}_{uuid.uuid5(uuid.NAMESPACE_URL, sample.sample_id).hex[:8]}",
            sample_index=sample_index,
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            question=sample.question,
            image_paths=tuple(sample.image_paths),
            image_hashes=tuple(sha256_file(path) for path in sample.image_paths),
            positive_answer=decision.answer,
            negative_answer=decision.negative_answer,
            positive_rule=positive_rule,
            negative_rule=negative_rule,
            evidence_summary=evidence_summary,
            evidence_claims=tuple(verifier.claims if verifier else ()),
            component_scores=decision.scores,
            trust=trust,
            timestamp=time.time(),
        )

    def _load_completed(self) -> set[str]:
        if not self.resume or not self.predictions_path.exists():
            return set()
        with self.predictions_path.open("r", encoding="utf-8") as handle:
            return {json.loads(line)["sample_id"] for line in handle if line.strip()}

    def _validate_resume_state(self) -> None:
        if not self.resume or not self.manifest_path.exists():
            return
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        stored_config = payload.get("config")
        stored_config_hash = (
            _scientific_config_hash(stored_config)
            if isinstance(stored_config, dict)
            else payload.get("config_hash")
        )
        expected = {
            "model": self.model.model_name,
            "embedding_model": self.memory.embedder.name,
            "method": self.method,
            "prompt_hash": self._prompt_hash,
            "config_hash": _scientific_config_hash(self.config_snapshot),
        }
        actual = {**payload, "config_hash": stored_config_hash}
        mismatches = {
            key: (actual.get(key), value)
            for key, value in expected.items()
            if actual.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Refusing to resume an incompatible experiment directory: "
                + _stable_json(mismatches)
            )
        self.total_prompt_tokens = int(payload.get("prompt_tokens", 0))
        self.total_completion_tokens = int(payload.get("completion_tokens", 0))
        self.total_latency = float(payload.get("latency_seconds", 0.0))

    def _write_manifest(self, status: str, processed: int) -> None:
        payload = {
            "status": status,
            "git_sha": _git_sha(),
            "model": self.model.model_name,
            "embedding_model": self.memory.embedder.name,
            "method": self.method,
            "seed": self.seed,
            "prompt_hash": self._prompt_hash,
            "config_hash": _scientific_config_hash(self.config_snapshot),
            "config": self.config_snapshot,
            "processed": processed,
            "memory_size": len(self.memory),
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "latency_seconds": self.total_latency,
            "response_trace": getattr(self.model, "trace_stats", None),
            "updated_at": time.time(),
        }
        self.manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(self, samples: Sequence[MultimodalSample]) -> None:
        existing_artifacts = [
            path
            for path in (self.predictions_path, self.memory_path, self.manifest_path)
            if path.exists()
        ]
        if not self.resume and existing_artifacts:
            raise FileExistsError(
                "Refusing to append to an existing non-resume run: "
                + ", ".join(str(path) for path in existing_artifacts)
            )
        self._validate_resume_state()
        completed = self._load_completed()
        if self.resume and self.memory_path.exists():
            self.memory.load_jsonl(self.memory_path)
        previous_images: Sequence[str] = ()
        processed = len(completed)
        self._write_manifest("running", processed)

        for sample_index, sample in enumerate(samples):
            if sample.sample_id in completed:
                previous_images = sample.image_paths or previous_images
                continue
            sample.validate_images()
            before_memory_size = len(self.memory)
            memory_context, retrieved = self._memory_context(sample, sample_index)

            contrastive_verdicts: List[Dict[str, Any]] = []
            evidence_certificate: Dict[str, Any] = {}
            causal_certificate: Dict[str, Any] = {}
            shared_certificate: Dict[str, Any] = {}
            if self.method in {"zero_shot", "cot"}:
                question = sample.question
                if self.method == "cot":
                    question += "\nThink step by step, then provide the required final-answer line."
                candidate = self._candidate(
                    sample,
                    self.method,
                    question,
                    sample.image_paths,
                    ANSWER_SYSTEM,
                    self._sampling(0.0, sample_index * 100),
                )
                decision = _majority_decision([candidate])
                candidates = [candidate]
                verdicts: Dict[str, EvidenceVerdict] = {}
                counterfactual_answers: List[str] = []
            elif self.method == "self_consistency":
                candidates = [
                    self._candidate(
                        sample,
                        f"sample_{offset}",
                        sample.question,
                        sample.image_paths,
                        ANSWER_SYSTEM,
                        self._sampling(0.7, sample_index * 100 + offset),
                    )
                    for offset in range(6)
                ]
                decision = _majority_decision(candidates)
                verdicts = {}
                counterfactual_answers = []
            else:
                fidelity = make_fidelity_views(sample.image_paths, self.image_cache_dir, sample.sample_id)
                blanks = make_blank_views(sample.image_paths, self.image_cache_dir, sample.sample_id)
                candidates = self._generate_candidates(
                    sample, sample_index, memory_context, fidelity
                )
                if self.method == "direct_ttcl":
                    counterfactual_answers = []
                    verdicts = {}
                    decision = _majority_decision(candidates)
                elif self.method in {"contrastive_budget", "contrastive_evidence"}:
                    counterfactual_answers = []
                    verdicts = {}
                    contrastive_decision, contrastive_verdicts = self._contrastive_adjudication(
                        sample, candidates, sample_index, fidelity
                    )
                    decision = (
                        _majority_decision(candidates)
                        if self.method == "contrastive_budget"
                        else contrastive_decision
                    )
                elif self.method in {"certificate_budget", "certificate_evidence"}:
                    counterfactual_answers = []
                    verdicts = {}
                    certificate_decision, evidence_certificate = self._certificate_adjudication(
                        sample,
                        candidates,
                        sample_index,
                        fidelity,
                        blanks,
                        previous_images,
                    )
                    if self.method == "certificate_budget":
                        decision = _greedy_decision(candidates)
                        decision.reason += "; budget_control=certificate_ignored"
                    else:
                        decision = certificate_decision
                elif self.method in {"shared_budget", "shared_evidence"}:
                    counterfactual_answers = []
                    verdicts = {}
                    shared_decision, shared_certificate = self._shared_adjudication(
                        sample, candidates, sample_index, fidelity, blanks, previous_images,
                    )
                    if self.method == "shared_budget":
                        decision = _majority_decision(candidates)
                        decision.eligible_for_memory = False
                        decision.reason += "; budget_control=shared_observations_ignored"
                    else:
                        decision = shared_decision
                elif self.method in {"causal_budget", "causal_evidence"}:
                    counterfactual_answers = []
                    verdicts = {}
                    causal_decision, causal_certificate = self._causal_adjudication(
                        sample,
                        candidates,
                        sample_index,
                        fidelity,
                    )
                    if self.method == "causal_budget":
                        decision = _greedy_decision(candidates)
                        decision.reason += "; budget_control=causal_ignored"
                    else:
                        decision = causal_decision
                else:
                    counterfactual_answers = (
                        self._counterfactual_answers(sample, sample_index, blanks, previous_images)
                        if self.ablation["sensitivity"]
                        else []
                    )
                    verdicts = (
                        self._verify_top_answers(sample, candidates, sample_index)
                        if self.ablation["evidence"]
                        else {}
                    )
                    components = [
                        name
                        for name in ("invariance", "sensitivity", "evidence")
                        if self.ablation[name]
                    ]
                    evidence_decision = select_with_evidence(
                        candidates,
                        counterfactual_answers,
                        verdicts,
                        sample.dataset,
                        enabled_components=components,
                        use_logprob_tiebreak=bool(self.ablation["logprob_tiebreak"]),
                    )
                    decision = (
                        _majority_decision(candidates)
                        if self.method == "budget_matched"
                        else evidence_decision
                    )
                    if self.method == "budget_matched":
                        self._distill(sample, evidence_decision, sample_index)

            wrote_memory = False
            gate_passed = decision.eligible_for_memory or not self.ablation["memory_gate"]
            if self.method in {"evimem", "direct_ttcl"} and gate_passed:
                positive_rule, negative_rule, evidence_summary = self._distill(
                    sample, decision, sample_index
                )
                experience = self._build_experience(
                    sample,
                    sample_index,
                    decision,
                    verdicts,
                    positive_rule,
                    negative_rule,
                    evidence_summary,
                )
                if not self.ablation["positive_memory"]:
                    experience.positive_rule = ""
                if not self.ablation["negative_memory"]:
                    experience.negative_rule = ""
                self.memory.add(experience)
                self.memory.save_jsonl(self.memory_path)
                wrote_memory = True

            record = {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "sample_index": sample_index,
                "method": self.method,
                "answer": decision.answer,
                "response": decision.representative.response.content,
                "selection": {
                    "scores": asdict(decision.scores),
                    "borda_score": decision.borda_score,
                    "component_wins": decision.component_wins,
                    "eligible_for_memory": decision.eligible_for_memory,
                    "negative_answer": decision.negative_answer,
                    "reason": decision.reason,
                },
                "candidate_answers": [
                    {
                        "trajectory": item.trajectory,
                        "answer": item.normalized_answer,
                        "question_variant": item.question_variant,
                        "image_paths": list(item.image_paths),
                        "image_hashes": [sha256_file(path) for path in item.image_paths],
                        "memory_conditioned": item.memory_conditioned,
                    }
                    for item in candidates
                ],
                "counterfactual_answers": counterfactual_answers,
                "contrastive_verdicts": contrastive_verdicts,
                "evidence_certificate": evidence_certificate,
                "causal_certificate": causal_certificate,
                "shared_certificate": shared_certificate,
                "retrieved_experience_ids": [item.experience.experience_id for item in retrieved.episodes],
                "retrieved_episodes": [
                    {
                        "experience_id": item.experience.experience_id,
                        "rrf_score": item.score,
                        "similarity": item.similarity,
                    }
                    for item in retrieved.episodes
                ],
                "retrieved_rules": [
                    {
                        "experience_id": item.experience_id,
                        "rule_type": item.rule_type,
                        "rrf_score": item.score,
                        "similarity": item.similarity,
                    }
                    for item in retrieved.rules
                ],
                "memory_size_before": before_memory_size,
                "memory_size_after": len(self.memory),
                "wrote_memory": wrote_memory,
            }
            with self.predictions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += 1
            previous_images = sample.image_paths or previous_images
            self._write_manifest("running", processed)
            LOGGER.info(
                "[%s/%s] sample=%s answer=%s memory=%s",
                processed,
                len(samples),
                sample.sample_id,
                decision.answer,
                len(self.memory),
            )

        self._write_manifest("completed", processed)
