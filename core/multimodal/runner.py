"""Single-pass EviMem-TTCL experiment runner."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .answers import normalize_answer
from .client import VLMBackend, parse_evidence_verdict, parse_json_object
from .image_ops import make_blank_views, make_fidelity_views, sha256_file
from .memory import EvidenceMemory, RetrievalBundle
from .prompts import (
    ANSWER_SYSTEM,
    DISTILL_SYSTEM,
    PARAPHRASE_SYSTEM,
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


LOGGER = logging.getLogger(__name__)
METHODS = {
    "zero_shot",
    "cot",
    "self_consistency",
    "budget_matched",
    "direct_ttcl",
    "evidence_only",
    "evimem",
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
            "\n".join([ANSWER_SYSTEM, PARAPHRASE_SYSTEM, VERIFY_SYSTEM, DISTILL_SYSTEM])
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
        response = self._generate(
            PARAPHRASE_SYSTEM,
            question,
            (),
            self._sampling(0.0, sample_index * 100 + 1, max_tokens=256),
        )
        try:
            payload = parse_json_object(response.content)
            variants = [str(item).strip() for item in payload.get("questions", []) if str(item).strip()]
        except (ValueError, TypeError, json.JSONDecodeError):
            variants = []
        while len(variants) < 2:
            variants.append(question)
        return variants[:2]

    def _candidate(
        self,
        sample: MultimodalSample,
        trajectory: str,
        question: str,
        images: Sequence[str],
        system: str,
        sampling: SamplingConfig,
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
        system = ANSWER_SYSTEM + ("\n\n" + memory_context if memory_context else "")
        specs = [
            ("original_greedy", question, sample.image_paths, 0.0),
            ("original_sampled", question, sample.image_paths, 0.7),
        ]
        if self.ablation["text_variants"]:
            specs.extend(
                [
                    ("text_variant_1", variants[0], sample.image_paths, 0.0),
                    ("text_variant_2", variants[1], sample.image_paths, 0.0),
                ]
            )
        if self.ablation["image_variant"]:
            specs.append(("image_variant", question, fidelity_images or sample.image_paths, 0.0))
        if self.ablation["joint_variant"]:
            specs.append(("joint_variant", variants[0], fidelity_images or sample.image_paths, 0.0))
        return [
            self._candidate(
                sample,
                name,
                variant,
                images,
                system,
                self._sampling(temperature, sample_index * 100 + 10 + offset),
            )
            for offset, (name, variant, images, temperature) in enumerate(specs)
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
            return (
                str(payload.get("positive_rule", "")).strip(),
                str(payload.get("negative_rule", "")).strip(),
                str(payload.get("evidence_summary", "")).strip(),
            )
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
        expected = {
            "model": self.model.model_name,
            "embedding_model": self.memory.embedder.name,
            "method": self.method,
            "prompt_hash": self._prompt_hash,
            "config_hash": _sha256_text(_stable_json(self.config_snapshot)),
        }
        mismatches = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
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
            "config_hash": _sha256_text(_stable_json(self.config_snapshot)),
            "config": self.config_snapshot,
            "processed": processed,
            "memory_size": len(self.memory),
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "latency_seconds": self.total_latency,
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
                    {"trajectory": item.trajectory, "answer": item.normalized_answer}
                    for item in candidates
                ],
                "counterfactual_answers": counterfactual_answers,
                "retrieved_experience_ids": [item.experience.experience_id for item in retrieved.episodes],
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
