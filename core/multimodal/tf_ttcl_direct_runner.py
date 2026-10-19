"""Four paired, label-blind arms for faithful TF-TTCL multimodal transfer.

The full arm changes only answering inputs from text to original image + text.
It uses the original closed/open selection rules, generation PPL, textual
positive/negative rules, and online cosine retrieval. No answer refinement,
image augmentation, verification, or trust gating occurs here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .answers import normalize_answer
from .image_ops import sha256_file
from .prompts import format_question
from .schema import MultimodalSample, SamplingConfig
from .tf_ttcl_direct_selection import ScoredCandidate, select_closed, select_open
from .tf_ttcl_text_embedding import cosine


VERSION = "tf_ttcl_mm_direct_v2"
ARMS = ("greedy", "vote_no_rewrite", "select_no_memory", "tf_ttcl_mm_direct")
TEACHER_SYSTEM = ("You are a careful visual question answering teacher. First write two short "
                  "sentences explaining the evidence and inference; say if the image is not "
                  "needed. Never output just an option letter. Use exactly these two sections:\n"
                  "Reasoning: <two short sentences>\nFinal answer: <one option letter>.")
STUDENT_SYSTEM = ("You are a visual question answering student. Independently write two short "
                  "sentences explaining the evidence and inference; say if the image is not "
                  "needed. Never output just an option letter. Use exactly these two sections:\n"
                  "Reasoning: <two short sentences>\nFinal answer: <one option letter>.")
TUTOR_SYSTEM = ("Rewrite only the question text into four semantically equivalent phrasings. "
                "Preserve entities, numbers, negation, comparisons, image references, and the "
                "question target. Do not answer, add facts, or describe any image. Return JSON "
                "only: {\"questions\":[\"...\",\"...\",\"...\",\"...\"]}.")
RULE_HEADER = ("<BEGIN_RULES>\nLEARNED RULES (from previous problem-solving experience)\n"
               "Apply these rules when solving similar problems:\n"
               "[POSITIVE PATTERNS - What works well]\n{positive}\n"
               "[NEGATIVE PATTERNS - What to avoid]\n{negative}\n<END_RULES>")
_PROMPT_ROOT = Path(__file__).resolve().parents[1] / "source/prompts/close/rule"
POSITIVE_TEMPLATE = (_PROMPT_ROOT / "positive/single_extract.md").read_text(encoding="utf-8")
NEGATIVE_TEMPLATE = (_PROMPT_ROOT / "negative/single_extract.md").read_text(encoding="utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                  separators=(",", ":")).encode()).hexdigest()


def split_question_clause(question: str) -> tuple[str, str]:
    """Keep ScienceQA context/hint byte-identical; Tutor sees only the question clause."""
    prefix, separator, clause = question.rpartition("\n\nQuestion:\n")
    return (prefix + separator, clause) if separator else ("", question)


def parse_variants(text: str, question: str) -> list[str]:
    """Keep exactly four question-only variants; log original-text fallbacks."""
    try:
        payload = json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
        raw = payload.get("questions", []) if isinstance(payload, dict) else payload
    except (ValueError, TypeError, AttributeError):
        raw = []
    variants: list[str] = []
    source_numbers = sorted(re.findall(r"\d+(?:\.\d+)?", question))
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, str):
            continue
        variant = item.strip()
        if (variant and variant != question and len(variant) <= max(3 * len(question), 300)
                and sorted(re.findall(r"\d+(?:\.\d+)?", variant)) == source_numbers
                and variant not in variants):
            variants.append(variant)
        if len(variants) == 4:
            break
    return (variants + [question] * 4)[:4]


def has_reasoning_before_final(text: str) -> bool:
    """Do not require a literal heading; require actual prose before the answer."""
    marker = re.search(r"(?i)final answer\s*:", text)
    if not marker:
        return False
    prefix = text[:marker.start()].strip()
    return len(prefix) >= 25 and len(prefix.split()) >= 8


@dataclass(frozen=True)
class TextRule:
    rule_id: str
    kind: str
    sample_index: int
    text: str
    vector: tuple[float, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextRule":
        return cls(data["rule_id"], data["kind"], int(data["sample_index"]),
                   data["text"], tuple(data["vector"]))


def retrieve_rules(rules: Sequence[TextRule], query_vector: tuple[float, ...],
                   sample_index: int, top_k: int = 30) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {"positive": [], "negative": []}
    for kind in result:
        eligible = [rule for rule in rules if rule.kind == kind and rule.sample_index < sample_index]
        ranked = sorted(eligible, key=lambda rule: (-cosine(query_vector, rule.vector),
                                                    rule.sample_index, rule.rule_id))[:top_k]
        result[kind] = [{"rule_id": rule.rule_id, "sample_index": rule.sample_index,
                         "text": rule.text, "similarity": cosine(query_vector, rule.vector)}
                        for rule in ranked]
    return result


def _rule_prompt(retrieved: dict[str, list[dict[str, Any]]]) -> str:
    if not any(retrieved.values()):
        return ""
    return RULE_HEADER.format(
        positive="\n".join(item["text"] for item in retrieved["positive"]),
        negative="\n".join(item["text"] for item in retrieved["negative"]),
    )


class DirectRunner:
    def __init__(self, backend, embedder, samples: Sequence[MultimodalSample],
                 output: Path, input_sha256: str, model_revision: str,
                 embedding_revision: str, seed: int = 0) -> None:
        if not samples or len({sample.sample_id for sample in samples}) != len(samples):
            raise ValueError("Need nonempty ordered samples with unique IDs")
        if len({sample.dataset for sample in samples}) != 1:
            raise ValueError("Mixed datasets are not allowed")
        self.backend, self.embedder = backend, embedder
        self.samples, self.output = list(samples), output
        self.predictions = output / "predictions.jsonl"
        self.manifest = output / "manifest.json"
        self.seed = seed
        self.rules: list[TextRule] = []
        self.git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        self.prompt_sha = _digest([TEACHER_SYSTEM, STUDENT_SYSTEM, TUTOR_SYSTEM,
                                   RULE_HEADER, POSITIVE_TEMPLATE, NEGATIVE_TEMPLATE])
        self.config = {"version": VERSION, "dataset": samples[0].dataset,
                       "input_sha256": input_sha256,
                       "ordered_ids": [sample.sample_id for sample in samples],
                       "model": backend.model_name, "model_revision": model_revision,
                       "embedding_revision": embedding_revision, "seed": seed,
                       "prompt_sha256": self.prompt_sha, "arms": list(ARMS),
                       "candidate_max_tokens": 256, "tutor_max_tokens": 384,
                       "rule_max_tokens": 100, "rule_top_k_each": 30,
                       "teacher_temperature": 0.0, "student_temperature": 0.7,
                       "student_top_p": 0.9, "labels_read": False}
        self.config_sha = _digest(self.config)
        self.code_sha = _digest([sha256_file(str(Path(__file__))),
                                 sha256_file(str(Path(__file__).with_name("tf_ttcl_direct_selection.py"))),
                                 sha256_file(str(Path(__file__).with_name("tf_ttcl_text_embedding.py")))])

    def _write_manifest(self, status: str, processed: int) -> None:
        payload = {**self.config, "config_sha256": self.config_sha, "code_sha256": self.code_sha,
                   "git_sha": self.git_sha, "status": status, "processed": processed,
                   "trace_stats": getattr(self.backend, "trace_stats", None)}
        temporary = self.manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(self.manifest)

    def _resume(self) -> int:
        if not self.manifest.exists():
            if self.predictions.exists():
                raise ValueError("Predictions without manifest")
            return 0
        old = json.loads(self.manifest.read_text())
        for key, value in (("config_sha256", self.config_sha), ("code_sha256", self.code_sha),
                           ("git_sha", self.git_sha)):
            if old.get(key) != value:
                raise ValueError(f"Resume {key} mismatch")
        rows = [json.loads(line) for line in self.predictions.read_text().splitlines()] \
            if self.predictions.exists() else []
        if len(rows) > len(self.samples):
            raise ValueError("Too many committed predictions")
        for index, row in enumerate(rows):
            if (row.get("sample_index") != index or row.get("sample_id") != self.samples[index].sample_id
                    or set(row.get("arms", {})) != set(ARMS)):
                raise ValueError("Resume is not an ordered complete prefix")
            self.rules.extend(TextRule.from_dict(rule) for rule in row.get("written_rules", []))
        return len(rows)

    def _call(self, system: str, user: str, images: Sequence[str],
              sampling: SamplingConfig) -> dict[str, Any]:
        response = self.backend.generate(system, user, images, sampling)
        return {"system": system, "user": user, "image_paths": list(images),
                "image_hashes": [sha256_file(path) for path in images],
                "sampling": asdict(sampling), "response": asdict(response)}

    def _candidate(self, role: str, sample: MultimodalSample, question: str,
                   system: str, offset: int, index: int, sampled: bool) -> tuple[ScoredCandidate, dict]:
        user = format_question(question, sample.choices, sample.metadata)
        sampling = SamplingConfig(temperature=0.7 if sampled else 0.0,
                                  top_p=0.9 if sampled else None,
                                  max_tokens=256, seed=self.seed + 100 * index + offset,
                                  request_logprobs=True)
        call = self._call(system, user, sample.image_paths, sampling)
        from .schema import GenerationResponse
        response = GenerationResponse(**call["response"])
        answer = normalize_answer(response.content, sample.dataset,
                                  {**sample.metadata, "choices": sample.choices})
        if sample.dataset == "scienceqa_img" and answer not in {
                chr(97 + i) for i in range(len(sample.choices))}:
            answer = f"__unparsed_{role}__"
        if not answer:
            answer = f"__unparsed_{role}__"
        scored = ScoredCandidate.from_response(role, answer, response)
        call.update({"role": role, "question_variant": question,
                     "normalized_answer": answer, "ppl": scored.ppl})
        return scored, call

    def _select(self, candidates: list[ScoredCandidate], dataset: str):
        if dataset == "scienceqa_img":
            return select_closed(candidates)
        if dataset == "textvqa":
            teacher_vector = self.embedder.encode(candidates[0].response.content)
            similarities = [cosine(teacher_vector, self.embedder.encode(candidate.response.content))
                            for candidate in candidates[1:]]
            return select_open(candidates, similarities)
        raise ValueError(f"Unsupported direct-transfer dataset: {dataset}")

    @staticmethod
    def _selection_record(selection) -> dict[str, Any]:
        return {"answer": "" if selection.positive.answer.startswith("__unparsed_")
                else selection.positive.answer, "positive_role": selection.positive.role,
                "negative_role": selection.negative.role if selection.negative else None,
                "positive_roles": selection.positive_roles,
                "negative_roles": selection.negative_roles, "reason": selection.reason,
                "positive_ppl": selection.positive.ppl,
                "negative_ppl": selection.negative.ppl if selection.negative else None,
                "similarity_ranking": selection.similarity_ranking}

    @staticmethod
    def _usage(calls: Sequence[dict]) -> dict[str, float]:
        responses = [call["response"] for call in calls]
        return {"prompt_tokens": sum(x["prompt_tokens"] for x in responses),
                "completion_tokens": sum(x["completion_tokens"] for x in responses),
                "latency_seconds": sum(x["latency_seconds"] for x in responses)}

    def _summarize(self, sample: MultimodalSample, index: int, selection) -> tuple[list[TextRule], list[dict]]:
        if not selection.summarize_positive:
            return [], []
        question = format_question(sample.question, sample.choices, sample.metadata)
        prompts = [("positive", POSITIVE_TEMPLATE.format(question=question,
                   answer=selection.positive.response.content))]
        if selection.summarize_negative and selection.negative is not None:
            prompts.append(("negative", NEGATIVE_TEMPLATE.format(
                question=question, positive_answer=selection.positive.response.content,
                negative_answer=selection.negative.response.content)))
        written: list[TextRule] = []
        calls: list[dict] = []
        for ordinal, (kind, prompt) in enumerate(prompts):
            sampling = SamplingConfig(temperature=0.7, top_p=0.9, max_tokens=100,
                                      seed=self.seed + 100 * index + 20 + ordinal)
            call = self._call("", prompt, (), sampling)
            call["kind"] = kind
            calls.append(call)
            text = call["response"]["content"].strip()
            text = re.sub(r"^(?:Positive|Negative) Rule:\s*", "", text, flags=re.I)
            if text:
                written.append(TextRule(f"{kind}_{index}", kind, index,
                                        text, self.embedder.encode(text)))
        return written, calls

    def _sample(self, sample: MultimodalSample, index: int) -> dict[str, Any]:
        sample.validate_images()
        query_vector = self.embedder.encode(sample.question, query=True) if self.rules else ()
        retrieved = retrieve_rules(self.rules, query_vector, index) if self.rules else {
            "positive": [], "negative": []}
        rule_text = _rule_prompt(retrieved)
        teacher, teacher_call = self._candidate("teacher", sample, sample.question,
                                                TEACHER_SYSTEM, 0, index, False)
        vote = [teacher]
        vote_calls = [teacher_call]
        for number in range(1, 5):
            candidate, call = self._candidate(f"student_{number}", sample, sample.question,
                                               STUDENT_SYSTEM, number + 1, index, True)
            vote.append(candidate)
            vote_calls.append(call)
        vote_selection = self._select(vote, sample.dataset)
        context_prefix, question_clause = split_question_clause(sample.question)
        tutor_call = self._call(TUTOR_SYSTEM, question_clause, (), SamplingConfig(
            temperature=0.7, top_p=0.9, max_tokens=384,
            seed=self.seed + 100 * index + 1))
        variants = [context_prefix + item for item in
                    parse_variants(tutor_call["response"]["content"], question_clause)]
        nomem = [teacher]
        nomem_calls = [teacher_call, tutor_call]
        for number, variant in enumerate(variants, 1):
            candidate, call = self._candidate(f"student_{number}", sample, variant,
                                               STUDENT_SYSTEM, number + 1, index, True)
            nomem.append(candidate)
            nomem_calls.append(call)
        nomem_selection = self._select(nomem, sample.dataset)
        if rule_text:
            full_teacher, full_teacher_call = self._candidate(
                "teacher", sample, sample.question, TEACHER_SYSTEM + "\n" + rule_text,
                0, index, False)
        else:
            full_teacher, full_teacher_call = teacher, teacher_call
        full = [full_teacher]
        full_calls = [full_teacher_call, tutor_call]
        for number, variant in enumerate(variants, 1):
            system = STUDENT_SYSTEM + ("\n" + rule_text if rule_text else "")
            candidate, call = self._candidate(f"student_{number}", sample, variant,
                                               system, number + 1, index, True)
            full.append(candidate)
            full_calls.append(call)
        full_selection = self._select(full, sample.dataset)
        written, summary_calls = self._summarize(sample, index, full_selection)
        full_calls.extend(summary_calls)
        self.rules.extend(written)
        greedy_answer = "" if teacher.answer.startswith("__unparsed_") else teacher.answer
        return {"sample_index": index, "sample_id": sample.sample_id,
                "dataset": sample.dataset, "image_hashes": teacher_call["image_hashes"],
                "original_question": sample.question, "choices": list(sample.choices),
                "variants": variants, "retrieved_rules": retrieved,
                "memory_size_before": sum(rule.sample_index < index for rule in self.rules),
                "arms": {
                    "greedy": {"answer": greedy_answer, "calls": [teacher_call],
                               "usage": self._usage([teacher_call])},
                    "vote_no_rewrite": {**self._selection_record(vote_selection),
                                         "calls": vote_calls, "usage": self._usage(vote_calls)},
                    "select_no_memory": {**self._selection_record(nomem_selection),
                                          "calls": nomem_calls, "usage": self._usage(nomem_calls)},
                    "tf_ttcl_mm_direct": {**self._selection_record(full_selection),
                                           "calls": full_calls, "usage": self._usage(full_calls)}},
                "written_rules": [asdict(rule) for rule in written], "labels_read": False}

    def run(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        start = self._resume()
        self._write_manifest("running", start)
        for index in range(start, len(self.samples)):
            row = self._sample(self.samples[index], index)
            with self.predictions.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._write_manifest("running", index + 1)
            print(json.dumps({"index": index, "sample_id": row["sample_id"],
                              "answers": {arm: row["arms"][arm]["answer"] for arm in ARMS},
                              "new_rules": len(row["written_rules"])}), flush=True)
        self._write_manifest("completed", len(self.samples))
