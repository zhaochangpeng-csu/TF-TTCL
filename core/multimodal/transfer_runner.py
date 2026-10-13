"""Single-pass, label-blind four-arm pseudo-experience transfer experiment.

All arms see the same current-sample candidates, typed evidence and distilled
rule. Only historical writes/retrieval differ. Each sample is committed after
four refinement responses; restart reconstructs the exact past banks from the
committed prefix. Trace recording makes an interrupted sample replayable.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from . import typed_evidence
from .answers import normalize_answer
from .image_ops import make_blank_views, make_fidelity_views, sha256_file
from .prompts import format_question
from .schema import Candidate, GenerationResponse, MultimodalSample, SamplingConfig
from .transfer_policy import answer_content, deterministic_random_past, majority_pair, write_policy
from scripts.run_transfer_format_smoke import FORMAT_USER_SUFFIX, TRAJECTORY_SYSTEM, format_ok


VERSION = "transfer_online_v1_rule_only_typed_gate"
ARMS = ("none", "noisy", "gated_weighted", "random")
RULE_SYSTEM = """Distill a broadly reusable visual-question answering CHECKING PROCEDURE
from proposed reasoning. The proposed answers are not ground truth and may be
wrong. Never copy a conclusion, an option letter, or example-specific names,
colors, values, species, or facts into the procedure. Return JSON only with
exactly three string keys: positive_rule, negative_rule, evidence_summary.
positive_rule should say how to check relevant evidence and make the inference
on a different question. negative_rule should be empty when there is no
different proposed answer; otherwise describe the unsupported reasoning pattern
as a hypothesis, not a proven error. evidence_summary is a brief, source-image
description for audit; it is not retrieved in this rule-only pilot."""
RULE_SCHEMA = {"type": "object", "properties": {
    "positive_rule": {"type": "string", "maxLength": 400},
    "negative_rule": {"type": "string", "maxLength": 400},
    "evidence_summary": {"type": "string", "maxLength": 400}},
    "required": ["positive_rule", "negative_rule", "evidence_summary"],
    "additionalProperties": False}
REFINE_SYSTEM = """Answer the current visual question. Current candidates and
retrieved historical procedures can be wrong. Recheck the CURRENT image and
context; never copy a prior answer or prior observed fact. Keep these exact
sections in order: Observed, Reasoning, Final answer. Use one current option
letter after Final answer."""
REFINE_SUFFIX = """Give all three sections in this exact order, even if the image
is not required. Distinguish visible observations from text/knowledge. State
one or two short sentences in each of the first two sections.
Observed: <current visible evidence, image not required, or insufficient>
Reasoning: <brief inference for the current question>
Final answer: <one current option letter>"""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def answer_in_choices(answer: str, choices: Sequence[str]) -> bool:
    return len(answer) == 1 and "a" <= answer <= "z" and ord(answer) - ord("a") < len(choices)


def parse_rule(text: str, has_negative: bool, choices: Sequence[str]) -> dict[str, Any]:
    empty = {"positive_rule": "", "negative_rule": "", "evidence_summary": "", "valid": False}
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return empty
    if not isinstance(payload, dict) or set(payload) != {"positive_rule", "negative_rule", "evidence_summary"}:
        return empty
    if any(not isinstance(v, str) for v in payload.values()):
        return empty
    positive = payload["positive_rule"].strip()
    negative = payload["negative_rule"].strip() if has_negative else ""
    summary = payload["evidence_summary"].strip()
    if not 15 <= len(positive) <= 400 or len(negative) > 400 or len(summary) > 400:
        return empty
    if re.search(r"(?i)final answer\s*:|\boption\s+[a-e]\b", positive + " " + negative):
        return empty
    # Exact option text is a source-specific answer shortcut. Short common
    # words are excluded from this check because they may be procedure words.
    for choice in choices:
        stripped = choice.strip()
        if len(stripped) >= 8 and re.search(re.escape(stripped), positive + " " + negative, re.I):
            return empty
    return {"positive_rule": positive, "negative_rule": negative,
            "evidence_summary": summary, "valid": True}


_TOKENS = re.compile(r"[a-z]+|\d+", re.I)


def tokens(text: str) -> Counter[str]:
    return Counter(_TOKENS.findall(text.lower()))


@dataclass(frozen=True)
class RuleEntry:
    experience_id: str
    sample_index: int
    sample_id: str
    question: str
    positive_answer_text: str
    negative_answer_text: str | None
    positive_rule: str
    negative_rule: str
    evidence_summary: str
    trust: float
    source_image_hashes: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RuleEntry":
        return cls(**{**value, "source_image_hashes": tuple(value["source_image_hashes"])})


def lexical_retrieve(entries: Sequence[RuleEntry], question: str, sample_index: int,
                     top_k: int = 2) -> list[RuleEntry]:
    eligible = [e for e in entries if e.sample_index < sample_index]
    if not eligible or top_k <= 0:
        return []
    query = tokens(question)
    documents = [tokens(e.question) for e in eligible]
    df = Counter(word for doc in documents for word in doc)
    n = len(eligible)
    avg_len = sum(sum(doc.values()) for doc in documents) / n

    def score(item: RuleEntry, doc: Counter[str]) -> float:
        length = sum(doc.values())
        total = 0.0
        for word, qfreq in query.items():
            freq = doc.get(word, 0)
            if not freq:
                continue
            idf = math.log(1 + (n - df[word] + 0.5) / (df[word] + 0.5))
            total += idf * freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * length / max(avg_len, 1))) * min(qfreq, 2)
        return total * item.trust

    ranked = sorted(zip(eligible, documents), key=lambda x: (-score(*x), x[0].sample_index, x[0].experience_id))
    return [entry for entry, doc in ranked if score(entry, doc) > 0][:top_k]


def rule_context(entries: Sequence[RuleEntry]) -> str:
    if not entries:
        return "Historical checking procedures: none available."
    lines = ["Historical checking procedures (uncertified; verify on the current image):"]
    for i, entry in enumerate(entries, 1):
        lines.append(f"{i}. Check: {entry.positive_rule}")
        if entry.negative_rule:
            lines.append(f"   Possible failure: {entry.negative_rule}")
    return "\n".join(lines)


class TransferRunner:
    def __init__(self, backend, samples: Sequence[MultimodalSample], output: Path,
                 input_sha256: str, model_revision: str, seed: int = 0) -> None:
        if not samples or len({s.sample_id for s in samples}) != len(samples):
            raise ValueError("Nonempty unique ordered samples required")
        self.backend = backend
        self.samples = list(samples)
        self.output = output
        self.seed = seed
        self.input_sha256 = input_sha256
        self.model_revision = model_revision
        self.predictions = output / "predictions.jsonl"
        self.manifest_path = output / "manifest.json"
        self.image_cache = output / "image_cache"
        self.banks: dict[str, list[RuleEntry]] = {name: [] for name in ARMS}
        self.prompt_sha = digest([TRAJECTORY_SYSTEM, FORMAT_USER_SUFFIX, RULE_SYSTEM, RULE_SCHEMA,
                                  REFINE_SYSTEM, REFINE_SUFFIX, typed_evidence.SPECS])
        self.config = {"version": VERSION, "dataset": "scienceqa_img", "input_sha256": input_sha256,
                       "ordered_ids": [s.sample_id for s in samples], "model": backend.model_name,
                       "model_revision": model_revision, "seed": seed, "prompt_sha256": self.prompt_sha,
                       "candidate_temperatures": [0.0, 0.7, 0.7, 0.7], "top_rules": 2,
                       "candidate_max_tokens": 384, "rule_max_tokens": 256,
                       "refinement_max_tokens": 384, "typed_gate": typed_evidence.VERSION,
                       "memory_type": "rule_only_lexical_bm25", "labels_read": False}
        self.config_sha = digest(self.config)
        self.git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        self.code_sha = digest([sha256_file(str(Path(__file__))),
                                sha256_file(str(Path(__file__).with_name("transfer_policy.py"))),
                                sha256_file(str(Path(__file__).with_name("typed_evidence.py"))),
                                sha256_file(str(Path(__file__).parents[2] / "scripts" / "run_transfer_format_smoke.py"))])

    def _manifest(self, status: str, processed: int) -> dict[str, Any]:
        return {**self.config, "config_sha256": self.config_sha, "code_sha256": self.code_sha,
                "git_sha": self.git_sha, "status": status, "processed": processed,
                "arm_order": list(ARMS), "all_arms_complete": status == "completed",
                "trace_stats": getattr(self.backend, "trace_stats", None)}

    def _write_manifest(self, status: str, processed: int) -> None:
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._manifest(status, processed), ensure_ascii=False, indent=2) + "\n")
        temporary.replace(self.manifest_path)

    def _load_prefix(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            if self.predictions.exists():
                raise ValueError("Predictions exist without manifest")
            return []
        old = json.loads(self.manifest_path.read_text())
        if (old.get("config_sha256") != self.config_sha or old.get("code_sha256") != self.code_sha
                or old.get("git_sha") != self.git_sha):
            raise ValueError("Resume configuration/code/Git SHA mismatch")
        rows = [json.loads(line) for line in self.predictions.read_text().splitlines()] if self.predictions.exists() else []
        if len(rows) > len(self.samples):
            raise ValueError("Too many committed rows")
        for index, row in enumerate(rows):
            if row.get("sample_index") != index or row.get("sample_id") != self.samples[index].sample_id:
                raise ValueError("Resume is not an ordered prefix")
            if set(row.get("arms", {})) != set(ARMS):
                raise ValueError("Partial four-arm row in committed prefix")
            for arm in ARMS:
                written = row["arms"][arm].get("written")
                if written:
                    entry = RuleEntry.from_dict(written)
                    if entry.sample_index != index or entry.sample_id != row["sample_id"]:
                        raise ValueError("Bad restored experience index")
                    self.banks[arm].append(entry)
        return rows

    def _call(self, system: str, user: str, images: Sequence[str], index: int,
              offset: int, temperature: float = 0.0, max_tokens: int = 384,
              output_schema: dict | None = None) -> GenerationResponse:
        sampling = SamplingConfig(temperature=temperature, top_p=0.9 if temperature else None,
                                  max_tokens=max_tokens, seed=self.seed + index * 100 + offset,
                                  output_schema=output_schema)
        return self.backend.generate(system, user, images, sampling)

    def _candidates(self, sample: MultimodalSample, index: int) -> tuple[list[Candidate], list[dict[str, Any]]]:
        user = format_question(sample.question, sample.choices, sample.metadata) + "\n\n" + FORMAT_USER_SUFFIX
        out = []
        attempts = []
        for ordinal, temperature in enumerate(self.config["candidate_temperatures"]):
            response = self._call(TRAJECTORY_SYSTEM, user, sample.image_paths, index,
                                  10 + ordinal, temperature)
            answer = normalize_answer(response.content, sample.dataset, {**sample.metadata, "choices": sample.choices})
            format_valid = format_ok(response, sample.choices)
            valid = answer_in_choices(answer, sample.choices)
            attempts.append({"trajectory": f"candidate_{ordinal}", "temperature": temperature,
                             "answer": answer, "format_ok": format_valid, "answer_valid": valid,
                             "response": asdict(response)})
            if not valid:
                # A bad candidate is logged and excluded from grouping. It is
                # never silently treated as a positive or negative example.
                continue
            out.append(Candidate(f"candidate_{ordinal}", f"candidate_{ordinal}", response,
                                 answer, sample.question, sample.image_paths))
        return out, attempts

    def _typed_gate(self, sample: MultimodalSample, index: int,
                    previous_images: Sequence[str], candidate_answers: Sequence[str]) -> dict[str, Any]:
        program = typed_evidence.compile_program(sample.question, sample.choices)
        result = {"program": program, "gate_passed": False, "computed_answer": None,
                  "fallback_reason": "unsupported_task", "observations": [],
                  "fidelity_byte_identical": None}
        if program is None or len(sample.image_paths) != 1:
            return result
        fidelity = make_fidelity_views(sample.image_paths, self.image_cache, sample.sample_id)
        blank = make_blank_views(sample.image_paths, self.image_cache, sample.sample_id)
        result["fidelity_byte_identical"] = sha256_file(fidelity[0]) == sha256_file(sample.image_paths[0])
        views = [("real", sample.image_paths), ("fidelity", fidelity), ("blank", blank)]
        if previous_images:
            views.append(("previous_mismatch", previous_images))
        system, schema, token_limit = typed_evidence.SPECS[program["modality"]]
        parsed = []
        for ordinal, (view, images) in enumerate(views):
            response = self._call(system, "Transcribe only visible observations using the supplied schema.",
                                  images[:1], index, 60 + ordinal, max_tokens=token_limit,
                                  output_schema=schema)
            observation = typed_evidence.parse_observation(response.content, program)
            if response.finish_reason == "length":
                observation["complete"] = False
            parsed.append(observation)
            result["observations"].append({"view": view, "image_hashes": [sha256_file(p) for p in images[:1]],
                                           "response": asdict(response), "parsed": observation})
        result.update(typed_evidence.assess_certificate(program, parsed, sample.choices, candidate_answers))
        return result

    def _distill(self, sample: MultimodalSample, index: int, pair) -> tuple[dict[str, Any], GenerationResponse]:
        user = (f"Current task:\n{format_question(sample.question, sample.choices, sample.metadata)}\n\n"
                f"Positive pseudo-candidate (uncertified):\n{pair.positive.response.content}\n\n"
                f"Different pseudo-candidate (not proven wrong):\n"
                f"{pair.negative.response.content if pair.negative else 'none'}")
        response = self._call(RULE_SYSTEM, user, sample.image_paths, index, 40,
                              max_tokens=256, output_schema=RULE_SCHEMA)
        return parse_rule(response.content, pair.negative is not None, sample.choices), response

    def _refinement(self, sample: MultimodalSample, index: int, pair, rules: Sequence[RuleEntry]):
        positive_text = answer_content(pair.positive, sample.choices)
        negative_text = answer_content(pair.negative, sample.choices) if pair.negative else "none proposed"
        user = (format_question(sample.question, sample.choices, sample.metadata)
                + "\n\nCurrent unverified candidates:\n"
                + f"Most frequent proposal: {positive_text}; rationale: {pair.positive.response.content[:650]}\n"
                + f"Different proposal: {negative_text}; rationale: "
                + (pair.negative.response.content[:650] if pair.negative else "none")
                + "\n\n" + rule_context(rules) + "\n\n" + REFINE_SUFFIX)
        response = self._call(REFINE_SYSTEM, user, sample.image_paths, index, 90, max_tokens=384)
        answer = normalize_answer(response.content, sample.dataset, {**sample.metadata, "choices": sample.choices})
        valid = format_ok(response, sample.choices) and answer_in_choices(answer, sample.choices)
        return response, answer if valid else pair.positive.normalized_answer, valid, user

    def run(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        rows = self._load_prefix()
        if len(rows) == len(self.samples):
            self._write_manifest("completed", len(rows))
            return
        previous_images: Sequence[str] = self.samples[len(rows) - 1].image_paths if rows else ()
        self._write_manifest("running", len(rows))
        with self.predictions.open("a", encoding="utf-8") as handle:
            for index in range(len(rows), len(self.samples)):
                sample = self.samples[index]
                sample.validate_images()
                candidates, attempts = self._candidates(sample, index)
                if not candidates:
                    raise RuntimeError(f"No format-valid candidate for {sample.sample_id}")
                pair = majority_pair(candidates)
                evidence = self._typed_gate(sample, index, previous_images,
                                            [c.normalized_answer for c in candidates])
                rule, rule_response = self._distill(sample, index, pair)
                record = {"sample_id": sample.sample_id, "sample_index": index, "dataset": sample.dataset,
                          "question": sample.question, "choices": list(sample.choices),
                          "metadata": sample.metadata, "image_hashes": [sha256_file(p) for p in sample.image_paths],
                          "candidates": [{"trajectory": c.trajectory, "answer": c.normalized_answer,
                                          "raw_response": asdict(c.response)} for c in candidates],
                          "candidate_attempts": attempts,
                          "pseudo_pair": {"positive": pair.positive.normalized_answer,
                                          "negative": pair.negative.normalized_answer if pair.negative else None,
                                          "consensus": pair.consensus, "writable": pair.writable,
                                          "reason": pair.reason}, "evidence": evidence,
                          "distillation": {"response": asdict(rule_response), "parsed": rule}, "arms": {}}
                for arm in ARMS:
                    if arm == "none":
                        retrieved = []
                    elif arm == "random":
                        retrieved = deterministic_random_past(self.banks["random"], index, self.seed, 2)
                    else:
                        retrieved = lexical_retrieve(self.banks[arm], sample.question, index)
                    if any(entry.sample_index >= index for entry in retrieved):
                        raise RuntimeError("Current/future retrieval leak")
                    response, answer, valid, user = self._refinement(sample, index, pair, retrieved)
                    decision = write_policy(arm, pair, evidence_passed=evidence["gate_passed"],
                                            evidence_answer=evidence["computed_answer"],
                                            rule_valid=rule["valid"] and len(candidates) >= 2
                                            and format_ok(pair.positive.response, sample.choices)
                                            and (pair.negative is None or format_ok(pair.negative.response, sample.choices)))
                    entry = None
                    if decision.write:
                        entry = RuleEntry(
                            experience_id=f"rule_{index:06d}_{sample.sample_id}", sample_index=index,
                            sample_id=sample.sample_id, question=sample.question,
                            positive_answer_text=answer_content(pair.positive, sample.choices),
                            negative_answer_text=answer_content(pair.negative, sample.choices) if pair.negative else None,
                            positive_rule=rule["positive_rule"], negative_rule=rule["negative_rule"],
                            evidence_summary=rule["evidence_summary"], trust=decision.trust,
                            source_image_hashes=tuple(record["image_hashes"]))
                    record["arms"][arm] = {
                        "answer": answer, "format_ok": valid, "response": asdict(response),
                        "input_user_text": user,
                        "retrieved": [asdict(e) for e in retrieved],
                        "memory_size_before": len(self.banks[arm]),
                        "write_decision": asdict(decision), "written": asdict(entry) if entry else None,
                        "changed_from_pseudo_positive": answer != pair.positive.normalized_answer}
                # A row is appended only after all arms have finished. Memory
                # becomes visible to the next sample after the committed row.
                handle.write(canonical_json(record) + "\n")
                handle.flush()
                for arm in ARMS:
                    written = record["arms"][arm]["written"]
                    if written:
                        self.banks[arm].append(RuleEntry.from_dict(written))
                previous_images = sample.image_paths
                self._write_manifest("running", index + 1)
                if (index + 1) % 10 == 0 or index == 0:
                    print(f"[{index + 1}/{len(self.samples)}] {sample.sample_id} "
                          f"writes=" + ",".join(f"{arm}:{len(self.banks[arm])}" for arm in ARMS), flush=True)
        self._write_manifest("completed", len(self.samples))
