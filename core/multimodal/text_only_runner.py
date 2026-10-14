"""Label-blind, no-memory paired test of question-only augmentation.

All arms share the original greedy response and the answer-blind paraphrase
request. The two vote arms have identical image, decoding settings and seeds;
only their question text differs. A committed row contains all three arms.
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .answers import normalize_answer
from .image_ops import sha256_file
from .prompts import format_question
from .runner import _protected_query_features, _semantic_variant_is_safe, _split_context_and_query
from .schema import MultimodalSample, SamplingConfig
from .transfer_runner import answer_in_choices, canonical_json, digest
from scripts.run_transfer_format_smoke import FORMAT_USER_SUFFIX, TRAJECTORY_SYSTEM, format_ok


VERSION = "text_only_vote_v2"
ARMS = ("greedy", "same_text_vote", "text_aug_vote")
CANDIDATES = ("original_greedy", "same_text_1", "same_text_2", "text_aug_1", "text_aug_2")
TEXT_REWRITE_SYSTEM = """Rewrite the supplied QUESTION CLAUSE twice without answering it.
Make two distinct, fluent questions by changing word order or grammatical
phrasing. Keep every content word, entity, number, comparison, negation and
image reference. Do not add a factual claim, answer, option, or named entity.
Do not repeat the original verbatim, use an empty string, or use ellipses.
The surrounding passage and answer options are absent and will be restored
unchanged. Return JSON only with exactly two nonempty strings in `questions`.
Examples:
Input: What is the capital of Maryland?
Output: {"questions":["What city is the capital of Maryland?","What is Maryland's capital city?"]}
Input: Which country is highlighted?
Output: {"questions":["Which country has been highlighted?","What country is highlighted in the image?"]}"""
PARAPHRASE_SCHEMA = {"type": "object", "properties": {
    "questions": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2}},
    "required": ["questions"], "additionalProperties": False}
_FUNCTION_WORDS = {"a", "an", "are", "as", "at", "be", "by",
                   "do", "does", "for", "from", "how", "in", "is", "of", "on",
                   "that", "the", "these", "this", "to", "was", "were", "what",
                   "when", "where", "which", "who", "why", "with"}


def _retains_content_words(original: str, variant: str) -> bool:
    words = lambda text: set(re.findall(r"[a-z0-9]+", text.lower()))
    return (words(original) - _FUNCTION_WORDS).issubset(words(variant))


def safe_variants(question: str, response_text: str) -> tuple[list[str], list[bool]]:
    context, query = _split_context_and_query(question)
    try:
        payload = json.loads(response_text)
        proposed = payload["questions"]
        if not isinstance(proposed, list):
            proposed = []
    except (TypeError, ValueError, KeyError):
        proposed = []
    variants, changed = [], []
    for value in (proposed + [query, query])[:2]:
        valid = (isinstance(value, str) and "..." not in value and "…" not in value
                 and value.count("?") <= 1 and _semantic_variant_is_safe(query, value)
                 and _retains_content_words(query, value)
                 and _protected_query_features(query)[3] == _protected_query_features(value)[3])
        rewritten = value.strip() if valid else query
        variants.append(context + rewritten)
        changed.append(rewritten != query)
    return variants, changed


def choose_vote(candidates: Sequence[dict[str, Any]]) -> str:
    """Valid majority with original-greedy tie preference; no score access."""
    valid = [candidate["answer"] for candidate in candidates if candidate["format_ok"]]
    if not valid:
        return candidates[0]["answer"]
    counts = Counter(valid)
    return max(counts, key=lambda answer: (counts[answer], -valid.index(answer)))


class TextOnlyRunner:
    def __init__(self, backend, samples: Sequence[MultimodalSample], output: Path,
                 input_sha256: str, model_revision: str, seed: int = 0) -> None:
        if not samples or len({s.sample_id for s in samples}) != len(samples):
            raise ValueError("Nonempty unique ordered samples required")
        self.backend, self.samples, self.output = backend, list(samples), output
        self.predictions = output / "predictions.jsonl"
        self.manifest_path = output / "manifest.json"
        self.seed = seed
        self.git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        self.prompt_sha = digest([TEXT_REWRITE_SYSTEM, PARAPHRASE_SCHEMA,
                                  TRAJECTORY_SYSTEM, FORMAT_USER_SUFFIX])
        self.code_sha = digest([sha256_file(Path(__file__)),
                                sha256_file(Path(__file__).with_name("runner.py")),
                                sha256_file(Path(__file__).with_name("answers.py")),
                                sha256_file(Path(__file__).parents[2] / "scripts" / "run_transfer_format_smoke.py")])
        self.config = {"version": VERSION, "dataset": "scienceqa_img", "labels_read": False,
                       "memory_enabled": False, "input_sha256": input_sha256,
                       "ordered_ids": [s.sample_id for s in samples], "seed": seed,
                       "model": backend.model_name, "model_revision": model_revision,
                       "prompt_sha256": self.prompt_sha, "candidate_max_tokens": 384,
                       "paraphrase_max_tokens": 256, "sample_temperature": 0.7,
                       "sample_top_p": 0.9, "arms": list(ARMS)}
        self.config_sha = digest(self.config)

    def _manifest(self, status: str, processed: int) -> dict[str, Any]:
        return {**self.config, "status": status, "processed": processed,
                "all_arms_complete": status == "completed", "git_sha": self.git_sha,
                "code_sha256": self.code_sha, "config_sha256": self.config_sha,
                "trace_stats": getattr(self.backend, "trace_stats", None)}

    def _write_manifest(self, status: str, processed: int) -> None:
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._manifest(status, processed), indent=2) + "\n")
        temporary.replace(self.manifest_path)

    def _load_prefix(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            if self.predictions.exists():
                raise ValueError("Predictions exist without manifest")
            return []
        manifest = json.loads(self.manifest_path.read_text())
        if (manifest.get("config_sha256") != self.config_sha or
                manifest.get("code_sha256") != self.code_sha or manifest.get("git_sha") != self.git_sha):
            raise ValueError("Resume configuration/code/Git SHA mismatch")
        rows = [json.loads(line) for line in self.predictions.read_text().splitlines()] if self.predictions.exists() else []
        if len(rows) > len(self.samples):
            raise ValueError("Too many committed rows")
        for index, row in enumerate(rows):
            if (row.get("sample_index") != index or row.get("sample_id") != self.samples[index].sample_id
                    or set(row.get("arms", {})) != set(ARMS)):
                raise ValueError("Resume is not a complete ordered prefix")
        return rows

    def _call(self, system: str, user: str, images: Sequence[str], index: int,
              offset: int, temperature: float, max_tokens: int,
              output_schema: dict | None = None):
        return self.backend.generate(system, user, images, SamplingConfig(
            temperature=temperature, top_p=0.9 if temperature else None,
            max_tokens=max_tokens, seed=self.seed + index * 100 + offset,
            output_schema=output_schema))

    def _candidate(self, sample: MultimodalSample, index: int, name: str,
                   question: str, offset: int, temperature: float) -> dict[str, Any]:
        user = format_question(question, sample.choices, sample.metadata) + "\n\n" + FORMAT_USER_SUFFIX
        response = self._call(TRAJECTORY_SYSTEM, user, sample.image_paths, index,
                              offset, temperature, 384)
        answer = normalize_answer(response.content, sample.dataset,
                                  {**sample.metadata, "choices": list(sample.choices)})
        return {"name": name, "question": question, "image_hashes":
                [sha256_file(path) for path in sample.image_paths], "answer": answer,
                "format_ok": format_ok(response, sample.choices)
                and answer_in_choices(answer, sample.choices),
                "response": asdict(response)}

    def run(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        rows = self._load_prefix()
        if len(rows) == len(self.samples):
            self._write_manifest("completed", len(rows))
            return
        self._write_manifest("running", len(rows))
        with self.predictions.open("a", encoding="utf-8") as handle:
            for index in range(len(rows), len(self.samples)):
                sample = self.samples[index]
                sample.validate_images()
                _, query = _split_context_and_query(sample.question)
                rewrite = self._call(TEXT_REWRITE_SYSTEM, query, (), index, 1, 0.0,
                                     256, PARAPHRASE_SCHEMA)
                variants, changed = safe_variants(sample.question, rewrite.content)
                greedy = self._candidate(sample, index, CANDIDATES[0], sample.question, 10, 0.0)
                same = [self._candidate(sample, index, f"same_text_{i + 1}", sample.question,
                                        20 + i, 0.7) for i in range(2)]
                augmented = [self._candidate(sample, index, f"text_aug_{i + 1}", variants[i],
                                             20 + i, 0.7) for i in range(2)]
                candidates = [greedy, *same, *augmented]
                arms = {"greedy": {"answer": greedy["answer"], "candidate_names": [CANDIDATES[0]]},
                        "same_text_vote": {"answer": choose_vote([greedy, *same]),
                                           "candidate_names": list(CANDIDATES[:3])},
                        "text_aug_vote": {"answer": choose_vote([greedy, *augmented]),
                                         "candidate_names": [CANDIDATES[0], *CANDIDATES[3:]]}}
                row = {"sample_index": index, "sample_id": sample.sample_id,
                       "dataset": sample.dataset, "question": sample.question,
                       "choices": list(sample.choices), "metadata": sample.metadata,
                       "image_hashes": greedy["image_hashes"], "rewrite": {
                           "response": asdict(rewrite), "variants": variants,
                           "changed": changed}, "candidates": candidates, "arms": arms}
                handle.write(canonical_json(row) + "\n")
                handle.flush()
                self._write_manifest("running", index + 1)
                if index == 0 or (index + 1) % 10 == 0:
                    print(f"[{index + 1}/{len(self.samples)}] {sample.sample_id} "
                          f"changed={sum(changed)}", flush=True)
        self._write_manifest("completed", len(self.samples))
