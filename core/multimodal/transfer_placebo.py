"""Label-blind, exact-current-candidate placebo control for transfer v1.

Only the historical rule block in the recorded noisy-arm final prompt changes.
The source four-arm run is immutable and must pass its complete audit first.
This is a post-hoc development diagnostic, not an untouched evaluation.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from evaluate.multimodal.compare_transfer import audit_four
from scripts.create_label_isolated_subset import sha256_file
from .answers import normalize_answer
from .schema import GenerationResponse, MultimodalSample, SamplingConfig
from .transfer_runner import (REFINE_SYSTEM, answer_in_choices, canonical_json,
                              digest, rule_context)
from scripts.run_transfer_format_smoke import format_ok


VERSION = "transfer_placebo_v1_exact_candidate_replay"
_PLACEBO_WORDS = ("An earlier record was withheld and provides no information "
                  "about this task or its conclusion").split()
_WORDS = re.compile(r"\b[\w'-]+\b")


def _matched_placebo(source: str) -> str:
    count = len(_WORDS.findall(source))
    return " ".join(_PLACEBO_WORDS[i % len(_PLACEBO_WORDS)] for i in range(count))


def placebo_context(retrieved: Sequence[dict[str, Any]]) -> str:
    if not retrieved:
        return rule_context([])
    lines = ["Historical checking procedures (uncertified; verify on the current image):"]
    for number, entry in enumerate(retrieved, 1):
        lines.append(f"{number}. Check: {_matched_placebo(entry['positive_rule'])}")
        if entry["negative_rule"]:
            lines.append(f"   Possible failure: {_matched_placebo(entry['negative_rule'])}")
    return "\n".join(lines)


def replace_rule_block(record: dict[str, Any]) -> str:
    source = record["arms"]["noisy"]
    retrieved = source["retrieved"]
    # The original formatter is the source of truth; failure here indicates
    # protocol drift, rather than silently constructing a different prompt.
    from .transfer_runner import RuleEntry
    original = rule_context([RuleEntry.from_dict(entry) for entry in retrieved])
    prompt = source["input_user_text"]
    marker = "\n\n" + original + "\n\n"
    if prompt.count(marker) != 1:
        raise ValueError("Recorded noisy prompt does not contain its exact rule block once")
    return prompt.replace(marker, "\n\n" + placebo_context(retrieved) + "\n\n", 1)


class PlaceboRunner:
    def __init__(self, backend, samples: Sequence[MultimodalSample], source: Path,
                 output: Path, input_sha256: str, model_revision: str, seed: int = 0,
                 source_expected_n: int = 400):
        self.backend, self.samples, self.source, self.output = backend, list(samples), source, output
        self.predictions = output / "predictions.jsonl"
        self.manifest_path = output / "manifest.json"
        self.source_manifest, self.source_rows = audit_four(source, source_expected_n)
        if self.source_manifest["input_sha256"] != input_sha256:
            raise ValueError("Input hash differs from audited source run")
        if [s.sample_id for s in self.samples] != self.source_manifest["ordered_ids"][:len(self.samples)]:
            raise ValueError("Placebo samples differ from source prefix")
        if not 1 <= len(self.samples) <= source_expected_n:
            raise ValueError("Placebo count must be an audited source prefix")
        self.config = {"version": VERSION, "source_run_sha": self.source_manifest["git_sha"],
                       "source_input_sha256": input_sha256, "source_manifest_sha256": sha256_file(source / "manifest.json"),
                       "ordered_ids": [s.sample_id for s in self.samples], "model": backend.model_name,
                       "model_revision": model_revision, "seed": seed,
                       "prompt_sha256": digest([REFINE_SYSTEM, _PLACEBO_WORDS]),
                       "source_labels_previously_scored": True, "labels_read_during_inference": False,
                       "source_expected_n": source_expected_n}
        self.config_sha = digest(self.config)
        self.code_sha = digest([sha256_file(Path(__file__)),
                                sha256_file(Path(__file__).with_name("transfer_runner.py"))])
        self.git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    def _manifest(self, status: str, processed: int) -> dict[str, Any]:
        return {**self.config, "config_sha256": self.config_sha, "code_sha256": self.code_sha,
                "git_sha": self.git_sha, "status": status, "processed": processed,
                "trace_stats": getattr(self.backend, "trace_stats", None)}

    def _write_manifest(self, status: str, processed: int) -> None:
        path = self.manifest_path.with_suffix(".tmp")
        path.write_text(json.dumps(self._manifest(status, processed), ensure_ascii=False, indent=2) + "\n")
        path.replace(self.manifest_path)

    def _resume_prefix(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            if self.predictions.exists():
                raise ValueError("Placebo predictions exist without manifest")
            return []
        old = json.loads(self.manifest_path.read_text())
        if any(old.get(key) != value for key, value in
               (("config_sha256", self.config_sha), ("code_sha256", self.code_sha),
                ("git_sha", self.git_sha))):
            raise ValueError("Placebo resume fingerprint mismatch")
        rows = [json.loads(line) for line in self.predictions.read_text().splitlines()]
        if len(rows) > len(self.samples):
            raise ValueError("Too many placebo records")
        for index, row in enumerate(rows):
            if (row["sample_index"] != index or row["sample_id"] != self.samples[index].sample_id
                    or row["source_row_sha256"] != digest(self.source_rows[index])):
                raise ValueError("Placebo resume is not an exact source prefix")
        return rows

    def run(self) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        rows = self._resume_prefix()
        if len(rows) == len(self.samples):
            self._write_manifest("completed", len(rows))
            return
        self._write_manifest("running", len(rows))
        with self.predictions.open("a", encoding="utf-8") as handle:
            for index in range(len(rows), len(self.samples)):
                sample = self.samples[index]
                source = self.source_rows[index]
                sample.validate_images()
                prompt = replace_rule_block(source)
                noisy = source["arms"]["noisy"]
                if not noisy["retrieved"]:
                    if prompt != noisy["input_user_text"]:
                        raise ValueError("Empty-history prompt changed")
                    response = noisy["response"]
                    reused_source = True
                else:
                    result = self.backend.generate(
                        REFINE_SYSTEM, prompt, sample.image_paths,
                        SamplingConfig(temperature=0.0, max_tokens=384, seed=self.config["seed"] + index * 100 + 90))
                    response = asdict(result)
                    reused_source = False
                answer = normalize_answer(response["content"], sample.dataset,
                                          {**sample.metadata, "choices": sample.choices})
                valid = format_ok(GenerationResponse(**response), sample.choices) and answer_in_choices(answer, sample.choices)
                if not valid:
                    answer = source["pseudo_pair"]["positive"]
                row = {"sample_index": index, "sample_id": sample.sample_id,
                       "source_row_sha256": digest(source), "retrieved_source_ids":
                       [e["sample_id"] for e in noisy["retrieved"]], "input_user_text": prompt,
                       "response": response, "answer": answer, "format_ok": valid,
                       "reused_identical_source_response": reused_source}
                handle.write(canonical_json(row) + "\n")
                handle.flush()
                self._write_manifest("running", index + 1)
                if (index + 1) % 20 == 0 or index == 0:
                    print(f"[{index + 1}/{len(self.samples)}] placebo", flush=True)
        self._write_manifest("completed", len(self.samples))
