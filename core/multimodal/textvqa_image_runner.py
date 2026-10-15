"""Label-blind paired image-only and text+image TextVQA controls.

The original greedy answer and frozen rewrites come from an audited prior
text-only run. New arms use the same two image views and sampling seeds; only
the question differs. No labels or prior scores are read by this module.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .answers import extract_final_answer, normalize_answer
from .image_ops import make_textvqa_photometric_views, sha256_file
from .prompts import format_question
from .schema import MultimodalSample
from .text_only_runner import TextOnlyRunner, choose_vote
from .textvqa_text_runner import TEXTVQA_SUFFIX, TEXTVQA_SYSTEM, textvqa_format_ok
from .transfer_runner import canonical_json, digest


VERSION = "textvqa_image_joint_vote_v1"
ARMS = ("greedy", "same_text_vote", "text_aug_vote", "image_only_vote", "joint_vote")
NEW_CANDIDATES = ("image_only_1", "image_only_2", "joint_1", "joint_2")


class TextVQAImageRunner(TextOnlyRunner):
    def __init__(self, backend, samples: Sequence[MultimodalSample], output: Path,
                 input_sha256: str, model_revision: str, baseline_manifest: dict,
                 baseline_rows: Sequence[dict], baseline_predictions_sha256: str,
                 seed: int = 0) -> None:
        if any(sample.dataset != "textvqa" for sample in samples):
            raise ValueError("TextVQA samples required")
        if (len(samples) != len(baseline_rows)
                or [s.sample_id for s in samples] != [r["sample_id"] for r in baseline_rows]
                or baseline_manifest.get("input_sha256") != input_sha256
                or baseline_manifest.get("model_revision") != model_revision
                or baseline_manifest.get("seed") != seed
                or baseline_manifest.get("status") != "completed"):
            raise ValueError("Audited baseline is not paired with this run")
        super().__init__(backend, samples, output, input_sha256, model_revision, seed)
        self.baseline_rows = list(baseline_rows)
        self.views_root = output / "image_views"
        self.prompt_sha = digest([self.prompt_sha, TEXTVQA_SYSTEM, TEXTVQA_SUFFIX])
        self.code_sha = digest([self.code_sha, sha256_file(Path(__file__)),
                                sha256_file(Path(__file__).with_name("image_ops.py")),
                                sha256_file(Path(__file__).with_name("textvqa_text_runner.py"))])
        self.config.update(version=VERSION, dataset="textvqa", prompt_sha256=self.prompt_sha,
                           arms=list(ARMS), image_views="full_frame_unsharp60_contrast1.08",
                           baseline_git_sha=baseline_manifest["git_sha"],
                           baseline_predictions_sha256=baseline_predictions_sha256)
        self.config_sha = digest(self.config)

    def _load_prefix(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            if self.predictions.exists():
                raise ValueError("Predictions exist without manifest")
            return []
        manifest = json.loads(self.manifest_path.read_text())
        if (manifest.get("config_sha256") != self.config_sha
                or manifest.get("code_sha256") != self.code_sha
                or manifest.get("git_sha") != self.git_sha):
            raise ValueError("Resume configuration/code/Git SHA mismatch")
        rows = [json.loads(line) for line in self.predictions.read_text().splitlines()] if self.predictions.exists() else []
        if len(rows) > len(self.samples):
            raise ValueError("Too many committed rows")
        for index, row in enumerate(rows):
            if (row.get("sample_index") != index
                    or row.get("sample_id") != self.samples[index].sample_id
                    or row.get("base_row_sha256") != digest(self.baseline_rows[index])
                    or set(row.get("arms", {})) != set(ARMS)
                    or [c.get("name") for c in row.get("new_candidates", [])] != list(NEW_CANDIDATES)):
                raise ValueError("Resume is not a complete ordered paired prefix")
        return rows

    def _candidate(self, sample: MultimodalSample, index: int, name: str,
                   question: str, image_paths: Sequence[str], offset: int) -> dict:
        user = format_question(question, sample.choices, sample.metadata) + "\n\n" + TEXTVQA_SUFFIX
        response = self._call(TEXTVQA_SYSTEM, user, image_paths, index, offset, 0.7, 384)
        raw = extract_final_answer(response.content)
        answer = normalize_answer(raw, "textvqa")
        return {"name": name, "question": question,
                "image_hashes": [sha256_file(path) for path in image_paths],
                "answer": answer, "raw_answer": raw,
                "format_ok": textvqa_format_ok(response, answer),
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
                sample, base = self.samples[index], self.baseline_rows[index]
                sample.validate_images()
                views = make_textvqa_photometric_views(sample.image_paths, self.views_root,
                                                       sample.sample_id)
                variants = base["rewrite"]["variants"]
                image_only = [self._candidate(sample, index, NEW_CANDIDATES[i],
                                              sample.question, views[i], 20 + i)
                              for i in range(2)]
                joint = [self._candidate(sample, index, NEW_CANDIDATES[2 + i],
                                         variants[i], views[i], 20 + i)
                         for i in range(2)]
                anchor = base["candidates"][0]
                arms = {name: dict(base["arms"][name]) for name in ARMS[:3]}
                arms["image_only_vote"] = {
                    "answer": choose_vote([anchor, *image_only]),
                    "candidate_names": ["original_greedy", *NEW_CANDIDATES[:2]]}
                arms["joint_vote"] = {
                    "answer": choose_vote([anchor, *joint]),
                    "candidate_names": ["original_greedy", *NEW_CANDIDATES[2:]]}
                row = {"sample_index": index, "sample_id": sample.sample_id,
                       "base_row_sha256": digest(base),
                       "source_image_hashes": [sha256_file(p) for p in sample.image_paths],
                       "view_hashes": [[sha256_file(p) for p in view] for view in views],
                       "new_candidates": [*image_only, *joint], "arms": arms}
                handle.write(canonical_json(row) + "\n")
                handle.flush()
                self._write_manifest("running", index + 1)
                if index == 0 or (index + 1) % 10 == 0:
                    print(f"[{index + 1}/{len(self.samples)}] {sample.sample_id}", flush=True)
        self._write_manifest("completed", len(self.samples))
