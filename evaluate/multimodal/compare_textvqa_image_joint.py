#!/usr/bin/env python3
"""Audit image/joint TextVQA arms before opening isolated labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.multimodal.answers import extract_final_answer, normalize_answer
from core.multimodal.image_ops import make_textvqa_photometric_views, sha256_file as image_sha256
from core.multimodal.schema import GenerationResponse
from core.multimodal.text_only_runner import choose_vote
from core.multimodal.textvqa_image_runner import ARMS, NEW_CANDIDATES, VERSION
from core.multimodal.textvqa_text_runner import textvqa_format_ok
from core.multimodal.transfer_runner import digest
from evaluate.multimodal.compare_textvqa_text import audit as audit_baseline, _paired
from evaluate.multimodal.metrics import score_prediction
from scripts.create_label_isolated_subset import sha256_file
from core.multimodal.datasets import load_local_dataset


def audit(path: Path, baseline: Path, inference: Path, expected_n: int) -> tuple[dict, list[dict], list[dict]]:
    if expected_n < 1:
        raise ValueError("A nonempty paired run is required")
    baseline_n = json.loads((baseline / "manifest.json").read_text())["processed"]
    if baseline_n < expected_n:
        raise ValueError("Baseline shorter than paired run")
    base_manifest, all_base_rows = audit_baseline(baseline, inference, baseline_n)
    base_rows = all_base_rows[:expected_n]
    manifest = json.loads((path / "manifest.json").read_text())
    rows = [json.loads(line) for line in (path / "predictions.jsonl").read_text().splitlines()]
    examples = load_local_dataset(str(inference), "textvqa", str(inference.parent), limit=expected_n)
    if (manifest.get("status") != "completed" or manifest.get("all_arms_complete") is not True
            or manifest.get("labels_read") is not False or manifest.get("memory_enabled") is not False
            or manifest.get("version") != VERSION or manifest.get("processed") != expected_n
            or manifest.get("input_sha256") != sha256_file(inference)
            or manifest.get("baseline_predictions_sha256") != sha256_file(baseline / "predictions.jsonl")
            or manifest.get("baseline_git_sha") != base_manifest["git_sha"]
            or manifest.get("model_revision") != base_manifest["model_revision"]
            or manifest.get("seed") != base_manifest["seed"]
            or manifest.get("arms") != list(ARMS)
            or len(rows) != expected_n or len(examples) != expected_n
            or manifest.get("ordered_ids") != [row.get("sample_id") for row in rows]):
        raise ValueError("Incomplete or mismatched image/joint run; labels remain unopened")
    for index, (row, base, example) in enumerate(zip(rows, base_rows, examples)):
        sample = example.sample
        if (example.target is not None or row.get("sample_index") != index
                or row.get("sample_id") != sample.sample_id
                or row.get("base_row_sha256") != digest(base)
                or row.get("source_image_hashes") != [image_sha256(p) for p in sample.image_paths]
                or set(row.get("arms", {})) != set(ARMS)
                or [c.get("name") for c in row.get("new_candidates", [])] != list(NEW_CANDIDATES)
                or any(key in row for key in ("target", "label", "gold", "answers"))):
            raise ValueError("Bad paired image/joint row or target leakage")
        views = make_textvqa_photometric_views(sample.image_paths, path / "image_views", sample.sample_id)
        hashes = [[image_sha256(p) for p in view] for view in views]
        if row.get("view_hashes") != hashes:
            raise ValueError("Image view hash changed")
        candidates = row["new_candidates"]
        questions = [sample.question, sample.question, *base["rewrite"]["variants"]]
        for i, (candidate, question) in enumerate(zip(candidates, questions)):
            response = GenerationResponse(**candidate["response"])
            raw = extract_final_answer(response.content)
            answer = normalize_answer(raw, "textvqa")
            if (candidate["question"] != question or candidate["image_hashes"] != hashes[i % 2]
                    or candidate["raw_answer"] != raw or candidate["answer"] != answer
                    or candidate["format_ok"] != textvqa_format_ok(response, answer)):
                raise ValueError("Image/joint candidate contract broken")
        anchor = base["candidates"][0]
        expected = {"image_only_vote": choose_vote([anchor, *candidates[:2]]),
                    "joint_vote": choose_vote([anchor, *candidates[2:]])}
        for arm in ARMS[:3]:
            if row["arms"][arm] != base["arms"][arm]:
                raise ValueError("Frozen baseline arm changed")
        for arm, value in expected.items():
            names = ["original_greedy", *(NEW_CANDIDATES[:2] if arm == "image_only_vote" else NEW_CANDIDATES[2:])]
            if row["arms"][arm] != {"answer": value, "candidate_names": names}:
                raise ValueError("Image/joint vote differs from candidates")
    return manifest, rows, base_rows


def _raw_vote_answer(row: dict, base: dict, arm: str) -> str:
    selected = row["arms"][arm]["answer"]
    names = set(row["arms"][arm]["candidate_names"])
    for candidate in [*base["candidates"], *row["new_candidates"]]:
        if candidate["name"] in names and candidate["answer"] == selected:
            return candidate["raw_answer"]
    raise ValueError("Selected answer has no supporting candidate")


def compare(path: Path, baseline: Path, inference: Path, labels: Path) -> dict:
    manifest, rows, base_rows = audit(path, baseline, inference, 5000)
    targets = load_local_dataset(str(labels), "textvqa", str(labels.parent))
    if ([item.sample.sample_id for item in targets] != manifest["ordered_ids"]
            or any(item.target is None or len(item.target.answers) != 10 for item in targets)):
        raise ValueError("Isolated labels do not match the complete stream")
    scores = {arm: [score_prediction("textvqa", _raw_vote_answer(row, base, arm), item.target.answers)
                    for row, base, item in zip(rows, base_rows, targets)] for arm in ARMS}
    cost = {arm: {key: 0.0 for key in ("prompt_tokens", "completion_tokens", "latency_seconds")}
            for arm in ARMS}
    for row, base in zip(rows, base_rows):
        old, new, rewrite = base["candidates"], row["new_candidates"], base["rewrite"]["response"]
        requests = {
            "greedy": [old[0]["response"]],
            "same_text_vote": [rewrite, *[c["response"] for c in old[:3]]],
            "text_aug_vote": [rewrite, old[0]["response"], *[c["response"] for c in old[3:]]],
            "image_only_vote": [rewrite, old[0]["response"], *[c["response"] for c in new[:2]]],
            "joint_vote": [rewrite, old[0]["response"], *[c["response"] for c in new[2:]]],
        }
        for arm, responses in requests.items():
            for response in responses:
                for key in cost[arm]:
                    cost[arm][key] += response[key]
    results = {arm: {"vqa_accuracy": sum(scores[arm]) / 5000,
                     "score_sum": sum(scores[arm])} for arm in ARMS}
    return {"status": "scored_after_paired_audit", "n": 5000,
            "run_git_sha": manifest["git_sha"], "baseline_git_sha": manifest["baseline_git_sha"],
            "input_sha256": manifest["input_sha256"], "labels_sha256": sha256_file(labels),
            "results": results, "logical_cost": cost,
            "image_minus_same": _paired(scores["same_text_vote"], scores["image_only_vote"]),
            "joint_minus_text": _paired(scores["text_aug_vote"], scores["joint_vote"]),
            "joint_minus_image": _paired(scores["image_only_vote"], scores["joint_vote"]),
            "image_minus_greedy": _paired(scores["greedy"], scores["image_only_vote"]),
            "joint_minus_greedy": _paired(scores["greedy"], scores["joint_vote"]),
            "invalid_new_candidates": sum(not c["format_ok"] for row in rows for c in row["new_candidates"]),
            "metric_note": "Local deterministic VQA soft score; official EvalAI processor parity not yet verified."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an evaluation")
    result = compare(args.run, args.baseline, args.inference, args.labels)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
