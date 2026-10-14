#!/usr/bin/env python3
"""Audit all MathVista paired predictions before opening isolated labels."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.multimodal.datasets import load_local_dataset
from core.multimodal.image_ops import sha256_file as image_sha256
from core.multimodal.mathvista_text_runner import VERSION, math_format_ok
from core.multimodal.schema import GenerationResponse
from core.multimodal.text_only_runner import ARMS, CANDIDATES, choose_vote, safe_variants
from evaluate.multimodal.metrics import paired_bootstrap_delta_ci, paired_randomization_p_value, score_prediction
from scripts.create_label_isolated_subset import sha256_file


def audit(path: Path, inference: Path, expected_n: int) -> tuple[dict, list[dict]]:
    manifest = json.loads((path / "manifest.json").read_text())
    rows = [json.loads(line) for line in (path / "predictions.jsonl").read_text().splitlines()]
    samples = load_local_dataset(str(inference), "mathvista", str(inference.parent), limit=expected_n)
    if (manifest.get("status") != "completed" or manifest.get("all_arms_complete") is not True
            or manifest.get("labels_read") is not False or manifest.get("memory_enabled") is not False
            or manifest.get("version") != VERSION or manifest.get("processed") != expected_n
            or manifest.get("input_sha256") != sha256_file(inference)
            or len(rows) != expected_n or len(samples) != expected_n
            or manifest.get("ordered_ids") != [row.get("sample_id") for row in rows]
            or manifest.get("arms") != list(ARMS)):
        raise ValueError("Incomplete or mismatched MathVista run; labels remain unopened")
    for index, (row, example) in enumerate(zip(rows, samples)):
        sample = example.sample
        if (example.target is not None or row.get("sample_index") != index
                or row.get("sample_id") != sample.sample_id
                or row.get("question") != sample.question
                or row.get("choices") != list(sample.choices)
                or row.get("metadata") != sample.metadata
                or set(row.get("arms", {})) != set(ARMS)
                or len(row.get("candidates", [])) != 5
                or any(key in row for key in ("target", "label", "gold"))):
            raise ValueError("Bad paired row or target leakage")
        hashes = [image_sha256(image) for image in sample.image_paths]
        if not hashes or row.get("image_hashes") != hashes:
            raise ValueError("Image hashes changed")
        candidates = row["candidates"]
        if [candidate["name"] for candidate in candidates] != list(CANDIDATES):
            raise ValueError("Candidate order changed")
        variants, changed = safe_variants(sample.question, row["rewrite"]["response"]["content"])
        if row["rewrite"]["variants"] != variants or row["rewrite"]["changed"] != changed:
            raise ValueError("Question rewrite mismatch")
        questions = [sample.question] * 3 + variants
        for candidate, question in zip(candidates, questions):
            response = GenerationResponse(**candidate["response"])
            valid = math_format_ok(response, sample, candidate["answer"])
            if (candidate["question"] != question or candidate["image_hashes"] != hashes
                    or candidate["format_ok"] != valid):
                raise ValueError("MathVista image/question/format contract broken")
        expected_answers = (candidates[0]["answer"], choose_vote(candidates[:3]),
                            choose_vote([candidates[0], *candidates[3:]]))
        for arm, expected in zip(ARMS, expected_answers):
            if row["arms"][arm]["answer"] != expected:
                raise ValueError("Arm vote differs from committed candidates")
    return manifest, rows


def _paired(left: list[float], right: list[float]) -> dict:
    pairs = Counter((int(a), int(b)) for a, b in zip(left, right))
    return {"corrected": pairs[(0, 1)], "harmed": pairs[(1, 0)],
            "paired_delta_ci": paired_bootstrap_delta_ci(left, right),
            "paired_randomization_p": paired_randomization_p_value(left, right)}


def compare(path: Path, inference: Path, labels: Path, expected_n: int = 1000) -> dict:
    if expected_n != 1000:
        raise ValueError("Only the complete testmini can be scored")
    manifest, rows = audit(path, inference, expected_n)
    targets = load_local_dataset(str(labels), "mathvista", str(labels.parent))
    if ([item.sample.sample_id for item in targets] != manifest["ordered_ids"]
            or any(item.target is None for item in targets)):
        raise ValueError("Isolated labels do not match the complete stream")
    scores = {arm: [score_prediction("mathvista", row["arms"][arm]["answer"],
                                     item.target.answers, item.target.metadata)
                    for row, item in zip(rows, targets)] for arm in ARMS}
    results = {arm: {"accuracy": sum(scores[arm]) / expected_n,
                     "correct": sum(scores[arm]),
                     "comparisons": {base: _paired(scores[base], scores[arm])
                                     for base in ARMS if base != arm}} for arm in ARMS}
    cost = {arm: {key: 0.0 for key in ("prompt_tokens", "completion_tokens", "latency_seconds")}
            for arm in ARMS}
    for row in rows:
        candidates, rewrite = row["candidates"], row["rewrite"]["response"]
        for arm, responses in (("greedy", [candidates[0]["response"]]),
                               ("same_text_vote", [rewrite, *[c["response"] for c in candidates[:3]]]),
                               ("text_aug_vote", [rewrite, candidates[0]["response"],
                                                  candidates[3]["response"], candidates[4]["response"]])):
            for response in responses:
                for key in cost[arm]:
                    cost[arm][key] += response[key]
    groups = {}
    for group in ("multi_choice", "free_form"):
        indices = [i for i, row in enumerate(rows)
                   if row["metadata"].get("question_type") == group]
        groups[group] = {"n": len(indices),
                         "accuracy": {arm: sum(scores[arm][i] for i in indices) / len(indices)
                                      if indices else None for arm in ARMS},
                         "text_minus_same": _paired([scores["same_text_vote"][i] for i in indices],
                                                    [scores["text_aug_vote"][i] for i in indices])}
    return {"status": "scored_after_paired_audit", "n": expected_n,
            "run_git_sha": manifest["git_sha"], "input_sha256": manifest["input_sha256"],
            "labels_sha256": sha256_file(labels), "results": results,
            "logical_cost": cost, "question_type_slices": groups,
            "rewrite_changed_items": sum(any(row["rewrite"]["changed"]) for row in rows),
            "answer_disagreement_same_vs_text": sum(
                row["arms"]["same_text_vote"]["answer"] != row["arms"]["text_aug_vote"]["answer"]
                for row in rows),
            "format_invalid_candidates": sum(not c["format_ok"] for row in rows for c in row["candidates"]),
            "metric_note": "Local deterministic MathVista accuracy; not claimed official leaderboard parity."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an evaluation")
    result = compare(args.run, args.inference, args.labels, args.expected_n)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
