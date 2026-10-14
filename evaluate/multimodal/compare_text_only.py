#!/usr/bin/env python3
"""Audit all paired text-only arms before opening isolated ScienceQA labels."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.multimodal.datasets import load_local_dataset
from core.multimodal.schema import GenerationResponse
from core.multimodal.text_only_runner import ARMS, CANDIDATES, VERSION, choose_vote, safe_variants
from core.multimodal.transfer_runner import answer_in_choices
from evaluate.multimodal.metrics import (paired_bootstrap_delta_ci,
                                         paired_randomization_p_value, score_prediction)
from scripts.create_label_isolated_subset import sha256_file
from scripts.run_transfer_format_smoke import format_ok


def audit_text_only(path: Path, expected_n: int) -> tuple[dict, list[dict]]:
    """This API has no label path and cannot score the smoke or development run."""
    manifest = json.loads((path / "manifest.json").read_text())
    rows = [json.loads(line) for line in (path / "predictions.jsonl").read_text().splitlines()]
    if (manifest.get("status") != "completed" or manifest.get("all_arms_complete") is not True
            or manifest.get("labels_read") is not False or manifest.get("memory_enabled") is not False
            or manifest.get("version") != VERSION or manifest.get("processed") != expected_n
            or len(rows) != expected_n or manifest.get("ordered_ids") != [row.get("sample_id") for row in rows]
            or manifest.get("arms") != list(ARMS)):
        raise ValueError("Incomplete or mismatched text-only run; labels remain unopened")
    for index, row in enumerate(rows):
        if (row.get("sample_index") != index or set(row.get("arms", {})) != set(ARMS)
                or len(row.get("candidates", [])) != 5 or not row.get("image_hashes")
                or any(key in row for key in ("target", "label", "gold"))):
            raise ValueError("Bad paired row or target leakage")
        candidates = row["candidates"]
        if [candidate["name"] for candidate in candidates] != list(CANDIDATES):
            raise ValueError("Candidate order changed")
        variants, changed = safe_variants(row["question"], row["rewrite"]["response"]["content"])
        if row["rewrite"]["variants"] != variants or row["rewrite"]["changed"] != changed:
            raise ValueError("Question rewrite mismatch")
        expected_questions = [row["question"]] * 3 + variants
        for candidate, question in zip(candidates, expected_questions):
            response = GenerationResponse(**candidate["response"])
            valid = format_ok(response, row["choices"]) and answer_in_choices(candidate["answer"], row["choices"])
            if (candidate["question"] != question or candidate["image_hashes"] != row["image_hashes"]
                    or candidate["format_ok"] != valid):
                raise ValueError("Text-only image/question/format contract broken")
        expected_answers = (candidates[0]["answer"], choose_vote(candidates[:3]),
                            choose_vote([candidates[0], *candidates[3:]]))
        for arm, expected in zip(ARMS, expected_answers):
            if row["arms"][arm]["answer"] != expected:
                raise ValueError("Arm vote differs from committed candidates")
    return manifest, rows


def _cost(responses: list[dict]) -> dict[str, float]:
    return {key: sum(response[key] for response in responses)
            for key in ("prompt_tokens", "completion_tokens", "latency_seconds")}


def compare(path: Path, labels_path: Path, expected_n: int = 400) -> dict:
    manifest, rows = audit_text_only(path, expected_n)
    targets = load_local_dataset(str(labels_path), "scienceqa_img", str(labels_path.parent))
    if (expected_n != 400 or [item.sample.sample_id for item in targets] != manifest["ordered_ids"]
            or any(item.target is None for item in targets)):
        raise ValueError("Isolated labels do not match the complete 400-item stream")
    scores = {arm: [score_prediction("scienceqa_img", row["arms"][arm]["answer"],
                                     item.target.answers, item.target.metadata)
                    for row, item in zip(rows, targets)] for arm in ARMS}
    results = {}
    for arm in ARMS:
        current = scores[arm]
        comparisons = {}
        for baseline in ARMS:
            if baseline == arm:
                continue
            pairs = Counter((int(a), int(b)) for a, b in zip(scores[baseline], current))
            comparisons[baseline] = {
                "corrected": pairs[(0, 1)], "harmed": pairs[(1, 0)],
                "paired_delta_ci": paired_bootstrap_delta_ci(scores[baseline], current),
                "paired_randomization_p": paired_randomization_p_value(scores[baseline], current)}
        results[arm] = {"accuracy": sum(current) / expected_n,
                        "correct": sum(current), "comparisons": comparisons}
    costs = {arm: {key: 0.0 for key in ("prompt_tokens", "completion_tokens", "latency_seconds")}
             for arm in ARMS}
    for row in rows:
        candidates = row["candidates"]
        rewrite = row["rewrite"]["response"]
        for arm, responses in (("greedy", [candidates[0]["response"]]),
                               ("same_text_vote", [rewrite, *[c["response"] for c in candidates[:3]]]),
                               ("text_aug_vote", [rewrite, candidates[0]["response"],
                                                  candidates[3]["response"], candidates[4]["response"]])):
            for key, value in _cost(responses).items():
                costs[arm][key] += value
    return {"status": "scored_after_paired_audit", "n": expected_n,
            "run_git_sha": manifest["git_sha"], "input_sha256": manifest["input_sha256"],
            "labels_sha256": sha256_file(labels_path), "results": results, "logical_cost": costs,
            "rewrite_changed_items": sum(any(row["rewrite"]["changed"]) for row in rows),
            "rewrite_changed_variants": sum(sum(row["rewrite"]["changed"]) for row in rows),
            "answer_disagreement_same_vs_text": sum(
                row["arms"]["same_text_vote"]["answer"] != row["arms"]["text_aug_vote"]["answer"]
                for row in rows),
            "interpretation": "Single-model, single-stream development test of text augmentation, not memory or evidence verification."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an evaluation")
    result = compare(args.run, args.labels, args.expected_n)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
