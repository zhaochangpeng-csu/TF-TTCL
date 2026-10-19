#!/usr/bin/env python3
"""Offline-only ScienceQA scoring after every direct-transfer arm passes audit."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.multimodal.datasets import load_local_dataset
from core.multimodal.tf_ttcl_direct_runner import ARMS
from evaluate.multimodal.metrics import (paired_bootstrap_delta_ci,
                                         paired_randomization_p_value,
                                         score_prediction)
from scripts.audit_tf_ttcl_mm_direct import audit
from scripts.create_label_isolated_subset import sha256_file


def _cost(rows: list[dict], trace_dir: Path) -> dict:
    fields = ("prompt_tokens", "completion_tokens", "latency_seconds")
    logical = {arm: {field: sum(row["arms"][arm]["usage"][field] for row in rows)
                     for field in fields} for arm in ARMS}
    physical = {field: 0.0 for field in fields}
    count = 0
    for path in trace_dir.glob("*.json"):
        response = json.loads(path.read_text())["response"]
        count += 1
        for field in fields:
            physical[field] += response[field]
    return {"logical_per_arm": logical, "unique_executed_requests": count,
            "physical_shared_trace_total": physical,
            "logical_costs_include_shared_calls": True}


def compare(run: Path, inference: Path, labels: Path, expected_n: int = 400) -> dict:
    manifest, rows = audit(run, inference, expected_n)
    if expected_n != 400:
        raise ValueError("Smoke must remain unscored; only complete dev400 may open labels")
    targets = load_local_dataset(str(labels), "scienceqa_img", str(labels.parent))
    if len(targets) != expected_n or [target.sample.sample_id for target in targets] != manifest["ordered_ids"]:
        raise ValueError("Isolated labels do not match the completed ordered stream")
    if any(target.target is None for target in targets):
        raise ValueError("Missing offline target")
    scores = {arm: [score_prediction("scienceqa_img", row["arms"][arm]["answer"],
                                     target.target.answers, target.target.metadata)
                    for row, target in zip(rows, targets)] for arm in ARMS}
    baselines = ("greedy", "vote_no_rewrite", "select_no_memory")
    results = {}
    for arm in ARMS:
        values = scores[arm]
        result = {"accuracy": sum(values) / expected_n,
                  "front_middle_back_accuracy": [sum(values[lo:hi]) / (hi - lo)
                      for lo, hi in ((0, expected_n // 3), (expected_n // 3, 2 * expected_n // 3),
                                     (2 * expected_n // 3, expected_n))],
                  "answer_change_vs_greedy": sum(row["arms"][arm]["answer"] !=
                     row["arms"]["greedy"]["answer"] for row in rows),
                  "selection_reasons": dict(Counter(row["arms"][arm].get("reason", "greedy")
                                                    for row in rows))}
        for base in baselines:
            if arm == base:
                continue
            changed = Counter((int(a), int(b)) for a, b in zip(scores[base], values))
            result[f"vs_{base}"] = {
                "corrected": changed[(0, 1)], "harmed": changed[(1, 0)],
                "paired_delta_ci": paired_bootstrap_delta_ci(scores[base], values),
                "paired_randomization_p": paired_randomization_p_value(scores[base], values),
            }
        results[arm] = result
    by_subject: dict[str, dict[str, float]] = {}
    groups: dict[str, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        groups[str(target.sample.metadata.get("subject", "unknown"))].append(index)
    for subject, indices in groups.items():
        if len(indices) >= 20:
            by_subject[subject] = {"n": len(indices), **{
                arm: sum(scores[arm][index] for index in indices) / len(indices) for arm in ARMS}}
    return {"status": "scored_after_complete_four_arm_audit", "n": expected_n,
            "run_git_sha": manifest["git_sha"], "input_sha256": manifest["input_sha256"],
            "labels_sha256": sha256_file(labels), "results": results,
            "cost": _cost(rows, run / "response_trace"),
            "rule_writes": sum(len(row["written_rules"]) for row in rows),
            "rule_retrieval_coverage": sum(any(row["retrieved_rules"].values()) for row in rows),
            "distinct_tutor_variants": dict(Counter(len(set(row["variants"])) for row in rows)),
            "by_subject_exploratory": by_subject,
            "interpretation": "One ScienceQA train development stream and one ordering seed; subject slices are exploratory, not confirmatory."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite scored result")
    result = compare(args.run, args.inference, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "n": result["n"],
                      "accuracy": {arm: result["results"][arm]["accuracy"] for arm in ARMS}},
                     indent=2))


if __name__ == "__main__":
    main()
