#!/usr/bin/env python3
"""Four-arm scoring gate: finish all inference before loading any labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.multimodal.datasets import load_local_dataset
from core.multimodal.token_budget import load_plan, namespace
from evaluate.multimodal.audit_causal import load_predictions, manifest_for
from evaluate.multimodal.compare_causal import compare_runs, validate_completed_runs
from evaluate.multimodal.metrics import (
    bootstrap_mean_ci, paired_bootstrap_delta_ci, paired_randomization_p_value, score_prediction,
)


def validate_four(paths: dict, expected_n: int):
    three = {name: paths[name] for name in ("greedy", "shared_budget", "shared_evidence")}
    order, rows, manifests = validate_completed_runs(three, "shared", expected_n)
    path = paths["efficient_budget"]
    manifest = manifest_for(path)
    if manifest.get("status") != "completed" or manifest.get("method") != "efficient_budget":
        raise ValueError("Independent budget run incomplete; refusing to read labels")
    independent_order, independent = load_predictions(path)
    if independent_order != order or manifest.get("processed") != expected_n:
        raise ValueError("Independent budget stream mismatch; refusing to read labels")
    manifests["efficient_budget"], rows["efficient_budget"] = manifest, independent
    reference = manifests["shared_budget"]
    if reference["config"].get("evidence", {}).get("protocol") != "typed_v3":
        raise ValueError("Not a frozen typed-v3 comparison")
    for name, other in manifests.items():
        if (namespace(other["config"]) != namespace(reference["config"])
                or other.get("git_sha") != reference.get("git_sha")):
            raise ValueError(f"{name} scientific configuration mismatch; refusing to read labels")
    load_plan(manifest["config"]["budget_control"]["token_plan"], manifest["config"], order)
    for row in independent.values():
        if not 6 <= len(row["candidate_answers"]) <= 12:
            raise ValueError("Invalid independent sampling count")
        if (row.get("wrote_memory") or row.get("retrieved_experience_ids")
                or row.get("memory_size_before") or row.get("memory_size_after")
                or row.get("shared_certificate")):
            raise ValueError("Independent budget accessed evidence/memory")
    return order, rows, manifests


def compare_four(paths: dict, labels: str, expected_n: int = 400,
                 bootstrap_samples: int = 10000, seed: int = 0):
    order, rows, manifests = validate_four(paths, expected_n)
    config = manifests["shared_budget"]["config"]["experiment"]
    three = {name: paths[name] for name in ("greedy", "shared_budget", "shared_evidence")}
    result = compare_runs(three, labels, config["dataset"], config.get("image_root"),
                          bootstrap_samples, seed, "shared", expected_n)
    targets = {e.target.sample_id: e.target for e in load_local_dataset(
        labels, config["dataset"], config.get("image_root")) if e.target is not None}
    scores = {name: [score_prediction(config["dataset"], str(run[sid]["answer"]),
                    targets[sid].answers, targets[sid].metadata) for sid in order]
              for name, run in rows.items()}
    manifest = manifests["efficient_budget"]
    mean, low, high = bootstrap_mean_ci(scores["efficient_budget"], bootstrap_samples, seed)
    result["runs"]["efficient_budget"] = {
        "accuracy": mean, "ci95": [low, high],
        **{key: manifest.get(key) for key in ("prompt_tokens", "completion_tokens", "latency_seconds",
                                            "git_sha", "prompt_hash", "response_trace")},
    }
    for baseline, candidate in (("greedy", "efficient_budget"), ("efficient_budget", "shared_evidence")):
        left, right = scores[baseline], scores[candidate]
        delta, low, high = paired_bootstrap_delta_ci(left, right, bootstrap_samples, seed)
        result["comparisons"][f"{candidate}_minus_{baseline}"] = {
            "delta": delta, "delta_ci95": [low, high],
            "paired_randomization_p": paired_randomization_p_value(left, right, bootstrap_samples, seed),
            "corrected": sum(b > a for a, b in zip(left, right)),
            "regressed": sum(b < a for a, b in zip(left, right)),
            "answer_changes": [{"sample_id": sid, "baseline_answer": rows[baseline][sid]["answer"],
                "candidate_answer": rows[candidate][sid]["answer"], "baseline_score": left[i],
                "candidate_score": right[i]} for i, sid in enumerate(order)
                if rows[baseline][sid]["answer"] != rows[candidate][sid]["answer"]],
        }
    cost = lambda name: manifests[name]["prompt_tokens"] + manifests[name]["completion_tokens"]
    relative = abs(cost("efficient_budget") - cost("shared_evidence")) / cost("shared_evidence")
    result["independent_budget_audit"] = {
        "relative_token_difference": relative, "within_five_percent": relative <= 0.05,
        "sampling_counts": {str(n): sum(len(r["candidate_answers"]) == n
                                          for r in rows["efficient_budget"].values()) for n in range(6, 13)},
        "cost_targets_contain_answers": False, "allocation": "cost_only_cumulative_6_to_12",
        "reference_costs_known_after_reference_inference": True,
    }
    result["continue_to_validation"] = bool(result["continue_to_validation"] and relative <= 0.05
        and result["comparisons"]["shared_evidence_minus_efficient_budget"]["delta"] > 0)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("greedy", "budget", "evidence", "efficient", "labels"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--expected-n", type=int, default=400)
    args = parser.parse_args()
    result = compare_four({"greedy": Path(args.greedy), "shared_budget": Path(args.budget),
        "shared_evidence": Path(args.evidence), "efficient_budget": Path(args.efficient)},
        args.labels, args.expected_n)
    print(json.dumps(result, indent=2))
