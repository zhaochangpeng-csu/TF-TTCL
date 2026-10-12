#!/usr/bin/env python3
"""Score three completed CCIC development runs; read labels only after checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.multimodal.datasets import load_local_dataset
from evaluate.multimodal.audit_causal import audit_runs, load_predictions, manifest_for
from evaluate.multimodal.metrics import (
    bootstrap_mean_ci,
    paired_bootstrap_delta_ci,
    paired_randomization_p_value,
    score_prediction,
)


def validate_completed_runs(paths: dict[str, Path], family: str = "causal",
                            expected_n: int | None = None) -> tuple[
    list[str], dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, Any]]
]:
    """Require a completed, matched stream before any label-loading call."""
    if family not in {"causal", "shared"}:
        raise ValueError("Unknown certificate family")
    expected_methods = {
        "greedy": "zero_shot",
        f"{family}_budget": f"{family}_budget",
        f"{family}_evidence": f"{family}_evidence",
    }
    if set(paths) != set(expected_methods):
        raise ValueError("Scoring requires all three inference arms")
    manifests = {name: manifest_for(path) for name, path in paths.items()}
    for name, manifest in manifests.items():
        if manifest.get("status") != "completed":
            raise ValueError(f"{name} is not completed; refusing to read labels")
        if manifest.get("method") != expected_methods[name]:
            raise ValueError(f"Unexpected method for {name}")
    loaded = {name: load_predictions(path) for name, path in paths.items()}
    reference = loaded["greedy"][0]
    if not reference:
        raise ValueError("Cannot score an empty stream")
    if expected_n is not None and len(reference) != expected_n:
        raise ValueError("Incomplete planned stream; refusing to read labels")
    for name, (order, _) in loaded.items():
        if order != reference or manifests[name].get("processed") != len(reference):
            raise ValueError(f"{name} does not contain the identical completed stream")
    return reference, {name: rows for name, (_, rows) in loaded.items()}, manifests


def compare_runs(
    paths: dict[str, Path], labels: str, dataset: str,
    image_root: str | None = None, bootstrap_samples: int = 10000, seed: int = 0,
    family: str = "causal",
    expected_n: int | None = None,
) -> dict[str, Any]:
    order, rows, manifests = validate_completed_runs(paths, family, expected_n)
    budget_name, evidence_name = f"{family}_budget", f"{family}_evidence"
    if family == "shared":
        from evaluate.multimodal.audit_shared import audit_runs as shared_audit
        audit = shared_audit(paths[budget_name], paths[evidence_name])
    else:
        audit = audit_runs(paths[budget_name], paths[evidence_name])
    targets = {
        example.target.sample_id: example.target
        for example in load_local_dataset(labels, dataset, image_root)
        if example.target is not None
    }
    if any(sample_id not in targets for sample_id in order):
        raise ValueError("Evaluation targets missing from the completed stream")
    scores = {
        name: [
            score_prediction(dataset, str(run[sample_id]["answer"]),
                             targets[sample_id].answers, targets[sample_id].metadata)
            for sample_id in order
        ]
        for name, run in rows.items()
    }
    aggregate = {}
    for name, values in scores.items():
        mean, low, high = bootstrap_mean_ci(values, bootstrap_samples, seed)
        manifest = manifests[name]
        aggregate[name] = {
            "accuracy": mean, "ci95": [low, high],
            "prompt_tokens": manifest.get("prompt_tokens", 0),
            "completion_tokens": manifest.get("completion_tokens", 0),
            "latency_seconds": manifest.get("latency_seconds", 0),
            "git_sha": manifest.get("git_sha"), "prompt_hash": manifest.get("prompt_hash"),
            "response_trace": manifest.get("response_trace"),
        }
    comparisons = {}
    for baseline, candidate in [
        ("greedy", budget_name), ("greedy", evidence_name),
        (budget_name, evidence_name),
    ]:
        left, right = scores[baseline], scores[candidate]
        delta, low, high = paired_bootstrap_delta_ci(left, right, bootstrap_samples, seed)
        changes = [
            {"sample_id": sample_id,
             "baseline_answer": rows[baseline][sample_id]["answer"],
             "candidate_answer": rows[candidate][sample_id]["answer"],
             "baseline_score": left[i], "candidate_score": right[i]}
            for i, sample_id in enumerate(order)
            if rows[baseline][sample_id]["answer"] != rows[candidate][sample_id]["answer"]
        ]
        comparisons[f"{candidate}_minus_{baseline}"] = {
            "delta": delta, "delta_ci95": [low, high],
            "paired_randomization_p": paired_randomization_p_value(
                left, right, bootstrap_samples, seed),
            "corrected": sum(b > a for a, b in zip(left, right)),
            "regressed": sum(b < a for a, b in zip(left, right)),
            "answer_changes": changes,
        }
    budget = comparisons[f"{evidence_name}_minus_{budget_name}"]
    selected = [
        scores[evidence_name][i]
        for i, sample_id in enumerate(order)
        if rows[evidence_name][sample_id].get(f"{family}_certificate", {}).get("gate_passed")
    ]
    result = {
        "dataset": dataset, "n": len(order), "runs": aggregate,
        "comparisons": comparisons, "label_free_protocol_audit": audit,
        "selective_accuracy": sum(selected) / len(selected) if selected else None,
        "continue_to_validation": bool(
            audit["protocol_ok"]
            and aggregate[evidence_name]["accuracy"] > aggregate["greedy"]["accuracy"]
            and budget["delta"] > 0
        ),
    }
    if family == "shared":
        anchors = [next(c["answer"] for c in rows[budget_name][sid]["candidate_answers"]
                        if c["trajectory"] == "original_greedy") for sid in order]
        anchor_scores = [score_prediction(dataset, str(answer), targets[sid].answers,
                                         targets[sid].metadata)
                         for sid, answer in zip(order, anchors)]
        delta, low, high = paired_bootstrap_delta_ci(
            anchor_scores, scores[evidence_name], bootstrap_samples, seed)
        result["internal_greedy_anchor"] = {
            "accuracy": sum(anchor_scores) / len(anchor_scores),
            "standalone_answer_mismatch_ids": [sid for sid, answer in zip(order, anchors)
                                                if answer != rows["greedy"][sid]["answer"]],
            "evidence_delta": delta, "delta_ci95": [low, high],
            "corrected": sum(b > a for a, b in zip(anchor_scores, scores[evidence_name])),
            "regressed": sum(b < a for a, b in zip(anchor_scores, scores[evidence_name])),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy", required=True)
    parser.add_argument("--budget", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--family", choices=["causal", "shared"], default="causal")
    parser.add_argument("--expected-n", type=int)
    args = parser.parse_args()
    result = compare_runs(
        {"greedy": Path(args.greedy), f"{args.family}_budget": Path(args.budget),
         f"{args.family}_evidence": Path(args.evidence)},
        args.labels, args.dataset, args.image_root, args.bootstrap_samples, args.seed,
        args.family,
        args.expected_n,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
