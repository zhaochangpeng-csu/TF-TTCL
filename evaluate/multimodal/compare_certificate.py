#!/usr/bin/env python3
"""Evaluate greedy, certificate-budget, and certificate-evidence paired runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from core.multimodal.datasets import load_local_dataset
from evaluate.multimodal.metrics import (
    bootstrap_mean_ci,
    paired_bootstrap_delta_ci,
    paired_randomization_p_value,
    score_prediction,
)


def _load_predictions(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    order: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row["sample_id"])
        if sample_id in rows:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in {path}")
        order.append(sample_id)
        rows[sample_id] = row
    return order, rows


def _manifest_for(predictions: Path) -> dict[str, Any]:
    manifest = predictions.parent / "manifest.json"
    return json.loads(manifest.read_text(encoding="utf-8"))


def _call_trace(row: dict[str, Any]) -> tuple[Any, ...]:
    candidates = tuple(
        (
            item.get("trajectory"),
            item.get("question_variant"),
            tuple(item.get("image_hashes", [])),
        )
        for item in row.get("candidate_answers", [])
    )
    certificate = row.get("evidence_certificate", {})
    views = tuple(
        (
            item.get("view"),
            item.get("candidate_1"),
            item.get("candidate_2"),
            tuple(item.get("image_hashes", [])),
        )
        for item in certificate.get("views", [])
    )
    return candidates, views


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
    args = parser.parse_args()

    paths = {
        "greedy": Path(args.greedy),
        "certificate_budget": Path(args.budget),
        "certificate_evidence": Path(args.evidence),
    }
    loaded = {name: _load_predictions(path) for name, path in paths.items()}
    reference_order = loaded["greedy"][0]
    reference_ids = set(reference_order)
    for name, (order, rows) in loaded.items():
        if set(rows) != reference_ids or order != reference_order:
            raise ValueError(f"{name} does not contain the identical ordered sample stream")

    targets = {
        item.target.sample_id: item.target
        for item in load_local_dataset(args.labels, args.dataset, args.image_root)
        if item.target is not None
    }
    missing = [sample_id for sample_id in reference_order if sample_id not in targets]
    if missing:
        raise ValueError(f"Missing targets for sample IDs: {missing[:10]}")

    scores: dict[str, list[float]] = {name: [] for name in paths}
    for sample_id in reference_order:
        target = targets[sample_id]
        for name, (_, rows) in loaded.items():
            scores[name].append(
                score_prediction(
                    args.dataset,
                    str(rows[sample_id]["answer"]),
                    target.answers,
                    target.metadata,
                )
            )

    aggregate: dict[str, Any] = {}
    for name, values in scores.items():
        mean, low, high = bootstrap_mean_ci(values, args.bootstrap_samples, args.seed)
        manifest = _manifest_for(paths[name])
        aggregate[name] = {
            "score": mean,
            "ci95": [low, high],
            "prompt_tokens": int(manifest.get("prompt_tokens", 0)),
            "completion_tokens": int(manifest.get("completion_tokens", 0)),
            "latency_seconds": float(manifest.get("latency_seconds", 0.0)),
        }

    greedy_scores = scores["greedy"]
    evidence_scores = scores["certificate_evidence"]
    delta, low, high = paired_bootstrap_delta_ci(
        greedy_scores, evidence_scores, args.bootstrap_samples, args.seed
    )
    evidence_rows = loaded["certificate_evidence"][1]
    budget_rows = loaded["certificate_budget"][1]
    greedy_rows = loaded["greedy"][1]
    changed = [
        sample_id
        for sample_id in reference_order
        if evidence_rows[sample_id]["answer"] != greedy_rows[sample_id]["answer"]
    ]
    gates = [evidence_rows[sample_id].get("evidence_certificate", {}) for sample_id in reference_order]
    fallback_reasons = Counter(
        str(gate.get("fallback_reason")) for gate in gates if not gate.get("gate_passed")
    )
    parse_error_views = sum(
        bool(view.get("verdict", {}).get("parse_error"))
        for gate in gates
        for view in gate.get("views", [])
    )
    budget_total = (
        aggregate["certificate_budget"]["prompt_tokens"]
        + aggregate["certificate_budget"]["completion_tokens"]
    )
    evidence_total = (
        aggregate["certificate_evidence"]["prompt_tokens"]
        + aggregate["certificate_evidence"]["completion_tokens"]
    )
    result = {
        "dataset": args.dataset,
        "n": len(reference_order),
        "runs": aggregate,
        "evidence_minus_greedy": {
            "delta": delta,
            "delta_ci95": [low, high],
            "paired_randomization_p": paired_randomization_p_value(
                greedy_scores, evidence_scores, args.bootstrap_samples, args.seed
            ),
            "candidate_better": sum(
                right > left for left, right in zip(greedy_scores, evidence_scores)
            ),
            "greedy_better": sum(
                left > right for left, right in zip(greedy_scores, evidence_scores)
            ),
            "ties": sum(
                abs(left - right) < 1e-12
                for left, right in zip(greedy_scores, evidence_scores)
            ),
            "answer_changed": len(changed),
            "changed_sample_ids": changed,
        },
        "certificate_gate": {
            "real_stable": sum(bool(gate.get("real_stable")) for gate in gates),
            "counterfactual_drop": sum(
                bool(gate.get("counterfactual_drop")) for gate in gates
            ),
            "gate_passed": sum(bool(gate.get("gate_passed")) for gate in gates),
            "changed_from_greedy": sum(
                bool(gate.get("changed_from_greedy")) for gate in gates
            ),
            "parse_error_views": parse_error_views,
            "fallback_reasons": dict(fallback_reasons),
        },
        "budget_audit": {
            "budget_answer_differs_from_standalone_greedy": sum(
                budget_rows[sample_id]["answer"] != greedy_rows[sample_id]["answer"]
                for sample_id in reference_order
            ),
            "budget_evidence_call_trace_mismatches": sum(
                _call_trace(budget_rows[sample_id]) != _call_trace(evidence_rows[sample_id])
                for sample_id in reference_order
            ),
            "budget_evidence_total_token_relative_difference": (
                abs(budget_total - evidence_total) / max(1, evidence_total)
            ),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
