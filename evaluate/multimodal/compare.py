#!/usr/bin/env python3
"""Strict paired comparison for two label-isolated multimodal runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from core.multimodal.datasets import load_local_dataset
from evaluate.multimodal.metrics import (
    paired_bootstrap_delta_ci,
    paired_randomization_p_value,
    score_prediction,
)


def _load_predictions(path: str) -> tuple[list[str], dict[str, dict[str, Any]]]:
    order: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in rows:
                raise ValueError(f"Duplicate sample_id {sample_id!r} in {path}")
            order.append(sample_id)
            rows[sample_id] = row
    return order, rows


def _prediction_text(dataset: str, row: dict[str, Any]) -> str:
    return str(row["answer"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two runs on identical samples")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    baseline_order, baseline_rows = _load_predictions(args.baseline)
    candidate_order, candidate_rows = _load_predictions(args.candidate)
    if set(baseline_rows) != set(candidate_rows):
        missing_candidate = sorted(set(baseline_rows) - set(candidate_rows))
        missing_baseline = sorted(set(candidate_rows) - set(baseline_rows))
        raise ValueError(
            "Runs must contain identical sample IDs; "
            f"missing_from_candidate={missing_candidate[:10]}, "
            f"missing_from_baseline={missing_baseline[:10]}"
        )

    targets = {
        item.target.sample_id: item.target
        for item in load_local_dataset(args.labels, args.dataset, args.image_root)
        if item.target is not None
    }
    unknown = [sample_id for sample_id in baseline_order if sample_id not in targets]
    if unknown:
        raise ValueError(f"Missing evaluation targets for sample IDs: {unknown[:10]}")

    baseline_scores: list[float] = []
    candidate_scores: list[float] = []
    transitions: list[dict[str, Any]] = []
    answer_changes: list[dict[str, Any]] = []
    for sample_id in baseline_order:
        target = targets[sample_id]
        left = score_prediction(
            args.dataset,
            _prediction_text(args.dataset, baseline_rows[sample_id]),
            target.answers,
            target.metadata,
        )
        right = score_prediction(
            args.dataset,
            _prediction_text(args.dataset, candidate_rows[sample_id]),
            target.answers,
            target.metadata,
        )
        baseline_scores.append(left)
        candidate_scores.append(right)
        baseline_answer = str(baseline_rows[sample_id]["answer"])
        candidate_answer = str(candidate_rows[sample_id]["answer"])
        if baseline_answer != candidate_answer:
            answer_changes.append(
                {
                    "sample_id": sample_id,
                    "baseline_answer": baseline_answer,
                    "candidate_answer": candidate_answer,
                    "baseline_score": left,
                    "candidate_score": right,
                }
            )
        if not abs(right - left) < 1e-12:
            transitions.append(
                {
                    "sample_id": sample_id,
                    "baseline_score": left,
                    "candidate_score": right,
                    "baseline_answer": baseline_rows[sample_id]["answer"],
                    "candidate_answer": candidate_rows[sample_id]["answer"],
                }
            )

    delta, low, high = paired_bootstrap_delta_ci(
        baseline_scores, candidate_scores, args.bootstrap_samples, args.seed
    )
    result = {
        "dataset": args.dataset,
        "n": len(baseline_scores),
        "baseline_score": sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0,
        "candidate_score": sum(candidate_scores) / len(candidate_scores) if candidate_scores else 0.0,
        "delta": delta,
        "delta_ci95": [low, high],
        "candidate_better": sum(right > left for left, right in zip(baseline_scores, candidate_scores)),
        "baseline_better": sum(left > right for left, right in zip(baseline_scores, candidate_scores)),
        "ties": sum(abs(left - right) < 1e-12 for left, right in zip(baseline_scores, candidate_scores)),
        "answer_changed": len(answer_changes),
        "paired_randomization_p": paired_randomization_p_value(
            baseline_scores, candidate_scores, args.bootstrap_samples, args.seed
        ),
        "transitions": transitions,
        "answer_changes": answer_changes,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
