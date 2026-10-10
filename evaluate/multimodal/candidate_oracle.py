#!/usr/bin/env python3
"""Measure candidate recall separately from evidence-selection quality."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from core.multimodal.datasets import load_local_dataset
from core.multimodal.schema import EvaluationTarget
from evaluate.multimodal.metrics import score_prediction


def summarize_candidate_oracle(
    rows: Sequence[Mapping[str, Any]],
    targets: Mapping[str, EvaluationTarget],
    dataset: str,
) -> dict[str, Any]:
    selected_scores: list[float] = []
    oracle_scores: list[float] = []
    unique_counts: list[int] = []
    regrets: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id not in targets:
            raise ValueError(f"Missing evaluation target for sample_id {sample_id!r}")
        target = targets[sample_id]
        selected_text = str(row["answer"])
        selected_score = score_prediction(
            dataset, selected_text, target.answers, target.metadata
        )
        candidates = [str(item["answer"]) for item in row.get("candidate_answers", [])]
        if not candidates:
            candidates = [str(row["answer"])]
        scores = [
            score_prediction(dataset, answer, target.answers, target.metadata)
            for answer in candidates
        ]
        oracle_score = max(scores)
        selected_scores.append(selected_score)
        oracle_scores.append(oracle_score)
        unique_counts.append(len(set(candidates)))
        if oracle_score > selected_score:
            regrets.append(
                {
                    "sample_id": sample_id,
                    "selected_answer": row["answer"],
                    "candidate_answers": candidates,
                    "correct_candidate_indices": [
                        index for index, score in enumerate(scores) if score == oracle_score
                    ],
                }
            )

    n = len(rows)
    return {
        "dataset": dataset,
        "n": n,
        "selected_score": sum(selected_scores) / n if n else 0.0,
        "candidate_oracle_score": sum(oracle_scores) / n if n else 0.0,
        "oracle_gap": (
            sum(right - left for left, right in zip(selected_scores, oracle_scores)) / n
            if n
            else 0.0
        ),
        "selection_regrets": len(regrets),
        "all_candidates_same": sum(count == 1 for count in unique_counts),
        "mean_unique_answers": sum(unique_counts) / n if n else 0.0,
        "regret_details": regrets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image-root")
    args = parser.parse_args()

    targets = {
        item.target.sample_id: item.target
        for item in load_local_dataset(args.labels, args.dataset, args.image_root)
        if item.target is not None
    }
    with Path(args.predictions).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    print(
        json.dumps(
            summarize_candidate_oracle(rows, targets, args.dataset),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
