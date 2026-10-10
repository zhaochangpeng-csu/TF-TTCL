#!/usr/bin/env python3
"""Evaluate inference-only JSONL against a separate label-bearing dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.multimodal.datasets import load_local_dataset
from evaluate.multimodal.metrics import bootstrap_mean_ci, score_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    targets = {
        item.target.sample_id: item.target
        for item in load_local_dataset(args.labels, args.dataset, args.image_root)
        if item.target is not None
    }
    rows = []
    with Path(args.predictions).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    scores = [
        score_prediction(
            args.dataset,
            row["answer"],
            targets[row["sample_id"]].answers,
            targets[row["sample_id"]].metadata,
        )
        for row in rows
        if row["sample_id"] in targets
    ]
    mean, low, high = bootstrap_mean_ci(scores, args.bootstrap_samples)
    print(json.dumps({"dataset": args.dataset, "n": len(scores), "score": mean, "ci95": [low, high]}, indent=2))


if __name__ == "__main__":
    main()
