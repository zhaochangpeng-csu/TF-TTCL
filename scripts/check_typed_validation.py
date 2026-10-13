#!/usr/bin/env python3
"""Validate an isolated regression stream without scoring/reading its label file."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.datasets import load_local_dataset
from scripts.select_typed_smoke import label_free
from scripts.create_label_isolated_subset import sha256_file


def check_stream(manifest_path: Path, expected_n: int = 2097) -> dict:
    m = json.loads(manifest_path.read_text())
    if (m.get("count") != expected_n or m.get("offset") != 0
            or m.get("seed") != 20261013 or m.get("selection") != "sha256(seed:id) rank slice"):
        raise ValueError("Unexpected validation permutation/count")
    # This source-hash check is a preparation check, before model inference.
    if sha256_file(Path(m["source"])) != m["source_sha256"]:
        raise ValueError("Validation source changed")
    inference = Path(m["inference_path"])
    if inference.resolve() == Path(m["labels_path"]).resolve():
        raise ValueError("Inference and labels must be different files")
    rows = [json.loads(line) for line in inference.read_text().splitlines() if line.strip()]
    if not all(label_free(row) for row in rows):
        raise ValueError("Forbidden label field in inference stream")
    ids = [str(row["id"]) for row in rows]
    if ids != m["selected_ids"] or len(ids) != expected_n or len(set(ids)) != len(ids):
        raise ValueError("Incomplete/mismatched ordered stream")
    if hashlib.sha256("\n".join(ids).encode()).hexdigest() != m["selected_ids_sha256"]:
        raise ValueError("ID checksum mismatch")
    examples = load_local_dataset(str(inference), "scienceqa_img", str(inference.parent))
    if any(example.target is not None for example in examples):
        raise ValueError("Inference adapter constructed an evaluation target")
    for example in examples:
        example.sample.validate_images()
    return {"n": expected_n, "source_sha256": m["source_sha256"],
            "ordered_ids_sha256": m["selected_ids_sha256"], "labels_read": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-n", type=int, default=2097)
    args = parser.parse_args()
    print(json.dumps(check_stream(Path(args.manifest), args.expected_n), indent=2))
