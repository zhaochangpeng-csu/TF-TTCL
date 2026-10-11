#!/usr/bin/env python3
"""Create a deterministic development subset with separate inference/label files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


LABEL_KEYS = {
    "answer",
    "answers",
    "label",
    "labels",
    "output",
    "gold",
    "annotation",
    "solution",
    "lecture",
    "rationale",
    "explanation",
    "response",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_subset(
    source_path: Path,
    inference_path: Path,
    labels_path: Path,
    manifest_path: Path,
    count: int,
    seed: int,
) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not 0 < count <= len(rows):
        raise ValueError(f"count must be between 1 and {len(rows)}")
    ids = [str(row.get("id", "")) for row in rows]
    if any(not sample_id for sample_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Source rows must have unique non-empty IDs")
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{row['id']}".encode("utf-8")
        ).hexdigest(),
    )[:count]

    for path in (inference_path, labels_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing subset artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    with inference_path.open("w", encoding="utf-8") as inference_handle, labels_path.open(
        "w", encoding="utf-8"
    ) as labels_handle:
        for row in ranked:
            public = {key: value for key, value in row.items() if key not in LABEL_KEYS}
            inference_handle.write(json.dumps(public, ensure_ascii=False) + "\n")
            labels_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    selected_ids = [str(row["id"]) for row in ranked]
    manifest = {
        "source": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "selection": "lowest_sha256(seed:id)",
        "seed": seed,
        "count": count,
        "selected_ids": selected_ids,
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(selected_ids).encode("utf-8")
        ).hexdigest(),
        "inference_path": str(inference_path.resolve()),
        "labels_path": str(labels_path.resolve()),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--inference-output", required=True)
    parser.add_argument("--labels-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest = create_subset(
        Path(args.source),
        Path(args.inference_output),
        Path(args.labels_output),
        Path(args.manifest),
        args.count,
        args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
