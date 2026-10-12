#!/usr/bin/env python3
"""Select supported calibration items using only the frozen lexical router."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.datasets import _LABEL_KEYS
from core.multimodal.typed_evidence import compile_program


def label_free(value):
    if isinstance(value, dict):
        return not (set(value) & _LABEL_KEYS) and all(label_free(v) for v in value.values())
    if isinstance(value, list):
        return all(label_free(v) for v in value)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.inference).read_text().splitlines() if line.strip()]
    if not all(label_free(row) for row in rows):
        raise ValueError("Smoke source contains forbidden label fields")
    selected = [row for row in rows if compile_program(row["question"], row.get("choices", []))]
    with Path(args.output).open("x") as handle:
        for row in selected:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({"n": len(selected), "ids": [row["id"] for row in selected]}))
