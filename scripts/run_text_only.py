#!/usr/bin/env python3
"""Run the frozen, label-blind ScienceQA train text-only paired comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.client import OpenAIVLMClient
from core.multimodal.datasets import load_local_dataset
from core.multimodal.replay import ResponseTraceBackend
from core.multimodal.text_only_runner import TextOnlyRunner, VERSION
from scripts.create_label_isolated_subset import sha256_file
from scripts.inspect_baseline_io import read_rows, require_label_free


def checked_samples(inference: Path, subset_manifest: Path, count: int):
    meta = json.loads(subset_manifest.read_text())
    if (meta.get("seed"), meta.get("offset"), meta.get("count")) != (20261011, 2100, 400):
        raise ValueError("Not the predeclared disjoint text-only train slice")
    if Path(meta["inference_path"]).resolve() != inference.resolve():
        raise ValueError("Subset/inference path mismatch")
    rows = read_rows(inference)
    if [str(row["id"]) for row in rows] != meta["selected_ids"]:
        raise ValueError("Ordered IDs changed")
    for row in rows:
        require_label_free(row)
    if not 1 <= count <= 400:
        raise ValueError("Count must be 1..400")
    examples = load_local_dataset(str(inference), "scienceqa_img", str(inference.parent), limit=count)
    if len(examples) != count or any(item.target is not None for item in examples):
        raise ValueError("Inference adapter exposed a target")
    samples = [item.sample for item in examples]
    if [sample.sample_id for sample in samples] != meta["selected_ids"][:count]:
        raise ValueError("Adapter changed ordered IDs")
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=(20, 400), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-url", default="http://127.0.0.1:9000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--revision", default="cc594898137f460bfe9f0759e9844b3ce807cfb5")
    args = parser.parse_args()
    samples = checked_samples(args.inference, args.subset_manifest, args.count)
    source_sha = sha256_file(args.inference)
    namespace = {"dataset": "scienceqa_img", "source_sha256": source_sha,
                 "ordered_ids_sha256": hashlib.sha256("\n".join(s.sample_id for s in samples).encode()).hexdigest(),
                 "seed": args.seed, "model_revision": args.revision, "protocol": VERSION}
    base = OpenAIVLMClient(args.base_url, args.model, os.getenv("OPENAI_API_KEY", "EMPTY"))
    backend = ResponseTraceBackend(base, args.output / "response_trace", namespace, "record")
    TextOnlyRunner(backend, samples, args.output, source_sha, args.revision, args.seed).run()
    print(json.dumps({"status": "completed", "count": len(samples), "labels_read": False,
                      "trace_stats": backend.trace_stats, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
