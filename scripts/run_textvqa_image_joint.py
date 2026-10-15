#!/usr/bin/env python3
"""Run image-only and joint TextVQA arms against a frozen, label-free baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.client import OpenAIVLMClient
from core.multimodal.replay import ResponseTraceBackend
from core.multimodal.textvqa_image_runner import TextVQAImageRunner, VERSION
from evaluate.multimodal.compare_textvqa_text import audit as audit_baseline
from scripts.create_label_isolated_subset import sha256_file
from scripts.run_textvqa_text_only import checked_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=(20, 5000), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-url", default="http://127.0.0.1:9000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--revision", default="cc594898137f460bfe9f0759e9844b3ce807cfb5")
    args = parser.parse_args()
    samples = checked_samples(args.inference, args.subset_manifest, args.count)
    base_manifest, all_base_rows = audit_baseline(args.baseline, args.inference, 5000)
    base_rows = all_base_rows[:args.count]
    source_sha = sha256_file(args.inference)
    baseline_sha = sha256_file(args.baseline / "predictions.jsonl")
    namespace = {"dataset": "textvqa", "source_sha256": source_sha,
                 "ordered_ids_sha256": hashlib.sha256("\n".join(s.sample_id for s in samples).encode()).hexdigest(),
                 "seed": args.seed, "model_revision": args.revision,
                 "baseline_predictions_sha256": baseline_sha, "protocol": VERSION}
    base = OpenAIVLMClient(args.base_url, args.model, os.getenv("OPENAI_API_KEY", "EMPTY"))
    backend = ResponseTraceBackend(base, args.output / "response_trace", namespace, "record")
    TextVQAImageRunner(backend, samples, args.output, source_sha, args.revision,
                       base_manifest, base_rows, baseline_sha, args.seed).run()
    print(json.dumps({"status": "completed", "count": len(samples), "labels_read": False,
                      "trace_stats": backend.trace_stats, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
