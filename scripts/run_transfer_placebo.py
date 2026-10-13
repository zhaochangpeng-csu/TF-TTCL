#!/usr/bin/env python3
"""Replay audited current candidates with a length-matched no-fact history."""
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
from core.multimodal.transfer_placebo import PlaceboRunner, VERSION
from scripts.create_label_isolated_subset import sha256_file
from scripts.run_transfer_online import checked_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-url", default="http://127.0.0.1:9000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--revision", default="cc594898137f460bfe9f0759e9844b3ce807cfb5")
    args = parser.parse_args()
    samples = checked_samples(args.inference, args.subset_manifest, args.count)
    source_hash = sha256_file(args.inference)
    namespace = {"dataset": "scienceqa_img", "source_sha256": source_hash,
                 "ordered_ids_sha256": hashlib.sha256("\n".join(s.sample_id for s in samples).encode()).hexdigest(),
                 "seed": args.seed, "model_revision": args.revision,
                 "protocol": VERSION, "trace_version": "source_exact_current_candidates_placebo_final_v1"}
    base = OpenAIVLMClient(args.base_url, args.model, os.getenv("OPENAI_API_KEY", "EMPTY"))
    backend = ResponseTraceBackend(base, args.output / "response_trace", namespace, "record")
    runner = PlaceboRunner(backend, samples, args.source, args.output,
                           source_hash, args.revision, args.seed)
    runner.run()
    print(json.dumps({"status": "completed", "count": len(samples),
                      "labels_read_during_inference": False,
                      "source_labels_previously_scored": True,
                      "trace_stats": backend.trace_stats}, indent=2))


if __name__ == "__main__":
    main()
