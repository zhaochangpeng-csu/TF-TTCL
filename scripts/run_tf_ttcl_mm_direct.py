#!/usr/bin/env python3
"""Run paired ScienceQA/TextVQA direct-transfer arms without opening labels."""
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
from core.multimodal.tf_ttcl_direct_runner import DirectRunner, VERSION
from core.multimodal.tf_ttcl_text_embedding import Qwen3TextEmbedder
from scripts.create_label_isolated_subset import sha256_file
from scripts.inspect_baseline_io import read_rows, require_label_free


def checked_samples(inference: Path, manifest_path: Path, count: int,
                    dataset: str = "scienceqa_img"):
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    if dataset == "scienceqa_img":
        if (meta.get("seed"), meta.get("offset"), meta.get("count")) != (20261019, 0, 400):
            raise ValueError("Not the predeclared unused ScienceQA train stream")
        if meta.get("excluded_id_count", 0) < 2500:
            raise ValueError("Previously used ScienceQA train IDs were not excluded")
    elif dataset == "textvqa":
        if (meta.get("dataset_id"), meta.get("split"), meta.get("seed"),
                meta.get("offset"), meta.get("count")) != (
                "official/TextVQA_0.5.1", "train", 20261020, 0, 400):
            raise ValueError("Not the predeclared fresh TextVQA train stream")
        if meta.get("inference_sha256") != sha256_file(inference):
            raise ValueError("TextVQA train inference hash mismatch")
    else:
        raise ValueError("Unsupported direct-transfer dataset")
    if Path(meta["inference_path"]).resolve() != inference.resolve():
        raise ValueError("Inference/manifest path mismatch")
    rows = read_rows(inference)
    if len(rows) != 400 or [str(row["id"]) for row in rows] != meta["selected_ids"]:
        raise ValueError("Inference stream differs from the frozen 400 IDs")
    for row in rows:
        require_label_free(row)
    if count not in (20, 400):
        raise ValueError("Only smoke20 and dev400 are supported")
    examples = load_local_dataset(str(inference), dataset, str(inference.parent), limit=count)
    if any(example.target is not None for example in examples):
        raise ValueError("Inference adapter exposed labels")
    samples = [example.sample for example in examples]
    if [sample.sample_id for sample in samples] != meta["selected_ids"][:count]:
        raise ValueError("Dataset adapter changed ordered IDs")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=(20, 400), required=True)
    parser.add_argument("--dataset", choices=("scienceqa_img", "textvqa"),
                        default="scienceqa_img")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-url", default="http://127.0.0.1:9000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--revision", default="cc594898137f460bfe9f0759e9844b3ce807cfb5")
    parser.add_argument("--embedding-path", default="/home/lthpc/private/models/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-revision")
    args = parser.parse_args()
    samples = checked_samples(args.inference, args.subset_manifest, args.count, args.dataset)
    for sample in samples:
        sample.validate_images()
    source_hash = sha256_file(args.inference)
    namespace = {"dataset": args.dataset, "source_sha256": source_hash,
                 "ordered_ids_sha256": hashlib.sha256("\n".join(
                     sample.sample_id for sample in samples).encode()).hexdigest(),
                 "seed": args.seed, "model_revision": args.revision,
                 "protocol": VERSION, "trace_version": "four_arm_shared_requests_v1"}
    weights = Path(args.embedding_path) / "model.safetensors"
    embedding_revision = args.embedding_revision or sha256_file(weights)
    embedder = Qwen3TextEmbedder(args.embedding_path)
    base = OpenAIVLMClient(args.base_url, args.model, os.getenv("OPENAI_API_KEY", "EMPTY"))
    available = [item.id for item in base.client.models.list().data]
    if args.model not in available:
        raise ValueError(f"Wrong served model: {available}")
    backend = ResponseTraceBackend(base, args.output / "response_trace", namespace, "record")
    runner = DirectRunner(backend, embedder, samples, args.output, source_hash,
                          args.revision, embedding_revision, args.seed)
    runner.run()
    print(json.dumps({"status": "completed", "count": len(samples),
                      "labels_read": False, "trace_stats": backend.trace_stats,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
