#!/usr/bin/env python3
"""Label-blind, one-request image logprob capability probe for direct TF-TTCL."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.client import OpenAIVLMClient
from core.multimodal.datasets import load_local_dataset
from core.multimodal.image_ops import sha256_file
from core.multimodal.prompts import format_question
from core.multimodal.schema import MultimodalSample, SamplingConfig
from core.multimodal.tf_ttcl_direct_selection import LogprobUnavailable, strict_generation_ppl
from scripts.inspect_baseline_io import require_label_free


PROBE_SYSTEM = "Answer the visual multiple-choice question with brief reasoning and one final option letter."


def probe(backend, sample: MultimodalSample, seed: int = 0) -> dict:
    sample.validate_images()
    result = {"sample_id": sample.sample_id,
              "image_hashes": [sha256_file(path) for path in sample.image_paths],
              "probes": [], "status": "supported"}
    user = format_question(sample.question, sample.choices, sample.metadata)
    for role, temperature, offset in (("teacher", 0.0, 0), ("student", 0.7, 1)):
        response = backend.generate(PROBE_SYSTEM, user, sample.image_paths,
                                    SamplingConfig(temperature=temperature,
                                                   top_p=0.9 if temperature else None,
                                                   max_tokens=128, seed=seed + offset,
                                                   request_logprobs=True))
        entry = {"role": role, "response": asdict(response),
                 "logprob_count": len(response.logprobs),
                 "completion_token_count": response.completion_tokens,
                 "all_logprobs_finite": all(math.isfinite(x) for x in response.logprobs)}
        try:
            entry["ppl"] = strict_generation_ppl(response)
        except LogprobUnavailable as error:
            entry["reason"] = str(error)
            result["status"] = "blocked"
        result["probes"].append(entry)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:9000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--revision", default="cc594898137f460bfe9f0759e9844b3ce807cfb5")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing capability probe")
    first = json.loads(args.inference.read_text(encoding="utf-8").splitlines()[0])
    require_label_free(first)
    example = load_local_dataset(str(args.inference), "scienceqa_img", str(args.image_root), limit=1)[0]
    if example.target is not None or not example.sample.image_paths:
        raise ValueError("Probe requires one label-free visual sample")
    backend = OpenAIVLMClient(args.base_url, args.model, os.getenv("OPENAI_API_KEY", "EMPTY"))
    available = [model.id for model in backend.client.models.list().data]
    if args.model not in available:
        raise ValueError(f"Target model not served: {args.model}; available={available}")
    result = probe(backend, example.sample)
    result.update({"model": args.model, "model_revision": args.revision,
                   "input_sha256": sha256_file(str(args.inference)),
                   "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "labels_read": False})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    public = {**result, "probes": [{key: value for key, value in entry.items()
                                   if key != "response"} for entry in result["probes"]]}
    print(json.dumps(public, ensure_ascii=False, indent=2))
    if result["status"] != "supported":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
