#!/usr/bin/env python3
"""Finite, unscored format smoke on a new label-free train stream.

This is NOT a memory comparison or an accuracy experiment. It tests whether
the frozen model can expose visible observations separately from inference,
instead of distilling invented rationales from answer-only baseline outputs.
No changes to the frozen validation prompts/operators are imported or made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.answers import normalize_answer
from core.multimodal.client import OpenAIVLMClient
from core.multimodal.datasets import load_local_dataset
from core.multimodal.prompts import format_question
from core.multimodal.replay import ResponseTraceBackend
from core.multimodal.schema import SamplingConfig
from scripts.create_label_isolated_subset import sha256_file
from scripts.inspect_baseline_io import read_rows, require_label_free


TRAJECTORY_SYSTEM = """Answer the visual question using the supplied image,
context and options. Give a brief, inspectable rationale; do not invent a
visible detail just to justify a conclusion. Separate directly visible facts
from knowledge or inference. If the question does not require the image,
explicitly say so instead of pretending the image proves it. Use exactly these
three labeled sections, keeping the rationale concise:
Observed: <visible task-relevant facts, or not required / insufficient>
Reasoning: <how those facts and any text or knowledge lead to the answer>
Final answer: <one current option letter only>
These observations and reasoning are proposals, not certified facts."""


def format_ok(response, choices):
    text = response.content
    final = re.search(r"\nFinal answer:[ \t]*([A-Za-z])[ \t]*\s*\Z", text)
    observed = re.search(r"(?ms)^Observed:\s*(.+?)\nReasoning:", text)
    reasoning = re.search(r"(?ms)^Reasoning:\s*(.+?)\nFinal answer:", text)
    return bool(response.finish_reason == "stop" and text.startswith("Observed:")
                and len(re.findall(r"(?m)^Final answer:", text)) == 1
                and final and observed and reasoning
                and observed.group(1).strip() and reasoning.group(1).strip()
                and 0 <= ord(final.group(1).upper()) - ord("A") < len(choices))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:9000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--revision", default="cc594898137f460bfe9f0759e9844b3ce807cfb5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.count <= 20:
        raise ValueError("This format smoke is capped at 20 unscored samples")
    for row in read_rows(args.inference):
        require_label_free(row)
    examples = load_local_dataset(str(args.inference), "scienceqa_img", str(args.inference.parent), limit=args.count)
    if len(examples) != args.count or any(e.target is not None for e in examples):
        raise ValueError("Incomplete or non-isolated input")
    if args.output.exists():
        raise FileExistsError("New output directory required; do not overwrite a smoke")
    args.output.mkdir(parents=True)
    namespace = {"dataset": "scienceqa_img", "source_sha256": sha256_file(args.inference),
                 "seed": args.seed, "model_revision": args.revision, "purpose": "transfer_format_smoke_v1"}
    backend = ResponseTraceBackend(OpenAIVLMClient(args.base_url, args.model, os.getenv("OPENAI_API_KEY", "EMPTY")),
                                   args.output / "response_trace", namespace, "record")
    ids = [e.sample.sample_id for e in examples]
    manifest = {"status": "running", "labels_read": False, "purpose": "format_only_not_accuracy_not_memory",
                "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_worktree_status": subprocess.check_output(["git", "status", "--porcelain"], text=True).strip(),
                "script_sha256": sha256_file(Path(__file__)), "model": args.model,
                "model_revision": args.revision, "seed": args.seed, "input_sha256": namespace["source_sha256"],
                "ordered_ids": ids, "prompt_sha256": hashlib.sha256(TRAJECTORY_SYSTEM.encode()).hexdigest(),
                "max_tokens": 384, "expected_calls": args.count * 2, "memory_enabled": False}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    try:
        with (args.output / "predictions.jsonl").open("x") as handle:
            for index, example in enumerate(examples):
                sample = example.sample
                sample.validate_images()
                for offset, temperature in enumerate((0.0, 0.7)):
                    sampling = SamplingConfig(temperature, 0.9 if temperature else None, 384, args.seed + index * 100 + offset)
                    text = format_question(sample.question, sample.choices, sample.metadata)
                    response = backend.generate(TRAJECTORY_SYSTEM, text, sample.image_paths, sampling)
                    row = {"sample_id": sample.sample_id, "sample_index": index, "sampling": asdict(sampling),
                           "input": sample.to_public_dict(), "user_text": text, "raw_response": asdict(response),
                           "normalized_answer": normalize_answer(response.content, sample.dataset, {**sample.metadata, "choices": list(sample.choices)}),
                           "format_ok": format_ok(response, sample.choices)}
                    results.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                print(f"[{index + 1}/{args.count}] sample={sample.sample_id} format={[r['format_ok'] for r in results[-2:]]}", flush=True)
    except Exception as exc:
        manifest.update(status="failed", completed_calls=len(results), error=str(exc))
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        raise
    manifest.update(status="completed", completed_calls=len(results),
                    format_passed=all(r["format_ok"] for r in results),
                    format_failures=[(r["sample_id"], r["sampling"]["temperature"]) for r in results if not r["format_ok"]],
                    prompt_tokens=sum(r["raw_response"]["prompt_tokens"] for r in results),
                    completion_tokens=sum(r["raw_response"]["completion_tokens"] for r in results),
                    latency_seconds=sum(r["raw_response"]["latency_seconds"] for r in results))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if not manifest["format_passed"]:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
