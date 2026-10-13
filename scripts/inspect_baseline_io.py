#!/usr/bin/env python3
"""Read-only, label-free case cards for inspecting real baseline I/O.

Prediction `answer` fields are model outputs, not dataset targets. Input records
are checked recursively BEFORE the dataset adapter is called. This tool has no
label-file argument, never makes a model call, and never changes run artifacts.
Exact recorded requests are distinguished from reconstructed requests; a
reconstructed request is not evidence that the wire input was captured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.datasets import load_local_dataset
from core.multimodal.prompts import ANSWER_SYSTEM, format_question
from core.multimodal.typed_evidence import compile_program
from scripts.create_label_isolated_subset import LABEL_KEYS, sha256_file


def require_label_free(value):
    if isinstance(value, dict):
        forbidden = {str(key).lower() for key in value} & LABEL_KEYS
        if forbidden:
            raise ValueError(f"Forbidden label fields in input: {sorted(forbidden)}")
        for item in value.values():
            require_label_free(item)
    elif isinstance(value, list):
        for item in value:
            require_label_free(item)


def read_rows(path):
    # Only completed newline-terminated records belong to an in-flight snapshot.
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith(b"\n") and line.strip()]


def trace_lookup(directory, request):
    key = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    path = directory / f"{key}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("request") != request:
        raise ValueError(f"Trace request mismatch: {path}")
    return {"provenance": "recorded_exact", "trace_key": key, **payload}


def request_for(sample, record, manifest, candidate, source_sha):
    """Only the frozen four-arm regression's ANSWER_SYSTEM calls are supported."""
    method = manifest["method"]
    supported = {"zero_shot", "shared_budget", "shared_evidence", "efficient_budget"}
    if method not in supported:
        raise ValueError(f"No exact request reconstruction for method {method}")
    if candidate.get("memory_conditioned") or record.get("memory_size_before", 0):
        raise ValueError("Memory-conditioned inputs need actual context, not reconstruction")
    config = manifest["config"]
    seed = int(manifest["seed"])
    base = int(record["sample_index"]) * 100
    candidates = record["candidate_answers"]
    offset = next(i for i, item in enumerate(candidates) if item is candidate)
    temperature = 0.0
    if method == "zero_shot":
        request_seed = seed + base
    else:
        request_seed = seed + base + 10 + offset
        if candidate["trajectory"] == "original_sampled" or method == "efficient_budget" and offset > 0:
            temperature = 0.7
    return {
        "namespace": {"dataset": sample.dataset, "source_sha256": source_sha,
                      "seed": seed, "model_revision": config["model"].get("revision", "unspecified")},
        "model": manifest["model"], "system_prompt": ANSWER_SYSTEM,
        "user_text": format_question(candidate["question_variant"], sample.choices, sample.metadata),
        "image_hashes": candidate["image_hashes"],
        "sampling": {"temperature": temperature, "top_p": 0.9 if temperature else None,
                     "max_tokens": config["model"].get("max_tokens", 256),
                     "seed": request_seed, "request_logprobs": config["model"].get("request_logprobs", False)},
    }


def build_report(inference, run_paths, count=8, sample_ids=(), include_diagnostics=False):
    if count < 0:
        raise ValueError("count must be nonnegative")
    raw_rows = read_rows(inference)
    for row in raw_rows:
        require_label_free(row)
    if not raw_rows:
        raise ValueError("Empty inference snapshot")
    ids = [str(row["id"]) for row in raw_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate inference IDs")
    source_sha = sha256_file(inference)
    runs = {}
    for name, directory in run_paths.items():
        manifest = json.loads((directory / "manifest.json").read_text())
        config = manifest["config"]
        if Path(config["experiment"]["data_path"]).resolve() != inference.resolve():
            raise ValueError(f"Run {name} uses another inference path")
        rows = read_rows(directory / "predictions.jsonl")
        expected = ids[:len(rows)]
        if [row["sample_id"] for row in rows] != expected:
            raise ValueError(f"Run {name} is not a prefix of this input stream")
        for index, row in enumerate(rows):
            if row.get("sample_index") != index or row.get("method") != manifest["method"]:
                raise ValueError(f"Run {name} has mismatched index/method")
        trace = config["model"].get("response_trace_dir")
        runs[name] = (manifest, {row["sample_id"]: row for row in rows}, Path(trace) if trace else None)
    datasets = {m["config"]["experiment"]["dataset"] for m, _, _ in runs.values()}
    if len(datasets) != 1:
        raise ValueError("Provide runs from exactly one dataset")
    examples = load_local_dataset(str(inference), datasets.pop(), str(inference.parent))
    if any(e.target is not None for e in examples):
        raise ValueError("Input adapter created a target")
    samples = {e.sample.sample_id: e.sample for e in examples}
    selected = list(dict.fromkeys([*ids[:count], *map(str, sample_ids)]))
    if set(selected) - set(ids):
        raise ValueError("Requested ID is absent from this stream")
    if include_diagnostics:
        seen = set()
        # One case per supported operator and first two candidate disagreements;
        # never use accuracy, labels, or corrected/regressed status to select.
        disagreements = 0
        for sid in ids:
            sample = samples[sid]
            program = compile_program(sample.question, sample.choices)
            if program and program["operator"] not in seen:
                seen.add(program["operator"])
                selected.append(sid)
            disagree = any(len({c["answer"] for c in rows[sid]["candidate_answers"]}) > 1
                           for _, rows, _ in runs.values() if sid in rows)
            if disagree and disagreements < 2:
                selected.append(sid)
                disagreements += 1
        selected = list(dict.fromkeys(selected))
    cases = []
    for sid in selected:
        sample = samples[sid]
        images = []
        for image in sample.image_paths:
            path = Path(image)
            info = {"path": image, "sha256": sha256_file(path)}
            try:
                from PIL import Image
                with Image.open(path) as obj:
                    info.update(width=obj.width, height=obj.height, mode=obj.mode)
            except ImportError:
                info["dimensions_unavailable"] = "Pillow not installed"
            images.append(info)
        case = {"sample_id": sid, "stream_index": ids.index(sid),
                "input": sample.to_public_dict(), "images": images,
                "typed_program": compile_program(sample.question, sample.choices), "runs": {}}
        for name, (manifest, rows, trace) in runs.items():
            if sid not in rows:
                case["runs"][name] = {"status": "not_in_snapshot"}
                continue
            row = rows[sid]
            calls = []
            for candidate in row["candidate_answers"]:
                request = request_for(sample, row, manifest, candidate, source_sha)
                captured = trace_lookup(trace, request) if trace else None
                calls.append({"trajectory": candidate["trajectory"], "normalized_answer": candidate["answer"],
                              "image_paths": candidate["image_paths"],
                              **(captured or {"provenance": "reconstructed_not_wire_capture",
                                              "request": request, "response": None})})
            certificate = row.get("shared_certificate", {})
            case["runs"][name] = {"status": "observed", "answer": row["answer"],
                                   "selected_raw_output": row["response"], "usage": row["usage_delta"],
                                   "selection": row["selection"], "candidate_calls": calls,
                                   "shared_certificate": certificate}
        cases.append(case)
    question_groups = Counter((s.question, tuple(s.choices)) for s in samples.values())
    stats = {"n": len(samples), "with_context": sum(s.question.startswith("Context:\n") for s in samples.values()),
             "choice_count": dict(Counter(str(len(s.choices)) for s in samples.values())),
             "subjects": dict(Counter(str(s.metadata.get("subject")) for s in samples.values())),
             "repeated_question_and_choices_groups": sum(n > 1 for n in question_groups.values()),
             "items_in_repeated_question_and_choices_groups": sum(n for n in question_groups.values() if n > 1),
             "typed_supported": sum(compile_program(s.question, s.choices) is not None for s in samples.values())}
    run_stats = {}
    for name, (manifest, rows, _) in runs.items():
        values = list(rows.values())
        run_stats[name] = {"manifest_status": manifest["status"], "snapshot_records": len(values),
                           "git_sha": manifest["git_sha"], "prompt_hash": manifest["prompt_hash"],
                           "answer_only_outputs": sum(bool(re.fullmatch(r"\s*Final answer:\s*[A-Za-z]\s*", r["response"])) for r in values),
                           "no_reasoning_outputs_including_bare_letters": sum(bool(re.fullmatch(r"\s*(?:Final answer:\s*)?[A-Za-z]\s*", r["response"])) for r in values),
                           "candidate_disagreements": sum(len({c["answer"] for c in r["candidate_answers"]}) > 1 for r in values),
                           "byte_identical_fidelity_candidates": sum(any(
                               c["trajectory"] == "image_variant" and c["image_hashes"] == r["candidate_answers"][0]["image_hashes"]
                               for c in r["candidate_answers"]) for r in values),
                           "prompt_tokens": sum(r["usage_delta"]["prompt_tokens"] for r in values),
                           "completion_tokens": sum(r["usage_delta"]["completion_tokens"] for r in values)}
    return {"labels_read": False, "inference_sha256": source_sha, "selection": {
        "first_n": count, "explicit_ids": list(sample_ids), "diagnostics": include_diagnostics,
        "uses_labels_or_correctness": False}, "dataset": stats, "runs": run_stats, "cases": cases}


def render_markdown(report):
    lines = ["# Label-free baseline input/output inspection", "",
             "No labels read; no accuracy-based case selection. Requests without immutable traces are reconstructed, not wire captures.",
             "", "## Dataset and run snapshot", "", "```json",
             json.dumps({k: report[k] for k in ("inference_sha256", "selection", "dataset", "runs")}, ensure_ascii=False, indent=2), "```"]
    for case in report["cases"]:
        lines += ["", f"## Sample {case['sample_id']} (stream index {case['stream_index']})", "",
                  "```json", json.dumps(case["input"], ensure_ascii=False, indent=2), "```", ""]
        for img in case["images"]:
            lines += [f"Image: {img['path']} · SHA256 {img['sha256']} · {img.get('width', '?')}×{img.get('height', '?')}", ""]
        lines += [f"Typed program: {json.dumps(case['typed_program'])}", ""]
        for name, data in case["runs"].items():
            lines += [f"### {name}", ""]
            if data["status"] != "observed":
                lines += ["Not present in this snapshot."]
                continue
            lines += ["Selected raw output:", "```text", data["selected_raw_output"], "```", "",
                      f"Candidate answers: {[(c['trajectory'], c['normalized_answer']) for c in data['candidate_calls']]}", "",
                      f"Usage: {json.dumps(data['usage'])}", ""]
            for call in data["candidate_calls"]:
                lines += [f"#### {call['trajectory']} — {call['provenance']}", "", "```json",
                          json.dumps(call, ensure_ascii=False, indent=2), "```", ""]
            if data["shared_certificate"]:
                lines += ["Evidence observations and deterministic decision:", "```json",
                          json.dumps(data["shared_certificate"], ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="name=directory")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--ids", default="")
    parser.add_argument("--include-diagnostics", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="New JSON path; Markdown beside it")
    args = parser.parse_args()
    runs = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        if name in runs:
            raise ValueError("Duplicate run name")
        runs[name] = Path(path)
    markdown = args.output.with_suffix(".md")
    if args.output.exists() or markdown.exists():
        raise FileExistsError("Refusing to overwrite an inspection snapshot")
    report = build_report(args.inference, runs, args.count, tuple(filter(None, args.ids.split(","))), args.include_diagnostics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    markdown.write_text(render_markdown(report))
    print(json.dumps({"output": str(args.output), "cases": len(report["cases"]),
                      "labels_read": False, "dataset": report["dataset"], "runs": report["runs"]}, indent=2))


if __name__ == "__main__":
    main()
