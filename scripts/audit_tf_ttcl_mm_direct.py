#!/usr/bin/env python3
"""Label-free protocol audit and input/output casebook for direct transfer."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.schema import GenerationResponse
from core.multimodal.tf_ttcl_direct_runner import ARMS, VERSION, split_question_clause
from core.multimodal.tf_ttcl_direct_selection import ScoredCandidate, select_closed
from scripts.inspect_baseline_io import read_rows, require_label_free
from scripts.create_label_isolated_subset import sha256_file


def audit(output: Path, inference: Path, count: int) -> tuple[dict, list[dict]]:
    manifest = json.loads((output / "manifest.json").read_text())
    rows = read_rows(output / "predictions.jsonl")
    inputs = read_rows(inference)
    if (manifest.get("version"), manifest.get("status"), manifest.get("processed")) != (
            VERSION, "completed", count):
        raise ValueError("Run is not complete")
    if manifest.get("labels_read") is not False or manifest.get("input_sha256") != sha256_file(inference):
        raise ValueError("Label-free input manifest mismatch")
    if len(rows) != count or len(inputs) < count:
        raise ValueError("Wrong row count")
    reasoning_outputs = 0
    answering_outputs = 0
    for item in inputs:
        require_label_free(item)
    for index, (row, source) in enumerate(zip(rows, inputs)):
        if (row.get("sample_index"), row.get("sample_id"), row.get("labels_read")) != (
                index, str(source["id"]), False):
            raise ValueError(f"Row {index}: order or label boundary changed")
        if set(row.get("arms", {})) != set(ARMS) or len(row.get("variants", [])) != 4:
            raise ValueError(f"Row {index}: incomplete arms or variants")
        context_prefix, _ = split_question_clause(row["original_question"])
        if any(not variant.startswith(context_prefix) for variant in row["variants"]):
            raise ValueError(f"Row {index}: Tutor changed the protected context")
        if any(rule["sample_index"] >= index for bucket in row["retrieved_rules"].values()
               for rule in bucket):
            raise ValueError(f"Row {index}: current/future rule retrieved")
        if any(len(bucket) > 30 for bucket in row["retrieved_rules"].values()):
            raise ValueError(f"Row {index}: more than top-30 rules")
        if any(rule["sample_index"] != index for rule in row["written_rules"]):
            raise ValueError(f"Row {index}: write index mismatch")
        for name, arm in row["arms"].items():
            calls = arm["calls"]
            answering = [call for call in calls if call.get("role")]
            if len(answering) != (1 if name == "greedy" else 5):
                raise ValueError(f"Row {index} {name}: candidate count mismatch")
            for call in answering:
                answering_outputs += 1
                reasoning_outputs += bool(re.search(r"(?is)reasoning\s*:.{25,}?final answer\s*:",
                                                   call["response"]["content"]))
                expected_images = source.get("image_paths", source.get("image_path", source.get("image")))
                expected_images = expected_images if isinstance(expected_images, list) else [expected_images]
                if (call["image_hashes"] != row["image_hashes"]
                        or call["image_paths"] != expected_images
                        or not call["sampling"]["request_logprobs"]):
                    raise ValueError(f"Row {index} {name}: wrong image or missing PPL request")
                response = GenerationResponse(**call["response"])
                candidate = ScoredCandidate.from_response(call["role"], call["normalized_answer"], response)
                if not math.isclose(candidate.ppl, call["ppl"], rel_tol=1e-9):
                    raise ValueError(f"Row {index} {name}: PPL mismatch")
            for call in calls:
                if not call.get("role") and call["image_hashes"]:
                    raise ValueError(f"Row {index} {name}: Tutor/rule call saw image")
            if name != "greedy":
                candidates = [ScoredCandidate.from_response(call["role"], call["normalized_answer"],
                              GenerationResponse(**call["response"])) for call in answering]
                selected = select_closed(candidates)
                if (arm["positive_role"], arm["negative_role"], arm["reason"]) != (
                        selected.positive.role,
                        selected.negative.role if selected.negative else None, selected.reason):
                    raise ValueError(f"Row {index} {name}: selection mismatch")
            usage = arm["usage"]
            if usage["completion_tokens"] != sum(call["response"]["completion_tokens"] for call in calls):
                raise ValueError(f"Row {index} {name}: token accounting mismatch")
    if count == 20 and reasoning_outputs / max(answering_outputs, 1) < 0.8:
        raise ValueError("Smoke failed: fewer than 80% of answers contain a reasoning trajectory")
    return manifest, rows


def casebook(rows: list[dict]) -> str:
    lines = ["# TF-TTCL multimodal direct transfer: label-free I/O inspection", "",
             "No evaluation labels, accuracy, or correctness-based case selection were used.", ""]
    for row in rows:
        lines.extend([f"## {row['sample_index']}: {row['sample_id']}", "",
                      f"Question: {row['original_question']}", "",
                      f"Choices: {json.dumps(row['choices'], ensure_ascii=False)}", "",
                      f"Original image hashes: {json.dumps(row['image_hashes'])}", "",
                      f"Tutor variants: {json.dumps(row['variants'], ensure_ascii=False)}", ""])
        for name in ARMS:
            arm = row["arms"][name]
            lines.extend([f"### {name}", "",
                          f"Output: {arm['answer']} · reason: {arm.get('reason', 'greedy')}", "",
                          f"Tokens: {arm['usage']['prompt_tokens']} prompt + "
                          f"{arm['usage']['completion_tokens']} completion", ""])
            for call in arm["calls"]:
                role = call.get("role", call.get("kind", "tutor"))
                lines.extend([f"- {role}: question/input = {call['user'][:900]!r}; "
                              f"output = {call['response']['content'][:900]!r}; "
                              f"PPL = {call.get('ppl')}"])
            lines.append("")
        lines.extend(["Retrieved rules: " + json.dumps(row["retrieved_rules"], ensure_ascii=False),
                      "Written rules: " + json.dumps([r["text"] for r in row["written_rules"]],
                                                    ensure_ascii=False), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--casebook", type=Path)
    args = parser.parse_args()
    _, rows = audit(args.output, args.inference, args.count)
    if args.casebook:
        args.casebook.parent.mkdir(parents=True, exist_ok=True)
        if args.casebook.exists():
            raise FileExistsError(args.casebook)
        args.casebook.write_text(casebook(rows), encoding="utf-8")
    print(json.dumps({"audit": "passed", "samples": len(rows), "labels_read": False,
                      "rule_writes": sum(len(r["written_rules"]) for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
