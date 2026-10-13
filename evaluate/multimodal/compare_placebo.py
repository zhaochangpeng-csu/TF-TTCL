#!/usr/bin/env python3
"""Audit placebo replay before opening labels; compare paired source arms."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.multimodal.datasets import load_local_dataset
from core.multimodal.transfer_placebo import VERSION, replace_rule_block
from core.multimodal.transfer_runner import digest
from evaluate.multimodal.compare_transfer import audit_four, cost_summary
from evaluate.multimodal.metrics import paired_bootstrap_delta_ci, paired_randomization_p_value, score_prediction
from scripts.create_label_isolated_subset import sha256_file


def audit_placebo(source: Path, run: Path, expected_n: int,
                  source_expected_n: int = 400) -> tuple[dict, list[dict], list[dict]]:
    source_manifest, source_rows = audit_four(source, source_expected_n)
    manifest = json.loads((run / "manifest.json").read_text())
    rows = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
    if (manifest.get("status") != "completed" or manifest.get("version") != VERSION
            or manifest.get("processed") != expected_n or len(rows) != expected_n
            or manifest.get("source_expected_n") != source_expected_n
            or manifest.get("source_input_sha256") != source_manifest["input_sha256"]
            or manifest.get("source_run_sha") != source_manifest["git_sha"]
            or manifest.get("source_manifest_sha256") != sha256_file(source / "manifest.json")
            or manifest.get("labels_read_during_inference") is not False
            or manifest.get("ordered_ids") != source_manifest["ordered_ids"][:expected_n]):
        raise ValueError("Incomplete or mismatched placebo run; labels remain unopened")
    for index, (row, original) in enumerate(zip(rows, source_rows)):
        if (row.get("sample_index") != index or row.get("sample_id") != original["sample_id"]
                or row.get("source_row_sha256") != digest(original)
                or row.get("retrieved_source_ids") != [e["sample_id"] for e in original["arms"]["noisy"]["retrieved"]]
                or row.get("input_user_text") != replace_rule_block(original)
                or not row.get("response") or not row.get("answer")):
            raise ValueError("Placebo record differs from the audited source prompt or order")
        if row["reused_identical_source_response"] != (not original["arms"]["noisy"]["retrieved"]):
            raise ValueError("Placebo response reuse flag mismatched source retrieval")
        if row["reused_identical_source_response"] and row["response"] != original["arms"]["noisy"]["response"]:
            raise ValueError("Empty-history source response was not reused exactly")
        if any(e["sample_index"] >= index for e in original["arms"]["noisy"]["retrieved"]):
            raise ValueError("Current/future source memory in placebo")
    return manifest, rows, source_rows[:expected_n]


def compare(source: Path, run: Path, labels: Path, expected_n: int = 400) -> dict:
    if expected_n != 400:
        raise ValueError("Scoring is restricted to the completed 400-item replay")
    manifest, rows, source_rows = audit_placebo(source, run, expected_n)
    # This is the first point at which labels are opened. The source stream
    # had already been scored in the preceding pilot, so this is exploratory.
    targets = load_local_dataset(str(labels), "scienceqa_img", str(labels.parent))
    if len(targets) != expected_n or [e.sample.sample_id for e in targets] != manifest["ordered_ids"]:
        raise ValueError("Offline target order differs from completed placebo replay")
    if any(e.target is None for e in targets):
        raise ValueError("Missing offline target")
    predictions = {
        "none": [r["arms"]["none"]["answer"] for r in source_rows],
        "noisy": [r["arms"]["noisy"]["answer"] for r in source_rows],
        "random": [r["arms"]["random"]["answer"] for r in source_rows],
        "placebo": [r["answer"] for r in rows],
    }
    scores = {arm: [score_prediction("scienceqa_img", answer, e.target.answers, e.target.metadata)
                    for answer, e in zip(answers, targets)] for arm, answers in predictions.items()}
    results = {}
    for arm, values in scores.items():
        reference = scores["none"]
        changes = Counter((int(a), int(b)) for a, b in zip(reference, values))
        results[arm] = {"accuracy": sum(values) / expected_n,
                        "corrected_vs_none": changes[(0, 1)], "harmed_vs_none": changes[(1, 0)],
                        "paired_delta_ci_vs_none": paired_bootstrap_delta_ci(reference, values),
                        "paired_randomization_p_vs_none": paired_randomization_p_value(reference, values)}
    for other in ("noisy", "random"):
        results["placebo"][f"paired_delta_ci_vs_{other}"] = paired_bootstrap_delta_ci(scores[other], scores["placebo"])
        results["placebo"][f"paired_randomization_p_vs_{other}"] = paired_randomization_p_value(scores[other], scores["placebo"])
    source_cost = cost_summary(source_rows)
    placebo_final = {key: sum(row["response"][key] for row in rows)
                     for key in ("prompt_tokens", "completion_tokens", "latency_seconds")}
    return {"status": "scored_after_placebo_audit", "interpretation":
            "Post-hoc previously scored train stream; mechanistic diagnostic, not fresh confirmation.",
            "n": expected_n, "source_run_sha": manifest["source_run_sha"],
            "placebo_run_sha": manifest["git_sha"], "input_sha256": manifest["source_input_sha256"],
            "labels_sha256": sha256_file(labels), "results": results,
            "placebo_format_failures": sum(not row["format_ok"] for row in rows),
            "placebo_reused_identical_calls": sum(row["reused_identical_source_response"] for row in rows),
            "cost": {"source_shared_extraction": source_cost["shared_extraction"],
                     "source_arm_refinement": source_cost["arm_refinement"],
                     "placebo_refinement": placebo_final,
                     "placebo_logical": {key: source_cost["shared_extraction"][key] + placebo_final[key]
                                         for key in placebo_final}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite placebo comparison")
    result = compare(args.source, args.run, args.labels, args.expected_n)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
