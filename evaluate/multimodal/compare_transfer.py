#!/usr/bin/env python3
"""Audit four-arm online transfer before opening the isolated label file."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.multimodal.datasets import load_local_dataset
from core.multimodal.transfer_runner import ARMS, VERSION
from evaluate.multimodal.metrics import paired_bootstrap_delta_ci, paired_randomization_p_value, score_prediction
from scripts.create_label_isolated_subset import sha256_file


def audit_four(path: Path, expected_n: int) -> tuple[dict, list[dict]]:
    """No label path exists in this API; failures stop before scoring."""
    manifest = json.loads((path / "manifest.json").read_text())
    rows = [json.loads(line) for line in (path / "predictions.jsonl").read_text().splitlines()]
    if (manifest.get("status") != "completed" or manifest.get("all_arms_complete") is not True
            or manifest.get("labels_read") is not False or manifest.get("version") != VERSION
            or manifest.get("processed") != expected_n or len(rows) != expected_n
            or manifest.get("ordered_ids") != [row.get("sample_id") for row in rows]
            or manifest.get("arm_order") != list(ARMS)):
        raise ValueError("Incomplete or mismatched four-arm run; labels remain unopened")
    past_ids = {arm: set() for arm in ARMS}
    for index, row in enumerate(rows):
        if row.get("sample_index") != index or set(row.get("arms", {})) != set(ARMS):
            raise ValueError("Missing arm or bad stream index")
        if len(row.get("candidate_attempts", [])) != 4 or not row.get("distillation"):
            raise ValueError("Missing shared candidate/distillation calls")
        for arm in ARMS:
            data = row["arms"][arm]
            if not data.get("response") or not data.get("answer"):
                raise ValueError("Missing final response")
            if data.get("memory_size_before") != len(past_ids[arm]):
                raise ValueError("Memory size differs from committed prefix")
            if any(e["sample_index"] >= index for e in data["retrieved"]):
                raise ValueError("Current/future retrieval contamination")
            if any(e["experience_id"] not in past_ids[arm] for e in data["retrieved"]):
                raise ValueError("Retrieved experience absent from own past bank")
            if data["written"] and (data["written"]["sample_index"] != index
                                    or data["written"]["sample_id"] != row["sample_id"]):
                raise ValueError("Bad written experience provenance")
        if row["arms"]["none"]["retrieved"] or row["arms"]["none"]["written"]:
            raise ValueError("No-memory arm leaked memory")
        if row["arms"]["noisy"]["written"] != row["arms"]["random"]["written"]:
            raise ValueError("Random arm write pool differs from noisy arm")
        if row["arms"]["gated_weighted"]["written"] and not row["evidence"]["gate_passed"]:
            raise ValueError("Ungated experience in gated bank")
        for arm in ARMS:
            written = row["arms"][arm]["written"]
            if written:
                past_ids[arm].add(written["experience_id"])
    return manifest, rows


def cost_summary(rows):
    common = {key: 0.0 for key in ("prompt_tokens", "completion_tokens", "latency_seconds")}
    arms = {arm: dict(common) for arm in ARMS}
    for row in rows:
        responses = [a["response"] for a in row["candidate_attempts"]]
        responses.append(row["distillation"]["response"])
        responses.extend(o["response"] for o in row["evidence"]["observations"])
        for response in responses:
            for key in common:
                common[key] += response[key]
        for arm in ARMS:
            for key in common:
                arms[arm][key] += row["arms"][arm]["response"][key]
    logical = {arm: {key: common[key] + arms[arm][key] for key in common} for arm in ARMS}
    actual = {key: common[key] + sum(arms[arm][key] for arm in ARMS) for key in common}
    return {"shared_extraction": common, "arm_refinement": arms,
            "logical_per_arm": logical, "actual_all_arms": actual,
            "shared_trace_calls_may_reduce_executed_cost": True}


def compare(path: Path, labels_path: Path, expected_n: int = 400) -> dict:
    manifest, rows = audit_four(path, expected_n)
    # Only this point may open labels. Guard ordered IDs and complete targets.
    targets = load_local_dataset(str(labels_path), "scienceqa_img", str(labels_path.parent))
    if [e.sample.sample_id for e in targets] != manifest["ordered_ids"] or len(targets) != expected_n:
        raise ValueError("Isolated labels do not match the frozen stream")
    if any(e.target is None for e in targets):
        raise ValueError("Missing offline target")
    scores = {arm: [score_prediction("scienceqa_img", row["arms"][arm]["answer"],
                                     target.target.answers, target.target.metadata)
                    for row, target in zip(rows, targets)] for arm in ARMS}
    results = {}
    baseline = scores["none"]
    for arm in ARMS:
        values = scores[arm]
        changes = Counter((int(a), int(b)) for a, b in zip(baseline, values))
        segment = [sum(values[lo:hi]) / max(1, hi - lo) for lo, hi in
                   ((0, expected_n // 3), (expected_n // 3, 2 * expected_n // 3),
                    (2 * expected_n // 3, expected_n))]
        results[arm] = {"accuracy": sum(values) / expected_n,
                        "corrected_vs_none": changes[(0, 1)],
                        "harmed_vs_none": changes[(1, 0)],
                        "front_middle_back_accuracy": segment,
                        "writes": sum(row["arms"][arm]["written"] is not None for row in rows),
                        "retrieval_coverage": sum(bool(row["arms"][arm]["retrieved"]) for row in rows),
                        "answer_changed_from_pseudo_positive": sum(row["arms"][arm]["changed_from_pseudo_positive"] for row in rows),
                        "format_failures": sum(not row["arms"][arm]["format_ok"] for row in rows),
                        "paired_delta_ci_vs_none": paired_bootstrap_delta_ci(baseline, values),
                        "paired_randomization_p_vs_none": paired_randomization_p_value(baseline, values)}
    costs = cost_summary(rows)
    common_reference = costs["logical_per_arm"]["none"]
    budget = {arm: {kind: (costs["logical_per_arm"][arm][kind] - common_reference[kind])
                          / max(1, common_reference[kind])
                    for kind in ("prompt_tokens", "completion_tokens")}
              for arm in ARMS}
    return {"status": "scored_after_four_arm_audit", "n": expected_n,
            "run_git_sha": manifest["git_sha"], "input_sha256": manifest["input_sha256"],
            "labels_sha256": sha256_file(labels_path), "results": results,
            "cost": costs, "relative_budget_difference_vs_none": budget,
            "gate_coverage": sum(row["evidence"]["gate_passed"] for row in rows),
            "typed_supported": sum(row["evidence"]["program"] is not None for row in rows),
            "distillation_valid": sum(row["distillation"]["parsed"]["valid"] for row in rows),
            "candidate_answer_disagreements": sum(len({c["answer"] for c in row["candidates"]}) > 1 for row in rows),
            "interpretation": "Single-seed development pilot; shared candidates and distillation, not a budget-matched or cross-model result."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an evaluation result")
    result = compare(args.run, args.labels, args.expected_n)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
