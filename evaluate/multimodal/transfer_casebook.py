#!/usr/bin/env python3
"""Inspect four-arm transfer decisions, with optional post-audit labels.

The casebook is read-only and retains model text verbatim. It selects cases by
output differences and gate coverage, never by correctness. Supplying labels
is allowed only after the complete four-arm audit passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.multimodal.datasets import load_local_dataset
from evaluate.multimodal.compare_transfer import ARMS, audit_four
from evaluate.multimodal.metrics import score_prediction
from scripts.inspect_baseline_io import read_rows, require_label_free
from scripts.create_label_isolated_subset import sha256_file


def build_casebook(run: Path, inference: Path, expected_n: int, labels: Path | None,
                   max_cases: int = 50) -> dict:
    manifest, rows = audit_four(run, expected_n)
    public = read_rows(inference)
    if len(public) != expected_n or sha256_file(inference) != manifest["input_sha256"]:
        raise ValueError("Input stream does not match completed run")
    for record in public:
        require_label_free(record)
    if [str(record["id"]) for record in public[:expected_n]] != manifest["ordered_ids"]:
        raise ValueError("Input casebook stream does not match completed run")
    target_map = None
    if labels is not None:
        targets = load_local_dataset(str(labels), "scienceqa_img", str(labels.parent))
        if len(targets) != expected_n or [e.sample.sample_id for e in targets] != manifest["ordered_ids"]:
            raise ValueError("Offline target order mismatch")
        target_map = {e.sample.sample_id: e.target for e in targets}
    choices = []
    for index, row in enumerate(rows):
        answers = {arm: row["arms"][arm]["answer"] for arm in ARMS}
        changed = len(set(answers.values())) > 1
        gate = row["evidence"]["gate_passed"]
        if changed or gate or index < 3:
            choices.append((index, changed, gate))
    # First three examples, all changed answers, then gate cases by stream
    # order; max_cases caps report size without inspecting correctness.
    chosen = sorted(choices, key=lambda x: (0 if x[0] < 3 else 1 if x[1] else 2, x[0]))[:max_cases]
    cases = []
    for index, changed, gate in chosen:
        row = rows[index]
        sid = row["sample_id"]
        case = {"sample_index": index, "sample_id": sid,
                "selection_reason": "first_examples" if index < 3 else "arm_disagreement" if changed else "gate_passed",
                "input": public[index], "candidate_attempts": row["candidate_attempts"],
                "pseudo_pair": row["pseudo_pair"], "evidence": row["evidence"],
                "distillation": row["distillation"],
                "arms": {arm: row["arms"][arm] for arm in ARMS}}
        if target_map is not None:
            target = target_map[sid]
            if target is None:
                raise ValueError("Missing offline target")
            case["offline"] = {"targets": list(target.answers),
                               "score": {arm: score_prediction("scienceqa_img", row["arms"][arm]["answer"],
                                                                 target.answers, target.metadata) for arm in ARMS}}
        cases.append(case)
    return {"all_arms_complete": True, "labels_read": labels is not None,
            "run_git_sha": manifest["git_sha"], "expected_n": expected_n,
            "selection": "first_3_plus_answer_disagreements_plus_gates_no_correctness_selection",
            "total_arm_disagreements": sum(len({row["arms"][arm]["answer"] for arm in ARMS}) > 1 for row in rows),
            "total_gates": sum(row["evidence"]["gate_passed"] for row in rows),
            "cases": cases}


def render_markdown(report: dict) -> str:
    lines = ["# Four-arm online transfer casebook", "",
             f"Run SHA: `{report['run_git_sha']}`; items: {report['expected_n']}; "
             f"labels read: {report['labels_read']}.", "",
             f"Answer disagreements: {report['total_arm_disagreements']}; "
             f"gated examples: {report['total_gates']}.", "",
             "Cases are selected by stream order, arm disagreement and gate coverage, not correctness.", ""]
    for case in report["cases"]:
        lines += [f"## {case['sample_index']}: {case['sample_id']} ({case['selection_reason']})", "",
                  "Input:", "```json", json.dumps(case["input"], ensure_ascii=False, indent=2), "```", "",
                  "Candidate proposals:", ""]
        for attempt in case["candidate_attempts"]:
            lines += [f"- {attempt['trajectory']}: `{attempt['answer']}`; format={attempt['format_ok']}; "
                      f"raw={json.dumps(attempt['response']['content'], ensure_ascii=False)}"]
        lines += ["", "Pseudo-pair and distilled procedure:", "```json",
                  json.dumps({"pair": case["pseudo_pair"], "rule": case["distillation"]["parsed"]},
                             ensure_ascii=False, indent=2), "```", ""]
        evidence = case["evidence"]
        lines += [f"Typed gate: {evidence['gate_passed']}; operator: "
                  f"{evidence['program']['operator'] if evidence['program'] else 'unsupported'}; "
                  f"reason: {evidence['fallback_reason']}", ""]
        if "offline" in case:
            lines += ["Offline target and score (opened after four-arm audit):", "```json",
                      json.dumps(case["offline"], ensure_ascii=False, indent=2), "```", ""]
        for arm in ARMS:
            data = case["arms"][arm]
            lines += [f"### {arm}: {data['answer']}", "", "Retrieved past rules:", ""]
            if data["retrieved"]:
                for item in data["retrieved"]:
                    lines += [f"- From {item['sample_id']} (index {item['sample_index']}, trust "
                              f"{item['trust']}): {item['positive_rule']}"]
            else:
                lines += ["- None"]
            lines += ["", "Exact final user input:", "```text", data["input_user_text"], "```", "",
                      "Final raw response:", "```text", data["response"]["content"], "```", "",
                      f"Write decision: {json.dumps(data['write_decision'], ensure_ascii=False)}", ""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, default=400)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    markdown = args.output.with_suffix(".md")
    if args.output.exists() or markdown.exists():
        raise FileExistsError("Refusing to overwrite an existing casebook")
    report = build_casebook(args.run, args.inference, args.expected_n, args.labels, args.max_cases)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    markdown.write_text(render_markdown(report))
    print(json.dumps({k: v for k, v in report.items() if k != "cases"} |
                     {"case_count": len(report["cases"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
