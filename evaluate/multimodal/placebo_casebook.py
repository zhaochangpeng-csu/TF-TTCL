#!/usr/bin/env python3
"""Inspect exact paired placebo/source I/O, selecting cases without labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.multimodal.datasets import load_local_dataset
from evaluate.multimodal.compare_placebo import audit_placebo
from evaluate.multimodal.metrics import score_prediction
from scripts.create_label_isolated_subset import sha256_file
from scripts.inspect_baseline_io import read_rows, require_label_free


ARMS = ("none", "noisy", "gated_weighted", "random", "placebo")


def build_casebook(source: Path, run: Path, inference: Path, expected_n: int,
                   labels: Path | None = None, source_expected_n: int = 400) -> dict:
    manifest, rows, source_rows = audit_placebo(source, run, expected_n, source_expected_n)
    public = read_rows(inference)
    if (len(public) != source_expected_n or sha256_file(inference) != manifest["source_input_sha256"]
            or [str(row["id"]) for row in public[:expected_n]] != manifest["ordered_ids"]):
        raise ValueError("Casebook input differs from audited stream")
    for row in public:
        require_label_free(row)
    targets = None
    if labels is not None:
        examples = load_local_dataset(str(labels), "scienceqa_img", str(labels.parent))
        if len(examples) != source_expected_n or [e.sample.sample_id for e in examples[:expected_n]] != manifest["ordered_ids"]:
            raise ValueError("Casebook labels differ from audited stream")
        if any(e.target is None for e in examples[:expected_n]):
            raise ValueError("Casebook target missing")
        targets = examples
    cases = []
    disagreements = 0
    for index, (row, old) in enumerate(zip(rows, source_rows)):
        answers = {arm: old["arms"][arm]["answer"] for arm in ARMS[:-1]}
        answers["placebo"] = row["answer"]
        changed = len(set(answers.values())) > 1
        disagreements += int(changed)
        if index >= 3 and not changed:
            continue
        case = {"sample_index": index, "sample_id": row["sample_id"],
                "selection_reason": "first_examples" if index < 3 else "arm_disagreement",
                "input": public[index], "candidate_attempts": old["candidate_attempts"],
                "pseudo_pair": old["pseudo_pair"], "distilled_rule": old["distillation"]["parsed"],
                "retrieved": old["arms"]["noisy"]["retrieved"],
                "arms": {arm: {"answer": old["arms"][arm]["answer"],
                               "input_user_text": old["arms"][arm]["input_user_text"],
                               "raw_response": old["arms"][arm]["response"]["content"]}
                         for arm in ARMS[:-1]}}
        case["arms"]["placebo"] = {"answer": row["answer"],
                                    "input_user_text": row["input_user_text"],
                                    "raw_response": row["response"]["content"]}
        if targets is not None:
            target = targets[index].target
            case["offline"] = {"targets": list(target.answers),
                               "score": {arm: score_prediction("scienceqa_img", answer,
                                                                 target.answers, target.metadata)
                                         for arm, answer in answers.items()}}
        cases.append(case)
    return {"all_arms_complete": True, "source_labels_previously_scored": True,
            "labels_read_for_casebook": labels is not None, "n": expected_n,
            "selection": "first_3_plus_all_answer_disagreements_no_correctness_selection",
            "answer_disagreements": disagreements, "cases": cases}


def render_markdown(report: dict) -> str:
    lines = ["# Exact-candidate placebo/source casebook", "",
             f"Items: {report['n']}; answer disagreements: {report['answer_disagreements']}; "
             f"labels read for casebook: {report['labels_read_for_casebook']}.", "",
             "Cases are selected by stream position and model-answer disagreement, never correctness.", ""]
    for case in report["cases"]:
        lines += [f"## {case['sample_index']}: {case['sample_id']} ({case['selection_reason']})", "",
                  "Input:", "```json", json.dumps(case["input"], ensure_ascii=False, indent=2), "```", "",
                  "Current candidate proposals:", "```json",
                  json.dumps(case["candidate_attempts"], ensure_ascii=False, indent=2), "```", "",
                  "Pseudo-pair and distilled rule:", "```json",
                  json.dumps({"pair": case["pseudo_pair"], "rule": case["distilled_rule"]},
                             ensure_ascii=False, indent=2), "```", "",
                  "Retrieved source rules:", "```json",
                  json.dumps(case["retrieved"], ensure_ascii=False, indent=2), "```", ""]
        if "offline" in case:
            lines += ["Offline target (opened only after full audit):", "```json",
                      json.dumps(case["offline"], ensure_ascii=False, indent=2), "```", ""]
        for arm in ARMS:
            data = case["arms"][arm]
            lines += [f"### {arm}: {data['answer']}", "", "Exact final user input:", "```text",
                      data["input_user_text"], "```", "", "Raw final output:", "```text",
                      data["raw_response"], "```", ""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--expected-n", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    markdown = args.output.with_suffix(".md")
    if args.output.exists() or markdown.exists():
        raise FileExistsError("Refusing to overwrite an existing casebook")
    report = build_casebook(args.source, args.run, args.inference, args.expected_n, args.labels)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    markdown.write_text(render_markdown(report))
    print(json.dumps({k: v for k, v in report.items() if k != "cases"} |
                     {"case_count": len(report["cases"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
