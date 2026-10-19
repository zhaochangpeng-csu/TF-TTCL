#!/usr/bin/env python3
"""Post-hoc ScienceQA memory-change casebook; never used during inference."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.multimodal.datasets import load_local_dataset
from evaluate.multimodal.metrics import score_prediction
from scripts.audit_tf_ttcl_mm_direct import audit


def analyze(run: Path, inference: Path, labels: Path, count: int = 400) -> dict:
    manifest, rows = audit(run, inference, count)
    if count != 400 or manifest["dataset"] != "scienceqa_img":
        raise ValueError("Only the completed ScienceQA dev400 is eligible for post-hoc analysis")
    targets = load_local_dataset(str(labels), "scienceqa_img", str(labels.parent))
    if len(targets) != count or [item.sample.sample_id for item in targets] != manifest["ordered_ids"]:
        raise ValueError("Labels differ from the audited stream")
    cases = []
    consensus = {arm: {"all_same": 0, "all_same_correct": 0}
                 for arm in ("select_no_memory", "tf_ttcl_mm_direct")}
    changed_teacher = 0
    for row, target in zip(rows, targets):
        plain = row["arms"]["select_no_memory"]
        memory = row["arms"]["tf_ttcl_mm_direct"]
        changed_teacher += int(plain["calls"][0]["normalized_answer"] !=
                               memory["calls"][0]["normalized_answer"])
        for arm_name, arm in (("select_no_memory", plain), ("tf_ttcl_mm_direct", memory)):
            answers = [call["normalized_answer"] for call in arm["calls"] if call.get("role")]
            if len(set(answers)) == 1:
                consensus[arm_name]["all_same"] += 1
                consensus[arm_name]["all_same_correct"] += int(score_prediction(
                    "scienceqa_img", arm["answer"], target.target.answers) == 1)
        if plain["answer"] == memory["answer"]:
            continue
        plain_score = score_prediction("scienceqa_img", plain["answer"], target.target.answers)
        memory_score = score_prediction("scienceqa_img", memory["answer"], target.target.answers)
        outcome = "improved" if memory_score > plain_score else (
            "worsened" if memory_score < plain_score else "changed_same_score")
        cases.append({"index": row["sample_index"], "id": row["sample_id"],
                      "outcome": outcome, "question": row["original_question"],
                      "choices": row["choices"], "gold": list(target.target.answers),
                      "no_memory_answer": plain["answer"], "memory_answer": memory["answer"],
                      "no_memory_teacher_answer": plain["calls"][0]["normalized_answer"],
                      "memory_teacher_answer": memory["calls"][0]["normalized_answer"],
                      "no_memory_selection": plain["reason"],
                      "memory_selection": memory["reason"],
                      "no_memory_candidate_answers": [call["normalized_answer"]
                                                       for call in plain["calls"] if call.get("role")],
                      "memory_candidate_answers": [call["normalized_answer"]
                                                    for call in memory["calls"] if call.get("role")],
                      "retrieved_rule_counts": {kind: len(bucket)
                                                for kind, bucket in row["retrieved_rules"].items()},
                      "retrieved_rule_examples": {kind: [item["text"] for item in bucket[:2]]
                                                  for kind, bucket in row["retrieved_rules"].items()}})
    outcomes = Counter(item["outcome"] for item in cases)
    return {"status": "posthoc_after_completed_audit", "run_git_sha": manifest["git_sha"],
            "count": count, "answer_changed_items": len(cases),
            "all_same_consensus": consensus, "teacher_answer_changed_items": changed_teacher,
            "outcomes": dict(outcomes),
            "selection_reasons": {outcome: dict(Counter(
                item["memory_selection"] for item in cases if item["outcome"] == outcome))
                for outcome in ("improved", "worsened", "changed_same_score")},
            "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.run, args.inference, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
