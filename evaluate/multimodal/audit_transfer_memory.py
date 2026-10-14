#!/usr/bin/env python3
"""Label-free audit of rule content and retrieval applicability in transfer v1.

The indicators here are lexical proxies. A category match cannot certify that
a rule applies, and mentioning source option text is not automatically wrong.
No target labels are accepted or opened.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.multimodal.transfer_quality import assess_rule, literal_mention
from core.multimodal.transfer_runner import RuleEntry, lexical_retrieve
from evaluate.multimodal.compare_transfer import audit_four


ARMS = ("noisy", "gated_weighted", "random")
_ASSERTION_START = re.compile(
    r"^(?:the|this) (?:image|picture|map|diagram|highlighted|punnett square|table)\b", re.I
)
_INTERNAL_CAPITALIZED = re.compile(r"\b[A-Z][a-z]{2,}\b")


def internal_capitalized_tokens(rule: str) -> list[str]:
    """Heuristic source-anchor candidates, excluding the first sentence word."""
    return sorted({match.group() for match in _INTERNAL_CAPITALIZED.finditer(rule)
                   if match.start() != 0})


def rule_signals(row: dict) -> dict:
    entry = row["arms"]["noisy"]["written"]
    if entry is None:
        return {"written": False}
    positive = entry["positive_rule"]
    negative = entry["negative_rule"]
    combined = positive + " " + negative
    choices = [str(choice) for choice in row["choices"]]
    answer_text = entry["positive_answer_text"]
    negative_text = entry.get("negative_answer_text")
    mention = literal_mention(combined, answer_text)
    question_tail = row["question"].rsplit("\n\nQuestion:\n", 1)[-1].strip().lower()
    map_task = bool(re.search(r"\b(?:country|state|continent|island)\b", question_tail)
                    and re.search(r"\b(?:highlighted|shown|located|intersect)\b", question_tail))
    return {"written": True, "source_answer_mentioned": mention,
            "negative_answer_in_positive_rule": bool(negative_text)
                                                and literal_mention(positive, negative_text),
            "source_choice_mentions": [choice for choice in choices if literal_mention(combined, choice)],
            "assertion_start": bool(_ASSERTION_START.search(positive.strip())),
            "map_task": map_task,
            "source_answer_text": answer_text,
            "positive_rule": positive}


def summarize(rows: list[dict]) -> dict:
    by_id = {row["sample_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate sample IDs")
    signals = {row["sample_id"]: rule_signals(row) for row in rows}
    written = [row for row in rows if signals[row["sample_id"]]["written"]]
    content = {"written": len(written),
               "source_answer_mentioned": sum(signals[r["sample_id"]]["source_answer_mentioned"] for r in written),
               "negative_answer_in_positive_rule": sum(signals[r["sample_id"]]["negative_answer_in_positive_rule"]
                                                       for r in written),
               "any_source_choice_mentioned": sum(bool(signals[r["sample_id"]]["source_choice_mentions"]) for r in written),
               "assertion_start": sum(signals[r["sample_id"]]["assertion_start"] for r in written),
               "map_tasks_written": sum(signals[r["sample_id"]]["map_task"] for r in written),
               "map_task_source_answer_mentioned": sum(signals[r["sample_id"]]["map_task"]
                                                        and signals[r["sample_id"]]["source_answer_mentioned"]
                                                        for r in written)}
    retrieval = {}
    for arm in ARMS:
        events = []
        changed = []
        for index, row in enumerate(rows):
            current = row["metadata"]
            for entry in row["arms"][arm]["retrieved"]:
                source = by_id.get(entry["sample_id"])
                if source is None or source["sample_index"] >= index:
                    raise ValueError("Retrieved item is absent from the past source stream")
                meta = source["metadata"]
                same = {field: bool(meta.get(field)) and meta.get(field) == current.get(field)
                        for field in ("subject", "topic", "category", "skill")}
                signal = signals[source["sample_id"]]
                events.append({"current_index": index, "current_id": row["sample_id"],
                               "source_index": source["sample_index"], "source_id": source["sample_id"],
                               "same": same,
                               "same_question": row["question"] == source["question"],
                               "source_answer_mentioned": signal.get("source_answer_mentioned", False),
                               "negative_answer_in_positive_rule": signal.get("negative_answer_in_positive_rule", False),
                               "source_map_answer_mentioned": signal.get("map_task", False)
                                                              and signal.get("source_answer_mentioned", False),
                               "different_image": row["image_hashes"] != source["image_hashes"]})
            if row["arms"][arm]["answer"] != row["arms"]["none"]["answer"]:
                changed.append({"index": index, "sample_id": row["sample_id"],
                                "retrieved_source_ids": [e["sample_id"] for e in row["arms"][arm]["retrieved"]],
                                "same_skill_count": sum(by_id[e["sample_id"]]["metadata"].get("skill")
                                                        == current.get("skill")
                                                        and bool(current.get("skill"))
                                                        for e in row["arms"][arm]["retrieved"])})
        retrieval[arm] = {"rows_with_retrieval": sum(bool(r["arms"][arm]["retrieved"]) for r in rows),
                          "events": len(events),
                          **{f"same_{field}_events": sum(e["same"][field] for e in events)
                             for field in ("subject", "topic", "category", "skill")},
                          "same_question_events": sum(e["same_question"] for e in events),
                          "source_answer_mention_events": sum(e["source_answer_mentioned"] for e in events),
                          "negative_answer_in_positive_rule_events": sum(
                              e["negative_answer_in_positive_rule"] for e in events),
                          "source_map_answer_on_different_image_events": sum(
                              e["source_map_answer_mentioned"] and e["different_image"] for e in events),
                          "answer_changed_vs_none": len(changed),
                          "changed_cases": changed}
    examples = [{"index": row["sample_index"], "sample_id": row["sample_id"],
                 "question": row["question"], "answer_text": signals[row["sample_id"]]["source_answer_text"],
                 "rule": signals[row["sample_id"]]["positive_rule"]}
                for row in written if signals[row["sample_id"]]["map_task"]
                and signals[row["sample_id"]]["source_answer_mentioned"]][:8]
    negative_mentions = [{"index": row["sample_index"], "sample_id": row["sample_id"],
                       "positive_answer": row["arms"]["noisy"]["written"]["positive_answer_text"],
                       "negative_answer": row["arms"]["noisy"]["written"]["negative_answer_text"],
                       "rule": signals[row["sample_id"]]["positive_rule"]}
                      for row in written if signals[row["sample_id"]]["negative_answer_in_positive_rule"]][:12]
    accepted_bank: list[RuleEntry] = []
    rejection_reasons: Counter[str] = Counter()
    filtered_retrieval_rows = 0
    filtered_retrieval_events = 0
    filtered_same_skill_events = 0
    filtered_source_negative_mention_events = 0
    accepted_capitalized_examples = []
    for index, row in enumerate(rows):
        selected = lexical_retrieve(accepted_bank, row["question"], index)
        filtered_retrieval_rows += bool(selected)
        filtered_retrieval_events += len(selected)
        for entry in selected:
            old = by_id[entry.sample_id]
            filtered_same_skill_events += bool(row["metadata"].get("skill")) and (
                row["metadata"].get("skill") == old["metadata"].get("skill"))
            filtered_source_negative_mention_events += signals[entry.sample_id].get(
                "negative_answer_in_positive_rule", False)
        new_entry = row["arms"]["noisy"]["written"]
        if new_entry:
            verdict = assess_rule(new_entry)
            if verdict["accepted"]:
                accepted_bank.append(RuleEntry.from_dict(new_entry))
                tokens = internal_capitalized_tokens(new_entry["positive_rule"])
                if tokens:
                    accepted_capitalized_examples.append({"index": index,
                        "sample_id": row["sample_id"], "tokens": tokens,
                        "positive_rule": new_entry["positive_rule"]})
            else:
                rejection_reasons.update(verdict["reasons"])
    filtered = {"accepted_writes": len(accepted_bank),
                "rejected_writes": len(written) - len(accepted_bank),
                "rejection_reason_counts": dict(rejection_reasons),
                "retrieval_rows": filtered_retrieval_rows,
                "retrieval_events": filtered_retrieval_events,
                "same_skill_events": filtered_same_skill_events,
                "negative_answer_in_positive_rule_events": filtered_source_negative_mention_events,
                "accepted_internal_capitalized_count": len(accepted_capitalized_examples),
                "accepted_internal_capitalized_examples": accepted_capitalized_examples[:12]}
    return {"n": len(rows), "labels_read": False,
            "definitions": {"same_skill": "exact official skill metadata match, audit proxy only",
                            "source_answer_mentioned": "whole-token literal source positive-answer text in retrieved rule",
                            "assertion_start": "rule starts with an image/map/diagram statement rather than a checking verb"},
            "content": content, "retrieval": retrieval, "map_fact_examples": examples,
            "negative_answer_mention_examples": negative_mentions,
            "procedural_filter_dry_run": filtered}


def render_markdown(report: dict) -> str:
    c = report["content"]
    lines = ["# Label-free memory content and retrieval audit", "",
             f"Samples: {report['n']}; labels read: false. Indicators are lexical proxies.", "",
             f"Noisy rules written: {c['written']}; source-answer text mentioned: "
             f"{c['source_answer_mentioned']}; any source choice mentioned: "
             f"{c['any_source_choice_mentioned']}; assertion-like start: {c['assertion_start']}.", "",
             f"Positive rules literally mentioning the paired negative answer: "
             f"{c['negative_answer_in_positive_rule']}.", "",
             f"Map-like rules written: {c['map_tasks_written']}; those mentioning the source answer: "
             f"{c['map_task_source_answer_mentioned']}.", "",
             "| Arm | Retrieved rows | Rule events | Same skill | Same category | Paired-negative mentions | Changed answers |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        d = report["retrieval"][arm]
        lines.append(f"| {arm} | {d['rows_with_retrieval']} | {d['events']} | "
                     f"{d['same_skill_events']} | {d['same_category_events']} | "
                     f"{d['negative_answer_in_positive_rule_events']} | "
                     f"{d['answer_changed_vs_none']} |")
    lines += ["", "Official skill/category matches measure task similarity, not rule validity.",
              "The answer-mention flag identifies a possible source-fact shortcut, not a proven error.", "",
              "## Map-answer examples (stream order, no correctness selection)", ""]
    for case in report["map_fact_examples"]:
        lines += [f"- {case['index']} / {case['sample_id']}: source answer "
                  f"`{case['answer_text']}`; rule: {case['rule']}"]
    lines += ["", "## Positive rules mentioning a paired negative answer", "",
              "Mentions can be valid conditionals; review the wording before treating one as a contradiction.", ""]
    for case in report["negative_answer_mention_examples"]:
        lines += [f"- {case['index']} / {case['sample_id']}: positive "
                  f"`{case['positive_answer']}`, negative `{case['negative_answer']}`; "
                  f"positive rule: {case['rule']}"]
    dry = report["procedural_filter_dry_run"]
    lines += ["", "## Procedural-only write filter, label-free replay of retrieval", "",
              f"Accepted writes: {dry['accepted_writes']}; rejected: {dry['rejected_writes']}; "
              f"retrieval rows: {dry['retrieval_rows']}; events: {dry['retrieval_events']}; "
              f"same-skill events: {dry['same_skill_events']}.", "",
              f"Accepted rules with an internal capitalized token: "
              f"{dry['accepted_internal_capitalized_count']}. This is a noisy source-anchor "
              "screen, not a named-entity or leakage classifier.", "",
              "Rejection reasons: `" + json.dumps(dry["rejection_reason_counts"], sort_keys=True) + "`.", "",
              "Potential source-anchored survivors:", ""]
    for case in dry["accepted_internal_capitalized_examples"]:
        lines.append(f"- {case['index']} / {case['sample_id']}: "
                     f"{', '.join(case['tokens'])}; {case['positive_rule']}")
    lines += ["",
              "This dry run replays BM25 ranking only. It makes no model calls and cannot measure accuracy.", ""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    markdown = args.output.with_suffix(".md")
    if args.output.exists() or markdown.exists():
        raise FileExistsError("Refusing to overwrite an existing audit")
    _, rows = audit_four(args.run, args.expected_n)
    report = summarize(rows)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    markdown.write_text(render_markdown(report))
    print(json.dumps({"n": report["n"], "content": report["content"],
                      "procedural_filter_dry_run": report["procedural_filter_dry_run"],
                      "retrieval": {arm: {k: v for k, v in data.items() if k != "changed_cases"}
                                    for arm, data in report["retrieval"].items()}}, indent=2))


if __name__ == "__main__":
    main()
