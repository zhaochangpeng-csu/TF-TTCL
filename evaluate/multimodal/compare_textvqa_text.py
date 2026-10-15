#!/usr/bin/env python3
"""Audit complete TextVQA paired predictions, then score isolated labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.multimodal.answers import extract_final_answer, normalize_answer
from core.multimodal.datasets import load_local_dataset
from core.multimodal.image_ops import sha256_file as image_sha256
from core.multimodal.schema import GenerationResponse
from core.multimodal.text_only_runner import ARMS, CANDIDATES, choose_vote, safe_variants
from core.multimodal.textvqa_text_runner import VERSION, textvqa_format_ok
from evaluate.multimodal.metrics import paired_bootstrap_delta_ci, paired_randomization_p_value, score_prediction
from scripts.create_label_isolated_subset import sha256_file


def audit(path: Path, inference: Path, expected_n: int) -> tuple[dict, list[dict]]:
    manifest = json.loads((path / "manifest.json").read_text())
    rows = [json.loads(line) for line in (path / "predictions.jsonl").read_text().splitlines()]
    examples = load_local_dataset(str(inference), "textvqa", str(inference.parent), limit=expected_n)
    if (manifest.get("status") != "completed" or manifest.get("all_arms_complete") is not True
            or manifest.get("labels_read") is not False or manifest.get("memory_enabled") is not False
            or manifest.get("version") != VERSION or manifest.get("processed") != expected_n
            or manifest.get("input_sha256") != sha256_file(inference)
            or manifest.get("arms") != list(ARMS)
            or len(rows) != expected_n or len(examples) != expected_n
            or manifest.get("ordered_ids") != [row.get("sample_id") for row in rows]):
        raise ValueError("Incomplete or mismatched TextVQA run; labels remain unopened")
    for index, (row, example) in enumerate(zip(rows, examples)):
        sample = example.sample
        if (example.target is not None or row.get("sample_index") != index
                or row.get("sample_id") != sample.sample_id
                or row.get("question") != sample.question
                or row.get("choices") != list(sample.choices)
                or row.get("metadata") != sample.metadata
                or set(row.get("arms", {})) != set(ARMS)
                or len(row.get("candidates", [])) != 5
                or any(key in row for key in ("target", "label", "gold", "answers"))):
            raise ValueError("Bad paired row or target leakage")
        hashes = [image_sha256(image) for image in sample.image_paths]
        if len(hashes) != 1 or row.get("image_hashes") != hashes:
            raise ValueError("Image hashes changed")
        candidates = row["candidates"]
        if [candidate["name"] for candidate in candidates] != list(CANDIDATES):
            raise ValueError("Candidate order changed")
        variants, changed = safe_variants(sample.question, row["rewrite"]["response"]["content"])
        if row["rewrite"]["variants"] != variants or row["rewrite"]["changed"] != changed:
            raise ValueError("Question rewrite mismatch")
        for candidate, question in zip(candidates, [sample.question] * 3 + variants):
            response = GenerationResponse(**candidate["response"])
            raw = extract_final_answer(response.content)
            answer = normalize_answer(raw, "textvqa")
            if (candidate["question"] != question or candidate["image_hashes"] != hashes
                    or candidate["raw_answer"] != raw or candidate["answer"] != answer
                    or candidate["format_ok"] != textvqa_format_ok(response, answer)):
                raise ValueError("TextVQA image/question/format contract broken")
        expected_answers = (candidates[0]["answer"], choose_vote(candidates[:3]),
                            choose_vote([candidates[0], *candidates[3:]]))
        for arm, expected in zip(ARMS, expected_answers):
            if row["arms"][arm]["answer"] != expected:
                raise ValueError("Arm vote differs from committed candidates")
    return manifest, rows


def _raw_vote_answer(row: dict, arm: str) -> str:
    selected = row["arms"][arm]["answer"]
    names = set(row["arms"][arm]["candidate_names"])
    for candidate in row["candidates"]:
        if candidate["name"] in names and candidate["answer"] == selected:
            return candidate["raw_answer"]
    raise ValueError("Selected answer has no supporting candidate")


def _paired(left: list[float], right: list[float]) -> dict:
    differences = [b - a for a, b in zip(left, right)]
    return {"improved_items": sum(delta > 1e-12 for delta in differences),
            "worsened_items": sum(delta < -1e-12 for delta in differences),
            "net_score_points": sum(differences),
            "paired_delta_ci": paired_bootstrap_delta_ci(left, right),
            "paired_randomization_p": paired_randomization_p_value(left, right)}


def compare(path: Path, inference: Path, labels: Path, expected_n: int = 5000) -> dict:
    if expected_n != 5000:
        raise ValueError("Only the complete TextVQA validation set can be scored")
    manifest, rows = audit(path, inference, expected_n)
    targets = load_local_dataset(str(labels), "textvqa", str(labels.parent))
    if ([item.sample.sample_id for item in targets] != manifest["ordered_ids"]
            or any(item.target is None or len(item.target.answers) != 10 for item in targets)):
        raise ValueError("Isolated labels do not match the complete stream")
    scores = {arm: [score_prediction("textvqa", _raw_vote_answer(row, arm), item.target.answers)
                    for row, item in zip(rows, targets)] for arm in ARMS}
    cost = {arm: {key: 0.0 for key in ("prompt_tokens", "completion_tokens", "latency_seconds")}
            for arm in ARMS}
    for row in rows:
        candidates, rewrite = row["candidates"], row["rewrite"]["response"]
        for arm, responses in (("greedy", [candidates[0]["response"]]),
                               ("same_text_vote", [rewrite, *[c["response"] for c in candidates[:3]]]),
                               ("text_aug_vote", [rewrite, candidates[0]["response"],
                                                  candidates[3]["response"], candidates[4]["response"]])):
            for response in responses:
                for key in cost[arm]:
                    cost[arm][key] += response[key]
    results = {arm: {"vqa_accuracy": sum(scores[arm]) / expected_n,
                     "score_sum": sum(scores[arm])} for arm in ARMS}
    return {"status": "scored_after_paired_audit", "n": expected_n,
            "run_git_sha": manifest["git_sha"], "input_sha256": manifest["input_sha256"],
            "labels_sha256": sha256_file(labels), "results": results,
            "text_minus_same": _paired(scores["same_text_vote"], scores["text_aug_vote"]),
            "text_minus_greedy": _paired(scores["greedy"], scores["text_aug_vote"]),
            "logical_cost": cost,
            "rewrite_changed_items": sum(any(row["rewrite"]["changed"]) for row in rows),
            "answer_disagreement_same_vs_text": sum(
                row["arms"]["same_text_vote"]["answer"] != row["arms"]["text_aug_vote"]["answer"]
                for row in rows),
            "format_invalid_candidates": sum(not c["format_ok"] for row in rows for c in row["candidates"]),
            "metric_note": "Local deterministic VQA soft score; official EvalAI processor parity not yet verified."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an evaluation")
    result = compare(args.run, args.inference, args.labels)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
