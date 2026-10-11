#!/usr/bin/env python3
"""Label-free protocol audit for paired CCIC budget/evidence runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_predictions(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    order: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row["sample_id"])
        if sample_id in rows:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in {path}")
        order.append(sample_id)
        rows[sample_id] = row
    return order, rows


def manifest_for(predictions: Path) -> dict[str, Any]:
    return json.loads((predictions.parent / "manifest.json").read_text(encoding="utf-8"))


def call_trace(row: dict[str, Any]) -> tuple[Any, ...]:
    """Describe adaptive call inputs and outputs without consulting labels."""
    candidates = tuple(
        (
            item.get("trajectory"),
            item.get("answer"),
            item.get("question_variant"),
            tuple(item.get("image_hashes", [])),
        )
        for item in row.get("candidate_answers", [])
    )
    certificate = row.get("causal_certificate", {})
    localizations = tuple(
        (
            item.get("kind"),
            item.get("view"),
            item.get("candidate_1"),
            item.get("candidate_2"),
            tuple(item.get("image_hashes", [])),
            json.dumps(item.get("proposal", {}), sort_keys=True),
        )
        for item in certificate.get("localizations", [])
    )
    interventions = tuple(
        (
            item.get("view"),
            item.get("expected_answer"),
            item.get("observed_answer"),
            tuple(item.get("image_hashes", [])),
        )
        for item in certificate.get("interventions", [])
    )
    return candidates, localizations, interventions


def audit_runs(budget_path: Path, evidence_path: Path) -> dict[str, Any]:
    budget_order, budget_rows = load_predictions(budget_path)
    evidence_order, evidence_rows = load_predictions(evidence_path)
    if budget_order != evidence_order:
        raise ValueError("Budget and evidence runs must have the identical ordered stream")

    budget_manifest = manifest_for(budget_path)
    evidence_manifest = manifest_for(evidence_path)
    if budget_manifest.get("method") != "causal_budget":
        raise ValueError("Budget manifest method must be causal_budget")
    if evidence_manifest.get("method") != "causal_evidence":
        raise ValueError("Evidence manifest method must be causal_evidence")

    gates = [
        evidence_rows[sample_id].get("causal_certificate", {})
        for sample_id in evidence_order
    ]
    fallback_reasons = Counter(
        str(gate.get("fallback_reason"))
        for gate in gates
        if not gate.get("gate_passed")
    )
    question_parse_errors = 0
    candidate_parse_errors = 0
    question_candidate_leaks = 0
    for gate in gates:
        for item in gate.get("localizations", []):
            parse_error = bool(item.get("proposal", {}).get("parse_error"))
            if item.get("kind") == "question_only":
                question_parse_errors += parse_error
                question_candidate_leaks += item.get("candidate_answers_exposed") is not False
            elif item.get("kind") == "candidate_conditional":
                candidate_parse_errors += parse_error

    budget_tokens = int(budget_manifest.get("prompt_tokens", 0)) + int(
        budget_manifest.get("completion_tokens", 0)
    )
    evidence_tokens = int(evidence_manifest.get("prompt_tokens", 0)) + int(
        evidence_manifest.get("completion_tokens", 0)
    )
    trace_mismatches = [
        sample_id
        for sample_id in evidence_order
        if call_trace(budget_rows[sample_id]) != call_trace(evidence_rows[sample_id])
    ]
    answer_differences = [
        sample_id
        for sample_id in evidence_order
        if budget_rows[sample_id].get("answer") != evidence_rows[sample_id].get("answer")
    ]
    result = {
        "n": len(evidence_order),
        "manifests_completed": (
            budget_manifest.get("status") == "completed"
            and evidence_manifest.get("status") == "completed"
        ),
        "candidate_disagreements": sum(
            gate.get("alternative_answer") is not None for gate in gates
        ),
        "localization_stable": sum(bool(gate.get("localization_stable")) for gate in gates),
        "candidate_regions_disjoint": sum(
            bool(gate.get("candidate_regions_disjoint")) for gate in gates
        ),
        "question_region_alternative_exclusive": sum(
            bool(gate.get("question_region_alternative_exclusive")) for gate in gates
        ),
        "causal_crossover": sum(bool(gate.get("causal_crossover")) for gate in gates),
        "gate_passed": sum(bool(gate.get("gate_passed")) for gate in gates),
        "answer_differences": answer_differences,
        "fallback_reasons": dict(fallback_reasons),
        "parse_errors": {
            "question_only": question_parse_errors,
            "candidate_conditional": candidate_parse_errors,
        },
        "question_only_candidate_leak_records": question_candidate_leaks,
        "budget_audit": {
            "call_trace_mismatches": trace_mismatches,
            "budget_total_tokens": budget_tokens,
            "evidence_total_tokens": evidence_tokens,
            "total_token_relative_difference": (
                abs(budget_tokens - evidence_tokens) / max(1, evidence_tokens)
            ),
        },
    }
    result["protocol_ok"] = bool(
        result["manifests_completed"]
        and not trace_mismatches
        and question_parse_errors == 0
        and candidate_parse_errors == 0
        and question_candidate_leaks == 0
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_runs(Path(args.budget), Path(args.evidence)),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
