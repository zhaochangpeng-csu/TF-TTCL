#!/usr/bin/env python3
"""Label-free audit of the shared-fact selection experiment."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluate.multimodal.audit_causal import load_predictions, manifest_for


def audit_runs(budget_path: Path, evidence_path: Path) -> dict[str, Any]:
    order, budget = load_predictions(budget_path)
    evidence_order, evidence = load_predictions(evidence_path)
    if order != evidence_order or not order:
        raise ValueError("Shared evidence requires identical nonempty ordered streams")
    bm, em = manifest_for(budget_path), manifest_for(evidence_path)
    if bm.get("method") != "shared_budget" or em.get("method") != "shared_evidence":
        raise ValueError("Unexpected shared evidence methods")
    gates = [evidence[sid].get("shared_certificate", {}) for sid in order]
    mismatches = []
    for sid in order:
        def trace(row):
            return ([(c["trajectory"], c["answer"], c["question_variant"], c["image_hashes"])
                     for c in row["candidate_answers"]], row.get("shared_certificate"))
        if trace(budget[sid]) != trace(evidence[sid]):
            mismatches.append(sid)
    parse_errors = sum(o["parsed"]["parse_error"] for g in gates for o in g["observations"])
    exposed = sum(any(o.get(key) is not False for key in (
        "question_exposed", "choices_exposed", "candidate_answers_exposed"))
        for g in gates for o in g["observations"])
    policy_errors = []
    memory_leaks = []
    for sid, g in zip(order, gates):
        expected = g.get("computed_answer") if g.get("gate_passed") else g.get("greedy_answer")
        if evidence[sid]["answer"] != expected:
            policy_errors.append(sid)
        if any(row.get("wrote_memory") or row.get("retrieved_experience_ids")
               or row.get("memory_size_before") or row.get("memory_size_after")
               for row in (budget[sid], evidence[sid])):
            memory_leaks.append(sid)
    bt = bm.get("prompt_tokens", 0) + bm.get("completion_tokens", 0)
    et = em.get("prompt_tokens", 0) + em.get("completion_tokens", 0)
    cfg_same = all(bm.get(key) == em.get(key) for key in ("model", "seed", "prompt_hash", "git_sha"))
    for section, keys in {
        "experiment": ("dataset", "data_path", "image_root", "limit", "order", "seed"),
        "model": ("model_name", "revision", "max_tokens", "request_logprobs"),
    }.items():
        cfg_same &= all(bm["config"][section].get(k) == em["config"][section].get(k) for k in keys)
    result = {
        "n": len(order), "manifests_completed": all(m.get("status") == "completed"
            and m.get("processed") == len(order) for m in (bm, em)),
        "matched_configuration": bool(cfg_same),
        "supported_tasks": dict(Counter(g["operator"] for g in gates if g["operator"])),
        "candidate_disagreements": sum(g["candidate_disagreement"] for g in gates),
        "supported_disagreements": sum(g["candidate_disagreement"] and g["operator"] is not None
                                       for g in gates),
        "facts_stable": sum(g["facts_stable"] for g in gates),
        "counterfactual_drop": sum(g["counterfactual_drop"] for g in gates),
        "gate_passed": sum(g["gate_passed"] for g in gates),
        "changed_from_greedy": sum(g["changed_from_greedy"] for g in gates),
        "fallback_reasons": dict(Counter(str(g["fallback_reason"]) for g in gates if not g["gate_passed"])),
        "parse_error_responses": parse_errors, "exposed_observation_records": exposed,
        "policy_error_ids": policy_errors, "memory_leak_ids": memory_leaks,
        "budget_audit": {"call_trace_mismatches": mismatches,
                         "budget_total_tokens": bt, "evidence_total_tokens": et,
                         "total_token_relative_difference": abs(bt - et) / max(1, et)},
    }
    result["protocol_ok"] = bool(result["manifests_completed"] and cfg_same
        and not mismatches and not parse_errors and not exposed
        and not policy_errors and not memory_leaks and bt == et)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    print(json.dumps(audit_runs(Path(args.budget), Path(args.evidence)), indent=2))


if __name__ == "__main__":
    main()
