"""Answer-free cost targets for an independent sampling-budget control.

The control receives only per-example costs, not evidence, candidate answers,
labels or reference predictions. Costs are known after the reference run;
this is a cost-matched development comparator, not a deployable online budget.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def namespace(config: dict) -> dict:
    experiment, model = config["experiment"], config["model"]
    return {"dataset": experiment["dataset"], "seed": experiment.get("seed", 0),
            "source_sha256": digest(experiment["data_path"]),
            "model": model["model_name"], "revision": model.get("revision"),
            "order": experiment.get("order", "forward"), "limit": experiment.get("limit")}


def export_plan(predictions: Path, output: Path) -> dict:
    manifest = json.loads(predictions.with_name("manifest.json").read_text())
    rows = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    if manifest.get("status") != "completed" or manifest.get("processed") != len(rows):
        raise ValueError("Cost plan requires a complete reference run")
    if manifest.get("method") != "shared_budget" or not rows:
        raise ValueError("Unexpected or empty cost reference")
    items = [{"sample_id": row["sample_id"], "target_tokens":
              row["usage_delta"]["prompt_tokens"] + row["usage_delta"]["completion_tokens"]}
             for row in rows]
    if any(type(item["target_tokens"]) is not int or item["target_tokens"] <= 0 for item in items):
        raise ValueError("Missing reference usage")
    if len({i["sample_id"] for i in items}) != len(items):
        raise ValueError("Duplicate cost target ID")
    if sum(i["target_tokens"] for i in items) != manifest["prompt_tokens"] + manifest["completion_tokens"]:
        raise ValueError("Reference per-example usage does not match manifest")
    plan = {"version": "cost_only_v1", "namespace": namespace(manifest["config"]), "items": items}
    with output.open("x") as handle:
        json.dump(plan, handle, indent=2)
    return plan


def load_plan(path: str, config: dict, sample_ids: list[str]) -> list[int]:
    plan = json.loads(Path(path).read_text())
    if (set(plan) != {"version", "namespace", "items"} or plan["version"] != "cost_only_v1"
            or plan["namespace"] != namespace(config)):
        raise ValueError("Budget plan namespace/schema mismatch")
    items = plan["items"]
    if any(set(item) != {"sample_id", "target_tokens"} or type(item["target_tokens"]) is not int
           or item["target_tokens"] <= 0 for item in items):
        raise ValueError("Budget plan must contain only IDs and positive integer costs")
    if [item["sample_id"] for item in items] != sample_ids:
        raise ValueError("Budget plan stream mismatch")
    return [item["target_tokens"] for item in items]


def wants_extra(actual_tokens: int, target_tokens: int, estimated_call_tokens: float) -> bool:
    """One more useful generation only if it moves estimated cost closer."""
    return abs(actual_tokens + estimated_call_tokens - target_tokens) < abs(actual_tokens - target_tokens)
