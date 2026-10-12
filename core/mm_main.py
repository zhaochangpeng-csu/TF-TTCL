#!/usr/bin/env python3
"""CLI entry point for EviMem-TTCL experiments."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.multimodal.client import OpenAIVLMClient
from core.multimodal.datasets import load_local_dataset
from core.multimodal.embeddings import HashMultimodalEmbedder, VLLMMultimodalEmbedder
from core.multimodal.memory import EvidenceMemory
from core.multimodal.runner import EviMemRunner
from core.multimodal.replay import ResponseTraceBackend
from core.multimodal.image_ops import sha256_file


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply dotted ``key=value`` overrides after parsing values as YAML.

    Keeping overrides in the recorded configuration snapshot makes pilot runs
    reproducible without multiplying near-identical configuration files.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected dotted.key=value")
        dotted_key, raw_value = item.split("=", 1)
        keys = [key for key in dotted_key.split(".") if key]
        if not keys:
            raise ValueError(f"Invalid override key in {item!r}")
        target = config
        for key in keys[:-1]:
            current = target.get(key)
            if current is None:
                current = {}
                target[key] = current
            if not isinstance(current, dict):
                raise ValueError(f"Cannot descend into non-mapping override key {key!r}")
            target = current
        target[keys[-1]] = yaml.safe_load(raw_value)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen-VLM EviMem-TTCL")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted config key; repeat for multiple values",
    )
    parser.add_argument("--validate-config", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = apply_overrides(config, args.override)
    experiment = config["experiment"]
    model_config = config["model"]
    memory_config = config.get("memory", {})

    examples = load_local_dataset(
        experiment["data_path"],
        experiment["dataset"],
        experiment.get("image_root"),
        experiment.get("limit"),
        experiment.get("order", "forward"),
        experiment.get("seed", 0),
    )
    samples = [example.sample for example in examples]
    if args.validate_config:
        for sample in samples:
            sample.validate_images()
        print(f"Configuration valid: {len(samples)} inference samples; labels are isolated.")
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    api_key = os.getenv(model_config.get("api_key_env", "OPENAI_API_KEY"), "EMPTY")
    model = OpenAIVLMClient(
        base_url=model_config["api_url"],
        model_name=model_config["model_name"],
        api_key=api_key,
        timeout=float(model_config.get("timeout", 300)),
    )
    if model_config.get("response_trace_dir"):
        model = ResponseTraceBackend(
            model,
            Path(model_config["response_trace_dir"]),
            namespace={
                "dataset": experiment["dataset"],
                "source_sha256": sha256_file(experiment["data_path"]),
                "seed": experiment.get("seed", 0),
                "model_revision": model_config.get("revision"),
            },
            mode=model_config.get("response_trace_mode", "record"),
        )
    embedder_type = memory_config.get("embedder", "hash")
    if embedder_type == "hash":
        embedder = HashMultimodalEmbedder(int(memory_config.get("dimensions", 256)))
    elif embedder_type == "vllm_multimodal":
        embedder = VLLMMultimodalEmbedder(
            base_url=memory_config["api_url"],
            model_name=memory_config["model_name"],
            api_key=os.getenv(memory_config.get("api_key_env", "EMBEDDING_API_KEY"), "EMPTY"),
            timeout=float(memory_config.get("timeout", 300)),
        )
    else:
        raise ValueError(f"Unsupported memory.embedder: {embedder_type}")
    memory = EvidenceMemory(
        embedder,
        max_experiences=int(memory_config.get("max_experiences", 1000)),
        rrf_k=int(memory_config.get("rrf_k", 60)),
        rule_min_similarity=float(memory_config.get("rule_min_similarity", 0.5)),
        episode_min_similarity=float(memory_config.get("episode_min_similarity", 0.75)),
    )
    runner = EviMemRunner(
        model=model,
        memory=memory,
        output_dir=Path(experiment["output_dir"]),
        method=experiment.get("method", "evimem"),
        seed=int(experiment.get("seed", 0)),
        max_tokens=int(model_config.get("max_tokens", 256)),
        request_logprobs=bool(model_config.get("request_logprobs", False)),
        resume=bool(experiment.get("resume", True)),
        config_snapshot=config,
        ablation=config.get("ablation", {}),
    )
    runner.run(samples)


if __name__ == "__main__":
    main()
