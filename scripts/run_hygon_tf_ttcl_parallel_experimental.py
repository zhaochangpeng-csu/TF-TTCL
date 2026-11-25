#!/usr/bin/env python3
"""Experimental same-question 7-service ScienceQA smoke; never opens labels.

This is a separate experimental run, not a replacement for the original
sequential baseline. One process must exclusively own its output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.create_label_isolated_subset import sha256_file
from scripts.run_hygon_tf_ttcl_direct import (EMBEDDING_FILES, EMBEDDING_SHA, INPUT_SHA,
    MODEL_REVISION, VERSION, load_checked, source_fingerprint)


def validate_urls(urls):
    if len(urls) != 7 or len(set(urls)) != 7:
        raise ValueError('Expected seven distinct local endpoints')
    for url in urls:
        parsed = urlsplit(url)
        if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or
                not parsed.port or parsed.path != '/v1' or parsed.query or parsed.fragment):
            raise ValueError('Only local HTTP v1 endpoints are accepted')
    return urls


def validate_idle_devices(output):
    """Require successful explicit zero memory and utilization for HCU 1–7."""
    fields = {'memory use (%)': {}, 'HCU use (%)': {}}
    for index, name, value in re.findall(
            r'HCU\[(\d+)\]\s*:\s*(HCU use \(%\)|HCU memory use \(%\)):\s*([0-9.]+)', output):
        key = 'memory use (%)' if name.startswith('HCU memory') else name
        if int(index) in fields[key]:
            raise ValueError('Duplicate device status')
        fields[key][int(index)] = float(value)
    for name, values in fields.items():
        if any(index not in values or values[index] != 0 for index in range(1, 8)):
            raise ValueError(f'HCUs 1–7 not all idle according to {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inference', 'reference', 'subset-manifest', 'output', 'embedding-path'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--base-url', action='append', default=[])
    parser.add_argument('--stack-id', required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--check-devices', action='store_true')
    args = parser.parse_args()
    urls = validate_urls(args.base_url)
    if not args.stack_id.strip():
        raise ValueError('A nonempty immutable stack ID is required')
    if args.check_devices:
        validate_idle_devices(subprocess.check_output(
            ['/opt/hyhal/bin/hy-smi', '--showmemuse', '--showuse'], text=True))
    if (args.output / 'manifest.json').exists() or (args.output / 'predictions.jsonl').exists():
        raise FileExistsError('Experimental smoke refuses reuse of an output stream')
    samples = load_checked(args.inference, args.reference, args.subset_manifest, 20)
    for name, digest in EMBEDDING_FILES.items():
        if sha256_file(args.embedding_path / name) != digest:
            raise ValueError(f'Embedding artifact mismatch: {name}')
    project = Path(__file__).resolve().parents[1]
    fingerprint = source_fingerprint(project)
    entry_sha = sha256_file(Path(__file__))
    context = {'experimental': True, 'source_fingerprint': fingerprint,
               'entry_sha256': entry_sha, 'stack_id': args.stack_id, 'base_urls': urls}
    if args.check_only:
        print(json.dumps({'status': 'passed', 'count': 20, 'labels_read': False, **context}))
        return
    from core.multimodal.client import OpenAIVLMClient
    from core.multimodal.replay import ResponseTraceBackend
    from core.multimodal.tf_ttcl_parallel_runner import ParallelDirectRunner, RoutedBackend
    from core.multimodal.tf_ttcl_text_embedding import Qwen3TextEmbedder
    model_name = 'Qwen/Qwen2.5-VL-7B-Instruct'
    clients = [OpenAIVLMClient(url, model_name) for url in urls]
    for url, client in zip(urls, clients):
        if model_name not in [model.id for model in client.client.models.list().data]:
            raise ValueError(f'Wrong model at {url}')
    stack = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    namespace = {'dataset': 'scienceqa_img', 'source_sha256': INPUT_SHA,
                 'ordered_ids_sha256': hashlib.sha256('\n'.join(s.sample_id for s in samples).encode()).hexdigest(),
                 'seed': 0, 'model_revision': MODEL_REVISION, 'protocol': VERSION,
                 'trace_version': 'four_arm_shared_requests_v1', 'stack_id': stack,
                 'source_fingerprint': fingerprint, 'experimental_parallel': True}
    backend = ResponseTraceBackend(RoutedBackend(clients), args.output / 'response_trace', namespace, 'record')
    embedder = Qwen3TextEmbedder(str(args.embedding_path))
    identity = f'experimental-archive:{fingerprint}:{stack}'
    with patch('core.multimodal.tf_ttcl_direct_runner.subprocess.check_output', return_value=identity):
        runner = ParallelDirectRunner(backend, embedder, samples, args.output, INPUT_SHA,
                                      MODEL_REVISION, EMBEDDING_SHA, 0, max_workers=7)
    runner.run()
    print(json.dumps({'status': 'completed', 'count': 20, 'labels_read': False, **context}))


if __name__ == '__main__':
    main()
