#!/usr/bin/env python3
"""Run a sharded, label-free fixed-history 2x2 TextVQA shadow replay.

The anchor is already scored, so this is exploratory, not confirmation. The
model sees no evaluation answers and this program never writes online rules.
Each shard commits atomic per-item rows; exact requests are separately cached.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.datasets import load_local_dataset
from core.multimodal.textvqa_shadow import ARMS, replay_item
from core.multimodal.tf_ttcl_direct_runner import _digest
from scripts.audit_tf_ttcl_mm_direct import audit
from scripts.compare_textvqa_format_memory import policy_prompt_hash, verify_online_and_sampling
from scripts.create_label_isolated_subset import sha256_file
from scripts.run_hygon_tf_ttcl_direct import EMBEDDING_FILES, EMBEDDING_SHA, MODEL_REVISION, source_fingerprint
from scripts.run_hygon_scienceqa_parallel import verify_weights


def checked_anchor(anchor, inference, prediction_sha, manifest_sha):
    for path, expected in ((anchor / 'predictions.jsonl', prediction_sha),
                           (anchor / 'manifest.json', manifest_sha)):
        if not re.fullmatch('[0-9a-f]{64}', expected) or sha256_file(path) != expected:
            raise ValueError('Anchor manifest or predictions hash differs')
    manifest, rows = audit(anchor, inference, 400)
    if (manifest.get('format_experiment'), manifest.get('format_policy'),
            manifest.get('prompt_sha256'), manifest.get('model_revision')) != (
            'textvqa_format_factorial_v2', 'short_final', policy_prompt_hash('short_final'), MODEL_REVISION):
        raise ValueError('Need completed frozen short-final anchor')
    if manifest.get('seed') != 0 or manifest.get('embedding_revision') != EMBEDDING_SHA:
        raise ValueError('Anchor seed or embedding differs')
    verify_online_and_sampling(rows, manifest['seed'], 'short_final')
    examples = load_local_dataset(str(inference), 'textvqa', str(inference.parent), limit=400)
    if len(examples) != 400 or any(ex.target is not None for ex in examples):
        raise ValueError('Expected 400 label-free inputs')
    samples = [ex.sample for ex in examples]
    if [s.sample_id for s in samples] != [r['sample_id'] for r in rows]:
        raise ValueError('Input order differs from audited anchor')
    return manifest, rows, samples


def atomic_json(path, value):
    # Never overwrite a prior committed row; a crash during writing only leaves
    # a uniquely named temp file, which is never consumed on resume.
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()


def run_shard(args, config, samples, rows, client, embedder, tokenizer):
    from core.multimodal.replay import ResponseTraceBackend
    root = args.output / f'shard-{args.shard_index:03d}-of-{args.shard_count:03d}'
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = root / 'config.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != config:
                raise ValueError('Refusing incompatible shard resume')
        else:
            atomic_json(manifest, config)
        indices = list(range(args.shard_index, args.limit, args.shard_count))
        trace = ResponseTraceBackend(client, root / 'response_trace', config, 'record')
        for index in indices:
            path = root / f'row-{index:06d}.json'
            if path.exists():
                prior = json.loads(path.read_text())
                if (prior.get('config_sha256') != _digest(config) or
                        prior.get('sample_index') != index or
                        prior.get('sample_id') != samples[index].sample_id or
                        set(prior.get('arms', {})) != set(ARMS) or
                        prior.get('labels_read') is not False or prior.get('rules_written') is not False):
                    raise ValueError('Existing row identity or completeness mismatch')
                continue
            result = replay_item(trace, embedder, tokenizer, samples[index], rows[index], index, 0)
            result['config_sha256'] = _digest(config)
            atomic_json(path, result)
            print(json.dumps({'index': index, 'sample_id': samples[index].sample_id,
                              'shard_index': args.shard_index, 'labels_read': False}), flush=True)
        print(json.dumps({'status': 'completed', 'shard': args.shard_index,
                          'samples': len(indices), 'trace_stats': trace.trace_stats}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('anchor', 'inference', 'output', 'embedding-path', 'model-path', 'weight-manifest'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--anchor-predictions-sha256', required=True)
    parser.add_argument('--anchor-manifest-sha256', required=True)
    parser.add_argument('--shard-count', type=int, required=True)
    parser.add_argument('--shard-index', type=int, required=True)
    parser.add_argument('--limit', type=int, default=400)
    parser.add_argument('--base-url', default='http://127.0.0.1:19000/v1')
    parser.add_argument('--stack-id', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if not (1 <= args.limit <= 400 and 1 <= args.shard_count <= args.limit and
            0 <= args.shard_index < args.shard_count and args.stack_id.strip()):
        parser.error('Invalid shard, limit, or stack ID')
    out = args.output.resolve()
    if any(path.resolve().is_relative_to(out) or out.is_relative_to(path.resolve())
           for path in (args.anchor, args.inference.parent)):
        raise ValueError('Output must be separate from anchor and input folders')
    manifest, rows, samples = checked_anchor(args.anchor, args.inference,
        args.anchor_predictions_sha256, args.anchor_manifest_sha256)
    for filename, expected in EMBEDDING_FILES.items():
        if sha256_file(args.embedding_path / filename) != expected:
            raise ValueError('Embedding artifact mismatch: ' + filename)
    verify_weights(args.model_path, args.weight_manifest)
    project = Path(__file__).resolve().parents[1]
    fingerprint = _digest([source_fingerprint(project), sha256_file(Path(__file__)),
                           sha256_file(project / 'core/multimodal/textvqa_shadow.py')])
    config = {'version': 'textvqa_shadow_fixed_history_v1', 'anchor_predictions_sha256': args.anchor_predictions_sha256,
              'anchor_manifest_sha256': args.anchor_manifest_sha256, 'input_sha256': sha256_file(args.inference),
              'source_fingerprint': fingerprint, 'embedding_sha256': EMBEDDING_SHA,
              'weights_manifest_sha256': sha256_file(args.weight_manifest), 'model_revision': MODEL_REVISION,
              'seed': manifest['seed'], 'limit': args.limit, 'shard_count': args.shard_count,
              'shard_index': args.shard_index, 'stack_id': args.stack_id, 'arms': list(ARMS),
              'labels_read': False, 'rules_written': False}
    if args.check_only:
        print(json.dumps({'status': 'passed', **config}, indent=2))
        return
    from transformers import AutoTokenizer
    from core.multimodal.client import OpenAIVLMClient
    from core.multimodal.tf_ttcl_text_embedding import Qwen3TextEmbedder
    client = OpenAIVLMClient(args.base_url, 'Qwen/Qwen2.5-VL-7B-Instruct')
    if client.model_name not in [model.id for model in client.client.models.list().data]:
        raise ValueError('Wrong served model')
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    embedder = Qwen3TextEmbedder(str(args.embedding_path))
    run_shard(args, config, samples, rows, client, embedder, tokenizer)


if __name__ == '__main__':
    main()
