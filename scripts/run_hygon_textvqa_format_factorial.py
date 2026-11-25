#!/usr/bin/env python3
"""Run one label-free TextVQA format policy on its own online memory stream.

Run each policy with a distinct local service. Output and immutable response
traces are separated by policy; no request or rule cache is shared across the
two streams. Labels are not opened by this inference program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.datasets import load_local_dataset
from core.multimodal.textvqa_format_factorial import POLICIES, TextVQAFormatRunner
from core.multimodal.tf_ttcl_direct_runner import OPEN_VERSION
from scripts.create_label_isolated_subset import sha256_file
from scripts.inspect_baseline_io import require_label_free
from scripts.run_hygon_tf_ttcl_direct import EMBEDDING_FILES, EMBEDDING_SHA, MODEL_REVISION, source_fingerprint

EXCLUDED_SHA_COUNTS = {
    'e7b3865aa534594634ab9f9c136344d1c2673f317a2578722f09677cc8189b74': 400,
    '10b8f9464bc9190a929650e83594b5f0f6a237598e9d9eff47f97f52eb25d978': 5000,
}


def digest_ids(ids):
    return hashlib.sha256('\n'.join(ids).encode()).hexdigest()


def load_checked(inference: Path, manifest: Path, expected_manifest_sha: str,
                 excludes: list[Path], count: int, *, excluded_sha_counts=EXCLUDED_SHA_COUNTS):
    if count not in (20, 400) or len(excludes) != len(excluded_sha_counts):
        raise ValueError('Need 20/400 samples and all required prior inference exclusions')
    if not re.fullmatch(r'[a-f0-9]{64}', expected_manifest_sha):
        raise ValueError('Manifest SHA-256 must be provided in lowercase hex')
    if sha256_file(manifest) != expected_manifest_sha:
        raise ValueError('Frozen subset manifest hash mismatch')
    meta = json.loads(manifest.read_text(encoding='utf-8'))
    if (meta.get('dataset_id'), meta.get('split'), meta.get('seed'), meta.get('count')) != (
            'official/TextVQA_0.5.1', 'train', 20261126, 400):
        raise ValueError('Expected frozen 400-question official TextVQA train export')
    if sha256_file(inference) != meta.get('inference_sha256'):
        raise ValueError('Inference hash does not match frozen manifest')
    rows = [json.loads(line) for line in inference.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows) != 400:
        raise ValueError('Expected all 400 inference records')
    for row in rows:
        require_label_free(row)
    ids = [str(row.get('id', '')) for row in rows]
    image_ids = [str(row.get('image_id', '')) for row in rows]
    if (any(not value for value in ids) or len(set(ids)) != 400 or ids != meta.get('selected_ids')
            or digest_ids(ids) != meta.get('selected_ids_sha256')):
        raise ValueError('Frozen ordered question IDs mismatch')
    image_dir = (inference.parent / 'images').resolve(strict=True)
    for row, image_id in zip(rows, image_ids):
        if (not image_id or not re.fullmatch(r'[A-Za-z0-9_-]+', image_id)
                or row.get('image') != f'images/{image_id}.jpg'):
            raise ValueError('Malformed or noncanonical image reference')
        path = (image_dir / f'{image_id}.jpg').resolve(strict=True)
        if path.parent != image_dir or not path.is_file():
            raise ValueError('Image escapes the frozen input directory')
    unique_images = sorted(set(image_ids))
    image_digest = digest_ids([f'{name}:{sha256_file(image_dir / (name + ".jpg"))}' for name in unique_images])
    if len(unique_images) != 400 or len(unique_images) != meta.get('image_count') or image_digest != meta.get('selected_images_sha256'):
        raise ValueError('Frozen image count or hashes mismatch')
    excluded_hashes = []
    for path in excludes:
        prior_hash = sha256_file(path)
        if prior_hash not in excluded_sha_counts or prior_hash in excluded_hashes:
            raise ValueError('Unknown or repeated prior inference exclusion')
        prior = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(prior) != excluded_sha_counts[prior_hash]:
            raise ValueError('Prior inference exclusion count mismatch')
        for row in prior:
            require_label_free(row)
        prior_ids = {str(row.get('id', '')) for row in prior}
        prior_images = {str(row.get('image_id', '')) for row in prior}
        if '' in prior_ids or '' in prior_images:
            raise ValueError('Prior inference needs question and image IDs')
        if set(ids) & prior_ids or set(image_ids) & (prior_images - {''}):
            raise ValueError('Question or image overlaps a prior development stream')
        excluded_hashes.append(prior_hash)
    examples = load_local_dataset(str(inference), 'textvqa', str(inference.parent), limit=count)
    if any(example.target is not None for example in examples):
        raise ValueError('Inference object exposed a target')
    samples = [example.sample for example in examples]
    if [sample.sample_id for sample in samples] != ids[:count]:
        raise ValueError('Dataset adapter changed stream ordering')
    for sample in samples:
        sample.validate_images()
    return samples, meta, sorted(excluded_hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inference', 'subset-manifest', 'output', 'embedding-path'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--excluded-inference', type=Path, action='append', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--policy', required=True, choices=POLICIES)
    parser.add_argument('--count', type=int, required=True, choices=(20, 400))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--base-url', default='http://127.0.0.1:19000/v1')
    parser.add_argument('--stack-id', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    samples, meta, excluded_hashes = load_checked(args.inference, args.subset_manifest,
                                                   args.manifest_sha256, args.excluded_inference, args.count)
    for name, expected in EMBEDDING_FILES.items():
        if sha256_file(args.embedding_path / name) != expected:
            raise ValueError(f'Embedding artifact mismatch: {name}')
    if not args.stack_id.strip():
        raise ValueError('A nonempty immutable stack ID is required')
    project = Path(__file__).resolve().parents[1]
    fingerprint = hashlib.sha256((source_fingerprint(project) + sha256_file(Path(__file__))).encode()).hexdigest()
    config = {'dataset': 'textvqa', 'input_sha256': meta['inference_sha256'],
              'manifest_sha256': args.manifest_sha256, 'selected_images_sha256': meta['selected_images_sha256'],
              'excluded_inference_sha256': excluded_hashes,
              'ordered_ids_sha256': digest_ids([sample.sample_id for sample in samples]),
              'count': args.count, 'seed': args.seed, 'policy': args.policy,
              'model_revision': MODEL_REVISION, 'embedding_sha256': EMBEDDING_SHA,
              'protocol': OPEN_VERSION, 'stack_id': args.stack_id, 'source_fingerprint': fingerprint}
    if args.check_only:
        print(json.dumps({'status': 'passed', 'labels_read': False, **config}, indent=2))
        return
    from core.multimodal.client import OpenAIVLMClient
    from core.multimodal.replay import ResponseTraceBackend
    from core.multimodal.tf_ttcl_text_embedding import Qwen3TextEmbedder
    model = 'Qwen/Qwen2.5-VL-7B-Instruct'
    client = OpenAIVLMClient(args.base_url, model)
    if model not in [entry.id for entry in client.client.models.list().data]:
        raise ValueError('Wrong served model')
    output = args.output / args.policy
    trace = ResponseTraceBackend(client, output / 'response_trace', config, 'record')
    embedder = Qwen3TextEmbedder(str(args.embedding_path))
    identity = f'archive:{fingerprint}:{hashlib.sha256(args.stack_id.encode()).hexdigest()}'
    with patch('core.multimodal.tf_ttcl_direct_runner.subprocess.check_output', return_value=identity):
        runner = TextVQAFormatRunner(trace, embedder, samples, output, meta['inference_sha256'],
                                     MODEL_REVISION, EMBEDDING_SHA, seed=args.seed, policy=args.policy)
    runner.run()
    print(json.dumps({'status': 'completed', 'labels_read': False, 'archive_identity': identity,
                      'trace_stats': trace.trace_stats, **config}, indent=2))


if __name__ == '__main__':
    main()
