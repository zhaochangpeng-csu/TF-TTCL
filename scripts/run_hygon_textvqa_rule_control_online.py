#!/usr/bin/env python3
"""One label-free, independently resumed online rule-control stream."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.textvqa_rule_control_online import POLICIES, TextVQARuleControlRunner
from scripts.create_label_isolated_subset import sha256_file
from scripts.run_hygon_tf_ttcl_direct import EMBEDDING_FILES, EMBEDDING_SHA, MODEL_REVISION, source_fingerprint
from scripts.run_hygon_scienceqa_parallel import verify_weights
from scripts.inspect_baseline_io import read_rows, require_label_free
from core.multimodal.datasets import load_local_dataset

EXCLUSION_SHAS = {
    'e7b3865aa534594634ab9f9c136344d1c2673f317a2578722f09677cc8189b74',
    'f3196d50185da68f150c616825630216fb8d23ebce4678ab26a9ba7b0960f5fd',
    '10b8f9464bc9190a929650e83594b5f0f6a237598e9d9eff47f97f52eb25d978',
    '48d194d3b31c85d8e99336e324bd1e74842bbafb0d9e27d22093593b914754bc',
}


def load_rule_fresh(inference, manifest, manifest_sha, exclusions, count):
    if count not in (20, 400) or not re.fullmatch('[0-9a-f]{64}', manifest_sha) or sha256_file(manifest) != manifest_sha:
        raise ValueError('Unexpected count or unpinned data manifest')
    meta = json.loads(manifest.read_text())
    if ((meta.get('dataset_id'), meta.get('split'), meta.get('count')) !=
            ('official/TextVQA_0.5.1', 'train', 400) or meta.get('inference_sha256') != sha256_file(inference)):
        raise ValueError('Fresh 400-item manifest mismatch')
    rows = read_rows(inference)
    for row in rows:
        require_label_free(row)
    ids = [str(x.get('id', '')) for x in rows]
    images = [str(x.get('image_id', '')) for x in rows]
    if (len(rows) != 400 or len(set(ids)) != 400 or '' in ids or
            len(set(images)) != 400 or '' in images or ids != meta.get('selected_ids')):
        raise ValueError('Expected ordered unique questions and images')
    if len(exclusions) != 4 or {sha256_file(p) for p in exclusions} != EXCLUSION_SHAS:
        raise ValueError('All four pinned earlier streams are required')
    for prior in exclusions:
        old = read_rows(prior)
        for row in old:
            require_label_free(row)
        if set(ids) & {str(x.get('id', '')) for x in old} or set(images) & {str(x.get('image_id', '')) for x in old}:
            raise ValueError('Prior question or image overlap')
    image_dir = (inference.parent / 'images').resolve(strict=True)
    for row, image in zip(rows, images):
        if not re.fullmatch('[A-Za-z0-9_-]+', image) or row.get('image') != f'images/{image}.jpg':
            raise ValueError('Invalid image reference')
        path = (image_dir / (image + '.jpg')).resolve(strict=True)
        if path.parent != image_dir or not path.is_file():
            raise ValueError('Invalid image path')
    digest = hashlib.sha256('\n'.join(f'{image}:{sha256_file(image_dir / (image + ".jpg"))}'
                                        for image in sorted(images)).encode()).hexdigest()
    if digest != meta.get('selected_images_sha256') or meta.get('image_count') != 400:
        raise ValueError('Fresh image hashes mismatch')
    examples = load_local_dataset(str(inference), 'textvqa', str(inference.parent), limit=count)
    if any(e.target is not None for e in examples) or [e.sample.sample_id for e in examples] != ids[:count]:
        raise ValueError('Label exposure or input order mismatch')
    return [e.sample for e in examples], meta


def control_source_fingerprint(project):
    files = ('core/multimodal/textvqa_selector_online.py',
             'core/multimodal/textvqa_rule_control_online.py',
             'core/multimodal/textvqa_rule_gate.py',
             'core/multimodal/textvqa_shadow.py',
             'scripts/run_hygon_textvqa_rule_control_online.py')
    return hashlib.sha256((source_fingerprint(project) + ''.join(
        sha256_file(project / path) for path in files)).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('inference', 'subset-manifest', 'output', 'embedding-path', 'model-path', 'weight-manifest'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--subset-manifest-sha256', required=True)
    p.add_argument('--excluded-inference', type=Path, action='append', required=True)
    p.add_argument('--policy', choices=POLICIES, required=True)
    p.add_argument('--count', choices=(20, 400), type=int, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--base-url', default='http://127.0.0.1:19000/v1')
    p.add_argument('--stack-id', required=True)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    samples, meta = load_rule_fresh(a.inference, a.subset_manifest, a.subset_manifest_sha256,
                                    a.excluded_inference, a.count)
    for name, expected in EMBEDDING_FILES.items():
        if sha256_file(a.embedding_path / name) != expected:
            raise ValueError('Embedding hash mismatch')
    verify_weights(a.model_path, a.weight_manifest)
    if not a.stack_id.strip():
        raise ValueError('Missing stack identity')
    project = Path(__file__).resolve().parents[1]
    fingerprint = control_source_fingerprint(project)
    namespace = {'dataset': 'textvqa', 'input_sha256': meta['inference_sha256'],
                 'subset_manifest_sha256': a.subset_manifest_sha256, 'count': a.count,
                 'selector': 'full_response', 'rule_control_policy': a.policy, 'seed': a.seed,
                 'stack_id': a.stack_id, 'model_revision': MODEL_REVISION,
                 'embedding_sha256': EMBEDDING_SHA, 'source_fingerprint': fingerprint,
                 'weight_manifest_sha256': sha256_file(a.weight_manifest), 'labels_read': False}
    if a.check_only:
        print(json.dumps({'status': 'passed', **namespace}, indent=2)); return
    from transformers import AutoTokenizer
    from core.multimodal.client import OpenAIVLMClient
    from core.multimodal.replay import ResponseTraceBackend
    from core.multimodal.tf_ttcl_text_embedding import Qwen3TextEmbedder
    client = OpenAIVLMClient(a.base_url, 'Qwen/Qwen2.5-VL-7B-Instruct')
    if client.model_name not in [x.id for x in client.client.models.list().data]:
        raise ValueError('Wrong served model')
    output = a.output / a.policy
    backend = ResponseTraceBackend(client, output / 'response_trace', namespace, 'record')
    tokenizer = AutoTokenizer.from_pretrained(str(a.model_path), local_files_only=True)
    embedder = Qwen3TextEmbedder(str(a.embedding_path))
    with patch('core.multimodal.tf_ttcl_direct_runner.subprocess.check_output',
               return_value='archive:' + fingerprint + ':' + a.stack_id):
        runner = TextVQARuleControlRunner(backend, embedder, tokenizer, samples, output,
            meta['inference_sha256'], MODEL_REVISION, EMBEDDING_SHA, seed=a.seed, policy=a.policy)
    runner.run()
    print(json.dumps({'status': 'completed', **namespace}, indent=2))


if __name__ == '__main__': main()
