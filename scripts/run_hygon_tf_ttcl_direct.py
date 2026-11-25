#!/usr/bin/env python3
"""Run the unmodified online DirectRunner on the frozen ScienceQA HCU stream.

The input, reference and manifest are inference-only files. This entry point
never opens a label file. It maps the original absolute image references into
the new machine only after verifying the content hashes against the original
NVIDIA predictions. Each output directory is a separate ordered memory stream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.datasets import load_local_dataset
from core.multimodal.prompts import format_question
from core.multimodal.schema import SamplingConfig
from core.multimodal.tf_ttcl_direct_runner import DirectRunner, TEACHER_SYSTEM, VERSION
from scripts.create_label_isolated_subset import sha256_file
from scripts.inspect_baseline_io import require_label_free

INPUT_SHA = '01c6dc83958ea1cee51ab38594b2499e3bfbc4034139674dc799e5dfeb354689'
REFERENCE_SHA = 'b8ac87c5bdf38923f7ec5e9e848b3f1b18e0f183b2ea0deffea98700ffaf4d71'
EMBEDDING_SHA = '0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd'
EMBEDDING_FILES = {
    'model.safetensors': EMBEDDING_SHA,
    'config.json': 'b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9',
    'tokenizer.json': 'def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a',
    'tokenizer_config.json': '253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0',
    'merges.txt': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5',
    'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
}
MODEL_REVISION = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'
OLD_IMAGE_ROOT = Path('/home/lthpc/private/code/zcp/TTT/TF-TTCL/data/multimodal/scienceqa_modelscope_train/images')


def checked_sample(index, example, ref, image_root, old_root=OLD_IMAGE_ROOT):
    sample = example.sample
    if example.target is not None or ref.get('labels_read') is not False or (
            ref.get('sample_index'), ref.get('sample_id')) != (index, sample.sample_id):
        raise ValueError(f'Label boundary or sample order mismatch: {index}')
    images = []
    for image in sample.image_paths:
        old_path = Path(image)
        if old_path.parent != old_root or old_path.name in ('', '.', '..'):
            raise ValueError(f'Unexpected old image path: {old_path}')
        new_path = (image_root / old_path.name).resolve()
        if new_path.parent != image_root.resolve():
            raise ValueError('Image path escapes approved image root')
        images.append(str(new_path))
    sample = replace(sample, image_paths=tuple(images))
    sample.validate_images()
    hashes = [sha256_file(Path(path)) for path in images]
    call = ref['arms']['greedy']['calls'][0]
    sampling = SamplingConfig(temperature=0.0, max_tokens=256, seed=100 * index, request_logprobs=True)
    if (not hashes or call['system'] != TEACHER_SYSTEM or
            call['user'] != format_question(sample.question, sample.choices, sample.metadata) or
            call['sampling'] != asdict(sampling) or call['image_hashes'] != hashes or
            ref['image_hashes'] != hashes):
        raise ValueError(f'Original prompt, sampling, or image hash mismatch: {index}')
    return sample


def load_checked(inference, reference, manifest_path, count, *, input_sha=INPUT_SHA, reference_sha=REFERENCE_SHA):
    if count not in (20, 400):
        raise ValueError('Only 20 or 400 samples are supported')
    if sha256_file(inference) != input_sha or sha256_file(reference) != reference_sha:
        raise ValueError('Input or original reference hash mismatch')
    rows = [json.loads(line) for line in inference.read_text().splitlines() if line.strip()]
    refs = [json.loads(line) for line in reference.read_text().splitlines() if line.strip()]
    meta = json.loads(manifest_path.read_text())
    if (meta.get('seed'), meta.get('offset'), meta.get('count')) != (20261019, 0, 400):
        raise ValueError('Unexpected ScienceQA subset manifest')
    if meta.get('excluded_id_count', 0) < 2500:
        raise ValueError('Previously used IDs were not excluded')
    if len(rows) != 400 or len(refs) != 400 or [str(r['id']) for r in rows] != meta['selected_ids']:
        raise ValueError('Frozen sample IDs do not match manifest')
    for row in rows:
        require_label_free(row)
    examples = load_local_dataset(str(inference), 'scienceqa_img', str(inference.parent))
    if len({item.sample.sample_id for item in examples}) != 400:
        raise ValueError('Duplicated sample ID')
    return [checked_sample(i, item, refs[i], inference.parent / 'images')
            for i, item in enumerate(examples[:count])]


def source_fingerprint(project: Path) -> str:
    files = sorted((project / 'core/multimodal').glob('*.py'))
    files += sorted((project / 'core/source/prompts').rglob('*.md'))
    files += [project / 'scripts/run_hygon_tf_ttcl_direct.py',
              project / 'scripts/inspect_baseline_io.py',
              project / 'scripts/create_label_isolated_subset.py']
    h = hashlib.sha256()
    for path in files:
        h.update(str(path.relative_to(project)).encode() + b'\0')
        h.update(bytes.fromhex(sha256_file(path)))
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('inference', 'reference', 'subset-manifest', 'output', 'embedding-path'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--count', type=int, choices=(20, 400), required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:19000/v1')
    p.add_argument('--stack-id', required=True, help='Immutable environment/model-service fingerprint for resume')
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    samples = load_checked(args.inference, args.reference, args.subset_manifest, args.count)
    for filename, expected in EMBEDDING_FILES.items():
        if sha256_file(args.embedding_path / filename) != expected:
            raise ValueError(f'Embedding artifact hash differs from original: {filename}')
    project = Path(__file__).resolve().parents[1]
    fingerprint = source_fingerprint(project)
    if args.check_only:
        print(json.dumps({'status': 'passed', 'count': len(samples), 'labels_read': False,
                          'source_fingerprint': fingerprint, 'embedding_sha256': EMBEDDING_SHA}))
        return
    if not args.stack_id.strip():
        raise ValueError('A nonempty immutable stack ID is required')
    from core.multimodal.client import OpenAIVLMClient
    from core.multimodal.replay import ResponseTraceBackend
    from core.multimodal.tf_ttcl_text_embedding import Qwen3TextEmbedder
    model_name = 'Qwen/Qwen2.5-VL-7B-Instruct'
    base = OpenAIVLMClient(args.base_url, model_name)
    if model_name not in [x.id for x in base.client.models.list().data]:
        raise ValueError('Wrong served model')
    namespace = {'dataset': 'scienceqa_img', 'source_sha256': INPUT_SHA,
                 'ordered_ids_sha256': hashlib.sha256('\n'.join(s.sample_id for s in samples).encode()).hexdigest(),
                 'seed': 0, 'model_revision': MODEL_REVISION, 'protocol': VERSION,
                 'trace_version': 'four_arm_shared_requests_v1', 'stack_id': args.stack_id,
                 'source_fingerprint': fingerprint}
    backend = ResponseTraceBackend(base, args.output / 'response_trace', namespace, 'record')
    embedder = Qwen3TextEmbedder(str(args.embedding_path))
    # The server copy is a git archive without .git. An explicit archive digest
    # replaces git identity, so DirectRunner's existing resume check still locks
    # source bytes and execution stack rather than claiming an unverifiable SHA.
    identity = f'archive:{fingerprint}:{hashlib.sha256(args.stack_id.encode()).hexdigest()}'
    with patch('core.multimodal.tf_ttcl_direct_runner.subprocess.check_output', return_value=identity):
        runner = DirectRunner(backend, embedder, samples, args.output, INPUT_SHA,
                              MODEL_REVISION, EMBEDDING_SHA, 0)
    runner.run()
    print(json.dumps({'status': 'completed', 'count': len(samples), 'labels_read': False,
                      'stack_id': args.stack_id, 'archive_identity': identity,
                      'trace_stats': backend.trace_stats}))


if __name__ == '__main__':
    main()
