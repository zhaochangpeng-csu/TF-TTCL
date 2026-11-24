#!/usr/bin/env python3
"""Replay original sampled students on Hygon without reading evaluation labels.

Sampling with identical seeds across different stacks is not guaranteed to
produce identical tokens, so answer/PPL differences are descriptive only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.answers import normalize_answer
from core.multimodal.client import OpenAIVLMClient
from core.multimodal.datasets import load_local_dataset
from core.multimodal.image_ops import sha256_file
from core.multimodal.prompts import format_question
from core.multimodal.schema import GenerationResponse, SamplingConfig
from core.multimodal.tf_ttcl_direct_runner import STUDENT_SYSTEM
from core.multimodal.tf_ttcl_direct_selection import strict_generation_ppl
from scripts.inspect_baseline_io import require_label_free

OLD_ROOT = Path('/home/lthpc/private/code/zcp/TTT/TF-TTCL/data/multimodal/scienceqa_modelscope_train/images')


def checked_call(index, item, reference, image_root):
    sample = item.sample
    if item.target is not None or reference['sample_index'] != index or reference['sample_id'] != sample.sample_id or reference.get('labels_read') is not False:
        raise ValueError(f'Label boundary or reference order mismatch at {index}')
    new_paths = []
    for value in sample.image_paths:
        old_path = Path(value)
        if old_path.parent != OLD_ROOT:
            raise ValueError(f'Unexpected image path: {old_path}')
        new_paths.append(str((image_root / old_path.name).resolve()))
    sample = replace(sample, image_paths=tuple(new_paths))
    sample.validate_images()
    calls = [call for call in reference['arms']['vote_no_rewrite']['calls'] if call.get('role') == 'student_1']
    if len(calls) != 1:
        raise ValueError(f'Expected one sampled student_1 at {index}')
    old = calls[0]
    user = format_question(sample.question, sample.choices, sample.metadata)
    sampling = SamplingConfig(temperature=0.7, top_p=0.9, max_tokens=256,
                              seed=100 * index + 2, request_logprobs=True)
    hashes = [sha256_file(path) for path in sample.image_paths]
    if (not hashes or old['system'] != STUDENT_SYSTEM or old['user'] != user or
            old['sampling'] != asdict(sampling) or old['image_hashes'] != hashes):
        raise ValueError(f'Prompt, sampling, or image hash mismatch at {index}')
    old_ppl = strict_generation_ppl(GenerationResponse(**old['response']))
    if not math.isclose(old_ppl, old['ppl'], rel_tol=1e-9):
        raise ValueError(f'Original PPL mismatch at {index}')
    return sample, old, user, sampling, hashes, old_ppl


def run(inference, reference, output, base_url, count=20, check_only=False):
    if output.exists():
        raise FileExistsError(output)
    source = [json.loads(line) for line in inference.read_text().splitlines() if line.strip()]
    refs = [json.loads(line) for line in reference.read_text().splitlines() if line.strip()]
    for row in source:
        require_label_free(row)
    examples = load_local_dataset(str(inference), 'scienceqa_img', str(inference.parent))
    if len(source) != 400 or len(examples) != 400 or len(refs) != 400 or not 1 <= count <= 400:
        raise ValueError('Expected the 400-record stream and a valid count')
    calls = [checked_call(i, examples[i], refs[i], inference.parent / 'images') for i in range(count)]
    if check_only:
        return {'status': 'passed', 'count': count, 'labels_read': False}
    client = OpenAIVLMClient(base_url, 'Qwen/Qwen2.5-VL-7B-Instruct')
    output.parent.mkdir(parents=True, exist_ok=True)
    matches = 0
    with output.open('x', encoding='utf-8') as handle:
        for i, (sample, old, user, sampling, hashes, old_ppl) in enumerate(calls):
            response = client.generate(STUDENT_SYSTEM, user, sample.image_paths, sampling)
            ppl = strict_generation_ppl(response)
            answer = normalize_answer(response.content, sample.dataset, {**sample.metadata, 'choices': sample.choices})
            if answer not in {chr(97 + j) for j in range(len(sample.choices))}:
                answer = ''
            old_answer = old['normalized_answer']
            if old_answer.startswith('__unparsed_'):
                old_answer = ''
            matches += answer == old_answer
            row = {'sample_index': i, 'sample_id': sample.sample_id, 'image_hashes': hashes,
                   'answer': answer, 'reference_answer': old_answer, 'ppl': ppl,
                   'reference_ppl': old_ppl, 'response': asdict(response), 'labels_read': False}
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
    return {'status': 'completed', 'count': count, 'answer_matches': matches,
            'answer_match_rate': matches / count, 'labels_read': False,
            'note': 'Sampled outputs across inference stacks are not expected to be seed-identical.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inference', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-url', required=True)
    p.add_argument('--count', type=int, default=20)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    print(json.dumps(run(a.inference, a.reference, a.output, a.base_url, a.count, a.check_only), indent=2))


if __name__ == '__main__':
    main()
