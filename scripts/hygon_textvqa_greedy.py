#!/usr/bin/env python3
"""Replay the original TextVQA greedy calls on a second inference stack.

This program opens no labels. Each worker writes a distinct JSONL shard; an
offline scorer may compare the completed shards with labels afterward.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.answers import normalize_answer
from core.multimodal.client import OpenAIVLMClient
from core.multimodal.datasets import load_local_dataset
from core.multimodal.image_ops import sha256_file
from core.multimodal.prompts import format_question
from core.multimodal.schema import SamplingConfig
from core.multimodal.tf_ttcl_direct_selection import strict_generation_ppl
from core.multimodal.tf_ttcl_direct_runner import TEACHER_SYSTEM_OPEN
from scripts.inspect_baseline_io import require_label_free


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--count', type=int, default=400)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--probe-sampling', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256_file(str(args.inference)) != 'e7b3865aa534594634ab9f9c136344d1c2673f317a2578722f09677cc8189b74':
        raise ValueError('Input hash mismatch')
    if sha256_file(str(args.reference)) != '4750646aa363532c6169f98c332e0e3ef32dc1e04ec13b8feb729b3874e762c4':
        raise ValueError('Reference hash mismatch')
    source = [json.loads(line) for line in args.inference.read_text().splitlines()]
    for row in source:
        require_label_free(row)
    examples = load_local_dataset(str(args.inference), 'textvqa', str(args.inference.parent))
    refs = [json.loads(line) for line in args.reference.read_text().splitlines()]
    if len(source) != 400 or len(examples) != 400 or len(refs) != 400 or not 0 <= args.worker < args.workers or not 1 <= args.count <= 400:
        raise ValueError('Expected 400 aligned records and a valid worker/count')
    def checked_call(index):
        item, ref = examples[index], refs[index]
        sample = item.sample
        if any(not Path(path).resolve().is_relative_to((args.inference.parent / 'images').resolve())
               for path in sample.image_paths):
            raise ValueError('Image outside approved input directory')
        if item.target is not None or sample.sample_id != ref['sample_id'] or ref['sample_index'] != index:
            raise ValueError(f'Label boundary or reference order mismatch at {index}')
        sample.validate_images()
        old = ref['arms']['greedy']['calls'][0]
        user = format_question(sample.question, sample.choices, sample.metadata)
        sampling = SamplingConfig(temperature=0.0, max_tokens=256, seed=100 * index, request_logprobs=True)
        hashes = [sha256_file(path) for path in sample.image_paths]
        if (not hashes or old['system'] != TEACHER_SYSTEM_OPEN or old['user'] != user
                or old['sampling'] != asdict(sampling) or old['image_hashes'] != hashes):
            raise ValueError(f'Input or image hash mismatch at {index}')
        return sample, ref, user, sampling, hashes
    if args.check_only:
        for index in range(args.worker, args.count, args.workers):
            checked_call(index)
        print(json.dumps({'status': 'passed', 'checked': len(range(args.worker, args.count, args.workers)),
                          'labels_read': False}))
        return
    client = OpenAIVLMClient(args.base_url, 'Qwen/Qwen2.5-VL-7B-Instruct')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with args.output.open('x', encoding='utf-8') as out:
        if args.probe_sampling:
            sample, _, user, _, _ = checked_call(0)
            probe = client.generate(TEACHER_SYSTEM_OPEN, user, sample.image_paths,
                                    SamplingConfig(temperature=0.7, top_p=0.9, max_tokens=128,
                                                   seed=1, request_logprobs=True))
            print(json.dumps({'sampled_probe_ppl': strict_generation_ppl(probe),
                              'completion_tokens': probe.completion_tokens,
                              'logprob_count': len(probe.logprobs)}), flush=True)
        for index in range(args.worker, args.count, args.workers):
            sample, ref, user, sampling, hashes = checked_call(index)
            response = client.generate(TEACHER_SYSTEM_OPEN, user, sample.image_paths, sampling)
            ppl = strict_generation_ppl(response)
            answer = normalize_answer(response.content, sample.dataset, {**sample.metadata, 'choices': sample.choices})
            result = {'sample_index': index, 'sample_id': sample.sample_id, 'image_hashes': hashes,
                      'answer': answer, 'reference_answer': ref['arms']['greedy']['answer'],
                      'response': asdict(response), 'ppl': ppl, 'labels_read': False}
            out.write(json.dumps(result, ensure_ascii=False) + '\n')
            out.flush()
            print(json.dumps({'index': index, 'match': answer == result['reference_answer'],
                              'elapsed_seconds': round(time.monotonic() - start, 2)}), flush=True)


if __name__ == '__main__':
    main()
