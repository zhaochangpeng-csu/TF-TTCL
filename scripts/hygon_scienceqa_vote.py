#!/usr/bin/env python3
"""Replay ScienceQA's no-rewrite five-candidate selection without labels."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.answers import normalize_answer
from core.multimodal.client import OpenAIVLMClient
from core.multimodal.datasets import load_local_dataset
from core.multimodal.schema import GenerationResponse, SamplingConfig
from core.multimodal.tf_ttcl_direct_runner import STUDENT_SYSTEM
from core.multimodal.tf_ttcl_direct_selection import ScoredCandidate, select_closed, strict_generation_ppl
from scripts.hygon_scienceqa_sampled_probe import checked_call
from scripts.inspect_baseline_io import require_label_free


def candidate(role, response, sample):
    answer = normalize_answer(response.content, sample.dataset,
                              {**sample.metadata, 'choices': sample.choices})
    if answer not in {chr(97 + j) for j in range(len(sample.choices))}:
        answer = f'__unparsed_{role}__'
    return ScoredCandidate.from_response(role, answer, response)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inference', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--greedy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-url', required=True)
    p.add_argument('--count', type=int, default=400)
    p.add_argument('--worker', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    source = [json.loads(line) for line in a.inference.read_text().splitlines() if line.strip()]
    refs = [json.loads(line) for line in a.reference.read_text().splitlines() if line.strip()]
    greedy = [json.loads(line) for line in a.greedy.read_text().splitlines() if line.strip()]
    examples = load_local_dataset(str(a.inference), 'scienceqa_img', str(a.inference.parent))
    for row in source:
        require_label_free(row)
    if (len(source), len(refs), len(greedy), len(examples)) != (400, 400, 400, 400):
        raise ValueError('Expected four aligned 400-item streams')
    if not 0 <= a.worker < a.workers or not 1 <= a.count <= 400:
        raise ValueError('Invalid worker or count')
    calls = []
    for index in range(a.worker, a.count, a.workers):
        ref, grow = refs[index], greedy[index]
        sample, old_student, user, _, hashes, _ = checked_call(index, examples[index], ref,
                                                                a.inference.parent / 'images')
        if (grow['sample_index'], grow['sample_id'], grow.get('labels_read'), grow['image_hashes']) != (
                index, sample.sample_id, False, hashes):
            raise ValueError(f'Greedy stream mismatch at {index}')
        response = GenerationResponse(**grow['response'])
        teacher = candidate('teacher', response, sample)
        teacher_answer = '' if teacher.answer.startswith('__unparsed_') else teacher.answer
        if (grow['answer'] != teacher_answer or grow['reference_answer'] != ref['arms']['greedy']['answer']
                or not math.isclose(teacher.ppl, grow['ppl'], rel_tol=1e-9)):
            raise ValueError(f'Greedy response mismatch at {index}')
        old_calls = [c for c in ref['arms']['vote_no_rewrite']['calls'] if c.get('role')]
        if [c['role'] for c in old_calls] != ['teacher', 'student_1', 'student_2', 'student_3', 'student_4']:
            raise ValueError(f'Original candidate roles mismatch at {index}')
        for n, old in enumerate(old_calls[1:], 1):
            sampling = SamplingConfig(temperature=0.7, top_p=0.9, max_tokens=256,
                                      seed=100 * index + n + 1, request_logprobs=True)
            if (old['system'] != STUDENT_SYSTEM or old['user'] != user
                    or old['sampling'] != asdict(sampling) or old['image_hashes'] != hashes):
                raise ValueError(f'Original student configuration mismatch at {index}/{n}')
            if not math.isclose(strict_generation_ppl(GenerationResponse(**old['response'])),
                                old['ppl'], rel_tol=1e-9):
                raise ValueError(f'Original student PPL mismatch at {index}/{n}')
        calls.append((index, sample, ref, grow, user, hashes, teacher))
    if a.check_only:
        print(json.dumps({'status': 'passed', 'checked': len(calls), 'labels_read': False}))
        return
    client = OpenAIVLMClient(a.base_url, 'Qwen/Qwen2.5-VL-7B-Instruct')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as out:
        for index, sample, ref, grow, user, hashes, teacher in calls:
            candidates = [teacher]
            students = []
            for n in range(1, 5):
                role = f'student_{n}'
                sampling = SamplingConfig(temperature=0.7, top_p=0.9, max_tokens=256,
                                          seed=100 * index + n + 1, request_logprobs=True)
                response = client.generate(STUDENT_SYSTEM, user, sample.image_paths, sampling)
                scored = candidate(role, response, sample)
                candidates.append(scored)
                students.append({'role': role, 'answer': scored.answer, 'ppl': scored.ppl,
                                 'sampling': asdict(sampling), 'response': asdict(response)})
            selection = select_closed(candidates)
            answer = '' if selection.positive.answer.startswith('__unparsed_') else selection.positive.answer
            row = {'sample_index': index, 'sample_id': sample.sample_id, 'labels_read': False,
                   'image_hashes': hashes, 'teacher': grow, 'students': students,
                   'answer': answer, 'positive_role': selection.positive.role,
                   'negative_role': selection.negative.role if selection.negative else None,
                   'positive_roles': selection.positive_roles, 'negative_roles': selection.negative_roles,
                   'reason': selection.reason,
                   'reference_answer': ref['arms']['vote_no_rewrite']['answer']}
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            out.flush()
            print(json.dumps({'index': index, 'answer': answer,
                              'same_as_reference': answer == row['reference_answer']}), flush=True)


if __name__ == '__main__':
    main()
