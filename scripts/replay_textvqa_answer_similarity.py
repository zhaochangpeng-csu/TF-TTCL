#!/usr/bin/env python3
"""Label-free fixed-candidate diagnostic, not an online memory intervention.

Repartition already-generated candidates by the recorded answer similarities.
All rules and candidates remain fixed; this cannot estimate a new memory stream.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.answers import normalize_answer
from core.multimodal.datasets import load_local_dataset
from core.multimodal.image_ops import sha256_file
from core.multimodal.prompts import format_question
from core.multimodal.schema import GenerationResponse
from core.multimodal.tf_ttcl_direct_selection import ScoredCandidate, select_open
from scripts.audit_tf_ttcl_mm_direct import audit
from scripts.compare_textvqa_format_memory import verify_online_and_sampling
from scripts.inspect_baseline_io import read_rows, require_label_free

ARMS = ('select_no_memory', 'tf_ttcl_mm_direct')
ROLES = ('teacher', 'student_1', 'student_2', 'student_3', 'student_4')


def request_key(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def replay_candidate_arm(row, sample, arm_name, trace_dir, namespace, model):
    if arm_name not in ARMS or row['dataset'] != 'textvqa':
        raise ValueError('Only TextVQA original candidate arms are supported')
    arm = row['arms'][arm_name]
    calls = [c for c in arm['calls'] if c.get('role')]
    if [c['role'] for c in calls] != list(ROLES):
        raise ValueError('Candidate roles/order mismatch')
    if row['original_question'] != sample.question or row['choices'] != list(sample.choices):
        raise ValueError('Source question or choices differ')
    image_hashes = [sha256_file(path) for path in sample.image_paths]
    if image_hashes != row['image_hashes']:
        raise ValueError('Actual image differs from recorded image')
    candidates, trace_keys = [], []
    for index, call in enumerate(calls):
        question = sample.question if index == 0 else row['variants'][index - 1]
        if (call['question_variant'] != question or
                call['user'] != format_question(question, sample.choices, sample.metadata) or
                call['image_paths'] != list(sample.image_paths) or call['image_hashes'] != image_hashes):
            raise ValueError('Recorded candidate input differs from source or variant')
        sampling = dict(call['sampling'])
        if sampling.get('output_schema') is None:
            sampling.pop('output_schema', None)
        request = {'namespace': namespace, 'model': model, 'system_prompt': call['system'],
                   'user_text': call['user'], 'image_hashes': image_hashes, 'sampling': sampling}
        key = request_key(request)
        trace = json.loads((trace_dir / (key + '.json')).read_text())
        if trace.get('request') != request or trace.get('response') != call['response']:
            raise ValueError('Immutable request/response trace mismatch')
        response = GenerationResponse(**call['response'])
        answer = normalize_answer(response.content, 'textvqa', {**sample.metadata, 'choices': sample.choices})
        if not answer:
            answer = '__unparsed_' + call['role'] + '__'
        if answer != call['normalized_answer']:
            raise ValueError('Recorded answer does not match original response')
        candidate = ScoredCandidate.from_response(call['role'], answer, response)
        if not math.isclose(candidate.ppl, call['ppl'], rel_tol=1e-9):
            raise ValueError('Recorded PPL does not match complete token logprobs')
        candidates.append(candidate)
        trace_keys.append(key)
    diagnostic = arm['open_answer_diagnostic']
    similarities = diagnostic['answer_similarity']
    if (diagnostic['teacher_answer'] != candidates[0].answer or
            diagnostic['student_answers'] != {c.role: c.answer for c in candidates[1:]} or
            set(similarities) != set(ROLES[1:])):
        raise ValueError('Recorded answer similarity is attached to different candidates')
    alt = select_open(candidates, [similarities[role] for role in ROLES[1:]])
    full_sim = dict(arm['similarity_ranking'])
    if set(full_sim) != set(ROLES[1:]):
        raise ValueError('Original ranking is incomplete')
    original = select_open(candidates, [full_sim[role] for role in ROLES[1:]])
    if (original.positive.role != arm['positive_role'] or original.negative.role != arm['negative_role']
            or original.reason != arm['reason']):
        raise ValueError('Original decision does not replay')
    return {'sample_index': row['sample_index'], 'sample_id': row['sample_id'],
            'original_answer': arm['answer'], 'original_positive_role': original.positive.role,
            'replay_answer': '' if alt.positive.answer.startswith('__unparsed_') else alt.positive.answer,
            'replay_positive_role': alt.positive.role, 'replay_negative_role': alt.negative.role,
            'replay_positive_roles': list(alt.positive_roles), 'replay_negative_roles': list(alt.negative_roles),
            'full_response_similarity_ranking': [list(pair) for pair in original.similarity_ranking],
            'answer_similarity_ranking': [list(pair) for pair in alt.similarity_ranking],
            'candidate_trace_keys': dict(zip(ROLES, trace_keys)),
            'candidate_ppl': {candidate.role: candidate.ppl for candidate in candidates}}


def replay(run, inference, arm, count):
    inputs = read_rows(inference)
    for item in inputs:
        require_label_free(item)
    manifest, rows = audit(run, inference, count)
    if manifest['dataset'] != 'textvqa' or manifest.get('format_policy') not in ('original', 'short_final'):
        raise ValueError('Expected a completed TextVQA factorial stream')
    verify_online_and_sampling(rows, manifest['seed'], manifest['format_policy'])
    examples = load_local_dataset(str(inference), 'textvqa', str(inference.parent), limit=count)
    if any(example.target is not None for example in examples):
        raise ValueError('Inference data includes targets')
    trace_dir = run / 'response_trace'
    first = next(trace_dir.glob('*.json'), None)
    if first is None:
        raise ValueError('Missing exact request traces')
    namespace = json.loads(first.read_text())['request']['namespace']
    for k, v in {'dataset': 'textvqa', 'input_sha256': manifest['input_sha256'],
                 'count': count, 'seed': manifest['seed'], 'policy': manifest['format_policy']}.items():
        if namespace.get(k) != v:
            raise ValueError('Trace namespace mismatches completed run: ' + k)
    records = [replay_candidate_arm(r, e.sample, arm, trace_dir, namespace, manifest['model'])
               for r, e in zip(rows, examples)]
    return {'protocol': 'fixed_candidate_answer_similarity_replay_v1', 'labels_read': False,
            'model_calls': 0, 'memory_note': 'Recorded candidates and memory are fixed; no online memory is regenerated.',
            'source_predictions_sha256': sha256_file(run / 'predictions.jsonl'),
            'source_manifest_sha256': sha256_file(run / 'manifest.json'),
            'source_inference_sha256': sha256_file(inference), 'arm': arm, 'count': count,
            'records': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--inference', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = replay(args.run, args.inference, args.arm, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'count': result['count'], 'arm': result['arm'], 'labels_read': False,
                      'model_calls': 0, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
