#!/usr/bin/env python3
"""Audit a complete label-free five-candidate replay, then score offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.datasets import _extract_answers, load_local_dataset
from core.multimodal.answers import normalize_answer
from core.multimodal.prompts import format_question
from core.multimodal.schema import GenerationResponse, SamplingConfig
from core.multimodal.tf_ttcl_direct_runner import STUDENT_SYSTEM, TEACHER_SYSTEM
from core.multimodal.tf_ttcl_direct_selection import ScoredCandidate, select_closed, strict_generation_ppl
from evaluate.multimodal.metrics import paired_bootstrap_delta_ci, paired_randomization_p_value, score_prediction
from scripts.create_label_isolated_subset import sha256_file
from scripts.inspect_baseline_io import require_label_free

INPUT_SHA256 = '01c6dc83958ea1cee51ab38594b2499e3bfbc4034139674dc799e5dfeb354689'
LABEL_SHA256 = '009cff0b2f94a472311dfb6f112558bfecbb81477f11ae100c15de4c0c451fb7'
REFERENCE_SHA256 = 'b8ac87c5bdf38923f7ec5e9e848b3f1b18e0f183b2ea0deffea98700ffaf4d71'
GREEDY_SHA256 = 'c79333d6dde5589905efeab450a08039b6d75fc1767aef4ed8e1a72037707a9e'


def candidate(role, response, sample):
    answer = normalize_answer(response.content, sample.dataset,
                              {**sample.metadata, 'choices': sample.choices})
    if answer not in {chr(97 + j) for j in range(len(sample.choices))}:
        answer = f'__unparsed_{role}__'
    return ScoredCandidate.from_response(role, answer, response)


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def compare(inference, reference, greedy, shards, labels, input_hash, reference_hash,
            greedy_hash, label_hash, expected_count=400):
    if not shards or len({p.resolve() for p in shards}) != len(shards):
        raise ValueError('Prediction shards must be nonempty and unique')
    if any(sha256_file(path) != expected for path, expected in
           ((inference, input_hash), (reference, reference_hash), (greedy, greedy_hash))):
        raise ValueError('Input/reference/greedy hash mismatch')
    raw, old, teachers = rows(inference), rows(reference), rows(greedy)
    examples = load_local_dataset(str(inference), 'scienceqa_img', str(inference.parent))
    if any(len(x) != expected_count for x in (raw, old, teachers, examples)):
        raise ValueError('Input/reference/greedy incomplete')
    for item in raw:
        require_label_free(item)
    ids = [item.sample.sample_id for item in examples]
    if len(set(ids)) != expected_count:
        raise ValueError('Duplicate sample IDs')
    predictions = {}
    for shard in shards:
        for row in rows(shard):
            index = row.get('sample_index')
            if type(index) is not int or not 0 <= index < expected_count or index in predictions:
                raise ValueError('Duplicate or invalid prediction index')
            predictions[index] = row
    if set(predictions) != set(range(expected_count)):
        raise ValueError('Prediction shards incomplete')
    old_answers, new_answers = [], []
    for i, item in enumerate(examples):
        sample, ref, teacher, row = item.sample, old[i], teachers[i], predictions[i]
        if (item.target is not None or ref.get('sample_index') != i or ref.get('sample_id') != ids[i]
                or ref.get('labels_read') is not False or row.get('sample_id') != ids[i]
                or row.get('labels_read') is not False or row.get('teacher') != teacher
                or teacher.get('sample_index') != i or teacher.get('sample_id') != ids[i]
                or teacher.get('labels_read') is not False):
            raise ValueError(f'Identity/label/teacher mismatch at {i}')
        old_calls = [c for c in ref['arms']['vote_no_rewrite']['calls'] if c.get('role')]
        if [c['role'] for c in old_calls] != ['teacher', 'student_1', 'student_2', 'student_3', 'student_4']:
            raise ValueError(f'Original roles mismatch at {i}')
        hashes = ref['image_hashes']
        question = format_question(sample.question, sample.choices, sample.metadata)
        if not hashes or row.get('image_hashes') != hashes or teacher.get('image_hashes') != hashes:
            raise ValueError(f'Image hash mismatch at {i}')
        for n, call in enumerate(old_calls):
            expected = SamplingConfig(temperature=0.7 if n else 0.0,
                                      top_p=0.9 if n else None, max_tokens=256,
                                      seed=100*i+n+1 if n else 100*i, request_logprobs=True)
            if (call['system'] != (STUDENT_SYSTEM if n else TEACHER_SYSTEM)
                    or call['user'] != question or call['sampling'] != asdict(expected)
                    or call['image_hashes'] != hashes):
                raise ValueError(f'Original request mismatch at {i}/{n}')
            if not math.isclose(strict_generation_ppl(GenerationResponse(**call['response'])),
                                call['ppl'], rel_tol=1e-9):
                raise ValueError(f'Original PPL mismatch at {i}/{n}')
        teacher_candidate = candidate('teacher', GenerationResponse(**teacher['response']), sample)
        if (not math.isclose(teacher_candidate.ppl, teacher['ppl'], rel_tol=1e-9)
                or teacher['answer'] != ('' if teacher_candidate.answer.startswith('__unparsed_')
                                         else teacher_candidate.answer)
                or teacher['reference_answer'] != ref['arms']['greedy']['answer']):
            raise ValueError(f'Teacher candidate mismatch at {i}')
        students = row.get('students')
        if not isinstance(students, list) or len(students) != 4:
            raise ValueError(f'Wrong student count at {i}')
        candidates = [teacher_candidate]
        for n, student in enumerate(students, 1):
            role = f'student_{n}'
            sampling = SamplingConfig(temperature=0.7, top_p=0.9, max_tokens=256,
                                      seed=100*i+n+1, request_logprobs=True)
            scored = candidate(role, GenerationResponse(**student['response']), sample)
            if (student.get('role') != role or student.get('sampling') != asdict(sampling)
                    or student.get('answer') != scored.answer
                    or not math.isclose(student['ppl'], scored.ppl, rel_tol=1e-9)):
                raise ValueError(f'Student mismatch at {i}/{n}')
            candidates.append(scored)
        selected = select_closed(candidates)
        final = '' if selected.positive.answer.startswith('__unparsed_') else selected.positive.answer
        if (row.get('answer') != final or row.get('positive_role') != selected.positive.role
                or row.get('negative_role') != (selected.negative.role if selected.negative else None)
                or row.get('positive_roles') != list(selected.positive_roles)
                or row.get('negative_roles') != list(selected.negative_roles)
                or row.get('reason') != selected.reason
                or row.get('reference_answer') != ref['arms']['vote_no_rewrite']['answer']):
            raise ValueError(f'Selection mismatch at {i}')
        old_answers.append(row['reference_answer'])
        new_answers.append(final)
    # No target is opened until the entire 400-item prediction audit succeeds.
    label_bytes = labels.read_bytes()
    if hashlib.sha256(label_bytes).hexdigest() != label_hash:
        raise ValueError('Label hash mismatch')
    targets = [json.loads(line) for line in label_bytes.splitlines() if line.strip()]
    if (len(targets) != expected_count or [str(t.get('id', '')) for t in targets] != ids
            or any(not _extract_answers(t, t.get('choices', t.get('options', []))) for t in targets)):
        raise ValueError('Label IDs or answers do not match')
    def score(answers):
        return [score_prediction('scienceqa_img', ans,
                                 _extract_answers(target, target.get('choices', target.get('options', []))),
                                 target) for ans, target in zip(answers, targets)]
    before, after = score(old_answers), score(new_answers)
    return {'status': 'scored_after_complete_label_free_alignment', 'count': expected_count,
            'input_sha256': input_hash, 'reference_sha256': reference_hash,
            'greedy_sha256': greedy_hash, 'labels_sha256': label_hash,
            'shard_sha256': {str(p): sha256_file(p) for p in shards},
            'answers_equal': sum(a == b for a, b in zip(old_answers, new_answers)),
            'answers_changed': sum(a != b for a, b in zip(old_answers, new_answers)),
            'nvidia_correct': int(sum(before)), 'hygon_correct': int(sum(after)),
            'nvidia_accuracy': sum(before)/expected_count, 'hygon_accuracy': sum(after)/expected_count,
            'improved': sum(a < b for a, b in zip(before, after)),
            'worsened': sum(a > b for a, b in zip(before, after)),
            'paired_delta_ci': paired_bootstrap_delta_ci(before, after),
            'paired_randomization_p': paired_randomization_p_value(before, after)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('inference', 'reference', 'greedy', 'labels', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--shard', type=Path, action='append', required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = compare(a.inference, a.reference, a.greedy, a.shard, a.labels,
                     INPUT_SHA256, REFERENCE_SHA256, GREEDY_SHA256, LABEL_SHA256)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
