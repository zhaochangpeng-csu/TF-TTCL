#!/usr/bin/env python3
"""Score a complete, label-blind TextVQA replay against NVIDIA reference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.answers import normalize_answer
from core.multimodal.datasets import _extract_answers, load_local_dataset
from core.multimodal.prompts import format_question
from core.multimodal.schema import GenerationResponse, SamplingConfig
from core.multimodal.tf_ttcl_direct_runner import TEACHER_SYSTEM_OPEN
from core.multimodal.tf_ttcl_direct_selection import strict_generation_ppl
from evaluate.multimodal.metrics import paired_bootstrap_delta_ci, paired_randomization_p_value, score_prediction
from scripts.create_label_isolated_subset import sha256_file
from scripts.inspect_baseline_io import require_label_free

INPUT_SHA256 = 'e7b3865aa534594634ab9f9c136344d1c2673f317a2578722f09677cc8189b74'
LABEL_SHA256 = '4d28522f666df8ea2d13c10542366aff45f14253e8e58afeb4929a8046e41a05'
REFERENCE_SHA256 = '4750646aa363532c6169f98c332e0e3ef32dc1e04ec13b8feb729b3874e762c4'


def _rows(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare(inference: Path, reference: Path, shards: list[Path], labels: Path,
            input_sha256: str, reference_sha256: str, labels_sha256: str,
            expected_count: int = 400) -> dict:
    """Do all label-free alignment checks before opening the label file."""
    if not shards or len(set(p.resolve() for p in shards)) != len(shards):
        raise ValueError('Prediction shards must be nonempty and unique')
    if sha256_file(inference) != input_sha256 or sha256_file(reference) != reference_sha256:
        raise ValueError('Inference or reference hash mismatch')
    source, old = _rows(inference), _rows(reference)
    examples = load_local_dataset(str(inference), 'textvqa', str(inference.parent))
    if len(source) != expected_count or len(old) != expected_count:
        raise ValueError('Inference or NVIDIA reference is incomplete')
    for row in source:
        require_label_free(row)
    ids = [example.sample.sample_id for example in examples]
    if len(set(ids)) != expected_count or any(not x for x in ids):
        raise ValueError('Inference IDs are missing or duplicated')
    predictions = {}
    for path in shards:
        for row in _rows(path):
            index = row.get('sample_index')
            if type(index) is not int or not 0 <= index < expected_count or index in predictions:
                raise ValueError('Duplicate or invalid prediction index')
            predictions[index] = row
    if set(predictions) != set(range(expected_count)):
        raise ValueError('Prediction shards are incomplete')
    old_answers, new_answers = [], []
    for index, example in enumerate(examples):
        sample, ref, new = example.sample, old[index], predictions[index]
        if (example.target is not None or ref.get('sample_index') != index
                or ref.get('sample_id') != ids[index] or ref.get('labels_read') is not False
                or new.get('sample_id') != ids[index] or new.get('labels_read') is not False):
            raise ValueError(f'ID/order/label boundary mismatch at {index}')
        call = ref['arms']['greedy']['calls'][0]
        sampling = SamplingConfig(temperature=0.0, max_tokens=256, seed=100 * index,
                                  request_logprobs=True)
        if (call['system'] != TEACHER_SYSTEM_OPEN
                or call['user'] != format_question(sample.question, sample.choices, sample.metadata)
                or call['sampling'] != asdict(sampling)
                or not call['image_hashes'] or new.get('image_hashes') != call['image_hashes']
                or ref['image_hashes'] != call['image_hashes']):
            raise ValueError(f'Reference prompt, seed, or image hash mismatch at {index}')
        old_answer = ref['arms']['greedy']['answer']
        old_normalized = call['normalized_answer']
        if old_normalized.startswith('__unparsed_'):
            old_normalized = ''
        if old_answer != old_normalized or new.get('reference_answer') != old_answer:
            raise ValueError(f'Reference answer mismatch at {index}')
        response = GenerationResponse(**new['response'])
        ppl = strict_generation_ppl(response)
        if not math.isclose(ppl, new['ppl'], rel_tol=1e-9):
            raise ValueError(f'Prediction PPL mismatch at {index}')
        normalized = normalize_answer(response.content, 'textvqa',
                                      {**sample.metadata, 'choices': sample.choices})
        if new.get('answer') != normalized:
            raise ValueError(f'Prediction answer mismatch at {index}')
        old_answers.append(old_answer)
        new_answers.append(normalized)

    # The isolated target file is first opened only after all output checks pass.
    label_bytes = labels.read_bytes()
    if hashlib.sha256(label_bytes).hexdigest() != labels_sha256:
        raise ValueError('Isolated labels hash mismatch')
    targets = [json.loads(line) for line in label_bytes.splitlines() if line.strip()]
    if (len(targets) != expected_count or [str(item.get('id', '')) for item in targets] != ids
            or any(not _extract_answers(item, item.get('choices', item.get('options', [])))
                   for item in targets)):
        raise ValueError('Isolated labels are missing or out of order')
    def score(answers):
        return [score_prediction('textvqa', answer,
                                 _extract_answers(target, target.get('choices', target.get('options', []))),
                                 {key: target[key] for key in ('answer_type', 'precision',
                                  'question_type', 'choices', 'options') if key in target})
                for answer, target in zip(answers, targets)]
    old_scores, new_scores = score(old_answers), score(new_answers)
    return {
        'status': 'scored_after_complete_label_free_alignment', 'count': expected_count,
        'input_sha256': input_sha256, 'reference_sha256': reference_sha256,
        'labels_sha256': labels_sha256,
        'shard_sha256': {str(path): sha256_file(path) for path in shards},
        'answers_equal': sum(a == b for a, b in zip(old_answers, new_answers)),
        'answers_changed': sum(a != b for a, b in zip(old_answers, new_answers)),
        'nvidia_score_sum': sum(old_scores), 'hygon_score_sum': sum(new_scores),
        'nvidia_accuracy': sum(old_scores) / expected_count,
        'hygon_accuracy': sum(new_scores) / expected_count,
        'improved': sum(a < b for a, b in zip(old_scores, new_scores)),
        'worsened': sum(a > b for a, b in zip(old_scores, new_scores)),
        'paired_delta_ci': paired_bootstrap_delta_ci(old_scores, new_scores),
        'paired_randomization_p': paired_randomization_p_value(old_scores, new_scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--shard', type=Path, action='append', required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.reference_sha256 != REFERENCE_SHA256:
        raise ValueError('Not the frozen NVIDIA reference')
    result = compare(args.inference, args.reference, args.shard, args.labels,
                     INPUT_SHA256, args.reference_sha256, LABEL_SHA256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
