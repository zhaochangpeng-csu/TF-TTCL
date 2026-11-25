#!/usr/bin/env python3
"""Compare complete online Hygon and NVIDIA runs; open targets only after audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.datasets import _extract_answers
from core.multimodal.tf_ttcl_direct_runner import ARMS
from evaluate.multimodal.metrics import paired_bootstrap_delta_ci, paired_randomization_p_value, score_prediction
from scripts.audit_hygon_tf_ttcl_direct import REFERENCE_SHA, audit
from scripts.create_label_isolated_subset import sha256_file

LABEL_SHA = '009cff0b2f94a472311dfb6f112558bfecbb81477f11ae100c15de4c0c451fb7'


def compare(run: Path, inference: Path, reference: Path, labels: Path,
            label_sha: str = LABEL_SHA) -> dict:
    """Check both complete runs and ordering before opening isolated labels."""
    if sha256_file(reference) != REFERENCE_SHA:
        raise ValueError('Original NVIDIA reference hash mismatch')
    manifest, rows = audit(run, inference, reference, 400)
    with reference.open(encoding='utf-8') as f:
        refs = [json.loads(line) for line in f if line.strip()]
    if (manifest.get('status') != 'completed' or manifest.get('processed') != 400
            or manifest.get('labels_read') is not False or len(rows) != 400 or len(refs) != 400):
        raise ValueError('Both runs must be complete and label-free')
    ids = []
    for index, (row, ref) in enumerate(zip(rows, refs)):
        if (row.get('sample_index') != index or ref.get('sample_index') != index
                or row.get('sample_id') != ref.get('sample_id')
                or row.get('labels_read') is not False or ref.get('labels_read') is not False
                or set(row.get('arms', {})) != set(ARMS) or set(ref.get('arms', {})) != set(ARMS)):
            raise ValueError(f'Run alignment or label boundary mismatch at {index}')
        ids.append(str(row['sample_id']))
    if len(set(ids)) != 400 or manifest.get('ordered_ids') != ids:
        raise ValueError('Run IDs are duplicated or manifest order differs')

    # First target access occurs only after the full audit and reference checks.
    label_bytes = labels.read_bytes()
    if hashlib.sha256(label_bytes).hexdigest() != label_sha:
        raise ValueError('Isolated label SHA mismatch')
    targets = [json.loads(line) for line in label_bytes.splitlines() if line.strip()]
    if len(targets) != 400 or [str(row.get('id', '')) for row in targets] != ids:
        raise ValueError('Label IDs do not match the audited streams')
    answers = [_extract_answers(target, target.get('choices', target.get('options', [])))
               for target in targets]
    if any(not answer for answer in answers):
        raise ValueError('Missing label')
    results = {}
    for arm in ARMS:
        old = [row['arms'][arm]['answer'] for row in refs]
        new = [row['arms'][arm]['answer'] for row in rows]
        old_scores = [score_prediction('scienceqa_img', prediction, truth)
                      for prediction, truth in zip(old, answers)]
        new_scores = [score_prediction('scienceqa_img', prediction, truth)
                      for prediction, truth in zip(new, answers)]
        results[arm] = {
            'nvidia_correct': int(sum(old_scores)), 'hygon_correct': int(sum(new_scores)),
            'nvidia_accuracy': sum(old_scores) / 400, 'hygon_accuracy': sum(new_scores) / 400,
            'answers_equal': sum(a == b for a, b in zip(old, new)),
            'answers_changed': sum(a != b for a, b in zip(old, new)),
            'improved': sum(a < b for a, b in zip(old_scores, new_scores)),
            'worsened': sum(a > b for a, b in zip(old_scores, new_scores)),
            'paired_delta_ci': paired_bootstrap_delta_ci(old_scores, new_scores),
            'paired_randomization_p': paired_randomization_p_value(old_scores, new_scores),
        }
    return {'status': 'scored_after_complete_online_audit', 'count': 400,
            'input_sha256': sha256_file(inference), 'reference_sha256': REFERENCE_SHA,
            'labels_sha256': label_sha, 'hygon_predictions_sha256': sha256_file(run / 'predictions.jsonl'),
            'hygon_git_sha': manifest.get('git_sha'), 'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--inference', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.run, args.inference, args.reference, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
