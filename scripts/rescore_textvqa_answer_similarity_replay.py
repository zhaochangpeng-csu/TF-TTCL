#!/usr/bin/env python3
"""Exploratory MMF re-score of fixed-candidate answer-similarity replay.

This diagnostic changes only the selected output from the recorded candidates.
In particular, it does not generate a new online memory or model response.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.answers import extract_final_answer
from core.multimodal.datasets import _extract_answers
from scripts.compare_textvqa_format_memory import contrast
from scripts.create_label_isolated_subset import sha256_file
from scripts.inspect_baseline_io import read_rows
from scripts.replay_textvqa_answer_similarity import replay
from scripts.rescore_textvqa_mmf import MMF_REVISION, MMF_SHA256, MMF_URL, load_evaluator, score_one


def score_records(source_rows, replay_records, targets, evaluator, arm):
    if not source_rows or len(source_rows) != len(replay_records) or len(targets) != len(source_rows):
        raise ValueError('Unequal or empty paired scoring inputs')
    before, after = [], []
    for row, record, truth in zip(source_rows, replay_records, targets):
        if (str(row['sample_id']) != str(record['sample_id']) or
                row['sample_index'] != record['sample_index']):
            raise ValueError('Replay ID or index does not match source')
        source_arm = row['arms'][arm]
        calls = [c for c in source_arm['calls'] if c.get('role')]
        by_role = {c['role']: c for c in calls}
        if len(calls) != 5 or len(by_role) != 5 or source_arm['positive_role'] != record['original_positive_role']:
            raise ValueError('Original candidate roles do not match replay')
        original = by_role[record['original_positive_role']]
        new = by_role[record['replay_positive_role']]
        if (source_arm['answer'] != record['original_answer'] or
                ('' if new['normalized_answer'].startswith('__unparsed_') else new['normalized_answer']) != record['replay_answer']):
            raise ValueError('Replay answers do not match source candidates')
        before.append(score_one(evaluator, extract_final_answer(original['response']['content']), truth))
        after.append(score_one(evaluator, extract_final_answer(new['response']['content']), truth))
    delta = [b - a for a, b in zip(before, after)]
    return {'original_mean': sum(before) / len(before), 'replay_mean': sum(after) / len(after),
            'paired_delta': contrast(delta), 'improved_items': sum(x > 1e-12 for x in delta),
            'worsened_items': sum(x < -1e-12 for x in delta),
            'unchanged_items': sum(abs(x) <= 1e-12 for x in delta),
            'selected_role_changed_items': sum(r['original_positive_role'] != r['replay_positive_role'] for r in replay_records),
            'normalized_answer_changed_items': sum(r['original_answer'] != r['replay_answer'] for r in replay_records)}


def rescore(run, inference, subset_manifest, labels, replay_path, evaluator, expected_manifest_sha):
    if not re.fullmatch(r'[0-9a-f]{64}', expected_manifest_sha):
        raise ValueError('A fixed subset manifest SHA256 is required')
    if len({p.resolve() for p in (inference, subset_manifest, labels, replay_path)}) != 4:
        raise ValueError('Inference, manifest, labels and replay files must be separate')
    if sha256_file(subset_manifest) != expected_manifest_sha:
        raise ValueError('Frozen subset manifest SHA256 mismatch')
    subset = json.loads(subset_manifest.read_text())
    if (subset.get('dataset_id') != 'official/TextVQA_0.5.1' or subset.get('split') != 'train'
            or subset.get('count') != 400 or subset.get('inference_sha256') != sha256_file(inference)
            or not re.fullmatch(r'[0-9a-f]{64}', str(subset.get('labels_sha256', '')))):
        raise ValueError('Frozen subset is not the expected label-isolated 400-item TextVQA train stream')
    recorded = json.loads(replay_path.read_text())
    if recorded.get('arm') not in ('select_no_memory', 'tf_ttcl_mm_direct'):
        raise ValueError('Unexpected source arm')
    # Rebuild and compare the entire immutable replay, including exact trace keys,
    # candidate PPL, source checksums and chosen roles, before reading labels.
    recomputed = replay(run, inference, recorded['arm'], 400)
    if recorded != recomputed:
        raise ValueError('Replay or its source artifacts changed since generation')
    rows = read_rows(run / 'predictions.jsonl')
    if [str(x['sample_id']) for x in rows] != [str(x) for x in subset.get('selected_ids', [])]:
        raise ValueError('Prediction IDs mismatch frozen subset order')
    if sha256_file(labels) != subset['labels_sha256']:
        raise ValueError('Isolated labels SHA256 mismatch')
    label_rows = read_rows(labels)
    if len(label_rows) != 400 or [str(x.get('id')) for x in label_rows] != [str(x['sample_id']) for x in rows]:
        raise ValueError('Label IDs/order mismatch')
    targets = [_extract_answers(row, row.get('choices', row.get('options', []))) for row in label_rows]
    result = score_records(rows, recorded['records'], targets, evaluator, recorded['arm'])
    return {'protocol': 'exploratory_fixed_candidate_answer_similarity_mmf_v1', 'count': 400,
            'source_predictions_sha256': recorded['source_predictions_sha256'],
            'source_manifest_sha256': recorded['source_manifest_sha256'],
            'source_inference_sha256': recorded['source_inference_sha256'],
            'subset_manifest_sha256': expected_manifest_sha, 'labels_sha256': subset['labels_sha256'],
            'replay_sha256': sha256_file(replay_path), 'arm': recorded['arm'],
            'upstream_revision': MMF_REVISION, 'upstream_url': MMF_URL, 'upstream_sha256': MMF_SHA256,
            'answer_extraction': 'existing raw final-answer extraction with last-line fallback',
            'results': result,
            'interpretation': 'Exploratory fixed-candidate output-only re-selection. The recorded online memory '
                              'and all earlier decisions are unchanged; this is not a new online memory run. '
                              'Paired resampling concerns the 400-item development stream, not confirmed generalization.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'inference', 'subset-manifest', 'labels', 'replay', 'mmf-evaluator', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--subset-manifest-sha256', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    evaluator = load_evaluator(args.mmf_evaluator)
    result = rescore(args.run, args.inference, args.subset_manifest, args.labels,
                     args.replay, evaluator, args.subset_manifest_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
