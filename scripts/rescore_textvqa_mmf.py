#!/usr/bin/env python3
"""Re-score completed TextVQA streams with a pinned Meta MMF evaluator.

Official reference: https://github.com/facebookresearch/mmf/blob/
2dc099af7ce267561e4c2ae033c27eb36be5b3ea/mmf/utils/m4c_evaluators.py
Requires the exact upstream file via --mmf-evaluator; never changes inference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.answers import extract_final_answer
from core.multimodal.datasets import _extract_answers
from scripts.compare_textvqa_format_memory import (ARMS, CONDITIONS, compare,
                                                   contrast, selected_response)

MMF_REVISION = '2dc099af7ce267561e4c2ae033c27eb36be5b3ea'
MMF_SHA256 = 'e10a99e15c4658f8d0d83a05cc536c835922e701171fce0016718bec021b8c0a'
MMF_URL = ('https://raw.githubusercontent.com/facebookresearch/mmf/' + MMF_REVISION +
           '/mmf/utils/m4c_evaluators.py')


def load_evaluator(path: Path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != MMF_SHA256:
        raise ValueError('Upstream evaluator SHA256 mismatch')
    spec = importlib.util.spec_from_file_location('pinned_mmf_evaluators', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TextVQAAccuracyEvaluator()


def score_one(evaluator, answer, targets):
    if len(targets) != 10 or any(not isinstance(a, str) for a in targets):
        raise ValueError('TextVQA scoring requires ten string answers')
    return evaluator.eval_pred_list([{'pred_answer': answer, 'gt_answers': targets}])


def score_records(rows, targets, evaluator):
    scores = {mode: {} for mode in ('raw_extracted', 'recorded_normalized')}
    summaries = {mode: {} for mode in scores}
    for condition, run_rows, arm_name in zip(CONDITIONS, (rows[0], rows[0], rows[1], rows[1]), ARMS * 2):
        raw, normalized, missing = [], [], 0
        for row, truth in zip(run_rows, targets):
            arm = row['arms'][arm_name]
            response = selected_response(arm)
            missing += not bool(re.search(r'(?i)final answer\s*:', response))
            raw.append(extract_final_answer(response))
            normalized.append(arm['answer'])
        for mode, answers in [('raw_extracted', raw), ('recorded_normalized', normalized)]:
            vals = [score_one(evaluator, answer, truth) for answer, truth in zip(answers, targets)]
            scores[mode][condition] = vals
            summaries[mode][condition] = {'mean': sum(vals) / len(vals),
                'missing_final_answer_marker': missing,
                'answers_changed_by_old_normalization': sum(a != b for a, b in zip(raw, normalized))}
    contrasts = {}
    for mode, group in scores.items():
        bn, bm, sn, sm = (group[k] for k in CONDITIONS)
        contrasts[mode] = {name: contrast(vec) for name, vec in {
            'memory_baseline': [b-a for a,b in zip(bn,bm)],
            'memory_short': [d-c for c,d in zip(sn,sm)],
            'format_no_memory': [c-a for a,c in zip(bn,sn)],
            'format_memory': [d-b for b,d in zip(bm,sm)],
            'interaction': [(d-c)-(b-a) for a,b,c,d in zip(bn,bm,sn,sm)]}.items()}
    return summaries, contrasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('baseline', 'strengthened', 'inference', 'subset-manifest', 'labels', 'mmf-evaluator', 'output'):
        parser.add_argument('--' + arg, required=True, type=Path)
    parser.add_argument('--baseline-prompt-sha256', required=True)
    parser.add_argument('--strengthened-prompt-sha256', required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    evaluator = load_evaluator(a.mmf_evaluator)
    # Original compare performs both complete label-free audits and verifies
    # paired hashes, prompt identities and target identity before scoring.
    original = compare(a.baseline, a.strengthened, a.inference, a.subset_manifest,
                       a.labels, a.baseline_prompt_sha256, a.strengthened_prompt_sha256)
    rows = [[json.loads(line) for line in (p / 'predictions.jsonl').read_text().splitlines() if line]
            for p in (a.baseline, a.strengthened)]
    labels = [json.loads(line) for line in a.labels.read_text().splitlines() if line]
    if any([str(x['sample_id']) for x in group] != [str(x['id']) for x in labels] for group in rows):
        raise ValueError('Prediction and target IDs are not aligned')
    targets = [_extract_answers(row, row.get('choices', row.get('options', []))) for row in labels]
    summaries, contrasts = score_records(rows, targets, evaluator)
    result = {'upstream_revision': MMF_REVISION, 'upstream_url': MMF_URL,
              'upstream_sha256': MMF_SHA256, 'n': len(targets),
              'input_sha256': original['input_sha256'], 'labels_sha256': original['labels_sha256'],
              'results': summaries, 'contrasts': contrasts,
              'note': 'Post-hoc exploratory re-score; raw answer uses existing deterministic extraction, '
                      'including last-line fallback when no marker. Not verified against EvalAI server.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
