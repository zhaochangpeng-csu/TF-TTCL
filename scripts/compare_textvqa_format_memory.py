#!/usr/bin/env python3
"""Offline 2×2 TextVQA prompt-format × online-memory paired comparison.

Two complete independent streams are audited before isolated labels are read.
Significance is descriptive for one development stream, not a confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.datasets import _extract_answers
from core.multimodal.schema import SamplingConfig
from core.multimodal.textvqa_format_factorial import FACTORIAL_VERSION, SHORT_FINAL_SUFFIX
from core.multimodal.tf_ttcl_direct_runner import (TEACHER_SYSTEM_OPEN, STUDENT_SYSTEM_OPEN,
    TUTOR_SYSTEM, RULE_HEADER, POSITIVE_TEMPLATE_OPEN, NEGATIVE_TEMPLATE_OPEN, _digest, _rule_prompt)
from evaluate.multimodal.metrics import paired_bootstrap_delta_ci, paired_randomization_p_value, score_prediction
from scripts.audit_tf_ttcl_mm_direct import audit
from scripts.create_label_isolated_subset import sha256_file

ARMS = ('select_no_memory', 'tf_ttcl_mm_direct')
CONDITIONS = ('baseline_no_memory', 'baseline_memory', 'strengthened_no_memory', 'strengthened_memory')
CONFIG_KEYS = ('dataset', 'version', 'input_sha256', 'ordered_ids', 'model', 'model_revision',
               'embedding_revision', 'seed', 'arms', 'candidate_max_tokens', 'tutor_max_tokens',
               'rule_max_tokens', 'rule_top_k_each', 'teacher_temperature', 'student_temperature',
               'student_top_p', 'labels_read')


def selected_response(arm):
    matches = [call for call in arm['calls'] if call.get('role') == arm['positive_role']]
    if len(matches) != 1:
        raise ValueError('Selected role must have exactly one answering call')
    answer = matches[0]['normalized_answer']
    if arm['answer'] != ('' if answer.startswith('__unparsed_') else answer):
        raise ValueError('Final answer differs from the selected candidate')
    return matches[0]['response']['content']


def format_ok(response):
    """Surface format only: one nonempty final-answer line, without trailing text."""
    markers = list(re.finditer(r'(?im)^[ \t]*final answer[ \t]*:[ \t]*([^\n\r]+)[ \t]*$', response))
    return (len(markers) == 1 and bool(markers[0].group(1).strip())
            and not response[markers[0].end():].strip())


def contrast(differences):
    zero = [0.0] * len(differences)
    return {'mean_and_95pct_ci': paired_bootstrap_delta_ci(zero, differences),
            'paired_sign_randomization_p': paired_randomization_p_value(zero, differences)}


def policy_prompt_hash(policy):
    if policy not in ('original', 'short_final'):
        raise ValueError('Unknown format policy')
    suffix = '' if policy == 'original' else SHORT_FINAL_SUFFIX
    return _digest([TEACHER_SYSTEM_OPEN + suffix, STUDENT_SYSTEM_OPEN + suffix, TUTOR_SYSTEM,
                    RULE_HEADER, POSITIVE_TEMPLATE_OPEN, NEGATIVE_TEMPLATE_OPEN])


def verify_online_and_sampling(rows, seed, policy):
    if policy not in ('original', 'short_final'):
        raise ValueError('Unknown format policy')
    suffix = '' if policy == 'original' else SHORT_FINAL_SUFFIX
    prior = {}
    for index, row in enumerate(rows):
        for kind, bucket in row['retrieved_rules'].items():
            for rule in bucket:
                old = prior.get(rule['rule_id'])
                if (old is None or old['sample_index'] >= index or old['sample_index'] != rule['sample_index']
                        or old['kind'] != kind or old['text'] != rule['text']):
                    raise ValueError(f'Retrieval not produced by earlier sample: {index}')
        for arm_name in ARMS:
            answering = [c for c in row['arms'][arm_name]['calls'] if c.get('role')]
            if [c['role'] for c in answering] != ['teacher'] + [f'student_{n}' for n in range(1, 5)]:
                raise ValueError(f'Candidate roles or order changed: {index}')
            rule_text = _rule_prompt(row['retrieved_rules']) if arm_name == 'tf_ttcl_mm_direct' else ''
            for n, call in enumerate(answering):
                system = (TEACHER_SYSTEM_OPEN if n == 0 else STUDENT_SYSTEM_OPEN) + suffix
                if rule_text:
                    system += '\n' + rule_text
                if call.get('system') != system:
                    raise ValueError(f'Candidate system prompt changed: {index}/{arm_name}/{n}')
                expected = SamplingConfig(temperature=0.0 if n == 0 else 0.7,
                    top_p=None if n == 0 else 0.9, max_tokens=256,
                    seed=seed + 100 * index + (0 if n == 0 else n + 1), request_logprobs=True)
                if call['sampling'] != asdict(expected):
                    raise ValueError(f'Candidate sampling changed: {index}/{arm_name}/{n}')
        for rule in row['written_rules']:
            if rule['sample_index'] != index or rule['rule_id'] in prior:
                raise ValueError(f'Invalid rule write: {index}')
            prior[rule['rule_id']] = rule


def compare(baseline, strengthened, inference, subset_manifest, labels,
            baseline_prompt_sha, strengthened_prompt_sha, *, expected_count=400,
            audit_fn=audit):
    if expected_count != 400:
        raise ValueError('The 20-item smoke must remain unscored')
    if not re.fullmatch(r'[0-9a-f]{64}', baseline_prompt_sha) or not re.fullmatch(r'[0-9a-f]{64}', strengthened_prompt_sha) or baseline_prompt_sha == strengthened_prompt_sha:
        raise ValueError('Both distinct frozen prompt hashes are required')
    if (baseline_prompt_sha, strengthened_prompt_sha) != (policy_prompt_hash('original'), policy_prompt_hash('short_final')):
        raise ValueError('Frozen prompt hashes do not match the expected policies')
    if baseline.resolve() == strengthened.resolve():
        raise ValueError('The two independent streams need separate run directories')
    subset = json.loads(subset_manifest.read_text())
    if (subset.get('dataset_id') != 'official/TextVQA_0.5.1' or subset.get('split') != 'train'
            or subset.get('count') != expected_count or subset.get('inference_sha256') != sha256_file(inference)):
        raise ValueError('Frozen TextVQA subset manifest mismatch')
    if not re.fullmatch(r'[0-9a-f]{64}', str(subset.get('labels_sha256', ''))):
        raise ValueError('Subset manifest lacks isolated label hash')
    # No target is accessed until both full-run label-free audits complete.
    bm, br = audit_fn(baseline, inference, expected_count)
    sm, sr = audit_fn(strengthened, inference, expected_count)
    if bm.get('prompt_sha256') != baseline_prompt_sha or sm.get('prompt_sha256') != strengthened_prompt_sha:
        raise ValueError('Prompt hash mismatch')
    if (bm.get('format_experiment'), bm.get('format_policy'),
            sm.get('format_experiment'), sm.get('format_policy')) != (
            FACTORIAL_VERSION, 'original', FACTORIAL_VERSION, 'short_final'):
        raise ValueError('Manifest format policies do not match the two frozen treatments')
    for key in CONFIG_KEYS:
        if key not in bm or key not in sm or bm[key] != sm[key]:
            raise ValueError(f'Unequal or missing paired configuration: {key}')
    if bm['dataset'] != 'textvqa' or bm['labels_read'] is not False:
        raise ValueError('Invalid dataset or label boundary')
    ids = [str(r.get('sample_id', '')) for r in br]
    if (len(br) != expected_count or len(sr) != expected_count or len(set(ids)) != expected_count
            or ids != [str(r.get('sample_id', '')) for r in sr]
            or ids != [str(i) for i in subset.get('selected_ids', [])]
            or ids != [str(i) for i in bm['ordered_ids']]):
        raise ValueError('Ordered IDs differ between streams and frozen subset')
    verify_online_and_sampling(br, bm['seed'], 'original')
    verify_online_and_sampling(sr, sm['seed'], 'short_final')
    responses = {}
    for cond, rows, arm in zip(CONDITIONS, (br, br, sr, sr), ARMS * 2):
        records = [row['arms'][arm] for row in rows]
        # Force all format and usage reads before accessing any labels.
        answers = [record['answer'] for record in records]
        texts = [selected_response(record) for record in records]
        usages = [record['usage'] for record in records]
        if any(not isinstance(a, str) for a in answers):
            raise ValueError('Non-string answer')
        responses[cond] = (answers, texts, usages)
    raw = labels.read_bytes()
    if hashlib.sha256(raw).hexdigest() != subset['labels_sha256']:
        raise ValueError('Isolated target hash mismatch')
    targets = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(targets) != expected_count or [str(x.get('id', '')) for x in targets] != ids:
        raise ValueError('Targets differ from the frozen sample order')
    truths = [_extract_answers(row, row.get('choices', row.get('options', []))) for row in targets]
    if any(not truth for truth in truths):
        raise ValueError('Missing isolated targets')
    scores, summaries = {}, {}
    for cond, (answers, texts, usages) in responses.items():
        vals = [score_prediction('textvqa', answer, truth) for answer, truth in zip(answers, truths)]
        scores[cond] = vals
        summaries[cond] = {'local_vqa_score': sum(vals) / expected_count,
            'mean_answer_words': sum(len(a.split()) for a in answers) / expected_count,
            'short_answer_at_most_5_words_rate': sum(0 < len(a.split()) <= 5 for a in answers) / expected_count,
            'selected_output_format_rate': sum(format_ok(t) for t in texts) / expected_count,
            'logical_prompt_tokens': sum(u['prompt_tokens'] for u in usages),
            'logical_completion_tokens': sum(u['completion_tokens'] for u in usages),
            'logical_latency_seconds': sum(u['latency_seconds'] for u in usages)}
    bn, bm_, sn, sm_ = (scores[k] for k in CONDITIONS)
    vectors = {'format_gain_no_memory': [c-a for a,c in zip(bn,sn)],
               'format_gain_memory': [d-b for b,d in zip(bm_,sm_)],
               'memory_gain_baseline': [b-a for a,b in zip(bn,bm_)],
               'memory_gain_strengthened': [d-c for c,d in zip(sn,sm_)],
               'interaction': [(d-c)-(b-a) for a,b,c,d in zip(bn,bm_,sn,sm_)]}
    ratios = {}
    for memory in ('no_memory', 'memory'):
        old = summaries['baseline_' + memory]
        new = summaries['strengthened_' + memory]
        old_tokens = old['logical_prompt_tokens'] + old['logical_completion_tokens']
        new_tokens = new['logical_prompt_tokens'] + new['logical_completion_tokens']
        ratios[memory] = {'strengthened_over_baseline': new_tokens / old_tokens if old_tokens else None,
                          'within_5_percent': abs(new_tokens - old_tokens) <= .05 * old_tokens if old_tokens else False}
    return {'status': 'scored_after_both_complete_label_free_audits', 'n': expected_count,
            'dataset': 'textvqa_train', 'input_sha256': sha256_file(inference),
            'labels_sha256': subset['labels_sha256'], 'prompt_hashes': [baseline_prompt_sha, strengthened_prompt_sha],
            'scores': summaries, 'contrasts': {name: {**contrast(vec),
                'improved_items': sum(x > 1e-12 for x in vec),
                'worsened_items': sum(x < -1e-12 for x in vec)} for name, vec in vectors.items()},
            'cost_matching': ratios,
            'cost_note': 'Logical per-arm costs include shared calls; do not add as physical cost.',
            'metric_note': 'Local TextVQA soft score; official EvalAI parity not verified.',
            'inference_note': 'One developmental stream; sign-randomization p-values assume exchangeability.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'strengthened', 'inference', 'subset-manifest', 'labels', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--baseline-prompt-sha256', required=True)
    parser.add_argument('--strengthened-prompt-sha256', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.baseline, args.strengthened, args.inference, args.subset_manifest,
                     args.labels, args.baseline_prompt_sha256, args.strengthened_prompt_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
