#!/usr/bin/env python3
"""Audit online rule-control streams without opening labels.

The existing selector audit checks candidates, response traces, embeddings,
retrieval and PPL; this wrapper replaces only its unmodified-rule prompt
checker with a verifier that recomputes this experiment's controlled rules.
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.schema import SamplingConfig
from core.multimodal.datasets import load_local_dataset
from core.multimodal.prompts import format_question
from core.multimodal.textvqa_format_factorial import SHORT_FINAL_SUFFIX
from core.multimodal.textvqa_rule_control_online import POLICIES, VERSION, TextVQARuleControlRunner
from core.multimodal.tf_ttcl_direct_runner import (TEACHER_SYSTEM_OPEN, STUDENT_SYSTEM_OPEN,
    TUTOR_SYSTEM, POSITIVE_TEMPLATE_OPEN, NEGATIVE_TEMPLATE_OPEN, _rule_prompt,
    parse_variants, split_question_clause)
from scripts.audit_textvqa_selector_online import audit_selector


def verify_controls(rows, seed, policy, tokenizer, samples):
    if policy not in POLICIES:
        raise ValueError('Unknown rule-control policy')
    helper = TextVQARuleControlRunner.__new__(TextVQARuleControlRunner)
    helper.rule_policy, helper.tokenizer, helper.seed = policy, tokenizer, seed
    prior = {}
    if len(rows) != len(samples):
        raise ValueError('Sample count mismatch')
    for index, (row, sample) in enumerate(zip(rows, samples)):
        if row.get('sample_index') != index or row.get('labels_read') is not False:
            raise ValueError(f'{index}: invalid order or label boundary')
        for kind, bucket in row['retrieved_rules'].items():
            for rule in bucket:
                old = prior.get(rule['rule_id'])
                if (old is None or old['sample_index'] >= index or
                        (old['sample_index'], old['kind'], old['text']) !=
                        (rule['sample_index'], kind, rule['text'])):
                    raise ValueError(f'{index}: retrieval was not written by an earlier sample')
        controlled, gate = helper._control(row['retrieved_rules'], row['sample_id'])
        if row.get('controlled_rules') != controlled or row.get('rule_gate') != gate:
            raise ValueError(f'{index}: controlled rules or gate audit differ')
        for arm_name in ('vote_no_rewrite', 'select_no_memory', 'tf_ttcl_mm_direct'):
            calls = [c for c in row['arms'][arm_name]['calls'] if c.get('role')]
            if [c['role'] for c in calls] != ['teacher'] + [f'student_{n}' for n in range(1, 5)]:
                raise ValueError(f'{index}: candidate roles/order mismatch')
            rule_text = _rule_prompt(controlled) if arm_name == 'tf_ttcl_mm_direct' else ''
            for n, call in enumerate(calls):
                system = (TEACHER_SYSTEM_OPEN if n == 0 else STUDENT_SYSTEM_OPEN) + SHORT_FINAL_SUFFIX
                if rule_text:
                    system += '\n' + rule_text
                expected = SamplingConfig(temperature=0.0 if n == 0 else 0.7,
                    top_p=None if n == 0 else 0.9, max_tokens=256,
                    seed=seed + 100 * index + (0 if n == 0 else n + 1), request_logprobs=True)
                if call.get('system') != system or call['sampling'] != asdict(expected):
                    raise ValueError(f'{index}/{arm_name}/{n}: prompt or sampling mismatch')
        no_mem_calls = row['arms']['select_no_memory']['calls']
        full_calls = row['arms']['tf_ttcl_mm_direct']['calls']
        if (len(row['arms']['greedy']['calls']), len(row['arms']['vote_no_rewrite']['calls']),
                len(no_mem_calls), len(full_calls)) != (1, 5, 6, 8):
            raise ValueError(f'{index}: unexpected Tutor/candidate/summary call count')
        tutor_calls = [c for c in no_mem_calls if not c.get('role') and not c.get('kind')]
        full_tutors = [c for c in full_calls if not c.get('role') and not c.get('kind')]
        if len(tutor_calls) != 1 or full_tutors != tutor_calls:
            raise ValueError(f'{index}: missing or inconsistent Tutor call')
        prefix, question = split_question_clause(sample.question)
        tutor = tutor_calls[0]
        expected_tutor_sampling = SamplingConfig(temperature=0.7, top_p=0.9,
            max_tokens=384, seed=seed + 100 * index + 1)
        if (tutor.get('system'), tutor.get('user'), tutor.get('image_paths'), tutor.get('image_hashes'),
                tutor.get('sampling')) != (TUTOR_SYSTEM, question, [], [], asdict(expected_tutor_sampling)):
            raise ValueError(f'{index}: Tutor input or sampling mismatch')
        variants = [prefix + x for x in parse_variants(tutor['response']['content'], question)]
        if row['variants'] != variants:
            raise ValueError(f'{index}: Tutor variants differ from response')
        greedy = row['arms']['greedy']['calls']
        vote = [c for c in row['arms']['vote_no_rewrite']['calls'] if c.get('role')]
        nomem = [c for c in no_mem_calls if c.get('role')]
        if len(greedy) != 1 or greedy[0] != vote[0] or greedy[0] != nomem[0]:
            raise ValueError(f'{index}: shared teacher call mismatch')
        selected = row['arms']['tf_ttcl_mm_direct']
        by_role = {c['role']: c for c in full_calls if c.get('role')}
        positive = by_role.get(selected.get('positive_role'))
        negative = by_role.get(selected.get('negative_role'))
        summaries = [c for c in full_calls if c.get('kind')]
        if positive is None or negative is None or [c['kind'] for c in summaries] != ['positive', 'negative']:
            raise ValueError(f'{index}: positive/negative rule summaries mismatch')
        formatted = format_question(sample.question, sample.choices, sample.metadata)
        prompts = [POSITIVE_TEMPLATE_OPEN.format(question=formatted, answer=positive['response']['content']),
                   NEGATIVE_TEMPLATE_OPEN.format(question=formatted, correct_answer=positive['response']['content'],
                                                 incorrect_answer=negative['response']['content'])]
        for ordinal, (call, prompt) in enumerate(zip(summaries, prompts)):
            sampling = SamplingConfig(temperature=0.7, top_p=0.9, max_tokens=100,
                                      seed=seed + 100 * index + 20 + ordinal)
            if (call.get('system'), call.get('user'), call.get('image_paths'), call.get('image_hashes'),
                    call.get('sampling')) != ('', prompt, [], [], asdict(sampling)):
                raise ValueError(f'{index}: rule summary input or sampling mismatch')
        for rule in row['written_rules']:
            if rule['sample_index'] != index or rule['rule_id'] in prior:
                raise ValueError(f'{index}: invalid or duplicated rule write')
            prior[rule['rule_id']] = rule


def audit_rule_control(output, inference, count, embedder, tokenizer, *,
                       expected_fingerprint=None, expected_weight_sha=None):
    saved = json.loads((output / 'manifest.json').read_text())
    policy = saved.get('rule_control_policy')
    if (saved.get('rule_control_experiment') != VERSION or policy not in POLICIES or
            saved.get('similarity_selector') != 'full_response'):
        raise ValueError('Invalid rule-control experiment or baseline selector')
    examples = load_local_dataset(str(inference), 'textvqa', str(inference.parent), limit=count)
    if len(examples) != count or any(example.target is not None for example in examples):
        raise ValueError('Incomplete input or label exposure')
    samples = [example.sample for example in examples]
    # Scoped single-thread audit: the existing checker assumes original rules
    # appear verbatim in every memory prompt; verify_controls enforces the
    # controlled-prompt equivalent without weakening any other checks.
    with patch('scripts.audit_textvqa_selector_online.verify_online_and_sampling',
               side_effect=lambda rows, seed, _: verify_controls(rows, seed, policy, tokenizer, samples)):
        manifest, rows = audit_selector(output, inference, count, embedder)
    trace = next((output / 'response_trace').glob('*.json'), None)
    if trace is None:
        raise ValueError('Missing trace namespace')
    namespace = json.loads(trace.read_text())['request']['namespace']
    if namespace.get('rule_control_policy') != policy:
        raise ValueError('Trace rule-control policy mismatch')
    if expected_fingerprint is not None and namespace.get('source_fingerprint') != expected_fingerprint:
        raise ValueError('Trace source fingerprint mismatch')
    if expected_weight_sha is not None and namespace.get('weight_manifest_sha256') != expected_weight_sha:
        raise ValueError('Trace tokenizer/model manifest mismatch')
    return manifest, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'inference', 'embedding-path', 'model-path', 'weight-manifest'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--count', required=True, type=int)
    a = p.parse_args()
    from transformers import AutoTokenizer
    from core.multimodal.tf_ttcl_text_embedding import Qwen3TextEmbedder
    from scripts.run_hygon_tf_ttcl_direct import EMBEDDING_FILES
    from scripts.create_label_isolated_subset import sha256_file
    from scripts.run_hygon_scienceqa_parallel import verify_weights
    from scripts.run_hygon_textvqa_rule_control_online import control_source_fingerprint
    for name, expected in EMBEDDING_FILES.items():
        if sha256_file(a.embedding_path / name) != expected:
            raise ValueError('Embedding hash mismatch')
    verify_weights(a.model_path, a.weight_manifest)
    tokenizer = AutoTokenizer.from_pretrained(str(a.model_path), local_files_only=True)
    manifest, rows = audit_rule_control(a.output, a.inference, a.count,
        Qwen3TextEmbedder(str(a.embedding_path)), tokenizer,
        expected_fingerprint=control_source_fingerprint(Path(__file__).resolve().parents[1]),
        expected_weight_sha=sha256_file(a.weight_manifest))
    print(json.dumps({'status': 'passed', 'policy': manifest['rule_control_policy'],
                      'count': len(rows), 'labels_read': False}))


if __name__ == '__main__':
    main()
