"""Fixed-history TextVQA rule-content × instruction-position shadow replay.

Exploratory counterfactual replay: no new rules are written.  Each item uses
only the audited rules, variants and candidate seeds of the completed anchor
stream. Neutral placeholders are controls for text length, not no memory.
"""
from __future__ import annotations

from copy import deepcopy

from .image_ops import sha256_file
from .textvqa_format_factorial import SHORT_FINAL_SUFFIX
from .tf_ttcl_direct_runner import (DirectRunner, STUDENT_SYSTEM_OPEN,
                                    TEACHER_SYSTEM_OPEN, _rule_prompt)

ARMS = tuple(f'{content}_{position}' for content in ('real', 'neutral')
             for position in ('before', 'after'))


def neutralize_rules(retrieved, tokenizer):
    """Preserve each bucket and approximate each rule's tokenizer length.

    Repeated neutral text intentionally contains no task-specific examples.
    Counts are measured both per rule and later from actual model usage.
    Tokenization is not necessarily additive across textual boundaries.
    """
    output, counts = deepcopy(retrieved), []
    for kind in ('positive', 'negative'):
        for item in output[kind]:
            target = len(tokenizer.encode(item['text'], add_special_tokens=False))
            phrase = 'Reference note. '
            choices = []
            # Longest candidate is safely larger than typical 100-token rules;
            # choose the closest measured token count, with deterministic ties.
            lo, hi = 1, max(2, target * 2 + 2)
            while lo <= hi:
                mid = (lo + hi) // 2
                text = (phrase * mid).strip()
                n = len(tokenizer.encode(text, add_special_tokens=False))
                choices.append((abs(n - target), n, mid, text))
                if n < target:
                    lo = mid + 1
                else:
                    hi = mid - 1
            _, actual, _, text = min(choices)
            counts.append({'rule_id': item['rule_id'], 'kind': kind,
                           'real_tokens': target, 'neutral_tokens': actual})
            item['text'] = text
    return output, counts


def system_prompt(role, retrieved, position):
    base = TEACHER_SYSTEM_OPEN if role == 'teacher' else STUDENT_SYSTEM_OPEN
    rules = _rule_prompt(retrieved)
    if position == 'before':
        return base + SHORT_FINAL_SUFFIX + ('\n' + rules if rules else '')
    if position == 'after':
        return base + ('\n' + rules if rules else '') + SHORT_FINAL_SUFFIX
    raise ValueError('Unknown reminder position')


def replay_item(backend, embedder, tokenizer, sample, anchor_row, index, seed):
    """Run 5 TF-TTCL candidates for each intervention; do not mutate history."""
    if (anchor_row['sample_index'], anchor_row['sample_id']) != (index, sample.sample_id):
        raise ValueError('Anchor sample identity mismatch')
    if anchor_row.get('original_question') != sample.question:
        raise ValueError('Anchor question mismatch')
    if [sha256_file(path) for path in sample.image_paths] != anchor_row['image_hashes']:
        raise ValueError('Anchor image hash mismatch')
    variants = anchor_row['variants']
    if len(variants) != 4 or any(not isinstance(v, str) or not v for v in variants):
        raise ValueError('Need four anchor tutor variants')
    retrieved = anchor_row['retrieved_rules']
    if set(retrieved) != {'positive', 'negative'} or any(
            rule['sample_index'] >= index for bucket in retrieved.values() for rule in bucket):
        raise ValueError('Shadow memory contains current or future rules')
    neutral, counts = neutralize_rules(retrieved, tokenizer)
    runner = DirectRunner.__new__(DirectRunner)
    runner.backend, runner.embedder, runner.seed = backend, embedder, seed
    output = {}
    for content in ('real', 'neutral'):
        bucket = retrieved if content == 'real' else neutral
        for position in ('before', 'after'):
            candidates, calls = [], []
            for n, question in enumerate([sample.question, *variants]):
                role = 'teacher' if n == 0 else f'student_{n}'
                candidate, call = runner._candidate(role, sample, question,
                    system_prompt(role, bucket, position), 0 if n == 0 else n + 1,
                    index, n != 0)
                candidates.append(candidate)
                calls.append(call)
            selection = runner._select(candidates, 'textvqa')
            output[f'{content}_{position}'] = {
                **runner._selection_record(selection), 'calls': calls,
                'usage': runner._usage(calls)}
    return {'sample_index': index, 'sample_id': sample.sample_id,
            'image_hashes': anchor_row['image_hashes'], 'variants': variants,
            'retrieved_rules': retrieved, 'neutral_rule_token_counts': counts,
            'anchor_no_memory': anchor_row['arms']['select_no_memory']['answer'],
            'anchor_memory': anchor_row['arms']['tf_ttcl_mm_direct']['answer'],
            'arms': output, 'labels_read': False, 'rules_written': False}
