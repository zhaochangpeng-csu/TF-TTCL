"""Lexical, label-free rule intervention primitives; not a semantic verifier.

Only the first sentence of each retrieved rule is examined. This module does
not read labels, choose candidates, write rules, or implement an online run.
"""
from __future__ import annotations

import hashlib
import re
from copy import deepcopy

from .textvqa_shadow import neutralize_rules

_SENTENCE = re.compile(r'[.!?\r\n]')
_ACTION = re.compile(r'\b(?:avoid|omit|skip|forgo)\b', re.I)
_TARGET = re.compile(r'\b(?:reasoning|explanation|explaining)\b', re.I)
_DIRECT = re.compile(r"\b(?:do\s+not|don't|never)\s+explain\b", re.I)
# Conservative exclusion for an obvious reversal: avoid omitting reasoning.
_DOUBLE = re.compile(
    r"\b(?:avoid\s+(?:(?:unnecessarily|needlessly)\s+)?(?:omitting|skipping|forgoing)"
    r"|(?:do\s+not|don't|never)\s+(?:omit|skip|forgo|avoid))\b", re.I)
_NEGATED_REQUIREMENT = re.compile(r'\b(?:without|no|lacking)\b', re.I)
_ERROR_SUFFIX = re.compile(r'^\s+(?:errors?|mistakes?|failures?|omissions?|problems?|gaps?)\b', re.I)
_OTHER_OBJECT = re.compile(r'\b(?:failing|repetition|repeat|repeating|referencing|refer|reference)\b', re.I)
_RESTRICTIVE_CLAUSE = re.compile(r'^\s+that\s+(?:misinterprets|repeats)\b', re.I)
_WINDOW = 60
GATE_VERSION = 'first_sentence_lexical_reasoning_conflict_v1'


def match_rule(text: str) -> dict | None:
    """Return exact lexical evidence, or None; no truth or semantics claimed."""
    if not isinstance(text, str):
        raise TypeError('rule text must be a string')
    text = text.strip().replace('’', "'")
    boundary = _SENTENCE.search(text)
    sentence = text[:boundary.start()] if boundary else text
    direct = _DIRECT.search(sentence)
    if direct:
        return {'pattern': 'direct_negated_explain', 'matched_text': direct.group(),
                'span': [direct.start(), direct.end()], 'first_sentence': sentence}
    blocked = list(_DOUBLE.finditer(sentence))
    matches = []
    for action in _ACTION.finditer(sentence):
        if any(b.start() <= action.start() < b.end() for b in blocked):
            continue
        for target in _TARGET.finditer(sentence):
            if target.start() < action.end():
                continue
            between = sentence[action.end():target.start()]
            suffix = sentence[target.end():]
            if (_NEGATED_REQUIREMENT.search(between) or _OTHER_OBJECT.search(between)
                    or _ERROR_SUFFIX.match(suffix) or _RESTRICTIVE_CLAUSE.match(suffix)):
                continue
            gap = len(between)
            if gap <= _WINDOW:
                start, end = min(action.start(), target.start()), max(action.end(), target.end())
                matches.append((start, end, gap))
    if not matches:
        return None
    start, end, gap = min(matches)
    return {'pattern': 'avoid_omit_skip_forgo_near_reasoning',
            'matched_text': sentence[start:end], 'span': [start, end],
            'gap_characters': gap, 'first_sentence': sentence}


def substitute(retrieved: dict, tokenizer, *, mode: str, seed: int, sample_id: str):
    """Replace targeted or matched-random rules with length-approximate text.

    The random control selects the same number in each bucket, by hashing
    seed, sample identity, bucket and unique rule ID. Random selection may
    overlap targeted rules. Bucket order, rule identities and count are kept.
    """
    if mode not in ('targeted', 'matched_random'):
        raise ValueError('unknown mode')
    if not isinstance(seed, int) or not isinstance(sample_id, str) or not sample_id:
        raise ValueError('integer seed and nonempty sample ID required')
    if set(retrieved) != {'positive', 'negative'}:
        raise ValueError('expected positive and negative buckets')
    ids = [r['rule_id'] for b in retrieved.values() for r in b]
    if len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError('rule IDs must be nonempty and unique')
    neutral, counts = neutralize_rules(retrieved, tokenizer)
    count_by_id = {c['rule_id']: c for c in counts}
    result, records, targets = deepcopy(retrieved), [], {}
    for kind in ('positive', 'negative'):
        bucket = retrieved[kind]
        hits = {r['rule_id']: m for r in bucket if (m := match_rule(r['text']))}
        targets[kind] = list(hits)
        if mode == 'targeted':
            selected = set(hits)
        else:
            def order(rule):
                key = f'{seed}\0{sample_id}\0{kind}\0{rule["rule_id"]}'.encode()
                return hashlib.sha256(key).digest(), rule['rule_id']
            selected = {r['rule_id'] for r in sorted(bucket, key=order)[:len(hits)]}
        for position, (source, replacement) in enumerate(zip(bucket, neutral[kind])):
            rid = source['rule_id']
            if rid not in selected:
                continue
            result[kind][position]['text'] = replacement['text']
            counts_one = count_by_id[rid]
            records.append({'rule_id': rid, 'bucket': kind, 'position': position,
                'source_sample_index': source.get('sample_index'), 'original_text': source['text'],
                'replacement_text': replacement['text'], 'real_tokens': counts_one['real_tokens'],
                'replacement_tokens': counts_one['neutral_tokens'], 'mode': mode, 'seed': seed,
                'sample_id': sample_id, 'lexical_match': hits.get(rid)})
    return result, {'gate_version': GATE_VERSION, 'mode': mode, 'seed': seed,
                    'sample_id': sample_id, 'targeted_ids_by_bucket': targets,
                    'replacements': records}
