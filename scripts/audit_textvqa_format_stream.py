#!/usr/bin/env python3
"""Audit one label-free TextVQA format-policy stream before scoring."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.textvqa_format_factorial import FACTORIAL_VERSION
from scripts.audit_tf_ttcl_mm_direct import audit, casebook
from scripts.compare_textvqa_format_memory import policy_prompt_hash, verify_online_and_sampling


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--inference', type=Path, required=True)
    p.add_argument('--policy', choices=('original', 'short_final'), required=True)
    p.add_argument('--count', choices=(20, 400), type=int, required=True)
    p.add_argument('--casebook', type=Path)
    args = p.parse_args()
    manifest, rows = audit(args.output, args.inference, args.count)
    if (manifest.get('format_experiment'), manifest.get('format_policy'), manifest.get('prompt_sha256')) != (
            FACTORIAL_VERSION, args.policy, policy_prompt_hash(args.policy)):
        raise ValueError('Policy or prompt hash does not match the frozen protocol')
    verify_online_and_sampling(rows, manifest['seed'], args.policy)
    if args.casebook:
        with args.casebook.open('x', encoding='utf-8') as handle:
            handle.write(casebook(rows))
    print(json.dumps({'audit': 'passed', 'policy': args.policy, 'samples': len(rows),
                      'labels_read': False, 'rule_writes': sum(len(row['written_rules']) for row in rows)}))


if __name__ == '__main__':
    main()
