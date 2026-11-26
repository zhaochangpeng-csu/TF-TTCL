import json
from pathlib import Path
import tempfile
import unittest
import hashlib

from core.multimodal.textvqa_rule_gate import match_rule, substitute
from scripts.inspect_textvqa_rule_gate_coverage import report


class SimpleTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class GateTests(unittest.TestCase):
    def test_first_sentence_action_precedes_target(self):
        for text in ('Avoid unnecessary reasoning.', 'Omit the explanation.',
                     'Skip explaining.', 'Forgo lengthy reasoning.'):
            with self.subTest(text=text):
                self.assertEqual(match_rule(text)['pattern'], 'avoid_omit_skip_forgo_near_reasoning')
        self.assertIsNone(match_rule('Use concise reasoning. Avoid explanation.'))
        self.assertIsNone(match_rule('Use concise reasoning.Avoid explanation.'))
        self.assertIsNone(match_rule('Avoid ' + 'x' * 59 + ' reasoning.'))
        self.assertIsNotNone(match_rule('Avoid ' + 'x' * 58 + ' reasoning.'))

    def test_direct_negation_and_double_negatives(self):
        for text in ('Do not explain your work.', "Don't explain.", 'Never explain.'):
            with self.subTest(text=text):
                self.assertEqual(match_rule(text)['pattern'], 'direct_negated_explain')
        for text in ('Avoid omitting reasoning.', 'Avoid skipping the explanation.',
                     'Do not skip explaining.', "Don't omit reasoning.",
                     'Never skip reasoning.', 'Do not avoid explaining.',
                     'Use reasoning to avoid mistakes.', 'Avoid reasoning errors.',
                     'Avoid providing an answer without context or reasoning.',
                     'Avoid failing to provide reasoning.',
                     'Avoid unnecessary repetition in the reasoning section.',
                     'Avoid directly referencing the image in the reasoning step.',
                     'Avoid unnecessary reasoning that misinterprets the question.',
                     'Explain briefly. Avoid skipping reasoning.'):
            with self.subTest(text=text): self.assertIsNone(match_rule(text))

    def test_target_and_control_preserve_buckets_and_lengths(self):
        rules = {'positive': [
                    {'rule_id': 'p0', 'text': 'Avoid unnecessary reasoning.', 'sample_index': 0},
                    {'rule_id': 'p1', 'text': 'Read the image.', 'sample_index': 1},
                    {'rule_id': 'p2', 'text': 'Do not explain details.', 'sample_index': 2}],
                 'negative': [
                    {'rule_id': 'n0', 'text': 'Skip the explanation.', 'sample_index': 0},
                    {'rule_id': 'n1', 'text': 'Do not skip reasoning.', 'sample_index': 1}]}
        original = json.dumps(rules, sort_keys=True)
        targeted, info = substitute(rules, SimpleTokenizer(), mode='targeted', seed=7, sample_id='x')
        control, random_info = substitute(rules, SimpleTokenizer(), mode='matched_random', seed=7, sample_id='x')
        again, again_info = substitute(rules, SimpleTokenizer(), mode='matched_random', seed=7, sample_id='x')
        self.assertEqual(json.dumps(rules, sort_keys=True), original)
        self.assertEqual((control, random_info), (again, again_info))
        self.assertEqual([len(targeted[k]) for k in ('positive', 'negative')], [3, 2])
        for kind in rules:
            self.assertEqual([r['rule_id'] for r in rules[kind]], [r['rule_id'] for r in targeted[kind]])
            self.assertEqual(sum(r['bucket'] == kind for r in info['replacements']),
                             sum(r['bucket'] == kind for r in random_info['replacements']))
        self.assertEqual(set(info['targeted_ids_by_bucket']['positive']), {'p0', 'p2'})
        self.assertEqual(info['targeted_ids_by_bucket']['negative'], ['n0'])
        self.assertTrue(all(abs(r['real_tokens']-r['replacement_tokens']) <= 1
                            for r in info['replacements']))
        self.assertEqual(targeted['negative'][1]['text'], rules['negative'][1]['text'])

    def test_duplicate_ids_and_unknown_modes_are_rejected(self):
        r = {'positive': [{'rule_id': 'x', 'text': 'Avoid reasoning.'}],
             'negative': [{'rule_id': 'x', 'text': 'Reason.'}]}
        with self.assertRaisesRegex(ValueError, 'unique'): substitute(r, SimpleTokenizer(), mode='targeted', seed=0, sample_id='1')
        with self.assertRaisesRegex(ValueError, 'unknown'): substitute(r, SimpleTokenizer(), mode='bad', seed=0, sample_id='1')

    def test_coverage_is_pinned_ordered_and_reads_rules_not_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'predictions.jsonl'
            rows = [{'sample_index': 0, 'sample_id': 's0', 'labels_read': False,
                     'retrieved_rules': {'positive': [], 'negative': []}},
                    {'sample_index': 1, 'sample_id': 's1', 'labels_read': False,
                     'retrieved_rules': {'positive': [{'rule_id': 'p0', 'sample_index': 0,
                                                    'text': 'Avoid reasoning.'}], 'negative': []}}]
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            found = report(p, digest, 2)
            self.assertEqual(found['matched_retrieval_instances'], 1)
            self.assertEqual(found['samples_with_match'], 1)
            with self.assertRaisesRegex(ValueError, 'hash'): report(p, '0'*64, 2)
            rows[1]['retrieved_rules']['positive'][0]['sample_index'] = 1
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError, 'future'):
                report(p, hashlib.sha256(p.read_bytes()).hexdigest(), 2)


if __name__ == '__main__': unittest.main()
