import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.score_textvqa_rule_control_online import gate_summary, score, score_records, trace_cost


class Evaluator:
    def eval_pred_list(self, records):
        return float(records[0]['pred_answer'].lower() in records[0]['gt_answers'])


def make_row(answer, policy='original'):
    response = {'content': 'Reasoning: Evidence is visible.\nFinal answer: ' + answer}
    call = {'role': 'teacher', 'normalized_answer': answer, 'response': response}
    arm = {'answer': answer, 'positive_role': 'teacher', 'calls': [call]}
    return {'sample_index': 0, 'sample_id': 'one', 'arms': {
        'greedy': {'calls': [call]}, 'select_no_memory': arm, 'tf_ttcl_mm_direct': arm},
        'retrieved_rules': {'positive': [{'rule_id': 'p'}], 'negative': [{'rule_id': 'n'}]},
        'rule_gate': {'targeted_ids_by_bucket': {'positive': ['p'], 'negative': []},
                      'replacements': [] if policy == 'original' else [
                          {'bucket': 'positive', 'real_tokens': 4, 'replacement_tokens': 5}]}}


class RuleControlScoreTests(unittest.TestCase):
    def test_paired_scoring_and_replacements(self):
        streams = {'original': [make_row('no')], 'targeted': [make_row('yes', 'targeted')],
                   'matched_random': [make_row('no', 'matched_random')]}
        summaries, comparisons, segments = score_records(streams, [['yes'] * 10], Evaluator())
        self.assertEqual(summaries['targeted_memory']['mean'], 1)
        self.assertEqual(summaries['targeted_memory']['missing_final_answer_marker'], 0)
        self.assertEqual(comparisons['targeted_minus_original_memory']['improved_items'], 1)
        self.assertEqual(comparisons['targeted_minus_matched_random_memory']['improved_items'], 1)
        self.assertEqual(segments['targeted'][0]['memory_minus_no_memory'], 0)
        g = gate_summary(streams['targeted'])['by_bucket']
        self.assertEqual(g['positive']['replacements'], 1)
        self.assertEqual(g['positive']['real_tokens_replaced'], 4)
        self.assertEqual(g['positive']['replacement_tokens'], 5)
        self.assertEqual(g['negative']['replacements'], 0)

    def test_empty_gate_is_counted_without_inference_of_causality(self):
        row = make_row('yes')
        row['retrieved_rules'] = {'positive': [], 'negative': []}
        row['rule_gate'] = {'targeted_ids_by_bucket': {'positive': [], 'negative': []}, 'replacements': []}
        result = gate_summary([row])
        self.assertEqual(result['samples_with_any_match'], 0)
        self.assertEqual(result['samples_with_any_replacement'], 0)
        self.assertEqual(result['by_bucket']['positive']['samples_with_matches'], 0)

    def test_rejects_misaligned_streams(self):
        row = make_row('yes'); changed = make_row('yes'); changed['sample_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'order or IDs'):
            score_records({'original': [row], 'targeted': [changed], 'matched_random': [row]},
                          [['yes'] * 10], Evaluator())

    def test_smoke_and_failed_audit_never_read_labels(self):
        with self.assertRaisesRegex(ValueError, 'Smoke'):
            score(SimpleNamespace(count=20), None, None, None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            args = SimpleNamespace(count=400, inference=path/'inference', subset_manifest=path/'manifest',
                subset_manifest_sha256='a'*64, excluded_inference=[], weight_manifest=path/'weights',
                original=path/'original', targeted=path/'targeted', matched_random=path/'random',
                labels=path/'must-not-read')
            with patch('scripts.score_textvqa_rule_control_online.load_rule_fresh', return_value=([], {})), \
                 patch('scripts.score_textvqa_rule_control_online.control_source_fingerprint', return_value='source'), \
                 patch('scripts.score_textvqa_rule_control_online.sha256_file', return_value='weight'), \
                 patch('scripts.score_textvqa_rule_control_online.audit_rule_control', side_effect=ValueError('bad audit')):
                with self.assertRaisesRegex(ValueError, 'bad audit'):
                    score(args, SimpleNamespace(), None, Evaluator())

    def test_physical_trace_checks_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); (path/'response_trace').mkdir()
            (path/'response_trace'/'wrong.json').write_text(json.dumps({'request': {}, 'response': {}}))
            with self.assertRaisesRegex(ValueError, 'hash'):
                trace_cost(path, {}, 'source', 'weight')

    def test_scoring_reads_labels_only_after_all_three_audits(self):
        from scripts.compare_textvqa_format_memory import CONFIG_KEYS
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            labels = path / 'labels.jsonl'
            labels.write_text(''.join(json.dumps({'id': str(i), 'answers': ['yes'] * 10}) + '\n'
                                      for i in range(400)))
            data = {'labels_sha256': hashlib.sha256(labels.read_bytes()).hexdigest()}
            paths = {policy: path / policy for policy in ('original', 'targeted', 'matched_random')}
            for folder in paths.values():
                folder.mkdir(); (folder/'predictions.jsonl').write_text('fixed')
            (path/'inference').write_text('inference'); (path/'weights').write_text('weights')
            common = {key: 'same' for key in CONFIG_KEYS}
            common.update({'rule_control_experiment': 'v1', 'rule_gate_version': 'v1',
                           'similarity_selector': 'full_response', 'seed': 0})
            audit_calls = []
            def audited(folder, inference, count, embedder, tokenizer, **kwargs):
                self.assertTrue(labels.exists())
                policy = folder.name; audit_calls.append(policy)
                rows = []
                for i in range(400):
                    row = make_row('yes', policy); row.update(sample_index=i, sample_id=str(i))
                    usage = {'prompt_tokens': 10, 'completion_tokens': 2, 'latency_seconds': 1.0}
                    for arm in row['arms'].values(): arm['usage'] = usage
                    rows.append(row)
                return {**common, 'rule_control_policy': policy, 'config_sha256': policy}, rows
            def cheap_contrast(v): return {'mean_and_95pct_ci': (sum(v)/len(v), 0, 0), 'paired_sign_randomization_p': 1}
            args = SimpleNamespace(count=400, inference=path/'inference', subset_manifest=path/'manifest',
                subset_manifest_sha256='a'*64, excluded_inference=[], weight_manifest=path/'weights',
                labels=labels, **paths)
            with patch('scripts.score_textvqa_rule_control_online.load_rule_fresh', return_value=([], data)), \
                 patch('scripts.score_textvqa_rule_control_online.control_source_fingerprint', return_value='source'), \
                 patch('scripts.score_textvqa_rule_control_online.audit_rule_control', side_effect=audited), \
                 patch('scripts.score_textvqa_rule_control_online.trace_cost', return_value={}), \
                 patch('scripts.score_textvqa_rule_control_online.contrast', side_effect=cheap_contrast):
                result = score(args, SimpleNamespace(), None, Evaluator())
            self.assertEqual(audit_calls, ['original', 'targeted', 'matched_random'])
            self.assertEqual(result['n'], 400)
            self.assertEqual(result['metrics']['targeted_memory']['mean'], 1)
            self.assertTrue(result['cost_matching']['targeted_tf_ttcl_mm_direct_vs_matched_random']['within_5_percent'])
            self.assertEqual([x['start_index'] for x in result['stream_segments_100']['targeted']], [0, 100, 200, 300])
            self.assertTrue(all(x['count'] == 100 and x['unchanged_items'] == 100
                                for x in result['stream_segments_100']['targeted']))

if __name__ == '__main__': unittest.main()
