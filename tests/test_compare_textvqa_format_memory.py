import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import asdict

from scripts.compare_textvqa_format_memory import (CONFIG_KEYS, compare, format_ok,
                                                    policy_prompt_hash, verify_online_and_sampling)
from core.multimodal.schema import SamplingConfig
from core.multimodal.textvqa_format_factorial import FACTORIAL_VERSION, SHORT_FINAL_SUFFIX
from core.multimodal.tf_ttcl_direct_runner import TEACHER_SYSTEM_OPEN, STUDENT_SYSTEM_OPEN, _rule_prompt

BASE_HASH = policy_prompt_hash('original')
SHORT_HASH = policy_prompt_hash('short_final')


class FormatMemoryComparisonTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        baseline, strengthened = root / 'baseline', root / 'strengthened'
        baseline.mkdir(); strengthened.mkdir()
        inference = root / 'inference.jsonl'
        inference.write_text('no labels are in this input\n')
        ids = [str(i) for i in range(400)]
        labels = root / 'labels.jsonl'
        labels.write_text(''.join(json.dumps({'id': i, 'answers': ['yes'] * 10}) + '\n' for i in ids))
        subset = root / 'subset.json'
        subset.write_text(json.dumps({'dataset_id': 'official/TextVQA_0.5.1', 'split': 'train',
            'count': 400, 'selected_ids': ids, 'inference_sha256': hashlib.sha256(inference.read_bytes()).hexdigest(),
            'labels_sha256': hashlib.sha256(labels.read_bytes()).hexdigest()}))
        meta = {key: 'same' for key in CONFIG_KEYS}
        meta.update(dataset='textvqa', labels_read=False, ordered_ids=ids, seed=0)
        def arm(answer, index, suffix):
            calls = [{'role': 'teacher', 'normalized_answer': answer, 'response': {'content': 'Reasoning: evidence.\nFinal answer: '+answer},
                      'system': TEACHER_SYSTEM_OPEN + suffix,
                      'sampling': asdict(SamplingConfig(temperature=0.0, max_tokens=256, seed=100*index, request_logprobs=True))}]
            calls.extend({'role': f'student_{n}', 'normalized_answer': answer, 'response': {'content': 'Reasoning: evidence.\nFinal answer: '+answer},
                          'system': STUDENT_SYSTEM_OPEN + suffix,
                          'sampling': asdict(SamplingConfig(temperature=.7, top_p=.9, max_tokens=256,
                                                           seed=100*index+n+1, request_logprobs=True))}
                         for n in range(1, 5))
            return {'answer': answer, 'positive_role': 'teacher',
                    'calls': calls,
                    'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'latency_seconds': 1}}
        def rows(yes_a, yes_b, suffix):
            return [{'sample_id': i, 'retrieved_rules': {'positive': [], 'negative': []}, 'written_rules': [],
                     'arms': {'select_no_memory': arm('yes' if n < yes_a else 'no', n, suffix),
                              'tf_ttcl_mm_direct': arm('yes' if n < yes_b else 'no', n, suffix)}} for n,i in enumerate(ids)]
        data = {baseline: ({**meta, 'prompt_sha256': BASE_HASH, 'format_experiment': FACTORIAL_VERSION,
                            'format_policy': 'original'}, rows(100, 160, '')),
                strengthened: ({**meta, 'prompt_sha256': SHORT_HASH, 'format_experiment': FACTORIAL_VERSION,
                                'format_policy': 'short_final'}, rows(120, 240, SHORT_FINAL_SUFFIX))}
        return baseline, strengthened, inference, subset, labels, data

    def test_factorial_effect_and_interaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            b, s, i, sub, labels, data = self.fixture(tmp)
            result = compare(b, s, i, sub, labels, BASE_HASH, SHORT_HASH,
                             audit_fn=lambda path, *_: data[path])
            self.assertAlmostEqual(result['scores']['baseline_no_memory']['local_vqa_score'], .25)
            self.assertAlmostEqual(result['contrasts']['format_gain_no_memory']['mean_and_95pct_ci'][0], .05)
            self.assertAlmostEqual(result['contrasts']['format_gain_memory']['mean_and_95pct_ci'][0], .20)
            self.assertAlmostEqual(result['contrasts']['interaction']['mean_and_95pct_ci'][0], .15)
            self.assertEqual(result['scores']['strengthened_memory']['logical_prompt_tokens'], 1200)
            self.assertEqual(result['scores']['strengthened_memory']['selected_output_format_rate'], 1)
            self.assertTrue(result['cost_matching']['memory']['within_5_percent'])

    def test_no_targets_opened_when_audit_or_config_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            b, s, i, sub, labels, data = self.fixture(tmp)
            labels.unlink()
            def audit(path, *_):
                if path == s: raise ValueError('audit fails')
                return data[path]
            with self.assertRaisesRegex(ValueError, 'audit fails'):
                compare(b, s, i, sub, labels, BASE_HASH, SHORT_HASH, audit_fn=audit)
            data[s][0]['seed'] = 1
            with self.assertRaisesRegex(ValueError, 'configuration: seed'):
                compare(b, s, i, sub, labels, BASE_HASH, SHORT_HASH, audit_fn=lambda p,*_: data[p])

    def test_prompt_and_ids_checked_before_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            b, s, i, sub, labels, data = self.fixture(tmp)
            labels.unlink()
            with self.assertRaisesRegex(ValueError, 'Frozen prompt hashes'):
                compare(b, s, i, sub, labels, 'c'*64, SHORT_HASH, audit_fn=lambda p,*_: data[p])
            data[s][1][0]['sample_id'] = 'another'
            with self.assertRaisesRegex(ValueError, 'Ordered IDs differ'):
                compare(b, s, i, sub, labels, BASE_HASH, SHORT_HASH, audit_fn=lambda p,*_: data[p])

    def test_invalid_sampling_and_memory_rejected_before_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            b, s, i, sub, labels, data = self.fixture(tmp)
            labels.unlink()
            data[s][1][0]['arms']['select_no_memory']['calls'][1]['sampling']['seed'] = 99
            with self.assertRaisesRegex(ValueError, 'sampling changed'):
                compare(b, s, i, sub, labels, BASE_HASH, SHORT_HASH, audit_fn=lambda p,*_: data[p])
            data[s][1][0]['arms']['select_no_memory']['calls'][1]['sampling']['seed'] = 2
            data[s][1][0]['retrieved_rules']['positive'] = [{'rule_id': 'fake', 'sample_index': -1, 'text': 'rule'}]
            with self.assertRaisesRegex(ValueError, 'Retrieval not produced'):
                compare(b, s, i, sub, labels, BASE_HASH, SHORT_HASH, audit_fn=lambda p,*_: data[p])

    def test_modified_candidate_prompt_and_manifest_policy_rejected_before_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            b, s, i, sub, labels, data = self.fixture(tmp)
            labels.unlink()
            data[s][1][3]['arms']['tf_ttcl_mm_direct']['calls'][2]['system'] += '\ninjected'
            with self.assertRaisesRegex(ValueError, 'Candidate system prompt changed'):
                compare(b, s, i, sub, labels, BASE_HASH, SHORT_HASH, audit_fn=lambda p,*_: data[p])
            data[s][1][3]['arms']['tf_ttcl_mm_direct']['calls'][2]['system'] = STUDENT_SYSTEM_OPEN + SHORT_FINAL_SUFFIX
            data[s][0]['format_policy'] = 'original'
            with self.assertRaisesRegex(ValueError, 'Manifest format policies'):
                compare(b, s, i, sub, labels, BASE_HASH, SHORT_HASH, audit_fn=lambda p,*_: data[p])

    def test_memory_prompt_is_rebuilt_from_previously_written_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            b, _, _, _, _, data = self.fixture(tmp)
            rows = data[b][1]
            rule = {'rule_id': 'p0', 'sample_index': 0, 'kind': 'positive', 'text': 'read the image'}
            rows[0]['written_rules'] = [rule]
            rows[1]['retrieved_rules']['positive'] = [rule]
            suffix = '\n' + _rule_prompt(rows[1]['retrieved_rules'])
            for call in rows[1]['arms']['tf_ttcl_mm_direct']['calls']:
                call['system'] += suffix
            verify_online_and_sampling(rows, 0, 'original')
            rows[1]['arms']['tf_ttcl_mm_direct']['calls'][0]['system'] += ' altered'
            with self.assertRaisesRegex(ValueError, 'Candidate system prompt changed'):
                verify_online_and_sampling(rows, 0, 'original')

    def test_surface_format(self):
        self.assertTrue(format_ok('Reasoning: yes.\nFinal answer: 42'))
        self.assertFalse(format_ok('Final answer: 42\nExplanation follows'))
        self.assertFalse(format_ok('Final answer: \n'))
        self.assertFalse(format_ok('Final answer: 42\nFinal answer: 43'))


if __name__ == '__main__':
    unittest.main()
