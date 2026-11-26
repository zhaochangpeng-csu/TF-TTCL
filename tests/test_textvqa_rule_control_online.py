import json
import hashlib
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.multimodal.schema import GenerationResponse, MultimodalSample
from core.multimodal.datasets import load_local_dataset
from core.multimodal.replay import ResponseTraceBackend
from core.multimodal.textvqa_rule_control_online import TextVQARuleControlRunner
from core.multimodal.textvqa_selector_online import TextVQASelectorRunner
from scripts.audit_textvqa_rule_control_online import audit_rule_control, verify_controls
from scripts.create_label_isolated_subset import sha256_file
from scripts.run_hygon_textvqa_rule_control_online import load_rule_fresh
from tests.test_textvqa_format_factorial import Backend, Embedder
from tests.test_textvqa_shadow import Words


class GateBackend(Backend):
    def generate(self, system, user, images, sampling):
        if '<task>' in user:
            return GenerationResponse('Avoid unnecessary reasoning.', prompt_tokens=5, completion_tokens=1)
        return super().generate(system, user, images, sampling)


class RuleControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        image = self.root / 'image.jpg'; image.write_bytes(b'image fixture')
        self.samples = [MultimodalSample(str(i), 'textvqa', f'Question {i}?', (str(image),)) for i in range(3)]

    def runner(self, policy, path):
        return TextVQARuleControlRunner(GateBackend(), Embedder(), Words(), self.samples,
                                       path, 'source', 'model', 'embedding', seed=0, policy=policy)

    def test_exact_baseline_inputs_and_outputs_match_original_selector(self):
        baseline = TextVQASelectorRunner(GateBackend(), Embedder(), self.samples,
                    self.root/'selector', 'source', 'model', 'embedding', selector='full_response')
        control = self.runner('original', self.root/'original')
        for index, sample in enumerate(self.samples):
            expected = baseline._sample(sample, index)
            actual = control._sample(sample, index)
            self.assertEqual(actual.pop('controlled_rules'), expected['retrieved_rules'])
            gate = actual.pop('rule_gate')
            self.assertEqual(gate['mode'], 'original')
            self.assertEqual(actual, expected)
            self.assertEqual(actual['fixed_no_memory_selection'], expected['fixed_no_memory_selection'])

    def test_only_memory_prompt_is_substituted_and_history_stays_original(self):
        runs = {mode: self.runner(mode, self.root/mode) for mode in ('original', 'targeted', 'matched_random')}
        first = {mode: runner._sample(self.samples[0], 0) for mode, runner in runs.items()}
        for row in first.values():
            self.assertEqual(row['retrieved_rules'], {'positive': [], 'negative': []})
            self.assertFalse(row['labels_read'])
        second = {mode: runner._sample(self.samples[1], 1) for mode, runner in runs.items()}
        self.assertEqual(second['targeted']['retrieved_rules'], second['original']['retrieved_rules'])
        for mode in ('targeted', 'matched_random'):
            row = second[mode]
            self.assertEqual(len(row['rule_gate']['replacements']), 2)
            self.assertEqual(row['controlled_rules']['negative'][0]['rule_id'], 'negative_0')
            self.assertNotEqual(row['controlled_rules']['negative'][0]['text'], row['retrieved_rules']['negative'][0]['text'])
            self.assertEqual(row['arms']['select_no_memory'], second['original']['arms']['select_no_memory'])
            self.assertEqual(row['arms']['greedy'], second['original']['arms']['greedy'])
            for bucket in row['retrieved_rules'].values():
                self.assertTrue(all(rule['sample_index'] < 1 for rule in bucket))
            self.assertNotIn('Avoid unnecessary reasoning.', row['arms']['tf_ttcl_mm_direct']['calls'][0]['system'])
        self.assertEqual(len({id(runner.rules) for runner in runs.values()}), 3)
        self.assertTrue(all(rule.text == 'Avoid unnecessary reasoning.' for rule in runs['targeted'].rules))

    def test_cross_policy_resume_is_rejected(self):
        path = self.root/'resume'
        first = self.runner('targeted', path)
        first.run()
        original = first.predictions.read_bytes()
        self.runner('targeted', path).run()
        self.assertEqual(first.predictions.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, 'Resume'):
            self.runner('matched_random', path).run()
        with self.assertRaises(ValueError):
            self.runner('unknown', self.root/'unknown')

    def test_audit_recomputes_controls_and_rejects_tampering(self):
        source = self.root/'inference.jsonl'
        source.write_text(''.join(json.dumps({'id': str(i), 'question': f'Question {i}?',
                            'image': 'image.jpg'})+'\n' for i in range(3)))
        samples = [x.sample for x in load_local_dataset(str(source), 'textvqa')]
        for policy in ('original', 'targeted', 'matched_random'):
            output = self.root/('audit_'+policy)
            namespace = {'dataset': 'textvqa', 'selector': 'full_response', 'count': 3,
                'input_sha256': sha256_file(source), 'seed': 0, 'labels_read': False,
                'model_revision': 'model', 'embedding_sha256': 'embedding',
                'source_fingerprint': 'fixed', 'stack_id': 'stack', 'rule_control_policy': policy}
            trace = ResponseTraceBackend(GateBackend(), output/'response_trace', namespace, 'record')
            with patch('core.multimodal.tf_ttcl_direct_runner.subprocess.check_output', return_value='archive:fixed:stack'):
                runner = TextVQARuleControlRunner(trace, Embedder(), Words(), samples, output,
                    sha256_file(source), 'model', 'embedding', policy=policy)
            runner.run()
            manifest, rows = audit_rule_control(output, source, 3, Embedder(), Words())
            self.assertEqual(manifest['rule_control_policy'], policy)
            self.assertEqual(len(rows), 3)
            with self.assertRaisesRegex(ValueError, 'source fingerprint'):
                audit_rule_control(output, source, 3, Embedder(), Words(), expected_fingerprint='wrong')
            with self.assertRaisesRegex(ValueError, 'tokenizer/model manifest'):
                audit_rule_control(output, source, 3, Embedder(), Words(), expected_weight_sha='wrong')
            rows[1]['controlled_rules']['negative'][0]['text'] = 'tampered'
            runner.predictions.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError, 'controlled rules'):
                audit_rule_control(output, source, 3, Embedder(), Words())

    def test_input_requires_four_disjoint_pinned_streams(self):
        root = self.root/'fresh'; (root/'images').mkdir(parents=True)
        rows = []
        for i in range(400):
            image = f'im{i}'; (root/'images'/(image+'.jpg')).write_bytes(b'image')
            rows.append({'id': str(i), 'image_id': image, 'image': f'images/{image}.jpg', 'question': 'What?'})
        inference = root/'inference.jsonl'
        inference.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        images_sha = hashlib.sha256('\n'.join(f'{r["image_id"]}:{sha256_file(root/r["image"])}'
            for r in sorted(rows, key=lambda r: r['image_id'])).encode()).hexdigest()
        manifest = root/'manifest.json'
        manifest.write_text(json.dumps({'dataset_id': 'official/TextVQA_0.5.1', 'split': 'train',
            'count': 400, 'inference_sha256': sha256_file(inference), 'selected_ids': [r['id'] for r in rows],
            'image_count': 400, 'selected_images_sha256': images_sha}))
        old = []
        for i in range(4):
            path = self.root/f'old{i}.jsonl'
            path.write_text(json.dumps({'id': f'old{i}', 'image_id': f'oldimage{i}', 'question': 'What?'})+'\n')
            old.append(path)
        with patch('scripts.run_hygon_textvqa_rule_control_online.EXCLUSION_SHAS', {sha256_file(p) for p in old}):
            samples, _ = load_rule_fresh(inference, manifest, sha256_file(manifest), old, 20)
            self.assertEqual(len(samples), 20)
            with self.assertRaisesRegex(ValueError, 'four pinned'):
                load_rule_fresh(inference, manifest, sha256_file(manifest), old[:3], 20)
        old[0].write_text(json.dumps({'id': 'old0', 'image_id': 'im0', 'question': 'What?'})+'\n')
        with patch('scripts.run_hygon_textvqa_rule_control_online.EXCLUSION_SHAS', {sha256_file(p) for p in old}):
            with self.assertRaisesRegex(ValueError, 'overlap'):
                load_rule_fresh(inference, manifest, sha256_file(manifest), old, 20)

    def test_tutor_summary_and_no_rewrite_sampling_are_recomputed(self):
        runner = self.runner('targeted', self.root/'strict-audit')
        rows = [runner._sample(sample, i) for i, sample in enumerate(self.samples)]
        verify_controls(rows, 0, 'targeted', Words(), self.samples)

        bad = copy.deepcopy(rows)
        for arm in ('select_no_memory', 'tf_ttcl_mm_direct'):
            next(c for c in bad[1]['arms'][arm]['calls'] if not c.get('role') and not c.get('kind'))['user'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'Tutor input'):
            verify_controls(bad, 0, 'targeted', Words(), self.samples)

        bad = copy.deepcopy(rows); bad[1]['variants'][0] = 'Invented variant'
        with self.assertRaisesRegex(ValueError, 'Tutor variants'):
            verify_controls(bad, 0, 'targeted', Words(), self.samples)

        bad = copy.deepcopy(rows)
        next(c for c in bad[1]['arms']['tf_ttcl_mm_direct']['calls'] if c.get('kind') == 'negative')['user'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'rule summary input'):
            verify_controls(bad, 0, 'targeted', Words(), self.samples)

        bad = copy.deepcopy(rows)
        bad[1]['arms']['vote_no_rewrite']['calls'][1]['sampling']['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'sampling mismatch'):
            verify_controls(bad, 0, 'targeted', Words(), self.samples)


if __name__ == '__main__':
    unittest.main()
