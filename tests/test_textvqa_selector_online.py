import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from core.multimodal.datasets import load_local_dataset
from core.multimodal.schema import GenerationResponse
from core.multimodal.replay import ResponseTraceBackend
from core.multimodal.tf_ttcl_direct_selection import ScoredCandidate
from core.multimodal.textvqa_selector_online import CachedEmbedder, TextVQASelectorRunner, choose
from scripts.audit_textvqa_selector_online import audit_selector
from scripts.create_label_isolated_subset import sha256_file
from scripts.run_hygon_textvqa_selector_online import load_fresh
from scripts.score_textvqa_selector_online import score_records, score
from tests.test_textvqa_format_factorial import Backend, Embedder


class SelectionEmbedder:
    def encode(self, text, *, query=False):
        # Similarity rankings deliberately disagree between text and answer.
        return {'teacher raw': (1., 0.), 'student a raw': (1., 0.),
                'student b raw': (0.9, 0.1), 'student c raw': (0., 1.),
                'student d raw': (-1., 0.), 'teacher': (1., 0.),
                'a': (0., 1.), 'b': (-1., 0.), 'c': (1., 0.),
                'd': (0.9, 0.1)}[text]


class SelectorOnlineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'image.jpg').write_bytes(b'fixture')
        self.source = self.root / 'inference.jsonl'
        self.source.write_text(''.join(json.dumps({'id': str(i), 'question': f'Question {i}?', 'image': 'image.jpg'})+'\n' for i in range(2)))
        self.samples = [x.sample for x in load_local_dataset(str(self.source), 'textvqa')]

    def make_runner(self, selector, output):
        trace = ResponseTraceBackend(Backend(), output / 'response_trace',
            {'dataset': 'textvqa', 'selector': selector, 'count': 2,
             'input_sha256': sha256_file(self.source), 'seed': 0, 'labels_read': False,
             'model_revision': 'model', 'embedding_sha256': 'embedding',
             'source_fingerprint': 'fixed', 'stack_id': 'stack'}, 'record')
        with patch('core.multimodal.tf_ttcl_direct_runner.subprocess.check_output', return_value='archive:fixed:stack'):
            return TextVQASelectorRunner(trace, Embedder(), self.samples, output,
                                         sha256_file(self.source), 'model', 'embedding', selector=selector)

    def test_answer_and_full_text_can_choose_different_partitions(self):
        candidates = []
        for i, (raw, answer) in enumerate(zip(['teacher raw', 'student a raw', 'student b raw', 'student c raw', 'student d raw'],
                                               ['teacher', 'a', 'b', 'c', 'd'])):
            role = 'teacher' if i == 0 else f'student_{i}'
            response = GenerationResponse(raw, completion_tokens=1, logprobs=[-.2 if i == 0 else -.1])
            candidates.append(ScoredCandidate.from_response(role, answer, response))
        full = choose(candidates, SelectionEmbedder(), 'full_response')
        short = choose(candidates, SelectionEmbedder(), 'normalized_answer')
        self.assertEqual(full.positive_roles, ('teacher', 'student_1', 'student_2'))
        self.assertEqual(short.positive_roles, ('teacher', 'student_3', 'student_4'))

    def test_streams_audit_independently_and_forbid_cross_policy_resume(self):
        for selector in ('full_response', 'normalized_answer'):
            output = self.root / selector
            runner = self.make_runner(selector, output); runner.run()
            manifest, rows = audit_selector(output, self.source, 2, Embedder())
            self.assertEqual(manifest['similarity_selector'], selector)
            self.assertEqual(set(rows[0]['fixed_no_memory_selection']), {'full_response', 'normalized_answer'})
            self.assertTrue(rows[1]['retrieved_rules']['positive'])
            with self.assertRaisesRegex(ValueError, 'Resume'):
                self.make_runner('normalized_answer' if selector == 'full_response' else 'full_response', output).run()

    def test_audit_rejects_forged_similarity(self):
        output = self.root / 'forged'
        runner = self.make_runner('normalized_answer', output); runner.run()
        rows = [json.loads(line) for line in runner.predictions.read_text().splitlines()]
        rows[0]['fixed_no_memory_selection']['full_response']['similarity_ranking'][0][1] += 0.01
        runner.predictions.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        with self.assertRaisesRegex(ValueError, 'similarity'):
            audit_selector(output, self.source, 2, Embedder())

    def test_score_fixed_candidate_reuses_recorded_output(self):
        output = self.root / 'score'
        runner = self.make_runner('full_response', output); runner.run()
        rows = [json.loads(line) for line in runner.predictions.read_text().splitlines()]
        class Evaluator:
            def eval_pred_list(self, rows):
                return float(rows[0]['pred_answer'].lower() in rows[0]['gt_answers'])
        summary, comparisons = score_records(rows, rows, [['red'] * 10] * 2, Evaluator())
        self.assertIn('fixed_candidate_answer_minus_full', comparisons)
        self.assertEqual(summary['greedy']['mean'], 1.0)
        self.assertEqual(comparisons['online_answer_minus_full_memory']['improved_items'], 0)

    def test_exact_embedding_cache_distinguishes_query_mode(self):
        class Count:
            def __init__(self): self.calls = []
            def encode(self, text, *, query=False): self.calls.append((text, query)); return (float(query),)
        backend = Count(); cached = CachedEmbedder(backend)
        self.assertEqual(cached.encode('x'), cached.encode('x'))
        self.assertEqual(cached.encode('x', query=True), (1.,))
        self.assertEqual(backend.calls, [('x', False), ('x', True)])

    def test_smoke_cannot_be_scored(self):
        with self.assertRaisesRegex(ValueError, 'Smoke'):
            score(SimpleNamespace(count=20), None, None)

    def test_tampered_trace_and_future_rules_fail_audit(self):
        output = self.root / 'trace'
        runner = self.make_runner('full_response', output); runner.run()
        trace = next((output / 'response_trace').glob('*.json'))
        data = json.loads(trace.read_text()); data['response']['content'] += 'modified'
        trace.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'trace'):
            audit_selector(output, self.source, 2, Embedder())
        output2 = self.root / 'future'
        runner = self.make_runner('full_response', output2); runner.run()
        rows = [json.loads(line) for line in runner.predictions.read_text().splitlines()]
        rows[1]['retrieved_rules']['positive'][0]['sample_index'] = 1
        runner.predictions.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError, 'future'):
            audit_selector(output2, self.source, 2, Embedder())

    def test_fresh_data_rejects_prior_question_or_image_overlap(self):
        root = self.root / 'fresh'; (root / 'images').mkdir(parents=True)
        rows = []
        for i in range(400):
            image = f'im{i}'
            (root / 'images' / (image + '.jpg')).write_bytes(b'image')
            rows.append({'id': str(i), 'image_id': image, 'image': f'images/{image}.jpg', 'question': 'What?'})
        inference = root / 'inference.jsonl'
        inference.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        images_sha = hashlib.sha256('\n'.join(f'{r["image_id"]}:{sha256_file(root / r["image"])}'
            for r in sorted(rows, key=lambda r: r['image_id'])).encode()).hexdigest()
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps({'dataset_id': 'official/TextVQA_0.5.1', 'split': 'train', 'count': 400,
            'inference_sha256': sha256_file(inference), 'selected_ids': [r['id'] for r in rows],
            'image_count': 400, 'selected_images_sha256': images_sha}))
        old = []
        for i in range(3):
            f = self.root / f'old{i}.jsonl'
            f.write_text(json.dumps({'id': f'old{i}', 'image_id': f'oldimage{i}', 'question': 'What?'})+'\n')
            old.append(f)
        with patch('scripts.run_hygon_textvqa_selector_online.PRIOR_SHAS', {sha256_file(p) for p in old}):
            samples, _ = load_fresh(inference, manifest, sha256_file(manifest), old, 20)
            self.assertEqual(len(samples), 20)
        old[0].write_text(json.dumps({'id': 'old0', 'image_id': 'im0', 'question': 'What?'})+'\n')
        with patch('scripts.run_hygon_textvqa_selector_online.PRIOR_SHAS', {sha256_file(p) for p in old}):
            with self.assertRaisesRegex(ValueError, 'overlap'):
                load_fresh(inference, manifest, sha256_file(manifest), old, 20)


if __name__ == '__main__':
    unittest.main()
