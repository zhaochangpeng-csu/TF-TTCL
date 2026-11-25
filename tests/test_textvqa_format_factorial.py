import json
import tempfile
import unittest
from pathlib import Path

from core.multimodal.schema import GenerationResponse, MultimodalSample
from core.multimodal.textvqa_format_factorial import (SHORT_FINAL_SUFFIX, TextVQAFormatRunner,
                                                      build_format_runners)
from core.multimodal.tf_ttcl_direct_runner import DirectRunner


class Embedder:
    def encode(self, text, *, query=False):
        return (1.0, 0.0) if query else (0.0, 1.0)


class Backend:
    model_name = 'fake-vlm'

    def generate(self, system, user, images, sampling):
        if 'Rewrite only the question' in system:
            return GenerationResponse(json.dumps({'questions': [f'Question variant {i}?' for i in range(4)]}),
                                      prompt_tokens=5, completion_tokens=1)
        if '<task>' in user:
            return GenerationResponse('STRICT_RULE' if 'STRICT_PROBE' in user else 'ORIGINAL_RULE',
                                      prompt_tokens=5, completion_tokens=1)
        prefix = 'STRICT_PROBE' if SHORT_FINAL_SUFFIX in system else 'ORIGINAL_PROBE'
        answer = 'red' if sampling.temperature == 0 or sampling.seed % 5 < 3 else 'blue'
        return GenerationResponse(f'Reasoning: {prefix} visible evidence. I read the image carefully.\n'
                                  f'Final answer: {answer}', prompt_tokens=10,
                                  completion_tokens=2, logprobs=[-.1, -.2], finish_reason='stop')


class FormatFactorialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        image = self.root / 'image.jpg'
        image.write_bytes(b'image-fixture')
        self.samples = [MultimodalSample(str(i), 'textvqa', f'Question {i}?', (str(image),))
                        for i in range(2)]

    def test_original_policy_runs_original_algorithm_and_prompt(self):
        original = DirectRunner(Backend(), Embedder(), self.samples, self.root / 'base',
                                'source', 'model', 'embedding')
        comparison = TextVQAFormatRunner(Backend(), Embedder(), self.samples, self.root / 'factorial',
                                          'source', 'model', 'embedding', policy='original')
        self.assertEqual(original.teacher_system, comparison.teacher_system)
        self.assertEqual(original.student_system, comparison.student_system)
        self.assertEqual(original.prompt_sha, comparison.prompt_sha)
        self.assertEqual(original._sample(self.samples[0], 0), comparison._sample(self.samples[0], 0))

    def test_policy_changes_answering_prompt_only_and_records_frozen_hash(self):
        runs = build_format_runners(Backend(), Embedder(), self.samples, self.root / 'pair',
                                     'source', 'model', 'embedding')
        base, short = runs['original'], runs['short_final']
        self.assertNotEqual(base.prompt_sha, short.prompt_sha)
        self.assertEqual(short.teacher_system, base.teacher_system + SHORT_FINAL_SUFFIX)
        self.assertEqual(short.student_system, base.student_system + SHORT_FINAL_SUFFIX)
        self.assertEqual(base.positive_template, short.positive_template)
        self.assertEqual(base.negative_template, short.negative_template)
        for runner in runs.values():
            runner.run()
            manifest = json.loads(runner.manifest.read_text())
            self.assertEqual(manifest['format_policy'], runner.config['format_policy'])
            self.assertEqual(manifest['prompt_sha256'], runner.prompt_sha)
            self.assertFalse(manifest['labels_read'])
        base_rows = [json.loads(x) for x in base.predictions.read_text().splitlines()]
        short_rows = [json.loads(x) for x in short.predictions.read_text().splitlines()]
        self.assertEqual(base_rows[0]['variants'], short_rows[0]['variants'])
        self.assertTrue(all(r['text'] == 'ORIGINAL_RULE' for r in base_rows[1]['retrieved_rules']['positive']))
        self.assertTrue(all(r['text'] == 'STRICT_RULE' for r in short_rows[1]['retrieved_rules']['positive']))
        self.assertTrue(base_rows[1]['retrieved_rules']['positive'])
        self.assertTrue(short_rows[1]['retrieved_rules']['positive'])
        for row in short_rows:
            self.assertEqual(set(row['arms']), {'greedy', 'vote_no_rewrite', 'select_no_memory', 'tf_ttcl_mm_direct'})
            for arm in row['arms'].values():
                for call in arm['calls']:
                    if call.get('role'):
                        self.assertIn(SHORT_FINAL_SUFFIX, call['system'])
                        self.assertEqual(len(call['response']['logprobs']), call['response']['completion_tokens'])

    def test_rejects_invalid_policy_dataset_and_cross_policy_resume(self):
        with self.assertRaises(ValueError):
            TextVQAFormatRunner(Backend(), Embedder(), self.samples, self.root / 'bad',
                                 'source', 'model', 'embedding', policy='invalid')
        with self.assertRaises(ValueError):
            TextVQAFormatRunner(Backend(), Embedder(), [MultimodalSample('1', 'scienceqa_img', 'Q?')],
                                 self.root / 'bad', 'source', 'model', 'embedding')
        output = self.root / 'resume'
        TextVQAFormatRunner(Backend(), Embedder(), self.samples, output,
                             'source', 'model', 'embedding', policy='original').run()
        with self.assertRaisesRegex(ValueError, 'Resume'):
            TextVQAFormatRunner(Backend(), Embedder(), self.samples, output,
                                 'source', 'model', 'embedding', policy='short_final').run()


if __name__ == '__main__':
    unittest.main()
