import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from core.multimodal.datasets import load_local_dataset
from core.multimodal.prompts import format_question
from core.multimodal.schema import GenerationResponse, SamplingConfig
from core.multimodal.tf_ttcl_direct_runner import TEACHER_SYSTEM
from scripts.compare_hygon_scienceqa_greedy import compare
from scripts.create_label_isolated_subset import sha256_file


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inference = self.root / 'inference.jsonl'
        self.reference = self.root / 'reference.jsonl'
        self.labels = self.root / 'labels.jsonl'
        self.shard = self.root / 'shard.jsonl'
        inputs = [{'id': str(i), 'question': f'Q{i}', 'choices': ['x', 'y'],
                   'image': f'{i}.png'} for i in range(3)]
        self.write(self.inference, inputs)
        self.write(self.labels, [{**item, 'answer': 0} for item in inputs])
        examples = load_local_dataset(str(self.inference), 'scienceqa_img')
        old, new = [], []
        for i, example in enumerate(examples):
            old_answer = ['a', 'b', 'a'][i]
            new_answer = ['a', 'a', 'b'][i]
            call = {'system': TEACHER_SYSTEM,
                    'user': format_question(example.sample.question, example.sample.choices,
                                            example.sample.metadata),
                    'image_hashes': [f'hash{i}'],
                    'sampling': asdict(SamplingConfig(temperature=0.0, max_tokens=256,
                                                       seed=100*i, request_logprobs=True)),
                    'normalized_answer': old_answer}
            old.append({'sample_index': i, 'sample_id': str(i), 'labels_read': False,
                        'image_hashes': [f'hash{i}'],
                        'arms': {'greedy': {'answer': old_answer, 'calls': [call]}}})
            response = GenerationResponse(content='Final answer: '+new_answer,
                                          completion_tokens=1, logprobs=[-0.5])
            new.append({'sample_index': i, 'sample_id': str(i), 'labels_read': False,
                        'image_hashes': [f'hash{i}'], 'answer': new_answer,
                        'reference_answer': old_answer, 'response': asdict(response),
                        'ppl': math.exp(0.5)})
        self.new = new
        self.write(self.reference, old)
        self.write(self.shard, new)

    @staticmethod
    def write(path, rows):
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def run_compare(self, labels=None):
        return compare(self.inference, self.reference, [self.shard], labels or self.labels,
                       sha256_file(self.inference), sha256_file(self.reference),
                       sha256_file(self.labels), expected_count=3)

    def test_paired_statistics_and_alignment(self):
        result = self.run_compare()
        self.assertEqual((result['nvidia_correct'], result['hygon_correct']), (2, 2))
        self.assertEqual((result['answers_equal'], result['improved'], result['worsened']),
                         (1, 1, 1))
        self.assertEqual(result['paired_delta_ci'][0], 0)

    def test_incomplete_output_rejected_before_labels_are_opened(self):
        self.write(self.shard, self.new[:2])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.run_compare(self.root / 'missing_labels.jsonl')

    def test_duplicate_and_wrong_image_hash_rejected(self):
        self.write(self.shard, [self.new[0], self.new[0], self.new[2]])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.run_compare()
        self.new[1]['image_hashes'] = ['wrong']
        self.write(self.shard, self.new)
        with self.assertRaisesRegex(ValueError, 'image hash mismatch'):
            self.run_compare()

    def test_label_hash_mismatch_rejected(self):
        self.labels.write_text(self.labels.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'labels hash mismatch'):
            compare(self.inference, self.reference, [self.shard], self.labels,
                    sha256_file(self.inference), sha256_file(self.reference), 'wrong',
                    expected_count=3)


if __name__ == '__main__':
    unittest.main()
