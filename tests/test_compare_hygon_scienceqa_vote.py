import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from core.multimodal.datasets import load_local_dataset
from core.multimodal.prompts import format_question
from core.multimodal.schema import GenerationResponse, SamplingConfig
from core.multimodal.tf_ttcl_direct_runner import STUDENT_SYSTEM, TEACHER_SYSTEM
from core.multimodal.tf_ttcl_direct_selection import select_closed
from scripts.compare_hygon_scienceqa_vote import candidate, compare
from scripts.create_label_isolated_subset import sha256_file


class VoteComparisonTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.inference, self.ref, self.greedy, self.shard, self.labels = [root / x for x in
            ('input.jsonl', 'ref.jsonl', 'greedy.jsonl', 'shard.jsonl', 'labels.jsonl')]
        raw = [{'id': str(i), 'question': 'Q?', 'choices': ['x', 'y'], 'image': 'img.png'} for i in range(3)]
        self.write(self.inference, raw)
        self.write(self.labels, [{**row, 'answer': 0} for row in raw])
        examples = load_local_dataset(str(self.inference), 'scienceqa_img')
        refs, grows, votes = [], [], []
        for i, item in enumerate(examples):
            sample = item.sample
            hashes = [f'image{i}']
            def response(answer):
                return GenerationResponse(content='Final answer: '+answer, completion_tokens=1, logprobs=[-0.5])
            def sampling(n):
                return SamplingConfig(temperature=0.7 if n else 0.0, top_p=0.9 if n else None,
                                      max_tokens=256, seed=100*i+n+1 if n else 100*i,
                                      request_logprobs=True)
            calls = [{'role': 'teacher' if n == 0 else f'student_{n}',
                      'system': TEACHER_SYSTEM if n == 0 else STUDENT_SYSTEM,
                      'user': format_question(sample.question, sample.choices, sample.metadata),
                      'image_hashes': hashes, 'sampling': asdict(sampling(n)),
                      'response': asdict(response('a')), 'ppl': math.exp(0.5)} for n in range(5)]
            refs.append({'sample_index': i, 'sample_id': str(i), 'labels_read': False,
                         'image_hashes': hashes, 'arms': {'greedy': {'answer': 'a'},
                           'vote_no_rewrite': {'answer': 'a', 'calls': calls}}})
            grow = {'sample_index': i, 'sample_id': str(i), 'image_hashes': hashes,
                    'labels_read': False, 'answer': 'a', 'reference_answer': 'a',
                    'response': asdict(response('a')), 'ppl': math.exp(0.5)}
            grows.append(grow)
            answers = ['a']*4 if i != 1 else ['b']*4
            students = [{'role': f'student_{n}', 'answer': answers[n-1], 'ppl': math.exp(0.5),
                         'sampling': asdict(sampling(n)), 'response': asdict(response(answers[n-1]))}
                        for n in range(1, 5)]
            selection = select_closed([candidate('teacher', response('a'), sample)] +
                [candidate(f'student_{n}', response(answers[n-1]), sample) for n in range(1, 5)])
            votes.append({'sample_index': i, 'sample_id': str(i), 'labels_read': False,
                          'image_hashes': hashes, 'teacher': grow, 'students': students,
                          'answer': selection.positive.answer, 'positive_role': selection.positive.role,
                          'negative_role': selection.negative.role if selection.negative else None,
                          'positive_roles': selection.positive_roles, 'negative_roles': selection.negative_roles,
                          'reason': selection.reason, 'reference_answer': 'a'})
        self.votes = votes
        for p, data in ((self.ref, refs), (self.greedy, grows), (self.shard, votes)):
            self.write(p, data)

    @staticmethod
    def write(path, values):
        path.write_text(''.join(json.dumps(x)+'\n' for x in values))

    def run_compare(self, labels=None):
        return compare(self.inference, self.ref, self.greedy, [self.shard], labels or self.labels,
                       sha256_file(self.inference), sha256_file(self.ref), sha256_file(self.greedy),
                       sha256_file(self.labels), expected_count=3)

    def test_scores(self):
        result = self.run_compare()
        self.assertEqual((result['nvidia_correct'], result['hygon_correct']), (3, 2))
        self.assertEqual((result['answers_equal'], result['worsened']), (2, 1))

    def test_incomplete_or_duplicate_rejected_before_labels(self):
        self.write(self.shard, self.votes[:2])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.run_compare(self.labels.parent/'missing')
        self.write(self.shard, [self.votes[0], self.votes[0], self.votes[2]])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.run_compare()

    def test_invalid_sampling_or_selection_rejected(self):
        self.votes[0]['students'][1]['sampling']['seed'] = -1
        self.write(self.shard, self.votes)
        with self.assertRaisesRegex(ValueError, 'Student mismatch'):
            self.run_compare()
        self.votes[0]['students'][1]['sampling']['seed'] = 3
        self.votes[0]['answer'] = 'b'
        self.write(self.shard, self.votes)
        with self.assertRaisesRegex(ValueError, 'Selection mismatch'):
            self.run_compare()

    def test_incomplete_logprobs_and_modified_teacher_rejected(self):
        self.votes[0]['students'][0]['response']['logprobs'] = []
        self.write(self.shard, self.votes)
        with self.assertRaisesRegex(ValueError, 'generated token logprobs'):
            self.run_compare()
        self.votes[0]['students'][0]['response']['logprobs'] = [-0.5]
        self.votes[0]['teacher']['answer'] = 'b'
        self.write(self.shard, self.votes)
        with self.assertRaisesRegex(ValueError, 'teacher mismatch'):
            self.run_compare()


if __name__ == '__main__':
    unittest.main()
