import json
import math
import tempfile
import unittest
from pathlib import Path

from core.multimodal.image_ops import sha256_file
from core.multimodal.schema import MultimodalSample
from scripts.replay_textvqa_answer_similarity import replay_candidate_arm, request_key


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        image = self.root / 'image.jpg'
        image.write_bytes(b'test-image')
        self.sample = MultimodalSample('id', 'textvqa', 'what?', (str(image),), (), {})
        self.namespace = {'dataset': 'textvqa'}
        self.model = 'model'
        self.trace_dir = self.root / 'traces'
        self.trace_dir.mkdir()
        self.calls = []
        for i, role in enumerate(('teacher', 'student_1', 'student_2', 'student_3', 'student_4')):
            q = 'what?' if i == 0 else f'variant {i}'
            lp = [-0.1 if i == 3 else -1.0]
            response = {'content': f'Final answer: answer{i}', 'prompt_tokens': 5,
                        'completion_tokens': 1, 'logprobs': lp, 'latency_seconds': 0.1,
                        'finish_reason': 'stop'}
            sampling = {'temperature': 0.0, 'output_schema': None, 'request_logprobs': True}
            self.calls.append({'role': role, 'question_variant': q, 'system': 'system', 'user': q,
                'image_paths': [str(image)], 'image_hashes': [sha256_file(image)],
                'sampling': sampling, 'response': response, 'normalized_answer': f'answer{i}',
                'ppl': math.exp(-lp[0])})
        self.row = {'sample_index': 0, 'sample_id': 'id', 'dataset': 'textvqa',
            'original_question': 'what?', 'choices': [], 'variants': [f'variant {i}' for i in range(1,5)],
            'image_hashes': [sha256_file(image)], 'arms': {'select_no_memory': {
                'calls': self.calls, 'answer': 'answer0', 'positive_role': 'teacher',
                'negative_role': 'student_3', 'reason': 'teacher_similarity_top_half',
                'similarity_ranking': [('student_1', .9), ('student_2', .8), ('student_3', .2), ('student_4', .1)],
                'open_answer_diagnostic': {'teacher_answer': 'answer0',
                  'student_answers': {f'student_{i}':f'answer{i}' for i in range(1,5)},
                  'answer_similarity': {'student_1': .1, 'student_2': .2, 'student_3': .9, 'student_4': .8}}}}}
        for call in self.calls:
            self.write_trace(call)

    def write_trace(self, call):
        sampling = dict(call['sampling']); sampling.pop('output_schema')
        request = {'namespace': self.namespace, 'model': self.model, 'system_prompt': call['system'],
                   'user_text': call['user'], 'image_hashes': call['image_hashes'], 'sampling': sampling}
        (self.trace_dir / (request_key(request) + '.json')).write_text(json.dumps({'request':request,'response':call['response']}))

    def run_replay(self):
        return replay_candidate_arm(self.row, self.sample, 'select_no_memory', self.trace_dir, self.namespace, self.model)

    def test_fixed_candidates_partition_by_answer_similarity(self):
        result = self.run_replay()
        self.assertEqual(result['original_positive_role'], 'teacher')
        self.assertEqual(result['replay_positive_role'], 'student_3')
        self.assertEqual(result['replay_answer'], 'answer3')
        self.assertEqual(len(result['candidate_trace_keys']), 5)
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_tampered_prompt_rejected(self):
        self.calls[2]['user'] = 'another input'
        with self.assertRaises(ValueError): self.run_replay()

    def test_missing_logprob_is_not_imputed(self):
        self.calls[3]['response']['logprobs'] = []
        self.write_trace(self.calls[3])
        with self.assertRaises(ValueError): self.run_replay()

    def test_tampered_trace_response_rejected(self):
        self.calls[0]['response']['content'] = 'Final answer: modified'
        with self.assertRaises(ValueError): self.run_replay()

    def test_tied_similarities_preserve_student_order(self):
        self.row['arms']['select_no_memory']['open_answer_diagnostic']['answer_similarity'] = {
            f'student_{i}': 0.5 for i in range(1, 5)}
        result = self.run_replay()
        self.assertEqual(result['replay_positive_roles'], ['teacher', 'student_1', 'student_2'])

    def test_mismatched_similarity_rejected(self):
        self.row['arms']['select_no_memory']['open_answer_diagnostic']['student_answers']['student_1'] = 'wrong'
        with self.assertRaises(ValueError): self.run_replay()

    def test_image_hash_is_verified(self):
        Path(self.sample.image_paths[0]).write_bytes(b'changed')
        with self.assertRaises(ValueError): self.run_replay()


if __name__ == '__main__':
    unittest.main()
