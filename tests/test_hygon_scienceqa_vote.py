import unittest

from core.multimodal.schema import GenerationResponse, MultimodalSample
from core.multimodal.tf_ttcl_direct_selection import select_closed
from scripts.hygon_scienceqa_vote import candidate


class VoteCandidateTests(unittest.TestCase):
    def setUp(self):
        self.sample = MultimodalSample('1', 'scienceqa_img', 'Which?', choices=('one', 'two', 'three', 'four'))

    def test_unparsed_candidates_are_distinct_per_role(self):
        response = GenerationResponse('not a valid option', completion_tokens=1, logprobs=[-.2])
        first = candidate('student_1', response, self.sample)
        second = candidate('student_2', response, self.sample)
        self.assertEqual(first.answer, '__unparsed_student_1__')
        self.assertNotEqual(first.answer, second.answer)

    def test_closed_tied_cluster_uses_lowest_ppl(self):
        def make(role, answer, logprob):
            return candidate(role, GenerationResponse(f'Final answer: {answer}', completion_tokens=1,
                                                       logprobs=[logprob]), self.sample)
        candidates = [make('teacher', 'A', -.5), make('student_1', 'A', -.4),
                      make('student_2', 'B', -.2), make('student_3', 'B', -.3),
                      make('student_4', 'C', -.1)]
        selection = select_closed(candidates)
        self.assertEqual(selection.positive.answer, 'b')
        self.assertEqual(selection.positive.role, 'student_2')
        self.assertEqual(selection.reason, 'tied_cluster_min_ppl')


if __name__ == '__main__':
    unittest.main()
