import math
import unittest

from core.multimodal.schema import GenerationResponse
from core.multimodal.tf_ttcl_direct_selection import (
    LogprobUnavailable, ScoredCandidate, select_closed, select_open,
    strict_generation_ppl,
)


def candidates(answers, nlls):
    return [ScoredCandidate.from_response(role, answer, GenerationResponse(
        content=f"Final answer: {answer}", completion_tokens=2,
        logprobs=[-nll, -nll], finish_reason="stop"))
        for role, answer, nll in zip(
            ("teacher", "student_1", "student_2", "student_3", "student_4"),
            answers, nlls)]


class DirectSelectionTests(unittest.TestCase):
    def test_ppl_requires_complete_finite_generated_token_scores(self):
        self.assertAlmostEqual(strict_generation_ppl(GenerationResponse(
            content="ok", completion_tokens=2, logprobs=[-1.0, -2.0])), math.exp(1.5))
        for values, count in (([], 1), ([-1.0], 2), ([float("nan")], 1),
                              ([float("-inf")], 1), ([0.1], 1)):
            with self.subTest(values=values, count=count), self.assertRaises(LogprobUnavailable):
                strict_generation_ppl(GenerationResponse(content="x", completion_tokens=count,
                                                          logprobs=values))

    def test_closed_all_same_selects_min_ppl_and_positive_rule_only(self):
        selection = select_closed(candidates(["a"] * 5, [3, 2, 1, 4, 5]))
        self.assertEqual(selection.positive.role, "student_2")
        self.assertIsNone(selection.negative)
        self.assertTrue(selection.summarize_positive)
        self.assertFalse(selection.summarize_negative)

    def test_closed_all_distinct_keeps_teacher_without_rules(self):
        selection = select_closed(candidates(list("abcde"), [5, 1, 1, 1, 1]))
        self.assertEqual(selection.positive.role, "teacher")
        self.assertIsNone(selection.negative)
        self.assertFalse(selection.summarize_positive)

    def test_closed_majority_and_hard_negative_use_min_ppl(self):
        selection = select_closed(candidates(["a", "a", "a", "b", "c"], [3, 2, 1, .5, .1]))
        self.assertEqual(selection.positive.role, "student_2")
        self.assertEqual(selection.negative.role, "student_4")
        self.assertEqual(selection.reason, "largest_cluster")

    def test_closed_tied_largest_cluster_uses_global_min_ppl(self):
        selection = select_closed(candidates(["a", "a", "b", "b", "c"], [2, 3, 4, 1, .1]))
        self.assertEqual(selection.positive.role, "student_3")
        self.assertEqual(selection.negative.role, "student_4")
        self.assertEqual(selection.reason, "tied_cluster_min_ppl")

    def test_open_teacher_top_two_and_bottom_two(self):
        selection = select_open(candidates(["x", "a", "b", "c", "d"],
                                           [4, 3, 2, 1, .5]), [.1, .9, .8, .2])
        self.assertEqual(selection.positive_roles, ("teacher", "student_2", "student_3"))
        self.assertEqual(selection.negative_roles, ("student_4", "student_1"))
        self.assertEqual(selection.positive.role, "student_3")
        self.assertEqual(selection.negative.role, "student_4")
        with self.assertRaises(ValueError):
            select_open(candidates(["x"] * 5, [1] * 5), [.1, .2])


if __name__ == "__main__":
    unittest.main()
