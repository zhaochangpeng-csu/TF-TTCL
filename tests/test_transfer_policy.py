from __future__ import annotations

import unittest
from types import SimpleNamespace

from core.multimodal.schema import Candidate, GenerationResponse
from core.multimodal.transfer_policy import answer_content, deterministic_random_past, majority_pair, write_policy


def candidate(name, answer, logprobs=()):
    return Candidate(name, name, GenerationResponse("Final answer: " + answer, logprobs=list(logprobs)), answer, "q", ())


class TransferPolicyTests(unittest.TestCase):
    def test_wrong_or_uncertified_consensus_is_not_silently_filtered(self):
        pair = majority_pair([candidate("one", "a"), candidate("two", "a"), candidate("three", "b")])
        self.assertEqual(pair.positive.normalized_answer, "a")
        self.assertEqual(pair.negative.normalized_answer, "b")
        self.assertTrue(write_policy("noisy", pair).write)
        self.assertEqual(write_policy("noisy", pair).trust, 1.0)
        self.assertFalse(write_policy("gated_weighted", pair).write)
        self.assertFalse(write_policy("none", pair, evidence_passed=True, evidence_answer="a").write)

    def test_evidence_may_not_change_pseudo_label_and_confound_memory(self):
        pair = majority_pair([candidate("one", "a"), candidate("two", "a"), candidate("three", "b")])
        self.assertFalse(write_policy("gated_weighted", pair, evidence_passed=True, evidence_answer="b").write)
        gated = write_policy("gated_weighted", pair, evidence_passed=True, evidence_answer="a")
        self.assertTrue(gated.write)
        self.assertAlmostEqual(gated.trust, 2 / 3)

    def test_all_same_and_all_distinct(self):
        same = majority_pair([candidate("one", "a"), candidate("two", "a")])
        self.assertIsNone(same.negative)
        self.assertTrue(same.writable)
        distinct = majority_pair([candidate("one", "a"), candidate("two", "b")])
        self.assertFalse(distinct.writable)
        self.assertFalse(write_policy("noisy", distinct).write)

    def test_min_ppl_and_missing_logprob_fallback(self):
        pair = majority_pair([candidate("a1", "a", [-0.8]), candidate("b1", "b", [-0.1]),
                              candidate("a2", "a", [-0.4]), candidate("b2", "b", [-0.2])])
        self.assertEqual(pair.positive.candidate_id, "b1")
        self.assertEqual(pair.negative.candidate_id, "a2")
        missing = majority_pair([candidate("a1", "a"), candidate("b1", "b"), candidate("a2", "a"), candidate("b2", "b")])
        self.assertEqual(missing.positive.candidate_id, "a1")

    def test_option_letter_not_semantic_answer(self):
        c = candidate("one", "b")
        self.assertEqual(answer_content(c, ["attract", "repel"]), "repel")
        self.assertEqual(answer_content(c, ["repel", "attract"]), "attract")
        with self.assertRaises(ValueError):
            answer_content(candidate("bad", "z"), ["yes", "no"])

    def test_random_control_same_write_pool_and_temporal_safe_resume(self):
        items = [SimpleNamespace(sample_index=i, experience_id=f"e{i}") for i in range(10)]
        selected = deterministic_random_past(items, 5, 1, 4)
        self.assertTrue(all(item.sample_index < 5 for item in selected))
        self.assertEqual([i.experience_id for i in selected],
                         [i.experience_id for i in deterministic_random_past(list(reversed(items)), 5, 1, 4)])
        pair = majority_pair([candidate("one", "a"), candidate("two", "a")])
        self.assertEqual(write_policy("noisy", pair), write_policy("random", pair))


if __name__ == "__main__":
    unittest.main()
