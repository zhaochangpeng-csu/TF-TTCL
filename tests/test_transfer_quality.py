from __future__ import annotations

import unittest

from core.multimodal.transfer_quality import assess_rule


class TransferQualityTests(unittest.TestCase):
    def test_rejects_source_map_conclusion_conflicting_with_pseudo_pair(self):
        result = assess_rule({"positive_answer_text": "Dominica", "negative_answer_text": "Grenada",
                              "positive_rule": "The highlighted area corresponds to Grenada.",
                              "negative_rule": ""})
        self.assertFalse(result["accepted"])
        self.assertIn("negative_answer_asserted_in_positive_rule", result["reasons"])

    def test_keeps_a_conditional_physics_procedure(self):
        result = assess_rule({"positive_answer_text": "attract", "negative_answer_text": "repel",
                              "positive_rule": "Check the poles. If unlike poles face, they attract; "
                                               "if like poles face, they repel.",
                              "negative_rule": ""})
        self.assertTrue(result["accepted"])


if __name__ == "__main__":
    unittest.main()
