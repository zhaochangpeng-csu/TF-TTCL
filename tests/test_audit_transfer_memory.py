from __future__ import annotations

import unittest

from evaluate.multimodal.audit_transfer_memory import (
    internal_capitalized_tokens, literal_mention, summarize,
)


def row(index, sid, image, answer, retrieved=(), written=None):
    arms = {arm: {"answer": answer, "retrieved": [], "written": None}
            for arm in ("none", "noisy", "gated_weighted", "random")}
    arms["noisy"].update(retrieved=list(retrieved), written=written)
    return {"sample_index": index, "sample_id": sid,
            "question": "Which country is highlighted?", "choices": ["Grenada", "Barbados"],
            "metadata": {"subject": "social science", "topic": "geography",
                         "category": "Maps", "skill": "Identify a highlighted country"},
            "image_hashes": [image], "arms": arms}


class MemoryAuditTests(unittest.TestCase):
    def test_internal_capitalized_screen_is_only_a_heuristic(self):
        self.assertEqual(internal_capitalized_tokens(
            "Identify the items that Tyrone and Mason desire."), ["Mason", "Tyrone"])
        self.assertEqual(internal_capitalized_tokens("Check the image for matching shapes."), [])

    def test_literal_choice_and_different_image_hazard_are_measured(self):
        entry = {"sample_index": 0, "sample_id": "old", "positive_answer_text": "Grenada",
                 "positive_rule": "The highlighted country is Grenada.", "negative_rule": ""}
        source = row(0, "old", "image1", "a", written=entry)
        target = row(1, "new", "image2", "b", retrieved=[entry])
        result = summarize([source, target])
        self.assertTrue(literal_mention(entry["positive_rule"], "Grenada"))
        self.assertFalse(literal_mention("caterpillar", "cat"))
        self.assertEqual(result["content"]["map_task_source_answer_mentioned"], 1)
        self.assertEqual(result["retrieval"]["noisy"]["same_skill_events"], 1)
        self.assertEqual(result["retrieval"]["noisy"]["source_map_answer_on_different_image_events"], 1)

    def test_future_retrieval_fails_closed(self):
        future = {"sample_index": 1, "sample_id": "new"}
        source = row(0, "old", "image1", "a", retrieved=[future])
        target = row(1, "new", "image2", "b")
        with self.assertRaisesRegex(ValueError, "absent from the past"):
            summarize([source, target])

    def test_positive_rule_repeating_negative_answer_is_counted(self):
        entry = {"sample_index": 0, "sample_id": "old", "positive_answer_text": "Dominica",
                 "negative_answer_text": "Grenada",
                 "positive_rule": "The highlighted area corresponds to Grenada.", "negative_rule": ""}
        source = row(0, "old", "image1", "d", written=entry)
        target = row(1, "new", "image2", "c", retrieved=[entry])
        result = summarize([source, target])
        self.assertEqual(result["content"]["negative_answer_in_positive_rule"], 1)
        self.assertEqual(result["retrieval"]["noisy"]["negative_answer_in_positive_rule_events"], 1)


if __name__ == "__main__":
    unittest.main()
