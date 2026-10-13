from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.multimodal.schema import MultimodalSample
from core.multimodal.transfer_placebo import PlaceboRunner, placebo_context, replace_rule_block
from core.multimodal.transfer_runner import TransferRunner
from evaluate.multimodal.compare_placebo import audit_placebo, compare
from tests.test_transfer_runner import FakeBackend


class TransferPlaceboTests(unittest.TestCase):
    def test_placebo_matches_rule_word_counts_without_source_facts(self):
        rule = {"positive_rule": "The highlighted country is Grenada on the map.",
                "negative_rule": "Do not say Grenada from the old image."}
        context = placebo_context([rule])
        self.assertNotIn("Grenada", context)
        self.assertNotIn("country", context)
        self.assertEqual(len(context.split("Check: ")[1].split("\n")[0].split()),
                         len(rule["positive_rule"].split()))
        self.assertIn("Possible failure:", context)

    def test_exact_source_prefix_and_resumable_label_free_replay(self):
        samples = [MultimodalSample(str(i), "scienceqa_img", f"same question {i}?", (),
                                    ("yes", "no"), {"question_type": "multi_choice"})
                   for i in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = FakeBackend()
            TransferRunner(backend, samples, root / "source", "sourcehash", "revision").run()
            runner = PlaceboRunner(backend, samples, root / "source", root / "placebo",
                                   "sourcehash", "revision", source_expected_n=2)
            runner.run()
            manifest, rows, source_rows = audit_placebo(root / "source", root / "placebo", 2, 2)
            self.assertEqual(manifest["status"], "completed")
            self.assertTrue(rows[0]["reused_identical_source_response"])
            self.assertFalse(rows[1]["reused_identical_source_response"])
            self.assertEqual(rows[1]["input_user_text"], replace_rule_block(source_rows[1]))
            self.assertNotIn(source_rows[1]["arms"]["noisy"]["retrieved"][0]["positive_rule"],
                             rows[1]["input_user_text"])
            calls = len(backend.calls)
            PlaceboRunner(backend, samples, root / "source", root / "placebo", "sourcehash",
                          "revision", source_expected_n=2).run()
            self.assertEqual(len(backend.calls), calls)
            (root / "placebo" / "manifest.json").write_text(json.dumps({"status": "running"}))
            labels = root / "labels.jsonl"
            labels.write_text("INVALID_AND_MUST_NOT_BE_READ")
            with self.assertRaisesRegex(ValueError, "Incomplete or mismatched"):
                compare(root / "source", root / "placebo", labels, expected_n=400)


if __name__ == "__main__":
    unittest.main()
