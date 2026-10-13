from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.multimodal.schema import MultimodalSample
from core.multimodal.transfer_runner import TransferRunner
from evaluate.multimodal.transfer_casebook import build_casebook, render_markdown
from scripts.create_label_isolated_subset import sha256_file
from tests.test_transfer_runner import FakeBackend


class TransferCasebookTests(unittest.TestCase):
    def test_completed_casebook_inspects_exact_prompt_and_optional_offline_target(self):
        sample = MultimodalSample("a", "scienceqa_img", "What is shown?", (), ("yes", "no"),
                                  {"question_type": "multi_choice"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            inference = root / "inference.jsonl"
            inference.write_text(json.dumps({"id": "a", "question": sample.question,
                                             "choices": list(sample.choices),
                                             "question_type": "multi_choice"}) + "\n")
            labels = root / "labels.jsonl"
            labels.write_text(json.dumps({"id": "a", "question": sample.question,
                                          "choices": list(sample.choices), "answer": 0}) + "\n")
            TransferRunner(FakeBackend(), [sample], run, sha256_file(inference), "revision").run()
            unscored = build_casebook(run, inference, 1, None)
            self.assertFalse(unscored["labels_read"])
            self.assertNotIn("offline", unscored["cases"][0])
            self.assertIn("Exact final user input:", render_markdown(unscored))
            scored = build_casebook(run, inference, 1, labels)
            self.assertTrue(scored["labels_read"])
            self.assertEqual(scored["cases"][0]["offline"]["score"]["none"], 1.0)

    def test_incomplete_run_rejects_before_opening_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            (run / "manifest.json").write_text(json.dumps({"status": "running"}))
            (run / "predictions.jsonl").write_text("")
            inference = root / "inference.jsonl"
            inference.write_text("THIS_IS_NOT_JSON")
            labels = root / "labels.jsonl"
            labels.write_text("THIS_IS_NOT_JSON_EITHER")
            with self.assertRaisesRegex(ValueError, "Incomplete or mismatched"):
                build_casebook(run, inference, 1, labels)


if __name__ == "__main__":
    unittest.main()
