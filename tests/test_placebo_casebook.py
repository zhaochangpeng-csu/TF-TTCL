from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.multimodal.schema import MultimodalSample
from core.multimodal.transfer_placebo import PlaceboRunner
from core.multimodal.transfer_runner import TransferRunner
from evaluate.multimodal.placebo_casebook import build_casebook, render_markdown
from scripts.create_label_isolated_subset import sha256_file
from tests.test_transfer_runner import FakeBackend


class PlaceboCasebookTests(unittest.TestCase):
    def test_casebook_requires_completed_audit_before_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = MultimodalSample("a", "scienceqa_img", "Which?", (), ("yes", "no"),
                                      {"question_type": "multi_choice"})
            inference = root / "inference.jsonl"
            inference.write_text(json.dumps({"id": "a", "question": "Which?",
                                             "choices": ["yes", "no"]}) + "\n")
            labels = root / "labels.jsonl"
            labels.write_text("INVALID_UNREAD")
            backend = FakeBackend()
            source = root / "source"
            run = root / "run"
            TransferRunner(backend, [sample], source, sha256_file(inference), "revision").run()
            PlaceboRunner(backend, [sample], source, run, sha256_file(inference),
                          "revision", source_expected_n=1).run()
            report = build_casebook(source, run, inference, 1, None, 1)
            self.assertEqual(len(report["cases"]), 1)
            self.assertFalse(report["labels_read_for_casebook"])
            self.assertIn("Exact final user input", render_markdown(report))
            (run / "manifest.json").write_text(json.dumps({"status": "running"}))
            with self.assertRaisesRegex(ValueError, "Incomplete or mismatched"):
                build_casebook(source, run, inference, 1, labels, 1)


if __name__ == "__main__":
    unittest.main()
