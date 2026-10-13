from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.multimodal.schema import GenerationResponse, MultimodalSample
from core.multimodal.transfer_runner import TransferRunner
from evaluate.multimodal.compare_transfer import audit_four, compare
from tests.test_transfer_runner import FakeBackend


class TransferEvaluatorTests(unittest.TestCase):
    def test_incomplete_run_rejects_before_invalid_label_file_is_opened(self):
        sample = MultimodalSample("a", "scienceqa_img", "What is shown?", (), ("yes", "no"),
                                  {"question_type": "multi_choice"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            runner = TransferRunner(FakeBackend(), [sample], path / "run", "source", "revision")
            runner.run()
            labels = path / "labels.jsonl"
            labels.write_text("INVALID_LABEL_FILE_NEVER_READ")
            with self.assertRaisesRegex(ValueError, "Incomplete or mismatched"):
                compare(path / "run", labels, expected_n=2)
            manifest, rows = audit_four(path / "run", expected_n=1)
            self.assertEqual(len(rows), 1)
            self.assertFalse(manifest["labels_read"])


if __name__ == "__main__":
    unittest.main()
