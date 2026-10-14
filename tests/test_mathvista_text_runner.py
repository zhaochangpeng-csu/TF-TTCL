from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.multimodal.mathvista_text_runner import MathVistaTextRunner, math_format_ok
from core.multimodal.schema import GenerationResponse, MultimodalSample
from evaluate.multimodal.compare_mathvista_text import audit


class FakeBackend:
    model_name = "fake-vlm"

    def generate(self, system, user, images, sampling):
        if not images:
            return GenerationResponse(content='{"questions":["What is the number?",'
                                              '"Which number is shown?"]}', finish_reason="stop")
        return GenerationResponse(content="Observed: A value is visible.\n"
                                          "Reasoning: Read the value.\nFinal answer: 3",
                                  prompt_tokens=10, completion_tokens=15,
                                  finish_reason="stop")


class MathVistaTextRunnerTests(unittest.TestCase):
    def test_format_supports_free_form_and_options(self):
        free = MultimodalSample("1", "mathvista", "What is the number?", (), (),
                                {"question_type": "free_form", "answer_type": "integer"})
        valid = GenerationResponse("Observed: Three objects.\nReasoning: Count them.\nFinal answer: 3",
                                   finish_reason="stop")
        self.assertTrue(math_format_ok(valid, free, "3"))
        self.assertFalse(math_format_ok(GenerationResponse("Final answer: 3", finish_reason="stop"), free, "3"))
        choice = MultimodalSample("2", "mathvista", "Which?", (), ("three", "four"),
                                  {"question_type": "multi_choice"})
        self.assertTrue(math_format_ok(valid, choice, "three"))
        self.assertFalse(math_format_ok(valid, choice, "five"))

    def test_complete_pairing_and_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            image.write_bytes(b"fixed-image")
            record = {"id": "1", "question": "What is the number?", "image": "image.png",
                      "choices": [], "question_type": "free_form", "answer_type": "integer"}
            inference = root / "inference.jsonl"
            inference.write_text(json.dumps(record) + "\n")
            sample = MultimodalSample("1", "mathvista", record["question"], (str(image),), (),
                                      {"id": "1", "question_type": "free_form", "answer_type": "integer"})
            run = root / "run"
            MathVistaTextRunner(FakeBackend(), [sample], run, __import__("hashlib").sha256(
                inference.read_bytes()).hexdigest(), "fake-revision").run()
            manifest, rows = audit(run, inference, 1)
            self.assertEqual(manifest["version"], "mathvista_text_only_vote_v1")
            self.assertEqual(rows[0]["arms"]["text_aug_vote"]["answer"], "3")
            rows[0]["candidates"][4]["image_hashes"] = ["bad"]
            (run / "predictions.jsonl").write_text(json.dumps(rows[0]) + "\n")
            with self.assertRaisesRegex(ValueError, "contract broken"):
                audit(run, inference, 1)


if __name__ == "__main__":
    unittest.main()
