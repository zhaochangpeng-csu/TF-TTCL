import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from core.multimodal.answers import normalize_answer
from core.multimodal.datasets import load_local_dataset
from core.multimodal.schema import GenerationResponse
from core.multimodal.textvqa_text_runner import TextVQATextRunner, textvqa_format_ok
from evaluate.multimodal.compare_textvqa_text import _raw_vote_answer, audit


class FakeBackend:
    model_name = "fake-vlm"

    def generate(self, system, user, images, sampling):
        if not images:
            return GenerationResponse(content='{"questions":["What word is visible on the sign?",'
                                              '"Which word is on the sign?"]}', finish_reason="stop")
        return GenerationResponse(content="Observed: STOP is written on the sign.\n"
                                          "Reasoning: Read the sign.\nFinal answer: STOP",
                                  prompt_tokens=10, completion_tokens=15, finish_reason="stop")


class TextVQATextRunnerTests(unittest.TestCase):
    def test_format_and_raw_answer(self):
        valid = GenerationResponse("Observed: The sign says STOP.\nReasoning: Read it.\nFinal answer: STOP",
                                   finish_reason="stop")
        self.assertTrue(textvqa_format_ok(valid, "stop"))
        self.assertFalse(textvqa_format_ok(GenerationResponse("Final answer: STOP", finish_reason="stop"), "stop"))
        self.assertEqual(normalize_answer("STOP", "textvqa"), "stop")

    def test_complete_pairing_and_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            image.write_bytes(b"fixed-image")
            record = {"id": "1", "question": "What word is on the sign?", "image": "image.jpg", "choices": []}
            inference = root / "inference.jsonl"
            inference.write_text(json.dumps(record) + "\n")
            sample = load_local_dataset(str(inference), "textvqa")[0].sample
            run = root / "run"
            TextVQATextRunner(FakeBackend(), [sample], run,
                              hashlib.sha256(inference.read_bytes()).hexdigest(), "fake-revision").run()
            manifest, rows = audit(run, inference, 1)
            self.assertIs(manifest["labels_read"], False)
            self.assertEqual(rows[0]["arms"]["text_aug_vote"]["answer"], "stop")
            self.assertEqual(_raw_vote_answer(rows[0], "text_aug_vote"), "STOP")
            rows[0]["candidates"][4]["image_hashes"] = ["bad"]
            (run / "predictions.jsonl").write_text(json.dumps(rows[0]) + "\n")
            with self.assertRaisesRegex(ValueError, "contract broken"):
                audit(run, inference, 1)


if __name__ == "__main__":
    unittest.main()
