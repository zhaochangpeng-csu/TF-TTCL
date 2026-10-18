import tempfile
import unittest
from pathlib import Path

from core.multimodal.schema import GenerationResponse, MultimodalSample
from scripts.probe_tf_ttcl_mm_logprobs import probe


class FakeBackend:
    model_name = "fake-vlm"

    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, system, user, images, sampling):
        self.calls.append((system, user, images, sampling))
        return self.response


class LogprobProbeTests(unittest.TestCase):
    def test_visual_request_requires_logprobs(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.jpg"
            image.write_bytes(b"test-image")
            sample = MultimodalSample("1", "scienceqa_img", "Which option?", (str(image),), ("A", "B"))
            backend = FakeBackend(GenerationResponse("Final answer: A", completion_tokens=2,
                                                     logprobs=[-.2, -.4], finish_reason="stop"))
            result = probe(backend, sample)
            self.assertEqual(result["status"], "supported")
            self.assertTrue(backend.calls[0][2])
            self.assertTrue(backend.calls[0][3].request_logprobs)
            self.assertEqual(len(backend.calls), 2)
            self.assertEqual(backend.calls[1][3].temperature, 0.7)
            self.assertTrue(all(p["logprob_count"] == p["completion_token_count"]
                                for p in result["probes"]))

    def test_missing_logprobs_blocks_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.jpg"
            image.write_bytes(b"test-image")
            sample = MultimodalSample("1", "scienceqa_img", "Which option?", (str(image),), ("A", "B"))
            result = probe(FakeBackend(GenerationResponse("A", completion_tokens=1)), sample)
            self.assertEqual(result["status"], "blocked")
            self.assertTrue(all("ppl" not in p for p in result["probes"]))


if __name__ == "__main__":
    unittest.main()
