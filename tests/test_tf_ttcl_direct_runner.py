import json
import tempfile
import unittest
from pathlib import Path

from core.multimodal.schema import GenerationResponse, MultimodalSample
from core.multimodal.tf_ttcl_direct_runner import (DirectRunner, has_reasoning_before_final,
                                                    parse_variants, split_question_clause)
from core.multimodal.tf_ttcl_text_embedding import cosine
from scripts.create_label_isolated_subset import create_subset


class FakeEmbedder:
    def encode(self, text, *, query=False):
        return (1.0, 0.0) if "Question" in text or query else (0.0, 1.0)


class FakeBackend:
    model_name = "fake-vlm"

    def generate(self, system, user, images, sampling):
        if "Rewrite only the question" in system:
            content = json.dumps({"questions": [f"Question rephrased {i}?" for i in range(4)]})
            return GenerationResponse(content, prompt_tokens=8, completion_tokens=8)
        if "<task>" in user:
            return GenerationResponse("Identify the requested evidence before choosing.",
                                      prompt_tokens=8, completion_tokens=9)
        answer = "A" if sampling.temperature == 0 or sampling.seed % 5 < 3 else "B"
        return GenerationResponse(f"Final answer: {answer}", prompt_tokens=10,
                                  completion_tokens=2, logprobs=[-0.1, -0.2])


class DirectRunnerTests(unittest.TestCase):
    def test_reasoning_format_accepts_prose_without_heading(self):
        self.assertTrue(has_reasoning_before_final(
            "The image shows a red object beside two blue ones. Counting red objects gives one.\nFinal answer: A"))
        self.assertFalse(has_reasoning_before_final("Final answer: A"))
        self.assertFalse(has_reasoning_before_final("Final answer: A\nReasoning: too late"))

    def test_variants_keep_numbers_and_fallback(self):
        variants = parse_variants('{"questions":["Which 3?","Which 4?","How many 3?"]}',
                                  "How many 3?")
        self.assertEqual(variants, ["Which 3?", "How many 3?", "How many 3?", "How many 3?"])
        self.assertEqual(parse_variants('["Which 3?", "Which three?", "How many 4?"]',
                                        "How many 3?")[:2], ["Which 3?", "How many 3?"])
        prefix, clause = split_question_clause("Context:\nKeep this text.\n\nQuestion:\nWhich 3?")
        self.assertEqual(prefix, "Context:\nKeep this text.\n\nQuestion:\n")
        self.assertEqual(clause, "Which 3?")

    def test_cosine_requires_matching_dimensions(self):
        self.assertEqual(cosine((1.0, 0.0), (0.0, 1.0)), 0.0)
        with self.assertRaises(ValueError):
            cosine((1.0,), (1.0, 0.0))

    def test_unseen_subset_excludes_prior_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text("".join(json.dumps({"id": str(i), "question": "Q?", "answer": 0}) + "\n"
                                      for i in range(6)))
            prior = root / "prior.inference.jsonl"
            prior.write_text('{"id":"0"}\n{"id":"1"}\n')
            manifest = create_subset(source, root / "new.inference.jsonl", root / "new.labels.jsonl",
                                     root / "new.manifest.json", 3, 17,
                                     exclude_inference_paths=(prior,))
            self.assertTrue(set(manifest["selected_ids"]).isdisjoint({"0", "1"}))
            self.assertEqual(manifest["excluded_id_count"], 2)
            self.assertTrue(all("answer" not in json.loads(line) for line in
                                (root / "new.inference.jsonl").read_text().splitlines()))

    def test_two_sample_online_order_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            image.write_bytes(b"fake-image")
            samples = [MultimodalSample(str(i), "scienceqa_img",
                                        f"Context:\nKeep context {i}.\n\nQuestion:\nQuestion {i}?",
                                        (str(image),), ("first", "second")) for i in range(2)]
            output = root / "run"
            runner = DirectRunner(FakeBackend(), FakeEmbedder(), samples, output,
                                  "source-sha", "model-rev", "embed-rev")
            runner.run()
            rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["retrieved_rules"], {"positive": [], "negative": []})
            self.assertTrue(rows[0]["written_rules"])
            self.assertTrue(all(variant.startswith("Context:\nKeep context 1.\n\nQuestion:\n")
                                for variant in rows[1]["variants"]))
            self.assertTrue(all(rule["sample_index"] == 0 for bucket in rows[1]["retrieved_rules"].values()
                                for rule in bucket))
            self.assertEqual(set(rows[1]["arms"]), {"greedy", "vote_no_rewrite",
                                                    "select_no_memory", "tf_ttcl_mm_direct"})
            runner2 = DirectRunner(FakeBackend(), FakeEmbedder(), samples, output,
                                   "source-sha", "model-rev", "embed-rev")
            runner2.run()
            self.assertEqual(len((output / "predictions.jsonl").read_text().splitlines()), 2)
            with self.assertRaises(ValueError):
                DirectRunner(FakeBackend(), FakeEmbedder(), samples, output,
                             "other-source", "model-rev", "embed-rev").run()


if __name__ == "__main__":
    unittest.main()
