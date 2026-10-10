from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from core.mm_main import apply_overrides
from core.multimodal.client import build_multimodal_content, parse_evidence_verdict
from core.multimodal.answers import normalize_answer
from core.multimodal.datasets import load_local_dataset
from core.multimodal.embeddings import HashMultimodalEmbedder
from core.multimodal.image_ops import make_blank_views, make_fidelity_views
from core.multimodal.memory import EvidenceMemory
from core.multimodal.runner import EviMemRunner
from core.multimodal.schema import (
    Candidate,
    ComponentScores,
    EvidenceClaim,
    EvidenceVerdict,
    Experience,
    GenerationResponse,
    MultimodalSample,
    SamplingConfig,
)
from core.multimodal.scoring import select_with_evidence
from evaluate.multimodal.metrics import anls, score_prediction, vqa_accuracy
from scripts.prepare_multimodal_dataset import cached_modelscope_parquet_rows


def candidate(name: str, answer: str, logprob: float = -0.2) -> Candidate:
    return Candidate(
        candidate_id=name,
        trajectory=name,
        response=GenerationResponse(f"Final answer: {answer}", logprobs=[logprob]),
        normalized_answer=answer.lower(),
        question_variant="question",
        image_paths=(),
    )


class FakeVLM:
    model_name = "fake-vlm"

    def generate(self, system_prompt, user_text, image_paths, sampling):
        if "Rewrite a visual question" in system_prompt:
            content = '{"questions":["Rephrased one?","Rephrased two?"]}'
        elif "evidence verifier" in system_prompt:
            content = '{"answer_status":"entailed","claims":[{"text":"x","status":"entailed","evidence":"pixel"}],"rationale":"visible"}'
        elif "Distill reusable" in system_prompt:
            content = '{"positive_rule":"ground it","negative_rule":"do not guess","evidence_summary":"visible x"}'
        elif any("blank" in path for path in image_paths):
            content = "Final answer: B"
        else:
            content = "Final answer: A"
        return GenerationResponse(content, 10, 3, [-0.1], 0.01)


class SchemaAndDatasetTests(unittest.TestCase):
    def test_modelscope_cache_fallback_uses_url_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            downloads = root / "hub" / "datasets" / "downloads"
            downloads.mkdir(parents=True)
            shard = downloads / "contenthash"
            pq.write_table(pa.Table.from_pylist([{"pid": "7", "question": "Q?"}]), shard)
            (downloads / "contenthash.json").write_text(
                json.dumps(
                    {
                        "url": "https://modelscope.cn/api/v1/datasets/evalscope/MathVista/repo?"
                        "Revision=master&FilePath=data%2Ftestmini-00000.parquet"
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"MODELSCOPE_CACHE": str(root)}):
                rows, urls = cached_modelscope_parquet_rows("evalscope/MathVista", "testmini")
                self.assertEqual(list(rows)[0]["pid"], "7")
                self.assertEqual(len(urls), 1)

    def test_dotted_config_overrides_are_typed(self):
        config = {"experiment": {"limit": 20}, "model": {"request_logprobs": True}}
        updated = apply_overrides(
            config,
            ["experiment.limit=2", "model.request_logprobs=false", "ablation.evidence=true"],
        )
        self.assertEqual(updated["experiment"]["limit"], 2)
        self.assertFalse(updated["model"]["request_logprobs"])
        self.assertTrue(updated["ablation"]["evidence"])

    def test_inference_sample_has_no_label_field(self):
        sample = MultimodalSample("1", "scienceqa_img", "Which?", (), ("yes", "no"))
        public = sample.to_public_dict()
        self.assertNotIn("answer", public)
        self.assertNotIn("label", public)

    def test_dataset_adapter_isolates_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8), "red").save(root / "a.png")
            (root / "data.jsonl").write_text(
                json.dumps({"id": "q1", "question": "Which?", "image": "a.png", "choices": ["red", "blue"], "answer": 0}) + "\n",
                encoding="utf-8",
            )
            example = load_local_dataset(str(root / "data.jsonl"), "scienceqa_img")[0]
            self.assertEqual(example.target.answers, ("A", "red"))
            self.assertNotIn("answer", example.sample.metadata)
            self.assertNotIn("label", example.sample.to_public_dict())


class EvidenceScoringTests(unittest.TestCase):
    def test_borda_selects_invariant_grounded_answer(self):
        candidates = [candidate(f"a{i}", "A") for i in range(4)] + [candidate("b1", "B"), candidate("b2", "B")]
        verdicts = {
            "a": EvidenceVerdict("entailed", [EvidenceClaim("claim", "entailed")]),
            "b": EvidenceVerdict("contradicted", [EvidenceClaim("claim", "contradicted")]),
        }
        decision = select_with_evidence(candidates, ["b", "c"], verdicts, "scienceqa_img")
        self.assertEqual(decision.answer, "a")
        self.assertTrue(decision.eligible_for_memory)
        self.assertEqual(decision.negative_answer, "b")

    def test_missing_verifier_is_neutral_not_positive(self):
        decision = select_with_evidence([candidate("a", "A"), candidate("b", "B")], [], {}, "scienceqa_img")
        self.assertEqual(decision.scores.evidence, 0.5)
        self.assertFalse(decision.eligible_for_memory)

    def test_malformed_verdict_is_insufficient(self):
        verdict = parse_evidence_verdict("not json")
        self.assertTrue(verdict.parse_error)
        self.assertEqual(verdict.answer_status, "insufficient")


class MemoryTests(unittest.TestCase):
    def _experience(self, index: int, image: str) -> Experience:
        return Experience(
            experience_id=f"e{index}", sample_index=index, sample_id=str(index), dataset="x",
            question=f"question {index}", image_paths=(image,), image_hashes=("hash",),
            positive_answer="a", negative_answer="b", positive_rule="look", negative_rule="don't guess",
            evidence_summary="red square", evidence_claims=(),
            component_scores=ComponentScores(1, 1, 1), trust=1.0, timestamp=0,
        )

    def test_retrieval_never_reads_current_or_future(self):
        with tempfile.TemporaryDirectory() as directory:
            image = str(Path(directory) / "image.png")
            Image.new("RGB", (8, 8), "red").save(image)
            memory = EvidenceMemory(HashMultimodalEmbedder())
            memory.add(self._experience(0, image))
            memory.add(self._experience(2, image))
            sample = MultimodalSample("q", "x", "question", (image,))
            bundle = memory.retrieve(sample, sample_index=2)
            self.assertEqual([e.experience.experience_id for e in bundle.episodes], ["e0"])


class ImageAndClientTests(unittest.TestCase):
    def test_safe_views_are_deterministic_and_dimension_preserving(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jpg"
            Image.new("RGB", (13, 9), (10, 20, 30)).save(source)
            fidelity = make_fidelity_views([str(source)], root / "cache", "q1")[0]
            blank = make_blank_views([str(source)], root / "cache", "q1")[0]
            with Image.open(fidelity) as fidelity_image, Image.open(blank) as blank_image:
                self.assertEqual(fidelity_image.size, (13, 9))
                self.assertEqual(blank_image.size, (13, 9))
                self.assertEqual(blank_image.getpixel((0, 0)), (255, 255, 255))
            content = build_multimodal_content("question", [fidelity])
            self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/png;base64,"))


class MetricsTests(unittest.TestCase):
    def test_mathvista_type_aware_response_extraction(self):
        self.assertEqual(
            score_prediction(
                "mathvista",
                "Reasoning...\nFinal answer: \\( d \\approx 1.24 \\ \\text{cm} \\)",
                ["1.2"],
                {"answer_type": "float", "precision": 1, "question_type": "free_form"},
            ),
            1.0,
        )
        self.assertEqual(
            score_prediction(
                "mathvista",
                "Final answer: (B)",
                ["blue"],
                {
                    "answer_type": "text",
                    "question_type": "multi_choice",
                    "choices": ["red", "blue", "green"],
                },
            ),
            1.0,
        )

    def test_vqa_and_anls(self):
        self.assertAlmostEqual(vqa_accuracy("cat", ["cat"] * 10), 1.0)
        self.assertGreater(anls("hello", ["helo"]), 0.5)
        self.assertEqual(anls("abc", ["xyz"]), 0.0)

    def test_scienceqa_option_parsing_does_not_consume_word_suffixes(self):
        self.assertEqual(normalize_answer("Final answer: (A) red", "scienceqa_img"), "a")
        self.assertEqual(normalize_answer("Final answer: red", "scienceqa_img"), "red")
        self.assertEqual(normalize_answer("Final answer: blue", "scienceqa_img"), "blue")


class RunnerTests(unittest.TestCase):
    def test_end_to_end_output_has_no_ground_truth_and_writes_memory_after_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (8, 8), "red").save(image)
            sample = MultimodalSample("q1", "scienceqa_img", "Which option?", (str(image),), ("red", "blue"))
            memory = EvidenceMemory(HashMultimodalEmbedder())
            runner = EviMemRunner(FakeVLM(), memory, root / "run", seed=7, config_snapshot={"test": True})
            runner.run([sample])
            record = json.loads((root / "run" / "predictions.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["answer"], "a")
            self.assertNotIn("target", record)
            self.assertNotIn("label", record)
            self.assertEqual(record["memory_size_before"], 0)
            self.assertEqual(record["memory_size_after"], 1)

    def test_resume_rejects_config_or_cache_cross_contamination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (8, 8), "red").save(image)
            sample = MultimodalSample("q1", "scienceqa_img", "Which?", (str(image),))
            runner = EviMemRunner(
                FakeVLM(), EvidenceMemory(HashMultimodalEmbedder()), root / "run",
                method="zero_shot", config_snapshot={"seed": 0},
            )
            runner.run([sample])
            incompatible = EviMemRunner(
                FakeVLM(), EvidenceMemory(HashMultimodalEmbedder()), root / "run",
                method="zero_shot", config_snapshot={"seed": 1},
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                incompatible.run([sample])

    def test_non_resume_run_refuses_to_append_existing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (8, 8), "red").save(image)
            sample = MultimodalSample("q1", "scienceqa_img", "Which?", (str(image),))
            first = EviMemRunner(
                FakeVLM(), EvidenceMemory(HashMultimodalEmbedder()), root / "run",
                method="zero_shot", resume=False, config_snapshot={"run": 1},
            )
            first.run([sample])
            second = EviMemRunner(
                FakeVLM(), EvidenceMemory(HashMultimodalEmbedder()), root / "run",
                method="zero_shot", resume=False, config_snapshot={"run": 1},
            )
            with self.assertRaisesRegex(FileExistsError, "Refusing to append"):
                second.run([sample])


if __name__ == "__main__":
    unittest.main()
