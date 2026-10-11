from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from PIL import Image

from core.mm_main import apply_overrides
from core.multimodal.client import (
    build_multimodal_content,
    parse_contrastive_verdict,
    parse_evidence_verdict,
)
from core.multimodal.answers import normalize_answer
from core.multimodal.datasets import load_local_dataset
from core.multimodal.embeddings import HashMultimodalEmbedder
from core.multimodal.image_ops import make_blank_views, make_fidelity_views
from core.multimodal.memory import EvidenceMemory
from core.multimodal.runner import EviMemRunner, _semantic_variant_is_safe
from core.multimodal.schema import (
    Candidate,
    ComponentScores,
    EvidenceClaim,
    EvidenceVerdict,
    EvaluationTarget,
    Experience,
    GenerationResponse,
    MultimodalSample,
    SamplingConfig,
)
from core.multimodal.scoring import select_with_evidence
from evaluate.multimodal.metrics import (
    anls,
    paired_bootstrap_delta_ci,
    paired_randomization_p_value,
    score_prediction,
    vqa_accuracy,
)
from evaluate.multimodal.candidate_oracle import summarize_candidate_oracle
from scripts.prepare_multimodal_dataset import cached_modelscope_parquet_rows
from scripts.prepare_scienceqa import convert_scienceqa


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
        if "Rewrite only the supplied question clause" in system_prompt:
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
    def test_official_scienceqa_conversion_filters_images_and_omits_explanations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images" / "val" / "42"
            image_dir.mkdir(parents=True)
            Image.new("RGB", (8, 8), "blue").save(image_dir / "image.png")
            problems = {
                "42": {
                    "split": "val",
                    "question": "Which color?",
                    "hint": "Look at the square.",
                    "image": "image.png",
                    "choices": ["red", "blue"],
                    "answer": 1,
                    "lecture": "blue objects reflect blue light",
                    "solution": "The square is blue.",
                },
                "43": {
                    "split": "val",
                    "question": "Text only?",
                    "image": None,
                    "choices": ["yes", "no"],
                    "answer": 0,
                },
            }
            problems_path = root / "problems.json"
            problems_path.write_text(json.dumps(problems), encoding="utf-8")
            output = root / "scienceqa_img_val.jsonl"
            manifest = convert_scienceqa(problems_path, root / "images", output, "val")
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(manifest["count"], 1)
            self.assertEqual(row["answer"], 1)
            self.assertEqual(row["question_type"], "multi_choice")
            self.assertIn("Look at the square", row["question"])
            self.assertNotIn("lecture", row)
            self.assertNotIn("solution", row)

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

    def test_insufficient_claim_blocks_memory_even_when_answer_is_entailed(self):
        candidates = [candidate(f"a{i}", "A") for i in range(4)] + [candidate("b1", "B")]
        verdicts = {
            "a": EvidenceVerdict(
                "entailed", [EvidenceClaim("answer claim", "insufficient")]
            )
        }
        decision = select_with_evidence(candidates, ["b"], verdicts, "scienceqa_img")
        self.assertEqual(decision.answer, "a")
        self.assertFalse(decision.eligible_for_memory)

    def test_malformed_verdict_is_insufficient(self):
        verdict = parse_evidence_verdict("not json")
        self.assertTrue(verdict.parse_error)
        self.assertEqual(verdict.answer_status, "insufficient")

    def test_contrastive_verdict_parser_is_fail_closed(self):
        valid = parse_contrastive_verdict(
            '{"winner":"candidate_2","candidate_1_status":"contradicted",'
            '"candidate_2_status":"entailed","decisive_evidence":["visible mark"],'
            '"counterevidence":["candidate 1 conflicts"],"rationale":"direct"}'
        )
        self.assertEqual(valid.winner, "candidate_2")
        self.assertFalse(valid.parse_error)
        malformed = parse_contrastive_verdict('{"winner":"answer_b"}')
        self.assertEqual(malformed.winner, "undecidable")
        self.assertTrue(malformed.parse_error)
        wrong_evidence_type = parse_contrastive_verdict(
            '{"winner":"candidate_1","candidate_1_status":"entailed",'
            '"candidate_2_status":"contradicted","decisive_evidence":"visible mark",'
            '"counterevidence":[]}'
        )
        self.assertTrue(wrong_evidence_type.parse_error)
        self.assertEqual(wrong_evidence_type.decisive_evidence, [])


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
            memory = EvidenceMemory(
                HashMultimodalEmbedder(),
                rule_min_similarity=-1.0,
                episode_min_similarity=-1.0,
            )
            memory.add(self._experience(0, image))
            memory.add(self._experience(2, image))
            sample = MultimodalSample("q", "x", "question", (image,))
            bundle = memory.retrieve(sample, sample_index=2)
            self.assertEqual([e.experience.experience_id for e in bundle.episodes], ["e0"])

    def test_similarity_floor_does_not_force_unrelated_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            image = str(Path(directory) / "image.png")
            Image.new("RGB", (8, 8), "red").save(image)
            memory = EvidenceMemory(HashMultimodalEmbedder(2))
            experience = self._experience(0, image)
            experience.text_embedding = [0.0, 1.0]
            experience.image_embedding = [0.0, 1.0]
            experience.joint_embedding = [0.0, 1.0]
            memory.add(experience)
            sample = MultimodalSample("q", "x", "question", (image,))
            # Replace query methods with deterministic orthogonal vectors.
            memory.embedder.encode_text = lambda _: np.asarray([1.0, 0.0])
            memory.embedder.encode_image = lambda _: np.asarray([1.0, 0.0])
            memory.embedder.encode_joint = lambda _text, _images: np.asarray([1.0, 0.0])
            bundle = memory.retrieve(sample, sample_index=1)
            self.assertEqual(bundle.rules, ())
            self.assertEqual(bundle.episodes, ())


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
    def test_candidate_oracle_separates_generation_and_selection_errors(self):
        targets = {
            "q1": EvaluationTarget("q1", ("B", "blue"), {}),
            "q2": EvaluationTarget("q2", ("A", "red"), {}),
        }
        rows = [
            {
                "sample_id": "q1",
                "answer": "a",
                "candidate_answers": [{"answer": "a"}, {"answer": "b"}],
            },
            {
                "sample_id": "q2",
                "answer": "a",
                "candidate_answers": [{"answer": "a"}, {"answer": "a"}],
            },
        ]
        summary = summarize_candidate_oracle(rows, targets, "scienceqa_img")
        self.assertEqual(summary["selected_score"], 0.5)
        self.assertEqual(summary["candidate_oracle_score"], 1.0)
        self.assertEqual(summary["selection_regrets"], 1)
        self.assertEqual(summary["all_candidates_same"], 1)

    def test_paired_statistics_preserve_sample_pairing(self):
        baseline = [1.0, 0.0, 1.0, 0.0]
        candidate = [1.0, 1.0, 0.0, 1.0]
        delta, low, high = paired_bootstrap_delta_ci(
            baseline, candidate, samples=1000, seed=7
        )
        self.assertEqual(delta, 0.25)
        self.assertLessEqual(low, delta)
        self.assertGreaterEqual(high, delta)
        self.assertEqual(paired_randomization_p_value(baseline, baseline), 1.0)
        self.assertGreater(paired_randomization_p_value(baseline, candidate), 0.0)

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
        flea_metadata = {
            "answer_type": "text",
            "question_type": "multi_choice",
            "choices": ["Tilapia", "Common water flea", "Great diving beetle", "Tadpole"],
        }
        self.assertEqual(
            score_prediction(
                "mathvista", "Final answer: B", ["Common water flea"], flea_metadata
            ),
            1.0,
        )
        self.assertEqual(
            score_prediction(
                "mathvista", "Common water flea", ["Common water flea"], flea_metadata
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
    def test_semantic_augmentation_preserves_context_verbatim(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = EviMemRunner(
                FakeVLM(),
                EvidenceMemory(HashMultimodalEmbedder()),
                Path(directory) / "run",
            )
            source = "Context:\nA fact and a hint.\n\nQuestion:\nWhich option?"
            variants = runner._paraphrases(source, 0)
            self.assertEqual(
                variants,
                [
                    "Context:\nA fact and a hint.\n\nQuestion:\nRephrased one?",
                    "Context:\nA fact and a hint.\n\nQuestion:\nRephrased two?",
                ],
            )

    def test_semantic_augmentation_guard_rejects_logic_changes(self):
        self.assertTrue(
            _semantic_variant_is_safe(
                "Is Mars more than 3 times larger?",
                "Is Mars more than 3 times as large?",
            )
        )
        self.assertFalse(
            _semantic_variant_is_safe(
                "Is Mars more than 3 times larger?",
                "Is Mars less than 4 times larger?",
            )
        )
        self.assertFalse(
            _semantic_variant_is_safe(
                "Which animal is not nocturnal?",
                "Which animal is nocturnal?",
            )
        )
        self.assertFalse(
            _semantic_variant_is_safe(
                "Jupiter is larger than Mars. Which is larger?",
                "Which planet is larger?",
            )
        )

    def test_contrastive_adjudication_is_position_invariant(self):
        class ContrastiveFake(FakeVLM):
            def generate(self, system_prompt, user_text, image_paths, sampling):
                if "contrastive multimodal evidence" not in system_prompt:
                    return super().generate(system_prompt, user_text, image_paths, sampling)
                winner = "response_y" if "RESPONSE_X proposed answer: option A" in user_text else "response_x"
                first = "contradicted" if winner == "response_y" else "entailed"
                second = "entailed" if winner == "response_y" else "contradicted"
                content = json.dumps({
                    "winner": winner,
                    "response_x_status": first,
                    "response_y_status": second,
                    "decisive_evidence": ["visible feature"],
                    "counterevidence": ["other answer conflicts"],
                    "rationale": "same winner after swapping positions",
                })
                return GenerationResponse(content, 10, 3, [], 0.01)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (8, 8), "red").save(image)
            sample = MultimodalSample(
                "q1", "scienceqa_img", "Which?", (str(image),), ("red", "blue")
            )
            runner = EviMemRunner(
                ContrastiveFake(), EvidenceMemory(HashMultimodalEmbedder()), root / "run",
                method="contrastive_evidence",
            )
            candidates = [candidate(f"a{i}", "A") for i in range(4)] + [
                candidate("b1", "B"), candidate("b2", "B")
            ]
            decision, records = runner._contrastive_adjudication(
                sample, candidates, 0, (str(image),)
            )
            self.assertEqual(decision.answer, "b")
            self.assertTrue(decision.eligible_for_memory)
            self.assertEqual(len(records), 2)

    def test_contrastive_selection_does_not_require_negative_memory_gate(self):
        class InsufficientNegativeFake(FakeVLM):
            def generate(self, system_prompt, user_text, image_paths, sampling):
                if "contrastive multimodal evidence" not in system_prompt:
                    return super().generate(system_prompt, user_text, image_paths, sampling)
                first_is_a = "RESPONSE_X proposed answer: option A" in user_text
                content = json.dumps({
                    "winner": "undecidable",
                    "response_x_status": "insufficient" if first_is_a else "entailed",
                    "response_y_status": "entailed" if first_is_a else "insufficient",
                    "decisive_evidence": ["visible feature"],
                    "counterevidence": ["alternative lacks support"],
                    "rationale": "one answer is supported in both orders",
                })
                return GenerationResponse(content, 10, 3, [], 0.01)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (8, 8), "red").save(image)
            sample = MultimodalSample(
                "q1", "scienceqa_img", "Which?", (str(image),), ("red", "blue")
            )
            runner = EviMemRunner(
                InsufficientNegativeFake(),
                EvidenceMemory(HashMultimodalEmbedder()),
                root / "run",
                method="contrastive_evidence",
            )
            candidates = [candidate(f"a{i}", "A") for i in range(4)] + [
                candidate("b1", "B"), candidate("b2", "B")
            ]
            decision, records = runner._contrastive_adjudication(
                sample, candidates, 0, (str(image),)
            )
            self.assertEqual(decision.answer, "b")
            self.assertFalse(decision.eligible_for_memory)
            self.assertIn("memory=rejected_no_contradiction", decision.reason)
            self.assertTrue(all(record["raw_response"] for record in records))

    def test_memory_intervention_keeps_three_clean_anchor_trajectories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (8, 8), "red").save(image)
            sample = MultimodalSample(
                "q1", "scienceqa_img", "Which option?", (str(image),), ("red", "blue")
            )
            runner = EviMemRunner(
                FakeVLM(), EvidenceMemory(HashMultimodalEmbedder()), root / "run"
            )
            candidates = runner._generate_candidates(
                sample,
                sample_index=0,
                memory_context="Rules:\n- abstract prior rule",
                fidelity_images=(str(image),),
            )
            self.assertEqual(len(candidates), 6)
            self.assertEqual(sum(item.memory_conditioned for item in candidates), 3)

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
            self.assertTrue(record["candidate_answers"][0]["image_hashes"])
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

    def test_resume_toggle_is_operational_not_a_scientific_hash_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (8, 8), "red").save(image)
            sample = MultimodalSample("q1", "scienceqa_img", "Which?", (str(image),))
            first = EviMemRunner(
                FakeVLM(),
                EvidenceMemory(HashMultimodalEmbedder()),
                root / "run",
                method="zero_shot",
                resume=False,
                config_snapshot={"experiment": {"resume": False}, "seed": 0},
            )
            first.run([sample])
            resumed = EviMemRunner(
                FakeVLM(),
                EvidenceMemory(HashMultimodalEmbedder()),
                root / "run",
                method="zero_shot",
                resume=True,
                config_snapshot={"experiment": {"resume": True}, "seed": 0},
            )
            resumed.run([sample])
            rows = (root / "run" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
