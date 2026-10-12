from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.multimodal.typed_evidence import (
    LETTER_SCHEMA, assess_certificate, compile_program, execute_program,
    parse_observation,
)


MAGNET = {"operator": "facing_poles", "modality": "letters"}


def letters(labels=("S", "N", "N", "S"), vertical=False, overflow=False):
    return json.dumps({"overflow": overflow, "items": [
        {"text": label, "x": 500 if vertical else 100 + 200 * i,
         "y": 100 + 200 * i if vertical else 500}
        for i, label in enumerate(labels)]})


class TypedEvidenceTests(unittest.TestCase):
    def test_client_forwards_schema_and_cache_distinguishes_it(self):
        from core.multimodal.client import OpenAIVLMClient
        from core.multimodal.replay import ResponseTraceBackend
        from core.multimodal.schema import SamplingConfig
        from unittest.mock import Mock

        client = OpenAIVLMClient("http://127.0.0.1:9/v1", "fake")
        client.client = Mock()
        client.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=letters(())))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4))
        sampling = SamplingConfig(output_schema=LETTER_SCHEMA)
        client.generate("s", "q", (), sampling)
        request = client.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["response_format"]["json_schema"]["schema"], LETTER_SCHEMA)
        with tempfile.TemporaryDirectory() as d:
            trace = ResponseTraceBackend(client, Path(d), {"seed": 0}, "record")
            trace.generate("s", "q", (), SamplingConfig())
            trace.generate("s", "q", (), sampling)
            self.assertEqual(len(list(Path(d).glob("*.json"))), 2)
            requests = [json.loads(p.read_text())["request"] for p in Path(d).glob("*.json")]
            self.assertEqual(sum("output_schema" in r["sampling"] for r in requests), 1)

    def test_routing_is_bounded(self):
        self.assertEqual(compile_program("Will these magnets attract or repel?", ["repel", "attract"]), MAGNET)
        self.assertEqual(compile_program("How many red circles are shown?", ["1", "2"])["filters"],
                         {"color": "red", "shape": "circle"})
        self.assertIsNone(compile_program("How many invisible circles are shown?", ["1", "2"]))
        self.assertIsNone(compile_program("Which country is highlighted?", ["US", "UK"]))
        self.assertEqual(compile_program("Which city has the highest temperature?", ["X", "Y"])["operator"], "table_extreme")

    def test_horizontal_vertical_and_order_invariance(self):
        a = parse_observation(letters(), MAGNET)
        b = parse_observation(letters(vertical=True), MAGNET)
        c = parse_observation(letters(tuple(reversed(("S", "N", "N", "S")))), MAGNET)
        self.assertEqual(a["facts"], b["facts"])
        self.assertEqual(a["facts"], c["facts"])
        self.assertEqual(execute_program(MAGNET, b["facts"], ["repel", "attract"])["answer"], "a")

    def test_lowercase_letters_valid_but_not_magnetic_facts(self):
        r = parse_observation(letters(("g",)), MAGNET)
        self.assertFalse(r["parse_error"])
        self.assertFalse(r["visible"])
        self.assertFalse(r["literal_empty"])

    def test_schema_overflow_duplicate_and_bounds_fail_closed(self):
        for text in ('{"overflow":false,"items":[],"answer":"B"}',
                     '{"overflow":false,"overflow":true,"items":[]}', letters()[:-1],
                     letters(("N",) * 9)):
            self.assertTrue(parse_observation(text, MAGNET)["parse_error"])
        data = json.loads(letters())
        for value in (True, -1, 1001, float("nan")):
            data["items"][0]["x"] = value
            self.assertTrue(parse_observation(json.dumps(data), MAGNET)["parse_error"])
        overflow = parse_observation(letters(overflow=True), MAGNET)
        self.assertFalse(overflow["parse_error"])
        self.assertFalse(overflow["complete"])
        self.assertFalse(overflow["visible"])

    def test_stability_is_diagnosed_even_when_probe_malformed(self):
        real = parse_observation(letters(), MAGNET)
        cert = assess_certificate(MAGNET, [real, real, {"parse_error": True}], ["repel", "attract"], ["a", "b"])
        self.assertTrue(cert["facts_stable"])
        self.assertFalse(cert["gate_passed"])

    def test_blank_must_be_literal_empty_and_candidate_must_exist(self):
        real = parse_observation(letters(), MAGNET)
        blank = parse_observation(letters(()), MAGNET)
        other = parse_observation(letters(("S", "N", "S", "N")), MAGNET)
        accepted = assess_certificate(MAGNET, [real, real, blank, other], ["repel", "attract"], ["a", "b"])
        self.assertTrue(accepted["gate_passed"])
        for probe in (real, parse_observation(letters(("g",)), MAGNET),
                      parse_observation(letters(overflow=True), MAGNET)):
            self.assertFalse(assess_certificate(MAGNET, [real, real, probe],
                             ["repel", "attract"], ["a", "b"])["gate_passed"])
        self.assertFalse(assess_certificate(MAGNET, [real, real, blank],
                         ["repel", "attract"], ["b"])["gate_passed"])

    def test_numeric_tables_and_generic_body(self):
        choices = ["each bottle . . . the surroundings", "the surroundings . . . each bottle"]
        program = compile_program("During this time, thermal energy was transferred from () to ().", choices)
        raw = {"overflow": False, "columns": ["Initial temperature (°C)", "Final temperature (°C)"],
               "rows": [{"label": "Bottle 1", "values": ["2.2", "4.9"]}]}
        observation = parse_observation(json.dumps(raw), program)
        self.assertEqual(execute_program(program, observation["facts"], choices)["answer"], "b")
        raw["rows"][0]["values"][1] = None
        observation = parse_observation(json.dumps(raw), program)
        self.assertFalse(observation["parse_error"])
        self.assertFalse(observation["complete"])

    def test_numeric_cell_lexemes_are_bounded_and_never_rounded(self):
        program = {"operator": "temperature_delta", "modality": "table", "body": "each bottle"}
        raw = {"overflow": False, "columns": ["Initial temperature (°C)", "Final temperature (°C)"],
               "rows": [{"label": "Bottle", "values": ["18", "26"]}]}
        for value in (18, "1.234567890123456789", "1000000", "NaN", "1e3", "18 °C"):
            raw["rows"][0]["values"][0] = value
            self.assertTrue(parse_observation(json.dumps(raw), program)["parse_error"])
        raw["rows"][0]["values"][0] = "18.0000"
        obs = parse_observation(json.dumps(raw), program)
        self.assertTrue(obs["complete"])
        self.assertEqual(obs["facts"]["rows"][0]["values"], [18.0, 26.0])

    def test_extrema_ties_count_unknown_filters_and_overflow(self):
        p = compile_program("Which city has the lowest temperature?", ["X", "Y"])
        facts = {"columns": ["Temperature (°C)"], "rows": [
            {"label": "X", "values": [10]}, {"label": "Y", "values": [20]}]}
        self.assertEqual(execute_program(p, facts, ["X", "Y"])["answer"], "a")
        facts["rows"][1]["values"][0] = 10
        self.assertIsNone(execute_program(p, facts, ["X", "Y"])["answer"])
        p = compile_program("How many red circles are shown?", ["1", "2"])
        raw = {"overflow": False, "objects": [
            {"color": "red", "shape": "circle", "size": "", "x": 100, "y": 100},
            {"color": "blue", "shape": "circle", "size": "", "x": 500, "y": 100}]}
        o = parse_observation(json.dumps(raw), p)
        self.assertTrue(o["complete"])
        self.assertEqual(execute_program(p, o["facts"], ["1", "2"])["answer"], "a")
        raw["objects"][1]["color"] = ""
        self.assertFalse(parse_observation(json.dumps(raw), p)["complete"])

    def test_runner_uses_schema_and_never_exposes_question(self):
        from core.multimodal.runner import EviMemRunner
        from core.multimodal.memory import EvidenceMemory
        from core.multimodal.embeddings import HashMultimodalEmbedder
        from core.multimodal.schema import MultimodalSample, Candidate, GenerationResponse
        from PIL import Image

        class Recorder:
            model_name = "fake"
            def __init__(self):
                self.calls = []
                self.length = False
            def generate(self, system, text, images, sampling):
                self.calls.append((system, text, sampling))
                return GenerationResponse(letters(() if "blank" in images[0] else ("S", "N", "N", "S"), vertical=True),
                                          finish_reason="length" if self.length else "stop")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real, blank = root / "real.png", root / "blank.png"
            Image.new("RGB", (10, 10)).save(real)
            Image.new("RGB", (10, 10), "white").save(blank)
            sample = MultimodalSample("x", "scienceqa_img", "SECRET\n\nQuestion:\nWill these magnets attract or repel?",
                                      [str(real)], ["attract", "repel"])
            model = Recorder()
            runner = EviMemRunner(model, EvidenceMemory(HashMultimodalEmbedder(16)), root / "run",
                                 method="shared_evidence", config_snapshot={"evidence": {"protocol": "typed_v3"}})
            candidates = [Candidate("g", "original_greedy", GenerationResponse("A"), "a", "q", ()),
                          Candidate("s", "original_sampled", GenerationResponse("B"), "b", "q", ())]
            decision, cert = runner._shared_adjudication(sample, candidates, 0, [str(real)], [str(blank)], ())
            self.assertEqual(decision.answer, "b")
            self.assertTrue(cert["gate_passed"])
            self.assertFalse(decision.eligible_for_memory)
            for system, text, sampling in model.calls:
                self.assertNotIn("SECRET", system + text)
                self.assertNotIn("Will these magnets", system + text)
                self.assertNotIn("attract", system + text)
                self.assertNotIn("repel", system + text)
                self.assertEqual(sampling.output_schema, LETTER_SCHEMA)
                self.assertEqual(sampling.max_tokens, 512)
            model.length = True
            decision, cert = runner._shared_adjudication(sample, candidates, 0, [str(real)], [str(blank)], ())
            self.assertEqual(decision.answer, "a")
            self.assertFalse(cert["gate_passed"])
            self.assertTrue(all(not o["parsed"]["complete"] for o in cert["observations"]))


if __name__ == "__main__":
    unittest.main()
