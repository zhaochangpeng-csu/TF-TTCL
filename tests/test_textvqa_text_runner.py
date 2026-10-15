import hashlib
import json
from pathlib import Path

import pytest

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


def test_format_and_raw_answer():
    valid = GenerationResponse("Observed: The sign says STOP.\nReasoning: Read it.\nFinal answer: STOP",
                               finish_reason="stop")
    assert textvqa_format_ok(valid, "stop")
    assert not textvqa_format_ok(GenerationResponse("Final answer: STOP", finish_reason="stop"), "stop")
    assert normalize_answer("STOP", "textvqa") == "stop"


def test_complete_pairing_and_audit(tmp_path: Path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fixed-image")
    record = {"id": "1", "question": "What word is on the sign?", "image": "image.jpg", "choices": []}
    inference = tmp_path / "inference.jsonl"
    inference.write_text(json.dumps(record) + "\n")
    sample = load_local_dataset(str(inference), "textvqa")[0].sample
    run = tmp_path / "run"
    TextVQATextRunner(FakeBackend(), [sample], run,
                      hashlib.sha256(inference.read_bytes()).hexdigest(), "fake-revision").run()
    manifest, rows = audit(run, inference, 1)
    assert manifest["labels_read"] is False
    assert rows[0]["arms"]["text_aug_vote"]["answer"] == "stop"
    assert _raw_vote_answer(rows[0], "text_aug_vote") == "STOP"
    rows[0]["candidates"][4]["image_hashes"] = ["bad"]
    (run / "predictions.jsonl").write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="contract broken"):
        audit(run, inference, 1)
