"""Paired, no-memory MathVista question-rewrite baseline.

Reuses the frozen v2 rewrite guard and paired sampling design, but gives
MathVista's free-form and multiple-choice answers their own format contract.
"""
from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

from .answers import normalize_answer
from .image_ops import sha256_file
from .prompts import format_question
from .schema import MultimodalSample
from .text_only_runner import TextOnlyRunner
from .transfer_runner import digest


VERSION = "mathvista_text_only_vote_v1"
MATH_SYSTEM = """Answer the question using the supplied image when relevant.
Keep visible observations distinct from calculations. Do not invent an image
detail. Use exactly three concise labeled sections in this order:
Observed: <relevant visible facts, or not required / insufficient>
Reasoning: <brief calculation or inference>
Final answer: <one short answer, with no explanation after it>"""
MATH_SUFFIX = """Return exactly the three labeled sections Observed, Reasoning,
Final answer. Put only the requested value (or current option letter for a
multiple-choice question) on the final line. Do not add text after it."""
_FINAL = re.compile(r"(?m)^Final answer:[ \t]*(\S.*?)[ \t]*\s*\Z")


def math_format_ok(response, sample: MultimodalSample, answer: str) -> bool:
    text = response.content
    if (response.finish_reason != "stop" or not text.startswith("Observed:")
            or len(re.findall(r"(?m)^Final answer:", text)) != 1
            or not _FINAL.search(text) or not answer):
        return False
    observed = re.search(r"(?ms)^Observed:\s*(.+?)\nReasoning:", text)
    reasoning = re.search(r"(?ms)^Reasoning:\s*(.+?)\nFinal answer:", text)
    if not observed or not reasoning or not observed.group(1).strip() or not reasoning.group(1).strip():
        return False
    if sample.metadata.get("question_type") == "multi_choice":
        return answer in [normalize_answer(choice, "mathvista") for choice in sample.choices]
    return True


class MathVistaTextRunner(TextOnlyRunner):
    def __init__(self, backend, samples, output: Path, input_sha256: str,
                 model_revision: str, seed: int = 0) -> None:
        if any(sample.dataset != "mathvista" for sample in samples):
            raise ValueError("MathVista samples required")
        super().__init__(backend, samples, output, input_sha256, model_revision, seed)
        self.prompt_sha = digest([self.prompt_sha, MATH_SYSTEM, MATH_SUFFIX])
        self.code_sha = digest([self.code_sha, sha256_file(Path(__file__))])
        self.config.update(version=VERSION, dataset="mathvista", prompt_sha256=self.prompt_sha)
        self.config_sha = digest(self.config)

    def _candidate(self, sample, index, name, question, offset, temperature):
        user = format_question(question, sample.choices, sample.metadata) + "\n\n" + MATH_SUFFIX
        response = self._call(MATH_SYSTEM, user, sample.image_paths, index,
                              offset, temperature, 384)
        answer = normalize_answer(response.content, "mathvista",
                                  {**sample.metadata, "choices": list(sample.choices)})
        return {"name": name, "question": question,
                "image_hashes": [sha256_file(path) for path in sample.image_paths],
                "answer": answer, "format_ok": math_format_ok(response, sample, answer),
                "response": asdict(response)}
