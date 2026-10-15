"""Paired, no-memory TextVQA question-rewrite baseline."""
from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

from .answers import extract_final_answer, normalize_answer
from .image_ops import sha256_file
from .prompts import format_question
from .schema import MultimodalSample
from .text_only_runner import TextOnlyRunner
from .transfer_runner import digest


VERSION = "textvqa_text_only_vote_v1"
TEXTVQA_SYSTEM = """Answer the question using the supplied image. Read visible text
carefully; do not invent words you cannot see. Use exactly three concise labeled
sections in this order:
Observed: <relevant visible text or facts, or insufficient>
Reasoning: <brief inference from the observation>
Final answer: <one short answer only, no explanation after it>"""
TEXTVQA_SUFFIX = """Return exactly the three labeled sections Observed, Reasoning,
Final answer. The final line must contain only the answer, preserving its visible
wording when possible. Do not add text after it."""
_FINAL = re.compile(r"(?m)^Final answer:[ \t]*(\S.*?)[ \t]*\s*\Z")


def textvqa_format_ok(response, answer: str) -> bool:
    content = response.content
    if (response.finish_reason != "stop" or not content.startswith("Observed:")
            or len(re.findall(r"(?m)^Final answer:", content)) != 1
            or not _FINAL.search(content) or not answer):
        return False
    observed = re.search(r"(?ms)^Observed:\s*(.+?)\nReasoning:", content)
    reasoning = re.search(r"(?ms)^Reasoning:\s*(.+?)\nFinal answer:", content)
    return bool(observed and reasoning and observed.group(1).strip() and reasoning.group(1).strip())


class TextVQATextRunner(TextOnlyRunner):
    def __init__(self, backend, samples, output: Path, input_sha256: str,
                 model_revision: str, seed: int = 0) -> None:
        if any(sample.dataset != "textvqa" for sample in samples):
            raise ValueError("TextVQA samples required")
        super().__init__(backend, samples, output, input_sha256, model_revision, seed)
        self.prompt_sha = digest([self.prompt_sha, TEXTVQA_SYSTEM, TEXTVQA_SUFFIX])
        self.code_sha = digest([self.code_sha, sha256_file(Path(__file__))])
        self.config.update(version=VERSION, dataset="textvqa", prompt_sha256=self.prompt_sha)
        self.config_sha = digest(self.config)

    def _candidate(self, sample: MultimodalSample, index: int, name: str,
                   question: str, offset: int, temperature: float) -> dict:
        user = format_question(question, sample.choices, sample.metadata) + "\n\n" + TEXTVQA_SUFFIX
        response = self._call(TEXTVQA_SYSTEM, user, sample.image_paths, index,
                              offset, temperature, 384)
        raw_answer = extract_final_answer(response.content)
        answer = normalize_answer(raw_answer, "textvqa")
        return {"name": name, "question": question,
                "image_hashes": [sha256_file(path) for path in sample.image_paths],
                "answer": answer, "raw_answer": raw_answer,
                "format_ok": textvqa_format_ok(response, answer),
                "response": asdict(response)}
