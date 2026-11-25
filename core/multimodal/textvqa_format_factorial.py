"""Frozen prompt-format comparison with separate online memory streams.

Each stream runs the existing direct-transfer algorithm unchanged. The two
format policies have separate rule memories and output directories. Pass the
same exact-request trace backend to both streams to reuse identical requests
(for example, Tutor rewrites); the trace key must include every request field.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .image_ops import sha256_file
from .schema import MultimodalSample
from .tf_ttcl_direct_runner import DirectRunner, RULE_HEADER, TUTOR_SYSTEM, _digest


POLICIES = ('original', 'short_final')
SHORT_FINAL_SUFFIX = (
    '\nOn the Final answer line, provide only the answer required by the question. '
    'Do not repeat the question or include an explanation in the answer. '
    'Preserve all words, digits, and symbols needed for the answer; '
    'do not impose a fixed word limit. Keep the Reasoning section before the final line.'
)
FACTORIAL_VERSION = 'textvqa_format_factorial_v1'


class TextVQAFormatRunner(DirectRunner):
    """Original open-domain TF-TTCL, with one frozen final-line prompt policy."""

    def __init__(self, backend, embedder, samples: Sequence[MultimodalSample],
                 output: Path, input_sha256: str, model_revision: str,
                 embedding_revision: str, seed: int = 0, policy: str = 'original'):
        if policy not in POLICIES:
            raise ValueError(f'Unknown format policy: {policy}')
        if not samples or any(sample.dataset != 'textvqa' for sample in samples):
            raise ValueError('Only nonempty TextVQA streams are supported')
        super().__init__(backend, embedder, samples, output, input_sha256,
                         model_revision, embedding_revision, seed)
        if policy == 'short_final':
            self.teacher_system += SHORT_FINAL_SUFFIX
            self.student_system += SHORT_FINAL_SUFFIX
        self.prompt_sha = _digest([self.teacher_system, self.student_system, TUTOR_SYSTEM,
                                   RULE_HEADER, self.positive_template, self.negative_template])
        self.config.update(format_experiment=FACTORIAL_VERSION, format_policy=policy,
                           prompt_sha256=self.prompt_sha)
        self.config_sha = _digest(self.config)
        self.code_sha = _digest([self.code_sha, sha256_file(str(Path(__file__)))])


def build_format_runners(backend, embedder, samples: Sequence[MultimodalSample],
                         output: Path, input_sha256: str, model_revision: str,
                         embedding_revision: str, seed: int = 0) -> dict[str, TextVQAFormatRunner]:
    """Build two isolated streams; caller runs both before any label access.

    Each individual stream yields greedy, same-text candidate, no-memory and
    full-memory arms. The four factorial cells are no-memory/full-memory from
    each policy. Never copy rules between the two runner instances.
    """
    root = Path(output)
    return {policy: TextVQAFormatRunner(backend, embedder, samples, root / policy,
                                         input_sha256, model_revision, embedding_revision,
                                         seed=seed, policy=policy)
            for policy in POLICIES}
