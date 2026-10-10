"""Data contracts shared by the multimodal runner.

Inference-facing samples deliberately have no answer/label field.  Dataset
adapters return labels in a separate ``EvaluationTarget`` object that is never
passed to a model or memory component.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence


EvidenceStatus = Literal["entailed", "contradicted", "insufficient"]


@dataclass(frozen=True)
class MultimodalSample:
    sample_id: str
    dataset: str
    question: str
    image_paths: Sequence[str] = field(default_factory=tuple)
    choices: Sequence[str] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if not self.question.strip():
            raise ValueError("question must not be empty")

    def validate_images(self) -> None:
        missing = [path for path in self.image_paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing image files: {missing}")

    def to_public_dict(self) -> Dict[str, Any]:
        """Serialize only fields that are safe to expose during inference."""
        return asdict(self)


@dataclass(frozen=True)
class EvaluationTarget:
    sample_id: str
    answers: Sequence[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.0
    top_p: Optional[float] = None
    max_tokens: int = 256
    seed: Optional[int] = None
    request_logprobs: bool = False


@dataclass
class GenerationResponse:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    logprobs: List[float] = field(default_factory=list)
    latency_seconds: float = 0.0

    @property
    def mean_logprob(self) -> Optional[float]:
        if not self.logprobs:
            return None
        return sum(self.logprobs) / len(self.logprobs)


@dataclass
class Candidate:
    candidate_id: str
    trajectory: str
    response: GenerationResponse
    normalized_answer: str
    question_variant: str
    image_paths: Sequence[str]
    memory_conditioned: bool = False


@dataclass(frozen=True)
class EvidenceClaim:
    text: str
    status: EvidenceStatus
    evidence: str = ""


@dataclass
class EvidenceVerdict:
    answer_status: EvidenceStatus
    claims: List[EvidenceClaim] = field(default_factory=list)
    rationale: str = ""
    parse_error: bool = False

    @property
    def score(self) -> float:
        values = {"entailed": 1.0, "insufficient": 0.5, "contradicted": 0.0}
        parts = [values[self.answer_status]] + [values[c.status] for c in self.claims]
        return sum(parts) / len(parts)


@dataclass(frozen=True)
class ComponentScores:
    invariance: float
    sensitivity: float
    evidence: float
    mean_logprob: Optional[float] = None


@dataclass
class SelectionDecision:
    answer: str
    representative: Candidate
    scores: ComponentScores
    borda_score: float
    component_wins: int
    eligible_for_memory: bool
    negative_answer: Optional[str] = None
    negative_representative: Optional[Candidate] = None
    reason: str = ""


@dataclass
class Experience:
    experience_id: str
    sample_index: int
    sample_id: str
    dataset: str
    question: str
    image_paths: Sequence[str]
    image_hashes: Sequence[str]
    positive_answer: str
    negative_answer: Optional[str]
    positive_rule: str
    negative_rule: str
    evidence_summary: str
    evidence_claims: Sequence[EvidenceClaim]
    component_scores: ComponentScores
    trust: float
    timestamp: float
    text_embedding: Sequence[float] = field(default_factory=tuple, repr=False)
    image_embedding: Sequence[float] = field(default_factory=tuple, repr=False)
    joint_embedding: Sequence[float] = field(default_factory=tuple, repr=False)

    def to_dict(self, include_embeddings: bool = False) -> Dict[str, Any]:
        data = asdict(self)
        if not include_embeddings:
            data.pop("text_embedding", None)
            data.pop("image_embedding", None)
            data.pop("joint_embedding", None)
        return data
