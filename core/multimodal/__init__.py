"""EviMem-TTCL: evidence-aware online adaptation for frozen VLMs."""

from .schema import (
    Candidate,
    ComponentScores,
    EvidenceClaim,
    EvidenceVerdict,
    Experience,
    MultimodalSample,
    SamplingConfig,
    SelectionDecision,
)

__all__ = [
    "Candidate",
    "ComponentScores",
    "EvidenceClaim",
    "EvidenceVerdict",
    "Experience",
    "MultimodalSample",
    "SamplingConfig",
    "SelectionDecision",
]
