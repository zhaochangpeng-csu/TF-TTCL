"""Temporal-safe, trust-weighted contrastive evidence memory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .embeddings import MultimodalEmbedder
from .schema import ComponentScores, EvidenceClaim, Experience, MultimodalSample


@dataclass(frozen=True)
class RetrievedRule:
    content: str
    rule_type: str
    experience_id: str
    score: float


@dataclass(frozen=True)
class RetrievedEpisode:
    experience: Experience
    score: float


@dataclass(frozen=True)
class RetrievalBundle:
    rules: Sequence[RetrievedRule]
    episodes: Sequence[RetrievedEpisode]


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


class EvidenceMemory:
    def __init__(
        self,
        embedder: MultimodalEmbedder,
        max_experiences: int = 1000,
        rrf_k: int = 60,
    ) -> None:
        self.embedder = embedder
        self.max_experiences = max_experiences
        self.rrf_k = rrf_k
        self._experiences: List[Experience] = []

    def __len__(self) -> int:
        return len(self._experiences)

    @property
    def experiences(self) -> Sequence[Experience]:
        return tuple(self._experiences)

    def add(self, experience: Experience) -> None:
        if self._experiences and experience.sample_index <= self._experiences[-1].sample_index:
            raise ValueError("Experiences must be appended in strict stream order")
        if not experience.text_embedding:
            experience.text_embedding = self.embedder.encode_text(experience.question).tolist()
        if not experience.image_embedding:
            experience.image_embedding = self.embedder.encode_image(experience.image_paths).tolist()
        if not experience.joint_embedding:
            experience.joint_embedding = self.embedder.encode_joint(
                experience.question, experience.image_paths
            ).tolist()
        self._experiences.append(experience)
        if len(self._experiences) > self.max_experiences:
            self._experiences.pop(0)

    def retrieve(
        self,
        sample: MultimodalSample,
        sample_index: int,
        top_rules: int = 4,
        top_episodes: int = 1,
        modalities: Sequence[str] = ("text", "image", "joint"),
    ) -> RetrievalBundle:
        eligible = [e for e in self._experiences if e.sample_index < sample_index]
        if not eligible:
            return RetrievalBundle((), ())

        query_factories = {
            "text": lambda: self.embedder.encode_text(sample.question),
            "image": lambda: self.embedder.encode_image(sample.image_paths),
            "joint": lambda: self.embedder.encode_joint(sample.question, sample.image_paths),
        }
        attributes = {
            "text": "text_embedding",
            "image": "image_embedding",
            "joint": "joint_embedding",
        }
        unknown = set(modalities) - set(query_factories)
        if not modalities or unknown:
            raise ValueError(f"modalities must be a non-empty subset of {sorted(query_factories)}")
        fused: Dict[str, float] = {item.experience_id: 0.0 for item in eligible}
        for modality in modalities:
            query = query_factories[modality]()
            ranked = sorted(
                eligible,
                key=lambda item: _cosine(
                    query, np.asarray(getattr(item, attributes[modality]), dtype=np.float32)
                ),
                reverse=True,
            )
            for rank, item in enumerate(ranked, start=1):
                fused[item.experience_id] += item.trust / (self.rrf_k + rank)

        ordered = sorted(eligible, key=lambda item: fused[item.experience_id], reverse=True)
        episodes = tuple(
            RetrievedEpisode(item, fused[item.experience_id])
            for item in ordered[:top_episodes]
        )
        rules: List[RetrievedRule] = []
        for item in ordered:
            if item.positive_rule:
                rules.append(
                    RetrievedRule(
                        item.positive_rule, "positive", item.experience_id, fused[item.experience_id]
                    )
                )
            if item.negative_rule:
                rules.append(
                    RetrievedRule(
                        item.negative_rule, "negative", item.experience_id, fused[item.experience_id]
                    )
                )
            if len(rules) >= top_rules:
                break
        return RetrievalBundle(tuple(rules[:top_rules]), episodes)

    def save_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for experience in self._experiences:
                handle.write(json.dumps(experience.to_dict(include_embeddings=True), ensure_ascii=False) + "\n")

    def load_jsonl(self, path: Path) -> None:
        self._experiences.clear()
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                payload["evidence_claims"] = [EvidenceClaim(**item) for item in payload["evidence_claims"]]
                payload["component_scores"] = ComponentScores(**payload["component_scores"])
                self.add(Experience(**payload))
