"""
Sentence-level semantic grounding using all-MiniLM-L6-v2 embeddings. Each
generated summary sentence is compared against sentences from the supporting
reviews via cosine similarity. Sentences scoring below configurable thresholds
are flagged (weak < flag, medium between flag and strong, strong >= strong).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - allows running without dependency
    torch = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - allows running without dependency
    SentenceTransformer = None  # type: ignore

logger = logging.getLogger(__name__)


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    fragments = []
    sentence = []
    for char in text:
        sentence.append(char)
        if char in ".!?":
            fragments.append("".join(sentence).strip())
            sentence = []
    if sentence:
        fragments.append("".join(sentence).strip())
    return [frag for frag in fragments if frag]


@dataclass
class GroundingThresholds:
    flag: float
    strong: float


def categorize_similarity(score: float, thresholds: GroundingThresholds) -> str:
    if score < thresholds.flag:
        return "weak"
    if score < thresholds.strong:
        return "medium"
    return "strong"


class SemanticGrounder:
    """Semantic grounding helper around a MiniLM encoder."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        thresholds: GroundingThresholds = GroundingThresholds(flag=0.5, strong=0.7),
        device: str = None,
    ) -> None:
        self.model_name = model_name
        self.thresholds = thresholds
        cuda_available = bool(torch) and hasattr(torch, "cuda") and torch.cuda.is_available()
        self.device = device or ("cuda" if cuda_available else "cpu")
        self.available = SentenceTransformer is not None and torch is not None
        if self.available:
            try:
                self.model = SentenceTransformer(model_name, device=self.device)
            except Exception as exc:  # pragma: no cover - download failures
                logger.warning("Semantic grounding model unavailable: %s", exc)
                self.available = False
                self.model = None
        else:
            self.model = None

    def assess(self, summary: str, review_texts: List[str]) -> Dict:
        """Return similarity stats between summary sentences and review sentences."""
        summary_sentences = _split_sentences(summary)
        source_sentences: List[str] = []
        for text in review_texts:
            source_sentences.extend(_split_sentences(text))
        if not summary_sentences or not source_sentences:
            return {
                "strong": 0,
                "medium": 0,
                "weak": len(summary_sentences),
                "weak_sentence_indices": list(range(len(summary_sentences))),
                "sentence_scores": [
                    {"sentence": sent, "best_score": 0.0, "best_source": None}
                    for sent in summary_sentences
                ],
            }
        if not self.available or self.model is None:
            return self._heuristic_assessment(summary_sentences, source_sentences)
        summary_emb = self.model.encode(summary_sentences, convert_to_tensor=True, normalize_embeddings=True)
        source_emb = self.model.encode(source_sentences, convert_to_tensor=True, normalize_embeddings=True)
        scores = torch.matmul(summary_emb, source_emb.T)
        best_scores, best_indices = torch.max(scores, dim=1)
        best_scores = best_scores.detach().cpu().numpy()
        best_indices = best_indices.detach().cpu().numpy()
        categories = {"strong": 0, "medium": 0, "weak": 0}
        weak_indices: List[int] = []
        sentence_scores: List[Dict] = []
        for idx, (score, source_idx) in enumerate(zip(best_scores, best_indices)):
            label = categorize_similarity(float(score), self.thresholds)
            if label == "weak":
                weak_indices.append(idx)
            categories[label] += 1
            sentence_scores.append(
                {
                    "sentence": summary_sentences[idx],
                    "best_score": float(np.round(score, 4)),
                    "best_source": source_sentences[int(source_idx)],
                }
            )
        return {
            **categories,
            "weak_sentence_indices": weak_indices,
            "sentence_scores": sentence_scores,
        }

    def _heuristic_assessment(self, summary_sentences: List[str], source_sentences: List[str]) -> Dict:
        """Fallback scoring via token overlap when embeddings are unavailable."""
        categories = {"strong": 0, "medium": 0, "weak": 0}
        weak_indices: List[int] = []
        sentence_scores: List[Dict] = []
        for idx, summary_sent in enumerate(summary_sentences):
            summary_tokens = set(summary_sent.lower().split())
            best_score = 0.0
            best_source = None
            for source in source_sentences:
                source_tokens = set(source.lower().split())
                if not source_tokens:
                    continue
                overlap = len(summary_tokens & source_tokens) / len(source_tokens)
                if overlap > best_score:
                    best_score = overlap
                    best_source = source
            label = categorize_similarity(best_score, self.thresholds)
            if label == "weak":
                weak_indices.append(idx)
            categories[label] += 1
            sentence_scores.append(
                {"sentence": summary_sent, "best_score": round(best_score, 4), "best_source": best_source}
            )
        return {
            **categories,
            "weak_sentence_indices": weak_indices,
            "sentence_scores": sentence_scores,
        }
