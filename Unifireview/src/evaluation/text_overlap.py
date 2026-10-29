"""
Text overlap metrics using ROUGE (1/2/L F1) and BERTScore (mean F1) for system
vs. human summaries. ROUGE follows Lin (2004) via rouge-score; BERTScore uses
contextual embeddings (Zhang et al., 2020) with huggingface defaults.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

try:
    from bert_score import score as bert_score
except ImportError:  # pragma: no cover - optional dependency
    bert_score = None

try:
    from rouge_score import rouge_scorer
except ImportError:  # pragma: no cover - optional dependency
    rouge_scorer = None

BERT_AVAILABLE = bert_score is not None
ROUGE_AVAILABLE = rouge_scorer is not None


def rouge_all(hypothesis: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1/2/L F1 scores between hypothesis and reference."""
    if not hypothesis or not reference:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    if not ROUGE_AVAILABLE:
        hyp_tokens = hypothesis.split()
        ref_tokens = reference.split()
        overlap = len(set(hyp_tokens) & set(ref_tokens))
        denom = max(1, len(ref_tokens))
        f1 = overlap / denom
        return {"rouge1": round(f1, 4), "rouge2": round(f1, 4), "rougeL": round(f1, 4)}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def bert_score_f1(hypotheses: List[str], references: List[str], lang: str = "en") -> float:
    """Return mean BERTScore F1 for a list of hypothesis/reference summaries."""
    if not hypotheses or not references:
        return 0.0
    if len(hypotheses) != len(references):
        raise ValueError("Hypotheses and references must align 1:1.")
    if not BERT_AVAILABLE:
        # Fallback: simple token overlap proxy when bert-score is unavailable.
        overlaps = []
        for hyp, ref in zip(hypotheses, references):
            hyp_tokens = set(hyp.split())
            ref_tokens = set(ref.split())
            if not hyp_tokens or not ref_tokens:
                overlaps.append(0.0)
                continue
            overlaps.append(len(hyp_tokens & ref_tokens) / len(ref_tokens))
        return round(sum(overlaps) / len(overlaps), 4)
    precisions, recalls, f1s = bert_score(hypotheses, references, lang=lang)
    return round(float(f1s.mean().item()), 4)
