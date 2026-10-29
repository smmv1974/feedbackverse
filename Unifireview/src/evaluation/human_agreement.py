"""
Human-model agreement metrics: Cohen's kappa on binned sentiment, mean absolute
error and Pearson correlation on continuous sentiment, plus one-way ANOVA across
platform groups to detect systematic bias.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score, mean_absolute_error


def bin_scores(scores: Sequence[float], edges: Sequence[float]) -> List[int]:
    """Assign each score to a bin index based on monotonically increasing edges."""
    edges_arr = np.array(edges, dtype=float)
    bins = np.digitize(scores, edges_arr, right=False) - 1
    return bins.tolist()


def kappa_binned(
    human_scores: Sequence[float], model_scores: Sequence[float], edges: Sequence[float]
) -> float:
    human_bins = bin_scores(human_scores, edges)
    model_bins = bin_scores(model_scores, edges)
    return float(cohen_kappa_score(human_bins, model_bins))


def mae_continuous(human_scores: Sequence[float], model_scores: Sequence[float]) -> float:
    return float(mean_absolute_error(human_scores, model_scores))


def pearson_corr(human_scores: Sequence[float], model_scores: Sequence[float]) -> Dict[str, float]:
    try:
        r, p_val = stats.pearsonr(human_scores, model_scores)
        return {"r": float(r), "p": float(p_val)}
    except Exception:
        return {"r": float("nan"), "p": float("nan")}


def anova_by_platform(scores: Sequence[float], platforms: Iterable[str]) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = defaultdict(list)
    for platform, score in zip(platforms, scores):
        buckets[str(platform)].append(score)
    valid_groups = [vals for vals in buckets.values() if len(vals) > 1]
    if len(valid_groups) < 2:
        return {"F": float("nan"), "p": float("nan")}
    f_stat, p_val = stats.f_oneway(*valid_groups)
    return {"F": float(f_stat), "p": float(p_val)}
