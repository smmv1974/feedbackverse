"""
Validation script comparing model outputs against human references.
Computes ROUGE, BERTScore, Cohen's κ (binned), MAE, Pearson r/p, ANOVA, and
theme precision/recall. Expects references CSV with columns:
review_id, ref_summary, human_sentiment, human_themes, platform.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

from src.evaluation.human_agreement import (
    anova_by_platform,
    kappa_binned,
    mae_continuous,
    pearson_corr,
)
from src.evaluation.text_overlap import (
    BERT_AVAILABLE,
    ROUGE_AVAILABLE,
    bert_score_f1,
    rouge_all,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYNONYM_MAP = {
    "battery life": "battery",
    "battery": "battery",
    "camera": "camera",
    "photo": "camera",
    "service": "support",
    "customer support": "support",
    "support": "support",
    "price": "pricing",
    "cost": "pricing",
    "design": "design",
    "build quality": "design",
}


def load_analysis_results(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def normalize_themes(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, float) and pd.isna(raw):
        return []
    if isinstance(raw, str):
        candidates = [part.strip() for part in re.split(r"[;,]", raw)]
    else:
        candidates = [str(item).strip() for item in raw]
    normalized: List[str] = []
    for item in candidates:
        key = item.lower()
        mapped = SYNONYM_MAP.get(key, key)
        if mapped:
            normalized.append(mapped)
    return list(dict.fromkeys(normalized))


def build_lookup(analysis: Dict) -> Dict[str, Dict]:
    lookup: Dict[str, Dict] = {}
    for batch in analysis.get("batches", []):
        summary = batch.get("summary")
        themes = batch.get("themes", [])
        review_meta = {rid: {"batch_summary": summary, "batch_themes": themes} for rid in batch.get("review_ids", [])}
        for review_id, meta in review_meta.items():
            lookup.setdefault(review_id, {}).update(meta)
        for sentiment in batch.get("sentiments", []):
            lookup.setdefault(sentiment["review_id"], {}).update(
                {
                    "model_sentiment": sentiment["sentiment"],
                    "platform": sentiment.get("platform"),
                }
            )
    return lookup


def theme_pr(rec_human: List[List[str]], rec_model: List[List[str]]) -> Tuple[float, float]:
    if not rec_human:
        return 0.0, 0.0
    precisions = []
    recalls = []
    for human, model in zip(rec_human, rec_model):
        if not model:
            precisions.append(0.0)
        else:
            precisions.append(len(set(model) & set(human)) / len(model))
        if not human:
            recalls.append(0.0)
        else:
            recalls.append(len(set(model) & set(human)) / len(human))
    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)


def main(args: argparse.Namespace) -> None:
    references_path = Path(args.references)
    if not references_path.exists():
        logger.error("Reference file %s not found.", references_path)
        output_placeholder = {"error": "references_missing"}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(output_placeholder, fh, indent=2)
        return
    analysis_path = Path(args.analysis)
    if not analysis_path.exists():
        logger.error("Analysis results file %s not found.", analysis_path)
        output_placeholder = {"error": "analysis_results_missing"}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(output_placeholder, fh, indent=2)
        return
    analysis = load_analysis_results(analysis_path)
    lookup = build_lookup(analysis)
    refs = pd.read_csv(references_path)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    edges = config.get("evaluation", {}).get("sentiment_bin_edges", [-1, -0.5, 0, 0.5, 1])
    model_summaries: List[str] = []
    human_summaries: List[str] = []
    human_sentiments: List[float] = []
    model_sentiments: List[float] = []
    platforms: List[str] = []
    human_theme_lists: List[List[str]] = []
    model_theme_lists: List[List[str]] = []
    for row in refs.itertuples():
        row_lookup = lookup.get(row.review_id)
        if not row_lookup:
            continue
        if pd.isna(row.human_sentiment):
            continue
        ref_summary = getattr(row, "ref_summary", "")
        if pd.isna(ref_summary):
            ref_summary = ""
        model_summaries.append(row_lookup.get("batch_summary", ""))
        human_summaries.append(ref_summary)
        human_sentiments.append(float(row.human_sentiment))
        model_sentiments.append(float(row_lookup.get("model_sentiment", 0.0)))
        platform_value = getattr(row, "platform", None)
        if pd.isna(platform_value):
            platform_value = row_lookup.get("platform", "unknown")
        platforms.append(platform_value)
        human_theme_lists.append(normalize_themes(row.human_themes))
        model_theme_lists.append(normalize_themes(row_lookup.get("batch_themes", [])))
    if not model_summaries:
        metrics = {"error": "no_overlap_between_references_and_outputs"}
    else:
        rouge_components = [rouge_all(m, h) for m, h in zip(model_summaries, human_summaries)]
        rouge_scores = {
            "rouge1": round(sum(item["rouge1"] for item in rouge_components) / len(rouge_components), 4),
            "rouge2": round(sum(item["rouge2"] for item in rouge_components) / len(rouge_components), 4),
            "rougeL": round(sum(item["rougeL"] for item in rouge_components) / len(rouge_components), 4),
        }
        try:
            bert = bert_score_f1(model_summaries, human_summaries)
        except Exception as exc:
            logger.warning("BERTScore failed: %s", exc)
            bert = None
        kappa = kappa_binned(human_sentiments, model_sentiments, edges)
        mae = mae_continuous(human_sentiments, model_sentiments)
        pearson = pearson_corr(human_sentiments, model_sentiments)
        anova = anova_by_platform(model_sentiments, platforms)
        precision, recall = theme_pr(human_theme_lists, model_theme_lists)
        metrics = {
            **rouge_scores,
            "bertscore_F1": bert,
            "kappa_binned": kappa,
            "mae": mae,
            "pearson_r": pearson["r"],
            "pearson_p": pearson["p"],
            "anova_F": anova["F"],
            "anova_p": anova["p"],
            "theme_precision": precision,
            "theme_recall": recall,
            "n": len(model_summaries),
            "fallback_overlap": not (BERT_AVAILABLE and ROUGE_AVAILABLE),
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate model outputs against human references.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--analysis", default="data/results/analysis_results.json")
    parser.add_argument("--references", default="data/validation/references.csv")
    parser.add_argument("--output", default="data/results/validation_metrics.json")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
