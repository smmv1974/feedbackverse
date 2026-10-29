"""
Compare base vs alternate model runs. Flags when the absolute delta in corpus
sentiment exceeds config.evaluation.delta_sentiment_flag and reports theme
frequency differences.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import yaml


def load_results(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def aggregate_sentiment(results: Dict) -> float:
    sentiments = []
    for batch in results.get("batches", []):
        for rec in batch.get("sentiments", []):
            sentiments.append(rec.get("sentiment", 0.0))
    if not sentiments:
        return 0.0
    return sum(sentiments) / len(sentiments)


def top_theme_diff(base: Dict, alt: Dict, limit: int = 5) -> List[Dict]:
    def collect(data: Dict) -> Counter:
        counter = Counter()
        for batch in data.get("batches", []):
            counter.update([theme.lower() for theme in batch.get("themes", [])])
        return counter

    base_counter = collect(base)
    alt_counter = collect(alt)
    themes = set(base_counter) | set(alt_counter)
    diffs = []
    for theme in themes:
        diffs.append(
            {
                "theme": theme,
                "base_count": base_counter.get(theme, 0),
                "alt_count": alt_counter.get(theme, 0),
                "delta": alt_counter.get(theme, 0) - base_counter.get(theme, 0),
            }
        )
    diffs.sort(key=lambda item: abs(item["delta"]), reverse=True)
    return diffs[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base vs alt model outputs.")
    parser.add_argument("--base", required=True, help="Path to base analysis_results.json")
    parser.add_argument("--alt", required=True, help="Path to alt analysis_results.json")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="data/results/ablation_flag.json")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    threshold = config.get("evaluation", {}).get("delta_sentiment_flag", 0.3)
    base_results = load_results(Path(args.base))
    alt_results = load_results(Path(args.alt))
    base_avg = aggregate_sentiment(base_results)
    alt_avg = aggregate_sentiment(alt_results)
    delta = abs(base_avg - alt_avg)
    payload = {
        "base_avg_sentiment": base_avg,
        "alt_avg_sentiment": alt_avg,
        "delta_sentiment": delta,
        "flagged": delta > threshold,
        "threshold": threshold,
        "top_theme_differences": top_theme_diff(base_results, alt_results),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    main()
