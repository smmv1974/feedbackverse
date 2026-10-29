"""
Weekly sentiment drift analysis: average model sentiment grouped by ISO week.
Provides week-start timestamps (Mondays) to enable trend visualizations and CSV
exports for downstream dashboards.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Mapping

import pandas as pd


def week_start(date_val: datetime) -> datetime:
    """Return the Monday (ISO week start) for a given datetime."""
    return (date_val - timedelta(days=date_val.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def weekly_sentiment(records: Iterable[Mapping], output_csv: Path) -> pd.DataFrame:
    """
    Compute mean sentiment by ISO week from iterable sentiment records containing
    'review_date' (ISO string) and 'sentiment' values.
    """
    rows: List[Mapping] = list(records)
    if not rows:
        df = pd.DataFrame(columns=["week_start", "mean_sentiment", "count"])
        df.to_csv(output_csv, index=False)
        return df
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["review_date", "sentiment"])
    if df.empty:
        df = pd.DataFrame(columns=["week_start", "mean_sentiment", "count"])
        df.to_csv(output_csv, index=False)
        return df
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df = df.dropna(subset=["review_date"])
    df["week_start"] = df["review_date"].apply(lambda d: week_start(d).date().isoformat())
    grouped = (
        df.groupby("week_start")["sentiment"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "mean_sentiment"})
        .reset_index()
        .sort_values("week_start")
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_csv, index=False)
    return grouped
