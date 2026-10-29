"""
Preprocessing pipeline that deduplicates, filters, normalizes, and persists
multi-platform review corpora while producing auditable statistics. The
pipeline follows a reproducible sequence: load raw reviews, remove exact and
near duplicates (via normalized Levenshtein similarity), filter spam/short/non-
English entries, normalize ratings onto a 1–5 scale, and persist the cleaned
dataset to both CSV and SQLite alongside descriptive corpus metrics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:  # pragma: no cover - fallback using difflib
    from difflib import SequenceMatcher

    class _Levenshtein:
        @staticmethod
        def normalized_similarity(a: str, b: str) -> float:
            return SequenceMatcher(None, a, b).ratio()

    Levenshtein = _Levenshtein()

try:  # langdetect is optional during unit tests
    from langdetect import LangDetectException, detect
except ImportError:  # pragma: no cover - fallback when langdetect missing
    class LangDetectException(Exception):
        pass

    def detect(_: str) -> str:
        return "en"

logger = logging.getLogger(__name__)


URL_PATTERN = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
PROMO_PATTERN = re.compile(r"(promo\s?code|discount\s?code|use\s?code)", re.IGNORECASE)

REQUIRED_COLUMNS = [
    "platform",
    "category",
    "product_name",
    "rating",
    "review_text",
    "review_date",
    "reviewer_name",
    "verified_purchase",
]

OUTPUT_COLUMNS = [
    "review_id",
    "platform",
    "category",
    "product_name",
    "rating_norm",
    "review_text",
    "review_date",
    "reviewer_name",
    "verified_purchase",
    "word_count",
]

NEAR_DUP_THRESHOLD = 0.85
MIN_WORD_COUNT = 10


@dataclass
class PreprocessPaths:
    raw_dir: Path
    scraped_dir: Path
    processed_csv: Path
    processed_sqlite: Path
    descriptive_json: Path
    near_dup_csv: Path


def load_config(config_path: Path) -> Dict:
    """Load YAML configuration into memory."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _collect_source_files(paths: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for folder in paths:
        if not folder.exists():
            continue
        files.extend(sorted(folder.glob("*.csv")))
    return files


def load_raw_reviews(paths: PreprocessPaths) -> Tuple[pd.DataFrame, int]:
    """
    Load raw review CSVs from data/raw and data/scraped; errors if none exist to
    prevent reusing stale processed outputs.
    """
    files = _collect_source_files([paths.raw_dir, paths.scraped_dir])
    dfs: List[pd.DataFrame] = []
    for csv_path in files:
        try:
            dfs.append(pd.read_csv(csv_path))
            logger.info("Loaded raw file %s", csv_path)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", csv_path, exc)
    if not dfs:
        raise FileNotFoundError("No raw CSVs found under data/raw or data/scraped.")
    df = pd.concat(dfs, ignore_index=True)
    return df, len(df)


def normalize_rating(raw_value: Optional[str]) -> float:
    """
    Map heterogeneous rating formats onto a 1.0–5.0 scale. Supports literals
    like "4/5", "8 out of 10", percentages, and direct numeric scores.
    """
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return np.nan
    if isinstance(raw_value, (int, float)):
        value_str = str(raw_value)
    else:
        value_str = str(raw_value).strip()
        if not value_str:
            return np.nan
    ratio: Optional[float] = None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)", value_str)
    if match:
        num = float(match.group(1))
        denom = float(match.group(2))
        if denom:
            if 4.0 <= denom <= 5.5:
                return float(np.clip(num, 1.0, 5.0))
            ratio = max(0.0, min(1.0, num / denom))
    elif "%" in value_str:
        digits = re.findall(r"[0-9]+(?:\.[0-9]+)?", value_str)
        if digits:
            num = float(digits[0])
            ratio = max(0.0, min(1.0, num / 100.0))
    elif re.search(r"out\s+of\s+10", value_str, re.IGNORECASE):
        digits = re.findall(r"[0-9]+(?:\.[0-9]+)?", value_str)
        if digits:
            num = float(digits[0])
            ratio = max(0.0, min(1.0, num / 10.0))
    else:
        try:
            num = float(value_str)
        except ValueError:
            return np.nan
        if 0 <= num <= 5:
            return float(np.clip(num, 1.0, 5.0))
        if 0 <= num <= 10:
            ratio = num / 10.0
        elif 0 <= num <= 100:
            ratio = num / 100.0
        else:
            ratio = None
    if ratio is None:
        return np.nan
    return float(1.0 + 4.0 * ratio)


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text).strip().lower())
    return cleaned


def _hash_text(text: str) -> str:
    return hashlib.sha1(_normalize_text(text).encode("utf-8")).hexdigest()


def _assign_review_id(row: pd.Series) -> str:
    composite = "|".join(
        [
            str(row.get("platform", "")).lower(),
            str(row.get("reviewer_name", "")).strip(),
            str(row.get("review_date", "")),
            _normalize_text(row.get("review_text", "")),
        ]
    )
    return hashlib.md5(composite.encode("utf-8")).hexdigest()


def _is_spam(text: str) -> bool:
    if not isinstance(text, str):
        return True
    if URL_PATTERN.search(text) or EMAIL_PATTERN.search(text) or PROMO_PATTERN.search(text):
        return True
    word_count = len(text.split())
    return word_count < MIN_WORD_COUNT


def _is_english(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _window_filter(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if not start and not end:
        return df
    df = df.copy()
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    if start:
        df = df[df["review_date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["review_date"] <= pd.to_datetime(end)]
    df["review_date"] = df["review_date"].dt.strftime("%Y-%m-%d")
    return df


def remove_near_duplicates(
    df: pd.DataFrame,
    threshold: float = NEAR_DUP_THRESHOLD,
    group_keys: Tuple[str, ...] = ("platform", "product_name"),
) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    """
    Drop later duplicates where the normalized Levenshtein similarity between
    review texts exceeds the given threshold; return filtered frame and log rows.
    """
    kept_rows: List[pd.Series] = []
    logs: List[Dict[str, str]] = []
    cache: Dict[Tuple, List[Tuple[int, str]]] = {}
    for idx, row in df.iterrows():
        text = row["review_text"]
        if not isinstance(text, str):
            continue
        key = tuple(row.get(k) for k in group_keys)
        cache.setdefault(key, [])
        duplicate_found = False
        for kept_idx, kept_text in cache[key]:
            similarity = Levenshtein.normalized_similarity(text, kept_text)
            if similarity >= threshold:
                logs.append(
                    {
                        "kept_review_id": df.at[kept_idx, "review_id"],
                        "removed_review_id": row["review_id"],
                        "similarity": round(float(similarity), 4),
                        "kept_preview": kept_text[:120],
                        "removed_preview": text[:120],
                    }
                )
                duplicate_found = True
                break
        if not duplicate_found:
            kept_rows.append(row)
            cache[key].append((idx, text))
    filtered_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    return filtered_df, logs


def _persist_near_dup_log(logs: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["kept_review_id", "removed_review_id", "similarity", "kept_preview", "removed_preview"]
    df = pd.DataFrame(logs)
    if df.empty:
        df = pd.DataFrame(columns=columns)
    df.to_csv(path, index=False)


def _persist_sqlite(df: pd.DataFrame, sqlite_path: Path) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as conn:
        df.to_sql("reviews", conn, if_exists="replace", index=False)


def _descriptive_stats(df: pd.DataFrame, raw_n: int) -> Dict:
    rating_series = df["rating_norm"].dropna().astype(float)
    n_kept = len(df)
    stats = {
        "raw_n": raw_n,
        "n": n_kept,
        "kept_pct": round((n_kept / raw_n * 100) if raw_n else 0.0, 2),
        "median_rating": round(float(rating_series.median()) if not rating_series.empty else np.nan, 3),
        "pct_5": round(float((rating_series >= 4.75).mean() * 100) if not rating_series.empty else 0.0, 2),
        "pct_4": round(float(((rating_series >= 3.75) & (rating_series < 4.75)).mean() * 100) if not rating_series.empty else 0.0, 2),
        "mean_word_count": round(float(df["word_count"].mean()) if not df.empty else np.nan, 2),
        "domain_counts": df["category"].fillna("unknown").value_counts().to_dict(),
        "platform_split": df["platform"].fillna("unknown").value_counts().to_dict(),
    }
    return stats


def _write_json(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def preprocess(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """
    Run the preprocessing pipeline end-to-end and return the cleaned DataFrame.
    """
    config = load_config(Path(config_path))
    paths = PreprocessPaths(
        raw_dir=Path("data/raw"),
        scraped_dir=Path("data/scraped"),
        processed_csv=Path("data/processed/reviews_clean.csv"),
        processed_sqlite=Path("data/processed/reviews.sqlite"),
        descriptive_json=Path("data/results/descriptive.json"),
        near_dup_csv=Path("data/logs/near_duplicates.csv"),
    )
    raw_df, raw_n = load_raw_reviews(paths)
    if raw_df.empty:
        logger.warning("No reviews found to preprocess.")
        raise ValueError("No reviews available after loading raw CSVs.")
    df = _ensure_schema(raw_df)
    df = df.dropna(subset=["review_text"])
    df["review_text"] = df["review_text"].astype(str)
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df = df.dropna(subset=["review_date"])
    df["review_date"] = df["review_date"].dt.strftime("%Y-%m-%d")
    df["reviewer_name"] = df["reviewer_name"].fillna("Unknown")
    df["verified_purchase"] = df["verified_purchase"].fillna(False).astype(str).str.lower().isin(
        {"1", "true", "yes"}
    )
    df["rating_norm"] = df["rating"].apply(normalize_rating)
    df["word_count"] = df["review_text"].apply(lambda txt: len(str(txt).split()))
    df["review_id"] = df.apply(_assign_review_id, axis=1)
    df["text_hash"] = df["review_text"].apply(_hash_text)
    df = df[~df["text_hash"].duplicated(keep="first")]
    df = df[~df["review_text"].apply(_is_spam)]
    df = df[df["review_text"].apply(_is_english)]
    collection_window = config.get("data", {}).get("collection_window", {})
    df = _window_filter(
        df,
        collection_window.get("start"),
        collection_window.get("end"),
    )
    df, near_dup_logs = remove_near_duplicates(df)
    _persist_near_dup_log(near_dup_logs, paths.near_dup_csv)
    df = df[OUTPUT_COLUMNS]
    df = df.sort_values("review_date").reset_index(drop=True)
    paths.processed_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.processed_csv, index=False)
    _persist_sqlite(df, paths.processed_sqlite)
    stats = _descriptive_stats(df, raw_n)
    _write_json(stats, paths.descriptive_json)
    return df


if __name__ == "__main__":
    preprocess()
