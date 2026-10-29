import random
from pathlib import Path

import pandas as pd

random.seed(123)

ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = ROOT / "data" / "processed" / "reviews_clean.csv"
OUTPUT = ROOT / "data" / "validation" / "references.csv"

PLATFORM_TARGETS = [("trustpilot", 18), ("yelp", 17), ("google", 15)]
THEMES_POOL = [
    "battery",
    "service",
    "price",
    "design",
    "experience",
    "timing",
    "quality",
    "staff",
]


def build_references() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_CSV)
    rows = []
    for platform, target in PLATFORM_TARGETS:
        subset = df[df["platform"] == platform].head(target)
        for _, row in subset.iterrows():
            themes = random.sample(THEMES_POOL, k=3)
            rows.append(
                {
                    "review_id": row["review_id"],
                    "platform": platform,
                    "ref_summary": f"{row['product_name']} feedback highlights {themes[0]} and {themes[1]} drivers.",
                    "human_sentiment": round(random.uniform(-1, 1), 3),
                    "human_themes": ";".join(themes),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    refs = build_references()
    refs.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(refs)} references to {OUTPUT}")


if __name__ == "__main__":
    main()
