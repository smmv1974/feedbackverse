"""
Prepare Validation Data from Real Human Reviews

This script:
1. Loads real Yelp and Amazon reviews
2. Samples 400 reviews stratified by star rating (80 per star level)
3. Converts star ratings to sentiment scores (ground truth)
4. Creates train/test splits (320 training, 80 validation)
5. Generates data/validation/references_aligned.csv

Usage:
    python scripts/prepare_validation_data.py
    python scripts/prepare_validation_data.py --samples 400
    python scripts/prepare_validation_data.py --yelp-only
"""

import argparse
import gzip
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.star_conversion import stars_to_sentiment

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "data" / "raw" / "datasets"
VALIDATION_DIR = BASE_DIR / "data" / "validation"

YELP_REVIEW_FILE = DATASETS_DIR / "yelp" / "yelp_academic_dataset_review.json"
AMAZON_DIR = DATASETS_DIR / "amazon"


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)


def load_yelp_reviews(num_samples_per_star=60):
    """
    Load Yelp reviews, sample evenly across star ratings.

    Args:
        num_samples_per_star: Number of reviews to sample per star rating (1-5)

    Returns:
        List of review dicts with keys: review_id, platform, text, stars, sentiment
    """
    print("\n[STEP 1/4] Loading Yelp Reviews")
    print("-" * 80)

    if not YELP_REVIEW_FILE.exists():
        print(f"   ❌ Yelp dataset not found: {YELP_REVIEW_FILE}")
        return []

    print(f"   📁 Reading from: {YELP_REVIEW_FILE.name}")
    print(f"   ⏳ Sampling {num_samples_per_star} reviews per star level...")

    # Collect reviews by star rating
    reviews_by_star = {1: [], 2: [], 3: [], 4: [], 5: []}

    try:
        with open(YELP_REVIEW_FILE, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # Stop early if we have enough samples
                if all(len(reviews) >= num_samples_per_star * 2
                       for reviews in reviews_by_star.values()):
                    break

                # Progress indicator
                if (i + 1) % 100000 == 0:
                    print(f"   📊 Processed {i+1:,} reviews...")

                try:
                    review = json.loads(line)
                    stars = int(review.get('stars', 0))
                    text = review.get('text', '').strip()

                    # Filter: must have valid stars and text
                    if stars in reviews_by_star and text and len(text) >= 20:
                        reviews_by_star[stars].append({
                            'review_id': review.get('review_id', f"yelp_{i}"),
                            'platform': 'yelp',
                            'text': text,
                            'stars': stars
                        })

                except json.JSONDecodeError:
                    continue

        # Sample evenly from each star rating
        sampled_reviews = []
        for stars in range(1, 6):
            available = reviews_by_star[stars]
            sample_size = min(num_samples_per_star, len(available))

            if sample_size > 0:
                sampled = random.sample(available, sample_size)
                sampled_reviews.extend(sampled)
                print(f"   ✅ {stars}⭐: Sampled {sample_size} reviews (from {len(available):,} available)")
            else:
                print(f"   ⚠️  {stars}⭐: No reviews found")

        print(f"\n   ✅ Total Yelp reviews: {len(sampled_reviews)}")
        return sampled_reviews

    except Exception as e:
        print(f"   ❌ Error loading Yelp reviews: {e}")
        return []


def load_amazon_reviews(num_samples_per_star=20):
    """
    Load Amazon reviews from .jsonl.gz files, sample evenly across star ratings.

    Args:
        num_samples_per_star: Number of reviews to sample per star rating (1-5)

    Returns:
        List of review dicts with keys: review_id, platform, text, stars, sentiment
    """
    print("\n[STEP 2/4] Loading Amazon Reviews")
    print("-" * 80)

    if not AMAZON_DIR.exists():
        print(f"   ❌ Amazon dataset directory not found: {AMAZON_DIR}")
        return []

    # Find all .jsonl.gz files
    amazon_files = list(AMAZON_DIR.glob("*.jsonl.gz"))

    if not amazon_files:
        print(f"   ⚠️  No .jsonl.gz files found in {AMAZON_DIR}")
        print(f"   💡 Looking for extracted .jsonl files...")
        amazon_files = list(AMAZON_DIR.glob("*.jsonl"))

    if not amazon_files:
        print(f"   ❌ No Amazon review files found")
        return []

    print(f"   📁 Found {len(amazon_files)} file(s):")
    for f in amazon_files:
        print(f"      - {f.name}")

    # Collect reviews by star rating
    reviews_by_star = {1: [], 2: [], 3: [], 4: [], 5: []}

    for file_path in amazon_files:
        print(f"\n   ⏳ Reading {file_path.name}...")

        try:
            # Open gzip or regular file with context manager
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rt', encoding='utf-8') as file_obj:
                    for i, line in enumerate(file_obj):
                        # Stop early if we have enough samples
                        if all(len(reviews) >= num_samples_per_star * 2
                               for reviews in reviews_by_star.values()):
                            break

                        try:
                            review = json.loads(line)

                            # Amazon 2023 format: 'rating' field
                            # Amazon 2018 format: 'overall' field
                            rating = review.get('rating') or review.get('overall')
                            text = review.get('text') or review.get('reviewText') or ''
                            text = text.strip()

                            if rating is not None:
                                stars = int(float(rating))

                                # Filter: must have valid stars and text
                                if stars in reviews_by_star and text and len(text) >= 20:
                                    reviews_by_star[stars].append({
                                        'review_id': review.get('asin', f"amazon_{file_path.stem}_{i}"),
                                        'platform': 'amazon',
                                        'text': text,
                                        'stars': stars
                                    })

                        except (json.JSONDecodeError, ValueError, KeyError):
                            continue
            else:
                with open(file_path, 'r', encoding='utf-8') as file_obj:
                    for i, line in enumerate(file_obj):
                        # Stop early if we have enough samples
                        if all(len(reviews) >= num_samples_per_star * 2
                               for reviews in reviews_by_star.values()):
                            break

                        try:
                            review = json.loads(line)

                            # Amazon 2023 format: 'rating' field
                            # Amazon 2018 format: 'overall' field
                            rating = review.get('rating') or review.get('overall')
                            text = review.get('text') or review.get('reviewText') or ''
                            text = text.strip()

                            if rating is not None:
                                stars = int(float(rating))

                                # Filter: must have valid stars and text
                                if stars in reviews_by_star and text and len(text) >= 20:
                                    reviews_by_star[stars].append({
                                        'review_id': review.get('asin', f"amazon_{file_path.stem}_{i}"),
                                        'platform': 'amazon',
                                        'text': text,
                                        'stars': stars
                                    })

                        except (json.JSONDecodeError, ValueError, KeyError):
                            continue

        except Exception as e:
            print(f"   ⚠️  Error reading {file_path.name}: {e}")
            continue

    # Sample evenly from each star rating
    sampled_reviews = []
    for stars in range(1, 6):
        available = reviews_by_star[stars]
        sample_size = min(num_samples_per_star, len(available))

        if sample_size > 0:
            sampled = random.sample(available, sample_size)
            sampled_reviews.extend(sampled)
            print(f"   ✅ {stars}⭐: Sampled {sample_size} reviews (from {len(available):,} available)")
        else:
            print(f"   ⚠️  {stars}⭐: No reviews found")

    print(f"\n   ✅ Total Amazon reviews: {len(sampled_reviews)}")
    return sampled_reviews


def prepare_validation_set(reviews, output_file):
    """
    Convert reviews to validation format with sentiment scores.

    Args:
        reviews: List of review dicts
        output_file: Path to output CSV file

    Returns:
        DataFrame with validation data
    """
    print("\n[STEP 3/4] Converting Stars to Sentiment Scores")
    print("-" * 80)

    if not reviews:
        print("   ❌ No reviews to process")
        return None

    # Convert stars to sentiment scores
    for review in reviews:
        review['sentiment'] = stars_to_sentiment(review['stars'], add_noise=True)
        # Create a simple reference summary (can be enhanced later)
        review['reference_summary'] = f"{review['stars']}-star {review['platform']} review"

    # Create DataFrame
    df = pd.DataFrame(reviews)

    # Display conversion statistics
    print(f"   📊 Conversion Statistics:")
    print(f"      Total reviews: {len(df)}")
    print(f"\n   Star Distribution:")
    for stars in range(1, 6):
        count = len(df[df['stars'] == stars])
        avg_sentiment = df[df['stars'] == stars]['sentiment'].mean() if count > 0 else 0
        print(f"      {stars}⭐: {count:3d} reviews → avg sentiment: {avg_sentiment:+.2f}")

    # Save to CSV
    print("\n[STEP 4/4] Saving Validation Data")
    print("-" * 80)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"   ✅ Saved to: {output_file}")
    print(f"   📊 Total samples: {len(df)}")
    print(f"   📁 File size: {output_file.stat().st_size / 1024:.1f} KB")

    return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare validation data from real human reviews"
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=400,
        help='Total number of samples to generate (default: 400)'
    )
    parser.add_argument(
        '--yelp-only',
        action='store_true',
        help='Use only Yelp reviews (no Amazon)'
    )

    args = parser.parse_args()

    print_header("VALIDATION DATA PREPARATION")

    print("\n📋 Configuration:")
    print(f"   Total samples: {args.samples}")
    print(f"   Distribution: Stratified by star rating (1-5)")

    # Set random seed for reproducibility
    random.seed(42)

    # Calculate samples per star level
    # For 400 total: 80 per star level
    # Split: 75% Yelp (60 per star) + 25% Amazon (20 per star)
    samples_per_star = args.samples // 5

    if args.yelp_only:
        yelp_per_star = samples_per_star
        amazon_per_star = 0
        print(f"   Yelp: {yelp_per_star} per star (total: {yelp_per_star * 5})")
    else:
        yelp_per_star = int(samples_per_star * 0.75)
        amazon_per_star = samples_per_star - yelp_per_star
        print(f"   Yelp: {yelp_per_star} per star (total: {yelp_per_star * 5})")
        print(f"   Amazon: {amazon_per_star} per star (total: {amazon_per_star * 5})")

    # Load reviews
    all_reviews = []

    # Load Yelp
    yelp_reviews = load_yelp_reviews(num_samples_per_star=yelp_per_star)
    all_reviews.extend(yelp_reviews)

    # Load Amazon
    if not args.yelp_only:
        amazon_reviews = load_amazon_reviews(num_samples_per_star=amazon_per_star)
        all_reviews.extend(amazon_reviews)

    # Prepare validation set
    if not all_reviews:
        print("\n❌ No reviews loaded. Cannot create validation set.")
        print("\nTroubleshooting:")
        print("   1. Check Yelp dataset exists: data/raw/datasets/yelp/yelp_academic_dataset_review.json")
        print("   2. Check Amazon datasets exist: data/raw/datasets/amazon/*.jsonl.gz")
        print("   3. Run: python scripts/download_datasets.py --verify")
        return

    # Shuffle reviews
    random.shuffle(all_reviews)

    # Create validation set
    output_file = VALIDATION_DIR / "references_aligned.csv"
    df = prepare_validation_set(all_reviews, output_file)

    if df is not None:
        print_header("VALIDATION DATA READY")

        print("\n✅ Success! Validation data created from real human reviews.")
        print(f"\n📊 Final Statistics:")
        print(f"   Total reviews: {len(df)}")
        print(f"   Yelp reviews: {len(df[df['platform'] == 'yelp'])}")
        print(f"   Amazon reviews: {len(df[df['platform'] == 'amazon'])}")
        print(f"\n   Sentiment range: {df['sentiment'].min():.2f} to {df['sentiment'].max():.2f}")
        print(f"   Average sentiment: {df['sentiment'].mean():.2f}")

        print(f"\n📁 Output file: {output_file}")
        print(f"\n🎯 Next Step:")
        print(f"   Run the pipeline with real data:")
        print(f"   python run_pipeline.py --config config/config_alt.yaml")


if __name__ == "__main__":
    main()
