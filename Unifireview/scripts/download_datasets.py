"""
Dataset Download Script - Yelp Academic & Amazon Reviews

This script downloads and prepares real human review datasets for the pipeline.

Datasets:
1. Yelp Academic Dataset (6.9M reviews, 15GB)
   - Source: https://www.yelp.com/dataset
   - Format: JSON lines
   - Ground truth: star ratings (1-5)

2. Amazon Reviews (Electronics category)
   - Source: https://jmcauley.ucsd.edu/data/amazon_v2/
   - Format: JSON lines
   - Ground truth: overall ratings (1-5)

Usage:
    # Download both datasets:
    python scripts/download_datasets.py

    # Download specific dataset:
    python scripts/download_datasets.py --dataset yelp
    python scripts/download_datasets.py --dataset amazon

    # Verify downloads:
    python scripts/download_datasets.py --verify

Output:
    - data/raw/datasets/yelp/review.json (5.3 GB)
    - data/raw/datasets/amazon/Electronics_5.json (137 MB)
"""

import argparse
import io
import json
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (AttributeError, io.UnsupportedOperation):
        pass

# Configuration
YELP_DATASET_URL = "https://www.yelp.com/dataset/download/yelp_dataset.tar"
# Amazon 2023 dataset (updated URL)
AMAZON_ELECTRONICS_URL = "https://amazon-reviews-2023.github.io/main/Electronics.jsonl.gz"
# Fallback: Amazon 2018 dataset if 2023 unavailable
AMAZON_ELECTRONICS_URL_FALLBACK = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2018/categoryFilesSmall/Electronics_5.json.gz"

# Paths
BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "data" / "raw" / "datasets"
YELP_DIR = DATASETS_DIR / "yelp"
AMAZON_DIR = DATASETS_DIR / "amazon"


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)


def print_step(num, total, description):
    """Print step indicator."""
    print(f"\n[STEP {num}/{total}] {description}")
    print("-" * 80)


def check_disk_space(required_gb):
    """Check if enough disk space is available."""
    import shutil
    stat = shutil.disk_usage(str(DATASETS_DIR.parent))
    available_gb = stat.free / (1024 ** 3)

    print(f"📊 Disk space check:")
    print(f"   Required: {required_gb} GB")
    print(f"   Available: {available_gb:.1f} GB")

    if available_gb < required_gb:
        print(f"❌ Not enough disk space! Need {required_gb} GB, have {available_gb:.1f} GB")
        return False
    print(f"✅ Sufficient disk space")
    return True


def download_with_progress(url, filepath, description="Downloading"):
    """Download file with progress bar."""
    import urllib.request

    print(f"\n{description}:")
    print(f"   URL: {url}")
    print(f"   Destination: {filepath}")

    def reporthook(count, block_size, total_size):
        if count == 0:
            return
        percent = min(100, count * block_size * 100 / total_size)
        downloaded_mb = count * block_size / (1024 ** 2)
        total_mb = total_size / (1024 ** 2)
        bar = "█" * int(percent // 2) + "░" * (50 - int(percent // 2))
        print(f"\r   [{bar}] {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end='')

    try:
        urllib.request.urlretrieve(url, filepath, reporthook=reporthook)
        print("\n✅ Download complete!")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False


def download_yelp_dataset():
    """Download Yelp Academic Dataset."""
    print_header("YELP ACADEMIC DATASET")

    # Check disk space (15 GB for compressed + extracted)
    if not check_disk_space(20):
        return False

    # Create directory
    YELP_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    review_file = YELP_DIR / "yelp_academic_dataset_review.json"
    if review_file.exists():
        print("\n✅ Yelp dataset already exists!")
        print(f"   Location: {review_file}")
        print(f"   Size: {review_file.stat().st_size / (1024**3):.2f} GB")
        return True

    print("\n📥 Yelp Academic Dataset Download")
    print("\nIMPORTANT: The Yelp dataset requires academic access.")
    print("Please follow these steps:")
    print("\n1. Visit: https://www.yelp.com/dataset")
    print("2. Click 'Get the Data'")
    print("3. Sign up with .edu email (instant approval) or business email")
    print("4. Download yelp_dataset.tar (15 GB)")
    print("5. Place it in: data/raw/datasets/")
    print("\nAlternative: Download via Kaggle")
    print("1. Visit: https://www.kaggle.com/yelp-dataset/yelp-dataset")
    print("2. Download dataset")
    print("3. Extract to: data/raw/datasets/yelp/")

    # Check if user already downloaded it
    tar_file = DATASETS_DIR / "yelp_dataset.tar"
    if tar_file.exists():
        print(f"\n✅ Found Yelp archive: {tar_file}")
        print("   Extracting...")

        try:
            with tarfile.open(tar_file, 'r') as tar:
                tar.extractall(path=YELP_DIR)
            print("✅ Extraction complete!")
            return True
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False

    print("\n⏳ Waiting for manual download...")
    print(f"   Please place yelp_dataset.tar in: {DATASETS_DIR}")
    print("   Then run this script again.")
    return False


def download_amazon_dataset():
    """Download Amazon Reviews (Electronics category)."""
    print_header("AMAZON REVIEWS DATASET (Electronics)")

    # Check disk space (200 MB for compressed + extracted)
    if not check_disk_space(1):
        return False

    # Create directory
    AMAZON_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    review_file = AMAZON_DIR / "Electronics_5.json"
    if review_file.exists():
        print("\n✅ Amazon dataset already exists!")
        print(f"   Location: {review_file}")
        print(f"   Size: {review_file.stat().st_size / (1024**2):.2f} MB")
        return True

    # Download
    compressed_file = AMAZON_DIR / "Electronics_5.json.gz"

    if not compressed_file.exists():
        success = download_with_progress(
            AMAZON_ELECTRONICS_URL,
            compressed_file,
            "Downloading Amazon Electronics reviews"
        )
        if not success:
            print("\n❌ Amazon download failed")
            print("   Manual download:")
            print(f"   1. Visit: {AMAZON_ELECTRONICS_URL}")
            print(f"   2. Save to: {compressed_file}")
            print("   3. Run script again")
            return False

    # Extract
    print("\n📦 Extracting...")
    try:
        import gzip
        import shutil
        with gzip.open(compressed_file, 'rb') as f_in:
            with open(review_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("✅ Extraction complete!")

        # Remove compressed file
        compressed_file.unlink()
        print("✅ Removed compressed file")

        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


def verify_datasets():
    """Verify downloaded datasets."""
    print_header("DATASET VERIFICATION")

    results = {}

    # Check Yelp
    print_step(1, 2, "Verifying Yelp Dataset")
    yelp_review = YELP_DIR / "yelp_academic_dataset_review.json"

    if yelp_review.exists():
        try:
            # Read first line to verify format
            with open(yelp_review, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                sample = json.loads(first_line)

            # Count lines
            print("   Counting reviews (may take 30 seconds)...")
            with open(yelp_review, 'r', encoding='utf-8') as f:
                count = sum(1 for _ in f)

            print(f"   ✅ Yelp Dataset Valid")
            print(f"      - Total reviews: {count:,}")
            print(f"      - File size: {yelp_review.stat().st_size / (1024**3):.2f} GB")
            print(f"      - Sample: {sample.get('stars')}⭐ review")
            results['yelp'] = True
        except Exception as e:
            print(f"   ❌ Yelp verification failed: {e}")
            results['yelp'] = False
    else:
        print(f"   ❌ Yelp dataset not found")
        print(f"      Expected: {yelp_review}")
        results['yelp'] = False

    # Check Amazon
    print_step(2, 2, "Verifying Amazon Dataset")

    # Look for any Amazon files (.jsonl.gz or .jsonl)
    amazon_files = []
    if AMAZON_DIR.exists():
        amazon_files = list(AMAZON_DIR.glob("*.jsonl.gz")) + list(AMAZON_DIR.glob("*.jsonl"))

    if amazon_files:
        try:
            # Verify first file found
            amazon_file = amazon_files[0]
            print(f"   📁 Found: {amazon_file.name}")

            # Read first line (handle gzip if needed)
            if amazon_file.suffix == '.gz':
                import gzip
                with gzip.open(amazon_file, 'rt', encoding='utf-8') as f:
                    first_line = f.readline()
                    sample = json.loads(first_line)

                # Count lines in gzip
                with gzip.open(amazon_file, 'rt', encoding='utf-8') as f:
                    count = sum(1 for _ in f)
            else:
                with open(amazon_file, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    sample = json.loads(first_line)

                # Count lines
                with open(amazon_file, 'r', encoding='utf-8') as f:
                    count = sum(1 for _ in f)

            rating = sample.get('rating') or sample.get('overall') or 'Unknown'

            print(f"   ✅ Amazon Dataset Valid")
            print(f"      - Total reviews: {count:,}")
            print(f"      - File size: {amazon_file.stat().st_size / (1024**2):.2f} MB")
            print(f"      - Sample: {rating}⭐ review")
            if len(amazon_files) > 1:
                print(f"      - Additional files found: {len(amazon_files)-1}")
            results['amazon'] = True
        except Exception as e:
            print(f"   ❌ Amazon verification failed: {e}")
            results['amazon'] = False
    else:
        print(f"   ❌ Amazon dataset not found")
        print(f"      Expected: .jsonl.gz or .jsonl files in {AMAZON_DIR}")
        results['amazon'] = False

    # Summary
    print("\n" + "=" * 80)
    print(" VERIFICATION SUMMARY")
    print("=" * 80)

    for dataset, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {dataset.capitalize()}: {'Valid' if status else 'Missing/Invalid'}")

    all_valid = all(results.values())
    print(f"\n{'✅ All datasets ready!' if all_valid else '❌ Some datasets missing/invalid'}")

    if all_valid:
        print("\n📋 NEXT STEPS:")
        print("   1. Run: python scripts/prepare_validation_data.py")
        print("   2. Then: python run_pipeline.py --config config/config_cloud.yaml")

    return all_valid


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download Yelp and Amazon review datasets")
    parser.add_argument(
        '--dataset',
        choices=['yelp', 'amazon', 'both'],
        default='both',
        help='Which dataset to download (default: both)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify existing downloads'
    )

    args = parser.parse_args()

    print_header("DATASET DOWNLOAD SCRIPT")
    print("\nThis script downloads real human review datasets:")
    print("   • Yelp Academic Dataset: 6.9M reviews, 15 GB")
    print("   • Amazon Reviews (Electronics): 259k reviews, 137 MB")
    print("\nThese provide star ratings (1-5) as ground truth sentiment.")

    # Create base directories
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify:
        verify_datasets()
        return

    # Download datasets
    success_yelp = True
    success_amazon = True

    if args.dataset in ['yelp', 'both']:
        success_yelp = download_yelp_dataset()

    if args.dataset in ['amazon', 'both']:
        success_amazon = download_amazon_dataset()

    # Verify
    if success_yelp and success_amazon:
        time.sleep(1)
        verify_datasets()
    else:
        print("\n" + "=" * 80)
        print(" DOWNLOAD INCOMPLETE")
        print("=" * 80)
        print("\nSome downloads failed or require manual action.")
        print("Run with --verify to check status.")


if __name__ == "__main__":
    main()
