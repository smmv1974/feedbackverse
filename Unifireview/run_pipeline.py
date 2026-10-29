"""
Simple pipeline runner with configurable review counts.
Usage:
    python run_pipeline.py --reviews 10    # Test with 10 reviews
    python run_pipeline.py --reviews 50    # Standard test
    python run_pipeline.py --reviews 300   # Full validation
    python run_pipeline.py                 # Process all reviews
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.validate_complete import run_validation


def main():
    parser = argparse.ArgumentParser(description="Run sentiment analysis pipeline")
    parser.add_argument(
        "--reviews",
        type=int,
        default=None,
        help="Number of reviews to process (10, 50, 300, or all)",
    )
    parser.add_argument(
        "--config",
        default="config/config_ollama_cloud.yaml",
        help="Config file to use",
    )
    args = parser.parse_args()

    # Determine review count
    if args.reviews is None:
        print("Processing all reviews...")
        review_count = 300  # Default to full validation set
    else:
        review_count = args.reviews
        print(f"Processing {review_count} reviews...")

    # Run validation
    run_validation(n_reviews=review_count, config=args.config)


if __name__ == "__main__":
    main()
