"""
Comprehensive validation script with human-labeled reviews.
Tests: 10, 50, or 300 reviews
Calculates: Kappa, Pearson, MAE
Compares to documented targets
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path

# Add parent directory to Python path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

# Set encoding (API key should be set in environment before running)
os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.llm_analysis.ollama_cloud_analyzer import OllamaCloudAnalyzer


def bin_sentiment(s):
    """Bin sentiment scores into 5 categories for Kappa calculation."""
    if s < -0.6: return 0  # Very negative
    elif s < -0.2: return 1  # Negative
    elif s < 0.2: return 2  # Neutral
    elif s < 0.6: return 3  # Positive
    else: return 4  # Very positive


def run_validation(n_reviews=50, config='config/config_ollama_cloud.yaml'):
    """Run validation on n_reviews from human-labeled dataset."""

    print("\n" + "="*80)
    print(f" VALIDATION RUN - {n_reviews} REVIEWS")
    print("="*80)
    print(f"Config: {config}")
    print(f"API Key: {'[SET]' if os.getenv('OLLAMA_API_KEY') else '[MISSING]'}")
    print()

    # Load validation data
    print("[INFO] Loading validation data...")
    val_df = pd.read_csv('data/validation/references_aligned.csv')
    print(f"[OK] Loaded {len(val_df)} total validation reviews")

    # Sample reviews
    test_df = val_df.head(n_reviews)
    print(f"[INFO] Testing with {len(test_df)} reviews")
    print()

    # Initialize analyzer
    print("[INFO] Initializing Ollama Cloud analyzer...")
    analyzer = OllamaCloudAnalyzer(config)
    print(f"[OK] Models:")
    for task, model in analyzer.models.items():
        print(f"   - {task}: {model}")
    print()

    # Process reviews
    print("[INFO] Processing reviews...")
    print("="*80)

    results = []
    start_time = time.time()

    for idx, row in test_df.iterrows():
        review_start = time.time()

        # Prepare review data
        review_data = {
            'review_id': row['review_id'],
            'review_text': row['text'],
            'text': row['text'],
            'rating_norm': (row['stars'] - 1) / 4.0 * 2 - 1,  # 1-5 stars → -1 to +1
            'platform': row.get('platform', 'validation'),
            'review_date': row.get('review_date', '2024-01-01')
        }

        try:
            # Analyze
            print(f"   [{idx+1:3d}/{len(test_df)}] Processing review {row['review_id'][:15]}...", end=' ')
            batch_result = analyzer.analyze_batch([review_data], batch_index=idx)
            sentiment = batch_result['sentiments'][0]['sentiment']

            review_time = time.time() - review_start

            # Store result
            results.append({
                'review_id': row['review_id'],
                'predicted': sentiment,
                'actual': row['sentiment'],
                'rating': row['stars'],
                'text_preview': row['text'][:80] + "..."
            })

            # Visual feedback
            diff = abs(sentiment - row['sentiment'])
            match = "[OK] GOOD" if diff < 0.3 else "[~] OK" if diff < 0.5 else "[X] OFF"
            print(f"{match} (pred:{sentiment:+.2f} | actual:{row['sentiment']:+.2f} | diff:{diff:.2f}) [{review_time:.1f}s]")

        except Exception as e:
            print(f"[ERROR] {e}")
            results.append({
                'review_id': row['review_id'],
                'predicted': 0.0,
                'actual': row['sentiment'],
                'rating': row['stars'],
                'text_preview': row['text'][:80] + "..."
            })

        # Progress update every 10 reviews
        if (idx + 1) % 10 == 0:
            elapsed = (time.time() - start_time) / 60
            avg_time = elapsed / (idx + 1)
            remaining = avg_time * (len(test_df) - (idx + 1))
            print(f"\n   [PROGRESS] {idx+1}/{len(test_df)} | Elapsed: {elapsed:.1f}m | ETA: {remaining:.1f}m\n")

    total_time = time.time() - start_time

    print()
    print("="*80)
    print(" CALCULATING METRICS")
    print("="*80)
    print()

    # Extract predictions and actuals
    pred = [r['predicted'] for r in results]
    actual = [r['actual'] for r in results]

    # Bin for Kappa
    pred_bins = [bin_sentiment(s) for s in pred]
    actual_bins = [bin_sentiment(s) for s in actual]

    # Calculate metrics
    mae = mean_absolute_error(actual, pred)
    pearson_r, pearson_p = pearsonr(pred, actual)
    kappa = cohen_kappa_score(actual_bins, pred_bins)

    print(f"[TIME] Processing Time: {total_time/60:.1f} minutes ({total_time/len(results):.1f}s per review)")
    print()

    print("[METRICS] VALIDATION METRICS:")
    print(f"   Cohen's Kappa:       {kappa:.3f}  (target: 0.70-0.80)")
    kappa_status = "[OK] MEETS TARGET" if kappa >= 0.70 else "[~] BELOW TARGET" if kappa >= 0.50 else "[X] NEEDS IMPROVEMENT"
    print(f"                        {kappa_status}")
    print()

    print(f"   Pearson Correlation: {pearson_r:.3f}  (target: 0.75-0.88)")
    pearson_status = "[OK] MEETS TARGET" if pearson_r >= 0.75 else "[~] BELOW TARGET" if pearson_r >= 0.60 else "[X] NEEDS IMPROVEMENT"
    print(f"                        {pearson_status}")
    print()

    print(f"   Mean Absolute Error: {mae:.3f}  (target: 0.25-0.35)")
    mae_status = "[OK] MEETS TARGET" if mae <= 0.35 else "[~] SLIGHTLY HIGH" if mae <= 0.45 else "[X] TOO HIGH"
    print(f"                        {mae_status}")
    print()

    # Compare to baseline
    baseline_file = Path("data/results/validation_metrics.json")
    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline = json.load(f)

        print("[COMPARISON] COMPARISON TO BASELINE:")
        print(f"   Kappa:   {kappa:.3f} vs {baseline.get('kappa_binned', 0):.3f} (baseline)")
        kappa_improvement = ((kappa - baseline.get('kappa_binned', 0)) / abs(baseline.get('kappa_binned', 0.01))) * 100 if baseline.get('kappa_binned', 0) != 0 else 0
        print(f"            {kappa_improvement:+.0f}% improvement")
        print()

        print(f"   Pearson: {pearson_r:.3f} vs {baseline.get('pearson_r', 0):.3f} (baseline)")
        pearson_improvement = pearson_r - baseline.get('pearson_r', 0)
        print(f"            {pearson_improvement:+.3f} points better")
        print()

        print(f"   MAE:     {mae:.3f} vs {baseline.get('mae', 1):.3f} (baseline)")
        mae_improvement = ((baseline.get('mae', 1) - mae) / baseline.get('mae', 1)) * 100
        print(f"            {mae_improvement:+.0f}% reduction (lower is better)")
        print()

    # Sample predictions
    print("[SAMPLES] SAMPLE PREDICTIONS (First 10):")
    for i, r in enumerate(results[:10], 1):
        diff = abs(r['predicted'] - r['actual'])
        match = "[OK]" if diff < 0.3 else "[~]" if diff < 0.5 else "[X]"
        print(f"{i:2d}. {match} Pred: {r['predicted']:+.2f} | Actual: {r['actual']:+.2f} | Diff: {diff:.2f} | Rating: {r['rating']} stars")
        if i <= 5:  # Show text preview for first 5
            print(f"    \"{r['text_preview']}\"")
    print()

    # Save results
    output_file = f"data/results/validation_{n_reviews}_reviews.json"
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_reviews': len(results),
        'time_minutes': total_time / 60,
        'model': 'ollama-cloud-120b',
        'config': config,
        'metrics': {
            'cohen_kappa': float(kappa),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'mae': float(mae)
        },
        'predictions': results
    }

    Path("data/results").mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[SAVED] Results saved: {output_file}")
    print()

    # Final verdict
    print("="*80)
    if kappa >= 0.70 and pearson_r >= 0.75 and mae <= 0.35:
        print(" [SUCCESS] All metrics meet or exceed targets!")
    elif kappa >= 0.60 and pearson_r >= 0.70 and mae <= 0.40:
        print(" [VERY GOOD] Close to targets, strong performance!")
    elif kappa >= 0.50 and pearson_r >= 0.60:
        print(" [PARTIAL SUCCESS] Good correlation, approaching targets")
    else:
        print(" [NEEDS IMPROVEMENT] Below targets")
    print("="*80)
    print()

    return {
        'kappa': kappa,
        'pearson': pearson_r,
        'mae': mae,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description="Validate sentiment analysis with human-labeled reviews")
    parser.add_argument('--reviews', type=int, default=50, help='Number of reviews to test (10, 50, or 300)')
    parser.add_argument('--config', default='config/config_ollama_cloud.yaml', help='Config file')
    args = parser.parse_args()

    # Validate review count
    if args.reviews not in [10, 50, 300]:
        print(f"[WARNING] {args.reviews} reviews requested. Recommended: 10 (quick), 50 (standard), or 300 (full)")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Run validation
    run_validation(n_reviews=args.reviews, config=args.config)


if __name__ == "__main__":
    main()
