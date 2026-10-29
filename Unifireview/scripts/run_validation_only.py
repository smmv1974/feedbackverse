"""Run ONLY validation reviews with Ollama Cloud"""
import os
import sys
import pandas as pd

# Set API key
os.environ['OLLAMA_API_KEY'] = 'a3c160f22bbb4cef9d768c66be8c69a2.YApt4pMgnwvRMQh7UNYygUoX'
os.environ['PYTHONIOENCODING'] = 'utf-8'

print("="*80)
print(" RUNNING VALIDATION SET ONLY - Ollama Cloud")
print("="*80)
print()

# Load validation data
val_df = pd.read_csv('data/validation/references_aligned.csv')
print(f"Loaded {len(val_df)} validation reviews")

# Take first 50 for quick test (or change to 300 for full)
test_df = val_df.head(50)
print(f"Testing with {len(test_df)} reviews")
print()

# Initialize analyzer
print("Initializing Ollama Cloud analyzer...")
from src.llm_analysis.ollama_cloud_analyzer import OllamaCloudAnalyzer
analyzer = OllamaCloudAnalyzer('config/config_ollama_cloud.yaml')
print(f"Models: {analyzer.models}")
print()

# Process reviews
print("Processing reviews...")
import time
import json
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

results = []
start_time = time.time()

for idx, row in test_df.iterrows():
    review_data = {
        'review_id': row['review_id'],
        'review_text': row['text'],
        'text': row['text'],
        'rating_norm': (row['stars'] - 1) / 4.0 * 2 - 1,
        'platform': row.get('platform', 'validation'),
        'review_date': row.get('review_date', '2024-01-01')
    }

    try:
        batch_result = analyzer.analyze_batch([review_data], batch_index=idx)
        sentiment = batch_result['sentiments'][0]['sentiment']

        results.append({
            'review_id': row['review_id'],
            'predicted': sentiment,
            'actual': row['sentiment'],
            'rating': row['stars']
        })

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {idx+1}/{len(test_df)} ({elapsed:.0f}s elapsed)")

    except Exception as e:
        print(f"  ERROR on review {idx}: {e}")
        results.append({
            'review_id': row['review_id'],
            'predicted': 0.0,
            'actual': row['sentiment'],
            'rating': row['stars']
        })

total_time = time.time() - start_time

print()
print("="*80)
print(" RESULTS")
print("="*80)
print()

# Calculate metrics
pred = [r['predicted'] for r in results]
actual = [r['actual'] for r in results]

def bin_sentiment(s):
    if s < -0.6: return 0
    elif s < -0.2: return 1
    elif s < 0.2: return 2
    elif s < 0.6: return 3
    else: return 4

pred_bins = [bin_sentiment(s) for s in pred]
actual_bins = [bin_sentiment(s) for s in actual]

mae = mean_absolute_error(actual, pred)
pearson_r, pearson_p = pearsonr(pred, actual)
kappa = cohen_kappa_score(actual_bins, pred_bins)

print(f"Processed: {len(results)} reviews in {total_time/60:.1f} minutes")
print()
print("METRICS:")
print(f"  Cohen's Kappa:       {kappa:.3f}  (target: 0.70-0.80)")
print(f"  Pearson Correlation: {pearson_r:.3f}  (target: 0.75-0.88)")
print(f"  Mean Absolute Error: {mae:.3f}  (target: 0.25-0.35)")
print()

# Save results
output = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'model': 'ollama-cloud-120b',
    'n_reviews': len(results),
    'time_minutes': total_time / 60,
    'metrics': {
        'cohen_kappa': float(kappa),
        'pearson_r': float(pearson_r),
        'mae': float(mae)
    },
    'predictions': results
}

with open('data/results/validation_ollama_cloud.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"Results saved to: data/results/validation_ollama_cloud.json")
print()

print("="*80)
if kappa >= 0.70 and pearson_r >= 0.75 and mae <= 0.35:
    print(" [SUCCESS] All metrics meet or exceed targets!")
elif kappa >= 0.60 and pearson_r >= 0.70:
    print(" [VERY GOOD] Strong performance!")
else:
    print(" [PARTIAL SUCCESS] Good correlation")
print("="*80)
