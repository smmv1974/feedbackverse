#!/usr/bin/env python
# quick_test_analysis.py
"""
Quick test script for analysis with better error handling
Tests with just 2 small batches
"""

import pandas as pd
import json
import logging
from pathlib import Path
import sys

sys.path.append('src')

from llm_analysis.ollama_analyzer import OllamaAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run quick analysis test"""
    
    # Load test data (just first 10 reviews)
    logger.info("Loading test data...")
    df = pd.read_csv("data/processed/reviews_clean.csv")
    df = df.head(10)  # Just 10 reviews for quick test
    logger.info(f"Loaded {len(df)} reviews for testing")
    
    # Initialize analyzer
    logger.info("Initializing OllamaAnalyzer...")
    analyzer = OllamaAnalyzer(config_path="config/config.yaml")
    
    # Test connection
    if not analyzer.test_connection(test_all_models=True):
        logger.error("Cannot connect to all Ollama models. Exiting.")
        return
    
    all_results = []
    
    # Process in one batch
    logger.info(f"\nAnalyzing {len(df)} reviews...")
    reviews = df['review_text'].tolist()
    product_name = df['product_name'].iloc[0] if 'product_name' in df.columns else "Sample Product"
    
    try:
        result = analyzer.analyze_batch(reviews, product_name)
        
        result['batch_id'] = "test_batch_1"
        result['platform'] = df['platform'].iloc[0]
        result['product_name'] = product_name
        result['review_ids'] = df.index.tolist()
        
        all_results.append(result)
        
        logger.info(f"✓ Analysis complete!")
        logger.info(f"  Sentiment score: {result['sentiment']['sentiment_score']:.2f}")
        logger.info(f"  Themes extracted: {len(result['themes'])}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save results
    output_file = "data/results/analysis_results.json"
    Path("data/results").mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✓ Saved results to {output_file}")
    logger.info("\n✓ Quick test complete! Ready for Streamlit dashboard.")
    
    # Print result summary
    print("\n" + "="*80)
    print("ANALYSIS RESULT")
    print("="*80)
    print(f"\nSummary:\n{result['summary'][:200]}...")
    print(f"\nSentiment: {result['sentiment']['sentiment_score']:.2f}")
    print(f"\nTop 3 Themes:")
    for i, theme in enumerate(result['themes'][:3], 1):
        print(f"  {i}. {theme['theme']} ({theme['sentiment']})")


if __name__ == "__main__":
    main()
