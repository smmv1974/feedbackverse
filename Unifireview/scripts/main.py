# main.py
"""
Main Pipeline for Multi-Platform Review Analysis
Complete end-to-end execution optimized for 24-hour timeline

Usage:
    python main.py --mode all  # Run full pipeline
    python main.py --mode collect  # Only collect data
    python main.py --mode analyze  # Only analyze existing data
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create directories
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("results/figures").mkdir(parents=True, exist_ok=True)
Path("results/tables").mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(exist_ok=True)


class ReviewAnalysisPipeline:
    """Complete pipeline for review collection and analysis"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize pipeline with configuration"""
        self.config = self.load_config(config_path)
        self.all_reviews = []
    
    def load_config(self, path: str) -> dict:
        """Load configuration file"""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found at {path}, using defaults")
            return self._default_config()
    
    def _default_config(self) -> dict:
        """Default configuration"""
        return {
            'data_collection': {
                'limits': {'reviews_per_platform': 60},
                'rate_limiting': {'delay_between_platforms': 10}
            },
            'llm': {
                'models': {
                    'summarization': 'llama3.2:3b-instruct-q4_0',
                    'sentiment': 'llama3.2:3b-instruct-q4_0',
                    'themes': 'llama3.2:3b-instruct-q4_0'
                },
                'generation_params': {
                    'temperature': 0.3,
                    'max_tokens': 500
                },
                'batch_processing': {'batch_size': 50}
            }
        }
    
    # ============ PHASE 1: DATA COLLECTION ============
    
    def collect_all_reviews(self):
        """Collect reviews from all platforms"""
        logger.info("=" * 80)
        logger.info("PHASE 1: DATA COLLECTION")
        logger.info("=" * 80)
        
        limit = self.config['data_collection']['limits']['reviews_per_platform']
        
        # 1. Trustpilot
        logger.info("\n[1/3] Collecting from Trustpilot...")
        trustpilot_reviews = self._collect_trustpilot(limit)
        logger.info(f"✓ Collected {len(trustpilot_reviews)} Trustpilot reviews")
        
        import time
        time.sleep(self.config['data_collection']['rate_limiting']['delay_between_platforms'])
        
        # 2. Yelp
        logger.info("\n[2/3] Collecting from Yelp...")
        yelp_reviews = self._collect_yelp(limit)
        logger.info(f"✓ Collected {len(yelp_reviews)} Yelp reviews")
        
        time.sleep(self.config['data_collection']['rate_limiting']['delay_between_platforms'])
        
        # 3. Google Reviews
        logger.info("\n[3/3] Collecting from Google Reviews...")
        google_reviews = self._collect_google(limit)
        logger.info(f"✓ Collected {len(google_reviews)} Google reviews")
        
        # Combine all
        self.all_reviews = trustpilot_reviews + yelp_reviews + google_reviews
        logger.info(f"\n✓ TOTAL COLLECTED: {len(self.all_reviews)} reviews across 3 platforms")
        
        # Save raw data
        df = pd.DataFrame(self.all_reviews)
        df.to_csv("data/raw/all_reviews_raw.csv", index=False)
        logger.info(f"✓ Saved raw data to: data/raw/all_reviews_raw.csv")
        
        return self.all_reviews
    
    def _collect_trustpilot(self, limit: int) -> list:
        """Collect from Trustpilot"""
        from src.data_collection.trustpilot_collector import TrustpilotCollector
        
        collector = TrustpilotCollector(rate_limit=2.0)
        reviews = collector.collect_reviews(
            company_url="www.trustpilot.com/review/www.apple.com",
            limit=limit
        )
        collector.save_to_csv(reviews, "data/raw/trustpilot_raw.csv")
        return reviews
    
    def _collect_yelp(self, limit: int) -> list:
        """Collect from Yelp"""
        from src.data_collection.yelp_collector import YelpCollector
        
        collector = YelpCollector()
        reviews = collector.collect_reviews_scraping(
            business_url="https://www.yelp.com/biz/apple-fifth-avenue-new-york",
            limit=limit
        )
        collector.save_to_csv(reviews, "data/raw/yelp_raw.csv")
        return reviews
    
    def _collect_google(self, limit: int) -> list:
        """Collect from Google Reviews"""
        from src.data_collection.google_collector import GoogleReviewsCollector
        
        collector = GoogleReviewsCollector()
        reviews = collector.collect_reviews_manual("Apple Store", "New York")
        collector.save_to_csv(reviews, "data/raw/google_raw.csv")
        return reviews[:limit]
    
    # ============ PHASE 2: PREPROCESSING ============
    
    def preprocess_reviews(self, input_file: str = None):
        """Clean and standardize reviews"""
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: PREPROCESSING")
        logger.info("=" * 80)
        
        # Load data
        if input_file:
            df = pd.read_csv(input_file)
        else:
            df = pd.DataFrame(self.all_reviews)
        
        logger.info(f"Initial dataset: {len(df)} reviews")
        
        # 1. Remove duplicates
        logger.info("\n[1/5] Removing duplicates...")
        df_dedup = self._remove_duplicates(df)
        logger.info(f"✓ After deduplication: {len(df_dedup)} reviews ({len(df) - len(df_dedup)} removed)")
        
        # 2. Filter spam
        logger.info("\n[2/5] Filtering spam...")
        df_clean = self._filter_spam(df_dedup)
        logger.info(f"✓ After spam filtering: {len(df_clean)} reviews ({len(df_dedup) - len(df_clean)} removed)")
        
        # 3. Quality checks
        logger.info("\n[3/5] Quality checks...")
        df_quality = self._quality_filter(df_clean)
        logger.info(f"✓ After quality filter: {len(df_quality)} reviews")
        
        # 4. Language detection
        logger.info("\n[4/5] Language detection...")
        df_lang = self._detect_language(df_quality)
        logger.info(f"✓ Language distribution: {df_lang['language'].value_counts().to_dict()}")
        
        # 5. Standardize
        logger.info("\n[5/5] Standardizing schema...")
        df_final = self._standardize_schema(df_lang)
        
        # Save processed data
        df_final.to_csv("data/processed/reviews_clean.csv", index=False)
        logger.info(f"\n✓ PREPROCESSING COMPLETE: {len(df_final)} reviews ready for analysis")
        logger.info(f"✓ Saved to: data/processed/reviews_clean.csv")
        
        return df_final
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate reviews"""
        # Simple exact duplicates
        df = df.drop_duplicates(subset=['review_text'], keep='first')
        
        # For fuzzy matching, would use fuzzywuzzy (skipped for speed)
        return df
    
    def _filter_spam(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter spam reviews"""
        spam_patterns = ['buy now', 'click here', 'limited time']
        
        def is_spam(text):
            if pd.isna(text):
                return True
            text_lower = str(text).lower()
            return any(pattern in text_lower for pattern in spam_patterns)
        
        df = df[~df['review_text'].apply(is_spam)]
        return df
    
    def _quality_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter low-quality reviews"""
        # Minimum word count
        df['word_count'] = df['review_text'].str.split().str.len()
        df = df[df['word_count'] >= 10]
        
        # Remove null text
        df = df.dropna(subset=['review_text'])
        
        return df
    
    def _detect_language(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect review language"""
        from langdetect import detect, LangDetectException
        
        def safe_detect(text):
            try:
                return detect(str(text))
            except LangDetectException:
                return 'unknown'
        
        df['language'] = df['review_text'].apply(safe_detect)
        
        # Filter to English only (for simplicity)
        df = df[df['language'] == 'en']
        
        return df
    
    def _standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize to unified schema"""
        df['review_id'] = [f"REV_{i:05d}" for i in range(len(df))]
        df['category'] = 'electronics'  # Could be enhanced with classification
        
        # Ensure all required columns exist
        required_cols = ['review_id', 'platform', 'rating', 'review_text', 
                        'review_date', 'reviewer_name', 'category']
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        
        return df[required_cols + ['verified_purchase', 'word_count', 'language']]
    
    # ============ PHASE 3: LLM ANALYSIS ============
    
    def analyze_with_llm(self, input_file: str = "data/processed/reviews_clean.csv"):
        """Run LLM analysis pipeline"""
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3: LLM ANALYSIS")
        logger.info("=" * 80)
        
        # Load processed data
        df = pd.read_csv(input_file)
        logger.info(f"Analyzing {len(df)} reviews")
        
        # Initialize LLM analyzer
        from src.llm_analysis.ollama_analyzer import OllamaAnalyzer
        
        analyzer = OllamaAnalyzer(
            model_name=self.config['llm']['models']['summarization'],
            temperature=self.config['llm']['generation_params']['temperature']
        )
        
        # Check Ollama connectivity
        if not analyzer.test_connection():
            logger.error("Cannot connect to Ollama. Please start Ollama server.")
            return None
        
        # Batch processing
        batch_size = self.config['llm']['batch_processing']['batch_size']
        results = []
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            logger.info(f"\nProcessing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}...")
            
            # 1. Summarization
            summary = analyzer.summarize(batch_df['review_text'].tolist())
            
            # 2. Sentiment Analysis
            sentiment = analyzer.analyze_sentiment(summary)
            
            # 3. Theme Extraction
            themes = analyzer.extract_themes(batch_df['review_text'].tolist())
            
            results.append({
                'batch_id': i//batch_size,
                'review_count': len(batch_df),
                'summary': summary,
                'sentiment': sentiment,
                'themes': themes
            })
            
            logger.info(f"✓ Batch complete - Sentiment: {sentiment.get('score', 'N/A')}")
        
        # Save results
        with open("data/results/analysis_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✓ LLM ANALYSIS COMPLETE: {len(results)} batches processed")
        logger.info(f"✓ Results saved to: data/results/analysis_results.json")
        
        return results
    
    # ============ PHASE 4: VISUALIZATION ============
    
    def generate_visualizations(self):
        """Generate charts and tables for paper"""
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 4: VISUALIZATION")
        logger.info("=" * 80)
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Load processed data
        df = pd.read_csv("data/processed/reviews_clean.csv")
        
        # Load analysis results
        with open("data/results/analysis_results.json", 'r') as f:
            results = json.load(f)
        
        # 1. Platform distribution
        logger.info("Generating platform distribution chart...")
        plt.figure(figsize=(10, 6))
        df['platform'].value_counts().plot(kind='bar')
        plt.title('Review Distribution by Platform')
        plt.xlabel('Platform')
        plt.ylabel('Number of Reviews')
        plt.tight_layout()
        plt.savefig('results/figures/platform_distribution.png', dpi=300)
        plt.close()
        
        # 2. Rating distribution
        logger.info("Generating rating distribution chart...")
        plt.figure(figsize=(10, 6))
        df['rating'].hist(bins=5, edgecolor='black')
        plt.title('Rating Distribution')
        plt.xlabel('Rating (1-5)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('results/figures/rating_distribution.png', dpi=300)
        plt.close()
        
        # 3. Sentiment over time (if dates available)
        # ... additional charts ...
        
        logger.info("✓ Visualizations saved to: results/figures/")
        
    def run_full_pipeline(self):
        """Execute complete pipeline"""
        start_time = datetime.now()
        logger.info("\n" + "=" * 80)
        logger.info("STARTING COMPLETE PIPELINE")
        logger.info("=" * 80)
        
        try:
            # Phase 1: Collect
            self.collect_all_reviews()
            
            # Phase 2: Preprocess
            self.preprocess_reviews()
            
            # Phase 3: Analyze
            self.analyze_with_llm()
            
            # Phase 4: Visualize
            self.generate_visualizations()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60
            
            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETE!")
            logger.info(f"Total execution time: {duration:.2f} minutes")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


# ============ COMMAND LINE INTERFACE ============

def main():
    parser = argparse.ArgumentParser(description='Review Analysis Pipeline')
    parser.add_argument('--mode', choices=['all', 'collect', 'preprocess', 'analyze', 'visualize'],
                       default='all', help='Pipeline mode')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    pipeline = ReviewAnalysisPipeline(config_path=args.config)
    
    if args.mode == 'all':
        pipeline.run_full_pipeline()
    elif args.mode == 'collect':
        pipeline.collect_all_reviews()
    elif args.mode == 'preprocess':
        pipeline.preprocess_reviews()
    elif args.mode == 'analyze':
        pipeline.analyze_with_llm()
    elif args.mode == 'visualize':
        pipeline.generate_visualizations()


if __name__ == "__main__":
    main()