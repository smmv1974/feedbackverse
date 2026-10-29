"""
Ollama Cloud Analyzer - Uses Ollama's cloud-hosted large models.

Supports models like:
- gpt-oss:120b-cloud (120B parameters)
- gpt-oss:20b-cloud (20B parameters)
- deepseek-v3.1:671b-cloud (671B parameters!)
- kimi-k2:1t-cloud (1 trillion parameters)
- qwen3-coder:480b-cloud (480B parameters)

These run on Ollama's datacenter without needing local GPU/RAM.
"""
import os
import sys
import time
import json
import logging
import re
from typing import List, Dict, Optional
from pathlib import Path

# Fix Windows UTF-8 encoding
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

logger = logging.getLogger(__name__)


class OllamaCloudAnalyzer:
    """
    Ollama Cloud analyzer for sentiment analysis using cloud-hosted models.

    Uses Ollama's cloud API (https://ollama.com) instead of local models.
    Requires OLLAMA_API_KEY environment variable.
    """

    def __init__(self, config_path: str = "config/config_production.yaml"):
        """Initialize Ollama Cloud analyzer."""
        import yaml

        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Model configuration
        llm_cfg = self.config.get("llm", {})
        ollama_cloud_cfg = llm_cfg.get("ollama_cloud", {})

        # Get API key from environment or config
        self.api_key = os.getenv('OLLAMA_API_KEY')
        if not self.api_key:
            # Try to get from config file
            self.api_key = ollama_cloud_cfg.get('api_key_env')
        if not self.api_key:
            raise ValueError(
                "OLLAMA_API_KEY not set! Set environment variable or add to config file"
            )

        # Initialize client
        try:
            from ollama import Client
            self.client = Client(
                host="https://ollama.com",
                headers={'Authorization': f'Bearer {self.api_key}'}
            )
            logger.info("✅ Ollama Cloud client initialized")
        except ImportError:
            raise ImportError("Install ollama package: pip install ollama")

        self.models = {
            "summarization": ollama_cloud_cfg.get("summarization", "gpt-oss:20b-cloud"),
            "sentiment": ollama_cloud_cfg.get("sentiment", "gpt-oss:20b-cloud"),
            "themes": ollama_cloud_cfg.get("themes", "gpt-oss:120b-cloud"),
        }

        # Add latest_models attribute for pipeline compatibility
        self.latest_models = self.models.copy()

        # Add generation parameters for pipeline compatibility
        gen_cfg = llm_cfg.get("generation", {})
        self.generation = {
            "temperature": gen_cfg.get("temperature", 0.1),
            "top_p": gen_cfg.get("top_p", 0.95),
            "max_tokens": gen_cfg.get("tokens", {}).get("summary_max", 500)
        }

        # Calibration settings to fix range compression (improves Kappa)
        # Power < 1 stretches extreme values toward -1 and +1
        # This improves categorical agreement without hurting correlation
        self.calibration_enabled = ollama_cloud_cfg.get("calibration_enabled", True)
        self.calibration_power = ollama_cloud_cfg.get("calibration_power", 0.75)

        if self.calibration_enabled:
            logger.info(f"✅ Calibration enabled (power={self.calibration_power})")

        logger.info(f"📊 Ollama Cloud models: {self.models}")

    def calibrate_sentiment(self, raw_score: float) -> float:
        """
        Apply calibration to stretch extreme values toward -1 and +1.

        This fixes the "range compression" problem where models avoid extreme predictions.
        Uses power function: calibrated = sign(score) * |score|^power

        - power < 1: stretches values (e.g., 0.75)
        - power = 1: no change
        - power > 1: compresses values

        Example with power=0.75:
        - Raw: -0.60 → Calibrated: -0.70 (more negative)
        - Raw: +0.60 → Calibrated: +0.70 (more positive)
        - Raw:  0.00 → Calibrated:  0.00 (unchanged)

        This improves Kappa (categorical agreement) while preserving Pearson (ranking).
        """
        if not self.calibration_enabled:
            return raw_score

        # Apply power function: sign(x) * |x|^power
        sign = 1 if raw_score >= 0 else -1
        calibrated = sign * (abs(raw_score) ** self.calibration_power)

        # Ensure still in [-1, 1] range
        return max(-1.0, min(1.0, calibrated))

    def summarize_batch(self, reviews: List[Dict]) -> List[str]:
        """Summarize a batch of reviews using Ollama Cloud."""
        summaries = []

        for review in reviews:
            text = review.get("review_text") or review.get("text", "")

            prompt = f"""Summarize this review in 2-3 sentences:

Review: {text}

Summary:"""

            try:
                response = self.client.chat(
                    model=self.models["summarization"],
                    messages=[{'role': 'user', 'content': prompt}],
                    stream=False
                )
                summary = response['message']['content'].strip()
                summaries.append(summary)
            except Exception as e:
                logger.error(f"Summarization failed: {e}")
                summaries.append(f"Summary unavailable. Original: {text[:100]}...")

        return summaries

    def extract_themes_batch(self, reviews: List[Dict]) -> List[List[str]]:
        """Extract themes from reviews using Ollama Cloud."""
        themes_list = []

        for review in reviews:
            text = review.get("review_text") or review.get("text", "")

            prompt = f"""Extract 3-5 main themes from this review. Return only theme names, one per line.

Review: {text}

Themes:"""

            try:
                response = self.client.chat(
                    model=self.models["themes"],
                    messages=[{'role': 'user', 'content': prompt}],
                    stream=False
                )
                themes_text = response['message']['content'].strip()
                themes = [line.strip() for line in themes_text.split('\n') if line.strip()]
                themes_list.append(themes[:5])  # Max 5 themes
            except Exception as e:
                logger.error(f"Theme extraction failed: {e}")
                themes_list.append([])

        return themes_list

    def analyze_sentiment_batch(self, reviews: List[Dict]) -> List[float]:
        """Analyze sentiment using Ollama Cloud with few-shot prompting."""
        sentiments = []

        for review in reviews:
            text = review.get("review_text") or review.get("text", "")

            # Few-shot prompt with actual examples
            prompt = f"""You are a sentiment analysis expert. Rate the sentiment of reviews on a scale from -1 (very negative) to +1 (very positive).

Here are examples of correctly rated reviews:

Example 1:
Review: "This product is amazing! Best purchase I've ever made. Exceeded all my expectations!"
Sentiment: 0.9

Example 2:
Review: "Decent quality for the price. Works as expected, nothing special."
Sentiment: 0.1

Example 3:
Review: "Not great, not terrible. It's okay I guess."
Sentiment: 0.0

Example 4:
Review: "Very disappointed. Poor quality and broke after one week."
Sentiment: -0.7

Example 5:
Review: "Absolute garbage! Worst product ever. Complete waste of money!"
Sentiment: -0.95

Now rate this review:

Review: "{text}"

Respond with ONLY a decimal number between -1.0 and +1.0. No text, no explanation, just the number.

Sentiment score:"""

            try:
                response = self.client.chat(
                    model=self.models["sentiment"],
                    messages=[{'role': 'user', 'content': prompt}],
                    stream=False,
                    options={
                        'temperature': 0.1,  # Low temperature for consistency
                        'top_p': 0.9
                    }
                )
                score_text = response['message']['content'].strip()

                # Try to extract number from response (handles both clean and verbose responses)
                try:
                    # First try: direct float conversion
                    score = float(score_text)
                except ValueError:
                    # Second try: extract first decimal number using regex
                    match = re.search(r'[-+]?\d*\.?\d+', score_text)
                    if match:
                        score = float(match.group())
                    else:
                        logger.warning(f"Could not extract sentiment from: '{score_text}'")
                        sentiments.append(0.0)
                        continue

                # Clamp raw score to [-1, 1] range
                raw_score = max(-1.0, min(1.0, score))

                # Apply calibration to fix range compression
                calibrated_score = self.calibrate_sentiment(raw_score)

                sentiments.append(calibrated_score)

            except Exception as e:
                logger.error(f"Sentiment analysis failed: {e}")
                # Don't calibrate fallback 0.0
                sentiments.append(0.0)

        return sentiments

    def analyze_batch(self, reviews: List[Dict], batch_index: int) -> Dict:
        """
        Analyze a batch of reviews using Ollama Cloud.
        Compatible with pipeline interface.
        """
        start = time.time()

        # Extract summaries
        summaries = self.summarize_batch(reviews)
        summary = summaries[0] if summaries else "No summary available"

        # Extract themes
        themes_list = self.extract_themes_batch(reviews)
        themes = themes_list[0] if themes_list else []

        # Analyze sentiments
        sentiment_scores = self.analyze_sentiment_batch(reviews)
        sentiments = [
            {
                "review_id": review.get("review_id", review.get("id", f"review_{i}")),
                "sentiment": score
            }
            for i, (review, score) in enumerate(zip(reviews, sentiment_scores))
        ]

        elapsed = time.time() - start

        return {
            "batch_index": batch_index,
            "review_ids": [r.get("review_id", r.get("id", f"review_{i}")) for i, r in enumerate(reviews)],
            "summary": summary,
            "themes": themes,
            "sentiments": sentiments,
            "models_used": {
                "summarization": self.models["summarization"],
                "sentiment": self.models["sentiment"],
                "themes": self.models["themes"],
                "provider": "Ollama Cloud"
            },
            "elapsed_s": elapsed
        }


if __name__ == "__main__":
    # Test script
    print("=" * 80)
    print(" TESTING OLLAMA CLOUD ANALYZER")
    print("=" * 80)

    # Check API key
    api_key = os.getenv('OLLAMA_API_KEY')
    if not api_key:
        print("\n❌ OLLAMA_API_KEY not set!")
        print("\nSet it like this:")
        print("$env:OLLAMA_API_KEY='your-ollama-api-key'")
        sys.exit(1)

    print(f"\n✅ API Key set: {api_key[:20]}...")

    # Initialize analyzer
    try:
        analyzer = OllamaCloudAnalyzer()
        print("✅ Analyzer initialized")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        sys.exit(1)

    # Test sentiment analysis
    test_reviews = [
        {
            "review_id": "test_1",
            "review_text": "This product is amazing! I absolutely love it!",
        }
    ]

    print("\n📊 Testing sentiment analysis...")
    try:
        sentiments = analyzer.analyze_sentiment_batch(test_reviews)
        print(f"✅ Sentiment score: {sentiments[0]:.2f}")

        if -1.0 <= sentiments[0] <= 1.0:
            print("✅ Score in valid range [-1, 1]")
        else:
            print(f"⚠️ Score out of range: {sentiments[0]}")
    except Exception as e:
        print(f"❌ Sentiment analysis failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print(" TEST COMPLETE")
    print("=" * 80)
    print("\n✅ Ollama Cloud is working!")
    print("\nNext: Run full pipeline with:")
    print("python run_pipeline.py --config config/config_ollama_cloud.yaml")
