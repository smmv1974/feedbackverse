"""
Cloud-based LLM analyzer with support for FREE cloud APIs.

Supports:
- Google Gemini API (FREE, recommended)
- Groq API (FREE, fast)
- OpenAI API (paid but cheap)
- Fallback to local Ollama if all cloud APIs fail

Setup:
1. Get API key from provider
2. Set environment variable (GOOGLE_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY)
3. Install: pip install google-generativeai groq openai
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8-sig") as fh:
        return yaml.safe_load(fh)


class CloudAnalyzer:
    """
    Cloud-based LLM analyzer with FREE API support and local fallback.

    Supports multiple providers:
    - Google Gemini (FREE, recommended)
    - Groq (FREE, very fast)
    - OpenAI (paid but cheap)
    - Local Ollama (fallback)
    """

    def __init__(self, config_path: str = "config/config_cloud.yaml"):
        self.config = load_config(config_path)
        self.cloud_cfg = self.config.get("llm", {}).get("cloud", {})
        self.enabled = self.cloud_cfg.get("enabled", False)

        # Generation parameters
        gen_cfg = self.config.get("llm", {}).get("generation", {})
        self.generation = gen_cfg
        self.temperature = gen_cfg.get("temperature", 0.1)
        self.max_tokens = gen_cfg.get("tokens", {}).get("summary_max", 1000)

        # Model tracking (for compatibility with OllamaAnalyzer interface)
        self.models = {
            "summarization": "cloud",
            "sentiment": "cloud",
            "themes": "cloud"
        }
        self.latest_models = self.models.copy()

        # Initialize API clients
        self.clients = {}
        self._init_apis()

        # Fallback to local Ollama
        self.local_fallback_enabled = self.cloud_cfg.get("local_fallback", {}).get("enabled", True)
        if self.local_fallback_enabled:
            try:
                from src.llm_analysis.ollama_analyzer import OllamaAnalyzer
                server_cfg = self.config.get("llm", {}).get("server", {})
                fallback_config_path = config_path.replace("cloud", "alt")
                if not Path(fallback_config_path).exists():
                    fallback_config_path = "config/config_alt.yaml"
                self.local_analyzer = OllamaAnalyzer(fallback_config_path)
                logger.info("Local Ollama fallback enabled")
            except Exception as e:
                logger.warning(f"Could not initialize local fallback: {e}")
                self.local_fallback_enabled = False

    def _init_apis(self):
        """Initialize cloud API clients."""
        # Google Gemini
        if self._has_api_key("GOOGLE_API_KEY"):
            try:
                import google.generativeai as genai
                api_key = os.getenv("GOOGLE_API_KEY")
                genai.configure(api_key=api_key)

                # Use Gemini 2.5 Flash (free tier)
                model_name = self.cloud_cfg.get("primary", {}).get("model", "gemini-2.5-flash")
                self.clients["google"] = genai.GenerativeModel(model_name)
                logger.info(f"✅ Google Gemini initialized ({model_name})")
            except ImportError:
                logger.warning("google-generativeai not installed. Run: pip install google-generativeai")
            except Exception as e:
                logger.warning(f"Failed to initialize Google Gemini: {e}")

        # Groq
        if self._has_api_key("GROQ_API_KEY"):
            try:
                from groq import Groq
                self.clients["groq"] = Groq(api_key=os.getenv("GROQ_API_KEY"))
                logger.info("✅ Groq initialized")
            except ImportError:
                logger.warning("groq not installed. Run: pip install groq")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq: {e}")

        # OpenAI
        if self._has_api_key("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                self.clients["openai"] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                logger.info("✅ OpenAI initialized")
            except ImportError:
                logger.warning("openai not installed. Run: pip install openai")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")

        if not self.clients:
            logger.error("❌ No cloud APIs available! Set GOOGLE_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY")

    def _has_api_key(self, env_var: str) -> bool:
        """Check if API key exists in environment."""
        return bool(os.getenv(env_var))

    def _call_google(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call Google Gemini API."""
        if "google" not in self.clients:
            return None

        for attempt in range(max_retries):
            try:
                response = self.clients["google"].generate_content(
                    prompt,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens,
                    }
                )
                return response.text
            except Exception as e:
                logger.warning(f"Google Gemini attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        return None

    def _call_groq(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call Groq API."""
        if "groq" not in self.clients:
            return None

        model = self.cloud_cfg.get("fallback_1", {}).get("model", "meta-llama/llama-4-scout-17b-16e-instruct")

        for attempt in range(max_retries):
            try:
                response = self.clients["groq"].chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Groq attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def _call_openai(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call OpenAI API."""
        if "openai" not in self.clients:
            return None

        model = self.cloud_cfg.get("fallback_2", {}).get("model", "gpt-4o-mini")

        for attempt in range(max_retries):
            try:
                response = self.clients["openai"].chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def _call_cloud_with_fallback(self, prompt: str) -> str:
        """
        Call cloud APIs with fallback chain:
        Google Gemini → Groq → OpenAI → Local Ollama
        """
        # Try Google Gemini first (FREE)
        result = self._call_google(prompt)
        if result:
            return result

        # Try Groq (FREE)
        result = self._call_groq(prompt)
        if result:
            return result

        # Try OpenAI (paid but cheap)
        result = self._call_openai(prompt)
        if result:
            return result

        # Fallback to local Ollama
        if self.local_fallback_enabled:
            logger.warning("All cloud APIs failed, falling back to local Ollama")
            # This will be handled by the caller
            return ""

        raise RuntimeError("All cloud APIs failed and no local fallback available")

    def summarize_batch(self, reviews: List[Dict]) -> List[str]:
        """Generate summaries for a batch of reviews."""
        summaries = []

        for review in reviews:
            text = review.get("text", "")
            prompt = f"""Summarize this review in 2-3 sentences. Focus on the main points and sentiment.

Review: {text}

Summary:"""

            try:
                summary = self._call_cloud_with_fallback(prompt)
                if not summary and self.local_fallback_enabled:
                    # Use local fallback
                    batch_summaries = self.local_analyzer.summarize_batch([review])
                    summary = batch_summaries[0] if batch_summaries else "No summary available"
                summaries.append(summary or "No summary available")
            except Exception as e:
                logger.error(f"Summarization failed: {e}")
                summaries.append("Error generating summary")

        return summaries

    def analyze_sentiment_batch(self, reviews: List[Dict]) -> List[float]:
        """
        Analyze sentiment for a batch of reviews.
        Returns sentiment scores from -1 (very negative) to +1 (very positive).
        """
        sentiments = []

        for review in reviews:
            text = review.get("text", "")
            prompt = f"""Analyze the sentiment of this review and provide a score from -1 to +1:
- -1 = Very negative (terrible, awful, worst)
- -0.5 = Negative (disappointed, bad)
- 0 = Neutral (okay, average)
- +0.5 = Positive (good, nice)
- +1 = Very positive (amazing, excellent, best)

Review: {text}

Respond with ONLY a number between -1 and +1 (e.g., 0.75 or -0.3). No explanation needed.

Sentiment score:"""

            try:
                response = self._call_cloud_with_fallback(prompt)

                if not response and self.local_fallback_enabled:
                    # Use local fallback
                    batch_sentiments = self.local_analyzer.analyze_sentiment_batch([review])
                    sentiment = batch_sentiments[0] if batch_sentiments else 0.0
                else:
                    # Parse sentiment from response
                    try:
                        # Extract first number found
                        import re
                        numbers = re.findall(r'-?\d+\.?\d*', response)
                        if numbers:
                            sentiment = float(numbers[0])
                            # Clamp to [-1, 1]
                            sentiment = max(-1.0, min(1.0, sentiment))
                        else:
                            logger.warning(f"Could not parse sentiment from: {response}")
                            sentiment = 0.0
                    except (ValueError, AttributeError):
                        sentiment = 0.0

                sentiments.append(sentiment)
            except Exception as e:
                logger.error(f"Sentiment analysis failed: {e}")
                sentiments.append(0.0)

        return sentiments

    def extract_themes_batch(self, reviews: List[Dict]) -> List[List[str]]:
        """Extract key themes from a batch of reviews."""
        all_themes = []

        for review in reviews:
            text = review.get("text", "")
            prompt = f"""Extract 3-5 key themes or topics from this review.
Each theme should be 1-3 words.

Review: {text}

Themes (one per line):"""

            try:
                response = self._call_cloud_with_fallback(prompt)

                if not response and self.local_fallback_enabled:
                    # Use local fallback
                    batch_themes = self.local_analyzer.extract_themes_batch([review])
                    themes = batch_themes[0] if batch_themes else []
                else:
                    # Parse themes from response
                    themes = [line.strip().lstrip('-•* ') for line in response.split('\n') if line.strip()]
                    themes = themes[:5]  # Max 5 themes

                all_themes.append(themes)
            except Exception as e:
                logger.error(f"Theme extraction failed: {e}")
                all_themes.append([])

        return all_themes

    def analyze_batch(self, reviews: List[Dict], batch_index: int) -> Dict:
        """
        Analyze a batch of reviews using cloud APIs.

        Compatible with OllamaAnalyzer interface.

        Args:
            reviews: List of review dicts with 'review_id' and 'review_text' keys
            batch_index: Index of this batch

        Returns:
            Dict with summary, themes, sentiments, and metadata
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

        # Track which APIs were used
        models_used = {
            "summarization": self._get_active_provider(),
            "themes": self._get_active_provider(),
            "sentiment": self._get_active_provider(),
        }
        self.latest_models = models_used

        return {
            "batch_index": batch_index,
            "review_ids": [r.get("review_id", r.get("id", f"review_{i}")) for i, r in enumerate(reviews)],
            "summary": summary,
            "themes": themes,
            "sentiments": sentiments,
            "models_used": models_used,
            "generation_params": {
                "temperature": self.generation.get("temperature"),
                "top_p": self.generation.get("top_p"),
                "repeat_penalty": self.generation.get("repeat_penalty"),
                "tokens": self.generation.get("tokens", {}),
            },
            "elapsed_s": elapsed,
        }

    def _get_active_provider(self) -> str:
        """Get the name of the currently active provider."""
        if "google" in self.clients:
            return "Google Gemini (cloud)"
        elif "groq" in self.clients:
            return "Groq (cloud)"
        elif "openai" in self.clients:
            return "OpenAI (cloud)"
        elif self.local_fallback_enabled:
            return "Local Ollama (fallback)"
        else:
            return "No provider available"
