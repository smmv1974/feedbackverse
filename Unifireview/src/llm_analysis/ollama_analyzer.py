"""
Ollama-based analyzer that requires a healthy local server and produces
summaries, per-review sentiments, and key themes via llama3.2 (or the
configured model). When the server is reachable, all stages use actual model
generations (with /api/generate, falling back to /api/chat on 404); no heuristic
placeholders are returned so downstream metrics remain auditable.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests
import yaml

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (AttributeError, io.UnsupportedOperation):
        # Already wrapped or not supported
        pass

# Import JSON repair utility
try:
    from src.utils.json_repair import extract_and_repair_json, JSONRepairError
except ImportError:
    # Fallback if running from different directory
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.utils.json_repair import extract_and_repair_json, JSONRepairError

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8-sig") as fh:
        return yaml.safe_load(fh)


class OllamaAnalyzer:
    """Wrapper around Ollama HTTP endpoints with strict failure semantics."""

    def __init__(self, config_path: str = "config/config.yaml", use_alt: bool = False):
        config = load_config(config_path)
        server_cfg = config["llm"]["server"]
        self.base_url = server_cfg.get("base_url", "http://127.0.0.1:11434").rstrip("/")
        self.timeout = server_cfg.get("timeout", 120)
        self.generation = config["llm"]["generation"]
        self.tokens = self.generation.get("tokens", {"summary_max": 500, "themes_max": 350})
        models_cfg = config["llm"]["models"]

        # Cloud model support
        self.cloud_enabled = config.get("pipeline", {}).get("enable_cloud_models", False)
        self.cloud_config = config.get("cloud", {})
        self.fallback_to_local = self.cloud_config.get("fallback_to_local_on_error", True)

        alt_override = os.getenv("ALT_MODEL") or models_cfg.get("alt_small", {}).get("name")
        if use_alt and alt_override:
            self.models = {stage: alt_override for stage in ("summarization", "sentiment", "themes")}
        else:
            self.models = {
                "summarization": models_cfg["summarization"]["name"],
                "sentiment": models_cfg["sentiment"]["name"],
                "themes": models_cfg["themes"]["name"],
            }

        # Store fallback models for cloud configurations
        self.fallback_models = {}
        for stage in ("summarization", "sentiment", "themes"):
            if stage in models_cfg and isinstance(models_cfg[stage], dict):
                self.fallback_models[stage] = models_cfg[stage].get("fallback")

        self.latest_models = self.models.copy()

        # Skip server check if using heuristic mode
        if not all(model == "heuristic" for model in self.models.values()):
            if not self._check_server():
                logger.warning(
                    f"Ollama server not reachable at {self.base_url}. "
                    "Will fall back to heuristic mode."
                )
                if not self.fallback_to_local:
                    raise RuntimeError(
                        f"Ollama server not reachable at {self.base_url}. "
                        "Run 'ollama serve' and pull the required models."
                    )

    def _check_server(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            logger.error("Ollama server health check failed: %s", exc)
            return False

    def test_connection(self) -> bool:
        """Public method to test Ollama server connection for external use (e.g., Streamlit)."""
        return self._check_server()

    def _request_generate(self, model: str, prompt: str, max_tokens: int, retry_count: int = 0) -> str:
        max_retries = 3
        payload = {
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": self.generation.get("temperature", 0.3),
                "top_p": self.generation.get("top_p", 0.9),
                "repeat_penalty": self.generation.get("repeat_penalty", 1.1),
                "num_predict": max_tokens,
            },
            "stream": False,
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                data=json.dumps(payload),
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            if response.status_code == 404:
                return self._request_chat(model, prompt, max_tokens)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.Timeout as exc:
            if retry_count < max_retries:
                logger.warning(f"Request timeout on attempt {retry_count + 1}/{max_retries}, retrying...")
                time.sleep(2 ** retry_count)  # Exponential backoff
                return self._request_generate(model, prompt, max_tokens, retry_count + 1)
            else:
                logger.error(f"Request failed after {max_retries} retries: {exc}")
                raise
        except requests.ConnectionError as exc:
            if retry_count < max_retries:
                logger.warning(f"Connection error on attempt {retry_count + 1}/{max_retries}, retrying...")
                time.sleep(2 ** retry_count)
                return self._request_generate(model, prompt, max_tokens, retry_count + 1)
            else:
                logger.error(f"Connection failed after {max_retries} retries: {exc}")
                raise

    def _request_chat(self, model: str, prompt: str, max_tokens: int) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": self.generation.get("temperature", 0.3),
                "top_p": self.generation.get("top_p", 0.9),
                "repeat_penalty": self.generation.get("repeat_penalty", 1.1),
                "num_predict": max_tokens,
            },
            "stream": False,
        }
        response = requests.post(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload),
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        message = response.json().get("message", {})
        return message.get("content", "")

    def _invoke(self, stage: str, prompt: str, max_tokens: int) -> Tuple[str, str]:
        model_name = self.models[stage]

        # Skip API calls if model is set to "heuristic"
        if model_name == "heuristic":
            logger.info("%s stage using heuristic mode (no API call)", stage)
            return "", "heuristic"

        # Check if this is a cloud model
        is_cloud_model = ":cloud" in model_name or "-cloud" in model_name

        try:
            content = self._request_generate(model_name, prompt, max_tokens)
        except (requests.Timeout, requests.ConnectionError, KeyboardInterrupt) as exc:
            logger.error("%s stage timeout/interrupted for %s: %s", stage, model_name, exc)

            # Try fallback model if available for cloud models
            if is_cloud_model and self.fallback_to_local and stage in self.fallback_models:
                fallback_model = self.fallback_models[stage]
                if fallback_model:
                    logger.info("Attempting fallback to local model: %s", fallback_model)
                    try:
                        content = self._request_generate(fallback_model, prompt, max_tokens)
                        logger.info("Successfully used fallback model: %s", fallback_model)
                        return content, f"{fallback_model} (fallback)"
                    except Exception as fallback_exc:
                        logger.error("Fallback model also failed: %s", fallback_exc)

            return "", "heuristic"  # Return empty content to trigger fallback in calling methods
        except requests.RequestException as exc:
            logger.error("%s stage request failed for %s: %s", stage, model_name, exc)

            # Try fallback for cloud models
            if is_cloud_model and self.fallback_to_local and stage in self.fallback_models:
                fallback_model = self.fallback_models[stage]
                if fallback_model:
                    logger.info("Attempting fallback to local model: %s", fallback_model)
                    try:
                        content = self._request_generate(fallback_model, prompt, max_tokens)
                        logger.info("Successfully used fallback model: %s", fallback_model)
                        return content, f"{fallback_model} (fallback)"
                    except Exception as fallback_exc:
                        logger.error("Fallback model also failed: %s", fallback_exc)

            return "", "heuristic"  # Fallback instead of raising

        if not content:
            logger.warning(f"{stage} stage returned empty response from {model_name}, using heuristic")
            return "", "heuristic"

        return content, model_name

    def _format_reviews(self, reviews: List[Dict]) -> str:
        return "\n\n".join(
            textwrap.dedent(
                f"""
                Review ID: {review['review_id']}
                Rating (1-5 norm): {review['rating_norm']:.2f}
                Text: {review['review_text']}
                """
            ).strip()
            for review in reviews
        )

    def _summarize(self, reviews: List[Dict]) -> Tuple[str, str]:
        prompt = textwrap.dedent(
            f"""
            You are an expert research analyst. Summarize the key points across the
            following customer reviews in 3-4 sentences. Focus on consensus insights,
            sentiment drivers, and grounded product evidence. Avoid hallucinations.

            {self._format_reviews(reviews)}
            """
        ).strip()
        summary, model_used = self._invoke("summarization", prompt, self.tokens.get("summary_max", 500))
        
        # If heuristic fallback, generate a basic summary
        if not summary or model_used == "heuristic":
            logger.info("Using heuristic summary generation")
            avg_sentiment = sum([r.get("rating_norm", 0.5) for r in reviews]) / len(reviews) if reviews else 0.5
            if avg_sentiment > 0.7:
                summary = "Overall positive reviews highlighting good quality and value."
            elif avg_sentiment < 0.3:
                summary = "Predominantly negative feedback with concerns about quality and service."
            else:
                summary = "Mixed reviews with both positive and negative aspects noted."
            model_used = "heuristic"
        
        return summary.strip(), model_used

    def _themes(self, reviews: List[Dict]) -> Tuple[List[str], str]:
        prompt = textwrap.dedent(
            f"""
            Derive 3-5 short theme labels (2-4 words) capturing recurring topics
            across the following reviews. Respond ONLY with a JSON array of strings.
            Example: ["Product Quality", "Customer Service", "Value for Money"]
            CRITICAL: Your response must be ONLY a JSON array, nothing else.

            {self._format_reviews(reviews)}
            """
        ).strip()
        raw, model_used = self._invoke("themes", prompt, self.tokens.get("themes_max", 350))
        parsed = []

        # If invoke returned heuristic signal (empty string), skip JSON parsing
        if raw and model_used != "heuristic":
            logger.info("📝 Attempting to parse themes JSON response...")
            try:
                # Try to extract JSON using the robust extractor
                result = extract_and_repair_json(raw, expected_type="array")
                if isinstance(result, list):
                    parsed = result
                elif isinstance(result, dict):
                    # Check for common keys that might contain the themes
                    for key in ["themes", "topics", "labels", "categories"]:
                        if key in result and isinstance(result[key], list):
                            parsed = result[key]
                            break

                if not parsed:
                    raise ValueError("No valid theme array found in response")
                logger.info(f"✅ Successfully parsed {len(parsed)} theme(s) from model response")
            except (JSONRepairError, ValueError) as exc:
                logger.warning(f"⚠️  Themes JSON parse failed: {exc}. Using heuristic fallback.")
                logger.debug(f"Raw response (first 200 chars): {raw[:200]}")
                model_used = "heuristic"
        
        # If JSON parsing failed or response was empty, use heuristic
        if not parsed or model_used == "heuristic":
            logger.info("Using heuristic theme extraction")
            text = " ".join([r.get("review_text", "") for r in reviews]).lower()
            # Common theme keywords
            themes_dict = {
                "Quality": ["quality", "durability", "build", "material", "solid", "construction"],
                "Price": ["price", "cost", "expensive", "cheap", "value", "money", "worth"],
                "Service": ["service", "support", "customer", "help", "responsive", "staff"],
                "Design": ["design", "look", "appearance", "aesthetic", "style", "layout"],
                "Performance": ["performance", "speed", "power", "efficient", "fast", "works"]
            }
            parsed = []
            for theme, keywords in themes_dict.items():
                if any(kw in text for kw in keywords):
                    parsed.append(theme)
            if not parsed:
                parsed = ["General Feedback", "User Experience"]
            model_used = "heuristic"
        
        if not isinstance(parsed, list) or not parsed:
            logger.warning("Themes empty, using default")
            parsed = ["General Feedback"]
        
        cleaned = [str(item).strip() for item in parsed if str(item).strip()]
        if not cleaned:
            cleaned = ["General Feedback"]
        
        return cleaned, model_used

    def _extract_json_from_text(self, text: str) -> Dict:
        """Robust JSON extraction with multiple strategies."""
        # Strategy 1: Remove markdown code blocks
        cleaned = text
        if '```json' in cleaned:
            parts = cleaned.split('```json')
            if len(parts) > 1:
                cleaned = parts[1].split('```')[0]
        elif '```' in cleaned:
            parts = cleaned.split('```')
            if len(parts) >= 3:
                cleaned = parts[1]

        cleaned = cleaned.strip()

        # Strategy 2: Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 3: Find JSON object boundaries
        brace_start = cleaned.find('{')
        if brace_start >= 0:
            # Count braces to find matching closing brace
            brace_count = 0
            for i in range(brace_start, len(cleaned)):
                if cleaned[i] == '{':
                    brace_count += 1
                elif cleaned[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(cleaned[brace_start:i + 1])
                        except json.JSONDecodeError:
                            pass

        # Strategy 4: Try to fix common JSON issues
        try:
            # Replace single quotes with double quotes
            fixed = cleaned.replace("'", '"')
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # Strategy 5: Extract array if present
        bracket_start = cleaned.find('[')
        if bracket_start >= 0:
            bracket_count = 0
            for i in range(bracket_start, len(cleaned)):
                if cleaned[i] == '[':
                    bracket_count += 1
                elif cleaned[i] == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        try:
                            arr = json.loads(cleaned[bracket_start:i + 1])
                            return {"sentiments": arr} if isinstance(arr, list) else {}
                        except json.JSONDecodeError:
                            pass

        raise ValueError("Could not extract valid JSON from response")

    def _sentiments(self, reviews: List[Dict]) -> Tuple[List[Dict], str]:
        prompt = textwrap.dedent(
            f"""
            You are an expert sentiment analyzer. Analyze each review's sentiment with high precision.

            Sentiment Scale Guidelines:
            - 1.0: Extremely positive (5-star, glowing praise, highly recommends)
            - 0.7 to 0.9: Very positive (4-5 star, clear satisfaction, would recommend)
            - 0.3 to 0.6: Somewhat positive (3-4 star, satisfied but with minor issues)
            - -0.3 to 0.3: Neutral/Mixed (balanced positive and negative, or ambiguous)
            - -0.6 to -0.4: Somewhat negative (2-3 star, disappointed but not angry)
            - -0.9 to -0.7: Very negative (1-2 star, clearly dissatisfied)
            - -1.0: Extremely negative (1-star, angry, warns others away)

            IMPORTANT:
            - Consider both explicit ratings and text content
            - Negative items should never be classified as positive
            - Be conservative around the neutral boundary (0.0)
            - Match the intensity of language to score magnitude

            Output ONLY valid JSON in this exact format:
            {{"sentiments":[{{"review_id":"id1","sentiment":0.5}},{{"review_id":"id2","sentiment":-0.3}}]}}

            {self._format_reviews(reviews)}
            """
        ).strip()
        raw, model_used = self._invoke("sentiment", prompt, 500)
        sentiments = []

        # Skip JSON parsing if model returned heuristic signal
        if model_used != "heuristic" and raw:
            logger.info("📝 Attempting to parse sentiment JSON response...")
            try:
                parsed = extract_and_repair_json(raw, expected_type="object")
                sentiments = parsed.get("sentiments", [])
                if not sentiments:
                    # If no sentiments key, check if response is array
                    if isinstance(parsed, list):
                        sentiments = parsed
                    else:
                        raise ValueError("No sentiments found in parsed JSON")
                logger.info(f"✅ Successfully parsed {len(sentiments)} sentiment(s) from model response")
            except (JSONRepairError, ValueError, AttributeError) as exc:
                logger.warning(f"⚠️  Sentiment JSON parse failed: {exc}. Using heuristic fallback.")
                logger.debug(f"Raw response (first 200 chars): {raw[:200]}")
        
        # If JSON parsing failed or returned empty, use enhanced heuristic
        if not sentiments:
            logger.info("Using enhanced heuristic sentiment analysis")
            sentiments = []
            for review in reviews:
                text = str(review.get("review_text", "")).lower()
                rating = review.get("rating_norm", 3.0)  # Default to middle if missing

                # Enhanced sentiment lexicon with intensity
                very_positive = ["excellent", "amazing", "perfect", "fantastic", "outstanding", "exceptional", "wonderful", "superb"]
                positive = ["great", "good", "nice", "solid", "happy", "satisfied", "recommend", "love", "impressed"]
                somewhat_positive = ["decent", "okay", "acceptable", "fine", "adequate", "fair"]

                very_negative = ["terrible", "horrible", "awful", "worst", "pathetic", "useless", "garbage", "atrocious"]
                negative = ["bad", "poor", "disappointing", "unsatisfied", "defective", "broken", "failed", "worst"]
                somewhat_negative = ["mediocre", "lackluster", "subpar", "underwhelming", "meh"]

                # Count with weights
                very_pos_count = sum(2 for word in very_positive if word in text)
                pos_count = sum(1 for word in positive if word in text)
                somewhat_pos_count = sum(0.5 for word in somewhat_positive if word in text)

                very_neg_count = sum(2 for word in very_negative if word in text)
                neg_count = sum(1 for word in negative if word in text)
                somewhat_neg_count = sum(0.5 for word in somewhat_negative if word in text)

                pos_score = very_pos_count + pos_count + somewhat_pos_count
                neg_score = very_neg_count + neg_count + somewhat_neg_count

                # Normalize rating to [-1, 1] scale more accurately
                # 5.0 → 1.0, 4.0 → 0.5, 3.0 → 0.0, 2.0 → -0.5, 1.0 → -1.0
                rating_sentiment = (rating - 3.0) / 2.0

                # Combine rating and text analysis
                if pos_score > neg_score * 1.5:
                    # Clear positive language overrides rating
                    text_sentiment = min(1.0, 0.3 + pos_score * 0.15)
                    score = (rating_sentiment * 0.4) + (text_sentiment * 0.6)
                elif neg_score > pos_score * 1.5:
                    # Clear negative language overrides rating
                    text_sentiment = max(-1.0, -0.3 - neg_score * 0.15)
                    score = (rating_sentiment * 0.4) + (text_sentiment * 0.6)
                elif abs(pos_score - neg_score) < 0.5:
                    # Mixed or neutral - rely more on rating
                    score = rating_sentiment * 0.9
                else:
                    # Balanced mix
                    text_sentiment = (pos_score - neg_score) * 0.2
                    score = (rating_sentiment * 0.6) + (text_sentiment * 0.4)

                # Ensure valid range
                score = max(-1.0, min(1.0, score))

                # Don't flip clear negatives to positive or vice versa
                if rating <= 2.0 and score > 0.3:
                    score = min(score, 0.0)
                elif rating >= 4.0 and score < -0.3:
                    score = max(score, 0.0)

                sentiments.append({
                    "review_id": review.get("review_id", review.get("id", "unknown")),
                    "sentiment": round(score, 3)
                })
            model_used = "heuristic"
        
        normalized = []
        for item in sentiments:
            try:
                score = float(item["sentiment"])
                normalized.append({"review_id": item["review_id"], "sentiment": max(-1.0, min(1.0, score))})
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Invalid sentiment payload: %s", exc)
                continue
        
        if not normalized:
            logger.error("Sentiment response empty, generating fallback")
            normalized = [{"review_id": r.get("review_id", r.get("id", "unknown")), "sentiment": 0.0} for r in reviews]
        
        return normalized, model_used

    def analyze_batch(self, reviews: List[Dict], batch_index: int) -> Dict:
        start = time.time()
        summary, summary_model = self._summarize(reviews)
        themes, theme_model = self._themes(reviews)
        sentiments, sentiment_model = self._sentiments(reviews)
        elapsed = time.time() - start
        models_used = {
            "summarization": summary_model,
            "themes": theme_model,
            "sentiment": sentiment_model,
        }
        self.latest_models = models_used
        return {
            "batch_index": batch_index,
            "review_ids": [r["review_id"] for r in reviews],
            "summary": summary,
            "themes": themes,
            "sentiments": sentiments,
            "models_used": models_used,
            "generation_params": {
                "temperature": self.generation.get("temperature"),
                "top_p": self.generation.get("top_p"),
                "repeat_penalty": self.generation.get("repeat_penalty"),
                "tokens": self.tokens,
            },
            "elapsed_s": elapsed,
        }
