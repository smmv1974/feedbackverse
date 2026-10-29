"""
Advanced JSON repair and extraction utility.
Handles malformed JSON from LLM responses with multiple repair strategies.
"""

import json
import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class JSONRepairError(Exception):
    """Raised when JSON cannot be repaired."""
    pass


def repair_json_string(text: str) -> str:
    """
    Attempt to repair common JSON syntax errors.

    Handles:
    - Missing commas between elements
    - Trailing commas
    - Unquoted keys
    - Comments
    - Single quotes instead of double quotes
    """
    # Remove comments
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    # Fix missing commas between object elements: }{ -> },{
    text = re.sub(r'\}\s*\{', '},{', text)

    # Fix missing commas between quoted strings: "" "" -> "","
    text = re.sub(r'"\s+"', '","', text)

    # Fix trailing commas before closing braces/brackets
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)

    # Fix missing commas after numbers/booleans/null: 123 "key" -> 123,"key"
    text = re.sub(r'(\d+|true|false|null)\s+"', r'\1,"', text)

    # Fix missing commas between array elements: ][ -> ],[
    text = re.sub(r'\]\s*\[', '],[', text)

    # Fix unquoted property names (simple cases)
    text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', text)

    # Fix single quotes to double quotes (carefully)
    # Only replace single quotes that look like string delimiters
    text = re.sub(r"'([^']*)'(\s*[,\]}:])", r'"\1"\2', text)

    return text


def extract_json_objects(text: str) -> list:
    """Extract all JSON objects from text."""
    objects = []
    brace_count = 0
    start_pos = None

    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_pos = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_pos is not None:
                obj_str = text[start_pos:i + 1]
                objects.append(obj_str)
                start_pos = None

    return objects


def extract_json_arrays(text: str) -> list:
    """Extract all JSON arrays from text."""
    arrays = []
    bracket_count = 0
    start_pos = None

    for i, char in enumerate(text):
        if char == '[':
            if bracket_count == 0:
                start_pos = i
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            if bracket_count == 0 and start_pos is not None:
                arr_str = text[start_pos:i + 1]
                arrays.append(arr_str)
                start_pos = None

    return arrays


def extract_sentiment_pattern(text: str) -> Optional[Dict]:
    """
    Extract sentiment data using regex pattern matching.
    Useful when JSON is too malformed to parse.
    """
    # Pattern: "review_id":"id123","sentiment":0.5
    pattern = r'"review_id"\s*:\s*"([^"]+)"\s*,?\s*"sentiment"\s*:\s*(-?\d+\.?\d*)'
    matches = re.findall(pattern, text)

    if matches:
        sentiments = []
        for review_id, sentiment_str in matches:
            try:
                sentiments.append({
                    "review_id": review_id,
                    "sentiment": float(sentiment_str)
                })
            except ValueError:
                logger.warning(f"Could not parse sentiment: {sentiment_str}")
                continue

        if sentiments:
            logger.info(f"Extracted {len(sentiments)} sentiments via pattern matching")
            return {"sentiments": sentiments}

    return None


def extract_and_repair_json(text: str, expected_type: str = "object") -> Any:
    """
    Main function to extract and repair JSON from text.

    Args:
        text: Text containing JSON (possibly malformed)
        expected_type: "object" or "array" - expected JSON type

    Returns:
        Parsed JSON object or array

    Raises:
        JSONRepairError: If JSON cannot be extracted/repaired
    """
    if not text or not isinstance(text, str):
        raise JSONRepairError("Empty or invalid input text")

    # Clean up markdown code blocks
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

    # Strategy 1: Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.debug(f"Direct parse failed: {e}")

    # Strategy 2: Try with basic repair
    try:
        repaired = repair_json_string(cleaned)
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        logger.debug(f"Repaired parse failed: {e}")

    # Strategy 3: Extract and parse JSON objects
    if expected_type == "object":
        json_objects = extract_json_objects(cleaned)
        for json_str in json_objects:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try repairing
                try:
                    repaired = repair_json_string(json_str)
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    continue

    # Strategy 4: Extract and parse JSON arrays
    json_arrays = extract_json_arrays(cleaned)
    for json_str in json_arrays:
        try:
            arr = json.loads(json_str)
            if expected_type == "array":
                return arr
            else:
                # Wrap array in object for sentiment data
                return {"sentiments": arr} if isinstance(arr, list) else {}
        except json.JSONDecodeError:
            # Try repairing
            try:
                repaired = repair_json_string(json_str)
                arr = json.loads(repaired)
                if expected_type == "array":
                    return arr
                else:
                    return {"sentiments": arr} if isinstance(arr, list) else {}
            except json.JSONDecodeError:
                continue

    # Strategy 5: Pattern-based extraction (last resort for sentiments)
    if "review_id" in cleaned and "sentiment" in cleaned:
        result = extract_sentiment_pattern(cleaned)
        if result:
            return result

    # All strategies failed
    raise JSONRepairError(f"Could not extract/repair JSON from text: {cleaned[:200]}...")


def safe_extract_json(text: str, expected_type: str = "object", default: Any = None) -> Any:
    """
    Safe version that returns default instead of raising exception.

    Args:
        text: Text containing JSON
        expected_type: "object" or "array"
        default: Default value to return on failure

    Returns:
        Parsed JSON or default value
    """
    try:
        return extract_and_repair_json(text, expected_type)
    except JSONRepairError as e:
        logger.warning(f"JSON extraction failed: {e}")
        return default


# Convenience functions
def extract_sentiment_json(text: str) -> Optional[Dict]:
    """Extract sentiment data from text."""
    return safe_extract_json(text, expected_type="object", default=None)


def extract_themes_json(text: str) -> Optional[list]:
    """Extract themes array from text."""
    result = safe_extract_json(text, expected_type="array", default=None)
    if isinstance(result, dict):
        # Check for themes in various keys
        for key in ["themes", "topics", "labels", "categories"]:
            if key in result and isinstance(result[key], list):
                return result[key]
    elif isinstance(result, list):
        return result
    return None
