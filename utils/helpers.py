"""Small shared utilities: text normalization and tolerant JSON parsing."""

import json
import re
from typing import Any, Optional


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace. Used before retrieval/objective checks."""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_json_block(raw: str) -> Optional[str]:
    """
    LLMs sometimes wrap JSON in markdown fences or add stray text.
    Pull out the first {...} block we can find.
    """
    if not raw:
        return None
    raw = raw.strip()
    # strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()

    # find the outermost matching braces
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]
    return None


def safe_json_parse(raw: str) -> Optional[dict[str, Any]]:
    """Best-effort parse of a JSON object from raw LLM text. Returns None on failure."""
    block = extract_json_block(raw)
    if block is None:
        return None
    try:
        return json.loads(block)
    except json.JSONDecodeError:
        # one common failure mode: trailing commas
        try:
            cleaned = re.sub(r",\s*([}\]])", r"\1", block)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def extract_numbers(text: str) -> set[str]:
    """Extract standalone numeric tokens (prices, quantities, dates) for consistency checks."""
    if not text:
        return set()
    return set(re.findall(r"\$?\d[\d,]*(?:\.\d+)?%?", text))


def extract_order_ids(text: str) -> set[str]:
    if not text:
        return set()
    return set(re.findall(r"\bORD-\d+\b", text, re.IGNORECASE))
