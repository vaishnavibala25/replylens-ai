"""
llm_service.py
---------------
Thin wrapper around an OpenAI-compatible chat completions endpoint, so the
provider (OpenAI, OpenRouter, a local server, etc.) is swappable purely
through environment variables:

    LLM_API_KEY
    LLM_BASE_URL   (e.g. https://api.openai.com/v1 or https://openrouter.ai/api/v1)
    LLM_MODEL      (e.g. gpt-4o-mini, anthropic/claude-3.5-sonnet, etc.)

If no API key is configured, the service reports itself as unavailable and
the rest of the app falls back to Demo Mode rather than crashing.
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
import requests

load_dotenv()


@dataclass
class LLMResponse:
    text: str
    ok: bool
    error: Optional[str] = None


class LLMService:
    def __init__(self):
        self.api_key = _config_value("LLM_API_KEY")
        self.base_url = _config_value("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = _config_value("LLM_MODEL", "gpt-4o-mini")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def configuration_summary(self) -> dict[str, str | bool]:
        """Return safe configuration details without exposing the API key."""
        return {
            "configured": self.is_configured,
            "model": self.model,
            "base_url": self.base_url,
        }

    def chat(self, system_prompt: str, user_prompt: str,
              temperature: float = 0.3, max_tokens: int = 700) -> LLMResponse:
        """Call the chat completions endpoint. Never raises; returns LLMResponse(ok=False, ...) on failure."""
        if not self.is_configured:
            return LLMResponse(text="", ok=False, error="LLM_API_KEY is not set.")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return LLMResponse(text=content, ok=True)
        except requests.exceptions.RequestException as e:
            return LLMResponse(text="", ok=False, error=f"Request error: {e}")
        except (KeyError, IndexError, ValueError) as e:
            return LLMResponse(text="", ok=False, error=f"Unexpected response shape: {e}")


def _streamlit_secret(name: str) -> str:
    """Read one Streamlit secret without requiring Streamlit runtime context."""
    try:
        import streamlit as st
        return str(st.secrets.get(name, "")).strip()
    except (FileNotFoundError, KeyError, RuntimeError, TypeError):
        return ""


def _config_value(name: str, default: str = "") -> str:
    """Prefer process/.env values, then fall back to Streamlit Cloud secrets."""
    return os.getenv(name, "").strip() or _streamlit_secret(name) or default


if __name__ == "__main__":
    print(LLMService().configuration_summary())
