"""
llm_client.py
-------------
Central OpenRouter LLM client for the Lead Qualification & Outreach Agent.

OpenRouter exposes an OpenAI-compatible API, so we use the standard `openai`
SDK pointed at the OpenRouter base URL.

Usage
-----
    from llm_client import get_llm_client, get_model, is_llm_enabled

    if is_llm_enabled():
        client = get_llm_client()
        model  = get_model()
        response = client.chat.completions.create(
            model=model,
            messages=[...],
        )

Environment variables
---------------------
  OPENROUTER_API_KEY   — your OpenRouter key (required for LLM mode)
  OPENROUTER_MODEL     — model slug (optional, default: openai/gpt-4o-mini)
"""

from __future__ import annotations

import os
from typing import Final

# ── Constants ────────────────────────────────────────────────────────────────

OPENROUTER_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"
DEFAULT_MODEL: Final[str] = "openai/gpt-4o-mini"

# The placeholder value written in .env.example — not a real key.
_PLACEHOLDER_PREFIX: Final[str] = "sk-or-v1-..."


# ── Public helpers ────────────────────────────────────────────────────────────

def is_llm_enabled() -> bool:
    """
    Return True when a real OpenRouter API key is present in the environment.
    Falls back gracefully to rule-based template mode when False.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    return bool(key) and key != _PLACEHOLDER_PREFIX and len(key) > 20


def get_model() -> str:
    """
    Return the model slug to use for LLM calls.
    Reads OPENROUTER_MODEL from the environment; falls back to DEFAULT_MODEL.
    """
    return os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def get_llm_client():  # -> openai.OpenAI
    """
    Return a configured OpenAI-SDK client pointed at OpenRouter.

    Raises
    ------
    ImportError
        If the `openai` package is not installed.
    RuntimeError
        If no valid OPENROUTER_API_KEY is set.
    """
    try:
        from openai import OpenAI  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required for LLM mode. "
            "Install it with: pip install openai==2.44.0"
        ) from exc

    if not is_llm_enabled():
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set or is still the placeholder value. "
            "Set a real key in your .env file to enable LLM drafting."
        )

    api_key = os.environ["OPENROUTER_API_KEY"].strip()

    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            # OpenRouter recommends these headers for attribution / rate-limit
            # identification.  They are optional but good practice.
            "HTTP-Referer": "https://github.com/lead-qualification-agent",
            "X-Title": "Lead Qualification & Outreach Agent",
        },
    )
