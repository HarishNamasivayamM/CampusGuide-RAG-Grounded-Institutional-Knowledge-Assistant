"""Shared chat-provider clients.

Groq is the default provider through its OpenAI-compatible API. Azure OpenAI
is retained as an optional provider for deployments that already use it.
"""

import logging
import os
from typing import Generator

from openai import AzureOpenAI, OpenAI

logger = logging.getLogger(__name__)


def _provider() -> str:
    """Return the configured chat provider, defaulting to Groq when present."""
    return os.getenv("LLM_PROVIDER", "groq" if os.getenv("GROQ_API_KEY") else "azure").lower()


def _gpt_client() -> OpenAI | AzureOpenAI:
    """Build the configured OpenAI-compatible chat client."""
    if _provider() == "groq":
        return OpenAI(
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=os.getenv("GROQ_API_KEY", ""),
        )

    return AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_KEY", ""),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
    )


def _gpt_deployment() -> str:
    if _provider() == "groq":
        return os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


def _gpt_credentials_configured() -> bool:
    if _provider() == "groq":
        return bool(os.getenv("GROQ_API_KEY"))
    return bool(os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_KEY"))


def call_gpt(
    messages: list,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> str:
    """Call the configured chat provider for one buffered response."""
    if not _gpt_credentials_configured():
        logger.error("LLM credentials are not configured for provider %s.", _provider())
        return "[LLM unavailable: credentials not configured]"

    try:
        client = _gpt_client()
        response = client.chat.completions.create(
            model=_gpt_deployment(),
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return f"[LLM error: {exc}]"


def stream_gpt(
    messages: list,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> Generator[str, None, None]:
    """Stream response text from the configured chat provider."""
    if not _gpt_credentials_configured():
        logger.error("LLM credentials are not configured for provider %s.", _provider())
        yield "[LLM unavailable: credentials not configured]"
        return

    try:
        client = _gpt_client()
        stream = client.chat.completions.create(
            model=_gpt_deployment(),
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as exc:
        logger.error("LLM stream failed: %s", exc)
        yield f"[LLM error: {exc}]"
