"""
common/llm_client.py
Shared LLM clients:
  - call_llm / stream_llm  — Theta EdgeCloud (Llama) — used for query rewriting only
  - call_gpt / stream_gpt  — Azure OpenAI (GPT-4o)   — used for answer generation and routing
"""

import json
import logging
import os
from typing import Generator

import requests
from openai import AzureOpenAI, OpenAI

logger = logging.getLogger(__name__)


def call_llm(
    messages: list,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> str:
    """
    Call Theta EdgeCloud's streaming SSE endpoint.

    Args:
        messages:    OpenAI-style list of {role, content} dicts.
        max_tokens:  Maximum tokens to generate.
        temperature: Sampling temperature (0.0 for deterministic extraction).

    Returns:
        Accumulated response string, or "[LLM error: ...]" on failure.
    """
    url = os.getenv("THETA_API_URL", "")
    key = os.getenv("THETA_API_KEY", "")

    if not url or not key:
        logger.error("THETA_API_URL or THETA_API_KEY not set.")
        return "[LLM unavailable: credentials not configured]"

    try:
        resp = requests.post(
            url,
            json={
                "input": {
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            },
            headers={"Authorization": f"Bearer {key}"},
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()

        content: list[str] = []
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            data = raw_line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                piece = chunk["choices"][0].get("delta", {}).get("content") or ""
                if piece:
                    content.append(piece)
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

        return "".join(content)

    except Exception as exc:
        logger.error(f"LLM call failed: {exc}")
        return f"[LLM error: {exc}]"


def stream_llm(
    messages: list,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> Generator[str, None, None]:
    """
    Same SSE endpoint as call_llm but yields each token piece instead of buffering.
    Used by Streamlit via st.write_stream() for perceived-latency improvement.

    Yields:
        Individual text pieces as they arrive from the SSE stream.
        On error, yields a single error string.
    """
    url = os.getenv("THETA_API_URL", "")
    key = os.getenv("THETA_API_KEY", "")

    if not url or not key:
        logger.error("THETA_API_URL or THETA_API_KEY not set.")
        yield "[LLM unavailable: credentials not configured]"
        return

    try:
        resp = requests.post(
            url,
            json={
                "input": {
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            },
            headers={"Authorization": f"Bearer {key}"},
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()

        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data:"):
                continue
            data = raw_line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                piece = chunk["choices"][0].get("delta", {}).get("content") or ""
                if piece:
                    yield piece
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    except Exception as exc:
        logger.error(f"LLM stream failed: {exc}")
        yield f"[LLM error: {exc}]"


# ── Azure OpenAI (GPT-4o) ─────────────────────────────────────────────────────

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
    """
    Call Azure OpenAI (GPT-4o) with a list of messages.
    Used for answer generation across all domain pipelines and for routing.

    Returns:
        Response string, or "[GPT error: ...]" on failure.
    """
    if not _gpt_credentials_configured():
        logger.error("LLM credentials are not configured for provider %s.", _provider())
        return "[GPT unavailable: credentials not configured]"

    try:
        client   = _gpt_client()
        response = client.chat.completions.create(
            model=_gpt_deployment(),
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.error(f"GPT call failed: {exc}")
        return f"[GPT error: {exc}]"


def stream_gpt(
    messages: list,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> Generator[str, None, None]:
    """
    Stream tokens from Azure OpenAI (GPT-4o).
    Used by Streamlit via st.write_stream() for perceived-latency improvement.

    Yields:
        Individual text pieces as they arrive.
        On error, yields a single error string.
    """
    if not _gpt_credentials_configured():
        logger.error("LLM credentials are not configured for provider %s.", _provider())
        yield "[GPT unavailable: credentials not configured]"
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
        logger.error(f"GPT stream failed: {exc}")
        yield f"[GPT error: {exc}]"
