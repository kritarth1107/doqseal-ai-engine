"""Async Azure OpenAI calls for chat (the same resource and keys as extraction)."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger("doqseal.chat.llm")


class LLMError(Exception):
    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


def configured() -> bool:
    return bool(
        (settings.azure_openai_endpoint or "").strip()
        and (settings.azure_openai_api_key or "").strip()
        and deployment()
    )


def deployment() -> str:
    return (
        (settings.chat_deployment or "").strip()
        or (settings.azure_openai_text_deployment or "").strip()
        or (settings.azure_openai_deployment or "").strip()
    )


def _supports_temperature(name: str) -> bool:
    # Reasoning deployments (gpt-5*, o-series) only accept the default temperature.
    return not re.match(r"^(gpt-5|o\d)", name.lower())


def _temperature() -> float | None:
    raw = (settings.chat_temperature or "").strip()
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def _url(name: str) -> str:
    endpoint = settings.azure_openai_endpoint.rstrip("/")
    return f"{endpoint}/openai/deployments/{name}/chat/completions" f"?api-version={settings.azure_openai_api_version}"


def _headers() -> dict[str, str]:
    return {"api-key": settings.azure_openai_api_key, "Content-Type": "application/json"}


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=15.0, read=settings.chat_timeout_seconds, write=30.0, pool=15.0)


def _payload(messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
    payload: dict[str, Any] = {"messages": messages, "max_completion_tokens": max_tokens}
    temperature = _temperature()
    if temperature is not None and _supports_temperature(deployment()):
        payload["temperature"] = temperature
    return payload


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    retryable = response.status_code in (408, 429) or response.status_code >= 500
    raise LLMError(f"model call failed with status {response.status_code}", retryable=retryable)


async def complete_json(
    messages: list[dict[str, Any]], *, max_tokens: int | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One JSON-mode completion. Returns (parsed, usage)."""
    if not configured():
        raise LLMError("model is not configured", retryable=False)
    payload = _payload(messages, max_tokens or settings.chat_judge_max_tokens)
    payload["response_format"] = {"type": "json_object"}
    try:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            response = await client.post(_url(deployment()), headers=_headers(), json=payload)
    except httpx.HTTPError as exc:
        raise LLMError(f"model call failed: {type(exc).__name__}") from exc
    _raise_for_status(response)
    body = response.json()
    content = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMError("model returned invalid JSON", retryable=True) from exc
    if not isinstance(parsed, dict):
        raise LLMError("model returned invalid JSON", retryable=True)
    return parsed, body.get("usage") or {}


async def stream_chat(messages: list[dict[str, Any]], *, max_tokens: int) -> AsyncIterator[dict[str, Any]]:
    """Streams {"text": str} deltas, then one {"usage": {...}} item.

    Cancelling the consumer closes the HTTP stream, which stops generation.
    """
    if not configured():
        raise LLMError("model is not configured", retryable=False)
    payload = _payload(messages, max_tokens)
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    try:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            async with client.stream("POST", _url(deployment()), headers=_headers(), json=payload) as response:
                if response.status_code >= 400:
                    await response.aread()
                    _raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        yield {"usage": chunk["usage"]}
                    for choice in chunk.get("choices") or []:
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            yield {"text": text}
    except httpx.HTTPError as exc:
        raise LLMError(f"model stream failed: {type(exc).__name__}") from exc
