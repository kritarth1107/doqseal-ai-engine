"""Azure OpenAI client used by chat: payload shape and stream parsing."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.chat import llm
from app.config import settings


@pytest.fixture()
def azure(monkeypatch):
    monkeypatch.setattr(settings, "azure_openai_endpoint", "https://example.invalid")
    monkeypatch.setattr(settings, "azure_openai_api_key", "k")
    monkeypatch.setattr(settings, "azure_openai_text_deployment", "text-model")
    monkeypatch.setattr(settings, "chat_deployment", "")
    monkeypatch.setattr(settings, "chat_temperature", "")
    seen: list[dict] = []
    real = httpx.AsyncClient

    def install(handler):
        def wrapped(request: httpx.Request) -> httpx.Response:
            seen.append({"url": str(request.url), "body": json.loads(request.content)})
            return handler(request)

        class Client(real):
            def __init__(self, *a, **k):
                super().__init__(*a, transport=httpx.MockTransport(wrapped), **k)

        monkeypatch.setattr(llm.httpx, "AsyncClient", Client)

    return install, seen


def test_json_completion_payload_and_parse(azure):
    install, seen = azure
    install(
        lambda r: httpx.Response(
            200, json={"choices": [{"message": {"content": '{"coverage": "full"}'}}], "usage": {"total_tokens": 9}}
        )
    )
    parsed, usage = asyncio.run(llm.complete_json([{"role": "user", "content": "q"}]))
    assert parsed == {"coverage": "full"} and usage == {"total_tokens": 9}
    body = seen[0]["body"]
    assert "/deployments/text-model/" in seen[0]["url"]
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_completion_tokens"] == settings.chat_judge_max_tokens
    assert "temperature" not in body and "max_tokens" not in body


def test_temperature_only_when_configured_and_supported(azure, monkeypatch):
    install, seen = azure
    install(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]}))
    monkeypatch.setattr(settings, "chat_temperature", "0.2")
    asyncio.run(llm.complete_json([]))
    assert seen[-1]["body"]["temperature"] == 0.2
    monkeypatch.setattr(settings, "chat_deployment", "gpt-5-chat")
    asyncio.run(llm.complete_json([]))
    assert "temperature" not in seen[-1]["body"]


def test_invalid_json_and_http_errors(azure):
    install, _ = azure
    install(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]}))
    with pytest.raises(llm.LLMError):
        asyncio.run(llm.complete_json([]))
    install(lambda r: httpx.Response(429, json={}))
    with pytest.raises(llm.LLMError) as err:
        asyncio.run(llm.complete_json([]))
    assert err.value.retryable
    install(lambda r: httpx.Response(400, json={}))
    with pytest.raises(llm.LLMError) as err:
        asyncio.run(llm.complete_json([]))
    assert not err.value.retryable


def test_stream_parsing(azure):
    install, seen = azure
    lines = [
        'data: {"choices":[{"delta":{"role":"assistant"}}]}',
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        ": keep-alive",
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}',
        "data: [DONE]",
    ]
    install(lambda r: httpx.Response(200, text="\n\n".join(lines) + "\n\n"))

    async def collect():
        return [item async for item in llm.stream_chat([{"role": "user", "content": "x"}], max_tokens=50)]

    items = asyncio.run(collect())
    assert items == [{"text": "Hel"}, {"text": "lo"}, {"usage": {"prompt_tokens": 5, "completion_tokens": 2}}]
    body = seen[0]["body"]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["max_completion_tokens"] == 50


def test_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "azure_openai_endpoint", "")
    with pytest.raises(llm.LLMError):
        asyncio.run(llm.complete_json([]))
