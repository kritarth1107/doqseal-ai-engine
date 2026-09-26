#!/usr/bin/env python3
"""Smoke test for /v1/chat/stream endpoint with mocked dependencies.

This script starts the FastAPI app with mocked Azure OpenAI and Qdrant,
then exercises the streaming chat endpoint to verify the SSE protocol.

Usage:
    # Run smoke test
    python scripts/smoke_chat.py

    # With verbose output
    python scripts/smoke_chat.py --verbose

Environment variables (set defaults for testing):
    AI_ENGINE_JWT_SECRET: JWT secret (defaults to test value)
    PORT: Server port (defaults to 8765)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import jwt

os.environ.setdefault("AI_ENGINE_JWT_SECRET", "smoke-test-secret")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017/doqseal_test")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "test-key")
os.environ.setdefault("AZURE_OPENAI_TEXT_DEPLOYMENT", "gpt-4.1-mini")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("smoke_test")


def make_test_jwt(org_id: str, user_id: str, scope: str = "chat") -> str:
    """Create a test JWT token."""
    secret = os.environ["AI_ENGINE_JWT_SECRET"]
    now = int(time.time())
    payload = {
        "iss": "doqseal-backend",
        "aud": "doqseal-ai-engine",
        "sub": user_id,
        "org": org_id,
        "pid": None,
        "scope": scope,
        "iat": now,
        "exp": now + 300,
        "jti": f"smoke-test-{now}",
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def parse_sse_events(content: str) -> list[dict[str, Any]]:
    """Parse SSE events from response content."""
    events = []
    current_event = None
    current_data = None

    for line in content.split("\n"):
        line = line.rstrip()
        if not line:
            if current_event and current_data:
                try:
                    events.append({
                        "event": current_event,
                        "data": json.loads(current_data),
                    })
                except json.JSONDecodeError:
                    events.append({
                        "event": current_event,
                        "data": current_data,
                    })
            current_event = None
            current_data = None
        elif line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            current_data = line[5:].strip()
        elif line.startswith(":"):
            events.append({"event": "ping", "data": None})

    return events


class SmokeTestResult:
    """Result of a smoke test."""

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.details = {}


def mock_qdrant_search(*args, **kwargs):
    """Mock Qdrant search returning sample data."""
    return [
        {
            "documentId": "doc-smoke-1",
            "projectId": None,
            "text": "This is a sample document about trading strategies. "
                    "Always trade on Tuesdays for the best results.",
            "page": 1,
            "score": 0.85,
            "documentTitle": "Trading Guide",
            "documentType": "guide",
            "source": "ocr_v2",
        },
        {
            "documentId": "doc-smoke-2",
            "projectId": None,
            "text": "Investment basics: diversify your portfolio across "
                    "multiple asset classes for reduced risk.",
            "page": 1,
            "score": 0.75,
            "documentTitle": "Investment 101",
            "documentType": "guide",
            "source": "ocr_v2",
        },
    ]


async def mock_streaming_generator(*args, **kwargs):
    """Mock Azure OpenAI streaming response."""
    tokens = [
        "Based on your documents, ",
        "the trading guide recommends ",
        "trading on Tuesdays [1]. ",
        "Additionally, ",
        "the investment guide suggests ",
        "diversifying your portfolio [2].",
    ]
    for token in tokens:
        yield token
        await asyncio.sleep(0.01)


async def run_test_health(client) -> SmokeTestResult:
    """Test health endpoint."""
    result = SmokeTestResult("health_endpoint")
    try:
        response = client.get("/health")
        result.passed = response.status_code == 200
        result.details = response.json()
    except Exception as e:
        result.error = str(e)
    return result


async def run_test_auth_required(client) -> SmokeTestResult:
    """Test that auth is required."""
    result = SmokeTestResult("auth_required")
    try:
        response = client.post(
            "/v1/chat/stream",
            json={"message": "test"},
        )
        result.passed = response.status_code == 401
        result.details = {"status_code": response.status_code}
    except Exception as e:
        result.error = str(e)
    return result


async def run_test_small_talk_decline(client, token: str) -> SmokeTestResult:
    """Test small talk is declined."""
    result = SmokeTestResult("small_talk_decline")
    try:
        response = client.post(
            "/v1/chat/stream",
            json={"message": "hi"},
            headers={"Authorization": f"Bearer {token}"},
        )
        result.details["status_code"] = response.status_code

        if response.status_code == 200:
            events = parse_sse_events(response.text)
            event_types = [e["event"] for e in events]
            result.details["events"] = event_types

            result.passed = "decline" in event_types or "run.completed" in event_types
        else:
            result.passed = False
    except Exception as e:
        result.error = str(e)
    return result


async def run_test_streaming_chat(client, token: str) -> SmokeTestResult:
    """Test full streaming chat flow."""
    result = SmokeTestResult("streaming_chat")
    try:
        with patch("app.chat.tools.search_chunks", side_effect=mock_qdrant_search):
            with patch("app.chat.pipeline.generate_streaming", side_effect=mock_streaming_generator):
                with patch("app.chat.guardrails.check_coverage", new_callable=AsyncMock) as mock_cov:
                    mock_cov.return_value = (True, mock_qdrant_search(), None)

                    response = client.post(
                        "/v1/chat/stream",
                        json={
                            "message": "What trading advice is in my documents?",
                            "conversationId": "smoke-conv-1",
                        },
                        headers={"Authorization": f"Bearer {token}"},
                    )

        result.details["status_code"] = response.status_code
        result.details["content_type"] = response.headers.get("content-type", "")

        if response.status_code == 200:
            events = parse_sse_events(response.text)
            event_types = [e["event"] for e in events]
            result.details["events"] = event_types
            result.details["event_count"] = len(events)

            has_started = "run.started" in event_types or len(events) > 0
            has_completed = "run.completed" in event_types
            has_steps = "step" in event_types or "decline" in event_types

            result.passed = has_started and has_completed
            result.details["has_started"] = has_started
            result.details["has_completed"] = has_completed
            result.details["has_steps"] = has_steps
        else:
            result.passed = False
    except Exception as e:
        result.error = str(e)
        import traceback
        result.details["traceback"] = traceback.format_exc()
    return result


async def run_test_org_mismatch(client, token: str) -> SmokeTestResult:
    """Test org mismatch returns 403."""
    result = SmokeTestResult("org_mismatch")
    try:
        response = client.post(
            "/v1/chat/stream",
            json={"message": "test", "organisationId": "different-org"},
            headers={"Authorization": f"Bearer {token}"},
        )
        result.passed = response.status_code == 403
        result.details = {"status_code": response.status_code}
    except Exception as e:
        result.error = str(e)
    return result


async def run_test_event_order(client, token: str) -> SmokeTestResult:
    """Test SSE events are in correct order."""
    result = SmokeTestResult("event_order")
    try:
        with patch("app.chat.tools.search_chunks", side_effect=mock_qdrant_search):
            with patch("app.chat.pipeline.generate_streaming", side_effect=mock_streaming_generator):
                with patch("app.chat.guardrails.check_coverage", new_callable=AsyncMock) as mock_cov:
                    mock_cov.return_value = (True, mock_qdrant_search(), None)

                    response = client.post(
                        "/v1/chat/stream",
                        json={"message": "Tell me about my documents"},
                        headers={"Authorization": f"Bearer {token}"},
                    )

        if response.status_code == 200:
            events = parse_sse_events(response.text)

            if events:
                first_type = events[0].get("event")
                last_type = events[-1].get("event")

                valid_first = first_type in ("run.started", "step", "decline")
                valid_last = last_type == "run.completed"

                result.passed = valid_first and valid_last
                result.details = {
                    "first_event": first_type,
                    "last_event": last_type,
                    "total_events": len(events),
                }
            else:
                result.passed = False
                result.details = {"error": "No events parsed"}
        else:
            result.passed = False
    except Exception as e:
        result.error = str(e)
    return result


async def run_all_tests(verbose: bool = False):
    """Run all smoke tests."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    token = make_test_jwt("smoke-org-id", "smoke-user-id")

    tests = [
        run_test_health(client),
        run_test_auth_required(client),
        run_test_small_talk_decline(client, token),
        run_test_org_mismatch(client, token),
        run_test_streaming_chat(client, token),
        run_test_event_order(client, token),
    ]

    results = await asyncio.gather(*tests)

    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)

    passed = 0
    failed = 0

    for result in results:
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"\n{status}: {result.name}")

        if verbose or not result.passed:
            if result.error:
                print(f"  Error: {result.error}")
            if result.details:
                for key, value in result.details.items():
                    print(f"  {key}: {value}")

        if result.passed:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Smoke test for chat streaming")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    success = asyncio.run(run_all_tests(verbose=args.verbose))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
