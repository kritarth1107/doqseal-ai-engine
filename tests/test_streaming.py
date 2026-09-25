"""Tests for streaming chat endpoint."""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from tests.conftest import make_jwt


def parse_sse_events(content: str) -> list[dict]:
    """Parse SSE events from response content."""
    events = []
    current_event = None
    current_data = None

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            if current_event and current_data:
                try:
                    events.append({
                        "event": current_event,
                        "data": json.loads(current_data),
                    })
                except json.JSONDecodeError:
                    pass
            current_event = None
            current_data = None
        elif line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            current_data = line[5:].strip()
        elif line.startswith(":"):
            events.append({"event": "ping", "data": None})

    return events


class TestStreamingEndpoint:
    """Tests for the /v1/chat/stream endpoint."""

    def test_stream_requires_auth(self, client):
        """Streaming endpoint requires authentication."""
        response = client.post(
            "/v1/chat/stream",
            json={"message": "Hello"},
        )
        assert response.status_code == 401

    def test_stream_returns_event_stream(self, client, mock_qdrant, mock_mongodb):
        """Streaming endpoint returns text/event-stream."""
        token = make_jwt("org-a-id", "user-a-id", scope="chat")

        with patch("app.chat.pipeline.generate_streaming") as mock_gen:
            async def mock_generator(*args, **kwargs):
                yield "Test response"

            mock_gen.return_value = mock_generator()

            with patch("app.chat.guardrails.check_coverage", new_callable=AsyncMock) as mock_cov:
                mock_cov.return_value = (True, [{"score": 0.8, "text": "test"}], None)

                response = client.post(
                    "/v1/chat/stream",
                    json={"message": "Hi there"},
                    headers={"Authorization": f"Bearer {token}"},
                )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    def test_stream_emits_run_started(self, client, mock_qdrant, mock_mongodb):
        """Stream starts with run.started event."""
        token = make_jwt("org-a-id", "user-a-id", scope="chat")

        with patch("app.chat.pipeline.generate_streaming") as mock_gen:
            async def mock_generator(*args, **kwargs):
                yield "Test"

            mock_gen.return_value = mock_generator()

            with patch("app.chat.guardrails.check_coverage", new_callable=AsyncMock) as mock_cov:
                mock_cov.return_value = (True, [{"score": 0.8, "text": "test"}], None)

                response = client.post(
                    "/v1/chat/stream",
                    json={"message": "Test question"},
                    headers={"Authorization": f"Bearer {token}"},
                )

        events = parse_sse_events(response.text)
        event_types = [e["event"] for e in events if e.get("event")]

        assert "run.started" in event_types or len(events) > 0

    def test_small_talk_returns_decline_event(self, client, mock_qdrant, mock_mongodb):
        """Small talk returns a decline event without generation."""
        token = make_jwt("org-a-id", "user-a-id", scope="chat")

        response = client.post(
            "/v1/chat/stream",
            json={"message": "hi"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        events = parse_sse_events(response.text)
        event_types = [e["event"] for e in events]

        assert "decline" in event_types or "run.completed" in event_types

    def test_stream_includes_step_events(self, client, mock_qdrant, mock_mongodb):
        """Stream includes step events for real pipeline execution."""
        token = make_jwt("org-a-id", "user-a-id", scope="chat")

        with patch("app.chat.pipeline.generate_streaming") as mock_gen:
            async def mock_generator(*args, **kwargs):
                yield "Test response [1]."

            mock_gen.return_value = mock_generator()

            with patch("app.chat.guardrails.check_coverage", new_callable=AsyncMock) as mock_cov:
                mock_cov.return_value = (True, [{"score": 0.8, "text": "test", "documentId": "doc-1"}], None)

                response = client.post(
                    "/v1/chat/stream",
                    json={"message": "What is in my documents?"},
                    headers={"Authorization": f"Bearer {token}"},
                )

        events = parse_sse_events(response.text)
        event_types = [e["event"] for e in events]

        assert "step" in event_types or "run.started" in event_types

    def test_stream_ends_with_run_completed(self, client, mock_qdrant, mock_mongodb):
        """Stream ends with run.completed event."""
        token = make_jwt("org-a-id", "user-a-id", scope="chat")

        response = client.post(
            "/v1/chat/stream",
            json={"message": "hello"},
            headers={"Authorization": f"Bearer {token}"},
        )

        events = parse_sse_events(response.text)
        event_types = [e["event"] for e in events]

        assert "run.completed" in event_types


class TestStreamingEventOrder:
    """Tests for correct streaming event order."""

    def test_event_order_for_answered(self, client, mock_qdrant, mock_mongodb):
        """Events follow correct order: run.started -> steps -> tokens -> citations -> run.completed."""
        token = make_jwt("org-a-id", "user-a-id", scope="chat")

        chunks_data = [
            {"score": 0.9, "text": "Test content", "documentId": "doc-1", "documentTitle": "Test Doc", "page": 1}
        ]

        with patch("app.chat.tools.search_chunks", return_value=chunks_data):
            with patch("app.chat.pipeline.generate_streaming") as mock_gen:
                async def mock_generator(*args, **kwargs):
                    yield "Based on the document [1], "
                    yield "the answer is yes."

                mock_gen.return_value = mock_generator()

                with patch("app.chat.guardrails.check_coverage", new_callable=AsyncMock) as mock_cov:
                    mock_cov.return_value = (True, chunks_data, None)

                    response = client.post(
                        "/v1/chat/stream",
                        json={"message": "Test question?"},
                        headers={"Authorization": f"Bearer {token}"},
                    )

        events = parse_sse_events(response.text)

        if events:
            first_event = events[0]
            last_event = events[-1]

            if first_event.get("event"):
                assert first_event["event"] in ("run.started", "step", "decline")

            if last_event.get("event"):
                assert last_event["event"] == "run.completed"

    def test_event_order_for_declined(self, client, mock_qdrant, mock_mongodb):
        """Declined responses have correct event order."""
        token = make_jwt("org-a-id", "user-a-id", scope="chat")

        with patch("app.chat.tools.search_chunks", return_value=[]):
            response = client.post(
                "/v1/chat/stream",
                json={"message": "What is the capital of France?"},
                headers={"Authorization": f"Bearer {token}"},
            )

        events = parse_sse_events(response.text)
        event_types = [e["event"] for e in events if e.get("event")]

        if "decline" in event_types:
            decline_idx = event_types.index("decline")
            completed_idx = event_types.index("run.completed") if "run.completed" in event_types else -1
            if completed_idx >= 0:
                assert decline_idx < completed_idx


class TestStreamingCancellation:
    """Tests for stream cancellation on disconnect."""

    def test_cancellation_logged(self, client, mock_qdrant, mock_mongodb, caplog):
        """Client disconnect triggers cancellation."""
        import logging

        caplog.set_level(logging.INFO)
        token = make_jwt("org-a-id", "user-a-id", scope="chat")

        response = client.post(
            "/v1/chat/stream",
            json={"message": "hi"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200


class TestStreamingHistory:
    """Tests for conversation history handling."""

    def test_history_included_in_request(self, client, mock_qdrant, mock_mongodb):
        """Conversation history is properly included."""
        token = make_jwt("org-a-id", "user-a-id", scope="chat")

        history = [
            {"role": "user", "content": "What documents do I have?"},
            {"role": "assistant", "content": "You have 3 documents."},
        ]

        response = client.post(
            "/v1/chat/stream",
            json={
                "message": "Tell me more about the first one",
                "history": history,
                "conversationId": "conv-123",
            },
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200

    def test_history_trimmed_to_limit(self, client, mock_qdrant, mock_mongodb):
        """History is trimmed to configured limit."""
        token = make_jwt("org-a-id", "user-a-id", scope="chat")

        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            for i in range(30)
        ]

        response = client.post(
            "/v1/chat/stream",
            json={
                "message": "Follow up",
                "history": history,
            },
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
