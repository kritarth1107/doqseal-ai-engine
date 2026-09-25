"""Pytest fixtures and configuration."""

import os
import time
from typing import Any
from unittest.mock import MagicMock, patch

import jwt
import pytest
from fastapi.testclient import TestClient

os.environ["AI_ENGINE_JWT_SECRET"] = "test-secret-key-for-testing-only"
os.environ["MONGODB_URI"] = "mongodb://localhost:27017/doqseal_test"
os.environ["QDRANT_URL"] = "http://localhost:6333"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com"
os.environ["AZURE_OPENAI_API_KEY"] = "test-key"
os.environ["AZURE_OPENAI_TEXT_DEPLOYMENT"] = "gpt-4.1-mini"

from app.auth import JWT_ALGORITHM, JWT_AUDIENCE, JWT_ISSUER
from app.main import app


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def jwt_secret():
    """Return the test JWT secret."""
    return "test-secret-key-for-testing-only"


def make_jwt(
    org_id: str,
    user_id: str,
    scope: str = "chat",
    *,
    secret: str = "test-secret-key-for-testing-only",
    project_id: str | None = None,
    expired: bool = False,
    wrong_issuer: bool = False,
    wrong_audience: bool = False,
) -> str:
    """Create a test JWT token."""
    now = int(time.time())
    payload = {
        "iss": "wrong-issuer" if wrong_issuer else JWT_ISSUER,
        "aud": "wrong-audience" if wrong_audience else JWT_AUDIENCE,
        "sub": user_id,
        "org": org_id,
        "pid": project_id,
        "scope": scope,
        "iat": now - 3600 if expired else now,
        "exp": now - 60 if expired else now + 300,
        "jti": f"test-jti-{now}",
    }
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


@pytest.fixture
def org_a_token():
    """JWT token for organisation A."""
    return make_jwt("org-a-id", "user-a-id")


@pytest.fixture
def org_b_token():
    """JWT token for organisation B."""
    return make_jwt("org-b-id", "user-b-id")


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant responses."""
    with patch("app.chat.tools.httpx.Client") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance

        mock_instance.get.return_value = MagicMock(status_code=200, json=lambda: {})

        mock_instance.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"result": []},
        )
        mock_instance.post.return_value.raise_for_status = MagicMock()

        yield mock_instance


@pytest.fixture
def mock_qdrant_with_data():
    """Mock Qdrant with sample data for org A."""

    def create_mock(org_id: str):
        sample_points = []
        if org_id == "org-a-id":
            sample_points = [
                {
                    "id": "point-1",
                    "score": 0.85,
                    "payload": {
                        "organisationId": "org-a-id",
                        "documentId": "doc-a-1",
                        "text": "Trading tips: Always trade on Tuesdays for best results.",
                        "documentTitle": "Trading Guide",
                        "page": 1,
                        "deletedAt": None,
                        "sharedWithOrganisation": True,
                    },
                },
                {
                    "id": "point-2",
                    "score": 0.75,
                    "payload": {
                        "organisationId": "org-a-id",
                        "documentId": "doc-a-2",
                        "text": "Investment basics: diversify your portfolio.",
                        "documentTitle": "Investment 101",
                        "page": 2,
                        "deletedAt": None,
                        "sharedWithOrganisation": True,
                    },
                },
            ]

        with patch("app.chat.tools.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance

            mock_instance.get.return_value = MagicMock(status_code=200)

            def mock_post(url, **kwargs):
                response = MagicMock()
                response.status_code = 200

                body = kwargs.get("json", {})
                filter_dict = body.get("filter", {})
                must_conditions = filter_dict.get("must", [])

                result_org = None
                for cond in must_conditions:
                    if cond.get("key") == "organisationId":
                        result_org = cond.get("match", {}).get("value")
                        break

                if result_org == "org-a-id":
                    response.json = lambda: {"result": sample_points}
                else:
                    response.json = lambda: {"result": []}

                response.raise_for_status = MagicMock()
                return response

            mock_instance.post.side_effect = mock_post

            return mock_instance

    return create_mock


@pytest.fixture
def mock_mongodb():
    """Mock MongoDB responses."""
    with patch("app.db.mongo.MongoClient") as mock_client:
        mock_db = MagicMock()
        mock_client.return_value.get_default_database.return_value = mock_db

        mock_db.documents.find.return_value.sort.return_value.limit.return_value = iter([])
        mock_db.extractions.find.return_value = iter([])
        mock_db.extractions.find_one.return_value = None
        mock_db.organisations.find_one.return_value = None

        yield mock_db


@pytest.fixture
def mock_azure_openai():
    """Mock Azure OpenAI responses."""
    with patch("app.chat.pipeline.httpx.AsyncClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value.__aenter__.return_value = mock_instance

        async def mock_post(*args, **kwargs):
            response = MagicMock()
            response.status_code = 200
            response.json = lambda: {
                "choices": [{"message": {"content": "Test response [1]."}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            }
            response.raise_for_status = MagicMock()
            return response

        mock_instance.post = mock_post

        yield mock_instance


@pytest.fixture
def mock_azure_openai_streaming():
    """Mock Azure OpenAI streaming responses."""

    async def mock_stream(*args, **kwargs):
        class MockResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            async def aiter_lines(self):
                yield 'data: {"choices":[{"delta":{"content":"Test "}}]}'
                yield 'data: {"choices":[{"delta":{"content":"answer "}}]}'
                yield 'data: {"choices":[{"delta":{"content":"[1]."}}]}'
                yield "data: [DONE]"

        class MockStream:
            async def __aenter__(self):
                return MockResponse()

            async def __aexit__(self, *args):
                pass

        return MockStream()

    with patch("app.chat.pipeline.httpx.AsyncClient") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value.__aenter__.return_value = mock_instance
        mock_instance.stream = mock_stream

        yield mock_instance
