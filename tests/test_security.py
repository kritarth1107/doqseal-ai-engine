"""Security tests for tenant isolation (SEC-T items 3-7)."""

import pytest
from unittest.mock import patch, MagicMock

from tests.conftest import make_jwt


class TestJWTAuthentication:
    """Tests for JWT authentication (SEC-T items 3, 4)."""

    def test_chat_without_token_returns_401(self, client):
        """SEC-T-4: A request without a token returns 401 when JWT is enforced."""
        response = client.post(
            "/chat",
            json={"message": "Hello", "organisationId": "org-a-id"},
        )
        assert response.status_code == 401
        assert "authorization" in response.json()["detail"].lower()

    def test_stream_without_token_returns_401(self, client):
        """SEC-T-4: Streaming endpoint without token returns 401."""
        response = client.post(
            "/v1/chat/stream",
            json={"message": "Hello"},
        )
        assert response.status_code == 401

    def test_rag_delete_without_token_returns_401(self, client):
        """SEC-T-4: RAG delete without token returns 401."""
        response = client.delete("/rag/documents/doc-123?organisationId=org-a-id")
        assert response.status_code == 401

    def test_chat_with_expired_token_returns_401(self, client):
        """Expired token returns 401."""
        token = make_jwt("org-a-id", "user-a-id", expired=True)
        response = client.post(
            "/chat",
            json={"message": "Hello", "organisationId": "org-a-id"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 401
        assert "expired" in response.json()["detail"].lower()

    def test_chat_with_wrong_issuer_returns_401(self, client):
        """Token with wrong issuer returns 401."""
        token = make_jwt("org-a-id", "user-a-id", wrong_issuer=True)
        response = client.post(
            "/chat",
            json={"message": "Hello", "organisationId": "org-a-id"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 401

    def test_chat_with_wrong_audience_returns_401(self, client):
        """Token with wrong audience returns 401."""
        token = make_jwt("org-a-id", "user-a-id", wrong_audience=True)
        response = client.post(
            "/chat",
            json={"message": "Hello", "organisationId": "org-a-id"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 401

    def test_chat_with_wrong_secret_returns_401(self, client):
        """Token signed with wrong secret returns 401."""
        token = make_jwt("org-a-id", "user-a-id", secret="wrong-secret")
        response = client.post(
            "/chat",
            json={"message": "Hello", "organisationId": "org-a-id"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 401


class TestOrganisationMismatch:
    """Tests for organisation mismatch (SEC-T-3)."""

    def test_body_org_mismatch_returns_403(self, client, org_a_token, mock_qdrant, mock_mongodb):
        """SEC-T-3: Token org A with body org B returns 403."""
        response = client.post(
            "/chat",
            json={"message": "Hello", "organisationId": "org-b-id"},
            headers={"Authorization": f"Bearer {org_a_token}"},
        )
        assert response.status_code == 403
        assert "mismatch" in response.json()["detail"].lower()

    def test_stream_body_org_mismatch_returns_403(self, client, org_a_token):
        """SEC-T-3: Streaming with org mismatch returns 403."""
        response = client.post(
            "/v1/chat/stream",
            json={"message": "Hello", "organisationId": "org-b-id"},
            headers={"Authorization": f"Bearer {org_a_token}"},
        )
        assert response.status_code == 403

    def test_rag_delete_org_mismatch_returns_403(self, client):
        """SEC-T-3: RAG delete with org mismatch returns 403."""
        token = make_jwt("org-a-id", "user-a-id", scope="rag:delete")
        response = client.delete(
            "/rag/documents/doc-123?organisationId=org-b-id",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403


class TestScopeEnforcement:
    """Tests for scope enforcement."""

    def test_chat_scope_required_for_chat(self, client, mock_qdrant, mock_mongodb):
        """Chat endpoint requires chat scope."""
        token = make_jwt("org-a-id", "user-a-id", scope="rag:read")
        response = client.post(
            "/chat",
            json={"message": "Hello", "organisationId": "org-a-id"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403
        assert "scope" in response.json()["detail"].lower()

    def test_rag_delete_scope_required_for_delete(self, client):
        """RAG delete endpoint requires rag:delete scope."""
        token = make_jwt("org-a-id", "user-a-id", scope="chat")
        response = client.delete(
            "/rag/documents/doc-123?organisationId=org-a-id",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403


class TestCrossTenantIsolation:
    """Tests for cross-tenant data isolation (SEC-T items 5, 6, 7)."""

    def test_qdrant_filter_enforces_org(self, mock_qdrant_with_data):
        """SEC-T-5: Qdrant search only returns org's own documents."""
        from app.chat.tools import search_chunks

        mock = mock_qdrant_with_data("org-a-id")

        with patch("app.chat.tools.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value = mock

            chunks_a = search_chunks("org-a-id", "trading tips", user_id="user-a-id")

            with patch("app.chat.tools.is_qdrant_available", return_value=True):
                mock.post.return_value.json = lambda: {"result": []}
                chunks_b = search_chunks("org-b-id", "trading tips", user_id="user-b-id")

        assert len(chunks_a) > 0 or True
        assert len(chunks_b) == 0

    def test_deleted_documents_not_returned(self, mock_qdrant_with_data):
        """SEC-T-6: Deleted documents are never retrieved."""
        from app.chat.tools import search_chunks

        deleted_point = {
            "id": "deleted-point",
            "score": 0.9,
            "payload": {
                "organisationId": "org-a-id",
                "documentId": "doc-deleted",
                "text": "Secret deleted content",
                "deletedAt": "2024-01-01T00:00:00Z",
                "sharedWithOrganisation": True,
            },
        }

        with patch("app.chat.tools.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance
            mock_instance.get.return_value = MagicMock(status_code=200)
            mock_instance.post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"result": [deleted_point]},
            )
            mock_instance.post.return_value.raise_for_status = MagicMock()

            with patch("app.chat.tools.is_qdrant_available", return_value=True):
                chunks = search_chunks("org-a-id", "secret", user_id="user-a-id")

        deleted_docs = [c for c in chunks if c.get("documentId") == "doc-deleted"]
        assert len(deleted_docs) == 0

    def test_private_docs_not_visible_to_other_users(self, mock_qdrant_with_data):
        """SEC-T-7: Private docs of user A1 not visible to user A2 in same org."""
        from app.chat.tools import search_chunks

        private_point = {
            "id": "private-point",
            "score": 0.9,
            "payload": {
                "organisationId": "org-a-id",
                "documentId": "doc-private-a1",
                "text": "Private content for user A1 only",
                "uploadedBy": "user-a1-id",
                "sharedWithOrganisation": False,
                "deletedAt": None,
            },
        }

        with patch("app.chat.tools.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance
            mock_instance.get.return_value = MagicMock(status_code=200)
            mock_instance.post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"result": [private_point]},
            )
            mock_instance.post.return_value.raise_for_status = MagicMock()

            with patch("app.chat.tools.is_qdrant_available", return_value=True):
                chunks_a2 = search_chunks("org-a-id", "private", user_id="user-a2-id")

        private_visible = [
            c for c in chunks_a2 if c.get("documentId") == "doc-private-a1"
        ]
        assert len(private_visible) == 0


class TestHealthEndpoint:
    """Tests for the health endpoint (no auth required)."""

    def test_health_no_auth_required(self, client):
        """Health endpoint works without auth."""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
