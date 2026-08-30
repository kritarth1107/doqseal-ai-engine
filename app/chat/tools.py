"""Chat tools — Qdrant retrieval and MongoDB document helpers."""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from app.config import settings
from app.db.mongo import get_db
from app.rag.embedder import embed_query

logger = logging.getLogger("doqseal.chat.tools")


def _collection_name(organisation_id: str) -> str:
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", organisation_id)
    return f"org_{safe_id}"


def _qdrant_headers() -> dict[str, str]:
    if settings.qdrant_api_key:
        return {"api-key": settings.qdrant_api_key}
    return {}


def is_qdrant_available() -> bool:
    try:
        with httpx.Client(timeout=5.0, headers=_qdrant_headers()) as client:
            response = client.get(f"{settings.qdrant_url.rstrip('/')}/collections")
            return response.status_code == 200
    except Exception:
        return False


def _collection_exists(client: httpx.Client, organisation_id: str) -> bool:
    response = client.get(
        f"{settings.qdrant_url.rstrip('/')}/collections/{_collection_name(organisation_id)}"
    )
    return response.status_code == 200


def search_documents(
    organisation_id: str,
    query: str,
    *,
    project_id: str | None = None,
    user_id: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve relevant chunks from Qdrant via vector search."""
    if not query.strip():
        return []

    if not is_qdrant_available():
        logger.info("Qdrant unavailable — skipping retrieval")
        return []

    collection = _collection_name(organisation_id)
    base_url = settings.qdrant_url.rstrip("/")

    try:
        vector = embed_query(query)
    except Exception as exc:
        logger.warning("Query embedding failed: %s", exc)
        return []

    try:
        with httpx.Client(timeout=15.0, headers=_qdrant_headers()) as client:
            if not _collection_exists(client, organisation_id):
                logger.info("Qdrant collection %s not found", collection)
                return []

            query_filter: dict[str, Any] | None = None
            if project_id:
                query_filter = {
                    "must": [{"key": "projectId", "match": {"value": project_id}}]
                }

            # Over-fetch then apply visibility in Python (handles legacy payloads)
            fetch_limit = max(limit * 4, 20) if user_id else limit

            response = client.post(
                f"{base_url}/collections/{collection}/points/search",
                json={
                    "vector": vector,
                    "limit": fetch_limit,
                    "with_payload": True,
                    "filter": query_filter,
                },
            )
            response.raise_for_status()
            points = response.json().get("result", [])
    except Exception as exc:
        logger.warning("Qdrant retrieval failed: %s", exc)
        return []

    chunks: list[dict[str, Any]] = []
    for point in points:
        payload = point.get("payload") or {}
        text = payload.get("text") or payload.get("chunk") or payload.get("content")
        if not text:
            continue

        # Visibility: private chunks only for the uploader; missing flag = shared (legacy)
        if user_id:
            shared = payload.get("sharedWithOrganisation")
            uploaded_by = payload.get("uploadedBy")
            if shared is False and uploaded_by and uploaded_by != user_id:
                continue

        chunks.append(
            {
                "documentId": payload.get("documentId"),
                "projectId": payload.get("projectId"),
                "snippet": str(text)[:500],
                "score": point.get("score"),
            }
        )
        if len(chunks) >= limit:
            break

    return chunks


def get_extraction(document_id: str) -> dict[str, Any] | None:
    db = get_db()
    extraction = db.extractions.find_one({"documentId": document_id})
    if not extraction:
        return None
    extraction.pop("_id", None)
    return extraction


def list_project_documents(
    organisation_id: str,
    project_id: str,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    db = get_db()
    cursor = (
        db.documents.find(
            {"organisationId": organisation_id, "projectId": project_id},
            {"_id": 0, "documentId": 1, "originalFilename": 1, "createdAt": 1},
        )
        .sort("createdAt", -1)
        .limit(limit)
    )
    return list(cursor)
