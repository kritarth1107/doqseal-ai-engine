"""Upsert document chunks into per-organisation Qdrant collections."""

from __future__ import annotations

import logging
import re
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import settings
from app.rag.chunker import build_chunks, build_chunks_v2
from app.rag.embedder import embed_passages

logger = logging.getLogger("doqseal.rag")

_client: QdrantClient | None = None
_VECTOR_SIZE = 768


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        kwargs: dict[str, Any] = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        _client = QdrantClient(**kwargs)
    return _client


def _collection_name(organisation_id: str) -> str:
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", organisation_id)
    return f"org_{safe_id}"


def _ensure_collection(organisation_id: str) -> str:
    client = _get_client()
    name = _collection_name(organisation_id)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection %s", name)
    return name


def _point_id(document_id: str, source: str, index: int) -> str:
    return str(uuid5(NAMESPACE_URL, f"{document_id}:{source}:{index}"))


def build_visibility_filter(
    organisation_id: str,
    user_id: str | None,
    *,
    project_id: str | None = None,
    include_deleted: bool = False,
) -> Filter:
    """Build a Qdrant filter enforcing tenant isolation and visibility.

    This filter is MANDATORY for all searches - it ensures:
    1. organisationId matches (defense in depth beyond collection name)
    2. Document not deleted (deletedAt is null/missing)
    3. Visibility: shared with org OR uploaded by this user
    """
    must_conditions = [
        FieldCondition(key="organisationId", match=MatchValue(value=organisation_id)),
    ]

    if not include_deleted:
        must_conditions.append(
            FieldCondition(key="deletedAt", match=MatchValue(value=None)),
        )

    if project_id:
        must_conditions.append(
            FieldCondition(key="projectId", match=MatchValue(value=project_id)),
        )

    should_conditions = []
    if user_id:
        should_conditions = [
            FieldCondition(
                key="sharedWithOrganisation", match=MatchValue(value=True)
            ),
            FieldCondition(key="uploadedBy", match=MatchValue(value=user_id)),
        ]
    else:
        should_conditions = [
            FieldCondition(
                key="sharedWithOrganisation", match=MatchValue(value=True)
            ),
        ]

    return Filter(
        must=must_conditions,
        should=should_conditions if user_id else None,
    )


def index_extraction(
    *,
    organisation_id: str,
    document_id: str,
    job_id: str,
    project_id: str | None,
    ocr_full_text: str | None,
    extraction_data: dict[str, Any] | None,
    uploaded_by: str | None = None,
    shared_with_organisation: bool = True,
    document_title: str | None = None,
    document_type: str | None = None,
    use_v2_chunker: bool = False,
) -> int:
    """Chunk, embed, and upsert extraction content. Returns number of points upserted."""
    if use_v2_chunker:
        chunks = build_chunks_v2(
            ocr_full_text,
            extraction_data,
            document_title=document_title,
            document_type=document_type,
            document_id=document_id,
        )
    else:
        chunks = build_chunks(ocr_full_text, extraction_data)

    if not chunks:
        logger.info("No RAG chunks for document %s", document_id)
        return 0

    vectors = embed_passages([chunk["text"] for chunk in chunks])
    collection_name = _ensure_collection(organisation_id)
    client = _get_client()

    points = []
    for chunk, vector in zip(chunks, vectors):
        payload = {
            "organisationId": organisation_id,
            "documentId": document_id,
            "jobId": job_id,
            "projectId": project_id,
            "uploadedBy": uploaded_by,
            "sharedWithOrganisation": shared_with_organisation,
            "deletedAt": None,
            "source": chunk["source"],
            "chunkIndex": chunk["index"],
            "text": chunk["text"],
        }

        if "page" in chunk:
            payload["page"] = chunk["page"]
        if "documentTitle" in chunk:
            payload["documentTitle"] = chunk["documentTitle"]
        if "documentType" in chunk:
            payload["documentType"] = chunk["documentType"]
        if "field" in chunk:
            payload["field"] = chunk["field"]
            payload["fieldValue"] = chunk.get("value")

        points.append(
            PointStruct(
                id=_point_id(document_id, chunk["source"], chunk["index"]),
                vector=vector,
                payload=payload,
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    logger.info(
        "Indexed %d chunks for document %s into %s",
        len(points),
        document_id,
        collection_name,
    )
    return len(points)


def mark_document_deleted(*, organisation_id: str, document_id: str) -> int:
    """Mark all chunks for a document as deleted (soft delete for RAG).

    Returns number of points updated.
    """
    client = _get_client()
    collection_name = _collection_name(organisation_id)

    if not client.collection_exists(collection_name):
        return 0

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    updated = 0
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="documentId", match=MatchValue(value=document_id)
                    ),
                    FieldCondition(
                        key="organisationId", match=MatchValue(value=organisation_id)
                    ),
                ]
            ),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for record in records:
            payload = dict(record.payload or {})
            payload["deletedAt"] = now
            client.set_payload(
                collection_name=collection_name,
                payload=payload,
                points=[record.id],
            )
            updated += 1

        if next_offset is None:
            break
        offset = next_offset

    logger.info(
        "Marked %d Qdrant chunks as deleted for document %s",
        updated,
        document_id,
    )
    return updated


def delete_document_chunks(*, organisation_id: str, document_id: str) -> int:
    """Remove all Qdrant points for a document. Returns deleted count (best effort)."""
    client = _get_client()
    collection_name = _collection_name(organisation_id)

    if not client.collection_exists(collection_name):
        return 0

    deleted = 0
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="documentId", match=MatchValue(value=document_id)
                    ),
                    FieldCondition(
                        key="organisationId", match=MatchValue(value=organisation_id)
                    ),
                ]
            ),
            limit=100,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )

        point_ids = [record.id for record in records]
        if point_ids:
            client.delete(collection_name=collection_name, points_selector=point_ids)
            deleted += len(point_ids)

        if next_offset is None:
            break
        offset = next_offset

    logger.info(
        "Deleted %d Qdrant chunks for document %s from %s",
        deleted,
        document_id,
        collection_name,
    )
    return deleted


def update_document_visibility(
    *,
    organisation_id: str,
    document_id: str,
    shared_with_organisation: bool,
) -> int:
    """Update visibility flag on all chunks for a document.

    Returns number of points updated.
    """
    client = _get_client()
    collection_name = _collection_name(organisation_id)

    if not client.collection_exists(collection_name):
        return 0

    updated = 0
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="documentId", match=MatchValue(value=document_id)
                    ),
                    FieldCondition(
                        key="organisationId", match=MatchValue(value=organisation_id)
                    ),
                ]
            ),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for record in records:
            payload = dict(record.payload or {})
            payload["sharedWithOrganisation"] = shared_with_organisation
            client.set_payload(
                collection_name=collection_name,
                payload=payload,
                points=[record.id],
            )
            updated += 1

        if next_offset is None:
            break
        offset = next_offset

    logger.info(
        "Updated visibility for %d chunks of document %s to shared=%s",
        updated,
        document_id,
        shared_with_organisation,
    )
    return updated
