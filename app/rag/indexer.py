"""Upsert document chunks into per-organisation Qdrant collections."""

from __future__ import annotations

import logging
import re
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.rag.chunker import build_chunks
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


def index_extraction(
    *,
    organisation_id: str,
    document_id: str,
    job_id: str,
    project_id: str,
    ocr_full_text: str | None,
    extraction_data: dict[str, Any] | None,
) -> int:
    """Chunk, embed, and upsert extraction content. Returns number of points upserted."""
    chunks = build_chunks(ocr_full_text, extraction_data)
    if not chunks:
        logger.info("No RAG chunks for document %s", document_id)
        return 0

    vectors = embed_passages([chunk["text"] for chunk in chunks])
    collection_name = _ensure_collection(organisation_id)
    client = _get_client()

    points = [
        PointStruct(
            id=_point_id(document_id, chunk["source"], chunk["index"]),
            vector=vector,
            payload={
                "organisationId": organisation_id,
                "documentId": document_id,
                "jobId": job_id,
                "projectId": project_id,
                "source": chunk["source"],
                "chunkIndex": chunk["index"],
                "text": chunk["text"],
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    client.upsert(collection_name=collection_name, points=points)
    logger.info(
        "Indexed %d chunks for document %s into %s",
        len(points),
        document_id,
        collection_name,
    )
    return len(points)


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
            scroll_filter={
                "must": [{"key": "documentId", "match": {"value": document_id}}]
            },
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
