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


_RX_RE = re.compile(
    r"prescri|\brx\b|\bmedicine(s)?\b|\btablet(s)?\b|\bdosage\b|\binvestigation(s)?\b",
    re.IGNORECASE,
)
_INVOICE_RE = re.compile(
    r"\binvoice\b|\bcash\s*memo\b|\breceipt\b|\bgst\b|\btax invoice\b|\bbill\b",
    re.IGNORECASE,
)
_NOTE_RE = re.compile(
    r"\bnote\b|\bitem list\b|\bmemo\b|\bhandwritten\b",
    re.IGNORECASE,
)


def _visible_to_user(doc: dict[str, Any], user_id: str | None) -> bool:
    if not user_id:
        return True
    shared = doc.get("sharedWithOrganisation")
    uploaded_by = doc.get("uploadedBy")
    if shared is False and uploaded_by and uploaded_by != user_id:
        return False
    return True


def classify_document(*, title: str, filename: str, extra: str, has_medicines: bool) -> str:
    """Return prescription | invoice | note | document. Invoices never count as prescriptions."""
    heading = f"{title} {filename}"
    blob = f"{heading} {extra}"
    if _INVOICE_RE.search(heading) or (
        _INVOICE_RE.search(blob) and not _RX_RE.search(heading)
    ):
        return "invoice"
    if _RX_RE.search(heading) or has_medicines:
        return "prescription"
    if _NOTE_RE.search(heading):
        return "note"
    if _RX_RE.search(extra or ""):
        return "prescription"
    return "document"


def list_document_library(
    organisation_id: str,
    *,
    project_id: str | None = None,
    user_id: str | None = None,
    limit: int = 80,
) -> dict[str, Any]:
    """Catalogue of Drive/project documents the user can see, for count questions."""
    db = get_db()
    query: dict[str, Any] = {
        "organisationId": organisation_id,
        "$or": [{"deletedAt": None}, {"deletedAt": {"$exists": False}}],
    }
    if project_id:
        query["projectId"] = project_id

    cursor = (
        db.documents.find(
            query,
            {
                "_id": 0,
                "documentId": 1,
                "projectId": 1,
                "originalFilename": 1,
                "displayTitle": 1,
                "status": 1,
                "uploadedBy": 1,
                "sharedWithOrganisation": 1,
            },
        )
        .sort("createdAt", -1)
        .limit(max(limit * 3, 120))
    )

    docs = [doc for doc in cursor if _visible_to_user(doc, user_id)][:limit]
    ids = [d["documentId"] for d in docs if d.get("documentId")]
    extraction_bits: dict[str, dict[str, Any]] = {}
    if ids:
        for row in db.extractions.find(
            {"documentId": {"$in": ids}},
            {
                "_id": 0,
                "documentId": 1,
                "data.summary": 1,
                "data.suggested_title": 1,
                "data.document_type": 1,
                "data.title": 1,
                "data.medicines": 1,
            },
        ):
            data = row.get("data") or {}
            medicines = data.get("medicines")
            extraction_bits[row.get("documentId") or ""] = {
                "text": " ".join(
                    str(data.get(key) or "")
                    for key in ("suggested_title", "title", "document_type", "summary")
                ),
                "hasMedicines": isinstance(medicines, list) and len(medicines) > 0,
                "medicineCount": len(medicines) if isinstance(medicines, list) else 0,
            }

    items: list[dict[str, Any]] = []
    counts = {"prescription": 0, "invoice": 0, "note": 0, "document": 0}
    for doc in docs:
        title = (doc.get("displayTitle") or doc.get("originalFilename") or "Untitled").strip()
        filename = (doc.get("originalFilename") or "").strip()
        bits = extraction_bits.get(doc.get("documentId") or "", {})
        kind = classify_document(
            title=title,
            filename=filename,
            extra=str(bits.get("text") or ""),
            has_medicines=bool(bits.get("hasMedicines")),
        )
        counts[kind] = counts.get(kind, 0) + 1
        items.append(
            {
                "documentId": doc.get("documentId"),
                "projectId": doc.get("projectId"),
                "title": title,
                "filename": filename,
                "status": doc.get("status"),
                "kind": kind,
                "prescription": kind == "prescription",
            }
        )

    return {
        "total": len(items),
        "prescriptionCount": counts.get("prescription", 0),
        "invoiceCount": counts.get("invoice", 0),
        "noteCount": counts.get("note", 0),
        "otherCount": counts.get("document", 0),
        "items": items,
    }


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
