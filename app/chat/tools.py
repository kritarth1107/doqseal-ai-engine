"""Chat tools — Qdrant retrieval and MongoDB document helpers with tenant isolation."""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from app.config import settings
from app.db.mongo import (
    aggregate_extraction_field,
    get_db,
    list_visible_documents,
    load_extraction,
)
from app.rag.embedder import embed_query
from app.rag.indexer import _collection_name, build_visibility_filter

logger = logging.getLogger("doqseal.chat.tools")


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


def search_chunks(
    organisation_id: str,
    query: str,
    *,
    user_id: str | None = None,
    project_id: str | None = None,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Search document chunks with mandatory tenant isolation filters.

    Returns chunks with scores for reranking. Never returns deleted or
    invisible documents.
    """
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

    visibility_filter = build_visibility_filter(organisation_id, user_id, project_id=project_id)

    filter_dict: dict[str, Any] = {"must": []}
    if visibility_filter.must:
        for cond in visibility_filter.must:
            filter_dict["must"].append({"key": cond.key, "match": {"value": cond.match.value}})

    if visibility_filter.should:
        filter_dict["should"] = []
        for cond in visibility_filter.should:
            filter_dict["should"].append({"key": cond.key, "match": {"value": cond.match.value}})

    try:
        with httpx.Client(timeout=15.0, headers=_qdrant_headers()) as client:
            if not _collection_exists(client, organisation_id):
                logger.info("Qdrant collection %s not found", collection)
                return []

            response = client.post(
                f"{base_url}/collections/{collection}/points/search",
                json={
                    "vector": vector,
                    "limit": limit,
                    "with_payload": True,
                    "filter": filter_dict,
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

        if payload.get("deletedAt"):
            continue

        chunks.append(
            {
                "documentId": payload.get("documentId"),
                "projectId": payload.get("projectId"),
                "text": str(text),
                "page": payload.get("page"),
                "score": point.get("score", 0),
                "documentTitle": payload.get("documentTitle"),
                "documentType": payload.get("documentType"),
                "source": payload.get("source"),
                "field": payload.get("field"),
                "fieldValue": payload.get("fieldValue"),
            }
        )

    return chunks


def search_documents(
    organisation_id: str,
    query: str,
    *,
    project_id: str | None = None,
    user_id: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Legacy search interface - wraps search_chunks for backward compatibility."""
    chunks = search_chunks(
        organisation_id,
        query,
        user_id=user_id,
        project_id=project_id,
        limit=max(limit * 4, 20),
    )

    results: list[dict[str, Any]] = []
    for chunk in chunks:
        results.append(
            {
                "documentId": chunk.get("documentId"),
                "projectId": chunk.get("projectId"),
                "snippet": str(chunk.get("text", ""))[:500],
                "score": chunk.get("score"),
            }
        )
        if len(results) >= limit:
            break

    return results


def get_extraction_fields(
    organisation_id: str,
    document_ids: list[str],
    fields: list[str] | None = None,
    *,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    """Get extraction fields for specified documents.

    Returns only documents visible to the user in the organisation.
    """
    if not document_ids:
        return []

    visible_docs = {
        d["documentId"] for d in list_visible_documents(organisation_id, user_id=user_id, limit=500)
    }

    results = []
    for doc_id in document_ids:
        if doc_id not in visible_docs:
            continue

        extraction = load_extraction(doc_id, organisation_id=organisation_id)
        if not extraction:
            continue

        data = extraction.get("data", {})
        if fields:
            filtered_data = {}
            for field in fields:
                parts = field.split(".")
                value = data
                for part in parts:
                    if isinstance(value, dict):
                        value = value.get(part)
                    else:
                        value = None
                        break
                if value is not None:
                    filtered_data[field] = value
            data = filtered_data

        results.append(
            {
                "documentId": doc_id,
                "fields": data,
                "confidence": extraction.get("fieldConfidence", {}),
            }
        )

    return results


def list_documents(
    organisation_id: str,
    *,
    user_id: str | None = None,
    project_id: str | None = None,
    document_type: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List documents visible to the user with optional type filtering."""
    docs = list_visible_documents(
        organisation_id,
        user_id=user_id,
        project_id=project_id,
        limit=limit * 2 if document_type else limit,
    )

    if document_type:
        db = get_db()
        doc_ids = [d["documentId"] for d in docs]
        type_map = {}
        for ext in db.extractions.find(
            {"documentId": {"$in": doc_ids}, "organisationId": organisation_id},
            {"documentId": 1, "data.document_type": 1},
        ):
            type_map[ext["documentId"]] = (ext.get("data") or {}).get("document_type")

        docs = [d for d in docs if _matches_type(type_map.get(d["documentId"]), document_type)][
            :limit
        ]

    results = []
    for doc in docs[:limit]:
        results.append(
            {
                "documentId": doc.get("documentId"),
                "projectId": doc.get("projectId"),
                "title": doc.get("displayTitle") or doc.get("originalFilename") or "Untitled",
                "filename": doc.get("originalFilename"),
                "status": doc.get("status"),
            }
        )

    return results


def _matches_type(doc_type: str | None, filter_type: str) -> bool:
    """Check if document type matches filter (case-insensitive, partial)."""
    if not doc_type:
        return False
    return filter_type.lower() in doc_type.lower()


def aggregate_field(
    organisation_id: str,
    field_path: str,
    operation: str,
    *,
    user_id: str | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Aggregate over extraction fields (count, sum, avg, min, max, distinct).

    Always scoped to organisation and visible documents.
    """
    return aggregate_extraction_field(
        organisation_id,
        field_path,
        operation,
        user_id=user_id,
        project_id=project_id,
    )


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


def classify_document(*, title: str, filename: str, extra: str, has_medicines: bool) -> str:
    """Return prescription | invoice | note | document. Invoices never count as prescriptions."""
    heading = f"{title} {filename}"
    blob = f"{heading} {extra}"
    if _INVOICE_RE.search(heading) or (_INVOICE_RE.search(blob) and not _RX_RE.search(heading)):
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
    docs = list_visible_documents(
        organisation_id,
        user_id=user_id,
        project_id=project_id,
        limit=limit,
    )

    db = get_db()
    ids = [d["documentId"] for d in docs if d.get("documentId")]
    extraction_bits: dict[str, dict[str, Any]] = {}

    if ids:
        for row in db.extractions.find(
            {"documentId": {"$in": ids}, "organisationId": organisation_id},
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


def get_extraction(
    document_id: str, *, organisation_id: str | None = None
) -> dict[str, Any] | None:
    """Get extraction for a document."""
    return load_extraction(document_id, organisation_id=organisation_id)


def list_project_documents(
    organisation_id: str,
    project_id: str,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List documents in a project."""
    return list_documents(
        organisation_id,
        project_id=project_id,
        limit=limit,
    )
