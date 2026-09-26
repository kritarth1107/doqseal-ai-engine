"""Tenant-safe retrieval for grounded chat.

Every search is scoped three ways:
1. the organisation's own Qdrant collection,
2. a mandatory `organisationId == org` payload filter (plus project when given),
3. a MongoDB check of every hit: the document must belong to the organisation,
   must not be deleted, and must be visible to the user (shared with the
   organisation or uploaded by them). Deleted or hidden documents are never
   returned, whatever their vectors still say.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger("doqseal.chat.retrieval")

_TOKEN_RE = re.compile(r"[\w][\w\-./]*", re.UNICODE)
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its me my of on "
    "or our show tell that the their them there these this to was we what when where which "
    "who why will with you your all any about please give list find get".split()
)
MAX_CHUNKS_PER_DOCUMENT = 6


@dataclass
class Evidence:
    document_id: str
    title: str
    text: str
    score: float
    page: int | None = None
    source: str | None = None
    project_id: str | None = None


def collection_name(organisation_id: str) -> str:
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", organisation_id)
    return f"org_{safe_id}"


def _qdrant_headers() -> dict[str, str]:
    return {"api-key": settings.qdrant_api_key} if settings.qdrant_api_key else {}


def qdrant_search(
    collection: str, vector: list[float], query_filter: dict[str, Any], limit: int
) -> list[dict[str, Any]]:
    """Raw Qdrant search. Returns [] when the collection does not exist."""
    base = settings.qdrant_url.rstrip("/")
    with httpx.Client(timeout=15.0, headers=_qdrant_headers()) as client:
        response = client.post(
            f"{base}/collections/{collection}/points/search",
            json={"vector": vector, "limit": limit, "with_payload": True, "filter": query_filter},
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()
        return response.json().get("result") or []


def embed(text: str) -> list[float]:
    from app.rag.embedder import embed_query

    return embed_query(text)


def _get_db():
    from app.db.mongo import get_db

    return get_db()


def visibility_query(organisation_id: str, user_id: str | None, project_id: str | None) -> dict[str, Any]:
    query: dict[str, Any] = {"organisationId": organisation_id, "deletedAt": None}
    if user_id:
        query["$or"] = [{"sharedWithOrganisation": {"$ne": False}}, {"uploadedBy": user_id}]
    else:
        # Without a verified user only organisation-shared documents are visible.
        query["sharedWithOrganisation"] = {"$ne": False}
    if project_id:
        query["projectId"] = project_id
    return query


def allowed_documents(
    organisation_id: str,
    user_id: str | None,
    document_ids: list[str],
    project_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    if not document_ids:
        return {}
    query = visibility_query(organisation_id, user_id, project_id)
    query["documentId"] = {"$in": sorted(set(document_ids))}
    rows = _get_db().documents.find(
        query,
        {"_id": 0, "documentId": 1, "organisationId": 1, "displayTitle": 1, "originalFilename": 1, "projectId": 1},
    )
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("organisationId") != organisation_id:
            continue
        out[row["documentId"]] = row
    return out


def list_library(
    organisation_id: str, user_id: str | None, project_id: str | None, limit: int = 100
) -> list[dict[str, Any]]:
    rows = (
        _get_db()
        .documents.find(
            visibility_query(organisation_id, user_id, project_id),
            {
                "_id": 0,
                "documentId": 1,
                "organisationId": 1,
                "displayTitle": 1,
                "originalFilename": 1,
                "status": 1,
                "createdAt": 1,
            },
        )
        .sort("createdAt", -1)
        .limit(limit)
    )
    return [r for r in rows if r.get("organisationId") == organisation_id]


def document_title(row: dict[str, Any]) -> str:
    return str(row.get("displayTitle") or row.get("originalFilename") or "Untitled document")[:200]


def query_terms(text: str) -> list[str]:
    terms = []
    for tok in _TOKEN_RE.findall(text.lower()):
        tok = tok.strip("-./")
        if len(tok) < 2 or tok in _STOPWORDS:
            continue
        terms.append(tok)
    return list(dict.fromkeys(terms))


def lexical_score(terms: list[str], text: str) -> float:
    if not terms:
        return 0.0
    haystack = text.lower()
    hits = sum(1 for t in terms if t in haystack)
    return hits / len(terms)


def _search_sync(
    organisation_id: str,
    query: str,
    user_id: str | None,
    project_id: str | None,
) -> list[Evidence]:
    if not query.strip():
        return []
    vector = embed(query)
    must: list[dict[str, Any]] = [{"key": "organisationId", "match": {"value": organisation_id}}]
    if project_id:
        must.append({"key": "projectId", "match": {"value": project_id}})
    points = qdrant_search(collection_name(organisation_id), vector, {"must": must}, settings.chat_retrieve_top_k)

    candidates: list[tuple[dict[str, Any], float]] = []
    for point in points:
        payload = point.get("payload") or {}
        if payload.get("organisationId") != organisation_id:
            continue  # defence in depth: never trust a mis-filed point
        if payload.get("deletedAt"):
            continue
        text = payload.get("text")
        doc_id = payload.get("documentId")
        if not text or not doc_id:
            continue
        candidates.append((payload, float(point.get("score") or 0.0)))

    allowed = allowed_documents(organisation_id, user_id, [p.get("documentId") for p, _ in candidates], project_id)
    terms = query_terms(query)
    seen_text: set[str] = set()
    ranked: list[Evidence] = []
    for payload, vscore in candidates:
        row = allowed.get(payload["documentId"])
        if row is None or vscore < settings.chat_min_score:
            continue
        text = str(payload["text"])
        key = text.strip()[:500]
        if key in seen_text:
            continue
        seen_text.add(key)
        page = payload.get("page")
        ranked.append(
            Evidence(
                document_id=payload["documentId"],
                title=document_title(row),
                text=text,
                score=0.75 * vscore + 0.25 * lexical_score(terms, text),
                page=page if isinstance(page, int) else None,
                source=payload.get("source"),
                project_id=row.get("projectId"),
            )
        )
    ranked.sort(key=lambda e: e.score, reverse=True)
    return select_within_budget(ranked)


def select_within_budget(ranked: list[Evidence]) -> list[Evidence]:
    chosen: list[Evidence] = []
    per_doc: dict[str, int] = {}
    total = 0
    for item in ranked:
        if len(chosen) >= settings.chat_context_chunks:
            break
        if per_doc.get(item.document_id, 0) >= MAX_CHUNKS_PER_DOCUMENT:
            continue
        size = len(item.text)
        if chosen and total + size > settings.chat_context_chars:
            continue
        chosen.append(item)
        per_doc[item.document_id] = per_doc.get(item.document_id, 0) + 1
        total += size
    return chosen


async def search(
    organisation_id: str,
    query: str,
    *,
    user_id: str | None,
    project_id: str | None = None,
) -> list[Evidence]:
    """Embedding and I/O run in a worker thread so the event loop stays free."""
    return await asyncio.to_thread(_search_sync, organisation_id, query, user_id, project_id)
