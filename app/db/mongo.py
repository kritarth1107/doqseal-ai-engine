"""MongoDB helpers with tenant isolation."""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pymongo import MongoClient

from app.config import settings

_client: MongoClient | None = None


def get_db():
    global _client
    if _client is None:
        _client = MongoClient(settings.mongodb_uri)
    return _client.get_default_database()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def load_job(job_id: str, *, organisation_id: str | None = None) -> dict[str, Any] | None:
    """Load extraction job by ID. Optionally verify org for defense in depth."""
    query: dict[str, Any] = {"jobId": job_id}
    if organisation_id:
        query["organisationId"] = organisation_id
    return get_db().extraction_jobs.find_one(query)


def load_project(project_id: str, *, organisation_id: str | None = None) -> dict[str, Any] | None:
    """Load project by ID. Optionally verify org for defense in depth."""
    query: dict[str, Any] = {"projectId": project_id, "deletedAt": None}
    if organisation_id:
        query["organisationId"] = organisation_id
    return get_db().projects.find_one(query)


def load_document(document_id: str, *, organisation_id: str | None = None) -> dict[str, Any] | None:
    """Load document by ID. Optionally verify org for defense in depth."""
    query: dict[str, Any] = {"documentId": document_id, "deletedAt": None}
    if organisation_id:
        query["organisationId"] = organisation_id
    return get_db().documents.find_one(query)


def load_extraction(
    document_id: str, *, organisation_id: str | None = None
) -> dict[str, Any] | None:
    """Load latest extraction for a document."""
    query: dict[str, Any] = {"documentId": document_id}
    if organisation_id:
        query["organisationId"] = organisation_id
    return get_db().extractions.find_one(query, sort=[("version", -1), ("createdAt", -1)])


def get_org_config(organisation_id: str) -> dict[str, Any]:
    """Load organisation config including guardrail overrides."""
    org = get_db().organisations.find_one({"publicId": organisation_id})
    if not org:
        return {}
    return org.get("aiConfig", {}) or {}


def mark_job_processing(job_id: str, document_id: str) -> None:
    now = utcnow()
    db = get_db()
    db.extraction_jobs.update_one(
        {"jobId": job_id},
        {"$set": {"status": "processing", "startedAt": now, "updatedAt": now}},
    )
    db.documents.update_one(
        {"documentId": document_id},
        {"$set": {"status": "processing", "updatedAt": now}},
    )


def mark_job_completed(
    job_id: str,
    document_id: str,
    organisation_id: str,
    project_id: str | None,
    extraction_payload: dict[str, Any],
) -> None:
    now = utcnow()
    db = get_db()

    latest = db.extractions.find_one(
        {"documentId": document_id},
        sort=[("version", -1), ("createdAt", -1)],
    )
    next_version = int((latest or {}).get("version") or 0) + 1

    db.extractions.delete_many({"documentId": document_id})

    db.extractions.insert_one(
        {
            "extractionId": str(uuid4()),
            "documentId": document_id,
            "jobId": job_id,
            "organisationId": organisation_id,
            "projectId": project_id,
            "version": next_version,
            "data": extraction_payload["data"],
            "fieldConfidence": extraction_payload["fieldConfidence"],
            "validationErrors": extraction_payload["validationErrors"],
            "status": extraction_payload["status"],
            "strategy": extraction_payload.get("strategy", "hybrid"),
            "ocrFullText": extraction_payload.get("ocrFullText"),
            "ocrLineCount": extraction_payload.get("ocrLineCount"),
            "ocrAverageConfidence": extraction_payload.get("ocrAverageConfidence"),
            "approvedAt": now,
            "createdAt": now,
            "updatedAt": now,
        }
    )

    db.extraction_jobs.update_one(
        {"jobId": job_id},
        {
            "$set": {
                "status": "completed",
                "completedAt": now,
                "error": None,
                "updatedAt": now,
            }
        },
    )

    document_update: dict[str, Any] = {"status": "completed", "updatedAt": now}
    display_title = extraction_payload.get("displayTitle")
    if isinstance(display_title, str) and display_title.strip():
        document_update["displayTitle"] = display_title.strip()

    db.documents.update_one(
        {"documentId": document_id},
        {"$set": document_update},
    )


def mark_job_failed(job_id: str, document_id: str, error: str) -> None:
    now = utcnow()
    db = get_db()
    db.extraction_jobs.update_one(
        {"jobId": job_id},
        {
            "$set": {
                "status": "failed",
                "error": error,
                "completedAt": now,
                "updatedAt": now,
            }
        },
    )
    db.documents.update_one(
        {"documentId": document_id},
        {"$set": {"status": "failed", "updatedAt": now}},
    )


def list_visible_documents(
    organisation_id: str,
    *,
    user_id: str | None = None,
    project_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """List documents visible to user, respecting visibility rules."""
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
                "createdAt": 1,
            },
        )
        .sort("createdAt", -1)
        .limit(max(limit * 3, 200))
    )

    docs = []
    for doc in cursor:
        if user_id:
            shared = doc.get("sharedWithOrganisation")
            uploaded_by = doc.get("uploadedBy")
            if shared is False and uploaded_by and uploaded_by != user_id:
                continue
        docs.append(doc)
        if len(docs) >= limit:
            break

    return docs


def aggregate_extraction_field(
    organisation_id: str,
    field_path: str,
    operation: str,
    *,
    user_id: str | None = None,
    project_id: str | None = None,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate over extraction fields for org documents.

    Supports: count, sum, avg, min, max, distinct
    """
    db = get_db()

    visible_doc_ids = [
        d["documentId"]
        for d in list_visible_documents(
            organisation_id, user_id=user_id, project_id=project_id, limit=1000
        )
    ]

    if not visible_doc_ids:
        return {"result": None, "count": 0}

    match_stage: dict[str, Any] = {
        "organisationId": organisation_id,
        "documentId": {"$in": visible_doc_ids},
    }
    if filters:
        match_stage.update(filters)

    field_ref = f"$data.{field_path}"

    if operation == "count":
        pipeline = [
            {"$match": match_stage},
            {"$match": {f"data.{field_path}": {"$exists": True, "$ne": None}}},
            {"$count": "result"},
        ]
    elif operation == "distinct":
        pipeline = [
            {"$match": match_stage},
            {"$group": {"_id": field_ref}},
            {"$group": {"_id": None, "result": {"$push": "$_id"}, "count": {"$sum": 1}}},
        ]
    elif operation in ("sum", "avg", "min", "max"):
        agg_op = f"${operation}"
        pipeline = [
            {"$match": match_stage},
            {
                "$group": {
                    "_id": None,
                    "result": {agg_op: {"$toDouble": field_ref}},
                    "count": {"$sum": 1},
                }
            },
        ]
    else:
        return {"error": f"Unknown operation: {operation}"}

    results = list(db.extractions.aggregate(pipeline))
    if not results:
        return {"result": None, "count": 0}

    return {
        "result": results[0].get("result"),
        "count": results[0].get("count", 0),
    }
