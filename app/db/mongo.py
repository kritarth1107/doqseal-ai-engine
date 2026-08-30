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


def load_job(job_id: str) -> dict[str, Any] | None:
    return get_db().extraction_jobs.find_one({"jobId": job_id})


def load_project(project_id: str) -> dict[str, Any] | None:
    return get_db().projects.find_one({"projectId": project_id, "deletedAt": None})


def load_document(document_id: str) -> dict[str, Any] | None:
    return get_db().documents.find_one({"documentId": document_id, "deletedAt": None})


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

    db.extractions.insert_one(
        {
            "extractionId": str(uuid4()),
            "documentId": document_id,
            "jobId": job_id,
            "organisationId": organisation_id,
            "projectId": project_id,
            "version": 1,
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