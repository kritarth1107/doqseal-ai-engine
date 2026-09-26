"""Internal endpoint: classify one document into a bundle template slot.

Only the backend calls this. It is disabled unless AI_ENGINE_SERVICE_TOKEN is
set, and every request must present that token. The organisation is checked
against the document record before any model call.
"""

from __future__ import annotations

import hmac
import logging
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.bundle.classify import (
    DEFAULT_KEY_FIELDS,
    MAX_SLOTS,
    ClassificationError,
    classify_document,
)
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bundle", tags=["bundle"])


class SlotDefinition(BaseModel):
    key: str = Field(..., min_length=1, max_length=80)
    label: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    hints: list[str] = Field(default_factory=list, max_length=20)


class DocumentContent(BaseModel):
    documentType: str | None = Field(default=None, max_length=200)
    text: str | None = Field(default=None, max_length=200_000)
    fields: dict[str, Any] = Field(default_factory=dict)


class ClassifyRequest(BaseModel):
    requestId: str = Field(..., min_length=1, max_length=100)
    organisationId: str = Field(..., min_length=1, max_length=100)
    bundleId: str = Field(..., min_length=1, max_length=100)
    documentId: str = Field(..., min_length=1, max_length=100)
    slots: list[SlotDefinition] = Field(..., min_length=1, max_length=MAX_SLOTS)
    document: DocumentContent
    keyFieldNames: list[str] = Field(
        default_factory=lambda: list(DEFAULT_KEY_FIELDS), max_length=20
    )


class ClassifyResponse(BaseModel):
    requestId: str
    organisationId: str
    bundleId: str
    documentId: str
    slot: str | None
    confidence: float
    reasons: list[str]
    alternatives: list[dict[str, Any]]
    keyFields: dict[str, str]
    model: str | None = None
    cached: bool = False


def _check_token(provided: str | None) -> None:
    expected = (settings.ai_engine_service_token or "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="bundle classification is not enabled")
    if not provided or not hmac.compare_digest(provided.strip(), expected):
        raise HTTPException(status_code=401, detail="invalid service token")


def document_belongs_to_org(organisation_id: str, document_id: str) -> bool:
    from app.db.mongo import get_db

    row = get_db().documents.find_one(
        {"documentId": document_id, "organisationId": organisation_id, "deletedAt": None},
        {"_id": 1},
    )
    return row is not None


@router.post("/classify", response_model=ClassifyResponse)
def classify(
    body: ClassifyRequest,
    x_service_token: str | None = Header(default=None),
    x_organisation_id: str | None = Header(default=None),
):
    _check_token(x_service_token)

    if not x_organisation_id or x_organisation_id.strip() != body.organisationId:
        raise HTTPException(status_code=400, detail="organisation mismatch")

    keys = [s.key for s in body.slots]
    if len(set(keys)) != len(keys):
        raise HTTPException(status_code=400, detail="slot keys must be unique")

    try:
        owned = document_belongs_to_org(body.organisationId, body.documentId)
    except Exception:
        logger.exception("bundle classify: document lookup failed")
        raise HTTPException(status_code=503, detail="document lookup unavailable")
    if not owned:
        raise HTTPException(status_code=404, detail="document not found")

    try:
        result = classify_document(
            organisation_id=body.organisationId,
            request_id=body.requestId,
            slots=[s.model_dump() for s in body.slots],
            document=body.document.model_dump(),
            key_field_names=body.keyFieldNames,
        )
    except ClassificationError as exc:
        logger.warning(
            "bundle classify failed org=%s document=%s retryable=%s reason=%s",
            body.organisationId,
            body.documentId,
            exc.retryable,
            exc,
        )
        status = 502 if exc.retryable else 422
        raise HTTPException(
            status_code=status,
            detail={"message": "classification failed", "retryable": exc.retryable},
        )

    return ClassifyResponse(
        requestId=body.requestId,
        organisationId=body.organisationId,
        bundleId=body.bundleId,
        documentId=body.documentId,
        **result,
    )
