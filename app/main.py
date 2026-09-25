"""DoqSeal AI Engine — FastAPI endpoints with JWT auth and streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.auth import (
    ServiceClaims,
    get_chat_claims,
    get_rag_delete_claims,
    resolve_context,
)
from app.chat import run_chat
from app.chat.pipeline import (
    ChatMessage,
    ChatRequest,
    run_chat_pipeline,
    stream_chat_pipeline,
)
from app.config import settings
from app.db.mongo import get_db
from app.rag.indexer import delete_document_chunks, mark_document_deleted

logger = logging.getLogger("doqseal.main")

app = FastAPI(
    title="DoqSeal AI Engine",
    description="Internal extraction and chat engine — not exposed to frontend",
    version="0.2.0",
)


class LegacyChatRequest(BaseModel):
    """Legacy chat request body."""

    message: str = Field(..., min_length=1, max_length=8000)
    organisationId: str = Field(..., min_length=1)
    projectId: str | None = None
    userId: str | None = None


class LegacyChatResponse(BaseModel):
    """Legacy chat response."""

    answer: str
    citations: list[dict]
    thinking: list[dict] = Field(default_factory=list)
    mode: str


class StreamChatRequest(BaseModel):
    """Streaming chat request per contract."""

    message: str = Field(..., min_length=1, max_length=8000)
    history: list[dict] = Field(default_factory=list, max_length=20)
    projectId: str | None = None
    conversationId: str | None = None
    organisationId: str | None = None  # Legacy fallback, ignored when JWT present


@app.get("/health")
def health():
    """Public-safe liveness. No paths, URIs, secrets, or infra hostnames."""
    import httpx

    from app.worker_heartbeat import heartbeat_age_seconds

    checks: dict[str, str] = {}
    status = "ok"

    try:
        get_db().command("ping")
        checks["mongodb"] = "up"
    except Exception:
        checks["mongodb"] = "down"
        status = "unhealthy"

    age = heartbeat_age_seconds()
    stale_limit = float(os.getenv("WORKER_HEARTBEAT_STALE_SEC", "180"))
    if age is None:
        checks["extraction_worker"] = "missing"
        status = "unhealthy"
    elif age > stale_limit:
        checks["extraction_worker"] = "stale"
        status = "unhealthy"
    else:
        checks["extraction_worker"] = "up"

    try:
        with httpx.Client(timeout=3.0) as client:
            response = client.get(f"{settings.ollama_url.rstrip('/')}/api/tags")
            checks["ollama"] = "up" if response.is_success else "down"
    except Exception:
        checks["ollama"] = "down"

    if (
        (settings.azure_openai_endpoint or "").strip()
        and (settings.azure_openai_api_key or "").strip()
    ):
        checks["azure_openai"] = "configured"
    else:
        checks["azure_openai"] = "missing"
        if checks.get("ollama") == "down" and status == "ok":
            status = "degraded"

    if (
        checks.get("azure_openai") != "configured"
        and checks.get("ollama") == "down"
        and status == "ok"
    ):
        status = "degraded"

    return {
        "status": status,
        "service": "doqseal-ai-engine",
        "version": app.version,
        "checks": checks,
    }


@app.post("/chat", response_model=LegacyChatResponse)
async def chat_legacy(
    body: LegacyChatRequest,
    claims: ServiceClaims | None = Depends(get_chat_claims),
):
    """Legacy non-streaming chat endpoint.

    Now uses the new grounded pipeline but returns in the legacy format.
    """
    organisation_id, user_id, project_id = resolve_context(
        claims, body.organisationId, body.userId, body.projectId
    )

    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    request = ChatRequest(
        message=message,
        organisation_id=organisation_id,
        user_id=user_id,
        project_id=project_id,
    )

    try:
        result = await run_chat_pipeline(request)
    except Exception as exc:
        logger.exception("Chat pipeline failed: %s", exc)
        result_legacy = run_chat(
            message,
            organisation_id,
            project_id=project_id,
            user_id=user_id,
        )
        return LegacyChatResponse(**result_legacy)

    citations = [
        {
            "documentId": c.document_id,
            "title": c.title,
            "page": c.page,
            "quote": c.quote,
            "n": c.n,
        }
        for c in result.citations
    ]

    return LegacyChatResponse(
        answer=result.answer,
        citations=citations,
        thinking=[],
        mode=result.mode,
    )


@app.post("/v1/chat/stream")
async def chat_stream(
    request: Request,
    body: StreamChatRequest,
    claims: ServiceClaims | None = Depends(get_chat_claims),
):
    """Streaming chat endpoint per contract.

    Returns SSE events: run.started, step, token, citation, decline, run.completed
    Sends heartbeat comments every 15s.
    Cancels on client disconnect.
    """
    organisation_id, user_id, project_id = resolve_context(
        claims, body.organisationId, None, body.projectId
    )

    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    history = []
    for h in (body.history or [])[-settings.chat_history_turns * 2 :]:
        role = h.get("role", "user")
        content = h.get("content", "")
        if role in ("user", "assistant") and content:
            history.append(ChatMessage(role=role, content=content[:2000]))

    chat_request = ChatRequest(
        message=message,
        organisation_id=organisation_id,
        user_id=user_id,
        project_id=project_id or (claims.project_id if claims else None),
        conversation_id=body.conversationId,
        history=history,
    )

    async def event_generator():
        last_heartbeat = time.time()
        heartbeat_interval = settings.stream_heartbeat_seconds

        try:
            async for event in stream_chat_pipeline(chat_request):
                if await request.is_disconnected():
                    logger.info("Client disconnected, cancelling stream")
                    break

                yield f"event: {event.event}\ndata: {json.dumps(event.data)}\n\n"

                now = time.time()
                if now - last_heartbeat >= heartbeat_interval:
                    yield ": ping\n\n"
                    last_heartbeat = now

        except asyncio.CancelledError:
            logger.info("Stream cancelled")
            raise
        except Exception as exc:
            logger.exception("Streaming error: %s", exc)
            error_event = {
                "code": "internal_error",
                "message": "An error occurred while processing your request",
            }
            yield f"event: error\ndata: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/rag/documents/{document_id}")
async def delete_rag_document(
    document_id: str,
    organisationId: str | None = None,
    claims: ServiceClaims | None = Depends(get_rag_delete_claims),
):
    """Delete document chunks from Qdrant.

    Requires rag:delete scope when JWT is enabled.
    """
    organisation_id, _, _ = resolve_context(claims, organisationId, None, None)

    deleted = delete_document_chunks(
        organisation_id=organisation_id,
        document_id=document_id,
    )

    return {"deleted": deleted, "documentId": document_id}


@app.post("/rag/documents/{document_id}/mark-deleted")
async def mark_rag_document_deleted(
    document_id: str,
    organisationId: str | None = None,
    claims: ServiceClaims | None = Depends(get_rag_delete_claims),
):
    """Soft-delete document chunks in Qdrant (marks deletedAt).

    Requires rag:delete scope when JWT is enabled.
    """
    organisation_id, _, _ = resolve_context(claims, organisationId, None, None)

    updated = mark_document_deleted(
        organisation_id=organisation_id,
        document_id=document_id,
    )

    return {"updated": updated, "documentId": document_id}
