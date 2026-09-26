import logging
import os
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.bundle.router import router as bundle_router
from app.chat.engine import ChatInput, ChatTurn, run_chat_events, run_chat_once
from app.chat.sse import sse_stream
from app.config import settings
from app.db.mongo import get_db
from app.rag.indexer import delete_document_chunks
from app.security import check_org, optional_claims, require_claims

logger = logging.getLogger("doqseal.main")

app = FastAPI(
    title="DoqSeal Main Backend",
    description="Internal extraction engine — not exposed to frontend",
    version="0.1.0",
)

app.include_router(bundle_router)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    organisationId: str = Field(..., min_length=1)
    projectId: str | None = None
    userId: str | None = None


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=20000)


class StreamChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=40)
    projectId: str | None = None
    conversationId: str | None = Field(default=None, max_length=100)
    # Optional; when present it must equal the organisation in the token.
    organisationId: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict]
    thinking: list[dict] = Field(default_factory=list)
    mode: str


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


def _step_summary(step: dict) -> str:
    detail = step.get("detail") or {}
    parts = []
    if "documents" in detail:
        parts.append(f"{detail['documents']} document{'s' if detail['documents'] != 1 else ''}")
    if "chunks" in detail:
        parts.append(f"{detail['chunks']} passage{'s' if detail['chunks'] != 1 else ''}")
    if detail.get("titles"):
        parts.append(", ".join(detail["titles"][:3]))
    return "; ".join(parts)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    """Non-streaming chat, kept for existing callers. Uses the grounded pipeline."""
    claims = optional_claims(request, scope="chat")
    check_org(claims, body.organisationId)
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    outcome = await run_chat_once(
        ChatInput(
            message=message,
            organisation_id=claims.organisation_id if claims else body.organisationId,
            user_id=claims.user_id if claims else body.userId,
            project_id=claims.project_id if claims else body.projectId,
        )
    )
    if outcome.error:
        raise HTTPException(status_code=503, detail=outcome.error["message"])
    return ChatResponse(
        answer=outcome.answer,
        citations=[{**c, "snippet": c.get("quote")} for c in outcome.citations],
        thinking=[
            {"title": step.get("label") or step.get("name"), "detail": _step_summary(step)}
            for step in outcome.steps
        ],
        mode=outcome.mode,
    )


@app.post("/v1/chat/stream")
async def chat_stream(request: Request, body: StreamChatRequest):
    """Streaming grounded chat (text/event-stream). Always needs a service token."""
    claims = require_claims(request, scope="chat")
    check_org(claims, body.organisationId)
    if body.projectId and claims.project_id and body.projectId != claims.project_id:
        raise HTTPException(status_code=403, detail="project mismatch")
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    inp = ChatInput(
        message=message,
        organisation_id=claims.organisation_id,
        user_id=claims.user_id,
        # A project only narrows the search inside the token's organisation.
        project_id=claims.project_id or body.projectId,
        conversation_id=body.conversationId,
        history=[ChatTurn(role=t.role, content=t.content) for t in body.history],
    )
    return StreamingResponse(
        sse_stream(
            run_chat_events(inp),
            heartbeat_seconds=settings.chat_heartbeat_seconds,
            is_disconnected=request.is_disconnected,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/rag/documents/{document_id}")
def delete_rag_document(request: Request, document_id: str, organisationId: str | None = None):
    claims = optional_claims(request, scope="rag:delete")
    check_org(claims, organisationId)
    organisation_id = claims.organisation_id if claims else (organisationId or "").strip()
    if not organisation_id:
        raise HTTPException(status_code=400, detail="organisationId is required")

    deleted = delete_document_chunks(
        organisation_id=organisation_id,
        document_id=document_id,
    )
    return {"deleted": deleted, "documentId": document_id}
