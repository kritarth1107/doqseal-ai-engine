from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.chat import run_chat
from app.db.mongo import get_db
from app.rag.indexer import delete_document_chunks

app = FastAPI(
    title="DoqSeal Main Backend",
    description="Internal extraction engine — not exposed to frontend",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    organisationId: str = Field(..., min_length=1)
    projectId: str | None = None
    userId: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict]
    mode: str


@app.get("/health")
def health():
    """Public-safe liveness. No paths, URIs, secrets, or infra hostnames."""
    import httpx

    from app.config import settings

    checks: dict[str, str] = {}
    status = "ok"

    try:
        get_db().command("ping")
        checks["mongodb"] = "up"
    except Exception:
        checks["mongodb"] = "down"
        status = "unhealthy"

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


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    result = run_chat(
        message,
        body.organisationId,
        project_id=body.projectId,
        user_id=body.userId,
    )
    return ChatResponse(**result)


@app.delete("/rag/documents/{document_id}")
def delete_rag_document(document_id: str, organisationId: str):
    if not organisationId.strip():
        raise HTTPException(status_code=400, detail="organisationId is required")

    deleted = delete_document_chunks(
        organisation_id=organisationId,
        document_id=document_id,
    )
    return {"deleted": deleted, "documentId": document_id}