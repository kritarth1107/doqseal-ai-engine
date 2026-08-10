from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.chat import run_chat
from app.config import settings
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


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict]
    mode: str


@app.get("/health")
def health():
    db = get_db()
    db.command("ping")
    return {
        "status": "ok",
        "service": "doqseal-main-backend",
        "queue": settings.extraction_queue,
        "storage_root": str(settings.storage_root),
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