from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.chat import run_chat
from app.config import settings
from app.db.mongo import get_db

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