from fastapi import FastAPI

from app.config import settings
from app.db.mongo import get_db

app = FastAPI(
    title="DoqSeal Main Backend",
    description="Internal extraction engine — not exposed to frontend",
    version="0.1.0",
)


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