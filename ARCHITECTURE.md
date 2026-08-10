# Architecture — doqseal-ai-engine

```
RabbitMQ extraction.jobs
        ↓
   worker.py
        ↓
   runner.py → OCR → VLM → validate
        ↓
   mongo.py → extractions collection
        ↓
   (planned) indexing.jobs → Qdrant
        ↓
   (planned) LangGraph chat ← local Qwen2.5-7B
```

## Components

| Module | Role |
|--------|------|
| `app/worker.py` | RabbitMQ consumer |
| `app/pipeline/` | OCR, VLM, validation |
| `app/rag/` | Chunk, embed, index (planned) |
| `app/chat/` | LangGraph agent (planned) |
| `app/main.py` | Health check FastAPI |
