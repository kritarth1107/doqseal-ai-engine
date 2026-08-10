# Agent instructions — doqseal-ai-engine

## Conventions

- Python 3.10+, FastAPI health + RabbitMQ worker
- Pipeline: `app/pipeline/` — preprocess → OCR → VLM → validate
- Config: `app/config.py` via env vars

## Do NOT

- Send document text to Claude/GPT/OpenAI APIs (DPDP — India self-hosted only)
- Commit `.venv/`, model weights, or `.env`
- Change Mongo write shapes without updating `docs/interfaces.md` + backend

## Dev without GPU

```bash
EXTRACTION_MODE=stub ./run-worker.sh
```

## Cross-repo

Must match [doqseal-backend](https://github.com/Noooblien/doqseal-backend) on `AES_SECRET`, queue names, Mongo schemas.
