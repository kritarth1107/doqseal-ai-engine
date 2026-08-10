# DoqSeal AI Engine

Python extraction worker — Indic OCR, VLM structured extraction, RAG indexing, LangGraph chat.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
./run-worker.sh
```

Health API: `./run-health.sh` → http://localhost:3031/health

## Modes

| `EXTRACTION_MODE` | Behavior |
|-------------------|----------|
| `stub` | Instant demo data |
| `ocr_only` | EasyOCR + regex |
| `hybrid` | VLM + OCR fallback (default) |

## Docs

- [AGENTS.md](./AGENTS.md)
- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [docs/PIPELINE.md](./docs/PIPELINE.md)
- [docs/RAG.md](./docs/RAG.md) (planned)
- [docs/LANGGRAPH.md](./docs/LANGGRAPH.md) (planned)
