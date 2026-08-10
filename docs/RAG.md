# RAG indexing

After extraction completes, the worker indexes document content into Qdrant for later retrieval.

## Flow

```
worker.py (job complete)
    ↓
app/rag/indexer.py
    ↓
chunker.py → embedder.py → Qdrant (org_{organisationId})
```

## Modules

| Module | Role |
|--------|------|
| `app/rag/chunker.py` | Split `ocrFullText` and extraction JSON into ~500-token chunks |
| `app/rag/embedder.py` | Lazy-load `sentence-transformers` and embed with `passage:` prefix |
| `app/rag/indexer.py` | Upsert vectors + metadata into per-org Qdrant collections |

## Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP endpoint |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | Sentence-transformers model (768-dim) |

## Collections

One Qdrant collection per organisation: `org_{organisationId}`.

Each point stores:

- `documentId`, `jobId`, `projectId`, `organisationId`
- `source` — `ocr` or `extraction`
- `chunkIndex`, `text`

Point IDs are deterministic (`uuid5`) so re-indexing the same document upserts in place.

## Trigger

Indexing runs inline in `worker.py` immediately after `mark_job_completed`. Failures are logged but do not fail the extraction job.

A separate `indexing.jobs` queue can be added later for async indexing.

## Planned

- LangGraph chat retrieval against Qdrant
- Query embedding with `query:` prefix (e5 convention)
