# Cross-service interfaces

Shared with [doqseal-ai-engine](https://github.com/Noooblien/doqseal-ai-engine).

## RabbitMQ: extraction.jobs

```json
{ "jobId": "uuid" }
```

## MongoDB collections

- `documents`, `extractions`, `extraction_jobs`, `projects`, `organisations`

## Encryption

- `AES_SECRET` env must be identical in backend and ai-engine
- Documents stored encrypted at `storagePath`

## Chat proxy (planned)

`POST /api/v1/chat` → ai-engine `POST /chat`
