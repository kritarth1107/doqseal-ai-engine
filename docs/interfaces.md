# Cross-service interfaces

Shared with [doqseal-backend](https://github.com/Noooblien/doqseal-backend).

## Authentication

Service-to-service JWT (HS256) required when `AI_ENGINE_JWT_SECRET` is set.

**Claims:**
- `iss`: "doqseal-backend"
- `aud`: "doqseal-ai-engine"
- `sub`: userId
- `org`: organisationId
- `pid`: projectId (optional)
- `scope`: "chat" | "rag:delete" | "rag:read"
- `iat`, `exp` (max 5 minutes), `jti`

**Header:** `Authorization: Bearer <jwt>`

Rollout safety: JWT enforced only when `AI_ENGINE_JWT_SECRET` is set. If unset, logs a warning and allows legacy behavior.

## REST Endpoints

### POST /v1/chat/stream

Streaming chat with SSE events. Requires `chat` scope.

**Request:**
```json
{
  "message": "string (required, max 8000 chars)",
  "history": [{"role": "user|assistant", "content": "string"}],
  "projectId": "string (optional)",
  "conversationId": "string (optional)"
}
```

**Response:** `text/event-stream`

**Events:**
- `run.started`: `{"runId", "conversationId"}`
- `step`: `{"id", "name", "status": "started|done", "label", "detail"}`
- `token`: `{"text"}`
- `citation`: `{"n", "documentId", "title", "page", "quote"}`
- `decline`: `{"reason": "not_covered|off_topic|small_talk", "message"}`
- `run.completed`: `{"mode": "answered|declined|partial", "usage", "latencyMs"}`
- `error`: `{"code", "message"}`
- Heartbeat: `: ping` every 15s

### POST /chat (Legacy)

Non-streaming chat. Requires `chat` scope.

**Request:**
```json
{
  "message": "string",
  "organisationId": "string",
  "projectId": "string (optional)",
  "userId": "string (optional)"
}
```

**Response:**
```json
{
  "answer": "string",
  "citations": [{"documentId", "title", "page", "quote", "n"}],
  "thinking": [],
  "mode": "answered|declined|partial"
}
```

### DELETE /rag/documents/{documentId}

Delete document vectors. Requires `rag:delete` scope.

### POST /rag/documents/{documentId}/mark-deleted

Soft-delete document vectors. Requires `rag:delete` scope.

### GET /health

Liveness check. No authentication required.

## RabbitMQ: extraction.jobs

```json
{ "jobId": "uuid" }
```

## MongoDB collections

- `documents`, `extractions`, `extraction_jobs`, `projects`, `organisations`

## Encryption

- `AES_SECRET` env must be identical in backend and ai-engine
- Documents stored encrypted at `storagePath`

## Qdrant

- Collection per org: `org_{organisationId}`
- Payload includes: `organisationId`, `documentId`, `deletedAt`, `uploadedBy`, `sharedWithOrganisation`
- Mandatory filters on every search: `organisationId` match, `deletedAt` null, visibility check
