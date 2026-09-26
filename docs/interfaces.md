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

## HTTP: POST /bundle/classify (internal)

Called only by the backend bundle pipeline. Disabled (503) unless
`AI_ENGINE_SERVICE_TOKEN` is set on the ai-engine.

Headers: `X-Service-Token: <AI_ENGINE_SERVICE_TOKEN>`, `X-Organisation-Id: <org>`
(must equal `organisationId` in the body).

Request: `{ requestId, organisationId, bundleId, documentId, slots: [{ key, label,
description?, hints[] }], document: { documentType?, text?, fields{} }, keyFieldNames[] }`

Response: `{ requestId, organisationId, bundleId, documentId, slot | null, confidence,
reasons[], alternatives[{ slot, confidence }], keyFields{}, model, cached }`

The document must exist for that organisation (404 otherwise) before any model
call. Uses the existing Azure OpenAI text deployment. Responses are cached per
(organisation, requestId) so retries do not call the model twice. Errors: 401 bad
token, 400 org mismatch, 404 unknown document, 502 retryable model error, 422
non-retryable.
