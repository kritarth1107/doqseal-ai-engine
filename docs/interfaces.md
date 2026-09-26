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

## Service authentication (backend -> ai-engine)

All internal HTTP calls carry a short-lived HS256 JWT:
`Authorization: Bearer <jwt>`, signed with `AI_ENGINE_JWT_SECRET`, which must be
identical on the backend and the ai-engine. `AI_ENGINE_SERVICE_TOKEN` is accepted as
an alias on both sides, so one secret covers chat and bundle classification.

Claims: `iss=doqseal-backend`, `aud=doqseal-ai-engine`, `sub` (user id), `org`
(organisation id from the authenticated session), optional `pid` (project),
`scope` (`chat`, `rag:read`, `rag:delete` or `bundle:classify`), `iat`, `exp`
(at most 5 minutes after `iat`), `jti`.

The organisation is always taken from the token. An `organisationId` in a body or
query string is only a consistency check: if it differs from `org` the call is
refused with 403. Missing or invalid token: 401. Wrong scope: 403.

| Endpoint | Secret unset | Secret set |
| --- | --- | --- |
| `POST /v1/chat/stream` | 503 | token required |
| `POST /bundle/classify` | 503 | token required |
| `POST /chat` (legacy) | allowed, logs `SECURITY:` warning | token required |
| `DELETE /rag/documents/{id}` (legacy) | allowed with `?organisationId=`, logs warning | token required |

## HTTP: POST /v1/chat/stream

Request: `{ message, history?: [{ role: "user"|"assistant", content }] (last 10 turns
are used), projectId?, conversationId?, organisationId? }`. Response:
`text/event-stream`, one `event:`/`data:` pair per event, a `: ping` comment every
15 s while idle.

Events, in order:

1. `run.started` `{ runId, conversationId }`
2. `step` `{ id, name, status: started|done, label, detail? }` for `understanding`,
   `retrieving`, `checking_coverage`, `reading`, `generating`, `verifying`
3. `token` `{ text }` (many)
4. `citation` `{ n, documentId, title, page, quote }` (one per `[n]` used)
5. `run.completed` `{ mode: answered|partial|declined, usage{promptTokens,
   completionTokens, totalTokens}, latencyMs }`

Instead of an answer the run can end with `decline` `{ reason: not_covered |
small_talk, message }` followed by `run.completed` (mode `declined`). A decline can
arrive after tokens if the finished answer fails citation checks; clients must then
replace the streamed text with the decline message. Failures end the stream with
`error` `{ code: retrieval_unavailable | model_unavailable, message }`.

Grounding: retrieval is limited to the token's organisation (per-org collection plus
an `organisationId` filter, payload re-checked), then to documents that still exist
in MongoDB, are not deleted and are visible to the user (shared with the
organisation, or uploaded by them). A model judge decides from the excerpts alone
whether they cover the question; if not, the run declines without generating. The
answer must cite only supplied excerpts; answers with no or unknown citations are
declined.

## HTTP: POST /chat (legacy)

Request `{ message, organisationId, projectId?, userId? }` (with a token, `org` and
`sub` come from it). Response `{ answer, citations[{ n, documentId, title, page,
quote, snippet }], thinking[{ title, detail }], mode }`. Runs the same grounded
pipeline without streaming; 503 when retrieval or the model is unavailable.

## HTTP: DELETE /rag/documents/{id}

Removes the document's vectors from the organisation's collection. Scope
`rag:delete`.

## HTTP: POST /bundle/classify (internal)

Called only by the backend bundle pipeline with a `bundle:classify` token whose `org`
must equal `organisationId` in the body.

Request: `{ requestId, organisationId, bundleId, documentId, slots: [{ key, label,
description?, hints[] }], document: { documentType?, text?, fields{} }, keyFieldNames[] }`

Response: `{ requestId, organisationId, bundleId, documentId, slot | null, confidence,
reasons[], alternatives[{ slot, confidence }], keyFields{}, model, cached }`

The document must exist for that organisation (404 otherwise) before any model
call. Uses the existing text model deployment. Responses are cached per
(organisation, requestId) so retries do not call the model twice. Errors: 401 bad
token, 403 wrong scope or org mismatch, 404 unknown document, 502 retryable model
error, 422 non-retryable, 503 when no secret is configured.
