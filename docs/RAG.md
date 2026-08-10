# RAG indexing (planned — Wave 1/2)

- Persist `ocrFullText` on extraction
- Chunk OCR + extraction JSON
- Embed with `multilingual-e5-base`
- Store in Qdrant `org_{organisationId}` collection
- Trigger via `indexing.jobs` queue after extraction
