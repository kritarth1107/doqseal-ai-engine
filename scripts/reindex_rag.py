#!/usr/bin/env python3
"""Re-index all documents with the improved v2 chunker.

This script reads all completed extractions from MongoDB and re-indexes
them into Qdrant using the new page-aware chunker with metadata.

WARNING: Do NOT run this against production without coordination.
This is a batch operation that will:
1. Read all extractions from MongoDB
2. Re-chunk using the v2 chunker
3. Upsert into Qdrant (same point IDs, so existing vectors are updated)

Usage:
    # Dry run (just count, no changes)
    python scripts/reindex_rag.py --dry-run

    # Single organisation
    python scripts/reindex_rag.py --org-id org_abc123

    # All organisations (requires --confirm)
    python scripts/reindex_rag.py --all --confirm

Environment variables:
    MONGODB_URI: MongoDB connection string
    QDRANT_URL: Qdrant HTTP endpoint
    QDRANT_API_KEY: Qdrant API key (optional)
"""

import argparse
import logging
import sys
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("reindex")


def get_extractions(db, organisation_id: str | None = None) -> list[dict[str, Any]]:
    """Load extractions to re-index."""
    query: dict[str, Any] = {}
    if organisation_id:
        query["organisationId"] = organisation_id

    cursor = db.extractions.find(
        query,
        {
            "extractionId": 1,
            "documentId": 1,
            "jobId": 1,
            "organisationId": 1,
            "projectId": 1,
            "data": 1,
            "ocrFullText": 1,
        },
    )

    return list(cursor)


def get_document_metadata(db, document_id: str) -> dict[str, Any]:
    """Get document metadata for chunking."""
    doc = db.documents.find_one(
        {"documentId": document_id},
        {
            "displayTitle": 1,
            "originalFilename": 1,
            "uploadedBy": 1,
            "sharedWithOrganisation": 1,
        },
    )
    return doc or {}


def reindex_extraction(
    extraction: dict[str, Any],
    doc_meta: dict[str, Any],
    *,
    dry_run: bool = False,
) -> int:
    """Re-index a single extraction."""
    from app.rag.indexer import index_extraction

    organisation_id = extraction.get("organisationId")
    document_id = extraction.get("documentId")
    job_id = extraction.get("jobId")
    project_id = extraction.get("projectId")
    ocr_text = extraction.get("ocrFullText")
    extraction_data = extraction.get("data")

    title = doc_meta.get("displayTitle") or doc_meta.get("originalFilename")
    doc_type = (extraction_data or {}).get("document_type")
    uploaded_by = doc_meta.get("uploadedBy")
    shared = doc_meta.get("sharedWithOrganisation", True)

    if dry_run:
        logger.info(
            "Would reindex: doc=%s org=%s title=%s",
            document_id,
            organisation_id,
            title,
        )
        return 0

    return index_extraction(
        organisation_id=organisation_id,
        document_id=document_id,
        job_id=job_id,
        project_id=project_id,
        ocr_full_text=ocr_text,
        extraction_data=extraction_data,
        uploaded_by=uploaded_by,
        shared_with_organisation=shared if shared is not None else True,
        document_title=title,
        document_type=doc_type,
        use_v2_chunker=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Re-index documents with improved chunker",
    )
    parser.add_argument(
        "--org-id",
        help="Organisation ID to re-index (required unless --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Re-index all organisations",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm bulk re-index operation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just count, don't make changes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of documents to process",
    )

    args = parser.parse_args()

    if not args.org_id and not args.all:
        parser.error("Either --org-id or --all is required")

    if args.all and not args.confirm and not args.dry_run:
        parser.error("--all requires --confirm (or use --dry-run)")

    from app.db.mongo import get_db

    db = get_db()

    logger.info("Fetching extractions...")
    extractions = get_extractions(db, args.org_id)

    if args.limit:
        extractions = extractions[: args.limit]

    logger.info("Found %d extractions to process", len(extractions))

    if not extractions:
        logger.info("Nothing to do")
        return

    total_chunks = 0
    processed = 0
    errors = 0

    for extraction in extractions:
        document_id = extraction.get("documentId")
        try:
            doc_meta = get_document_metadata(db, document_id)
            chunks = reindex_extraction(extraction, doc_meta, dry_run=args.dry_run)
            total_chunks += chunks
            processed += 1

            if processed % 100 == 0:
                logger.info("Processed %d/%d documents", processed, len(extractions))

        except Exception as exc:
            logger.error("Failed to reindex %s: %s", document_id, exc)
            errors += 1

    logger.info(
        "Completed: %d processed, %d errors, %d total chunks",
        processed,
        errors,
        total_chunks,
    )


if __name__ == "__main__":
    main()
