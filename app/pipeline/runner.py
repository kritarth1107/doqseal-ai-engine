"""Full extraction pipeline orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.config import settings
from app.pipeline.ocr import run_ocr
from app.pipeline.ocr_extract import extract_from_ocr
from app.pipeline.preprocess import guess_mime_type, load_document_pages
from app.pipeline.stub import generate_stub_extraction
from app.pipeline.validate import validate_extraction
from app.pipeline.vlm_extract import extract_with_vlm
from app.utils.decrypt import decrypt_document_file

logger = logging.getLogger("doqseal.pipeline")


def _load_plaintext(document: dict[str, Any], organisation_id: str) -> tuple[bytes, str]:
    storage_path = Path(document["storagePath"])
    if not storage_path.is_absolute():
        storage_path = settings.storage_root / storage_path

    ciphertext = storage_path.read_bytes()
    mime_type = document.get("mimeType", "application/pdf")

    if document.get("isEncrypted") and document.get("encryption"):
        if not settings.aes_secret:
            raise ValueError("AES_SECRET is required to decrypt documents")
        plaintext = decrypt_document_file(
            ciphertext,
            document["encryption"],
            organisation_id,
            settings.aes_secret,
        )
        mime_type = guess_mime_type(str(storage_path), mime_type)
        return plaintext, mime_type

    return ciphertext, mime_type


def run_extraction_pipeline(
    document: dict[str, Any],
    project: dict[str, Any],
    organisation_id: str,
) -> dict[str, Any]:
    mode = settings.extraction_mode.lower()

    if mode == "stub":
        payload = generate_stub_extraction(project)
        payload["strategy"] = "stub"
        return payload

    logger.info(
        "Running extraction for document %s (mode=%s)",
        document.get("documentId"),
        mode,
    )

    file_bytes, mime_type = _load_plaintext(document, organisation_id)
    pages = load_document_pages(file_bytes, mime_type)
    if not pages:
        raise ValueError("No pages could be loaded from document")

    ocr = run_ocr(pages)
    logger.info(
        "OCR complete: %d lines, avg confidence %.2f",
        len(ocr.lines),
        ocr.average_confidence,
    )

    extraction: dict[str, Any]

    if mode == "ocr_only":
        extraction = extract_from_ocr(project, ocr)
    else:
        try:
            extraction = extract_with_vlm(project, pages, ocr)
            logger.info("VLM extraction succeeded")
        except Exception as error:
            logger.warning("VLM failed, falling back to OCR-only: %s", error)
            extraction = extract_from_ocr(project, ocr)
            extraction["strategy"] = "ocr_fallback"
            extraction["vlmError"] = str(error)

    validated = validate_extraction(
        extraction.get("data", {}),
        extraction.get("fieldConfidence", {}),
        project,
        settings.confidence_threshold,
    )

    return {
        "data": validated["data"],
        "fieldConfidence": validated["fieldConfidence"],
        "validationErrors": validated["validationErrors"],
        "status": validated["status"],
        "strategy": extraction.get("strategy", mode),
        "ocrFullText": ocr.full_text,
        "ocrLineCount": len(ocr.lines),
        "ocrAverageConfidence": round(ocr.average_confidence, 3),
        "lowConfidenceFields": validated.get("lowConfidenceFields", []),
    }