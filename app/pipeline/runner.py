"""Full extraction pipeline orchestrator."""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings
from app.pipeline.ocr import ocr_result_from_text, run_ocr
from app.pipeline.ocr_extract import extract_from_ocr
from app.pipeline.preprocess import (
    extract_pdf_text_layers,
    guess_mime_type,
    load_document_pages,
)
from app.pipeline.stub import generate_stub_extraction
from app.pipeline.openai_extract import (
    azure_openai_configured,
    extract_with_azure_openai,
)
from app.pipeline.title import suggest_display_title
from app.pipeline.validate import validate_extraction
from app.pipeline.vlm_extract import extract_with_vlm
from app.utils.blob_storage import load_ciphertext
from app.utils.decrypt import decrypt_document_file

logger = logging.getLogger("doqseal.pipeline")


def _empty_ocr():
    return ocr_result_from_text("", confidence=0.0)


def _load_plaintext(document: dict[str, Any], organisation_id: str) -> tuple[bytes, str]:
    ciphertext = load_ciphertext(document)
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
        mime_type = guess_mime_type(
            str(document.get("storagePath") or document.get("storageUri") or ""),
            mime_type,
        )
        return plaintext, mime_type

    return ciphertext, mime_type


def _should_skip_vlm(
    ocr_text: str,
    average_confidence: float,
    *,
    is_image: bool,
) -> bool:
    # Photos / scans of handwritten forms need the vision model — OCR alone is poor.
    if is_image:
        return False
    text_len = len((ocr_text or "").strip())
    return (
        text_len >= settings.skip_vlm_min_text_chars
        and average_confidence >= settings.skip_vlm_min_ocr_confidence
    )


def run_extraction_pipeline(
    document: dict[str, Any],
    project: dict[str, Any],
    organisation_id: str,
) -> dict[str, Any]:
    mode = settings.extraction_mode.lower()

    if mode == "stub":
        payload = generate_stub_extraction(project)
        payload["strategy"] = "stub"
        title = suggest_display_title(
            payload.get("data"),
            original_filename=str(document.get("originalFilename") or ""),
        )
        if title:
            payload["displayTitle"] = title
            if isinstance(payload.get("data"), dict):
                payload["data"].setdefault("suggested_title", title)
        return payload

    logger.info(
        "Running extraction for document %s (mode=%s)",
        document.get("documentId"),
        mode,
    )

    file_bytes, mime_type = _load_plaintext(document, organisation_id)
    mime_l = mime_type.lower()
    is_pdf = "pdf" in mime_l
    is_image = any(tok in mime_l for tok in ("image/", "jpeg", "jpg", "png", "webp"))

    project = {
        **project,
        "_documentFilename": document.get("originalFilename") or "",
    }

    # Fast path: born-digital PDFs often have a text layer — skip EasyOCR + VLM
    pdf_text = ""
    pdf_pages_with_text = 0
    if is_pdf and settings.prefer_pdf_text:
        pdf_text, pdf_pages_with_text = extract_pdf_text_layers(file_bytes)
        logger.info(
            "PDF text layer: %d chars across %d pages",
            len(pdf_text),
            pdf_pages_with_text,
        )

    if (
        is_pdf
        and settings.prefer_pdf_text
        and len(pdf_text) >= settings.pdf_text_min_chars
    ):
        ocr = ocr_result_from_text(pdf_text, confidence=0.93)
        extraction = extract_from_ocr(project, ocr)
        extraction["strategy"] = "pdf_text"
        logger.info("Using PDF text-layer extraction (skipped OCR/VLM)")
    else:
        pages = load_document_pages(file_bytes, mime_type)
        if not pages:
            raise ValueError("No pages could be loaded from document")

        use_azure = (
            settings.vlm_provider.lower() == "azure_openai"
            and azure_openai_configured()
            and mode != "ocr_only"
        )
        # Fast handwritten/image path: GPT-4o only (skip EasyOCR) → target <10s
        fast_vision = use_azure and (
            is_image or settings.skip_ocr_for_vision
        )

        if fast_vision:
            ocr = _empty_ocr()
            try:
                extraction = extract_with_azure_openai(project, pages)
                logger.info("Azure OpenAI GPT-4o extraction succeeded")
            except Exception as error:
                logger.warning(
                    "Azure OpenAI failed, falling back to OCR/Ollama: %s", error
                )
                ocr = run_ocr(pages)
                try:
                    extraction = extract_with_vlm(project, pages, ocr)
                except Exception as ollama_err:
                    extraction = extract_from_ocr(project, ocr)
                    extraction["strategy"] = "ocr_fallback"
                    extraction["vlmError"] = f"{error}; {ollama_err}"
        else:
            ocr = run_ocr(pages)
            logger.info(
                "OCR complete: %d lines, avg confidence %.2f (image=%s)",
                len(ocr.lines),
                ocr.average_confidence,
                is_image,
            )

            if pdf_text and len(pdf_text) > 40:
                merged = f"{pdf_text}\n\n{ocr.full_text}".strip()
                ocr = ocr_result_from_text(
                    merged,
                    confidence=max(ocr.average_confidence, 0.7),
                )

            skip_vlm = mode == "ocr_only" or _should_skip_vlm(
                ocr.full_text,
                ocr.average_confidence,
                is_image=is_image,
            )

            if skip_vlm:
                extraction = extract_from_ocr(project, ocr)
                if mode != "ocr_only":
                    extraction["strategy"] = "ocr_fast"
                    logger.info(
                        "Skipping VLM (text=%d chars, conf=%.2f)",
                        len(ocr.full_text or ""),
                        ocr.average_confidence,
                    )
            else:
                try:
                    if use_azure:
                        extraction = extract_with_azure_openai(project, pages)
                        logger.info("Azure OpenAI GPT-4o extraction succeeded")
                    else:
                        extraction = extract_with_vlm(project, pages, ocr)
                        logger.info("Ollama VLM extraction succeeded")
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

    display_title = suggest_display_title(
        validated.get("data"),
        original_filename=str(document.get("originalFilename") or ""),
        ocr_text=ocr.full_text or "",
    )
    if display_title and isinstance(validated.get("data"), dict):
        validated["data"].setdefault("suggested_title", display_title)

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
        "displayTitle": display_title,
    }
