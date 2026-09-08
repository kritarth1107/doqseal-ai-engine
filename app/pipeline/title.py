"""Build a human-readable document title from extraction / OCR content."""

from __future__ import annotations

import re
from typing import Any


def _clean_title(value: str, max_len: int = 80) -> str:
    text = re.sub(r"\s+", " ", value).strip(" -_|.")
    text = re.sub(r"[\\\\/:*?\"<>|]+", " ", text)
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text


def _is_paragraph_title(text: str) -> bool:
    t = text.strip()
    return (
        len(t) > 90
        or t.count(" ") > 14
        or t.lower().startswith(("this ", "the document", "the prescription"))
    )


def suggest_display_title(
    data: dict[str, Any] | None,
    *,
    original_filename: str = "",
    ocr_text: str = "",
) -> str | None:
    """Return a short title describing the document, or None to keep the filename."""
    payload = data or {}

    for key in ("suggested_title", "title", "display_title"):
        raw = payload.get(key)
        if isinstance(raw, str) and len(raw.strip()) >= 3:
            if _is_paragraph_title(raw):
                continue
            return _clean_title(raw)

    patient = payload.get("patient")
    if isinstance(patient, dict):
        name = patient.get("name")
        if isinstance(name, str) and name.strip():
            return _clean_title(f"{name.strip()} — Prescription")

    entities = payload.get("key_entities")
    if isinstance(entities, dict):
        for label in (
            "Company name",
            "company_name",
            "Name of the company",
            "brand",
            "Brand",
            "Patient name",
            "patient_name",
            "Title",
            "Document title",
        ):
            value = entities.get(label)
            if isinstance(value, str) and len(value.strip()) >= 3:
                return _clean_title(value)

    doc_type = payload.get("document_type")
    if isinstance(doc_type, str) and doc_type.strip():
        return _clean_title(doc_type.replace("_", " ").title())

    # Light OCR fallback — never use long summaries
    hay = f"{original_filename}\n{(ocr_text or '')[:2500]}"
    company = re.search(
        r"(?:name of the company|company name)\s*[:\-]?\s*([A-Za-z0-9 &.\-]{3,80})",
        hay,
        re.I,
    )
    if company:
        return _clean_title(company.group(1))

    for line in (ocr_text or "").splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if len(cleaned) < 8 or len(cleaned) > 90:
            continue
        if re.search(r"^\d+$|page\s+\d+|form\s+no", cleaned, re.I):
            continue
        if _is_paragraph_title(cleaned):
            continue
        return _clean_title(cleaned)

    return None
