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
            return _clean_title(raw)

    entities = payload.get("key_entities")
    if isinstance(entities, dict):
        for label in (
            "Company name",
            "company_name",
            "Name of the company",
            "Patient name",
            "patient_name",
            "Title",
            "Document title",
        ):
            value = entities.get(label)
            if isinstance(value, str) and len(value.strip()) >= 3:
                about = payload.get("summary") or payload.get("document_type")
                if isinstance(about, str) and about.strip():
                    short = about.strip().split(".")[0][:40]
                    return _clean_title(f"{value.strip()} — {short}")
                return _clean_title(value)

    pointers = payload.get("pointers")
    if isinstance(pointers, list):
        for item in pointers[:8]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").lower()
            value = item.get("value")
            if not isinstance(value, str) or len(value.strip()) < 3:
                continue
            if any(
                token in label
                for token in ("company", "patient", "title", "name of", "subject")
            ):
                return _clean_title(value)

    summary = payload.get("summary")
    if isinstance(summary, str) and len(summary.strip()) >= 12:
        first = summary.strip().split(".")[0]
        if len(first) >= 8:
            return _clean_title(first)

    # Light content sniff (no fixed document-type catalog)
    hay = f"{original_filename}\n{ocr_text[:2500]}"
    company = re.search(
        r"(?:name of the company|company name)\s*[:\-]?\s*([A-Za-z0-9 &.\-]{3,80})",
        hay,
        re.I,
    )
    if company:
        return _clean_title(company.group(1))

    patient = re.search(
        r"(?:patient(?:'s)?\s*name|name of patient)\s*[:\-]?\s*([A-Za-z .]{3,60})",
        hay,
        re.I,
    )
    if patient:
        return _clean_title(patient.group(1))

    # First meaningful OCR line as last resort
    for line in (ocr_text or "").splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if len(cleaned) < 8 or len(cleaned) > 90:
            continue
        if re.search(r"^\d+$|page\s+\d+|form\s+no", cleaned, re.I):
            continue
        return _clean_title(cleaned)

    return None
