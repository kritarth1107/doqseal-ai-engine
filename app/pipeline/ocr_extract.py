"""OCR-only field extraction — real text parsing without VLM."""

from __future__ import annotations

import re
from typing import Any

from app.pipeline.ocr import OcrResult


def _find_after_label(text: str, labels: list[str]) -> str | None:
    for label in labels:
        pattern = rf"{re.escape(label)}\s*[:\-]?\s*(.+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).split("\n")[0].strip()
    return None


def _guess_from_schema(field: dict[str, Any], ocr_text: str) -> tuple[Any, float]:
    key = field.get("key", "")
    label = field.get("label", key)
    field_type = field.get("type", "string")
    labels = [label, key.replace("_", " ")]

    if field_type == "number":
        if key == "age":
            match = re.search(r"\b(?:age|aged)\s*[:\-]?\s*(\d{1,3})\b", ocr_text, re.I)
            if match:
                return int(match.group(1)), 0.82
        match = re.search(rf"{re.escape(label)}\s*[:\-]?\s*(\d+(?:\.\d+)?)", ocr_text, re.I)
        if match:
            return float(match.group(1)), 0.78
        return None, 0.0

    if field_type == "boolean":
        stamp_hints = ["stamp", "seal", "signature", "signed"]
        if any(hint in key.lower() for hint in stamp_hints):
            present = any(
                hint in ocr_text.lower()
                for hint in [label.lower(), key.replace("_", " ").lower(), "stamp", "seal"]
            )
            return present, 0.7 if present else 0.55
        return None, 0.0

    value = _find_after_label(ocr_text, labels)
    if value:
        return value, 0.8

    if key == "patient_name":
        match = re.search(
            r"(?:patient(?:'s)?\s*name|name\s*of\s*patient)\s*[:\-]?\s*([A-Za-z .]{3,60})",
            ocr_text,
            re.I,
        )
        if match:
            return match.group(1).strip(), 0.85

    if key == "sex" or key == "gender":
        match = re.search(r"\b(?:sex|gender)\s*[:\-]?\s*([MF]|male|female)\b", ocr_text, re.I)
        if match:
            val = match.group(1).upper()[0]
            return val, 0.83

    if key == "clinical_history":
        match = re.search(
            r"(?:clinical\s*history|history)\s*[:\-]?\s*(.+)",
            ocr_text,
            re.I,
        )
        if match:
            return match.group(1).split("\n")[0].strip(), 0.8

    if key == "medicines":
        meds = re.findall(r"(?:tab|cap|syr|inj)\.?\s+[A-Za-z0-9 +\-]{3,40}", ocr_text, re.I)
        if meds:
            return ", ".join(meds[:8]), 0.75

    return None, 0.0


def extract_from_ocr(project: dict[str, Any], ocr: OcrResult) -> dict[str, Any]:
    fields = project.get("fields") or []
    data: dict[str, Any] = {}
    confidence: dict[str, float] = {}

    base_conf = max(ocr.average_confidence, 0.5)

    for field in fields:
        key = field["key"]
        value, field_conf = _guess_from_schema(field, ocr.full_text)
        if value is not None and value != "":
            data[key] = value
            confidence[key] = round(min(0.95, (field_conf + base_conf) / 2), 2)
        elif field.get("required"):
            confidence[key] = round(base_conf * 0.4, 2)

    if not data and ocr.full_text:
        data["_raw_ocr_preview"] = ocr.full_text[:2000]
        confidence["_raw_ocr_preview"] = round(base_conf, 2)

    return {
        "data": data,
        "fieldConfidence": confidence,
        "strategy": "ocr",
    }