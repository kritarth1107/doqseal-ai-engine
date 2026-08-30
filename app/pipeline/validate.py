from __future__ import annotations

import re
from datetime import datetime
from typing import Any


def _coerce_value(field_type: str, raw: Any) -> Any:
    if raw is None:
        return None

    if field_type == "number":
        if isinstance(raw, (int, float)):
            return raw
        match = re.search(r"-?\d+(?:\.\d+)?", str(raw))
        return float(match.group()) if match else None

    if field_type == "boolean":
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in {"true", "yes", "y", "1", "present", "detected"}:
            return True
        if text in {"false", "no", "n", "0", "absent", "not detected"}:
            return False
        return None

    if field_type == "date":
        text = str(raw).strip()
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y"):
            try:
                return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return text or None

    if isinstance(raw, list):
        return ", ".join(str(item) for item in raw)
    return str(raw).strip() if str(raw).strip() else None


def _confidence_for_open_data(data: dict[str, Any], existing: dict[str, float]) -> dict[str, float]:
    confidence: dict[str, float] = dict(existing)
    for key, value in data.items():
        if key in confidence:
            continue
        if value in (None, "", [], {}):
            continue
        confidence[key] = 0.75
    return confidence


def validate_extraction(
    data: dict[str, Any],
    field_confidence: dict[str, float],
    project: dict[str, Any],
    confidence_threshold: float,
) -> dict[str, Any]:
    fields = project.get("fields") or []
    errors: list[str] = []

    # No project schema → keep open-ended extraction (do not wipe to {})
    if not fields:
        cleaned = {
            key: value
            for key, value in (data or {}).items()
            if value not in (None, "")
        }
        confidence = _confidence_for_open_data(cleaned, field_confidence or {})
        low_confidence_fields = [
            key for key, score in confidence.items() if score < confidence_threshold
        ]
        status = "approved_with_warnings" if low_confidence_fields else "approved"
        if not cleaned:
            status = "needs_review"
            errors.append("No structured fields could be extracted")
        return {
            "data": cleaned,
            "fieldConfidence": confidence,
            "validationErrors": errors,
            "status": status,
            "lowConfidenceFields": low_confidence_fields,
        }

    coerced: dict[str, Any] = {}

    for field in fields:
        key = field["key"]
        field_type = field.get("type", "string")
        raw_value = data.get(key)
        value = _coerce_value(field_type, raw_value)
        coerced[key] = value

        if field.get("required") and value in (None, "", []):
            errors.append(f"Missing required field: {key}")
            continue

        rules = field.get("validate") or {}
        if value is not None and field_type == "number" and isinstance(value, (int, float)):
            if "min" in rules and value < rules["min"]:
                errors.append(f"{key} below minimum ({rules['min']})")
            if "max" in rules and value > rules["max"]:
                errors.append(f"{key} above maximum ({rules['max']})")

    # Preserve useful open keys the model returned beyond the schema
    for key, value in (data or {}).items():
        if key in coerced:
            continue
        if key in {"document_type", "summary", "pages", "pointers", "key_entities", "auto_tags", "suggested_title"}:
            if value not in (None, "", []):
                coerced[key] = value

    low_confidence_fields = [
        key
        for key, score in field_confidence.items()
        if score < confidence_threshold
    ]

    if errors:
        status = "needs_review"
    elif low_confidence_fields:
        status = "approved_with_warnings"
    else:
        status = "approved"

    return {
        "data": coerced,
        "fieldConfidence": field_confidence,
        "validationErrors": errors,
        "status": status,
        "lowConfidenceFields": low_confidence_fields,
    }
