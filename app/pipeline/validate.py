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


def validate_extraction(
    data: dict[str, Any],
    field_confidence: dict[str, float],
    project: dict[str, Any],
    confidence_threshold: float,
) -> dict[str, Any]:
    fields = project.get("fields") or []
    errors: list[str] = []
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