from __future__ import annotations

from typing import Any


TRF_STUB = {
    "patient_name": "Mrs Afsana Ambir Pinjari",
    "age": 36,
    "sex": "F",
    "clinical_history": "CT Abd & spine",
    "center_stamp": True,
    "medical_officer_stamp": True,
    "medical_superintendent_stamp": False,
}

PRESCRIPTION_STUB = {
    "patient_name": "Mrs Afsana Ambir Pinjari",
    "age": 36,
    "doctor_name": "Dr. Rafat Rehman",
    "medicines": ["Tab Paracetamol 500mg", "Cap Amoxicillin 250mg"],
    "dosage_instructions": "BD after food for 5 days",
    "doctor_stamp": True,
}

INSURANCE_STUB = {
    "insurer_name": "Star Health",
    "policy_number": "SH-2026-88421",
    "pre_auth_id": "PA-77821",
    "patient_name": "Mrs Afsana Ambir Pinjari",
    "coverage_limit_inr": 150000,
    "approval_status": "approved",
}


def _confidence_for_fields(data: dict[str, Any]) -> dict[str, float]:
    return {key: 0.9 for key in data.keys()}


def _field_stub_value(field: dict[str, Any]) -> Any:
    key = field.get("key", "field")
    field_type = field.get("type", "string")

    if field_type == "number":
        return 36
    if field_type == "boolean":
        return True
    if field_type == "date":
        return "2026-03-07"
    return f"Stub value for {field.get('label', key)}"


def _template_for_project(project: dict[str, Any]) -> dict[str, Any] | None:
    name = (project.get("name") or "").lower()
    hint = (project.get("extractionHint") or "").lower()
    combined = f"{name} {hint}"

    if "prescription" in combined:
        return PRESCRIPTION_STUB
    if "insurance" in combined or "tpa" in combined or "pre-auth" in combined:
        return INSURANCE_STUB
    if "test request" in combined or "trf" in combined:
        return TRF_STUB
    return None


def generate_stub_extraction(project: dict[str, Any]) -> dict[str, Any]:
    template = _template_for_project(project)
    fields = project.get("fields") or []

    if template:
        data = dict(template)
    elif fields:
        data = {field["key"]: _field_stub_value(field) for field in fields}
    else:
        data = {
            "document_type": "General document",
            "summary": (
                f"Stub extraction for project '{project.get('name') or 'Drive'}'. "
                "Run OCR/hybrid mode for real field extraction."
            ),
            "pointers": [],
            "key_entities": {},
            "pages": [{"page": 1, "title": "Document", "summary": "Stub mode — no OCR"}],
            "auto_tags": ["stub"],
        }

    if "document_type" not in data:
        data["document_type"] = "Structured extraction"

    field_confidence = _confidence_for_fields(data)
    validation_errors: list[str] = []

    for field in fields:
        if field.get("required") and data.get(field["key"]) in (None, "", []):
            validation_errors.append(f"Missing required field: {field['key']}")

    if validation_errors:
        status = "needs_review"
    elif any(score < 0.85 for score in field_confidence.values()):
        status = "approved_with_warnings"
    else:
        status = "approved"

    return {
        "data": data,
        "fieldConfidence": field_confidence,
        "validationErrors": validation_errors,
        "status": status,
        "strategy": "stub",
    }