"""OCR-only field extraction — schema-aware or open-ended document parsing."""

from __future__ import annotations

import re
from typing import Any

from app.pipeline.ocr import OcrResult
from app.pipeline.title import suggest_display_title


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


def _extract_labeled_pointers(text: str, limit: int = 24) -> list[dict[str, Any]]:
    pointers: list[dict[str, Any]] = []
    seen: set[str] = set()

    for match in re.finditer(
        r"^([A-Za-z][A-Za-z0-9 /&\(\)\.\-]{2,60}?)\s*[:\-]\s*(.+)$",
        text,
        re.MULTILINE,
    ):
        label = re.sub(r"\s+", " ", match.group(1)).strip(" .")
        value = re.sub(r"\s+", " ", match.group(2)).strip()
        if len(value) < 2 or len(value) > 240:
            continue
        key = label.lower()
        if key in seen or key in {"page", "of", "and", "the"}:
            continue
        seen.add(key)
        pointers.append({"label": label, "value": value})
        if len(pointers) >= limit:
            break

    cin = re.search(r"\b([UL]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})\b", text)
    if cin and "cin" not in seen:
        pointers.insert(0, {"label": "CIN", "value": cin.group(1)})

    company = re.search(
        r"(?:name of the company|company name|name of company)\s*[:\-]?\s*([A-Za-z0-9 &.\-]{3,120})",
        text,
        re.I,
    )
    if company:
        pointers.insert(0, {"label": "Company name", "value": company.group(1).strip()})

    return pointers[:limit]


def _page_heading(chunk: str, index: int) -> str:
    for line in chunk.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if 8 <= len(cleaned) <= 80 and not re.match(r"^page\s+\d+", cleaned, re.I):
            return cleaned
    return f"Page {index}"


def _parse_hint_labels(hint: str) -> list[str]:
    """Split free-text extraction context into checklist-style labels."""
    if not hint or not hint.strip():
        return []
    parts = re.split(r"[\n,;•]+|(?:\s+-\s+)", hint)
    labels: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = re.sub(
            r"^(?:check|verify|extract|look\s+for|find)\s+",
            "",
            part.strip(),
            flags=re.I,
        )
        cleaned = cleaned.strip(" .:-")
        if len(cleaned) < 2 or len(cleaned) > 80:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        labels.append(cleaned)
    return labels[:40]


def _extract_from_hint(ocr_text: str, hint: str) -> dict[str, Any]:
    """Try to fill checklist items from extractionHint against OCR text."""
    checklist: dict[str, Any] = {}
    pointers: list[dict[str, Any]] = []
    for label in _parse_hint_labels(hint):
        lower = label.lower()
        # Presence / stamp style
        if any(token in lower for token in ("stamp", "seal", "signature", "signed")):
            present = any(
                token in ocr_text.lower()
                for token in (lower, "stamp", "seal", "signature")
            )
            checklist[label] = present
            pointers.append({"label": label, "value": present})
            continue

        value = _find_after_label(ocr_text, [label])
        if value:
            checklist[label] = value
            pointers.append({"label": label, "value": value})
        else:
            # Fuzzy: label appears somewhere in text
            if re.search(re.escape(label), ocr_text, re.I):
                checklist[label] = True
                pointers.append({"label": label, "value": True})
            else:
                checklist[label] = None
    return {"checklist": checklist, "hint_pointers": pointers}


def _open_ended_extraction(ocr: OcrResult, project: dict[str, Any]) -> dict[str, Any]:
    text = ocr.full_text or ""
    filename = str(project.get("_documentFilename") or "")
    base_conf = max(ocr.average_confidence, 0.5)
    pointers = _extract_labeled_pointers(text)
    in_project = bool(project.get("projectId"))
    hint = (project.get("extractionHint") or "").strip()

    preview = re.sub(r"\s+", " ", text).strip()[:400]
    data: dict[str, Any] = {
        "summary": preview
        or (
            f"Document processed"
            f"{' for project ' + str(project.get('name')) if in_project else ' from Drive'}."
        ),
        "pointers": pointers,
        "key_entities": {item["label"]: item["value"] for item in pointers[:12]},
    }

    if in_project:
        data["project_context"] = project.get("name") or "Project"
        if hint:
            data["project_hint"] = hint
            hint_payload = _extract_from_hint(text, hint)
            if hint_payload["checklist"]:
                data["checklist"] = hint_payload["checklist"]
            # Prefer hint-driven pointers first
            merged = hint_payload["hint_pointers"] + [
                p
                for p in pointers
                if p["label"].lower()
                not in {h["label"].lower() for h in hint_payload["hint_pointers"]}
            ]
            data["pointers"] = merged[:24]
            data["key_entities"] = {
                item["label"]: item["value"] for item in data["pointers"][:12]
            }
        if project.get("description"):
            data["project_description"] = project["description"]
    else:
        data["source"] = "Organisation Drive"

    page_chunks = re.split(r"\f|\n(?=Page\s+\d+)", text)
    pages: list[dict[str, Any]] = []
    for index, chunk in enumerate(page_chunks[:8], start=1):
        chunk_clean = chunk.strip()
        if len(re.sub(r"\s+", " ", chunk_clean)) < 40:
            continue
        pages.append(
            {
                "page": index,
                "title": _page_heading(chunk_clean, index),
                "summary": re.sub(r"\s+", " ", chunk_clean).strip()[:220],
            }
        )
    if pages:
        data["pages"] = pages

    title = suggest_display_title(data, original_filename=filename, ocr_text=text)
    if title:
        data["suggested_title"] = title

    confidence = {
        "summary": round(base_conf * 0.85, 2),
        "pointers": round(base_conf * 0.8, 2) if data["pointers"] else round(base_conf * 0.4, 2),
        "key_entities": round(base_conf * 0.8, 2)
        if data["key_entities"]
        else round(base_conf * 0.4, 2),
    }
    if data.get("checklist"):
        confidence["checklist"] = round(base_conf * 0.78, 2)
    if title:
        confidence["suggested_title"] = round(base_conf * 0.75, 2)

    if not data["pointers"] and text:
        data["ocr_preview"] = text[:2500]
        confidence["ocr_preview"] = round(base_conf, 2)

    return {
        "data": data,
        "fieldConfidence": confidence,
        "strategy": "ocr",
    }


def extract_from_ocr(project: dict[str, Any], ocr: OcrResult) -> dict[str, Any]:
    fields = project.get("fields") or []

    if not fields:
        return _open_ended_extraction(ocr, project)

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
        open_payload = _open_ended_extraction(ocr, project)
        data.update(open_payload["data"])
        confidence.update(open_payload["fieldConfidence"])
    else:
        title = suggest_display_title(
            data,
            original_filename=str(project.get("_documentFilename") or ""),
            ocr_text=ocr.full_text or "",
        )
        if title:
            data["suggested_title"] = title
            confidence["suggested_title"] = round(base_conf * 0.75, 2)

    return {
        "data": data,
        "fieldConfidence": confidence,
        "strategy": "ocr",
    }
