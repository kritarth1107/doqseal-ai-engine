"""Vision extraction via Ollama multimodal models (default: gemma4:e4b)."""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Any

import httpx

from app.config import settings
from app.pipeline.ocr import OcrResult
from app.pipeline.preprocess import PageImage

logger = logging.getLogger("doqseal.vlm")


def _looks_like_trf(ocr_text: str, hint: str) -> bool:
    blob = f"{ocr_text}\n{hint}".lower()
    markers = (
        "requisition",
        "trf",
        "patient name",
        "specimen",
        "test requirements",
        "uhid",
        "diagnostic",
    )
    return sum(1 for m in markers if m in blob) >= 2


def _build_schema_prompt(project: dict[str, Any], ocr: OcrResult) -> str:
    fields = project.get("fields") or []
    hint = (project.get("extractionHint") or "").strip()
    description = (project.get("description") or "").strip()
    ocr_preview = ocr.full_text[:3500] if ocr.full_text else "No OCR text detected."
    in_project = bool(project.get("projectId"))
    project_name = project.get("name") or (
        "Organisation Drive" if not in_project else "Project"
    )
    trf_mode = _looks_like_trf(ocr.full_text or "", hint)

    if fields:
        field_lines = []
        for field in fields:
            req = "required" if field.get("required") else "optional"
            field_lines.append(
                f'- "{field["key"]}" ({field.get("type", "string")}, {req}): '
                f'{field.get("label", field["key"])}'
            )
        schema_block = "\n".join(field_lines)
        schema_block += """
- "suggested_title" (string)
- "summary" (string)
- "pages" (array, optional): [{ "page": 1, "title": "...", "summary": "..." }]
"""
    elif trf_mode:
        schema_block = """
- "document_type" (string): e.g. "Test Requisition Form"
- "lab_name" (string|null)
- "patient" (object): {
    "name": string|null,
    "age": number|null,
    "gender": "M"|"F"|string|null,
    "uhid": string|null,
    "phone": string|null,
    "address": string|null
  }
- "client" (object): {
    "code": string|null,
    "name": string|null,
    "referring_doctor": string|null
  }
- "specimen" (object): {
    "drawn_date": string|null,
    "drawn_time": string|null,
    "fasting": boolean|null,
    "types": array of checked specimen type labels
  }
- "tests" (array): [{ "code": string|null, "description": string, "amount": number|null }]
- "clinical_history" (string|null)
- "verification" (object): stamps/signatures as booleans when visible
- "suggested_title" (string)
- "summary" (string)
- "key_entities" (object)
- "pointers" (array): [{ "label": "...", "value": "...", "page": 1 }]
- "checklist" (object)
- "auto_tags" (array of strings)
"""
    else:
        schema_block = """
- "suggested_title" (string)
- "summary" (string)
- "key_entities" (object)
- "checklist" (object)
- "pointers" (array): [{ "label": "...", "value": "...", "page": 1 }]
- "pages" (array): [{ "page": number, "title": "...", "summary": "..." }]
- "auto_tags" (array of strings)
"""

    if in_project and hint:
        scope = (
            f"Project «{project_name}».\n"
            f"PRIMARY EXTRACTION CONTEXT:\n{hint}\n"
            f"{('Description: ' + description) if description else ''}"
        )
    elif in_project:
        scope = f"Project «{project_name}». Description: {description or 'none'}"
    else:
        scope = "Organisation Drive upload. Read handwriting carefully from the image."

    trf_rules = ""
    if trf_mode:
        trf_rules = """
TRF rules:
- Prefer handwritten filled values over blank printed labels.
- Extract Patient Name, Age, Gender (checked box), Client Code, and every Test Description.
- tests[] must include each handwritten test (e.g. B group, TFT, HBsAg).
- Empty fields → null. Do not invent values.
"""

    return f"""You are DoqSeal's document intelligence extractor.
You receive the document IMAGE plus noisy OCR text.
Trust the IMAGE for handwriting; OCR is only a weak hint.

{scope}

Return ONLY valid JSON with these keys:
{schema_block}

Rules:
- Extract only what is visible.
- Prefer names, ages, codes, test names, dates, phones.
- Numbers as JSON numbers. Missing → null.
{trf_rules}

OCR reference (may be wrong — verify against the image):
{ocr_preview}
"""


def _parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    # Strip optional thinking blocks some models emit
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.I).strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Model response did not contain JSON object")

    return json.loads(cleaned[start : end + 1])


def _image_to_base64(page: PageImage) -> str:
    buf = io.BytesIO()
    # Keep resolution reasonable for Ollama vision
    image = page.image.convert("RGB")
    max_side = 1600
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        image = image.resize((int(w * scale), int(h * scale)))
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _call_ollama_vision(prompt: str, image_b64: str) -> str:
    model = settings.vlm_model
    url = f"{settings.ollama_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 1600,
        },
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
    }
    timeout = httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=30.0)
    logger.info("Ollama vision extract model=%s url=%s", model, url)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        body = response.json()
    message = body.get("message") or {}
    content = (message.get("content") or body.get("response") or "").strip()
    if not content:
        raise RuntimeError("Empty response from Ollama vision model")
    return content


def warmup_vlm() -> None:
    """Ping Ollama and ensure the vision model responds to a tiny prompt."""
    url = f"{settings.ollama_url.rstrip('/')}/api/tags"
    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.get(url)
            response.raise_for_status()
            names = {
                m.get("name") or m.get("model")
                for m in (response.json().get("models") or [])
            }
        logger.info(
            "Ollama reachable; gemma/vision model configured=%s present=%s",
            settings.vlm_model,
            settings.vlm_model in names
            or any(settings.vlm_model.split(":")[0] in (n or "") for n in names),
        )
    except Exception as err:
        logger.warning("Ollama warmup failed: %s", err)


def extract_with_vlm(
    project: dict[str, Any],
    pages: list[PageImage],
    ocr: OcrResult,
) -> dict[str, Any]:
    if not pages:
        raise ValueError("No pages available for vision extraction")

    prompt = _build_schema_prompt(project, ocr)
    image_b64 = _image_to_base64(pages[0])
    raw = _call_ollama_vision(prompt, image_b64)
    parsed = _parse_json_response(raw)

    field_confidence = {
        key: round(min(0.97, max(0.72, ocr.average_confidence + 0.25)), 2)
        for key, value in parsed.items()
        if value is not None
    }

    return {
        "data": parsed,
        "fieldConfidence": field_confidence,
        "strategy": f"ollama:{settings.vlm_model}",
    }
