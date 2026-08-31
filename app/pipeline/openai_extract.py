"""Fast vision extraction via Azure OpenAI GPT-4o (target <10s)."""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Any

import httpx
from PIL import Image, ImageOps

from app.config import settings
from app.pipeline.preprocess import PageImage

logger = logging.getLogger("doqseal.openai_vlm")


def azure_openai_configured() -> bool:
    return bool(
        (settings.azure_openai_endpoint or "").strip()
        and (settings.azure_openai_api_key or "").strip()
        and (settings.azure_openai_deployment or "").strip()
    )


def _pil_to_b64_jpeg(image: Image.Image, *, max_side: int = 1600, quality: int = 85) -> str:
    img = ImageOps.exif_transpose(image.convert("RGB"))
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_json_response(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    # Strip markdown fences if present
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def _expand_tests(value: Any) -> Any:
    if isinstance(value, list):
        value = ", ".join(str(v).strip() for v in value if str(v).strip())
    if not isinstance(value, str) or not value.strip():
        return value
    mapping = {
        r"\bcbc\b": "Complete Blood Count (CBC)",
        r"\btsh\b": "Thyroid Stimulating Hormone (TSH)",
        r"\btft\b": "Thyroid Function Test (TFT)",
        r"\bhbsag\b": "Hepatitis B Surface Antigen (HBsAg)",
        r"\bhba1c\b": "Glycated Hemoglobin (HbA1c)",
        r"\bcrp\b": "C-Reactive Protein (CRP)",
        r"\besr\b": "Erythrocyte Sedimentation Rate (ESR)",
        r"\bcreat(?:inine)?\b": "Creatinine",
        r"\blipid\s*pr(?:ofile)?\b": "Lipid Profile",
        r"\bb\s*group\b": "Blood Group",
        r"\blfts?\b": "Liver Function Test (LFT)",
        r"\burea\b": "Urea",
    }
    parts = [p.strip() for p in re.split(r"[,;\n]+", value) if p.strip()]
    expanded: list[str] = []
    for part in parts:
        out = part
        replaced = False
        for pattern, full in mapping.items():
            if re.search(pattern, part, re.I):
                # If already contains full form, keep as-is
                if full.split(" (")[0].lower() in part.lower() and "(" in part:
                    out = part
                else:
                    out = full
                replaced = True
                break
        expanded.append(out if replaced else part)
    # Dedupe preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for item in expanded:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    return ", ".join(uniq)


def _build_prompt(project: dict[str, Any]) -> str:
    fields = project.get("fields") or []
    hint = (project.get("extractionHint") or "").strip()
    name = (project.get("name") or "").strip()

    if fields:
        field_lines = "\n".join(
            f'- "{f["key"]}" ({f.get("type", "string")}): {f.get("label", f["key"])}'
            for f in fields
            if f.get("key")
        )
    else:
        field_lines = """
- "patient_name" (string|null)
- "patient_age" (number|null)
- "patient_gender" (string|null): Male / Female / Transgender from checked box
- "client_code" (string|null)
- "tests_requested" (string|null): comma-separated; expand abbreviations to full forms
- "lab_name" (string|null)
""".strip()

    return f"""You are extracting structured data from a medical Test Requisition Form photo (often handwritten).

Project: {name or "TRF"}
Instructions: {hint or "Extract patient and test fields accurately."}

Return ONE minified JSON object with exactly these keys when present:
{field_lines}
- "suggested_title" (string): "{{Patient Name}} — {{tests}}"
- "summary" (string): one short sentence from filled values only

Rules:
- Read ONLY handwritten ink and clearly checked boxes.
- Prefer null over guessing.
- Expand clear test abbreviations (CBC→Complete Blood Count (CBC), TSH→Thyroid Stimulating Hormone (TSH), TFT→Thyroid Function Test (TFT), HBsAg→Hepatitis B Surface Antigen (HBsAg), HbA1c→Glycated Hemoglobin (HbA1c), CRP→C-Reactive Protein (CRP), ESR→Erythrocyte Sedimentation Rate (ESR), Creat→Creatinine, Lipid pr→Lipid Profile, B group→Blood Group).
- Ignore Specimen Type checkboxes (Serum, EDTA, Urine) — those are not tests.
- Ignore printed purple labels / OCR noise.
- JSON only, no markdown.
"""


def extract_with_azure_openai(
    project: dict[str, Any],
    pages: list[PageImage],
) -> dict[str, Any]:
    if not azure_openai_configured():
        raise RuntimeError("Azure OpenAI is not configured")
    if not pages:
        raise ValueError("No pages available for vision extraction")

    image_b64 = _pil_to_b64_jpeg(pages[0].image)
    prompt = _build_prompt(project)

    endpoint = settings.azure_openai_endpoint.rstrip("/")
    deployment = settings.azure_openai_deployment
    api_version = settings.azure_openai_api_version
    url = (
        f"{endpoint}/openai/deployments/{deployment}/chat/completions"
        f"?api-version={api_version}"
    )

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 1200,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "api-key": settings.azure_openai_api_key,
        "Content-Type": "application/json",
    }

    logger.info(
        "Azure OpenAI vision extract deployment=%s detail=high",
        deployment,
    )
    timeout = httpx.Timeout(connect=20.0, read=60.0, write=30.0, pool=20.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()

    content = (
        (((body.get("choices") or [{}])[0].get("message") or {}).get("content"))
        or ""
    ).strip()
    parsed = _parse_json_response(content)
    if not parsed:
        raise RuntimeError("Azure OpenAI returned empty/invalid JSON")

    if "tests_requested" in parsed:
        parsed["tests_requested"] = _expand_tests(parsed.get("tests_requested"))

    # Drop empty strings
    cleaned = {
        k: v
        for k, v in parsed.items()
        if v is not None and v != ""
    }

    field_confidence = {
        key: 0.93
        for key, value in cleaned.items()
        if value is not None and value != ""
    }
    for key in (
        "patient_name",
        "patient_age",
        "patient_gender",
        "client_code",
        "tests_requested",
    ):
        if cleaned.get(key) not in (None, ""):
            field_confidence[key] = 0.95

    return {
        "data": cleaned,
        "fieldConfidence": field_confidence,
        "strategy": f"azure-openai:{deployment}",
    }
