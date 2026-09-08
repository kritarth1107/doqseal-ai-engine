"""Token-efficient Azure OpenAI extraction (vision + text)."""

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


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _pil_to_b64_jpeg(
    image: Image.Image,
    *,
    max_side: int,
    quality: int,
) -> str:
    img = ImageOps.exif_transpose(image.convert("RGB"))
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_json_response(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
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
                if full.split(" (")[0].lower() in part.lower() and "(" in part:
                    out = part
                else:
                    out = full
                replaced = True
                break
        expanded.append(out if replaced else part)
    seen: set[str] = set()
    uniq: list[str] = []
    for item in expanded:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    return ", ".join(uniq)


# Long project extraction contexts (Rx schemas, checklists, etc.)
HINT_MAX_CHARS = 8000
USER_CONTEXT_MAX_CHARS = 2000

_RICH_HINT_MARKERS = (
    "medicines",
    "investigations",
    "clinical_notes",
    "letterhead",
    "follow_up",
    "json object",
    "structured json",
    "dosage",
    "prescription",
    "return the result",
)


def _field_lines(project: dict[str, Any], *, defaults: list[str]) -> str:
    fields = project.get("fields") or []
    if fields:
        return "\n".join(
            f'- "{f["key"]}"'
            for f in fields
            if f.get("key")
        )
    return "\n".join(f'- "{key}"' for key in defaults)


def _raw_hint(project: dict[str, Any]) -> str:
    return str(project.get("extractionHint") or "").strip()


def _is_rich_hint(hint: str, project: dict[str, Any]) -> bool:
    """True when the project supplies structured extraction instructions."""
    if project.get("fields"):
        return False
    if len(hint) >= 400:
        return True
    lower = hint.lower()
    return any(marker in lower for marker in _RICH_HINT_MARKERS)


def _guidance(project: dict[str, Any]) -> str:
    user_context = _clip(
        str(project.get("_userContext") or "").strip(), USER_CONTEXT_MAX_CHARS
    )
    if not user_context:
        return ""
    return f"USER FIX (priority): {user_context}\n"


def _build_vision_prompt(project: dict[str, Any]) -> str:
    hint = _clip(_raw_hint(project), HINT_MAX_CHARS)
    guidance = _guidance(project)

    # Structured project context → follow the user's schema, not TRF defaults
    if _is_rich_hint(hint, project):
        return (
            "Extract structured data from this document image.\n"
            "Follow the EXTRACTION CONTEXT below exactly for sections, field names, "
            "lists (e.g. medicines, investigations), null handling, abbreviation expansion, "
            "and any required summary.\n"
            f"{guidance}"
            f"EXTRACTION CONTEXT:\n{hint}\n\n"
            "Rules: never invent values; use null when blank/illegible; "
            "preserve nested objects and arrays; return a single JSON object only."
        )

    defaults = [
        "patient_name",
        "patient_age",
        "patient_gender",
        "client_code",
        "tests_requested",
        "lab_name",
    ]
    return (
        "Extract TRF fields from this form image into minified JSON.\n"
        f"Keys:\n{_field_lines(project, defaults=defaults)}\n"
        f"{guidance}"
        f"Hint: {hint or 'handwritten medical TRF'}\n"
        "Rules: null if unsure; checked gender boxes; tests as comma string; "
        "ignore specimen type/labels; JSON only."
    )


def _build_text_prompt(project: dict[str, Any], *, source_label: str) -> str:
    hint = _clip(_raw_hint(project), HINT_MAX_CHARS)
    guidance = _guidance(project)

    if _is_rich_hint(hint, project):
        return (
            f"Extract structured data from {source_label} text.\n"
            "Follow the EXTRACTION CONTEXT below exactly for sections, field names, "
            "lists, null handling, and any required summary.\n"
            f"{guidance}"
            f"EXTRACTION CONTEXT:\n{hint}\n\n"
            "Rules: never invent values; use null when blank/illegible; "
            "preserve nested objects and arrays; return a single JSON object only."
        )

    defaults = [
        "document_type",
        "patient_name",
        "patient_age",
        "patient_gender",
        "client_code",
        "tests_requested",
    ]
    return (
        f"Extract fields from {source_label} text into minified JSON.\n"
        f"Keys:\n{_field_lines(project, defaults=defaults)}\n"
        f"{guidance}"
        f"Hint: {hint or 'business/medical document'}\n"
        "Rules: null if unsure; JSON only."
    )


def _normalize_value(value: Any) -> Any:
    """Keep nested lists/objects; only join flat primitive lists."""
    if isinstance(value, list):
        if not value:
            return None
        if all(isinstance(item, (dict, list)) for item in value):
            return value
        return ", ".join(str(x).strip() for x in value if str(x).strip())
    return value


def _finalize_payload(parsed: dict[str, Any], *, strategy: str) -> dict[str, Any]:
    # Keep model-produced summary when present; drop only unused marketing extras
    for drop in ("suggested_title", "key_entities"):
        parsed.pop(drop, None)

    if "tests_requested" in parsed:
        parsed["tests_requested"] = _expand_tests(parsed.get("tests_requested"))

    cleaned: dict[str, Any] = {}
    for key, value in parsed.items():
        if value is None or value == "":
            continue
        normalized = _normalize_value(value)
        if normalized is None or normalized == "":
            continue
        cleaned[key] = normalized

    field_confidence: dict[str, float] = {}
    for key, value in cleaned.items():
        if isinstance(value, (dict, list)):
            field_confidence[key] = 0.88
        else:
            field_confidence[key] = 0.93
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
        "strategy": strategy,
    }


def _chat_completions(
    messages: list[dict[str, Any]],
    *,
    deployment: str,
    max_tokens: int,
) -> dict[str, Any]:
    if not azure_openai_configured():
        raise RuntimeError("Azure OpenAI is not configured")

    endpoint = settings.azure_openai_endpoint.rstrip("/")
    api_version = settings.azure_openai_api_version
    url = (
        f"{endpoint}/openai/deployments/{deployment}/chat/completions"
        f"?api-version={api_version}"
    )
    payload = {
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    # Some GPT-5 models ignore/reject custom temperature; omit for reliability + cost.
    headers = {
        "api-key": settings.azure_openai_api_key,
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(connect=20.0, read=60.0, write=30.0, pool=20.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()

    usage = body.get("usage") or {}
    if usage:
        logger.info(
            "Azure OpenAI usage deployment=%s prompt=%s completion=%s total=%s",
            deployment,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )

    content = (
        (((body.get("choices") or [{}])[0].get("message") or {}).get("content"))
        or ""
    ).strip()
    parsed = _parse_json_response(content)
    if not parsed:
        raise RuntimeError("Azure OpenAI returned empty/invalid JSON")
    return parsed


def extract_with_azure_openai(
    project: dict[str, Any],
    pages: list[PageImage],
) -> dict[str, Any]:
    if not pages:
        raise ValueError("No pages available for vision extraction")

    rich = _is_rich_hint(_raw_hint(project), project)
    force_detail = (
        bool(project.get("_forceAi"))
        or bool((project.get("_userContext") or "").strip())
        or rich
    )
    detail = "high" if force_detail else settings.vision_detail
    max_side = settings.vision_max_side_high if force_detail else settings.vision_max_side
    quality = settings.vision_jpeg_quality
    max_tokens = (
        max(settings.vision_max_completion_tokens, 2500)
        if rich
        else settings.vision_max_completion_tokens
    )

    image_b64 = _pil_to_b64_jpeg(
        pages[0].image,
        max_side=max_side,
        quality=quality,
    )
    prompt = _build_vision_prompt(project)
    deployment = settings.azure_openai_deployment
    logger.info(
        "Azure OpenAI vision extract deployment=%s detail=%s max_side=%s "
        "prompt_chars=%d rich_hint=%s max_tokens=%d",
        deployment,
        detail,
        max_side,
        len(prompt),
        rich,
        max_tokens,
    )
    parsed = _chat_completions(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": detail,
                        },
                    },
                ],
            }
        ],
        deployment=deployment,
        max_tokens=max_tokens,
    )
    return _finalize_payload(
        parsed, strategy=f"azure-openai:{deployment}"
    )


def extract_text_with_azure_openai(
    project: dict[str, Any],
    document_text: str,
    *,
    source_label: str = "document",
) -> dict[str, Any]:
    """Structure plain text via cheaper text deployment when configured."""
    rich = _is_rich_hint(_raw_hint(project), project)
    deployment = (
        settings.azure_openai_text_deployment
        or settings.azure_openai_deployment
    )
    prompt = _build_text_prompt(project, source_label=source_label)
    clipped = _clip(document_text or "", settings.text_max_chars)
    max_tokens = (
        max(settings.text_max_completion_tokens, 2000)
        if rich
        else settings.text_max_completion_tokens
    )
    logger.info(
        "Azure OpenAI text extract deployment=%s chars=%d prompt_chars=%d "
        "rich_hint=%s max_tokens=%d",
        deployment,
        len(clipped),
        len(prompt),
        rich,
        max_tokens,
    )
    parsed = _chat_completions(
        [
            {
                "role": "user",
                "content": f"{prompt}\n\nTEXT:\n{clipped}",
            }
        ],
        deployment=deployment,
        max_tokens=max_tokens,
    )
    return _finalize_payload(
        parsed,
        strategy=f"azure-openai-text:{deployment}",
    )
