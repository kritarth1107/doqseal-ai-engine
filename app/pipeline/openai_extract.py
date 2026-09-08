"""Token-efficient Azure OpenAI extraction (vision + text) for any document type."""

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

HINT_MAX_CHARS = 8000
USER_CONTEXT_MAX_CHARS = 2000
TEXT_CHUNK_CHARS = 14000
TEXT_CHUNK_OVERLAP = 500

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

_TRF_HINT_MARKERS = (
    "trf",
    "test request form",
    "patient name",
    "tests requested",
    "client code",
    "lab name",
    "specimen",
)

_OPEN_SCHEMA = """Return a single JSON object with as much useful content as you can extract:
- "document_type": short type label (e.g. pitch_deck, invoice, contract, prescription, report, form, other)
- "suggested_title": concise human title for the document
- "summary": 4-8 sentence plain-English overview of the whole document
- "key_entities": object of notable names, companies, people, dates, metrics, amounts
- "fields": flat object of important scalar key/value pairs found in the document
- "sections": array of { "heading": string, "content": string } covering major sections
- "tables": array of { "name": string|null, "headers": string[], "rows": string[][] }
- "pages": array of { "page": number, "title": string|null, "bullets": string[], "key_points": string[] } when the source is multi-page / a deck
- "pointers": array of short actionable or notable highlights
- "auto_tags": array of short topic tags

Rules: extract EVERYTHING readable; never invent; use null for missing; keep nested arrays/objects; expand abbreviations when clear; JSON only."""


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
        r"\bcbp\b": "Complete Blood Picture (CBP)",
        r"\bcue\b": "Complete Urine Examination (CUE)",
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
        r"\becg\b": "Electrocardiogram (ECG)",
        r"\busg\b": "Ultrasonography (USG)",
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


def _field_lines(project: dict[str, Any], *, defaults: list[str]) -> str:
    fields = project.get("fields") or []
    if fields:
        return "\n".join(f'- "{f["key"]}"' for f in fields if f.get("key"))
    return "\n".join(f'- "{key}"' for key in defaults)


def _raw_hint(project: dict[str, Any]) -> str:
    return str(project.get("extractionHint") or "").strip()


def _looks_like_trf(hint: str, project: dict[str, Any]) -> bool:
    fields = project.get("fields") or []
    keys = {str(f.get("key") or "").lower() for f in fields}
    if keys & {"patient_name", "tests_requested", "client_code", "lab_name"}:
        return True
    lower = hint.lower()
    return any(marker in lower for marker in _TRF_HINT_MARKERS)


def _is_rich_hint(hint: str) -> bool:
    if len(hint) >= 400:
        return True
    lower = hint.lower()
    return any(marker in lower for marker in _RICH_HINT_MARKERS)


def _mode(project: dict[str, Any]) -> str:
    """schema | rich | trf | open"""
    hint = _raw_hint(project)
    fields = project.get("fields") or []
    if fields and not _looks_like_trf(hint, project):
        return "schema"
    if fields and _looks_like_trf(hint, project):
        return "trf"
    if _is_rich_hint(hint):
        return "rich"
    if _looks_like_trf(hint, project):
        return "trf"
    return "open"


def _guidance(project: dict[str, Any]) -> str:
    user_context = _clip(
        str(project.get("_userContext") or "").strip(), USER_CONTEXT_MAX_CHARS
    )
    if not user_context:
        return ""
    return f"USER FIX (priority): {user_context}\n"


def _open_prompt(*, source: str, guidance: str, hint: str) -> str:
    hint_block = f"\nEXTRA PROJECT CONTEXT:\n{hint}\n" if hint else ""
    return (
        f"You are a document intelligence engine. Extract structured data from this {source}.\n"
        f"{guidance}"
        f"{hint_block}\n"
        f"{_OPEN_SCHEMA}"
    )


def _build_vision_prompt(project: dict[str, Any]) -> str:
    hint = _clip(_raw_hint(project), HINT_MAX_CHARS)
    guidance = _guidance(project)
    mode = _mode(project)

    if mode == "rich":
        return (
            "Extract structured data from this document image.\n"
            "Follow the EXTRACTION CONTEXT below exactly for sections, field names, "
            "lists, null handling, abbreviation expansion, and any required summary.\n"
            f"{guidance}"
            f"EXTRACTION CONTEXT:\n{hint}\n\n"
            "Rules: never invent values; use null when blank/illegible; "
            "preserve nested objects and arrays; return a single JSON object only."
        )

    if mode == "schema":
        return (
            "Extract fields from this document image into JSON.\n"
            f"Keys:\n{_field_lines(project, defaults=[])}\n"
            f"{guidance}"
            f"Hint: {hint or 'follow the keys above'}\n"
            "Also include suggested_title and summary when helpful. "
            "Rules: null if unsure; preserve lists/objects; JSON only."
        )

    if mode == "trf":
        defaults = [
            "patient_name",
            "patient_age",
            "patient_gender",
            "client_code",
            "tests_requested",
            "lab_name",
        ]
        return (
            "Extract TRF / lab request fields from this form image into JSON.\n"
            f"Keys:\n{_field_lines(project, defaults=defaults)}\n"
            f"{guidance}"
            f"Hint: {hint or 'handwritten medical TRF'}\n"
            "Rules: null if unsure; checked gender boxes; tests as comma string; JSON only."
        )

    return _open_prompt(source="document image", guidance=guidance, hint=hint)


def _build_text_prompt(project: dict[str, Any], *, source_label: str) -> str:
    hint = _clip(_raw_hint(project), HINT_MAX_CHARS)
    guidance = _guidance(project)
    mode = _mode(project)

    if mode == "rich":
        return (
            f"Extract structured data from {source_label} text.\n"
            "Follow the EXTRACTION CONTEXT below exactly for sections, field names, "
            "lists, null handling, and any required summary.\n"
            f"{guidance}"
            f"EXTRACTION CONTEXT:\n{hint}\n\n"
            "Rules: never invent values; use null when blank/illegible; "
            "preserve nested objects and arrays; return a single JSON object only."
        )

    if mode == "schema":
        return (
            f"Extract fields from {source_label} text into JSON.\n"
            f"Keys:\n{_field_lines(project, defaults=[])}\n"
            f"{guidance}"
            f"Hint: {hint or 'follow the keys above'}\n"
            "Also include suggested_title and summary when helpful. JSON only."
        )

    if mode == "trf":
        defaults = [
            "document_type",
            "patient_name",
            "patient_age",
            "patient_gender",
            "client_code",
            "tests_requested",
            "lab_name",
        ]
        return (
            f"Extract TRF / lab request fields from {source_label} text into JSON.\n"
            f"Keys:\n{_field_lines(project, defaults=defaults)}\n"
            f"{guidance}"
            f"Hint: {hint or 'medical TRF'}\n"
            "Rules: null if unsure; JSON only."
        )

    return _open_prompt(
        source=f"{source_label} text",
        guidance=guidance,
        hint=hint,
    )


def _normalize_value(value: Any) -> Any:
    if isinstance(value, list):
        if not value:
            return None
        if all(isinstance(item, (dict, list)) for item in value):
            return value
        # Keep string lists (bullets / tags) as lists when >1 item
        if all(isinstance(item, (str, int, float, bool)) for item in value):
            if len(value) == 1:
                return value[0]
            return [str(x).strip() for x in value if str(x).strip()]
        return ", ".join(str(x).strip() for x in value if str(x).strip())
    return value


def _finalize_payload(parsed: dict[str, Any], *, strategy: str) -> dict[str, Any]:
    if "tests_requested" in parsed:
        parsed["tests_requested"] = _expand_tests(parsed.get("tests_requested"))

    # Promote nested "fields" object into top-level scalars when useful
    nested_fields = parsed.get("fields")
    if isinstance(nested_fields, dict):
        for key, value in nested_fields.items():
            if key not in parsed and value not in (None, ""):
                parsed[key] = value

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
            field_confidence[key] = 0.92

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
    headers = {
        "api-key": settings.azure_openai_api_key,
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(connect=20.0, read=120.0, write=30.0, pool=20.0)
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


def _chunk_text(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= TEXT_CHUNK_CHARS:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + TEXT_CHUNK_CHARS)
        if end < len(text):
            # Prefer break on page marker or newline
            window = text[start:end]
            cut = window.rfind("\n--- Page ")
            if cut < TEXT_CHUNK_CHARS // 3:
                cut = window.rfind("\n\n")
            if cut >= TEXT_CHUNK_CHARS // 3:
                end = start + cut
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - TEXT_CHUNK_OVERLAP, start + 1)
    return [c for c in chunks if c]


def _merge_partial_extractions(
    partials: list[dict[str, Any]],
    *,
    deployment: str,
    source_label: str,
) -> dict[str, Any]:
    if len(partials) == 1:
        return partials[0]
    payload = json.dumps(partials, ensure_ascii=False)
    prompt = (
        f"Merge these partial extractions of {source_label} into ONE complete JSON object.\n"
        "Deduplicate overlapping content. Prefer fuller pages/sections/tables.\n"
        f"{_OPEN_SCHEMA}\n\n"
        f"PARTIALS:\n{_clip(payload, 60000)}"
    )
    return _chat_completions(
        [{"role": "user", "content": prompt}],
        deployment=deployment,
        max_tokens=max(settings.text_max_completion_tokens, 4000),
    )


def extract_with_azure_openai(
    project: dict[str, Any],
    pages: list[PageImage],
) -> dict[str, Any]:
    if not pages:
        raise ValueError("No pages available for vision extraction")

    mode = _mode(project)
    force_detail = (
        bool(project.get("_forceAi"))
        or bool((project.get("_userContext") or "").strip())
        or mode in {"rich", "open", "schema"}
    )
    detail = "high" if force_detail else settings.vision_detail
    max_side = settings.vision_max_side_high if force_detail else settings.vision_max_side
    quality = settings.vision_jpeg_quality
    max_tokens = max(settings.vision_max_completion_tokens, 3000)
    vision_pages = pages[: max(1, int(getattr(settings, "max_vision_pages", 4) or 4))]

    content: list[dict[str, Any]] = [
        {"type": "text", "text": _build_vision_prompt(project)},
    ]
    for page in vision_pages:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        "data:image/jpeg;base64,"
                        + _pil_to_b64_jpeg(
                            page.image, max_side=max_side, quality=quality
                        )
                    ),
                    "detail": detail,
                },
            }
        )

    deployment = settings.azure_openai_deployment
    logger.info(
        "Azure OpenAI vision extract deployment=%s detail=%s pages=%d/%d "
        "mode=%s max_tokens=%d",
        deployment,
        detail,
        len(vision_pages),
        len(pages),
        mode,
        max_tokens,
    )
    parsed = _chat_completions(
        [{"role": "user", "content": content}],
        deployment=deployment,
        max_tokens=max_tokens,
    )
    return _finalize_payload(parsed, strategy=f"azure-openai:{deployment}")


def extract_text_with_azure_openai(
    project: dict[str, Any],
    document_text: str,
    *,
    source_label: str = "document",
) -> dict[str, Any]:
    """Structure plain text via cheaper text deployment (gpt-4.1-mini)."""
    mode = _mode(project)
    deployment = (
        settings.azure_openai_text_deployment
        or settings.azure_openai_deployment
    )
    prompt = _build_text_prompt(project, source_label=source_label)
    chunks = _chunk_text(_clip(document_text or "", settings.text_max_chars))
    if not chunks:
        raise ValueError("No text available for extraction")

    max_tokens = max(settings.text_max_completion_tokens, 3500)
    logger.info(
        "Azure OpenAI text extract deployment=%s chunks=%d total_chars=%d "
        "mode=%s max_tokens=%d",
        deployment,
        len(chunks),
        sum(len(c) for c in chunks),
        mode,
        max_tokens,
    )

    partials: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        chunk_prompt = prompt
        if len(chunks) > 1:
            chunk_prompt = (
                f"{prompt}\n\nThis is chunk {index + 1} of {len(chunks)}. "
                "Extract everything from this chunk; a later merge will combine results."
            )
        partial = _chat_completions(
            [
                {
                    "role": "user",
                    "content": f"{chunk_prompt}\n\nTEXT:\n{chunk}",
                }
            ],
            deployment=deployment,
            max_tokens=max_tokens,
        )
        partials.append(partial)

    parsed = _merge_partial_extractions(
        partials, deployment=deployment, source_label=source_label
    )
    return _finalize_payload(
        parsed,
        strategy=f"azure-openai-text:{deployment}",
    )
