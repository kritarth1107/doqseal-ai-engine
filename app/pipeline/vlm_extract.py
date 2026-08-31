"""Vision extraction via Ollama multimodal models (default: qwen3-vl:8b)."""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from typing import Any

import httpx
from PIL import Image, ImageEnhance, ImageOps

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
        "lupin",
    )
    return sum(1 for m in markers if m in blob) >= 2


def _is_handwritten_form(project: dict[str, Any], ocr: OcrResult) -> bool:
    hint = (project.get("extractionHint") or "").lower()
    name = (project.get("name") or "").lower()
    blob = f"{hint}\n{name}\n{(ocr.full_text or '')[:800]}".lower()
    return any(
        tok in blob
        for tok in (
            "handwrit",
            "trf",
            "requisition",
            "lupin",
            "patient name",
            "test request",
        )
    )


def _build_schema_prompt(
    project: dict[str, Any],
    ocr: OcrResult,
    *,
    use_ocr_hint: bool,
) -> str:
    fields = project.get("fields") or []
    hint = (project.get("extractionHint") or "").strip()
    description = (project.get("description") or "").strip()
    in_project = bool(project.get("projectId"))
    project_name = project.get("name") or (
        "Organisation Drive" if not in_project else "Project"
    )
    trf_mode = _looks_like_trf(ocr.full_text or "", hint) or _is_handwritten_form(
        project, ocr
    )

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
- "suggested_title" (string): "{Patient Name} — {tests}" when known
- "summary" (string): 1 short sentence from filled values only
"""
    elif trf_mode:
        schema_block = """
- "document_type" (string)
- "lab_name" (string|null)
- "patient_name" (string|null)
- "patient_age" (number|null)
- "patient_gender" (string|null)
- "client_code" (string|null)
- "client_name" (string|null)
- "uhid" (string|null)
- "phone" (string|null)
- "referring_doctor" (string|null)
- "clinical_history" (string|null)
- "tests_requested" (string|null): comma-separated handwritten test names only
- "specimen_drawn_date" (string|null)
- "specimen_drawn_time" (string|null)
- "fasting" (boolean|null)
- "suggested_title" (string)
- "summary" (string)
"""
    else:
        schema_block = """
- "suggested_title" (string)
- "summary" (string)
- "key_entities" (object)
- "pointers" (array): [{ "label": "...", "value": "...", "page": 1 }]
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
        scope = "Organisation Drive upload. Read blue/black ink handwriting from the photos."

    images_guide = """
You are given up to 3 photos of the SAME form:
1) full page
2) top band (Patient + Specimen + Client)
3) middle band (Specimen Type + Test Requirements)

Read ink strokes letter-by-letter. Printed purple labels are NOT values.
"""

    quality_rules = """
CRITICAL accuracy rules:
- Values must come from HANDWRITTEN ink or clearly CHECKED boxes in the photos.
- Do NOT copy wording from this prompt. Do NOT invent common lab tests.
- Do NOT use any OCR text as a source of patient names or tests — it is often garbage.
- Specimen Type checkboxes (Serum, EDTA, Urine, …) are NOT tests.
- tests_requested = only ink written under the Test Description column.
  Expand clear abbreviations when obvious from the ink (e.g. short forms of
  creatinine / CBC / lipid profile) — never invent tests that are not written.
- patient_name = ink in the Patient Name box only.
- patient_age = number next to Yrs/Months. patient_gender = checked Male/Female box.
- client_code = ink in Client Code only.
- Empty / unreadable → null. Prefer null over a guess.
- Never output OCR gibberish as a name.
"""

    ocr_block = ""
    if use_ocr_hint:
        preview = (ocr.full_text or "")[:1200]
        ocr_block = f"""
Printed-form OCR (IGNORE for handwritten fields; labels only):
{preview}
"""

    return f"""You are a careful medical-form transcriptionist for DoqSeal.
Transcribe filled Lupin / lab TRF fields from photographs.

{scope}
{images_guide}

Return ONLY valid JSON with these keys:
{schema_block}

{quality_rules}
{ocr_block}
"""


def _repair_json_text(blob: str) -> str:
    """Best-effort cleanup for common multimodal LLM JSON glitches."""
    text = blob.strip()
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I).strip()
    text = re.sub(r"<\|.*?\|>", "", text).strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.I)
    if fence:
        text = fence.group(1).strip()

    start = text.find("{")
    if start == -1:
        return text
    text = text[start:]

    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return text


def _parse_json_response(text: str) -> dict[str, Any]:
    cleaned = _repair_json_text(text)
    candidates = [cleaned]

    open_curly = cleaned.count("{") - cleaned.count("}")
    open_square = cleaned.count("[") - cleaned.count("]")
    if open_curly > 0 or open_square > 0:
        patched = cleaned.rstrip().rstrip(",")
        patched += "]" * max(0, open_square)
        patched += "}" * max(0, open_curly)
        candidates.append(patched)

    last_error: Exception | None = None
    for candidate in candidates:
        end = candidate.rfind("}")
        if end == -1:
            continue
        snippet = candidate[: end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except Exception as err:
            last_error = err
            continue

    preview = (text or "")[:800].replace("\n", " ")
    raise ValueError(
        f"Model response was not valid JSON ({last_error}); preview={preview!r}"
    )


def _prepare_vision_image(image: Image.Image, max_side: int = 2048) -> Image.Image:
    """Upscale small phone photos and gently boost ink contrast for VLM."""
    rgb = image.convert("RGB")
    rgb = ImageOps.autocontrast(rgb, cutoff=1)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.15)
    rgb = ImageEnhance.Sharpness(rgb).enhance(1.2)
    w, h = rgb.size
    if max(w, h) < 1400:
        scale = 1600 / float(max(w, h))
        rgb = rgb.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    w, h = rgb.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        rgb = rgb.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return rgb


def _pil_to_b64_jpeg(image: Image.Image, quality: int = 92) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _region_crops(full: Image.Image) -> list[Image.Image]:
    """Full page + top patient/client band + mid tests band."""
    w, h = full.size
    top = full.crop((0, 0, w, int(h * 0.42)))
    mid = full.crop((0, int(h * 0.28), w, int(h * 0.62)))
    return [full, top, mid]


def _images_for_vlm(page: PageImage) -> list[str]:
    full = _prepare_vision_image(page.image)
    return [_pil_to_b64_jpeg(img) for img in _region_crops(full)]


def _call_ollama_vision(prompt: str, image_b64_list: list[str]) -> str:
    model = settings.vlm_model
    url = f"{settings.ollama_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_predict": 2048,
        },
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": image_b64_list,
            }
        ],
    }
    timeout = httpx.Timeout(connect=120.0, read=600.0, write=120.0, pool=60.0)
    logger.info(
        "Ollama vision extract model=%s url=%s images=%d",
        model,
        url,
        len(image_b64_list),
    )
    last_error: Exception | None = None
    body: dict[str, Any] = {}
    for attempt in range(1, 4):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                body = response.json()
            break
        except Exception as err:
            last_error = err
            logger.warning(
                "Ollama vision attempt %d/3 failed: %s", attempt, err
            )
            if attempt < 3:
                time.sleep(5 * attempt)
    else:
        assert last_error is not None
        raise last_error
    message = body.get("message") or {}
    content = (message.get("content") or body.get("response") or "").strip()
    if not content:
        raise RuntimeError("Empty response from Ollama vision model")
    return content


def _looks_like_ocr_garbage(name: str | None) -> bool:
    if not name or not isinstance(name, str):
        return False
    cleaned = name.strip()
    if len(cleaned) < 3:
        return True
    if re.search(r"[0-9]{3,}", cleaned):
        return True
    if re.search(r"(lupin|requisition|patient|barcode|diagnostics)", cleaned, re.I):
        return True
    vowels = len(re.findall(r"[aeiouAEIOU]", cleaned))
    letters = len(re.findall(r"[A-Za-z]", cleaned))
    if letters >= 8 and vowels / max(letters, 1) < 0.18:
        return True
    return False


def _sanitize_extraction(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    name = out.get("patient_name")
    if isinstance(name, str) and _looks_like_ocr_garbage(name):
        logger.warning("Dropping garbage patient_name=%r", name)
        out["patient_name"] = None

    tests = out.get("tests_requested")
    if isinstance(tests, str):
        parts = [p.strip() for p in re.split(r"[,;\n]+", tests) if p.strip()]
        blocked = {
            "serum",
            "w. blood edta",
            "blood edta",
            "plasma",
            "urine",
            "urine random",
            "urine 24 hr",
            "urine 2d hr",
        }
        parts = [p for p in parts if p.lower() not in blocked]
        out["tests_requested"] = ", ".join(parts) if parts else None

    pname = out.get("patient_name")
    t = out.get("tests_requested")
    if pname and t:
        out["suggested_title"] = f"{pname} — {t}"
    elif pname:
        out["suggested_title"] = str(pname)
    return out


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

    handwritten = _is_handwritten_form(project, ocr)
    # OCR destroys handwritten TRFs — do not feed it as content.
    use_ocr_hint = not handwritten
    prompt = _build_schema_prompt(project, ocr, use_ocr_hint=use_ocr_hint)
    images = _images_for_vlm(pages[0])
    raw = _call_ollama_vision(prompt, images)
    try:
        parsed = _parse_json_response(raw)
    except Exception as err:
        logger.warning(
            "First JSON parse failed (%s); retrying vision once",
            err,
        )
        retry_prompt = (
            prompt
            + "\n\nReply with one minified JSON object only. "
            "Null for blank fields. No markdown."
        )
        raw = _call_ollama_vision(retry_prompt, images)
        parsed = _parse_json_response(raw)

    parsed = _sanitize_extraction(parsed)

    # If name still missing/garbage, one focused retry on the top/mid crops
    if handwritten and (
        not parsed.get("patient_name")
        or _looks_like_ocr_garbage(str(parsed.get("patient_name") or ""))
    ):
        logger.info("Retrying patient/tests focused pass")
        focus_prompt = (
            "Photo 1 is the top of a Lupin TRF (patient + client). "
            "Photo 2 is the test-requirements area. "
            "Return JSON with patient_name, patient_age, patient_gender, "
            "client_code, tests_requested only. "
            "Read blue ink carefully. Null if blank. No other keys. "
            "Do not invent tests."
        )
        focus_images = images[1:3] if len(images) >= 3 else images
        try:
            focus_raw = _call_ollama_vision(focus_prompt, focus_images)
            focus = _sanitize_extraction(_parse_json_response(focus_raw))
            for key in (
                "patient_name",
                "patient_age",
                "patient_gender",
                "client_code",
                "tests_requested",
            ):
                val = focus.get(key)
                if val is not None and val != "":
                    if key == "patient_name" and _looks_like_ocr_garbage(str(val)):
                        continue
                    parsed[key] = val
            parsed = _sanitize_extraction(parsed)
        except Exception as focus_err:
            logger.warning("Focused vision pass failed: %s", focus_err)

    field_confidence = {
        key: round(min(0.97, max(0.72, ocr.average_confidence + 0.25)), 2)
        for key, value in parsed.items()
        if value is not None and value != ""
    }
    if parsed.get("patient_name"):
        field_confidence["patient_name"] = 0.9
    if parsed.get("tests_requested"):
        field_confidence["tests_requested"] = 0.9

    return {
        "data": parsed,
        "fieldConfidence": field_confidence,
        "strategy": f"ollama:{settings.vlm_model}",
    }
