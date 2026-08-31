"""Qwen2.5-VL vision extraction — open-source hybrid layer."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

# Must run before importing torch/transformers — otherwise VLM load crashes with
# "Duplicate dispatch rule for <built-in function intern>"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from app.config import settings
from app.pipeline.ocr import OcrResult
from app.pipeline.preprocess import PageImage

logger = logging.getLogger("doqseal.vlm")

_vlm_model = None
_vlm_processor = None
_vlm_load_failed = False


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
    ocr_preview = ocr.full_text[:4000] if ocr.full_text else "No OCR text detected."
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
- "suggested_title" (string): Short human-readable title for this file
- "summary" (string): 1-3 sentence summary
- "pages" (array, optional): [{ "page": 1, "title": "...", "summary": "..." }]
"""
    elif trf_mode:
        schema_block = """
- "document_type" (string): e.g. "Test Requisition Form"
- "lab_name" (string|null): Diagnostics / lab brand if visible
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
- "verification" (object): stamps/signatures present as booleans when visible
- "suggested_title" (string): Patient name + form type when possible
- "summary" (string): 2-3 sentences covering patient, tests, lab
- "key_entities" (object): flat map of the most important fields
- "pointers" (array): [{ "label": "...", "value": "...", "page": 1 }]
- "checklist" (object): map extraction-context labels → values when provided
- "auto_tags" (array of strings)
"""
    else:
        schema_block = """
- "suggested_title" (string): Short clear title
- "summary" (string): 2-4 sentences
- "key_entities" (object): Important named values
- "checklist" (object): extraction-context labels → values when provided
- "pointers" (array): [{ "label": "...", "value": "...", "page": 1 }]
- "pages" (array): [{ "page": number, "title": "...", "summary": "..." }]
- "auto_tags" (array of strings)
"""

    if in_project and hint:
        scope = (
            f"This file belongs to project «{project_name}».\n"
            f"PRIMARY EXTRACTION CONTEXT (follow closely):\n{hint}\n"
            f"{('Project description: ' + description) if description else ''}"
        )
    elif in_project:
        scope = (
            f"This file belongs to project «{project_name}».\n"
            f"Project description: {description or 'Rely on the document itself.'}"
        )
    else:
        scope = (
            "This file is from Organisation Drive. "
            "Read the page carefully — including handwritten values."
        )

    trf_rules = ""
    if trf_mode:
        trf_rules = """
TRF / requisition rules:
- Prefer handwritten filled values over blank printed labels.
- Read Patient Name, Age, Gender/Sex, Client Code, and every Test Description row.
- Gender: use the checked checkbox (Male/Female).
- tests[] must list each handwritten test (e.g. B group, TFT, HBsAg).
- Do not invent values for empty fields — use null.
"""

    return f"""You are DoqSeal's document intelligence extractor.
You are given the document IMAGE plus noisy OCR reference text.
Trust the IMAGE for handwriting; use OCR only as a weak hint.

{scope}

Return ONLY a valid JSON object with these keys:
{schema_block}

Rules:
- Extract facts that are visible on the page.
- Prefer concrete entities: names, ages, codes, test names, dates, phones.
- For boolean stamp/signature fields, true if visible else false.
- For numbers, return numeric JSON values not strings.
- If a field is missing/blank, use null.
- suggested_title must help a user recognize the file.
{trf_rules}

OCR reference text (may be wrong — verify against the image):
{ocr_preview}
"""


def _parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Model response did not contain JSON object")

    return json.loads(cleaned[start : end + 1])


def _load_vlm():
    global _vlm_model, _vlm_processor, _vlm_load_failed

    if _vlm_load_failed:
        raise RuntimeError("VLM previously failed to load")

    if _vlm_model is not None:
        return _vlm_model, _vlm_processor

    import torch

    try:
        torch._dynamo.config.disable = True  # type: ignore[attr-defined]
    except Exception:
        pass

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_id = settings.vlm_model
    logger.info("Loading VLM %s (4bit=%s)...", model_id, settings.vlm_use_4bit)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if settings.vlm_use_4bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quant_config,
            trust_remote_code=True,
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    model.eval()
    _vlm_model = model
    _vlm_processor = processor
    return _vlm_model, _vlm_processor


def warmup_vlm() -> None:
    """Optionally preload VLM at boot (slow; off by default)."""
    logger.info("Warming up VLM…")
    _load_vlm()
    logger.info("VLM ready")


def extract_with_vlm(
    project: dict[str, Any],
    pages: list[PageImage],
    ocr: OcrResult,
) -> dict[str, Any]:
    global _vlm_load_failed

    try:
        model, processor = _load_vlm()
    except Exception as error:
        _vlm_load_failed = True
        raise RuntimeError(f"VLM load failed: {error}") from error

    import torch
    from qwen_vl_utils import process_vision_info

    prompt = _build_schema_prompt(project, ocr)
    primary_page = pages[0].image

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": primary_page},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=1400,
            temperature=0.1,
            do_sample=False,
        )

    trimmed = [out[len(inp) :] for inp, out in zip(inputs["input_ids"], generated)]
    response = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    parsed = _parse_json_response(response)
    field_confidence = {
        key: round(min(0.97, max(0.7, ocr.average_confidence + 0.2)), 2)
        for key, value in parsed.items()
        if value is not None
    }

    return {
        "data": parsed,
        "fieldConfidence": field_confidence,
        "strategy": "hybrid",
    }
