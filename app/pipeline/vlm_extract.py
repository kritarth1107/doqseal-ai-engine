"""Qwen2.5-VL vision extraction — open-source hybrid layer."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.config import settings
from app.pipeline.ocr import OcrResult
from app.pipeline.preprocess import PageImage

logger = logging.getLogger("doqseal.vlm")

_vlm_model = None
_vlm_processor = None
_vlm_load_failed = False


def _build_schema_prompt(project: dict[str, Any], ocr: OcrResult) -> str:
    fields = project.get("fields") or []
    field_lines = []
    for field in fields:
        req = "required" if field.get("required") else "optional"
        field_lines.append(
            f'- "{field["key"]}" ({field.get("type", "string")}, {req}): {field.get("label", field["key"])}'
        )

    schema_block = "\n".join(field_lines) or '- "summary" (string): Brief document summary'
    hint = project.get("extractionHint") or project.get("description") or ""
    ocr_preview = ocr.full_text[:3500] if ocr.full_text else "No OCR text detected."

    return f"""You are extracting structured data from an Indian medical/compliance document.
Project: {project.get("name", "Unknown")}
Context: {hint}

Return ONLY valid JSON object with these keys:
{schema_block}

Rules:
- Use Hindi or English source text as appropriate.
- For boolean stamp/signature fields, use true if visible/present else false.
- For numbers, return numeric JSON values not strings.
- If a field is missing, use null.
- Do not invent data not supported by the document or OCR.

OCR reference text:
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
            max_new_tokens=1024,
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
        key: round(min(0.97, max(0.65, ocr.average_confidence + 0.1)), 2)
        for key, value in parsed.items()
        if value is not None
    }

    return {
        "data": parsed,
        "fieldConfidence": field_confidence,
        "strategy": "hybrid",
        "rawModelPreview": response[:500],
    }