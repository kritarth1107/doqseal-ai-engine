"""Split OCR text and extraction JSON into retrieval chunks."""

from __future__ import annotations

import json
from typing import Any

# Rough heuristic: ~4 characters per token for mixed Latin/Devanagari text.
_CHARS_PER_TOKEN = 4


def _split_text(text: str, max_tokens: int) -> list[str]:
    text = text.strip()
    if not text:
        return []

    max_chars = max_tokens * _CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            split_at = text.rfind("\n", start, end)
            if split_at <= start:
                split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end if end > start else start + max_chars
    return chunks


def build_chunks(
    ocr_full_text: str | None,
    extraction_data: dict[str, Any] | None,
    *,
    max_tokens: int = 500,
) -> list[dict[str, Any]]:
    """Build indexed chunks from OCR plain text and structured extraction data."""
    chunks: list[dict[str, Any]] = []

    for index, text in enumerate(_split_text(ocr_full_text or "", max_tokens)):
        chunks.append({"source": "ocr", "index": index, "text": text})

    if extraction_data:
        serialized = json.dumps(extraction_data, ensure_ascii=False, indent=2)
        for index, text in enumerate(_split_text(serialized, max_tokens)):
            chunks.append({"source": "extraction", "index": index, "text": text})

    return chunks
