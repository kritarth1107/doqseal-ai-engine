"""Split OCR text and extraction JSON into retrieval chunks."""

from __future__ import annotations

import json
import re
from typing import Any

_CHARS_PER_TOKEN = 4
_DEFAULT_MAX_TOKENS = 800
_DEFAULT_OVERLAP_TOKENS = 120


def _split_text(
    text: str,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> list[str]:
    """Split text into chunks with optional overlap."""
    text = text.strip()
    if not text:
        return []

    max_chars = max_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * _CHARS_PER_TOKEN

    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        if end < len(text):
            split_at = text.rfind("\n\n", start, end)
            if split_at <= start:
                split_at = text.rfind("\n", start, end)
            if split_at <= start:
                split_at = text.rfind(". ", start, end)
            if split_at <= start:
                split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        next_start = end - overlap_chars if overlap_chars > 0 else end
        start = max(next_start, start + 1)

    return chunks


def _extract_page_blocks(text: str) -> list[tuple[int | None, str]]:
    """Extract page-delimited blocks from OCR text.

    Looks for markers like '--- Page N ---' or '[Page N]' in the text.
    Returns list of (page_number, text) tuples.
    """
    page_pattern = re.compile(
        r"(?:^|\n)\s*(?:[-=]{3,}\s*)?(?:Page|PAGE|Pg\.?)\s*(\d+)\s*(?:[-=]{3,})?\s*(?:\n|$)",
        re.IGNORECASE,
    )

    matches = list(page_pattern.finditer(text))

    if not matches:
        return [(None, text)]

    blocks: list[tuple[int | None, str]] = []

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            blocks.append((None, preamble))

    for i, match in enumerate(matches):
        page_num = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        page_text = text[start:end].strip()
        if page_text:
            blocks.append((page_num, page_text))

    return blocks


def build_chunks(
    ocr_full_text: str | None,
    extraction_data: dict[str, Any] | None,
    *,
    max_tokens: int = 500,
    document_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build indexed chunks from OCR plain text and structured extraction data.

    This is the legacy interface, maintained for backward compatibility.
    """
    chunks: list[dict[str, Any]] = []

    for index, text in enumerate(_split_text(ocr_full_text or "", max_tokens)):
        chunks.append({"source": "ocr", "index": index, "text": text})

    if extraction_data:
        serialized = json.dumps(extraction_data, ensure_ascii=False, indent=2)
        for index, text in enumerate(_split_text(serialized, max_tokens)):
            chunks.append({"source": "extraction", "index": index, "text": text})

    return chunks


def build_chunks_v2(
    ocr_full_text: str | None,
    extraction_data: dict[str, Any] | None,
    *,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS,
    document_title: str | None = None,
    document_type: str | None = None,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    """Build chunks with page awareness, overlap, and rich metadata.

    New chunking strategy for improved retrieval:
    - Page-aware: preserves page boundaries when detected
    - Overlap: 15% overlap between chunks for context continuity
    - Metadata: includes title, docType, page number per chunk
    - Field cards: indexes extraction fields as separate searchable units
    """
    chunks: list[dict[str, Any]] = []
    chunk_idx = 0

    page_blocks = _extract_page_blocks(ocr_full_text or "")

    for page_num, page_text in page_blocks:
        page_chunks = _split_text(page_text, max_tokens, overlap_tokens)

        for sub_idx, text in enumerate(page_chunks):
            header_parts = []
            if document_title:
                header_parts.append(f"Document: {document_title}")
            if document_type:
                header_parts.append(f"Type: {document_type}")
            if page_num is not None:
                header_parts.append(f"Page: {page_num}")

            if header_parts:
                text = f"[{' | '.join(header_parts)}]\n{text}"

            chunks.append(
                {
                    "source": "ocr_v2",
                    "index": chunk_idx,
                    "text": text,
                    "page": page_num,
                    "subIndex": sub_idx,
                    "documentTitle": document_title,
                    "documentType": document_type,
                }
            )
            chunk_idx += 1

    if extraction_data:
        field_cards = _build_field_cards(
            extraction_data,
            document_title=document_title,
            document_type=document_type,
        )
        for card_idx, card in enumerate(field_cards):
            chunks.append(
                {
                    "source": "field_card",
                    "index": card_idx,
                    "text": card["text"],
                    "field": card["field"],
                    "value": card["value"],
                    "documentTitle": document_title,
                    "documentType": document_type,
                }
            )

        full_json = json.dumps(extraction_data, ensure_ascii=False, indent=2)
        for idx, text in enumerate(_split_text(full_json, max_tokens, overlap_tokens)):
            chunks.append(
                {
                    "source": "extraction_v2",
                    "index": idx,
                    "text": text,
                    "documentTitle": document_title,
                    "documentType": document_type,
                }
            )

    return chunks


def _build_field_cards(
    extraction_data: dict[str, Any],
    document_title: str | None = None,
    document_type: str | None = None,
    prefix: str = "",
) -> list[dict[str, Any]]:
    """Build searchable field cards from extraction data."""
    cards: list[dict[str, Any]] = []

    skip_keys = {"_id", "extractionId", "documentId", "jobId", "createdAt", "updatedAt"}

    for key, value in extraction_data.items():
        if key in skip_keys or key.startswith("_"):
            continue

        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            cards.extend(
                _build_field_cards(
                    value,
                    document_title=document_title,
                    document_type=document_type,
                    prefix=full_key,
                )
            )
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                for i, item in enumerate(value[:10]):
                    cards.extend(
                        _build_field_cards(
                            item,
                            document_title=document_title,
                            document_type=document_type,
                            prefix=f"{full_key}[{i}]",
                        )
                    )
            else:
                str_value = ", ".join(str(v) for v in value[:20])
                if str_value.strip():
                    text_parts = [f"{_humanize_key(key)}: {str_value}"]
                    if document_title:
                        text_parts.insert(0, f"Document: {document_title}")
                    cards.append(
                        {
                            "field": full_key,
                            "value": str_value,
                            "text": "\n".join(text_parts),
                        }
                    )
        elif value is not None:
            str_value = str(value).strip()
            if str_value and str_value.lower() not in ("none", "null", "n/a", ""):
                text_parts = [f"{_humanize_key(key)}: {str_value}"]
                if document_title:
                    text_parts.insert(0, f"Document: {document_title}")
                cards.append(
                    {
                        "field": full_key,
                        "value": str_value,
                        "text": "\n".join(text_parts),
                    }
                )

    return cards


def _humanize_key(key: str) -> str:
    """Convert camelCase or snake_case to human-readable form."""
    key = re.sub(r"([a-z])([A-Z])", r"\1 \2", key)
    key = key.replace("_", " ")
    return key.title()
