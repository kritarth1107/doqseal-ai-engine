from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image

from app.config import settings


@dataclass
class PageImage:
    page_number: int
    image: Image.Image
    width: int
    height: int


def _enhance_image(image: Image.Image) -> Image.Image:
    """Light denoise + contrast for faded Indian scans."""
    rgb = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge((l_channel, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_rgb)


def _load_pdf_pages(data: bytes, max_pages: int) -> list[PageImage]:
    doc = fitz.open(stream=data, filetype="pdf")
    pages: list[PageImage] = []
    scale = max(1.0, float(settings.pdf_render_scale or 1.5))
    try:
        for index in range(min(len(doc), max_pages)):
            page = doc.load_page(index)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            image = _enhance_image(image)
            pages.append(
                PageImage(
                    page_number=index + 1,
                    image=image,
                    width=image.width,
                    height=image.height,
                )
            )
    finally:
        doc.close()
    return pages


def extract_pdf_text_layers(
    file_bytes: bytes,
    max_pages: int | None = None,
) -> tuple[str, int]:
    """
    Pull native PDF text (born-digital forms). Returns (text, page_count_with_text).
    Empty string when the PDF is scan-only.
    """
    limit = max_pages or settings.max_pdf_pages
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    chunks: list[str] = []
    pages_with_text = 0
    try:
        for index in range(min(len(doc), limit)):
            page = doc.load_page(index)
            text = (page.get_text("text") or "").strip()
            if text:
                pages_with_text += 1
                chunks.append(f"--- Page {index + 1} ---\n{text}")
    finally:
        doc.close()
    return "\n\n".join(chunks).strip(), pages_with_text


def _load_image_file(data: bytes) -> list[PageImage]:
    image = Image.open(io.BytesIO(data))
    image = _enhance_image(image)
    return [
        PageImage(
            page_number=1,
            image=image,
            width=image.width,
            height=image.height,
        )
    ]


def load_document_pages(
    file_bytes: bytes,
    mime_type: str,
    max_pages: int | None = None,
) -> list[PageImage]:
    limit = max_pages or settings.max_pdf_pages
    if "pdf" in mime_type.lower():
        return _load_pdf_pages(file_bytes, limit)
    return _load_image_file(file_bytes)


def page_to_cv2(page: PageImage) -> np.ndarray:
    return cv2.cvtColor(np.array(page.image.convert("RGB")), cv2.COLOR_RGB2BGR)


def guess_mime_type(path: str, fallback: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf" or ext.endswith(".pdf.enc"):
        return "application/pdf"
    if ext in {".png", ".png.enc"}:
        return "image/png"
    if ext in {".jpg", ".jpeg", ".jpg.enc", ".jpeg.enc"}:
        return "image/jpeg"
    return fallback