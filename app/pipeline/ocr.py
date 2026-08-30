from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config import settings
from app.pipeline.preprocess import PageImage, page_to_cv2

logger = logging.getLogger("doqseal.ocr")

_ocr_engine = None


@dataclass
class OcrLine:
    text: str
    confidence: float
    box: list[list[float]] = field(default_factory=list)


@dataclass
class OcrResult:
    full_text: str
    lines: list[OcrLine]
    average_confidence: float
    languages: list[str]


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        import easyocr
        import torch

        langs = [
            lang.strip()
            for lang in settings.ocr_languages.split(",")
            if lang.strip()
        ] or ["en"]

        use_gpu = torch.cuda.is_available()
        logger.info("Initializing EasyOCR (langs=%s, gpu=%s)...", langs, use_gpu)
        _ocr_engine = easyocr.Reader(langs, gpu=use_gpu)
    return _ocr_engine


def _parse_easyocr_result(result: list) -> list[OcrLine]:
    lines: list[OcrLine] = []
    for item in result:
        if len(item) < 3:
            continue
        box, text, score = item[0], item[1], item[2]
        cleaned = (text or "").strip()
        if not cleaned:
            continue
        lines.append(
            OcrLine(
                text=cleaned,
                confidence=float(score),
                box=box,
            )
        )
    return lines


def run_ocr(pages: list[PageImage]) -> OcrResult:
    engine = _get_ocr_engine()
    all_lines: list[OcrLine] = []

    for page in pages:
        image = page_to_cv2(page)
        result = engine.readtext(image)
        all_lines.extend(_parse_easyocr_result(result))

    if not all_lines:
        return OcrResult(
            full_text="",
            lines=[],
            average_confidence=0.0,
            languages=settings.ocr_languages.split(","),
        )

    full_text = "\n".join(line.text for line in all_lines)
    avg_conf = sum(line.confidence for line in all_lines) / len(all_lines)

    return OcrResult(
        full_text=full_text,
        lines=all_lines,
        average_confidence=avg_conf,
        languages=settings.ocr_languages.split(","),
    )


def ocr_result_from_text(text: str, confidence: float = 0.92) -> OcrResult:
    """Build an OCR-like result from native PDF text (no EasyOCR)."""
    lines = [
        OcrLine(text=line.strip(), confidence=confidence)
        for line in (text or "").splitlines()
        if line.strip()
    ]
    return OcrResult(
        full_text=(text or "").strip(),
        lines=lines,
        average_confidence=confidence if lines else 0.0,
        languages=settings.ocr_languages.split(","),
    )


def warmup_ocr() -> None:
    """Load EasyOCR once at worker boot so the first real job isn't a cold start."""
    logger.info("Warming up EasyOCR…")
    _get_ocr_engine()
    logger.info("EasyOCR ready")
