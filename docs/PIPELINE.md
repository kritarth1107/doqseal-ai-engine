# Extraction pipeline

1. **preprocess.py** — PDF→images, CLAHE, max 3 pages
2. **ocr.py** — EasyOCR (en, hi), expand to Indic tier
3. **vlm_extract.py** — Qwen2.5-VL structured JSON
4. **ocr_extract.py** — regex fallback
5. **validate.py** — schema + confidence threshold

Low OCR confidence → `needs_review` status.
