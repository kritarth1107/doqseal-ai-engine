import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    mongodb_uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017/doqseal")
    amqp_uri: str = os.getenv("AMQP_URI", "amqp://doqseal:doqseal@localhost:5672")
    extraction_queue: str = os.getenv("EXTRACTION_QUEUE", "extraction.jobs")
    storage_root: Path = Path(
        os.getenv("STORAGE_ROOT", str(Path(__file__).resolve().parents[2] / "storage"))
    ).resolve()
    azure_storage_connection_string: str = os.getenv(
        "AZURE_STORAGE_CONNECTION_STRING", ""
    )
    azure_storage_container: str = os.getenv("AZURE_STORAGE_CONTAINER", "documents")
    aes_secret: str = os.getenv("AES_SECRET", "")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "3031"))

    # Extraction pipeline
    extraction_mode: str = os.getenv("EXTRACTION_MODE", "hybrid")  # hybrid | ocr_only | stub
    # Vision via Azure OpenAI GPT-4o (default) or Ollama multimodal fallback.
    vlm_provider: str = os.getenv("VLM_PROVIDER", "azure_openai")  # azure_openai | ollama
    vlm_model: str = os.getenv("VLM_MODEL", "qwen3-vl:8b")
    vlm_use_4bit: bool = os.getenv("VLM_USE_4BIT", "true").lower() == "true"
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    azure_openai_api_version: str = os.getenv(
        "AZURE_OPENAI_API_VERSION", "2024-08-01-preview"
    )
    # When true, handwritten/image TRFs skip EasyOCR and go straight to GPT-4o.
    skip_ocr_for_vision: bool = (
        os.getenv("SKIP_OCR_FOR_VISION", "true").lower() == "true"
    )
    max_pdf_pages: int = int(os.getenv("MAX_PDF_PAGES", "3"))
    ocr_languages: str = os.getenv("OCR_LANGUAGES", "en,hi")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
    # Speed knobs — target <10s for typical 2-page digital PDFs
    pdf_render_scale: float = float(os.getenv("PDF_RENDER_SCALE", "1.5"))
    prefer_pdf_text: bool = os.getenv("PREFER_PDF_TEXT", "true").lower() == "true"
    pdf_text_min_chars: int = int(os.getenv("PDF_TEXT_MIN_CHARS", "180"))
    skip_vlm_min_ocr_confidence: float = float(
        os.getenv("SKIP_VLM_MIN_OCR_CONFIDENCE", "0.72")
    )
    skip_vlm_min_text_chars: int = int(os.getenv("SKIP_VLM_MIN_TEXT_CHARS", "220"))
    warmup_models: bool = os.getenv("WARMUP_MODELS", "true").lower() == "true"
    warmup_vlm: bool = os.getenv("WARMUP_VLM", "false").lower() == "true"

    # RAG indexing
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "intfloat/multilingual-e5-base"
    )

    # Chat / vision LLM (Ollama)
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "qwen3-vl:8b")


settings = Settings()