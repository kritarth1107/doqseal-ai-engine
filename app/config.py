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
    # Vision via Azure OpenAI GPT-5.4 (default) or Ollama multimodal fallback.
    vlm_provider: str = os.getenv("VLM_PROVIDER", "azure_openai")  # azure_openai | ollama
    vlm_model: str = os.getenv("VLM_MODEL", "qwen3-vl:8b")
    vlm_use_4bit: bool = os.getenv("VLM_USE_4BIT", "true").lower() == "true"
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    # Vision / handwriting (expensive). Text structuring uses the cheaper deployment.
    azure_openai_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
    azure_openai_text_deployment: str = os.getenv(
        "AZURE_OPENAI_TEXT_DEPLOYMENT", "gpt-4.1-mini"
    )
    azure_openai_api_version: str = os.getenv(
        "AZURE_OPENAI_API_VERSION", "2024-08-01-preview"
    )
    # When true, handwritten/image TRFs skip EasyOCR and go straight to vision.
    skip_ocr_for_vision: bool = (
        os.getenv("SKIP_OCR_FOR_VISION", "true").lower() == "true"
    )
    # Token / cost knobs (vision images dominate spend)
    vision_detail: str = os.getenv("VISION_DETAIL", "low")  # low default; high on reprocess
    vision_max_side: int = int(os.getenv("VISION_MAX_SIDE", "1024"))
    vision_max_side_high: int = int(os.getenv("VISION_MAX_SIDE_HIGH", "1280"))
    vision_jpeg_quality: int = int(os.getenv("VISION_JPEG_QUALITY", "70"))
    vision_max_completion_tokens: int = int(
        os.getenv("VISION_MAX_COMPLETION_TOKENS", "3000")
    )
    text_max_chars: int = int(os.getenv("TEXT_MAX_CHARS", "45000"))
    text_max_completion_tokens: int = int(
        os.getenv("TEXT_MAX_COMPLETION_TOKENS", "3500")
    )
    text_chunk_parallelism: int = int(os.getenv("TEXT_CHUNK_PARALLELISM", "3"))
    chat_max_completion_tokens: int = int(
        os.getenv("CHAT_MAX_COMPLETION_TOKENS", "250")
    )
    max_pdf_pages: int = int(os.getenv("MAX_PDF_PAGES", "40"))
    # OCR/render cap — vision only uses max_vision_pages; never OCR 40 pages for demos
    max_ocr_pages: int = int(os.getenv("MAX_OCR_PAGES", "8"))
    max_vision_pages: int = int(os.getenv("MAX_VISION_PAGES", "4"))
    ocr_languages: str = os.getenv("OCR_LANGUAGES", "en,hi")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
    # Speed knobs — prefer PDF text path to avoid vision entirely
    pdf_render_scale: float = float(os.getenv("PDF_RENDER_SCALE", "1.25"))
    prefer_pdf_text: bool = os.getenv("PREFER_PDF_TEXT", "true").lower() == "true"
    pdf_text_min_chars: int = int(os.getenv("PDF_TEXT_MIN_CHARS", "80"))
    skip_vlm_min_ocr_confidence: float = float(
        os.getenv("SKIP_VLM_MIN_OCR_CONFIDENCE", "0.72")
    )
    skip_vlm_min_text_chars: int = int(os.getenv("SKIP_VLM_MIN_TEXT_CHARS", "180"))
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

    # Chat pipeline settings
    chat_context_tokens: int = int(os.getenv("CHAT_CONTEXT_TOKENS", "16000"))
    chat_output_tokens: int = int(os.getenv("CHAT_OUTPUT_TOKENS", "2000"))
    chat_temperature: float = float(os.getenv("CHAT_TEMPERATURE", "0.1"))
    chat_history_turns: int = int(os.getenv("CHAT_HISTORY_TURNS", "10"))
    chat_rerank_top_k: int = int(os.getenv("CHAT_RERANK_TOP_K", "8"))
    chat_retrieve_top_k: int = int(os.getenv("CHAT_RETRIEVE_TOP_K", "25"))

    # Guardrail thresholds (can be overridden per-org)
    guardrail_min_rerank_score: float = float(
        os.getenv("GUARDRAIL_MIN_RERANK_SCORE", "0.25")
    )
    guardrail_min_chunks: int = int(os.getenv("GUARDRAIL_MIN_CHUNKS", "1"))
    guardrail_coverage_threshold: float = float(
        os.getenv("GUARDRAIL_COVERAGE_THRESHOLD", "0.6")
    )

    # Streaming settings
    stream_heartbeat_seconds: int = int(os.getenv("STREAM_HEARTBEAT_SECONDS", "15"))


settings = Settings()