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
    vlm_model: str = os.getenv(
        "VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"
    )
    vlm_use_4bit: bool = os.getenv("VLM_USE_4BIT", "true").lower() == "true"
    max_pdf_pages: int = int(os.getenv("MAX_PDF_PAGES", "3"))
    ocr_languages: str = os.getenv("OCR_LANGUAGES", "en,hi")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))

    # RAG indexing
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "intfloat/multilingual-e5-base"
    )

    # Chat
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")


settings = Settings()