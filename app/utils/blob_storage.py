"""Azure Blob helpers for reading encrypted document ciphertext."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import unquote

from app.config import settings

logger = logging.getLogger("doqseal.blob")

_container_client = None


def is_azure_blob_enabled() -> bool:
    return bool(
        (settings.azure_storage_connection_string or "").strip()
        and (settings.azure_storage_container or "").strip()
    )


def _get_container_client():
    global _container_client
    if _container_client is not None:
        return _container_client

    if not is_azure_blob_enabled():
        raise ValueError(
            "Azure Blob is not configured. Set AZURE_STORAGE_CONNECTION_STRING "
            "and AZURE_STORAGE_CONTAINER."
        )

    from azure.storage.blob import BlobServiceClient

    service = BlobServiceClient.from_connection_string(
        settings.azure_storage_connection_string
    )
    _container_client = service.get_container_client(settings.azure_storage_container)
    return _container_client


def _blob_key_from_document(document: dict) -> str | None:
    storage_uri = (document.get("storageUri") or "").strip()
    storage_path = (document.get("storagePath") or "").strip()
    provider = (document.get("storageProvider") or "").strip()
    container = (settings.azure_storage_container or "").strip()

    if storage_uri.startswith("https://") and container:
        marker = f"/{container}/"
        idx = storage_uri.find(marker)
        if idx >= 0:
            return unquote(storage_uri[idx + len(marker) :].split("?", 1)[0])

    if provider == "azure-blob" or (
        is_azure_blob_enabled()
        and storage_path
        and not Path(storage_path).is_absolute()
        and not storage_path.startswith("file:")
    ):
        if storage_path.startswith("https://") and container:
            marker = f"/{container}/"
            idx = storage_path.find(marker)
            if idx >= 0:
                return unquote(storage_path[idx + len(marker) :].split("?", 1)[0])
        return storage_path.lstrip("/")

    return None


def load_ciphertext(document: dict) -> bytes:
    """Load encrypted bytes from Azure Blob or legacy local filesystem."""
    blob_key = _blob_key_from_document(document)
    if blob_key and is_azure_blob_enabled():
        client = _get_container_client()
        blob = client.get_blob_client(blob_key)
        logger.info("Downloading blob %s", blob_key)
        return blob.download_blob().readall()

    storage_path = Path(document["storagePath"])
    if not storage_path.is_absolute():
        storage_path = settings.storage_root / storage_path
    return storage_path.read_bytes()
