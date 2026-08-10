"""Lazy-loaded sentence-transformer embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_passages(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    prefixed = [f"passage: {text}" for text in texts]
    vectors = _get_model().encode(prefixed, normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]


def embed_query(text: str) -> list[float]:
    prefixed = f"query: {text.strip()}"
    vector = _get_model().encode(prefixed, normalize_embeddings=True)
    return vector.tolist()
