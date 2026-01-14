"""Ollama embeddings wrapper."""

from langchain_ollama import OllamaEmbeddings

from src.config import EMBEDDING_MODEL, OLLAMA_BASE_URL


def get_embeddings() -> OllamaEmbeddings:
    """Get the Ollama embeddings model."""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )
