"""HuggingFace embeddings with GPU support."""

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL


def get_embeddings() -> HuggingFaceEmbeddings:
    """Get the HuggingFace embeddings model with GPU acceleration."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )
