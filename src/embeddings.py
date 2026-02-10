"""HuggingFace embeddings with configurable GPU/CPU device."""

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_DEVICE, EMBEDDING_MODEL


def _resolve_device() -> str:
    """Resolve embedding device from config and validate CUDA availability."""
    requested = (EMBEDDING_DEVICE or "cuda:0").strip().lower()
    if requested == "gpu":
        requested = "cuda:0"

    try:
        import torch

        if requested.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "EMBEDDING_DEVICE is set to CUDA but torch has no CUDA support. "
                    "Install a CUDA-enabled torch build."
                )
            return requested
        return requested
    except Exception:
        if requested.startswith("cuda"):
            raise
        return requested


def get_embeddings() -> HuggingFaceEmbeddings:
    """Get the HuggingFace embeddings model on the configured device."""
    device = _resolve_device()
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
