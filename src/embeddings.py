"""HuggingFace embeddings with configurable GPU/CPU device and OOM safeguards."""

from __future__ import annotations

import threading

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    EMBEDDING_NORMALIZE,
    EMBEDDING_OOM_CPU_FALLBACK,
    EMBEDDING_OOM_RETRY_BATCH_SIZE,
)


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


def _build_hf_embeddings(device: str, batch_size: int) -> HuggingFaceEmbeddings:
    """Construct a HuggingFace embeddings client with consistent encode kwargs."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": EMBEDDING_NORMALIZE,
            "batch_size": max(1, batch_size),
        },
    )


def _is_cuda_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return "cuda out of memory" in message or "out of memory" in message


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


class ResilientHuggingFaceEmbeddings(Embeddings):
    """Embedding wrapper that retries CUDA OOM with safer settings."""

    def __init__(self, device: str) -> None:
        self._device = device
        self._batch_size = EMBEDDING_BATCH_SIZE
        self._retry_batch_size = EMBEDDING_OOM_RETRY_BATCH_SIZE
        self._cpu_fallback = EMBEDDING_OOM_CPU_FALLBACK
        self._primary = _build_hf_embeddings(device=device, batch_size=self._batch_size)
        self._cpu: HuggingFaceEmbeddings | None = None

    def _get_cpu(self) -> HuggingFaceEmbeddings:
        if self._cpu is None:
            self._cpu = _build_hf_embeddings(device="cpu", batch_size=self._retry_batch_size)
        return self._cpu

    def _embed_documents_internal(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._primary.embed_documents(texts)
        except Exception as exc:  # noqa: BLE001
            if not (self._device.startswith("cuda") and _is_cuda_oom(exc)):
                raise

            _empty_cuda_cache()
            retry_batch_size = min(self._batch_size, self._retry_batch_size)
            original_batch_size = self._primary.encode_kwargs.get("batch_size", self._batch_size)
            self._primary.encode_kwargs["batch_size"] = retry_batch_size

            try:
                return self._primary.embed_documents(texts)
            except Exception as retry_exc:  # noqa: BLE001
                if self._cpu_fallback:
                    _empty_cuda_cache()
                    return self._get_cpu().embed_documents(texts)
                raise RuntimeError(
                    "CUDA OOM while embedding documents. "
                    "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, "
                    "reduce EMBEDDING_BATCH_SIZE, or enable EMBEDDING_OOM_CPU_FALLBACK=true."
                ) from retry_exc
            finally:
                self._primary.encode_kwargs["batch_size"] = original_batch_size

    def _embed_query_internal(self, text: str) -> list[float]:
        try:
            return self._primary.embed_query(text)
        except Exception as exc:  # noqa: BLE001
            if not (self._device.startswith("cuda") and _is_cuda_oom(exc)):
                raise

            _empty_cuda_cache()
            retry_batch_size = min(self._batch_size, self._retry_batch_size)
            original_batch_size = self._primary.encode_kwargs.get("batch_size", self._batch_size)
            self._primary.encode_kwargs["batch_size"] = retry_batch_size

            try:
                return self._primary.embed_query(text)
            except Exception as retry_exc:  # noqa: BLE001
                if self._cpu_fallback:
                    _empty_cuda_cache()
                    return self._get_cpu().embed_query(text)
                raise RuntimeError(
                    "CUDA OOM while embedding query. "
                    "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, "
                    "reduce EMBEDDING_BATCH_SIZE, or enable EMBEDDING_OOM_CPU_FALLBACK=true."
                ) from retry_exc
            finally:
                self._primary.encode_kwargs["batch_size"] = original_batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_documents_internal(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_query_internal(text)


_cached_embeddings: Embeddings | None = None
_cached_embeddings_lock = threading.Lock()


def get_embeddings() -> Embeddings:
    """Get a process-wide cached embeddings model."""
    global _cached_embeddings
    if _cached_embeddings is not None:
        return _cached_embeddings

    with _cached_embeddings_lock:
        if _cached_embeddings is None:
            device = _resolve_device()
            _cached_embeddings = ResilientHuggingFaceEmbeddings(device=device)
    return _cached_embeddings
