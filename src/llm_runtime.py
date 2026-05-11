"""Shared llama.cpp runtime lifecycle and completion helpers."""

from __future__ import annotations

import gc
import re
import threading
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
try:
    from llama_cpp import Llama
except Exception:  # noqa: BLE001
    Llama = Any  # type: ignore[assignment,misc]

from src.config import (
    LLAMACPP_N_CTX,
    LLAMACPP_N_GPU_LAYERS,
    LLAMACPP_N_THREADS,
    LLAMACPP_TEMPERATURE,
    LLAMACPP_VERBOSE,
    MODELS_DIR,
    get_llm_model,
)

_chat_lock = threading.RLock()
_chat_llm: Llama | None = None
_chat_spec: str | None = None


def llamacpp_supports_gpu_offload() -> bool:
    """True only if this llama-cpp-python wheel was built with a GPU backend (e.g. CUDA)."""
    if Llama is Any:
        return False
    try:
        import llama_cpp

        return bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        return False


def _normalize_model_spec(spec: str) -> str:
    """Map legacy Ollama-style HF tags to repo_id/filename.gguf for llama.cpp."""
    s = spec.strip()
    if not s:
        return s
    for prefix in ("https://hf.co/", "https://huggingface.co/", "hf.co/"):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix) :]
            break
    if s.endswith(".gguf"):
        return s
    if ":" not in s or "/" not in s:
        return s
    repo_id, _, tag = s.rpartition(":")
    tag = tag.strip()
    if not repo_id or not tag:
        return s
    m = re.search(r"gpt-oss-(\d+)b", repo_id, re.IGNORECASE)
    if m:
        size = m.group(1)
        return f"{repo_id}/gpt-oss-{size}b-{tag}.gguf"
    return s


def resolve_model_path(spec: str) -> Path:
    """Resolve a model spec to a local path, downloading from HF when needed."""
    normalized = _normalize_model_spec(spec)
    if not normalized:
        raise ValueError("Model spec cannot be empty")

    candidate = Path(normalized)
    if candidate.exists():
        return candidate.resolve()

    if "/" not in normalized or not normalized.endswith(".gguf"):
        raise ValueError(
            "Model must be a local .gguf path or a HuggingFace spec like "
            "'repo_id/filename.gguf'. "
            "Ollama-style tags such as 'hf.co/org/repo:Q4_K_M' are not valid here "
            "(use 'org/repo/gpt-oss-20b-Q4_K_M.gguf' or a path under models/)."
        )

    repo_id, filename = normalized.rsplit("/", 1)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    local_file = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=MODELS_DIR)
    return Path(local_file).resolve()


def _build_llama(model_path: Path) -> Llama:
    if Llama is Any:
        raise RuntimeError(
            "llama-cpp-python is not installed. Install dependencies with `uv sync` "
            "and ensure a compatible wheel/build toolchain is available."
        )
    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "n_gpu_layers": LLAMACPP_N_GPU_LAYERS,
        "n_ctx": LLAMACPP_N_CTX,
        "verbose": LLAMACPP_VERBOSE,
    }
    if LLAMACPP_N_THREADS is not None:
        kwargs["n_threads"] = LLAMACPP_N_THREADS
    return Llama(**kwargs)


def reload_chat_llm(spec: str) -> None:
    """Reload the in-process chat model."""
    global _chat_llm, _chat_spec
    with _chat_lock:
        model_path = resolve_model_path(spec)
        old = _chat_llm
        _chat_llm = None
        if old is not None:
            del old
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        _chat_llm = _build_llama(model_path)
        _chat_spec = spec.strip()


def get_chat_llm() -> Llama:
    """Return chat model, loading lazily from current active config."""
    current_spec = get_llm_model().strip()
    if not current_spec:
        raise ValueError("LLM model cannot be empty")
    if _chat_llm is None or _chat_spec != current_spec:
        reload_chat_llm(current_spec)
    if _chat_llm is None:
        raise RuntimeError("Failed to initialize chat model")
    return _chat_llm


def chat_completion(messages: list[dict[str, Any]], temperature: float | None = None) -> str:
    """Run chat completion and return response text content."""
    with _chat_lock:
        llm = get_chat_llm()
        response = llm.create_chat_completion(
            messages=messages,
            temperature=LLAMACPP_TEMPERATURE if temperature is None else temperature,
        )
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return str(message.get("content", "")).strip()
