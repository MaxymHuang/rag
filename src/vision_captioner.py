"""Vision caption service for converting image artifacts into text documents."""

from __future__ import annotations

import base64
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.documents import Document
try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
except Exception:  # noqa: BLE001
    Llama = Any  # type: ignore[assignment,misc]
    Llava15ChatHandler = None

from src.config import (
    LLAMACPP_N_GPU_LAYERS,
    OCR_ENABLED,
    VISION_CAPTION_MODEL,
    VISION_MMPROJ_MODEL,
    VISION_CAPTION_PROVIDER,
    VISION_ENABLED,
)
from src.document_loader import VisualArtifact, safe_print
from src.llm_runtime import resolve_model_path


CAPTION_PROMPT = (
    "Describe this image for retrieval in a RAG system. "
    "Be concise, factual, and include key entities, labels, chart axes, table fields, "
    "and process/diagram relationships when present. Avoid speculation."
)


class VisionCaptionProvider(Protocol):
    """Provider interface for image caption generation."""

    def caption_image(self, image_bytes: bytes, metadata: dict) -> str:
        """Return a factual caption for one image."""


@dataclass
class LlamaCppVisionCaptionProvider:
    """llama.cpp-backed vision captioning provider."""

    model: str = VISION_CAPTION_MODEL
    mmproj_model: str = VISION_MMPROJ_MODEL
    _llm: Llama | None = None
    _llm_lock: threading.Lock = field(default_factory=threading.Lock)

    def _build_llm(self) -> Llama:
        if Llama is Any or Llava15ChatHandler is None:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install dependencies with `uv sync` "
                "and ensure a compatible wheel/build toolchain is available."
            )
        handler = Llava15ChatHandler(clip_model_path=str(resolve_model_path(self.mmproj_model)))
        return Llama(
            model_path=str(resolve_model_path(self.model)),
            chat_handler=handler,
            n_gpu_layers=LLAMACPP_N_GPU_LAYERS,
            n_ctx=2048,
            logits_all=True,
            verbose=False,
        )

    def _get_llm(self) -> Llama:
        with self._llm_lock:
            if self._llm is None:
                self._llm = self._build_llm()
            return self._llm

    def caption_image(self, image_bytes: bytes, metadata: dict) -> str:
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        image_mime = metadata.get("image_mime", "image/png")
        response = self._get_llm().create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
                        {"type": "text", "text": CAPTION_PROMPT},
                    ],
                }
            ],
        )
        choices = response.get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", "")).strip()


def _build_provider() -> VisionCaptionProvider:
    provider = VISION_CAPTION_PROVIDER.strip().lower()
    if provider == "llamacpp":
        return LlamaCppVisionCaptionProvider()
    raise ValueError(f"Unsupported vision caption provider: {VISION_CAPTION_PROVIDER}")


def _extract_ocr_text(image_bytes: bytes) -> str:
    """Best-effort OCR text extraction from image bytes."""
    if not OCR_ENABLED:
        return ""
    try:
        import io

        import pytesseract
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as exc:  # noqa: BLE001
        safe_print(f"  Warning: OCR failed: {exc}")
        return ""


def caption_visual_artifacts(artifacts: list[VisualArtifact]) -> tuple[list[Document], int]:
    """
    Convert image artifacts into caption documents.

    Returns:
        tuple[captions, failed_count]
    """
    if not VISION_ENABLED:
        return [], 0

    provider = _build_provider()
    caption_docs: list[Document] = []
    failed_count = 0

    for artifact in artifacts:
        metadata = dict(artifact.metadata)
        try:
            caption = provider.caption_image(artifact.content, metadata)
            if not caption:
                failed_count += 1
                continue

            ocr_text = _extract_ocr_text(artifact.content)
            page_or_slide = metadata.get("page_or_slide")
            source = metadata.get("source", "unknown")
            prefix = f"Image summary from {source}"
            if page_or_slide is not None:
                prefix += f" (page_or_slide={page_or_slide})"

            content_parts = [prefix + ":", caption]
            if ocr_text:
                content_parts.append(f"OCR text: {ocr_text}")
            content = "\n".join(content_parts)

            metadata.update(
                {
                    "modality": "image_caption",
                    "caption_model": VISION_CAPTION_MODEL,
                    "caption_provider": VISION_CAPTION_PROVIDER,
                    "caption_status": "ok",
                }
            )
            caption_docs.append(Document(page_content=content, metadata=metadata))
        except Exception as exc:  # noqa: BLE001
            failed_count += 1
            safe_print(f"  Warning: Vision captioning failed for {metadata.get('source', 'unknown')}: {exc}")

    return caption_docs, failed_count
