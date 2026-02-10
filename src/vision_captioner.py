"""Vision caption service for converting image artifacts into text documents."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document
from ollama import Client

from src.config import (
    OCR_ENABLED,
    OLLAMA_BASE_URL,
    VISION_CAPTION_MODEL,
    VISION_CAPTION_PROVIDER,
    VISION_ENABLED,
)
from src.document_loader import VisualArtifact, safe_print


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
class OllamaVisionCaptionProvider:
    """Ollama-backed vision captioning provider."""

    model: str = VISION_CAPTION_MODEL
    base_url: str = OLLAMA_BASE_URL

    def __post_init__(self) -> None:
        self._client = Client(host=self.base_url)

    def caption_image(self, image_bytes: bytes, metadata: dict) -> str:
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        response = self._client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": CAPTION_PROMPT,
                    "images": [image_b64],
                }
            ],
        )
        return response.get("message", {}).get("content", "").strip()


def _build_provider() -> VisionCaptionProvider:
    provider = VISION_CAPTION_PROVIDER.strip().lower()
    if provider == "ollama":
        return OllamaVisionCaptionProvider()
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
