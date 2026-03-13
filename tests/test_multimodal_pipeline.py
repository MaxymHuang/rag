"""Tests for multimodal ingestion and caption fallback behavior."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from langchain_core.documents import Document

from src.document_loader import VisualArtifact, load_multimodal_artifacts
from src.rag_chain import format_context
from src.services.rag_service import ingest_documents
from src.vision_captioner import caption_visual_artifacts


class MultimodalLoaderTests(TestCase):
    def test_pdf_with_embedded_image_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            docs_dir = Path(tmp)
            (docs_dir / "diagram.pdf").write_bytes(b"%PDF-1.4")

            fake_pdf_docs = [Document(page_content="PDF text", metadata={})]
            fake_artifacts = [
                VisualArtifact(
                    content=b"png",
                    metadata={
                        "source": str(docs_dir / "diagram.pdf"),
                        "file_type": "pdf",
                        "page_or_slide": 1,
                        "image_index": 1,
                        "image_ext": "png",
                        "image_mime": "image/png",
                        "modality": "image",
                    },
                )
            ]

            with (
                patch("src.document_loader.load_pdf_text", return_value=fake_pdf_docs),
                patch("src.document_loader.extract_pdf_images", return_value=fake_artifacts),
            ):
                text_docs, image_artifacts = load_multimodal_artifacts(docs_dir)

            self.assertEqual(len(text_docs), 1)
            self.assertEqual(len(image_artifacts), 1)
            self.assertEqual(text_docs[0].metadata["modality"], "text")
            self.assertEqual(image_artifacts[0].metadata["page_or_slide"], 1)

    def test_pptx_with_image_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            docs_dir = Path(tmp)
            (docs_dir / "slides.pptx").write_bytes(b"pptx-bytes")

            fake_ppt_docs = [Document(page_content="Slide text", metadata={})]
            fake_artifacts = [
                VisualArtifact(
                    content=b"jpeg",
                    metadata={
                        "source": str(docs_dir / "slides.pptx"),
                        "file_type": "pptx",
                        "page_or_slide": 2,
                        "image_index": 1,
                        "image_ext": "jpg",
                        "image_mime": "image/jpeg",
                        "modality": "image",
                    },
                )
            ]

            with (
                patch("src.document_loader.load_pptx_text", return_value=fake_ppt_docs),
                patch("src.document_loader.extract_pptx_images", return_value=fake_artifacts),
            ):
                text_docs, image_artifacts = load_multimodal_artifacts(docs_dir)

            self.assertEqual(len(text_docs), 1)
            self.assertEqual(len(image_artifacts), 1)
            self.assertEqual(image_artifacts[0].metadata["file_type"], "pptx")

    def test_standalone_image_is_loaded_as_artifact(self) -> None:
        with TemporaryDirectory() as tmp:
            docs_dir = Path(tmp)
            (docs_dir / "idea.png").write_bytes(b"\x89PNG\r\n")
            text_docs, image_artifacts = load_multimodal_artifacts(docs_dir)

            self.assertEqual(text_docs, [])
            self.assertEqual(len(image_artifacts), 1)
            self.assertEqual(image_artifacts[0].metadata["image_mime"], "image/png")
            self.assertEqual(image_artifacts[0].metadata["parent_source"], "idea.png")


class VisionAndIngestTests(TestCase):
    def test_vision_disabled_returns_no_caption_docs(self) -> None:
        artifact = VisualArtifact(content=b"image", metadata={"source": "x.png"})
        with patch("src.vision_captioner.VISION_ENABLED", False):
            docs, failed = caption_visual_artifacts([artifact])
        self.assertEqual(docs, [])
        self.assertEqual(failed, 0)

    def test_ingest_counts_text_and_caption_chunks(self) -> None:
        text_docs = [Document(page_content="hello world", metadata={"source": "a.pdf", "title": "a", "file_type": "pdf"})]
        image_artifacts = [VisualArtifact(content=b"i", metadata={"source": "a.pdf", "page_or_slide": 1})]
        caption_docs = [
            Document(
                page_content="Image summary from a.pdf: diagram with arrows",
                metadata={"source": "a.pdf", "modality": "image_caption", "page_or_slide": 1},
            )
        ]

        with (
            patch("src.services.rag_service.load_multimodal_artifacts", return_value=(text_docs, image_artifacts)),
            patch("src.services.rag_service.chunk_documents", return_value=text_docs),
            patch("src.services.rag_service.caption_visual_artifacts", return_value=(caption_docs, 0)),
            patch("src.services.rag_service.add_documents", return_value=2),
        ):
            result = ingest_documents(source="local")

        self.assertEqual(result["total_chunks"], 2)
        self.assertEqual(result["local_text_chunks"], 1)
        self.assertEqual(result["local_caption_chunks"], 1)
        self.assertEqual(result["local_image_artifacts"], 1)

    def test_context_shows_modality_and_page_or_slide(self) -> None:
        docs = [
            Document(
                page_content="Diagram caption text",
                metadata={"source": "deck.pptx", "file_type": "pptx", "modality": "image_caption", "page_or_slide": 4},
            )
        ]
        context = format_context(docs)
        self.assertIn("Modality: image_caption", context)
        self.assertIn("PageOrSlide: 4", context)
