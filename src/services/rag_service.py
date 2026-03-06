"""Shared RAG service operations for CLI and API consumers."""

from collections.abc import Callable
import threading
from typing import Literal

from langchain_core.documents import Document

from src.config import DOCS_DIR, EMBEDDING_MODEL, NOTION_DATABASE_ID, NOTION_TOKEN, get_llm_model
from src.document_loader import chunk_documents, load_multimodal_artifacts
from src.notion_loader import load_notion_documents
from src.rag_chain import query_rag
from src.vector_store import add_documents, clear_vector_store, get_document_count
from src.vision_captioner import caption_visual_artifacts

IngestSource = Literal["all", "local", "notion"]
QueryMode = Literal["hybrid", "vector", "keyword"]
ContextSource = Literal["local", "notion"]
ProgressCallback = Callable[[str, int, str], None]
_embedding_work_lock = threading.Lock()


def _emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    progress: int,
    message: str,
) -> None:
    if callback is None:
        return
    callback(stage, progress, message)


def ingest_documents(
    source: IngestSource = "all",
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Ingest documents from local files and/or Notion into the vector store."""
    all_chunks: list[Document] = []
    local_chunks_count = 0
    local_text_chunks_count = 0
    local_caption_chunks_count = 0
    local_image_artifacts_count = 0
    local_caption_failed_count = 0
    notion_chunks_count = 0
    notion_pages_count = 0

    _emit_progress(progress_callback, "starting", 0, "Starting ingestion")

    if source in ("all", "local"):
        _emit_progress(progress_callback, "local_loading", 10, f"Loading local docs from {DOCS_DIR}")
        local_docs, image_artifacts = load_multimodal_artifacts()

        _emit_progress(
            progress_callback,
            "image_extracting",
            20 if source == "all" else 35,
            f"Extracted {len(image_artifacts)} image artifacts from local files",
        )

        local_text_chunks = chunk_documents(local_docs)
        local_text_chunks_count = len(local_text_chunks)
        local_image_artifacts_count = len(image_artifacts)

        local_caption_chunks: list[Document] = []
        if image_artifacts:
            _emit_progress(
                progress_callback,
                "image_captioning",
                25 if source == "all" else 45,
                f"Captioning {len(image_artifacts)} images",
            )
            local_caption_chunks, local_caption_failed_count = caption_visual_artifacts(image_artifacts)

        local_caption_chunks_count = len(local_caption_chunks)
        local_chunks_count = local_text_chunks_count + local_caption_chunks_count
        all_chunks.extend(local_text_chunks)
        all_chunks.extend(local_caption_chunks)
        _emit_progress(
            progress_callback,
            "local_done",
            30 if source == "all" else 60,
            (
                f"Loaded {local_chunks_count} local chunks "
                f"({local_text_chunks_count} text, {local_caption_chunks_count} caption)"
            ),
        )

    if source in ("all", "notion"):
        if not NOTION_TOKEN or not NOTION_DATABASE_ID:
            if source == "notion":
                raise ValueError("NOTION_TOKEN and NOTION_DATABASE_ID must be set")
        else:
            _emit_progress(progress_callback, "notion_loading", 40, "Loading Notion documents")
            notion_docs = load_notion_documents()
            notion_pages_count = len(notion_docs)
            notion_chunks = chunk_documents(notion_docs)
            notion_chunks_count = len(notion_chunks)
            all_chunks.extend(notion_chunks)
            _emit_progress(
                progress_callback,
                "notion_done",
                70,
                f"Loaded {notion_chunks_count} Notion chunks from {notion_pages_count} pages",
            )

    if not all_chunks:
        _emit_progress(progress_callback, "completed", 100, "No documents found to ingest")
        return {
            "total_chunks": 0,
            "ingested_chunks": 0,
            "local_chunks": local_chunks_count,
            "local_text_chunks": local_text_chunks_count,
            "local_caption_chunks": local_caption_chunks_count,
            "local_image_artifacts": local_image_artifacts_count,
            "local_caption_failed": local_caption_failed_count,
            "notion_chunks": notion_chunks_count,
            "notion_pages": notion_pages_count,
        }

    _emit_progress(progress_callback, "embedding", 80, f"Embedding chunks with {EMBEDDING_MODEL}")
    with _embedding_work_lock:
        count = add_documents(all_chunks)
    _emit_progress(progress_callback, "completed", 100, f"Ingested {count} chunks")

    return {
        "total_chunks": len(all_chunks),
        "ingested_chunks": count,
        "local_chunks": local_chunks_count,
        "local_text_chunks": local_text_chunks_count,
        "local_caption_chunks": local_caption_chunks_count,
        "local_image_artifacts": local_image_artifacts_count,
        "local_caption_failed": local_caption_failed_count,
        "notion_chunks": notion_chunks_count,
        "notion_pages": notion_pages_count,
    }


def query_documents(
    question: str,
    search_mode: QueryMode = "hybrid",
    title_filter: str | None = None,
    history: list[dict[str, str]] | None = None,
    context_sources: list[ContextSource] | None = None,
) -> tuple[str, list[Document]]:
    """Query the RAG chain after ensuring the vector store is populated."""
    doc_count = get_document_count()
    if doc_count == 0:
        raise ValueError("No documents in vector store. Run ingest first.")
    if search_mode == "keyword":
        return query_rag(
            question,
            search_mode=search_mode,
            title_filter=title_filter,
            history=history,
            context_sources=context_sources,
        )
    with _embedding_work_lock:
        return query_rag(
            question,
            search_mode=search_mode,
            title_filter=title_filter,
            history=history,
            context_sources=context_sources,
        )


def get_status() -> dict:
    """Return current RAG service status."""
    notion_configured = bool(NOTION_TOKEN and NOTION_DATABASE_ID)
    return {
        "documents_directory": str(DOCS_DIR),
        "notion_configured": notion_configured,
        "notion_database_id": NOTION_DATABASE_ID if notion_configured else "",
        "chunk_count": get_document_count(),
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": get_llm_model(),
    }


def clear_documents() -> bool:
    """Clear all data from the vector store."""
    return clear_vector_store()

