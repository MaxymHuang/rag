"""Shared RAG service operations for CLI and API consumers."""

from collections.abc import Callable
from typing import Literal

from langchain_core.documents import Document

from src.config import DOCS_DIR, EMBEDDING_MODEL, LLM_MODEL, NOTION_DATABASE_ID, NOTION_TOKEN
from src.document_loader import chunk_documents, load_and_chunk_documents
from src.notion_loader import load_notion_documents
from src.rag_chain import query_rag
from src.vector_store import add_documents, clear_vector_store, get_document_count

IngestSource = Literal["all", "local", "notion"]
QueryMode = Literal["hybrid", "vector", "keyword"]
ProgressCallback = Callable[[str, int, str], None]


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
    notion_chunks_count = 0
    notion_pages_count = 0

    _emit_progress(progress_callback, "starting", 0, "Starting ingestion")

    if source in ("all", "local"):
        _emit_progress(progress_callback, "local_loading", 10, f"Loading local docs from {DOCS_DIR}")
        local_chunks = load_and_chunk_documents()
        local_chunks_count = len(local_chunks)
        all_chunks.extend(local_chunks)
        _emit_progress(
            progress_callback,
            "local_done",
            30 if source == "all" else 60,
            f"Loaded {local_chunks_count} local chunks",
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
            "notion_chunks": notion_chunks_count,
            "notion_pages": notion_pages_count,
        }

    _emit_progress(progress_callback, "embedding", 80, f"Embedding chunks with {EMBEDDING_MODEL}")
    count = add_documents(all_chunks)
    _emit_progress(progress_callback, "completed", 100, f"Ingested {count} chunks")

    return {
        "total_chunks": len(all_chunks),
        "ingested_chunks": count,
        "local_chunks": local_chunks_count,
        "notion_chunks": notion_chunks_count,
        "notion_pages": notion_pages_count,
    }


def query_documents(
    question: str,
    search_mode: QueryMode = "hybrid",
    title_filter: str | None = None,
) -> tuple[str, list[Document]]:
    """Query the RAG chain after ensuring the vector store is populated."""
    doc_count = get_document_count()
    if doc_count == 0:
        raise ValueError("No documents in vector store. Run ingest first.")
    return query_rag(question, search_mode=search_mode, title_filter=title_filter)


def get_status() -> dict:
    """Return current RAG service status."""
    notion_configured = bool(NOTION_TOKEN and NOTION_DATABASE_ID)
    return {
        "documents_directory": str(DOCS_DIR),
        "notion_configured": notion_configured,
        "notion_database_id": NOTION_DATABASE_ID if notion_configured else "",
        "chunk_count": get_document_count(),
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
    }


def clear_documents() -> bool:
    """Clear all data from the vector store."""
    return clear_vector_store()

