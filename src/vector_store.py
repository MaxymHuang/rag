"""ChromaDB vector store operations."""

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import CHROMA_DB_DIR, COLLECTION_NAME, TOP_K_RESULTS
from src.embeddings import get_embeddings


def get_vector_store() -> Chroma:
    """Get or create the ChromaDB vector store."""
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DB_DIR)
    )


def add_documents(documents: list[Document]) -> int:
    """Add documents to the vector store. Returns count of documents added."""
    if not documents:
        return 0
    
    vector_store = get_vector_store()
    vector_store.add_documents(documents)
    return len(documents)


def similarity_search(query: str, k: int = TOP_K_RESULTS) -> list[Document]:
    """Search for similar documents given a query."""
    vector_store = get_vector_store()
    return vector_store.similarity_search(query, k=k)


def clear_vector_store() -> bool:
    """Clear all data from the vector store."""
    if CHROMA_DB_DIR.exists():
        shutil.rmtree(CHROMA_DB_DIR)
        return True
    return False


def get_document_count() -> int:
    """Get the number of documents in the vector store."""
    if not CHROMA_DB_DIR.exists():
        return 0
    
    vector_store = get_vector_store()
    collection = vector_store._collection
    return collection.count()
