"""ChromaDB vector store operations with hybrid search."""

import shutil
import threading
from collections import defaultdict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from src.config import CHROMA_DB_DIR, COLLECTION_NAME, TOP_K_RESULTS
from src.embeddings import get_embeddings

# Cache for BM25 retriever (rebuilt on ingest)
_bm25_retriever = None
_all_documents = []
_vector_store: Chroma | None = None
_vector_store_lock = threading.Lock()


def get_vector_store() -> Chroma:
    """Get or create the ChromaDB vector store."""
    global _vector_store

    if _vector_store is not None:
        return _vector_store

    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

    with _vector_store_lock:
        if _vector_store is None:
            _vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=get_embeddings(),
                persist_directory=str(CHROMA_DB_DIR),
            )
    return _vector_store


def add_documents(documents: list[Document]) -> int:
    """Add documents to the vector store. Returns count of documents added."""
    global _bm25_retriever, _all_documents
    
    if not documents:
        return 0
    
    vector_store = get_vector_store()
    vector_store.add_documents(documents)
    
    # Update BM25 index
    _all_documents.extend(documents)
    _bm25_retriever = BM25Retriever.from_documents(_all_documents, k=TOP_K_RESULTS)
    
    return len(documents)


def _get_all_docs_from_store() -> list[Document]:
    """Retrieve all documents from ChromaDB for BM25 indexing."""
    vector_store = get_vector_store()
    collection = vector_store._collection
    
    if collection.count() == 0:
        return []
    
    # Get all documents
    results = collection.get(include=["documents", "metadatas"])
    docs = []
    for i, content in enumerate(results.get("documents", [])):
        metadata = results.get("metadatas", [{}])[i] or {}
        docs.append(Document(page_content=content, metadata=metadata))
    
    return docs


def _ensure_bm25_retriever():
    """Ensure BM25 retriever is initialized."""
    global _bm25_retriever, _all_documents
    
    if _bm25_retriever is None:
        _all_documents = _get_all_docs_from_store()
        if _all_documents:
            _bm25_retriever = BM25Retriever.from_documents(_all_documents, k=TOP_K_RESULTS)


def _build_title_filter(title_filter: str | None) -> dict | None:
    """Build ChromaDB filter dict for title substring matching."""
    if not title_filter:
        return None
    # ChromaDB uses $contains for substring matching
    return {"title": {"$contains": title_filter.lower()}}


def similarity_search(
    query: str, 
    k: int = TOP_K_RESULTS,
    title_filter: str | None = None
) -> list[Document]:
    """Search for similar documents using vector similarity only.
    
    Args:
        query: Search query
        k: Number of results to return
        title_filter: Optional title substring filter (case-insensitive)
    """
    vector_store = get_vector_store()
    filter_dict = _build_title_filter(title_filter)
    
    if filter_dict:
        return vector_store.similarity_search(query, k=k, filter=filter_dict)
    return vector_store.similarity_search(query, k=k)


def hybrid_search(
    query: str, 
    k: int = TOP_K_RESULTS, 
    vector_weight: float = 0.5,
    title_filter: str | None = None
) -> list[Document]:
    """
    Hybrid search combining vector similarity (semantic) and BM25 (keyword).
    
    Uses Reciprocal Rank Fusion (RRF) to combine results.
    This improves retrieval for queries with specific terms like dates, names, codes.
    
    Args:
        query: Search query
        k: Number of results to return
        vector_weight: Weight for vector search (0-1), BM25 gets 1-vector_weight
        title_filter: Optional title substring filter (case-insensitive)
    """
    _ensure_bm25_retriever()
    
    vector_store = get_vector_store()
    filter_dict = _build_title_filter(title_filter)
    
    # Get vector search results (with optional filter)
    if filter_dict:
        vector_results = vector_store.similarity_search(query, k=k * 2, filter=filter_dict)
    else:
        vector_results = vector_store.similarity_search(query, k=k * 2)
    
    # Get BM25 results and filter by title if needed
    bm25_results = []
    if _bm25_retriever is not None:
        _bm25_retriever.k = k * 2
        bm25_results = _bm25_retriever.invoke(query)
        
        # Apply title filter to BM25 results (BM25 doesn't support native filtering)
        if title_filter:
            title_lower = title_filter.lower()
            bm25_results = [
                doc for doc in bm25_results 
                if title_lower in doc.metadata.get("title", "").lower()
            ]
    
    if not bm25_results:
        return vector_results[:k]
    
    # Reciprocal Rank Fusion
    rrf_k = 60  # Standard RRF constant
    doc_scores = defaultdict(float)
    doc_map = {}
    
    # Score vector results
    for rank, doc in enumerate(vector_results):
        doc_id = doc.page_content[:100]  # Use content prefix as ID
        doc_scores[doc_id] += vector_weight * (1.0 / (rrf_k + rank + 1))
        doc_map[doc_id] = doc
    
    # Score BM25 results
    bm25_weight = 1.0 - vector_weight
    for rank, doc in enumerate(bm25_results):
        doc_id = doc.page_content[:100]
        doc_scores[doc_id] += bm25_weight * (1.0 / (rrf_k + rank + 1))
        doc_map[doc_id] = doc
    
    # Sort by combined score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [doc_map[doc_id] for doc_id, _ in sorted_docs[:k]]


def keyword_search(
    query: str, 
    k: int = TOP_K_RESULTS,
    title_filter: str | None = None
) -> list[Document]:
    """Pure keyword-based BM25 search (good for exact matches, dates, codes).
    
    Args:
        query: Search query
        k: Number of results to return
        title_filter: Optional title substring filter (case-insensitive)
    """
    _ensure_bm25_retriever()
    
    if _bm25_retriever is None:
        return []
    
    _bm25_retriever.k = k
    results = _bm25_retriever.invoke(query)
    
    # Apply title filter (BM25 doesn't support native filtering)
    if title_filter:
        title_lower = title_filter.lower()
        results = [
            doc for doc in results 
            if title_lower in doc.metadata.get("title", "").lower()
        ]
    
    return results[:k]


def clear_vector_store() -> bool:
    """Clear all data from the vector store."""
    global _bm25_retriever, _all_documents, _vector_store

    _bm25_retriever = None
    _all_documents = []
    _vector_store = None

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
