# RAG Chain Improvements

This document describes the enhancements made to the RAG system.

## 1. HuggingFace GPU Embeddings

**Previous**: Ollama embeddings with `nomic-embed-text-v1.5-GGUF` (768 dimensions)

**New**: HuggingFace embeddings with `BAAI/bge-large-en-v1.5` (1024 dimensions)

### Benefits
- Direct CUDA GPU acceleration via `sentence-transformers`
- Higher dimensional embeddings capture more semantic nuance
- Normalized embeddings for consistent similarity scores
- Industry-leading retrieval quality on MTEB benchmarks

### Configuration
```python
# src/config.py
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# src/embeddings.py
HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)
```

### Migration
After upgrading, rebuild the vector store:
```bash
rag clear --yes
rag ingest
```

---

## 2. Enhanced System Prompt

The system prompt now provides structured guidelines for the LLM:

### Key Improvements
1. **Accuracy**: Explicit instruction to only use provided context
2. **Citations**: Mandatory `[X]` source notation
3. **Reasoning**: Explains synthesis from multiple sources
4. **Uncertainty**: Explicitly states gaps in information
5. **Structure**: Uses bullet points for complex answers

### Edge Case Handling
- No relevant information → Clear statement
- Partial information → Provide available data + note gaps
- Conflicting sources → Present both with citations

---

## 3. Metadata Filtering

### New Metadata Fields

| Field | Description | Source |
|-------|-------------|--------|
| `source` | Full relative path | All documents |
| `title` | Lowercase filename (no extension) | Local files |
| `file_type` | File extension or "notion" | All documents |

### Title Filter Usage

Filter search results by title/filename substring (case-insensitive):

```bash
# CLI usage
rag query "What are the key points?" --filter-title "report"
rag query "Find meeting notes" -t "2024"

# Combine with search mode
rag query "exact date" --mode keyword --filter-title "schedule"
```

### Programmatic Usage

```python
from src.rag_chain import query_rag

answer, docs = query_rag(
    question="What is the budget?",
    search_mode="hybrid",
    title_filter="finance"
)
```

### How Filtering Works

- **Vector search**: Uses ChromaDB's native `$contains` filter
- **Keyword/BM25 search**: Post-filters results in Python
- **Hybrid search**: Filters both vector and BM25 results before RRF fusion

---

## Dependencies Added

```toml
langchain-huggingface>=0.1.0
sentence-transformers>=3.0.0
```

---

## Files Modified

| File | Changes |
|------|---------|
| `pyproject.toml` | New dependencies |
| `src/config.py` | Updated `EMBEDDING_MODEL` |
| `src/embeddings.py` | HuggingFace with GPU |
| `src/rag_chain.py` | New system prompt, title_filter param |
| `src/document_loader.py` | Extract title/file_type metadata |
| `src/notion_loader.py` | Add file_type metadata |
| `src/vector_store.py` | title_filter for all search functions |
| `src/cli.py` | `--filter-title` option |
