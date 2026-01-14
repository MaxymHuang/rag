# RAG Agent

A CLI-based RAG (Retrieval-Augmented Generation) agent using Ollama for local LLM inference and ChromaDB for vector storage.

## Prerequisites

1. **Ollama** installed and running: https://ollama.ai
2. **Python 3.10+**
3. **uv** package manager

Pull the required models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Installation

```bash
uv sync
```

## Usage

### Ingest Documents

Process and index documents from `agent-doc/`:

```bash
uv run rag ingest
```

### Query Documents

Ask questions about your documents:

```bash
uv run rag query "What powers does Congress have?"
```

Show source documents with your answer:

```bash
uv run rag query "What is the role of the President?" --show-sources
```

### Check Status

View the current state of the RAG agent:

```bash
uv run rag status
```

### Clear Vector Store

Remove all indexed documents:

```bash
uv run rag clear
```

## Configuration

Edit `src/config.py` to customize:

- `EMBEDDING_MODEL`: Ollama embedding model (default: `nomic-embed-text`)
- `LLM_MODEL`: Ollama LLM model (default: `llama3.2`)
- `CHUNK_SIZE`: Document chunk size (default: 500)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)
- `TOP_K_RESULTS`: Number of documents to retrieve (default: 4)

## Project Structure

```
rag/
├── pyproject.toml          # Project configuration
├── README.md
├── agent-doc/              # Source documents
├── src/
│   ├── cli.py              # CLI entry point
│   ├── config.py           # Configuration settings
│   ├── document_loader.py  # Document loading/chunking
│   ├── embeddings.py       # Ollama embeddings
│   ├── vector_store.py     # ChromaDB operations
│   └── rag_chain.py        # RAG query chain
└── data/
    └── chroma_db/          # Persistent vector storage
```
