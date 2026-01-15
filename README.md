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

Process and index documents from local files and/or Notion:

```bash
# Ingest from all sources (local + Notion)
uv run rag ingest

# Ingest from local files only
uv run rag ingest --source local

# Ingest from Notion only
uv run rag ingest --source notion
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

## Notion Integration

To load documents from a Notion database:

1. Create an integration at https://www.notion.so/my-integrations
2. Share your target Notion database with the integration (click "..." > "Connections" > select your integration)
3. Create a `.env` file in the project root:

```
NOTION_TOKEN=secret_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
NOTION_DATABASE_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

The database ID can be found in the Notion database URL (the 32-character hex string after the workspace name).

Each row in the database becomes a document. Both page properties (columns) and page content (body) are extracted.

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
├── agent-doc/              # Local source documents
├── src/
│   ├── cli.py              # CLI entry point
│   ├── config.py           # Configuration settings
│   ├── document_loader.py  # Local document loading/chunking
│   ├── notion_loader.py    # Notion API document loading
│   ├── embeddings.py       # Ollama embeddings
│   ├── vector_store.py     # ChromaDB operations
│   └── rag_chain.py        # RAG query chain
└── data/
    └── chroma_db/          # Persistent vector storage
```
