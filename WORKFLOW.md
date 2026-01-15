# RAG Agent Workflow

## Overview

This RAG (Retrieval-Augmented Generation) system allows you to query your documents using natural language. It combines vector similarity search with LLM generation to provide accurate, context-aware answers.

---

## Architecture

```
src/
├── config.py           # Configuration (models, paths, settings)
├── document_loader.py  # Load & chunk text files
├── embeddings.py       # Ollama embeddings wrapper
├── vector_store.py     # ChromaDB operations
├── rag_chain.py        # RAG pipeline (retrieval + generation)
└── cli.py              # CLI interface
```

---

## Phase 1: Document Ingestion

Run once to index your documents into the vector database.

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  agent-doc/     │      │  document_loader │      │   embeddings    │      │  vector_store   │
│  *.txt files    │ ───► │  load + chunk    │ ───► │  Ollama embed   │ ───► │  ChromaDB       │
│                 │      │  (1000 chars)    │      │  (nomic-embed)  │      │  (persistent)   │
└─────────────────┘      └──────────────────┘      └─────────────────┘      └─────────────────┘
```

### Steps:
1. **Load** - Read all `.txt` files from `agent-doc/` directory
2. **Chunk** - Split documents into 1000-character chunks with 200-character overlap
3. **Embed** - Convert each chunk into a vector using `nomic-embed-text` model
4. **Store** - Save vectors in ChromaDB (persisted to `data/chroma_db/`)

### Command:
```bash
rag ingest
```

---

## Phase 2: Query Processing

Execute each time you ask a question.

```
                                    ┌─────────────────┐
                                    │   ChromaDB      │
                                    │   Vector Store  │
                                    └────────┬────────┘
                                             │
                                             │ 2. Find top-4
                                             │    similar chunks
                                             ▼
┌─────────────────┐   1. Embed    ┌─────────────────┐
│  User Question  │ ────────────► │  Similarity     │
│  "What is X?"   │               │  Search         │
└─────────────────┘               └────────┬────────┘
                                           │
                                           │ 3. Retrieved chunks
                                           ▼
                                  ┌─────────────────┐
                                  │  Build Prompt   │
                                  │  Context + Q    │
                                  └────────┬────────┘
                                           │
                                           │ 4. Send to LLM
                                           ▼
                                  ┌─────────────────┐
                                  │  Mistral-7B     │
                                  │  (via Ollama)   │
                                  └────────┬────────┘
                                           │
                                           │ 5. Generated answer
                                           ▼
                                  ┌─────────────────┐
                                  │  Final Answer   │
                                  │  + Sources      │
                                  └─────────────────┘
```

### Steps:
1. **Embed Query** - Convert user question into a vector
2. **Search** - Find top-4 most similar document chunks in ChromaDB
3. **Build Prompt** - Combine retrieved chunks as context with the question
4. **Generate** - Send prompt to Mistral-7B LLM via Ollama
5. **Return** - Display answer and optionally show source documents

### Command:
```bash
rag query "your question here"
rag query "your question here" -s   # include source snippets
```

---

## CLI Commands Reference

| Command | Description |
|---------|-------------|
| `rag ingest` | Load, chunk, embed, and store documents |
| `rag query "question"` | Ask a question and get an answer |
| `rag query "question" -s` | Ask with source snippets displayed |
| `rag status` | Show current configuration and chunk count |
| `rag clear` | Delete all data from vector store |

---

## Configuration

Defined in `src/config.py`:

| Setting | Value |
|---------|-------|
| Embedding Model | `nomic-embed-text-v1.5` |
| LLM Model | `Mistral-7B-Instruct-v0.3` |
| Chunk Size | 1000 characters |
| Chunk Overlap | 200 characters |
| Top-K Results | 4 documents |

---

## Data Flow Summary

| Phase | Input | Process | Output |
|-------|-------|---------|--------|
| **Ingest** | `.txt` files | chunk → embed → store | Vectors in ChromaDB |
| **Query** | User question | embed → search → prompt → LLM | Answer + sources |
