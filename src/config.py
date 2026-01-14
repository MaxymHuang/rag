"""Configuration settings for the RAG agent."""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "agent-doc"
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# Ollama settings
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval settings
TOP_K_RESULTS = 4

# ChromaDB collection name
COLLECTION_NAME = "agent_docs"
