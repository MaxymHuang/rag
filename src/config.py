"""Configuration settings for the RAG agent."""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "agent-doc"
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# Ollama settings
EMBEDDING_MODEL = "hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF"
LLM_MODEL = "hf.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF:Q4_K_M"
OLLAMA_BASE_URL = "http://localhost:11434"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
TOP_K_RESULTS = 4

# ChromaDB collection name
COLLECTION_NAME = "agent_docs"
