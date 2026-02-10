"""Configuration settings for the RAG agent."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "agent-doc"
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# Embedding model (HuggingFace)
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda:0")
LLM_MODEL = "hf.co/unsloth/gpt-oss-20b-GGUF:Q4_K_M"
OLLAMA_BASE_URL = "http://localhost:11434"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
TOP_K_RESULTS = 8

# ChromaDB collection name
COLLECTION_NAME = "agent_docs"

# Supported document extensions
SUPPORTED_EXTENSIONS = [".txt", ".md", ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".pptx", ".ppt"]

# Notion settings
NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID", "")