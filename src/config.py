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
MODELS_DIR = PROJECT_ROOT / "models"

# Embedding model (HuggingFace)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5").strip()
AVAILABLE_EMBEDDING_MODELS = os.getenv("AVAILABLE_EMBEDDING_MODELS", EMBEDDING_MODEL).strip()
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda:0")
EMBEDDING_BATCH_SIZE = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "32")))
EMBEDDING_OOM_RETRY_BATCH_SIZE = max(1, int(os.getenv("EMBEDDING_OOM_RETRY_BATCH_SIZE", "8")))
EMBEDDING_NORMALIZE = os.getenv("EMBEDDING_NORMALIZE", "true").strip().lower() in {"1", "true", "yes", "on"}
EMBEDDING_OOM_CPU_FALLBACK = (
    os.getenv("EMBEDDING_OOM_CPU_FALLBACK", "false").strip().lower() in {"1", "true", "yes", "on"}
)
LLM_MODEL = os.getenv("LLM_MODEL", "unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-Q4_K_M.gguf")
AVAILABLE_LLM_MODELS = os.getenv("AVAILABLE_LLM_MODELS", "")
_ACTIVE_LLM_MODEL = LLM_MODEL
LLAMACPP_N_GPU_LAYERS = int(os.getenv("LLAMACPP_N_GPU_LAYERS", "-1"))
LLAMACPP_N_CTX = int(os.getenv("LLAMACPP_N_CTX", "4096"))
_LLAMACPP_N_THREADS_RAW = os.getenv("LLAMACPP_N_THREADS", "").strip()
LLAMACPP_N_THREADS = int(_LLAMACPP_N_THREADS_RAW) if _LLAMACPP_N_THREADS_RAW else None
LLAMACPP_TEMPERATURE = float(os.getenv("LLAMACPP_TEMPERATURE", "0.1"))
LLAMACPP_VERBOSE = os.getenv("LLAMACPP_VERBOSE", "false").strip().lower() in {"1", "true", "yes", "on"}

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
TOP_K_RESULTS = 8

# ChromaDB collection name
COLLECTION_NAME = "agent_docs"
VECTOR_DB_PROVIDER = os.getenv("VECTOR_DB_PROVIDER", "chroma").strip().lower()
AVAILABLE_VECTOR_DB_PROVIDERS = os.getenv("AVAILABLE_VECTOR_DB_PROVIDERS", "chroma").strip().lower()

# Supported document extensions
SUPPORTED_EXTENSIONS = [
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ".doc",
    ".xlsx",
    ".xls",
    ".csv",
    ".pptx",
    ".ppt",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".tiff",
]

# Multimodal vision settings
VISION_ENABLED = os.getenv("VISION_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
VISION_CAPTION_PROVIDER = os.getenv("VISION_CAPTION_PROVIDER", "llamacpp").strip().lower()
VISION_CAPTION_MODEL = os.getenv(
    "VISION_CAPTION_MODEL",
    "mys/ggml_llava-v1.5-7b/ggml-model-q5_k.gguf",
).strip()
VISION_MMPROJ_MODEL = os.getenv(
    "VISION_MMPROJ_MODEL",
    "mys/ggml_llava-v1.5-7b/mmproj-model-f16.gguf",
).strip()
VISION_MAX_IMAGES_PER_DOC = max(1, int(os.getenv("VISION_MAX_IMAGES_PER_DOC", "16")))
OCR_ENABLED = os.getenv("OCR_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}

# Notion settings
NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID", "")


def get_llm_model() -> str:
    """Return the active LLM model for runtime requests."""
    return _ACTIVE_LLM_MODEL


def set_llm_model(model: str) -> None:
    """Set the active LLM model for the running process."""
    global _ACTIVE_LLM_MODEL
    _ACTIVE_LLM_MODEL = model
    os.environ["LLM_MODEL"] = model