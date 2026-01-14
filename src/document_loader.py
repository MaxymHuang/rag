"""Document loading and chunking utilities."""

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """Load all text documents from the specified directory."""
    documents = []
    
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
    
    for file_path in docs_dir.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8")
        doc = Document(
            page_content=content,
            metadata={"source": str(file_path.name)}
        )
        documents.append(doc)
    
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> list[Document]:
    """Split documents into smaller chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks


def load_and_chunk_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """Load documents and split them into chunks."""
    documents = load_documents(docs_dir)
    chunks = chunk_documents(documents)
    return chunks
