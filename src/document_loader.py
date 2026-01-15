"""Document loading and chunking utilities for multiple file formats."""

import sys
from pathlib import Path
from typing import Type

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader,
    PyMuPDFLoader,
    Docx2txtLoader,  # More reliable for .docx, no LibreOffice needed
)

from src.config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS


def safe_print(msg: str) -> None:
    """Print with fallback for encoding issues on Windows."""
    try:
        print(msg)
    except UnicodeEncodeError:
        # Replace problematic characters with ?
        safe_msg = msg.encode("ascii", errors="replace").decode("ascii")
        print(safe_msg)


def load_pptx(file_path: str) -> list[Document]:
    """Load PowerPoint files using python-pptx directly."""
    from pptx import Presentation
    
    prs = Presentation(file_path)
    text_parts = []
    
    for slide_num, slide in enumerate(prs.slides, 1):
        slide_text = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                slide_text.append(shape.text)
        if slide_text:
            text_parts.append(f"[Slide {slide_num}]\n" + "\n".join(slide_text))
    
    content = "\n\n".join(text_parts)
    if not content.strip():
        return []
    
    return [Document(page_content=content, metadata={"source": file_path})]


def load_docx(file_path: str) -> list[Document]:
    """Load Word documents using python-docx directly."""
    from docx import Document as DocxDocument
    
    doc = DocxDocument(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    
    # Also extract text from tables
    for table in doc.tables:
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_text:
                paragraphs.append(" | ".join(row_text))
    
    content = "\n\n".join(paragraphs)
    if not content.strip():
        return []
    
    return [Document(page_content=content, metadata={"source": file_path})]


# Map file extensions to their respective loaders
LOADER_MAP: dict[str, Type[BaseLoader] | str] = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".pdf": PyMuPDFLoader,
    ".docx": "docx",  # Custom loader
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
    ".csv": CSVLoader,
    ".pptx": "pptx",  # Custom loader
}

# Legacy formats that need LibreOffice - we'll skip these with a warning
LEGACY_FORMATS = {".doc", ".ppt"}


def get_loader_for_file(file_path: Path) -> BaseLoader | None:
    """Get the appropriate loader for a file based on its extension."""
    ext = file_path.suffix.lower()
    
    if ext in LEGACY_FORMATS:
        return None  # Skip legacy formats
    
    if ext not in LOADER_MAP:
        return None
    
    loader_class = LOADER_MAP[ext]
    
    # Custom loaders return string markers
    if isinstance(loader_class, str):
        return loader_class  # type: ignore
    
    # CSVLoader and TextLoader need encoding specified
    if loader_class in (CSVLoader, TextLoader):
        return loader_class(str(file_path), encoding="utf-8")
    
    return loader_class(str(file_path))


def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """Load all supported documents from the specified directory (recursive)."""
    documents = []
    skipped_legacy = []
    
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
    
    # Collect all files with supported extensions (recursive search)
    files_to_load = []
    for ext in SUPPORTED_EXTENSIONS:
        files_to_load.extend(docs_dir.glob(f"**/*{ext}"))
    
    # Filter out temp files (e.g., ~$document.docx)
    files_to_load = [f for f in files_to_load if not f.name.startswith("~$")]
    
    # Sort for consistent ordering
    files_to_load = sorted(files_to_load, key=lambda x: str(x).lower())
    
    if not files_to_load:
        return documents
    
    safe_print(f"Found {len(files_to_load)} files to process...")
    
    for i, file_path in enumerate(files_to_load, 1):
        ext = file_path.suffix.lower()
        
        # Track skipped legacy files
        if ext in LEGACY_FORMATS:
            skipped_legacy.append(file_path.name)
            continue
        
        try:
            safe_print(f"  [{i}/{len(files_to_load)}] Loading: {file_path.name}")
            
            # Custom loaders
            if ext == ".docx":
                loaded_docs = load_docx(str(file_path))
            elif ext == ".pptx":
                loaded_docs = load_pptx(str(file_path))
            else:
                loader = get_loader_for_file(file_path)
                if loader is None:
                    continue
                loaded_docs = loader.load()
            
            # Add rich metadata for better context and filtering
            rel_path = file_path.relative_to(docs_dir)
            title = file_path.stem.lower()  # filename without extension, lowercase for filtering
            file_type = file_path.suffix.lower().lstrip(".")  # extension without dot
            
            for doc in loaded_docs:
                doc.metadata["source"] = str(rel_path)
                doc.metadata["title"] = title
                doc.metadata["file_type"] = file_type
            documents.extend(loaded_docs)
            
        except Exception as e:
            # Log error but continue with other files
            safe_print(f"  Warning: Failed to load {file_path.name}: {e}")
    
    if skipped_legacy:
        safe_print(f"\nSkipped {len(skipped_legacy)} legacy files (.doc/.ppt) - need LibreOffice to convert")
    
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
