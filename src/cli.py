"""CLI interface for the RAG agent."""

import sys
import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.document_loader import load_and_chunk_documents, chunk_documents
from src.notion_loader import load_notion_documents
from src.vector_store import add_documents, clear_vector_store, get_document_count
from src.rag_chain import query_rag
from src.config import DOCS_DIR, LLM_MODEL, EMBEDDING_MODEL, NOTION_TOKEN, NOTION_DATABASE_ID

# Use UTF-8 for Windows console if possible
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

console = Console(force_terminal=True)


def sanitize_text(text: str) -> str:
    """Replace non-ASCII characters for safe console output."""
    return text.encode("ascii", errors="replace").decode("ascii")


@click.group()
def main():
    """RAG Agent CLI - Query your documents with AI."""
    pass


@main.command()
@click.option(
    "--source", "-src",
    type=click.Choice(["all", "local", "notion"], case_sensitive=False),
    default="all",
    help="Document source: 'local' (agent-doc/), 'notion', or 'all' (default)"
)
def ingest(source: str):
    """Ingest documents from local files and/or Notion into the vector store."""
    all_chunks = []
    
    # Load from local files
    if source in ("all", "local"):
        console.print(f"\n[bold blue]Loading local documents from:[/] {DOCS_DIR}")
        try:
            local_chunks = load_and_chunk_documents()
            all_chunks.extend(local_chunks)
            console.print(f"  Loaded {len(local_chunks)} chunks from local files")
        except FileNotFoundError as e:
            if source == "local":
                console.print(f"[red]Error:[/] {e}")
                return
            console.print(f"  [yellow]Skipped:[/] {e}")
    
    # Load from Notion
    if source in ("all", "notion"):
        if not NOTION_TOKEN or not NOTION_DATABASE_ID:
            if source == "notion":
                console.print("[red]Error:[/] NOTION_TOKEN and NOTION_DATABASE_ID must be set")
                return
            console.print("  [dim]Skipped Notion: credentials not configured[/]")
        else:
            console.print("\n[bold blue]Loading documents from Notion...[/]")
            try:
                notion_docs = load_notion_documents()
                notion_chunks = chunk_documents(notion_docs)
                all_chunks.extend(notion_chunks)
                console.print(f"  Loaded {len(notion_chunks)} chunks from {len(notion_docs)} Notion pages")
            except Exception as e:
                if source == "notion":
                    console.print(f"[red]Error loading Notion:[/] {e}")
                    return
                console.print(f"  [yellow]Notion error:[/] {e}")
    
    if not all_chunks:
        console.print("\n[yellow]No documents found to ingest.[/]")
        return
    
    console.print(f"\n[bold]Total:[/] {len(all_chunks)} chunks")
    console.print(f"Embedding chunks using {EMBEDDING_MODEL}...")
    
    count = add_documents(all_chunks)
    
    console.print(f"\n[green]Successfully ingested {count} chunks into the vector store.[/]\n")


@main.command()
@click.argument("question")
@click.option("--show-sources", "-s", is_flag=True, help="Show source documents")
def query(question: str, show_sources: bool):
    """Ask a question about your documents."""
    doc_count = get_document_count()
    
    if doc_count == 0:
        console.print("[yellow]No documents in vector store. Run 'rag ingest' first.[/]")
        return
    
    console.print(f"\n[dim]Querying {doc_count} document chunks with {LLM_MODEL}...[/]\n")
    
    console.print("Thinking...")
    answer, sources = query_rag(question)
    
    # Display answer (sanitize for Windows console)
    safe_answer = sanitize_text(answer)
    console.print(Panel(
        Markdown(safe_answer),
        title="[bold green]Answer[/]",
        border_style="green"
    ))
    
    # Show sources if requested
    if show_sources and sources:
        console.print("\n[bold blue]Sources:[/]")
        for i, doc in enumerate(sources, 1):
            source_name = sanitize_text(doc.metadata.get("source", "unknown"))
            content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            preview = sanitize_text(content)
            console.print(Panel(
                preview,
                title=f"[dim]{i}. {source_name}[/]",
                border_style="dim"
            ))
    
    console.print()


@main.command()
@click.confirmation_option(prompt="Are you sure you want to clear the vector store?")
def clear():
    """Clear all data from the vector store."""
    if clear_vector_store():
        console.print("[green]Vector store cleared successfully.[/]")
    else:
        console.print("[yellow]Vector store was already empty.[/]")


@main.command()
def status():
    """Show the current status of the RAG agent."""
    doc_count = get_document_count()
    notion_configured = bool(NOTION_TOKEN and NOTION_DATABASE_ID)
    
    console.print("\n[bold]RAG Agent Status[/]\n")
    console.print(f"  Documents directory: {DOCS_DIR}")
    console.print(f"  Notion configured: {'Yes' if notion_configured else 'No'}")
    if notion_configured:
        console.print(f"  Notion database: {NOTION_DATABASE_ID[:8]}...")
    console.print(f"  Chunks in vector store: {doc_count}")
    console.print(f"  Embedding model: {EMBEDDING_MODEL}")
    console.print(f"  LLM model: {LLM_MODEL}")
    console.print()


if __name__ == "__main__":
    main()
