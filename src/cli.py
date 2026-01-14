"""CLI interface for the RAG agent."""

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.document_loader import load_and_chunk_documents
from src.vector_store import add_documents, clear_vector_store, get_document_count
from src.rag_chain import query_rag
from src.config import DOCS_DIR, LLM_MODEL, EMBEDDING_MODEL

console = Console()


@click.group()
def main():
    """RAG Agent CLI - Query your documents with AI."""
    pass


@main.command()
def ingest():
    """Ingest documents from agent-doc/ into the vector store."""
    console.print(f"\n[bold blue]Ingesting documents from:[/] {DOCS_DIR}\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Load and chunk documents
        task = progress.add_task("Loading and chunking documents...", total=None)
        try:
            chunks = load_and_chunk_documents()
            progress.update(task, description=f"Loaded {len(chunks)} chunks")
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/] {e}")
            return
        
        if not chunks:
            console.print("[yellow]No documents found to ingest.[/]")
            return
        
        # Add to vector store
        progress.update(task, description=f"Embedding {len(chunks)} chunks (using {EMBEDDING_MODEL})...")
        count = add_documents(chunks)
        progress.update(task, description="Done!")
    
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
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Thinking...", total=None)
        answer, sources = query_rag(question)
        progress.update(task, description="Done!")
    
    # Display answer
    console.print(Panel(
        Markdown(answer),
        title="[bold green]Answer[/]",
        border_style="green"
    ))
    
    # Show sources if requested
    if show_sources and sources:
        console.print("\n[bold blue]Sources:[/]")
        for i, doc in enumerate(sources, 1):
            source_name = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
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
    
    console.print("\n[bold]RAG Agent Status[/]\n")
    console.print(f"  Documents directory: {DOCS_DIR}")
    console.print(f"  Chunks in vector store: {doc_count}")
    console.print(f"  Embedding model: {EMBEDDING_MODEL}")
    console.print(f"  LLM model: {LLM_MODEL}")
    console.print()


if __name__ == "__main__":
    main()
