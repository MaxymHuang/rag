"""CLI interface for the RAG agent."""

import sys
import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.services.rag_service import clear_documents, get_status, ingest_documents, query_documents

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
    try:
        result = ingest_documents(source=source)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/] {e}")
        return
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        return
    except Exception as e:
        console.print(f"[red]Ingestion failed:[/] {e}")
        return

    if result["total_chunks"] == 0:
        console.print("\n[yellow]No documents found to ingest.[/]")
        return

    console.print(f"\n[bold]Total:[/] {result['total_chunks']} chunks")
    if result["local_chunks"]:
        console.print(f"  Local chunks: {result['local_chunks']}")
    if result["notion_chunks"]:
        console.print(
            f"  Notion chunks: {result['notion_chunks']} (from {result['notion_pages']} pages)"
        )
    console.print(f"\n[green]Successfully ingested {result['ingested_chunks']} chunks into the vector store.[/]\n")


@main.command()
@click.argument("question")
@click.option("--show-sources", "-s", is_flag=True, help="Show source documents")
@click.option(
    "--mode", "-m",
    type=click.Choice(["hybrid", "vector", "keyword"], case_sensitive=False),
    default="hybrid",
    help="Search mode: 'hybrid' (default), 'vector' (semantic), 'keyword' (exact match)"
)
@click.option("--verbose", "-v", is_flag=True, help="Show retrieved chunks before answering")
@click.option(
    "--filter-title", "-t",
    default=None,
    help="Filter results by title/filename (case-insensitive substring match)"
)
def query(question: str, show_sources: bool, mode: str, verbose: bool, filter_title: str | None):
    """Ask a question about your documents."""
    status_data = get_status()
    doc_count = status_data["chunk_count"]

    if doc_count == 0:
        console.print("[yellow]No documents in vector store. Run 'rag ingest' first.[/]")
        return

    filter_info = f", title filter: '{filter_title}'" if filter_title else ""
    console.print(
        f"\n[dim]Querying {doc_count} chunks ({mode} search{filter_info}) "
        f"with {status_data['llm_model']}...[/]\n"
    )

    console.print("Retrieving relevant documents...")
    try:
        answer, sources = query_documents(question, search_mode=mode, title_filter=filter_title)
    except ValueError as e:
        console.print(f"[yellow]{e}[/]")
        return

    # Verbose mode: show what was retrieved
    if verbose and sources:
        console.print(f"\n[bold cyan]Retrieved {len(sources)} chunks:[/]")
        for i, doc in enumerate(sources, 1):
            source_name = sanitize_text(doc.metadata.get("source", "unknown"))
            preview = doc.page_content[:150].replace("\n", " ")
            preview = sanitize_text(preview) + "..."
            console.print(f"  [{i}] {source_name}: {preview}")
        console.print()
    
    console.print("Generating answer...")
    
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
    if clear_documents():
        console.print("[green]Vector store cleared successfully.[/]")
    else:
        console.print("[yellow]Vector store was already empty.[/]")


@main.command()
def status():
    """Show the current status of the RAG agent."""
    status_data = get_status()
    console.print("\n[bold]RAG Agent Status[/]\n")
    console.print(f"  Documents directory: {status_data['documents_directory']}")
    console.print(f"  Notion configured: {'Yes' if status_data['notion_configured'] else 'No'}")
    if status_data["notion_configured"]:
        console.print(f"  Notion database: {status_data['notion_database_id'][:8]}...")
    console.print(f"  Chunks in vector store: {status_data['chunk_count']}")
    console.print(f"  Embedding model: {status_data['embedding_model']}")
    console.print(f"  LLM model: {status_data['llm_model']}")
    console.print()


if __name__ == "__main__":
    main()
