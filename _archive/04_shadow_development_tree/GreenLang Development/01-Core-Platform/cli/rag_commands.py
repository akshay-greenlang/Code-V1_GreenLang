# -*- coding: utf-8 -*-
"""
CLI commands for GreenLang RAG system.

Commands:
- gl rag up: Start Weaviate container
- gl rag ingest: Ingest documents into collection
- gl rag query: Query RAG system
"""

import typer
import subprocess
import asyncio
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from datetime import date

from greenlang.intelligence.rag.config import get_config, RAGConfig
from greenlang.intelligence.rag.models import DocMeta
from greenlang.intelligence.rag.hashing import file_hash

console = Console()
app = typer.Typer(
    name="rag",
    help="RAG (Retrieval-Augmented Generation) commands",
)


@app.command()
def up(
    detach: bool = typer.Option(
        False, "--detach", "-d", help="Run in background"
    ),
):
    """Start Weaviate container via Docker Compose."""
    console.print("[cyan]Starting Weaviate container...[/cyan]")

    # Find docker-compose file
    project_root = Path(__file__).parent.parent.parent
    compose_file = project_root / "docker" / "weaviate" / "docker-compose.yml"

    if not compose_file.exists():
        console.print(
            f"[red]Error: docker-compose.yml not found at {compose_file}[/red]"
        )
        console.print("[yellow]Expected location: docker/weaviate/docker-compose.yml[/yellow]")
        raise typer.Exit(1)

    # Build docker-compose command
    cmd = ["docker-compose", "-f", str(compose_file), "up"]
    if detach:
        cmd.append("-d")

    try:
        result = subprocess.run(cmd, check=True)
        if detach:
            console.print("[green][OK][/green] Weaviate started in background")
            console.print("[blue][INFO][/blue] Connect at: http://localhost:8080")
        else:
            console.print("[green][OK][/green] Weaviate stopped")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error starting Weaviate: {e}[/red]")
        raise typer.Exit(1)
    except FileNotFoundError:
        console.print("[red]Error: docker-compose not found in PATH[/red]")
        console.print("[yellow]Install Docker Compose: https://docs.docker.com/compose/install/[/yellow]")
        raise typer.Exit(1)


@app.command()
def down():
    """Stop Weaviate container."""
    console.print("[cyan]Stopping Weaviate container...[/cyan]")

    # Find docker-compose file
    project_root = Path(__file__).parent.parent.parent
    compose_file = project_root / "docker" / "weaviate" / "docker-compose.yml"

    if not compose_file.exists():
        console.print(f"[red]Error: docker-compose.yml not found[/red]")
        raise typer.Exit(1)

    cmd = ["docker-compose", "-f", str(compose_file), "down"]

    try:
        subprocess.run(cmd, check=True)
        console.print("[green][OK][/green] Weaviate stopped")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error stopping Weaviate: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def ingest(
    collection: str = typer.Option(..., "--collection", "-c", help="Collection name"),
    file_path: Path = typer.Option(..., "--file", "-f", help="File to ingest"),
    title: Optional[str] = typer.Option(None, "--title", help="Document title"),
    publisher: Optional[str] = typer.Option(None, "--publisher", help="Publisher name"),
    version: Optional[str] = typer.Option(None, "--version", help="Document version"),
    year: Optional[int] = typer.Option(None, "--year", help="Publication year"),
    uri: Optional[str] = typer.Option(None, "--uri", help="Source URI"),
):
    """Ingest document into collection."""
    console.print(f"[cyan]Ingesting {file_path} into collection '{collection}'...[/cyan]")

    # Validate file exists
    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)

    # Create DocMeta
    content_hash = file_hash(str(file_path))
    doc_meta = DocMeta(
        doc_id=content_hash[:32],
        title=title or file_path.stem,
        collection=collection,
        source_uri=uri or str(file_path),
        publisher=publisher or "Unknown",
        publication_date=date(year, 1, 1) if year else None,
        version=version or "1.0",
        content_hash=content_hash,
        doc_hash=content_hash,
    )

    # Run ingestion
    try:
        from greenlang.intelligence.rag import ingest as rag_ingest

        async def run_ingest():
            manifest = await rag_ingest.ingest_path(
                path=file_path,
                collection=collection,
                doc_meta=doc_meta,
            )
            return manifest

        manifest = asyncio.run(run_ingest())

        # Display results
        console.print(f"[green][OK][/green] Ingestion completed")
        console.print(f"   - Document: {doc_meta.title}")
        console.print(f"   - Collection: {collection}")
        console.print(f"   - Embeddings: {manifest.total_embeddings}")
        console.print(f"   - Duration: {manifest.ingestion_duration_seconds:.2f}s")

        # Write manifest
        manifest_path = file_path.parent / f"{file_path.stem}_MANIFEST.json"
        rag_ingest.write_manifest(manifest, manifest_path)
        console.print(f"   - Manifest: {manifest_path}")

    except ImportError as e:
        console.print(f"[red]Error: RAG ingest module not available: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error during ingestion: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def query_cmd(
    query_text: str = typer.Option(..., "--query", "-q", help="Query string"),
    collections: List[str] = typer.Option(
        None, "--collection", "-c", help="Collections to search (can be repeated)"
    ),
    top_k: int = typer.Option(6, "--top-k", "-k", help="Number of results"),
    fetch_k: int = typer.Option(30, "--fetch-k", help="Number of MMR candidates"),
    mmr_lambda: float = typer.Option(0.5, "--lambda", help="MMR lambda (0=diversity, 1=relevance)"),
):
    """Query RAG system."""
    console.print(f"[cyan]Querying: {query_text}[/cyan]")
    console.print(f"   - Collections: {', '.join(collections) if collections else 'all'}")
    console.print(f"   - Top-k: {top_k}")
    console.print(f"   - Fetch-k: {fetch_k}")
    console.print(f"   - MMR lambda: {mmr_lambda}")
    console.print()

    try:
        from greenlang.intelligence.rag import query as rag_query

        async def run_query():
            result = await rag_query.query(
                q=query_text,
                top_k=top_k,
                collections=collections,
                fetch_k=fetch_k,
                mmr_lambda=mmr_lambda,
            )
            return result

        result = asyncio.run(run_query())

        # Display results
        console.print(f"[green][OK][/green] Query completed")
        console.print(f"   - Retrieved: {result.total_chunks} chunks")
        console.print(f"   - Search time: {result.search_time_ms}ms")
        console.print(f"   - Total tokens: {result.total_tokens}")
        console.print()

        # Display results in table
        table = Table(title="Query Results")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Score", style="green", width=6)
        table.add_column("Document", style="yellow")
        table.add_column("Section", style="blue")
        table.add_column("Text Preview", style="white")

        for i, (chunk, score, citation) in enumerate(
            zip(result.chunks, result.relevance_scores, result.citations), 1
        ):
            text_preview = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
            table.add_row(
                str(i),
                f"{score:.3f}",
                citation.doc_title[:30],
                chunk.section_path[:40],
                text_preview,
            )

        console.print(table)
        console.print()

        # Display citations
        console.print("[bold]Citations:[/bold]")
        for i, citation in enumerate(result.citations, 1):
            console.print(f"[{i}] {citation.formatted}")

    except ImportError as e:
        console.print(f"[red]Error: RAG query module not available: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error during query: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def stats(
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Collection name"),
):
    """Show RAG system statistics."""
    console.print("[cyan]RAG System Statistics[/cyan]")
    console.print()

    try:
        from greenlang.intelligence.rag import RAGEngine

        config = get_config()
        engine = RAGEngine(config)

        if collection:
            # Show stats for specific collection
            stats = engine.get_collection_stats(collection)
            console.print(f"Collection: [bold]{collection}[/bold]")
            console.print(f"   - Documents: {stats.get('num_documents', 0)}")
            console.print(f"   - Chunks: {stats.get('num_chunks', 0)}")
            console.print(f"   - Total tokens: {stats.get('total_tokens', 0)}")
            console.print(f"   - Embedding dimension: {stats.get('embedding_dimension', 0)}")
        else:
            # Show stats for all collections
            collections = engine.list_collections()
            console.print(f"Total collections: {len(collections)}")
            console.print()

            table = Table(title="Collections")
            table.add_column("Collection", style="cyan")
            table.add_column("Status", style="green")

            for coll in collections:
                table.add_row(coll, "Configured")

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
