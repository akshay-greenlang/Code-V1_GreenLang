"""
GL-FOUND-X-003: GreenLang Normalizer CLI - Vocabulary Commands

This module implements vocabulary management commands for listing,
searching, and displaying controlled vocabulary entries.

Example:
    >>> glnorm vocab list
    >>> glnorm vocab search "natural gas"
    >>> glnorm vocab show FUEL_NAT_GAS_001
"""

from typing import List, Optional
from enum import Enum

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

# Create Typer app for vocab commands
app = typer.Typer(
    name="vocab",
    help="Manage and search controlled vocabularies.",
    no_args_is_help=True,
)


class EntityType(str, Enum):
    """Entity types for vocabulary filtering."""

    FUEL = "fuel"
    MATERIAL = "material"
    PROCESS = "process"
    ALL = "all"


class OutputFormat(str, Enum):
    """Output formats for vocabulary results."""

    TABLE = "table"
    JSON = "json"
    YAML = "yaml"


@app.command(name="list")
def list_vocabularies(
    entity_type: EntityType = typer.Option(
        EntityType.ALL,
        "--type",
        "-t",
        help="Filter by entity type: fuel, material, process, or all.",
        case_sensitive=False,
    ),
    version: Optional[str] = typer.Option(
        None,
        "--version",
        "-V",
        help="Specific vocabulary version to list.",
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.TABLE,
        "--format",
        "-f",
        help="Output format: table, json, or yaml.",
        case_sensitive=False,
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        "-k",
        envvar="GLNORM_API_KEY",
        help="API key for remote vocabulary service.",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        "-l",
        help="Use local vocabulary data.",
    ),
    include_deprecated: bool = typer.Option(
        False,
        "--include-deprecated",
        "-d",
        help="Include deprecated entries.",
    ),
) -> None:
    """
    List available vocabularies and their entries.

    Shows controlled vocabulary entries for fuels, materials, and processes.
    Can filter by entity type and vocabulary version.

    [bold]Examples:[/bold]

        glnorm vocab list

        glnorm vocab list --type fuel

        glnorm vocab list --version 2026.01.0 --format json

        glnorm vocab list --include-deprecated
    """
    try:
        entries = _get_vocabulary_entries(
            entity_type=entity_type,
            version=version,
            use_api=not local,
            api_key=api_key,
            include_deprecated=include_deprecated,
        )

        if not entries:
            console.print("[yellow]No vocabulary entries found.[/yellow]")
            raise typer.Exit(code=0)

        if format == OutputFormat.JSON:
            _output_json(entries)
        elif format == OutputFormat.YAML:
            _output_yaml(entries)
        else:
            _output_list_table(entries, entity_type)

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(code=1)


@app.command(name="search")
def search_vocabularies(
    term: str = typer.Argument(
        ...,
        help="Search term to find in vocabulary entries.",
    ),
    entity_type: EntityType = typer.Option(
        EntityType.ALL,
        "--type",
        "-t",
        help="Filter by entity type: fuel, material, process, or all.",
        case_sensitive=False,
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-n",
        help="Maximum number of results to return.",
    ),
    min_score: float = typer.Option(
        0.5,
        "--min-score",
        help="Minimum similarity score (0.0-1.0) for fuzzy matches.",
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.TABLE,
        "--format",
        "-f",
        help="Output format: table, json, or yaml.",
        case_sensitive=False,
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        "-k",
        envvar="GLNORM_API_KEY",
        help="API key for remote vocabulary service.",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        "-l",
        help="Use local vocabulary data.",
    ),
) -> None:
    """
    Search vocabularies for matching entries.

    Performs fuzzy matching against canonical names and aliases
    to find relevant vocabulary entries.

    [bold]Examples:[/bold]

        glnorm vocab search "natural gas"

        glnorm vocab search diesel --type fuel

        glnorm vocab search "portland cement" --limit 5 --format json

        glnorm vocab search steel --min-score 0.8
    """
    try:
        results = _search_vocabulary(
            term=term,
            entity_type=entity_type,
            limit=limit,
            min_score=min_score,
            use_api=not local,
            api_key=api_key,
        )

        if not results:
            console.print(f"[yellow]No matches found for '{term}'.[/yellow]")
            raise typer.Exit(code=0)

        if format == OutputFormat.JSON:
            _output_json(results)
        elif format == OutputFormat.YAML:
            _output_yaml(results)
        else:
            _output_search_table(results, term)

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(code=1)


@app.command(name="show")
def show_vocabulary(
    vocab_id: str = typer.Argument(
        ...,
        help="Vocabulary entry ID to display (e.g., FUEL_NAT_GAS_001).",
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.TABLE,
        "--format",
        "-f",
        help="Output format: table, json, or yaml.",
        case_sensitive=False,
    ),
    version: Optional[str] = typer.Option(
        None,
        "--version",
        "-V",
        help="Specific vocabulary version.",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        "-k",
        envvar="GLNORM_API_KEY",
        help="API key for remote vocabulary service.",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        "-l",
        help="Use local vocabulary data.",
    ),
) -> None:
    """
    Display detailed information about a vocabulary entry.

    Shows all metadata, aliases, and attributes for a specific
    vocabulary entry identified by its ID.

    [bold]Examples:[/bold]

        glnorm vocab show FUEL_NAT_GAS_001

        glnorm vocab show MAT_STEEL_001 --format json

        glnorm vocab show GL-FUEL-DIESEL --version 2026.01.0
    """
    try:
        entry = _get_vocabulary_entry(
            vocab_id=vocab_id,
            version=version,
            use_api=not local,
            api_key=api_key,
        )

        if not entry:
            console.print(f"[red]Error:[/red] Vocabulary entry '{vocab_id}' not found.")
            raise typer.Exit(code=1)

        if format == OutputFormat.JSON:
            _output_json(entry)
        elif format == OutputFormat.YAML:
            _output_yaml(entry)
        else:
            _output_entry_details(entry)

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(code=1)


@app.command(name="versions")
def list_versions(
    format: OutputFormat = typer.Option(
        OutputFormat.TABLE,
        "--format",
        "-f",
        help="Output format: table, json, or yaml.",
        case_sensitive=False,
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        "-k",
        envvar="GLNORM_API_KEY",
        help="API key for remote vocabulary service.",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        "-l",
        help="Use local vocabulary data.",
    ),
) -> None:
    """
    List available vocabulary versions.

    Shows all published vocabulary versions with their release dates
    and entry counts.

    [bold]Example:[/bold]

        glnorm vocab versions
    """
    try:
        versions = _get_vocabulary_versions(
            use_api=not local,
            api_key=api_key,
        )

        if not versions:
            console.print("[yellow]No vocabulary versions found.[/yellow]")
            raise typer.Exit(code=0)

        if format == OutputFormat.JSON:
            _output_json(versions)
        elif format == OutputFormat.YAML:
            _output_yaml(versions)
        else:
            _output_versions_table(versions)

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(code=1)


def _get_vocabulary_entries(
    entity_type: EntityType,
    version: Optional[str],
    use_api: bool,
    api_key: Optional[str],
    include_deprecated: bool,
) -> List[dict]:
    """Get vocabulary entries from local or API source."""
    try:
        from gl_normalizer_core import ReferenceResolver

        resolver = ReferenceResolver(load_defaults=True)
        entries = []

        vocabularies = resolver.list_vocabularies()
        for vocab_name in vocabularies:
            if entity_type != EntityType.ALL and entity_type.value != vocab_name.rstrip("s"):
                continue

            vocab_entries = resolver.get_vocabulary_entries(vocab_name)
            for entry in vocab_entries:
                entry_dict = {
                    "id": entry.id,
                    "name": entry.name,
                    "entity_type": vocab_name.rstrip("s"),
                    "aliases": entry.aliases,
                    "metadata": entry.metadata,
                    "status": "active",
                }

                if include_deprecated or entry_dict.get("status") == "active":
                    entries.append(entry_dict)

        return entries

    except ImportError:
        console.print(
            "[yellow]Warning:[/yellow] gl-normalizer-core not installed. "
            "Using mock data."
        )
        return _get_mock_entries(entity_type, include_deprecated)


def _search_vocabulary(
    term: str,
    entity_type: EntityType,
    limit: int,
    min_score: float,
    use_api: bool,
    api_key: Optional[str],
) -> List[dict]:
    """Search vocabulary entries with fuzzy matching."""
    try:
        from gl_normalizer_core import ReferenceResolver

        resolver = ReferenceResolver(load_defaults=True, min_confidence=min_score * 100)
        results = []

        vocabularies = resolver.list_vocabularies()
        for vocab_name in vocabularies:
            if entity_type != EntityType.ALL and entity_type.value != vocab_name.rstrip("s"):
                continue

            try:
                resolution = resolver.resolve(term, vocab_name, confidence_threshold=min_score * 100)

                if resolution.resolved:
                    results.append({
                        "id": resolution.resolved.resolved_id,
                        "name": resolution.resolved.resolved_name,
                        "entity_type": vocab_name.rstrip("s"),
                        "score": resolution.resolved.confidence_score / 100,
                        "match_method": resolution.resolved.match_method.value,
                        "vocabulary": vocab_name,
                    })

                # Add candidates as additional results
                for candidate in resolution.candidates[:limit]:
                    if candidate["id"] not in [r["id"] for r in results]:
                        results.append({
                            "id": candidate["id"],
                            "name": candidate["name"],
                            "entity_type": vocab_name.rstrip("s"),
                            "score": candidate["score"] / 100,
                            "match_method": "fuzzy",
                            "vocabulary": vocab_name,
                        })

            except Exception:
                continue

        # Sort by score and limit
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:limit]

    except ImportError:
        console.print(
            "[yellow]Warning:[/yellow] gl-normalizer-core not installed. "
            "Using mock search."
        )
        return _mock_search(term, entity_type, limit, min_score)


def _get_vocabulary_entry(
    vocab_id: str,
    version: Optional[str],
    use_api: bool,
    api_key: Optional[str],
) -> Optional[dict]:
    """Get a specific vocabulary entry by ID."""
    try:
        from gl_normalizer_core import ReferenceResolver

        resolver = ReferenceResolver(load_defaults=True)

        for vocab_name in resolver.list_vocabularies():
            entries = resolver.get_vocabulary_entries(vocab_name)
            for entry in entries:
                if entry.id == vocab_id:
                    return {
                        "id": entry.id,
                        "name": entry.name,
                        "entity_type": vocab_name.rstrip("s"),
                        "aliases": entry.aliases,
                        "metadata": entry.metadata,
                        "status": "active",
                        "vocabulary_version": version or "default",
                    }

        return None

    except ImportError:
        console.print(
            "[yellow]Warning:[/yellow] gl-normalizer-core not installed. "
            "Using mock data."
        )
        return _get_mock_entry(vocab_id)


def _get_vocabulary_versions(
    use_api: bool,
    api_key: Optional[str],
) -> List[dict]:
    """Get list of available vocabulary versions."""
    # Mock versions for now - would come from API or version registry
    return [
        {
            "version": "2026.01.0",
            "released_at": "2026-01-15T00:00:00Z",
            "entry_count": 156,
            "status": "current",
        },
        {
            "version": "2025.12.0",
            "released_at": "2025-12-01T00:00:00Z",
            "entry_count": 142,
            "status": "supported",
        },
        {
            "version": "2025.06.0",
            "released_at": "2025-06-01T00:00:00Z",
            "entry_count": 128,
            "status": "deprecated",
        },
    ]


def _get_mock_entries(entity_type: EntityType, include_deprecated: bool) -> List[dict]:
    """Return mock vocabulary entries for demo purposes."""
    entries = [
        {
            "id": "FUEL_NAT_GAS_001",
            "name": "Natural Gas",
            "entity_type": "fuel",
            "aliases": ["nat gas", "methane", "ng"],
            "metadata": {"category": "fossil", "ghg_factor": 2.02},
            "status": "active",
        },
        {
            "id": "FUEL_DIESEL_001",
            "name": "Diesel",
            "entity_type": "fuel",
            "aliases": ["diesel fuel", "derv", "agr diesel"],
            "metadata": {"category": "fossil", "ghg_factor": 2.68},
            "status": "active",
        },
        {
            "id": "MAT_STEEL_001",
            "name": "Steel",
            "entity_type": "material",
            "aliases": ["carbon steel", "mild steel"],
            "metadata": {"category": "metal", "density": 7850},
            "status": "active",
        },
        {
            "id": "MAT_CONC_001",
            "name": "Concrete",
            "entity_type": "material",
            "aliases": ["cement", "portland cement"],
            "metadata": {"category": "construction", "density": 2400},
            "status": "active",
        },
    ]

    if entity_type != EntityType.ALL:
        entries = [e for e in entries if e["entity_type"] == entity_type.value]

    return entries


def _mock_search(term: str, entity_type: EntityType, limit: int, min_score: float) -> List[dict]:
    """Mock search functionality."""
    entries = _get_mock_entries(entity_type, False)
    term_lower = term.lower()

    results = []
    for entry in entries:
        # Simple substring matching for mock
        if term_lower in entry["name"].lower():
            results.append({**entry, "score": 0.95, "match_method": "name"})
        elif any(term_lower in alias.lower() for alias in entry["aliases"]):
            results.append({**entry, "score": 0.85, "match_method": "alias"})

    return sorted(results, key=lambda x: x["score"], reverse=True)[:limit]


def _get_mock_entry(vocab_id: str) -> Optional[dict]:
    """Get mock entry by ID."""
    entries = _get_mock_entries(EntityType.ALL, True)
    for entry in entries:
        if entry["id"] == vocab_id:
            return entry
    return None


def _output_json(data) -> None:
    """Output data as JSON."""
    import json

    console.print(json.dumps(data, indent=2))


def _output_yaml(data) -> None:
    """Output data as YAML."""
    import yaml

    console.print(yaml.dump(data, default_flow_style=False, sort_keys=False))


def _output_list_table(entries: List[dict], entity_type: EntityType) -> None:
    """Output vocabulary list as formatted table."""
    title = f"Vocabulary Entries"
    if entity_type != EntityType.ALL:
        title += f" ({entity_type.value}s)"

    table = Table(
        title=title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("ID", style="dim")
    table.add_column("Name", style="green")
    table.add_column("Type")
    table.add_column("Aliases", style="dim")
    table.add_column("Status")

    for entry in entries:
        aliases = ", ".join(entry.get("aliases", [])[:3])
        if len(entry.get("aliases", [])) > 3:
            aliases += "..."

        status_style = "green" if entry.get("status") == "active" else "yellow"
        status = f"[{status_style}]{entry.get('status', 'active')}[/{status_style}]"

        table.add_row(
            entry["id"],
            entry["name"],
            entry.get("entity_type", ""),
            aliases,
            status,
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(entries)} entries[/dim]")


def _output_search_table(results: List[dict], term: str) -> None:
    """Output search results as formatted table."""
    table = Table(
        title=f"Search Results for '{term}'",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("ID", style="dim")
    table.add_column("Name", style="green")
    table.add_column("Type")
    table.add_column("Score", justify="right")
    table.add_column("Match", style="dim")

    for result in results:
        score = result.get("score", 0)
        score_style = "green" if score >= 0.9 else "yellow" if score >= 0.7 else "red"

        table.add_row(
            result["id"],
            result["name"],
            result.get("entity_type", ""),
            f"[{score_style}]{score:.0%}[/{score_style}]",
            result.get("match_method", ""),
        )

    console.print(table)


def _output_entry_details(entry: dict) -> None:
    """Output detailed vocabulary entry information."""
    panel_content = []

    panel_content.append(f"[bold]ID:[/bold] {entry['id']}")
    panel_content.append(f"[bold]Name:[/bold] [green]{entry['name']}[/green]")
    panel_content.append(f"[bold]Type:[/bold] {entry.get('entity_type', 'unknown')}")
    panel_content.append(f"[bold]Status:[/bold] {entry.get('status', 'active')}")

    if entry.get("vocabulary_version"):
        panel_content.append(f"[bold]Version:[/bold] {entry['vocabulary_version']}")

    panel_content.append("")
    panel_content.append("[bold]Aliases:[/bold]")
    for alias in entry.get("aliases", []):
        panel_content.append(f"  - {alias}")

    if entry.get("metadata"):
        panel_content.append("")
        panel_content.append("[bold]Metadata:[/bold]")
        for key, value in entry["metadata"].items():
            panel_content.append(f"  {key}: {value}")

    panel = Panel(
        "\n".join(panel_content),
        title=f"[bold cyan]Vocabulary Entry[/bold cyan]",
        border_style="cyan",
    )
    console.print(panel)


def _output_versions_table(versions: List[dict]) -> None:
    """Output vocabulary versions as formatted table."""
    table = Table(
        title="Vocabulary Versions",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Version", style="green")
    table.add_column("Released", style="dim")
    table.add_column("Entries", justify="right")
    table.add_column("Status")

    for version in versions:
        status = version.get("status", "unknown")
        status_style = (
            "green" if status == "current"
            else "yellow" if status == "supported"
            else "red"
        )

        released = version.get("released_at", "")[:10]

        table.add_row(
            version["version"],
            released,
            str(version.get("entry_count", "")),
            f"[{status_style}]{status}[/{status_style}]",
        )

    console.print(table)
