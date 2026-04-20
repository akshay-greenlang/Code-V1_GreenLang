# -*- coding: utf-8 -*-
"""
CLI: Climate Ledger (v3 L2 System of Record)
=============================================

Subcommands::

    gl ledger record     Record an entry into the ledger
    gl ledger verify     Verify an entity's provenance chain
    gl ledger export     Export an entity chain or the full global chain

Phase 2.1 of the FY27 plan.  See docs/migration/AGENT_BASE_CONSOLIDATION.md
for canonical-base usage by the calling agents and
docs/REPO_TOUR.md for the L2 placement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from greenlang.climate_ledger import ClimateLedger

app = typer.Typer(
    help="Climate Ledger operations (record / verify / export)",
    no_args_is_help=True,
)
console = Console()


def _make_ledger(
    agent: str,
    sqlite: Optional[str],
) -> ClimateLedger:
    """Construct a ClimateLedger with the requested backend."""
    if sqlite is not None:
        return ClimateLedger(
            agent_name=agent,
            storage_backend="sqlite",
            sqlite_path=Path(sqlite),
        )
    return ClimateLedger(agent_name=agent, storage_backend="memory")


@app.command("record")
def record(
    entity_type: str = typer.Argument(..., help="Entity category (e.g. 'emission', 'facility')"),
    entity_id: str = typer.Argument(..., help="Unique identifier for the entity"),
    operation: str = typer.Argument(..., help="Operation (e.g. 'calculate', 'ingest')"),
    content_hash: str = typer.Argument(..., help="SHA-256 hex of the payload being recorded"),
    agent: str = typer.Option("cli", "--agent", help="Agent name owning this record"),
    sqlite: Optional[str] = typer.Option(
        None,
        "--sqlite",
        help="Path to an append-only SQLite ledger file. If omitted, an in-memory ledger is used (records are NOT persisted).",
    ),
    metadata: Optional[str] = typer.Option(
        None, "--metadata", help="Optional JSON object of extra context"
    ),
) -> None:
    """Record a single provenance entry and print the resulting chain hash."""
    meta: Optional[dict] = None
    if metadata is not None:
        try:
            meta = json.loads(metadata)
        except json.JSONDecodeError as exc:
            console.print(f"[red]--metadata must be valid JSON: {exc}[/red]")
            raise typer.Exit(2)

    ledger = _make_ledger(agent, sqlite)
    try:
        chain_hash = ledger.record_entry(
            entity_type=entity_type,
            entity_id=entity_id,
            operation=operation,
            content_hash=content_hash,
            metadata=meta,
        )
    finally:
        ledger.close()

    console.print(f"[green][OK][/green] Recorded entry for {entity_type}/{entity_id}")
    console.print(f"chain_hash: [bold]{chain_hash}[/bold]")


@app.command("verify")
def verify(
    entity_id: str = typer.Argument(..., help="Entity whose chain will be verified"),
    agent: str = typer.Option("cli", "--agent", help="Agent name that owns the ledger"),
    sqlite: Optional[str] = typer.Option(
        None, "--sqlite", help="Path to the SQLite ledger file"
    ),
) -> None:
    """Verify that an entity's provenance chain is intact.

    Note: a fresh CLI invocation reconstructs the in-memory tracker from
    the SQLite backend; use the same --sqlite path that was used to
    record the entries.
    """
    if sqlite is None:
        console.print(
            "[yellow]No --sqlite path provided; verifying against an empty "
            "in-memory ledger will always show 0 entries.[/yellow]"
        )

    ledger = _make_ledger(agent, sqlite)
    try:
        valid, chain = ledger.verify(entity_id)
    finally:
        ledger.close()

    table = Table(title=f"Chain for entity {entity_id}")
    table.add_column("#", justify="right")
    table.add_column("operation")
    table.add_column("content_hash")
    table.add_column("chain_hash")
    for i, entry in enumerate(chain, start=1):
        table.add_row(
            str(i),
            str(entry.get("operation") or entry.get("action", "?")),
            str(entry.get("content_hash") or entry.get("data_hash", ""))[:20] + "...",
            str(entry.get("chain_hash") or entry.get("hash", ""))[:20] + "...",
        )
    console.print(table)

    if valid:
        console.print(f"[green][OK][/green] Chain valid ({len(chain)} entries)")
    else:
        console.print(f"[red][FAIL][/red] Chain verification FAILED for {entity_id}")
        raise typer.Exit(1)


@app.command("export")
def export(
    output: str = typer.Argument(..., help="Output JSON file path"),
    entity_id: Optional[str] = typer.Option(
        None, "--entity-id", help="Restrict export to a single entity"
    ),
    agent: str = typer.Option("cli", "--agent", help="Agent name that owns the ledger"),
    sqlite: Optional[str] = typer.Option(
        None, "--sqlite", help="Path to the SQLite ledger file"
    ),
) -> None:
    """Export the global chain or a single entity's chain to JSON."""
    ledger = _make_ledger(agent, sqlite)
    try:
        data = ledger.export(entity_id=entity_id)
    finally:
        ledger.close()

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    n = len(data) if isinstance(data, list) else data.get("entry_count", "?")
    console.print(f"[green][OK][/green] Exported {n} entries -> {out_path}")
