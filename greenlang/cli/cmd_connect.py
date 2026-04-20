# -*- coding: utf-8 -*-
"""
CLI: Connect (v3 L1 Data Foundation)
======================================

Subcommands::

    gl connect list       List all registered connectors
    gl connect test       Healthcheck a connector against provided credentials
    gl connect extract    Run a connector (with --dry-run for CI)

Phase 2.5 of the FY27 plan.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from greenlang.connect import (
    ConnectorError,
    SourceSpec,
    default_registry,
)

app = typer.Typer(
    help="Connect operations (list / test / extract)",
    no_args_is_help=True,
)
console = Console()


def _load_credentials(
    credentials_file: Optional[str],
    env_prefix: Optional[str],
) -> dict[str, str]:
    """Resolve credentials from a JSON file and/or ENV variables."""
    creds: dict[str, str] = {}
    if credentials_file:
        creds.update(
            json.loads(Path(credentials_file).read_text(encoding="utf-8"))
        )
    if env_prefix:
        prefix = env_prefix.upper()
        for key, value in os.environ.items():
            if key.startswith(prefix + "_"):
                creds.setdefault(key[len(prefix) + 1:].lower(), value)
    return creds


@app.command("list")
def list_connectors() -> None:
    """List all registered Connect connectors."""
    registry = default_registry()
    table = Table(title="Available Connectors")
    table.add_column("connector_id")
    table.add_column("required credentials")
    table.add_column("python package")
    for row in registry.describe():
        table.add_row(
            row["connector_id"],
            ", ".join(row["required_credentials"]) or "-",
            row["required_python_package"] or "-",
        )
    console.print(table)


@app.command("test")
def test(
    connector_id: str = typer.Argument(..., help="Connector id (see `gl connect list`)"),
    credentials_file: Optional[str] = typer.Option(
        None, "--credentials-file", help="JSON file of credential key:value pairs"
    ),
    env_prefix: Optional[str] = typer.Option(
        None, "--env-prefix", help="Read credentials from ENV vars with this prefix"
    ),
) -> None:
    """Healthcheck a connector against provided credentials."""
    registry = default_registry()
    try:
        connector = registry.get(connector_id)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    credentials = _load_credentials(credentials_file, env_prefix)
    result = asyncio.run(connector.healthcheck(credentials))

    ok_mark = "[green][OK][/green]" if result.ok else "[red][FAIL][/red]"
    console.print(f"{ok_mark} {connector_id}: {result.reason}")
    if result.missing_credentials:
        console.print(
            f"  missing credentials: {result.missing_credentials}"
        )
    console.print(
        f"  dependency available: {result.dependency_available}"
    )
    if not result.ok:
        raise typer.Exit(1)


@app.command("extract")
def extract(
    connector_id: str = typer.Argument(..., help="Connector id"),
    tenant_id: str = typer.Option("default", "--tenant-id", help="Tenant identifier"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Skip external calls; emits an empty result"
    ),
    credentials_file: Optional[str] = typer.Option(
        None, "--credentials-file", help="JSON file of credentials"
    ),
    env_prefix: Optional[str] = typer.Option(
        None, "--env-prefix", help="ENV-var prefix for credentials"
    ),
    filters_json: Optional[str] = typer.Option(
        None, "--filters", help="Inline JSON filters dict"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", help="Write full ConnectorResult JSON to this path"
    ),
) -> None:
    """Run a connector's extract() with optional dry-run and output."""
    registry = default_registry()
    try:
        connector = registry.get(connector_id)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    credentials = _load_credentials(credentials_file, env_prefix)
    filters = json.loads(filters_json) if filters_json else {}

    spec = SourceSpec(
        tenant_id=tenant_id,
        connector_id=connector_id,
        credentials=credentials,
        filters=filters,
        dry_run=dry_run,
    )

    try:
        result = asyncio.run(connector.extract(spec))
    except ConnectorError as exc:
        console.print(f"[red]{connector_id} failed: {exc}[/red]")
        raise typer.Exit(1)

    console.print(
        f"[green][OK][/green] {connector_id}: {result.row_count} row(s), "
        f"checksum={result.checksum[:16]}..."
    )
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "connector_id": result.connector_id,
                    "row_count": result.row_count,
                    "checksum": result.checksum,
                    "metadata": result.metadata,
                    "records": result.records,
                },
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
        console.print(f"[green][OK][/green] Full result -> {out_path}")
