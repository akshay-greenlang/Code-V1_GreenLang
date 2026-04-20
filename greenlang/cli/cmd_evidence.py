# -*- coding: utf-8 -*-
"""
CLI: Evidence Vault (v3 L2 System of Record)
==============================================

Subcommands::

    gl evidence collect    Add an evidence record to a vault
    gl evidence attach     Attach a raw-source file to an evidence record
    gl evidence bundle     Write a signed ZIP bundle for a case
    gl evidence list       List evidence records (optionally filtered by case/type)
    gl evidence export     Export evidence records to JSON

Phase 2.2 of the FY27 plan.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from greenlang.evidence_vault import EvidenceVault

app = typer.Typer(
    help="Evidence Vault operations (collect / attach / bundle / list / export)",
    no_args_is_help=True,
)
console = Console()


def _make_vault(vault_id: str, sqlite: Optional[str]) -> EvidenceVault:
    if sqlite is not None:
        return EvidenceVault(vault_id=vault_id, storage="sqlite", sqlite_path=Path(sqlite))
    return EvidenceVault(vault_id=vault_id, storage="memory")


@app.command("collect")
def collect(
    evidence_type: str = typer.Argument(..., help="e.g. 'emission_factor', 'invoice', 'supplier_decl'"),
    source: str = typer.Argument(..., help="Originating agent or system"),
    data_json: str = typer.Argument(..., help="Evidence payload as inline JSON"),
    vault_id: str = typer.Option("default", "--vault-id", help="Vault identifier"),
    case_id: Optional[str] = typer.Option(None, "--case-id", help="Optional case grouping"),
    sqlite: Optional[str] = typer.Option(None, "--sqlite", help="SQLite vault file path"),
    metadata: Optional[str] = typer.Option(None, "--metadata", help="Extra JSON metadata"),
) -> None:
    """Collect a single evidence record and print its evidence_id."""
    try:
        data = json.loads(data_json)
    except json.JSONDecodeError as exc:
        console.print(f"[red]data_json must be valid JSON: {exc}[/red]")
        raise typer.Exit(2)

    meta = None
    if metadata:
        try:
            meta = json.loads(metadata)
        except json.JSONDecodeError as exc:
            console.print(f"[red]--metadata must be valid JSON: {exc}[/red]")
            raise typer.Exit(2)

    vault = _make_vault(vault_id, sqlite)
    try:
        eid = vault.collect(
            evidence_type=evidence_type,
            source=source,
            data=data,
            metadata=meta,
            case_id=case_id,
        )
    finally:
        vault.close()

    console.print(f"[green][OK][/green] Collected evidence_id=[bold]{eid}[/bold]")


@app.command("attach")
def attach(
    evidence_id: str = typer.Argument(..., help="Evidence record to attach to"),
    file_path: str = typer.Argument(..., help="Local file path to attach"),
    vault_id: str = typer.Option("default", "--vault-id", help="Vault identifier"),
    sqlite: Optional[str] = typer.Option(None, "--sqlite", help="SQLite vault file path"),
) -> None:
    """Attach a raw-source file to an existing evidence record."""
    src = Path(file_path)
    if not src.exists():
        console.print(f"[red]file not found: {src}[/red]")
        raise typer.Exit(2)
    payload = src.read_bytes()

    vault = _make_vault(vault_id, sqlite)
    try:
        content_hash = vault.attach(evidence_id, src.name, payload)
    finally:
        vault.close()

    console.print(f"[green][OK][/green] Attached {src.name} -> content_hash={content_hash}")


@app.command("bundle")
def bundle(
    output: str = typer.Argument(..., help="Output ZIP path"),
    case_id: Optional[str] = typer.Option(None, "--case-id", help="Case to bundle"),
    vault_id: str = typer.Option("default", "--vault-id", help="Vault identifier"),
    sqlite: Optional[str] = typer.Option(None, "--sqlite", help="SQLite vault file path"),
) -> None:
    """Write a signed ZIP bundle (manifest + records + attachments + signature)."""
    vault = _make_vault(vault_id, sqlite)
    try:
        out_path = vault.bundle(output_path=Path(output), case_id=case_id)
    finally:
        vault.close()
    console.print(f"[green][OK][/green] Bundle written -> {out_path}")


@app.command("list")
def list_records(
    vault_id: str = typer.Option("default", "--vault-id", help="Vault identifier"),
    case_id: Optional[str] = typer.Option(None, "--case-id", help="Filter by case"),
    evidence_type: Optional[str] = typer.Option(None, "--type", help="Filter by evidence_type"),
    sqlite: Optional[str] = typer.Option(None, "--sqlite", help="SQLite vault file path"),
) -> None:
    """List evidence records."""
    vault = _make_vault(vault_id, sqlite)
    try:
        records = vault.list_evidence(evidence_type=evidence_type, case_id=case_id)
    finally:
        vault.close()

    table = Table(title=f"Vault {vault_id}")
    table.add_column("evidence_id")
    table.add_column("case_id")
    table.add_column("type")
    table.add_column("source")
    table.add_column("collected_at")
    for r in records:
        table.add_row(
            r["evidence_id"][:18] + "...",
            str(r.get("case_id") or "-"),
            r["evidence_type"],
            r["source"],
            r["collected_at"],
        )
    console.print(table)
    console.print(f"[green]{len(records)} record(s)[/green]")


@app.command("export")
def export(
    output: str = typer.Argument(..., help="Output JSON file"),
    vault_id: str = typer.Option("default", "--vault-id", help="Vault identifier"),
    case_id: Optional[str] = typer.Option(None, "--case-id", help="Filter by case"),
    sqlite: Optional[str] = typer.Option(None, "--sqlite", help="SQLite vault file path"),
) -> None:
    """Export evidence records to JSON."""
    vault = _make_vault(vault_id, sqlite)
    try:
        records = vault.list_evidence(case_id=case_id)
        payload = vault.export(evidence_ids=[r["evidence_id"] for r in records])
    finally:
        vault.close()

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    console.print(f"[green][OK][/green] Exported {payload['record_count']} records -> {out}")
