# -*- coding: utf-8 -*-
"""
CLI: Comply (v3 L4 Compliance Cloud umbrella)
===============================================

Subcommands::

    gl comply run         Run a Comply request end-to-end
    gl comply applies-to  Shortcut for Policy Graph applicability only

Phase 3.1 of the FY27 plan.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from greenlang.comply import ComplianceRunRequest, ComplyOrchestrator

app = typer.Typer(
    help="Comply operations (run / applies-to)",
    no_args_is_help=True,
)
console = Console()


@app.command("run")
def run(
    request_file: str = typer.Argument(
        ..., help="Path to a Comply request JSON (ComplianceRunRequest shape)"
    ),
    bundle_output: Optional[str] = typer.Option(
        None, "--bundle", help="Write a signed Evidence Vault ZIP to this path"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", help="Write full ComplianceRunResult JSON to this path"
    ),
) -> None:
    """Execute a Comply run against a JSON request file."""
    req_path = Path(request_file)
    if not req_path.exists():
        console.print(f"[red]request file not found: {req_path}[/red]")
        raise typer.Exit(2)

    try:
        payload = json.loads(req_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        console.print(f"[red]request must be valid JSON: {exc}[/red]")
        raise typer.Exit(2)

    request = ComplianceRunRequest(**payload)
    orchestrator = ComplyOrchestrator()
    result = orchestrator.run(request, bundle_output=bundle_output)

    # Summary table
    table = Table(title=f"Comply Run — case {result.case_id}")
    table.add_column("regulation")
    table.add_column("jurisdiction")
    table.add_column("deadline")
    table.add_column("total CO2e (kg)")
    table.add_column("evidence")
    for fr in result.framework_results:
        total = (
            f"{fr.total_co2e_kg:,.1f}" if fr.total_co2e_kg is not None else "-"
        )
        table.add_row(
            fr.regulation,
            fr.jurisdiction,
            fr.deadline or "rolling",
            total,
            str(fr.evidence_count),
        )
    console.print(table)

    console.print(
        f"[green][OK][/green] Case {result.case_id}: "
        f"{len(result.applicable_regulations)} regulation(s) applied"
    )
    if result.evidence_bundle_path:
        console.print(f"  evidence bundle: {result.evidence_bundle_path}")
    if result.ledger_global_chain_head:
        console.print(f"  ledger chain-head: {result.ledger_global_chain_head}")

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result.model_dump(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        console.print(f"[green][OK][/green] Full result written -> {out_path}")


@app.command("applies-to")
def applies_to(
    entity_json: str = typer.Argument(..., help="Entity attributes as inline JSON"),
    jurisdiction: str = typer.Argument(..., help="Jurisdiction (EU, US-CA, GB, GLOBAL, …)"),
    date: str = typer.Argument(..., help="Reporting date YYYY-MM-DD"),
) -> None:
    """Shortcut: Policy Graph applicability only (no Scope Engine / Vault / Ledger)."""
    from greenlang.policy_graph import PolicyGraph

    entity = json.loads(entity_json)
    pg = PolicyGraph()
    verdict = pg.applies_to(
        entity=entity,
        activity={"category": "comply_run"},
        jurisdiction=jurisdiction,
        date=date,
    )
    table = Table(title=f"Applicable on {date}")
    table.add_column("regulation")
    table.add_column("jurisdiction")
    table.add_column("deadline")
    for reg in verdict.applicable_regulations:
        table.add_row(reg.name, reg.jurisdiction, reg.deadline or "rolling")
    console.print(table)
