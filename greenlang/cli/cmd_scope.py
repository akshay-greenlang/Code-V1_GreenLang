# -*- coding: utf-8 -*-
"""
CLI: Scope Engine (v3 L3 Intelligence)
========================================

Subcommands::

    gl scope compute     Run a Scope-Engine computation from a JSON request
    gl scope frameworks  List supported frameworks + adapters

Phase 3.2 of the FY27 plan.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from greenlang.scope_engine import ScopeEngineService
from greenlang.scope_engine.models import ComputationRequest, Framework

app = typer.Typer(
    help="Scope Engine operations (compute / frameworks)",
    no_args_is_help=True,
)
console = Console()


@app.command("compute")
def compute(
    input_file: str = typer.Argument(..., help="Path to a ComputationRequest JSON file"),
    output: Optional[str] = typer.Option(
        None, "--output", help="Write full ComputationResponse JSON to this path"
    ),
) -> None:
    """Run a scope computation."""
    p = Path(input_file)
    if not p.exists():
        console.print(f"[red]input not found: {p}[/red]")
        raise typer.Exit(2)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        console.print(f"[red]input must be valid JSON: {exc}[/red]")
        raise typer.Exit(2)

    request = ComputationRequest.model_validate(raw)
    response = ScopeEngineService().compute(request)
    comp = response.computation

    table = Table(title=f"Scope Computation {comp.computation_id[:8]}...")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("scope 1 CO2e (kg)", f"{float(comp.breakdown.scope_1_co2e_kg):,.2f}")
    table.add_row("scope 2 location (kg)", f"{float(comp.breakdown.scope_2_location_co2e_kg):,.2f}")
    table.add_row("scope 2 market (kg)", f"{float(comp.breakdown.scope_2_market_co2e_kg):,.2f}")
    table.add_row("scope 3 CO2e (kg)", f"{float(comp.breakdown.scope_3_co2e_kg):,.2f}")
    table.add_row("total CO2e (kg)", f"{float(comp.total_co2e_kg):,.2f}")
    console.print(table)

    fw_table = Table(title="Framework Views")
    fw_table.add_column("framework")
    fw_table.add_column("rows")
    for fw, view in response.framework_views.items():
        fw_table.add_row(fw.value, str(len(view.rows)))
    console.print(fw_table)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(response.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green][OK][/green] Full response -> {out_path}")


@app.command("frameworks")
def frameworks() -> None:
    """List supported Scope-Engine frameworks + their adapter modules."""
    table = Table(title="Scope Engine Frameworks")
    table.add_column("framework")
    table.add_column("adapter")
    mapping = {
        Framework.GHG_PROTOCOL: "greenlang.scope_engine.adapters.ghg_protocol",
        Framework.ISO_14064: "greenlang.scope_engine.adapters.iso_14064",
        Framework.SBTI: "greenlang.scope_engine.adapters.sbti",
        Framework.CSRD_E1: "greenlang.scope_engine.adapters.csrd_e1",
        Framework.CBAM: "greenlang.scope_engine.adapters.cbam",
    }
    for fw, module in mapping.items():
        table.add_row(fw.value, module)
    console.print(table)
