# -*- coding: utf-8 -*-
"""v2 program commands for scale and productization gates."""

from __future__ import annotations

import typer
from rich.console import Console

from greenlang.v2.conformance import (
    agent_lifecycle_checks,
    connector_reliability_checks,
    V2_APP_PROFILE_DIRS,
    contract_checks,
    docs_contract_checks,
    release_gate_checks,
    runtime_convention_checks,
)
from greenlang.v2.profiles import V2_APP_PROFILES

app = typer.Typer()
console = Console()


def _print_results(title: str, checks) -> bool:
    console.print(f"[bold]{title}[/bold]")
    failed = False
    for check in checks:
        if check.ok:
            console.print(f"[green][OK][/green] {check.name}")
        else:
            failed = True
            console.print(f"[red][FAIL][/red] {check.name}")
            for detail in check.details:
                console.print(f"  - {detail}")
    return not failed


@app.command("status")
def status() -> None:
    """Show configured v2 app profile targets."""
    console.print("[bold]GreenLang v2 Targets[/bold]")
    for profile_dir in V2_APP_PROFILE_DIRS:
        console.print(f"- {profile_dir.as_posix()}")
    console.print("\n[bold]Runtime Profiles[/bold]")
    for key, profile in sorted(V2_APP_PROFILES.items()):
        console.print(f"- {key}: {profile.command_template}")


@app.command("validate-contracts")
def validate_contracts() -> None:
    checks = contract_checks()
    if _print_results("v2 contract checks", checks):
        raise typer.Exit(0)
    raise typer.Exit(1)


@app.command("runtime-checks")
def runtime_checks() -> None:
    checks = runtime_convention_checks()
    if _print_results("v2 runtime checks", checks):
        raise typer.Exit(0)
    raise typer.Exit(1)


@app.command("docs-check")
def docs_check() -> None:
    checks = docs_contract_checks()
    if _print_results("v2 docs checks", checks):
        raise typer.Exit(0)
    raise typer.Exit(1)


@app.command("agent-checks")
def agent_checks() -> None:
    checks = agent_lifecycle_checks()
    if _print_results("v2 agent lifecycle checks", checks):
        raise typer.Exit(0)
    raise typer.Exit(1)


@app.command("connector-checks")
def connector_checks() -> None:
    checks = connector_reliability_checks()
    if _print_results("v2 connector reliability checks", checks):
        raise typer.Exit(0)
    raise typer.Exit(1)


@app.command("gate")
def gate() -> None:
    checks = release_gate_checks()
    if _print_results("v2 release gates", checks):
        console.print("[bold green]v2 release gates passed[/bold green]")
        raise typer.Exit(0)
    console.print("[bold red]v2 release gates failed[/bold red]")
    raise typer.Exit(1)

