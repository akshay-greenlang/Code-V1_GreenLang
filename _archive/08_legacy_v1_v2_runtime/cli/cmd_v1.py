# -*- coding: utf-8 -*-
"""v1 program commands for platformization gates."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

from greenlang.v1.conformance import (
    V1_APP_PROFILE_DIRS,
    contract_checks,
    profile_full_backend_checks,
    profile_smoke_checks,
    release_gate_checks,
    signed_pack_enforcement_checks,
)
from greenlang.v1.profiles import V1_APP_PROFILES, get_profile
from greenlang.v1.runtime import generate_profile_smoke_artifacts

app = typer.Typer()
console = Console()


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "on"}


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
    """Show configured v1 app profile targets."""
    console.print("[bold]GreenLang v1 Targets[/bold]")
    for profile_dir in V1_APP_PROFILE_DIRS:
        console.print(f"- {profile_dir.as_posix()}")
    console.print("\n[bold]Runtime Profiles[/bold]")
    for key, profile in sorted(V1_APP_PROFILES.items()):
        console.print(f"- {key}: {profile.command_template}")


@app.command("validate-contracts")
def validate_contracts() -> None:
    """Validate v1 pack.yaml and gl.yaml contracts across the first app set."""
    checks = contract_checks()
    if _print_results("v1 contract checks", checks):
        raise typer.Exit(0)
    raise typer.Exit(1)


@app.command("check-policy")
def check_policy() -> None:
    """Enforce signed-pack baseline policy across v1 app profiles."""
    checks = signed_pack_enforcement_checks()
    if _print_results("v1 signed-pack checks", checks):
        raise typer.Exit(0)
    raise typer.Exit(1)


@app.command("gate")
def gate() -> None:
    """Run full v1 release gate checks."""
    checks = release_gate_checks()
    if _print_results("v1 release gates", checks):
        console.print("[bold green]v1 release gates passed[/bold green]")
        raise typer.Exit(0)
    console.print("[bold red]v1 release gates failed[/bold red]")
    raise typer.Exit(1)


@app.command("full-backend-checks")
def full_backend_checks() -> None:
    """Run full backend checks for CSRD/VCCI plus CBAM runtime anchor."""
    checks = profile_full_backend_checks()
    if _print_results("v1 full backend checks", checks):
        raise typer.Exit(0)
    raise typer.Exit(1)


@app.command("run-profile")
def run_profile(
    profile: str = typer.Argument(..., help="Profile key: cbam | csrd | vcci"),
    config_or_input: str = typer.Argument(..., help="Config/input file path"),
    imports: str = typer.Argument("", help="CBAM imports file path (use '-' for non-cbam)"),
    output_dir: str = typer.Argument("out", help="Output directory"),
    smoke: str = typer.Argument("false", help="true|false smoke artifact generation"),
) -> None:
    """
    Execute a v1 profile with unified entry semantics.

    For `cbam`, command expands to:
      gl run cbam <config.yaml> <imports.csv> <output_dir>

    For `csrd`/`vcci`, command expands to:
      gl run <profile> <input.csv|json> <output_dir>
    """
    selected = get_profile(profile)
    if _coerce_bool(smoke):
        artifacts = generate_profile_smoke_artifacts(
            profile=selected,
            input_path=Path(config_or_input),
            output_dir=Path(output_dir),
        )
        console.print(
            f"[green][OK][/green] profile {selected.key} smoke artifacts generated: "
            f"{', '.join(artifacts)}"
        )
        raise typer.Exit(0)

    if selected.key == "cbam":
        if imports in {"", "-"}:
            console.print("[red]CBAM profile requires imports argument[/red]")
            raise typer.Exit(2)
        cmd = [
            sys.executable,
            "-m",
            "greenlang.cli.main",
            "run",
            "cbam",
            config_or_input,
            imports,
            output_dir,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "greenlang.cli.main",
            "run",
            selected.key,
            config_or_input,
            output_dir,
        ]

    console.print(f"[cyan]Running profile {selected.key}...[/cyan]")
    result = subprocess.run(cmd, check=False)
    raise typer.Exit(result.returncode)


@app.command("smoke")
def smoke(output_root: str = typer.Option("tmp_v1_smoke", "--out", help="Output root directory")) -> None:
    """
    Run minimal smoke checks across all v1 profiles.
    """
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    checks = contract_checks()
    if not _print_results("v1 smoke contract checks", checks):
        raise typer.Exit(1)
    policy_checks = signed_pack_enforcement_checks()
    if not _print_results("v1 smoke policy checks", policy_checks):
        raise typer.Exit(1)
    runtime_checks = profile_smoke_checks(Path(output_root))
    if not _print_results("v1 smoke runtime checks", runtime_checks):
        raise typer.Exit(1)
    console.print(f"[green][OK][/green] v1 smoke checks passed (output root: {out_root.as_posix()})")
    raise typer.Exit(0)

