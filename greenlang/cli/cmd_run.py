# -*- coding: utf-8 -*-
"""Legacy gl run module kept as a thin proxy.

Canonical runtime path is ``greenlang.cli.main:run``.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.callback(invoke_without_command=True)
def run(
    ctx: typer.Context,
    pipeline: Optional[str] = typer.Argument(
        None, help="Pipeline file or pack reference"
    ),
    inputs: Optional[str] = typer.Option(
        None, "--inputs", "-i", help="Input data file (JSON/YAML)"
    ),
    artifacts: str = typer.Option(
        "out", "--artifacts", "-a", help="Artifacts directory"
    ),
    backend: str = typer.Option(
        "local", "--backend", "-b", help="Execution backend (deprecated)"
    ),
    profile: str = typer.Option("dev", "--profile", "-p", help="Configuration profile (deprecated)"),
    audit: bool = typer.Option(
        False, "--audit", help="Record execution in audit ledger"
    ),
):
    """Deprecated proxy for backward compatibility."""
    if ctx.invoked_subcommand is not None:
        return
    if pipeline is None:
        console.print("[yellow]No pipeline specified[/yellow]")
        raise typer.Exit(2)

    if backend != "local" or profile != "dev":
        console.print(
            "[yellow]Warning:[/yellow] --backend/--profile are deprecated here; "
            "use canonical `gl run` behavior from greenlang.cli.main."
        )

    # Ensure output directory exists to preserve legacy expectations.
    Path(artifacts).mkdir(parents=True, exist_ok=True)

    from .main import run as canonical_run

    canonical_run(
        pipeline=pipeline,
        input_or_config=inputs,
        cbam_imports=None,
        output_dir=artifacts,
        audit=audit,
        dry_run=False,
    )


@app.command("list")
def list_pipelines() -> None:
    """Deprecated placeholder; use canonical runtime commands."""
    console.print(
        "[yellow]`gl run list` is deprecated in this module.[/yellow] "
        "Use canonical `gl run <pipeline>` with explicit paths."
    )


@app.command("info")
def pipeline_info(
    pipeline: str = typer.Argument(..., help="Pipeline name or reference")
) -> None:
    """Deprecated placeholder; use canonical runtime commands."""
    console.print(
        f"[yellow]Pipeline info for '{pipeline}' is deprecated in this module.[/yellow] "
        "Use pipeline files and `gl run` directly."
    )
