# -*- coding: utf-8 -*-
"""GL-Comply-APP CLI — `comply` commands."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

try:
    import typer
except ImportError:
    typer = None  # type: ignore[assignment]

from agents.orchestrator_agent import ComplianceOrchestrator
from schemas.models import (
    ApplicabilityRequest,
    ComplianceRequest,
)
from services import applicability, registry


def _require_typer():
    if typer is None:
        sys.exit("typer is required; install greenlang with extras [cli]")


_require_typer()
app = typer.Typer(name="comply", help="GreenLang Unified Compliance Hub CLI")


@app.command("run")
def run(
    input_file: Path = typer.Argument(..., exists=True, readable=True),
    output: Path | None = typer.Option(None, "--output", "-o"),
):
    """Run compliance orchestration from a JSON request file."""
    raw = json.loads(input_file.read_text(encoding="utf-8"))
    request = ComplianceRequest.model_validate(raw)
    report = asyncio.run(ComplianceOrchestrator().run(request))
    payload = report.model_dump_json(indent=2)
    if output:
        output.write_text(payload, encoding="utf-8")
        typer.echo(f"Report written to {output}")
    else:
        typer.echo(payload)


@app.command("applicability")
def check_applicability(input_file: Path = typer.Argument(..., exists=True, readable=True)):
    """Check framework applicability for an entity."""
    raw = json.loads(input_file.read_text(encoding="utf-8"))
    request = ApplicabilityRequest.model_validate(raw)
    result = applicability.evaluate(request)
    typer.echo(result.model_dump_json(indent=2))


@app.command("frameworks")
def frameworks():
    """List registered compliance framework adapters."""
    for fw in registry.available():
        typer.echo(fw.value)


@app.command("status")
def status(job_id: str):
    """Fetch status for a compliance job (in-memory store; use API for prod)."""
    from services.store import default_store

    report = default_store().get(job_id)
    if report is None:
        typer.echo(f"Unknown job_id {job_id}", err=True)
        raise typer.Exit(code=1)
    typer.echo(report.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
