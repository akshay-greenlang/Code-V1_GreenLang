# -*- coding: utf-8 -*-
"""Scope Engine CLI (Typer-based)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import typer
except ImportError:
    typer = None  # type: ignore[assignment]

from greenlang.scope_engine.models import ComputationRequest
from greenlang.scope_engine.service import ScopeEngineService


def _require_typer():
    if typer is None:
        sys.exit("typer is required; install greenlang with extras [cli]")


def build_app():
    _require_typer()
    app = typer.Typer(name="scope-engine", help="Unified Scope 1/2/3 engine")

    @app.command()
    def compute(input_file: Path = typer.Argument(..., exists=True, readable=True)):
        """Run a scope-engine computation from a JSON request file."""
        raw = json.loads(input_file.read_text(encoding="utf-8"))
        request = ComputationRequest.model_validate(raw)
        response = ScopeEngineService().compute(request)
        typer.echo(response.model_dump_json(indent=2))

    return app


if __name__ == "__main__":
    build_app()()
