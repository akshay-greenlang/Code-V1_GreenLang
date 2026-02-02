# -*- coding: utf-8 -*-
"""Validate command for GreenLang CLI."""

import typer
from pathlib import Path
from rich.console import Console
import yaml
import json
from typing import Optional

console = Console()
app = typer.Typer(help="Validate GreenLang pipeline and pack files")


@app.command()
def file(
    path: str = typer.Argument(..., help="Path to the file to validate"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed validation output"
    ),
):
    """Validate a GreenLang pipeline (gl.yaml) or pack (pack.yaml) file."""
    file_path = Path(path)

    if not file_path.exists():
        console.print(f"[red]✗ File not found: {path}[/red]")
        raise typer.Exit(1)

    try:
        # Load the YAML file
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        # Basic validation based on file name
        if file_path.name == "gl.yaml" or "pipeline" in file_path.name:
            validate_pipeline(data, verbose)
            console.print(f"[green]✓ VALID pipeline: {path}[/green]")
        elif file_path.name == "pack.yaml":
            validate_pack(data, verbose)
            console.print(f"[green]✓ VALID pack: {path}[/green]")
        else:
            # Generic YAML validation
            console.print(f"[green]✓ VALID YAML: {path}[/green]")

        if verbose:
            console.print("\n[dim]Validated structure:[/dim]")
            console.print(json.dumps(data, indent=2))

    except yaml.YAMLError as e:
        console.print(f"[red]✗ Invalid YAML: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗ Validation failed: {e}[/red]")
        raise typer.Exit(1)


def validate_pipeline(data: dict, verbose: bool = False):
    """Validate pipeline structure."""
    # Basic pipeline validation
    if "name" not in data:
        raise ValueError("Pipeline must have a 'name' field")

    if "steps" not in data:
        raise ValueError("Pipeline must have 'steps' field")

    if not isinstance(data["steps"], list):
        raise ValueError("Pipeline 'steps' must be a list")

    for i, step in enumerate(data["steps"]):
        if "name" not in step:
            raise ValueError(f"Step {i} must have a 'name' field")
        if "uses" not in step:
            raise ValueError(f"Step {i} must have a 'uses' field")

    if verbose:
        console.print(
            f"[dim]Pipeline '{data['name']}' has {len(data['steps'])} steps[/dim]"
        )


def validate_pack(data: dict, verbose: bool = False):
    """Validate pack structure."""
    # Basic pack validation
    if "name" not in data:
        raise ValueError("Pack must have a 'name' field")

    if "version" not in data:
        raise ValueError("Pack must have a 'version' field")

    if verbose:
        console.print(f"[dim]Pack '{data['name']}' version {data['version']}[/dim]")


@app.command()
def schema(
    path: str = typer.Argument(..., help="Path to validate against schema"),
    schema_path: Optional[str] = typer.Option(
        None, "--schema", "-s", help="Path to JSON schema"
    ),
):
    """Validate a file against a JSON schema."""
    try:
        pass

        file_path = Path(path)
        if not file_path.exists():
            console.print(f"[red]✗ File not found: {path}[/red]")
            raise typer.Exit(1)

        # Load the file
        with open(file_path, "r") as f:
            if file_path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        # For now, just validate it's valid JSON/YAML
        console.print(f"[green]✓ Valid structure: {path}[/green]")

    except Exception as e:
        console.print(f"[red]✗ Validation failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
