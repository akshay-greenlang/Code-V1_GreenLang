# -*- coding: utf-8 -*-
"""
gl agent pack - Build a .glpack archive from an agent directory.

Validates the agent.pack.yaml manifest and source files, produces a
compressed tar archive (.glpack), and reports a summary with file count,
total size, and SHA-256 checksum.

Example:
    gl agent pack --agent-dir ./agents/carbon-calc
    gl agent pack --agent-dir . --output ./dist
    gl agent pack --agent-dir . --validate-only

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import logging
import os
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GLPACK_EXTENSION = ".glpack"
MAX_PACKAGE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

EXCLUDE_PATTERNS = {
    "__pycache__", ".pyc", ".pyo", ".git", ".env", ".venv",
    "node_modules", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".tox", ".DS_Store", "Thumbs.db", "dist",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _should_exclude(path: Path) -> bool:
    """Check whether a path should be excluded from the package."""
    parts = path.parts
    for part in parts:
        if part in EXCLUDE_PATTERNS:
            return True
        for pat in EXCLUDE_PATTERNS:
            if part.endswith(pat):
                return True
    return False


def _collect_files(agent_dir: Path) -> list[Path]:
    """Collect all files in the agent directory, excluding ignored patterns."""
    files: list[Path] = []
    for root, dirs, filenames in os.walk(agent_dir):
        root_path = Path(root)
        # Prune excluded directories in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDE_PATTERNS]
        for fn in filenames:
            fp = root_path / fn
            rel = fp.relative_to(agent_dir)
            if not _should_exclude(rel):
                files.append(fp)
    return sorted(files)


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a single file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_archive(path: Path) -> str:
    """Compute SHA-256 hash of the entire archive."""
    return _sha256_file(path)


def _validate_agent_dir(agent_dir: Path) -> list[str]:
    """Validate the agent directory has required structure.

    Returns:
        List of validation error messages (empty == valid).
    """
    errors: list[str] = []
    pack_yaml = agent_dir / "agent.pack.yaml"
    if not pack_yaml.exists():
        errors.append("Missing agent.pack.yaml manifest")

    agent_py = agent_dir / "agent.py"
    if not agent_py.exists():
        errors.append("Missing agent.py entry point")

    init_py = agent_dir / "__init__.py"
    if not init_py.exists():
        errors.append("Missing __init__.py package init")

    return errors


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

def pack(
    agent_dir: Path = typer.Option(
        Path("."),
        "--agent-dir", "-d",
        help="Path to the agent directory (must contain agent.pack.yaml).",
    ),
    output: Path = typer.Option(
        Path("./dist"),
        "--output", "-o",
        help="Output directory for the .glpack archive.",
    ),
    sign: bool = typer.Option(
        False,
        "--sign",
        help="GPG-sign the package (requires gpg configured).",
    ),
    validate_only: bool = typer.Option(
        False,
        "--validate-only",
        help="Validate the agent directory without building a package.",
    ),
) -> None:
    """
    Build a .glpack archive from a GreenLang agent directory.

    Validates structure, collects source files, compresses into a tar
    archive, and reports package summary.

    Example:
        gl agent pack --agent-dir ./agents/carbon-calc --output ./dist
    """
    agent_dir = agent_dir.resolve()
    if not agent_dir.is_dir():
        console.print(f"[red]Agent directory not found: {agent_dir}[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        "[bold cyan]GreenLang Agent Packager[/bold cyan]\n"
        f"Source: [bold]{agent_dir}[/bold]",
        border_style="cyan",
    ))

    # Validate
    errors = _validate_agent_dir(agent_dir)
    if errors:
        console.print("[bold red]Validation Errors:[/bold red]")
        for err in errors:
            console.print(f"  [red]- {err}[/red]")
        raise typer.Exit(1)

    console.print("[green]Validation passed.[/green]")

    if validate_only:
        console.print("[yellow]Validate-only mode: no package built.[/yellow]")
        return

    # Collect files
    files = _collect_files(agent_dir)
    console.print(f"[bold]Files to package:[/bold] {len(files)}")

    # Create output directory
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Determine archive name from pack.yaml or directory name
    agent_name = agent_dir.name
    try:
        import yaml  # type: ignore[import-untyped]
        pack_yaml = agent_dir / "agent.pack.yaml"
        meta = yaml.safe_load(pack_yaml.read_text(encoding="utf-8")) or {}
        agent_name = meta.get("id", agent_name)
        agent_version = meta.get("version", "0.0.0")
    except Exception:
        agent_version = "0.0.0"

    archive_name = f"{agent_name}-{agent_version}{GLPACK_EXTENSION}"
    archive_path = output / archive_name

    # Build archive
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Building .glpack archive...", total=None)

        with tarfile.open(archive_path, "w:gz") as tar:
            for fp in files:
                arcname = str(fp.relative_to(agent_dir))
                tar.add(str(fp), arcname=arcname)

        progress.update(task, description="[green]Archive built")

    # Compute checksum
    checksum = _sha256_archive(archive_path)
    archive_size = archive_path.stat().st_size

    # Check size limit
    if archive_size > MAX_PACKAGE_SIZE_BYTES:
        console.print(
            f"[red]Package exceeds size limit: "
            f"{archive_size / (1024*1024):.1f} MB > "
            f"{MAX_PACKAGE_SIZE_BYTES / (1024*1024):.0f} MB[/red]"
        )
        archive_path.unlink(missing_ok=True)
        raise typer.Exit(1)

    # GPG signing
    if sign:
        try:
            import subprocess

            result = subprocess.run(
                ["gpg", "--detach-sign", "--armor", str(archive_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                console.print("[green]Package signed with GPG.[/green]")
            else:
                console.print(f"[yellow]GPG signing failed: {result.stderr.strip()}[/yellow]")
        except FileNotFoundError:
            console.print("[yellow]GPG not found; skipping signing.[/yellow]")

    # Summary table
    summary = Table(title="Package Summary")
    summary.add_column("Field", style="cyan")
    summary.add_column("Value")

    summary.add_row("Agent", agent_name)
    summary.add_row("Version", agent_version)
    summary.add_row("Archive", str(archive_path))
    summary.add_row("Files", str(len(files)))
    summary.add_row("Size", f"{archive_size / 1024:.1f} KB")
    summary.add_row("SHA-256", checksum)
    summary.add_row("Built At", datetime.now(timezone.utc).isoformat())

    console.print(summary)
    console.print(f"\n[bold green]Package created:[/bold green] {archive_path}")
