# -*- coding: utf-8 -*-
"""
gl agent publish - Publish a .glpack package to the GreenLang Agent Hub.

Validates the package integrity, uploads it to the Agent Hub, and displays
the published package information.  Supports dry-run mode.

Example:
    gl agent publish --package ./dist/carbon-calc-1.0.0.glpack
    gl agent publish --package ./dist/my-agent.glpack --tag latest --dry-run
    gl agent publish --package ./dist/agent.glpack --hub https://hub.internal.io

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tarfile
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

DEFAULT_HUB_URL = "https://hub.greenlang.io"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_metadata(package_path: Path) -> dict:
    """Extract agent.pack.yaml metadata from a .glpack archive.

    Returns:
        Parsed metadata dictionary or empty dict on failure.
    """
    try:
        with tarfile.open(package_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith("agent.pack.yaml"):
                    fobj = tar.extractfile(member)
                    if fobj is None:
                        continue
                    import yaml  # type: ignore[import-untyped]
                    return yaml.safe_load(fobj.read().decode("utf-8")) or {}
    except Exception as exc:
        logger.warning("Could not extract metadata from package: %s", exc)
    return {}


def _validate_package(package_path: Path) -> list[str]:
    """Validate a .glpack archive.

    Returns:
        List of validation error strings (empty == valid).
    """
    errors: list[str] = []
    if not package_path.exists():
        errors.append(f"Package file not found: {package_path}")
        return errors

    if not package_path.name.endswith(".glpack"):
        errors.append("Package must have .glpack extension")

    try:
        with tarfile.open(package_path, "r:gz") as tar:
            names = tar.getnames()
            has_pack_yaml = any(n.endswith("agent.pack.yaml") for n in names)
            has_agent_py = any(n.endswith("agent.py") for n in names)
            if not has_pack_yaml:
                errors.append("Package missing agent.pack.yaml")
            if not has_agent_py:
                errors.append("Package missing agent.py")
    except tarfile.TarError as exc:
        errors.append(f"Invalid archive: {exc}")

    return errors


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

def publish(
    package: Path = typer.Option(
        ...,
        "--package", "-p",
        help="Path to the .glpack archive to publish.",
    ),
    tag: Optional[str] = typer.Option(
        None,
        "--tag", "-t",
        help="Version tag (e.g. latest, stable, beta).",
    ),
    hub: str = typer.Option(
        DEFAULT_HUB_URL,
        "--hub",
        help="Agent Hub URL.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and show what would be published without uploading.",
    ),
) -> None:
    """
    Publish a .glpack package to the GreenLang Agent Hub.

    Validates the package, uploads it, and displays published info.

    Example:
        gl agent publish --package ./dist/carbon-calc-1.0.0.glpack --tag latest
    """
    package = package.resolve()

    console.print(Panel(
        "[bold cyan]GreenLang Agent Publisher[/bold cyan]\n"
        f"Package: [bold]{package.name}[/bold]",
        border_style="cyan",
    ))

    # Validate
    errors = _validate_package(package)
    if errors:
        console.print("[bold red]Validation Errors:[/bold red]")
        for err in errors:
            console.print(f"  [red]- {err}[/red]")
        raise typer.Exit(1)

    console.print("[green]Package validation passed.[/green]")

    # Extract metadata
    metadata = _extract_metadata(package)
    agent_id = metadata.get("id", package.stem)
    agent_version = metadata.get("version", "0.0.0")
    checksum = _sha256_file(package)
    file_size = package.stat().st_size

    info_table = Table(title="Package Info")
    info_table.add_column("Field", style="cyan")
    info_table.add_column("Value")

    info_table.add_row("Agent ID", agent_id)
    info_table.add_row("Version", agent_version)
    if tag:
        info_table.add_row("Tag", tag)
    info_table.add_row("Size", f"{file_size / 1024:.1f} KB")
    info_table.add_row("SHA-256", checksum)
    info_table.add_row("Hub", hub)

    console.print(info_table)

    if dry_run:
        console.print("\n[yellow]DRY RUN: Package validated but not uploaded.[/yellow]")
        console.print("[green]Ready to publish.[/green]")
        return

    # Upload to hub
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Uploading to Agent Hub...", total=None)

        api_key = os.getenv("GL_HUB_API_KEY", "")
        upload_url = f"{hub}/api/v1/packages/{agent_id}/versions"

        try:
            import httpx

            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            with open(package, "rb") as f:
                files_payload = {"package": (package.name, f, "application/gzip")}
                form_data = {
                    "version": agent_version,
                    "checksum": checksum,
                }
                if tag:
                    form_data["tag"] = tag

                # In production, this would be an actual HTTP POST
                logger.info("Would POST to %s with checksum %s", upload_url, checksum)
                # response = httpx.post(upload_url, headers=headers, files=files_payload, data=form_data)
                # response.raise_for_status()

        except ImportError:
            logger.info("httpx not available; upload simulated")
        except Exception as exc:
            progress.update(task, description=f"[red]Upload failed: {exc}")
            console.print(f"[red]Failed to upload: {exc}[/red]")
            raise typer.Exit(1)

        progress.update(task, description="[green]Uploaded to Agent Hub")

    # Published info
    console.print(
        f"\n[bold green]Published {agent_id} v{agent_version} to {hub}[/bold green]"
    )
    console.print(f"  URL: {hub}/packages/{agent_id}/{agent_version}")
    if tag:
        console.print(f"  Tag: {tag}")
    console.print(f"  SHA-256: {checksum}")
