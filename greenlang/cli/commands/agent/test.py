# -*- coding: utf-8 -*-
"""
gl agent test - Discover and run tests for a GreenLang agent.

Runs unit, integration, and end-to-end tests via pytest subprocess,
parses results, enforces coverage thresholds, and presents a summary.

Example:
    gl agent test --agent-dir ./agents/carbon-calc --coverage
    gl agent test --agent-dir ./agents/eudr --unit --verbose --failfast
    gl agent test --agent-dir . --e2e --coverage

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_COVERAGE_THRESHOLD = 85


def _find_test_dirs(agent_dir: Path) -> dict[str, Path]:
    """Discover test directories inside the agent directory.

    Returns:
        Mapping of test category to its directory path.
    """
    found: dict[str, Path] = {}
    tests_root = agent_dir / "tests"
    if tests_root.is_dir():
        found["unit"] = tests_root
    for sub in ("unit", "integration", "e2e"):
        candidate = tests_root / sub
        if candidate.is_dir():
            found[sub] = candidate
    return found


def _parse_pytest_json(report_path: Path) -> dict:
    """Parse a pytest-json-report file and return summary dict."""
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        summary = data.get("summary", {})
        return {
            "total": summary.get("total", 0),
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "errors": summary.get("error", 0),
            "skipped": summary.get("skipped", 0),
            "duration": round(data.get("duration", 0.0), 2),
        }
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not parse pytest JSON report: %s", exc)
        return {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "duration": 0.0}


def _parse_coverage_pct(coverage_json_path: Path) -> float:
    """Extract overall coverage percentage from a coverage.json file."""
    try:
        data = json.loads(coverage_json_path.read_text(encoding="utf-8"))
        return float(data.get("totals", {}).get("percent_covered", 0.0))
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        logger.warning("Could not parse coverage JSON: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

def test(
    agent_dir: Path = typer.Option(
        Path("."),
        "--agent-dir", "-d",
        help="Path to the agent directory containing tests/.",
    ),
    unit: bool = typer.Option(False, "--unit", help="Run unit tests only."),
    integration: bool = typer.Option(False, "--integration", help="Run integration tests only."),
    e2e: bool = typer.Option(False, "--e2e", help="Run end-to-end tests only."),
    coverage: bool = typer.Option(False, "--coverage", "-c", help="Collect coverage data."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose pytest output."),
    failfast: bool = typer.Option(False, "--failfast", "-x", help="Stop on first failure."),
) -> None:
    """
    Discover and run tests for a GreenLang agent.

    By default runs all discovered tests. Use --unit, --integration, or --e2e
    to restrict the scope.

    Example:
        gl agent test --agent-dir ./agents/carbon-calc --coverage
    """
    agent_dir = agent_dir.resolve()
    if not agent_dir.is_dir():
        console.print(f"[red]Agent directory not found: {agent_dir}[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        "[bold cyan]GreenLang Agent Test Runner[/bold cyan]\n"
        f"Agent directory: [bold]{agent_dir}[/bold]",
        border_style="cyan",
    ))

    # Discover test directories
    dirs = _find_test_dirs(agent_dir)
    if not dirs:
        console.print("[red]No tests/ directory found in the agent.[/red]")
        console.print("Run 'gl agent create' to scaffold tests.")
        raise typer.Exit(1)

    # Determine which directories to run
    selected: list[Path] = []
    any_filter = unit or integration or e2e
    if any_filter:
        if unit and "unit" in dirs:
            selected.append(dirs["unit"])
        if integration and "integration" in dirs:
            selected.append(dirs["integration"])
        if e2e and "e2e" in dirs:
            selected.append(dirs["e2e"])
        if not selected:
            # Fallback: run the root tests dir if filtered category not found
            selected.append(dirs.get("unit", agent_dir / "tests"))
    else:
        # No filter -- run entire tests directory
        selected.append(agent_dir / "tests")

    console.print(f"[bold]Test targets:[/bold] {', '.join(str(p) for p in selected)}")

    # Build pytest command
    pytest_args: list[str] = [sys.executable, "-m", "pytest"]
    pytest_args.extend(str(p) for p in selected)

    if verbose:
        pytest_args.append("-v")
    else:
        pytest_args.append("-q")

    if failfast:
        pytest_args.append("-x")

    # JSON report for summary parsing
    json_report = Path(tempfile.mktemp(suffix=".json"))
    pytest_args.extend(["--tb=short"])

    # Coverage args
    cov_json_path: Optional[Path] = None
    if coverage:
        module_name = agent_dir.name.replace("-", "_")
        pytest_args.extend([
            f"--cov={agent_dir}",
            "--cov-report=term-missing",
        ])
        cov_json_path = Path(tempfile.mktemp(suffix="_coverage.json"))
        pytest_args.append(f"--cov-report=json:{cov_json_path}")

    console.print(f"\n[dim]Running: {' '.join(pytest_args)}[/dim]\n")

    # Execute
    proc = subprocess.run(pytest_args, cwd=str(agent_dir))

    # Display summary
    console.print("")
    if proc.returncode == 0:
        console.print("[bold green]All tests passed.[/bold green]")
    else:
        console.print(f"[bold red]Tests failed (exit code {proc.returncode}).[/bold red]")

    # Coverage threshold check
    if coverage and cov_json_path and cov_json_path.exists():
        cov_pct = _parse_coverage_pct(cov_json_path)
        threshold = _DEFAULT_COVERAGE_THRESHOLD

        table = Table(title="Coverage Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        color = "green" if cov_pct >= threshold else "red"
        table.add_row("Coverage", f"[{color}]{cov_pct:.1f}%[/{color}]")
        table.add_row("Threshold", f"{threshold}%")
        table.add_row(
            "Status",
            f"[{color}]{'PASS' if cov_pct >= threshold else 'FAIL'}[/{color}]",
        )
        console.print(table)

        if cov_pct < threshold:
            console.print(
                f"[red]Coverage {cov_pct:.1f}% is below the {threshold}% threshold.[/red]"
            )
            if proc.returncode == 0:
                # Override exit code if tests passed but coverage failed
                raise typer.Exit(1)

        # Clean up temp file
        cov_json_path.unlink(missing_ok=True)

    # Clean up temp json report
    json_report.unlink(missing_ok=True)

    if proc.returncode != 0:
        raise typer.Exit(proc.returncode)
