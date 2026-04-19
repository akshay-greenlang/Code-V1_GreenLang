# -*- coding: utf-8 -*-
"""
Productivity Helpers
====================

Productivity utilities for rapid agent development:
- Agent scaffolding from CSV row
- Batch agent creation
- Bulk validation
- Progress tracking
- Time estimation

Supports GreenLang's 1-agent-per-hour goal.

Usage:
    gl agent batch create agents.csv
    gl agent batch validate ./agents/
    gl agent batch test ./agents/ --parallel
"""

import os
import sys
import csv
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.live import Live
from pydantic import BaseModel, Field

# Create sub-app for productivity commands
productivity_app = typer.Typer(
    name="batch",
    help="Batch operations and productivity tools",
    no_args_is_help=True,
)

console = Console()
logger = logging.getLogger(__name__)


# =============================================================================
# Productivity Models
# =============================================================================

class BatchOperation(BaseModel):
    """Batch operation result."""
    operation: str
    total: int
    successful: int
    failed: int
    skipped: int
    duration_seconds: float
    results: List[Dict[str, Any]] = Field(default_factory=list)


class AgentDefinition(BaseModel):
    """Agent definition from CSV/JSON."""
    name: str
    template: str = "custom"
    pack: Optional[str] = None
    description: Optional[str] = None
    category: str = "custom"
    author: str = "GreenLang Team"


# =============================================================================
# Batch Commands
# =============================================================================

@productivity_app.command("create")
def batch_create_command(
    input_file: Path = typer.Argument(
        ...,
        help="CSV or JSON file with agent definitions",
        exists=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for agents",
    ),
    parallel: int = typer.Option(
        4,
        "--parallel", "-p",
        help="Number of parallel workers",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be created",
    ),
):
    """
    Create multiple agents from a CSV or JSON file.

    CSV Format:
        name,template,pack,description,category
        carbon-calc,calculator,emissions,"Carbon calculator",calculator
        eudr-check,regulatory,eudr,"EUDR compliance checker",regulatory

    Example:
        gl agent batch create agents.csv
        gl agent batch create agents.json --parallel 8
    """
    console.print(Panel(
        "[bold cyan]Batch Agent Creation[/bold cyan]\n"
        f"Input: {input_file}",
        border_style="cyan"
    ))

    # Load definitions
    definitions = _load_agent_definitions(input_file)

    if not definitions:
        console.print("[yellow]No agent definitions found[/yellow]")
        raise typer.Exit(0)

    console.print(f"[bold]Agents to create:[/bold] {len(definitions)}")

    if dry_run:
        console.print("\n[yellow]Dry run - showing planned operations:[/yellow]")
        for defn in definitions:
            console.print(f"  - {defn.name} (template: {defn.template})")
        raise typer.Exit(0)

    # Create agents
    result = _batch_create_agents(definitions, output_dir, parallel)
    _display_batch_result(result)


@productivity_app.command("validate")
def batch_validate_command(
    agents_dir: Path = typer.Argument(
        ...,
        help="Directory containing agents to validate",
        exists=True,
    ),
    parallel: int = typer.Option(
        4,
        "--parallel", "-p",
        help="Number of parallel workers",
    ),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        help="Enable strict validation",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output validation report",
    ),
):
    """
    Validate multiple agents in a directory.

    Example:
        gl agent batch validate ./agents/
        gl agent batch validate ./packs/eudr/agents --strict
    """
    console.print(Panel(
        "[bold cyan]Batch Validation[/bold cyan]\n"
        f"Directory: {agents_dir}",
        border_style="cyan"
    ))

    # Find agents
    agents = _find_agents(agents_dir)

    if not agents:
        console.print("[yellow]No agents found[/yellow]")
        raise typer.Exit(0)

    console.print(f"[bold]Agents to validate:[/bold] {len(agents)}")

    # Validate agents
    result = _batch_validate_agents(agents, parallel, strict)
    _display_batch_result(result)

    if output:
        _save_batch_report(result, output)


@productivity_app.command("test")
def batch_test_command(
    agents_dir: Path = typer.Argument(
        ...,
        help="Directory containing agents to test",
        exists=True,
    ),
    parallel: int = typer.Option(
        4,
        "--parallel", "-p",
        help="Number of parallel workers",
    ),
    golden_only: bool = typer.Option(
        False,
        "--golden-only",
        help="Run only golden tests",
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast", "-x",
        help="Stop on first failure",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output test report",
    ),
):
    """
    Run tests for multiple agents.

    Example:
        gl agent batch test ./agents/
        gl agent batch test ./packs/ --golden-only
    """
    console.print(Panel(
        "[bold cyan]Batch Test Runner[/bold cyan]\n"
        f"Directory: {agents_dir}",
        border_style="cyan"
    ))

    # Find agents
    agents = _find_agents(agents_dir)

    if not agents:
        console.print("[yellow]No agents found[/yellow]")
        raise typer.Exit(0)

    console.print(f"[bold]Agents to test:[/bold] {len(agents)}")

    # Run tests
    result = _batch_test_agents(agents, parallel, golden_only, fail_fast)
    _display_batch_result(result)

    if output:
        _save_batch_report(result, output)


@productivity_app.command("certify")
def batch_certify_command(
    agents_dir: Path = typer.Argument(
        ...,
        help="Directory containing agents to certify",
        exists=True,
    ),
    level: str = typer.Option(
        "bronze",
        "--level", "-l",
        help="Minimum certification level target",
    ),
    parallel: int = typer.Option(
        2,
        "--parallel", "-p",
        help="Number of parallel workers",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output certification report",
    ),
):
    """
    Certify multiple agents.

    Example:
        gl agent batch certify ./agents/
        gl agent batch certify ./packs/ --level silver
    """
    console.print(Panel(
        "[bold cyan]Batch Certification[/bold cyan]\n"
        f"Directory: {agents_dir}\n"
        f"Target Level: {level.upper()}",
        border_style="cyan"
    ))

    # Find agents
    agents = _find_agents(agents_dir)

    if not agents:
        console.print("[yellow]No agents found[/yellow]")
        raise typer.Exit(0)

    console.print(f"[bold]Agents to certify:[/bold] {len(agents)}")

    # Certify agents
    result = _batch_certify_agents(agents, level, parallel)
    _display_batch_result(result)

    if output:
        _save_batch_report(result, output)


@productivity_app.command("scaffold")
def batch_scaffold_command(
    csv_file: Path = typer.Argument(
        ...,
        help="CSV file with agent definitions",
        exists=True,
    ),
    row: Optional[int] = typer.Option(
        None,
        "--row", "-r",
        help="Specific row to scaffold (1-indexed)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory",
    ),
):
    """
    Scaffold agent from a CSV row.

    Useful for processing large agent inventories.

    Example:
        gl agent batch scaffold agents.csv --row 5
        gl agent batch scaffold inventory.csv
    """
    console.print(Panel(
        "[bold cyan]Agent Scaffolding from CSV[/bold cyan]\n"
        f"File: {csv_file}",
        border_style="cyan"
    ))

    # Load CSV
    definitions = _load_agent_definitions(csv_file)

    if row is not None:
        if row < 1 or row > len(definitions):
            console.print(f"[red]Invalid row: {row}. File has {len(definitions)} rows.[/red]")
            raise typer.Exit(1)
        definitions = [definitions[row - 1]]
        console.print(f"[bold]Scaffolding row {row}:[/bold] {definitions[0].name}")
    else:
        console.print(f"[bold]Scaffolding {len(definitions)} agents[/bold]")

    # Scaffold
    from .create_command import create_agent_from_template

    for defn in definitions:
        console.print(f"\n[cyan]Creating: {defn.name}[/cyan]")
        try:
            create_agent_from_template(
                template=defn.template,
                name=defn.name,
                pack=defn.pack,
                output_dir=output_dir / defn.name if output_dir else None,
                description=defn.description,
                author=defn.author,
            )
        except Exception as e:
            console.print(f"[red]Failed: {str(e)}[/red]")


@productivity_app.command("estimate")
def batch_estimate_command(
    agents_dir: Path = typer.Argument(
        ...,
        help="Directory containing agents",
        exists=True,
    ),
):
    """
    Estimate time to complete agent pipeline.

    Example:
        gl agent batch estimate ./agents/
    """
    console.print(Panel(
        "[bold cyan]Pipeline Time Estimation[/bold cyan]\n"
        f"Directory: {agents_dir}",
        border_style="cyan"
    ))

    # Find agents
    agents = _find_agents(agents_dir)

    if not agents:
        console.print("[yellow]No agents found[/yellow]")
        raise typer.Exit(0)

    # Estimate times (based on typical durations)
    estimates = {
        "validation": 5,  # seconds per agent
        "unit_tests": 30,
        "golden_tests": 120,
        "integration_tests": 60,
        "certification": 180,
        "deployment": 120,
    }

    table = Table(title=f"Time Estimates for {len(agents)} Agents")
    table.add_column("Phase", style="cyan")
    table.add_column("Per Agent", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Parallel (4)", justify="right")

    total_time = 0
    for phase, seconds in estimates.items():
        total_seconds = seconds * len(agents)
        parallel_seconds = total_seconds / 4
        total_time += parallel_seconds

        table.add_row(
            phase.replace("_", " ").title(),
            f"{seconds}s",
            str(timedelta(seconds=total_seconds)),
            str(timedelta(seconds=int(parallel_seconds))),
        )

    console.print(table)

    console.print(f"\n[bold]Total estimated time (parallel):[/bold] {timedelta(seconds=int(total_time))}")
    console.print(f"[bold]Agents per hour:[/bold] {int(3600 / (total_time / len(agents)))}")


# =============================================================================
# List Agents Function (Used by cli_main.py)
# =============================================================================

def list_agents_impl(
    pack: Optional[str] = None,
    status: Optional[str] = None,
    category: Optional[str] = None,
    format: str = "table",
    limit: int = 50,
) -> None:
    """
    List agents implementation.

    Args:
        pack: Filter by pack
        status: Filter by status
        category: Filter by category
        format: Output format
        limit: Maximum results
    """
    console.print(Panel(
        "[bold cyan]Agent Registry[/bold cyan]",
        border_style="cyan"
    ))

    # Find local agents
    agents = []

    search_paths = ["./agents", "./packs"]
    for search_path in search_paths:
        path = Path(search_path)
        if path.exists():
            for agent_dir in path.rglob("*"):
                if agent_dir.is_dir() and (agent_dir / "agent.py").exists():
                    agent_info = _get_agent_info(agent_dir)
                    if agent_info:
                        agents.append(agent_info)

    # Apply filters
    if pack:
        agents = [a for a in agents if a.get("pack") == pack]
    if status:
        agents = [a for a in agents if a.get("status") == status]
    if category:
        agents = [a for a in agents if a.get("category") == category]

    # Limit results
    agents = agents[:limit]

    if not agents:
        console.print("[yellow]No agents found[/yellow]")
        return

    if format == "json":
        console.print_json(data=agents)
        return

    if format == "yaml":
        import yaml
        console.print(yaml.dump(agents, default_flow_style=False))
        return

    # Table format
    table = Table(title=f"Agents ({len(agents)})")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Version")
    table.add_column("Category")
    table.add_column("Status")
    table.add_column("Pack")

    for agent in agents:
        status_style = {
            "active": "[green]active[/green]",
            "draft": "[yellow]draft[/yellow]",
            "deprecated": "[red]deprecated[/red]",
        }.get(agent.get("status", "draft"), agent.get("status", "N/A"))

        table.add_row(
            agent.get("id", "N/A"),
            agent.get("name", "N/A"),
            agent.get("version", "N/A"),
            agent.get("category", "N/A"),
            status_style,
            agent.get("pack", "-"),
        )

    console.print(table)


# =============================================================================
# Core Batch Functions
# =============================================================================

def _load_agent_definitions(input_file: Path) -> List[AgentDefinition]:
    """Load agent definitions from CSV or JSON."""
    definitions = []

    if input_file.suffix.lower() == ".json":
        with open(input_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    definitions.append(AgentDefinition(**item))
    elif input_file.suffix.lower() == ".csv":
        with open(input_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter out empty values
                row = {k: v for k, v in row.items() if v}
                definitions.append(AgentDefinition(**row))

    return definitions


def _find_agents(directory: Path) -> List[Path]:
    """Find all agents in a directory."""
    agents = []

    for item in directory.iterdir():
        if item.is_dir():
            # Check if it's an agent directory
            if (item / "agent.py").exists() or (item / "pack.yaml").exists():
                agents.append(item)
            else:
                # Recurse one level
                for sub_item in item.iterdir():
                    if sub_item.is_dir():
                        if (sub_item / "agent.py").exists() or (sub_item / "pack.yaml").exists():
                            agents.append(sub_item)

    return agents


def _batch_create_agents(
    definitions: List[AgentDefinition],
    output_dir: Optional[Path],
    parallel: int,
) -> BatchOperation:
    """Create multiple agents in parallel."""
    from .create_command import create_agent_from_template

    start_time = time.time()
    results = []
    successful = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Creating agents...", total=len(definitions))

        for defn in definitions:
            try:
                out_dir = output_dir / defn.name if output_dir else None
                create_agent_from_template(
                    template=defn.template,
                    name=defn.name,
                    pack=defn.pack,
                    output_dir=out_dir,
                    description=defn.description,
                    author=defn.author,
                )
                results.append({"name": defn.name, "status": "success"})
                successful += 1
            except Exception as e:
                results.append({"name": defn.name, "status": "failed", "error": str(e)})
                failed += 1

            progress.update(task, advance=1)

    return BatchOperation(
        operation="create",
        total=len(definitions),
        successful=successful,
        failed=failed,
        skipped=0,
        duration_seconds=time.time() - start_time,
        results=results,
    )


def _batch_validate_agents(
    agents: List[Path],
    parallel: int,
    strict: bool,
) -> BatchOperation:
    """Validate multiple agents."""
    from .validate_command import validate_agent_implementation

    start_time = time.time()
    results = []
    successful = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Validating agents...", total=len(agents))

        for agent_path in agents:
            try:
                result = validate_agent_implementation(agent_path, include_tests=strict)
                if result.valid:
                    results.append({"name": agent_path.name, "status": "valid", "score": result.score})
                    successful += 1
                else:
                    results.append({
                        "name": agent_path.name,
                        "status": "invalid",
                        "score": result.score,
                        "errors": len(result.errors),
                    })
                    failed += 1
            except Exception as e:
                results.append({"name": agent_path.name, "status": "error", "error": str(e)})
                failed += 1

            progress.update(task, advance=1)

    return BatchOperation(
        operation="validate",
        total=len(agents),
        successful=successful,
        failed=failed,
        skipped=0,
        duration_seconds=time.time() - start_time,
        results=results,
    )


def _batch_test_agents(
    agents: List[Path],
    parallel: int,
    golden_only: bool,
    fail_fast: bool,
) -> BatchOperation:
    """Run tests for multiple agents."""
    import subprocess

    start_time = time.time()
    results = []
    successful = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running tests...", total=len(agents))

        for agent_path in agents:
            tests_dir = agent_path / "tests"
            if not tests_dir.exists():
                results.append({"name": agent_path.name, "status": "skipped", "reason": "no tests"})
                progress.update(task, advance=1)
                continue

            try:
                pytest_args = [sys.executable, "-m", "pytest", str(tests_dir), "-q"]
                if golden_only:
                    pytest_args.extend(["-k", "golden"])

                result = subprocess.run(pytest_args, capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    results.append({"name": agent_path.name, "status": "passed"})
                    successful += 1
                else:
                    results.append({"name": agent_path.name, "status": "failed", "output": result.stdout[:500]})
                    failed += 1
                    if fail_fast:
                        break

            except subprocess.TimeoutExpired:
                results.append({"name": agent_path.name, "status": "timeout"})
                failed += 1
            except Exception as e:
                results.append({"name": agent_path.name, "status": "error", "error": str(e)})
                failed += 1

            progress.update(task, advance=1)

    return BatchOperation(
        operation="test",
        total=len(agents),
        successful=successful,
        failed=failed,
        skipped=len(agents) - successful - failed,
        duration_seconds=time.time() - start_time,
        results=results,
    )


def _batch_certify_agents(
    agents: List[Path],
    level: str,
    parallel: int,
) -> BatchOperation:
    """Certify multiple agents."""
    from .certify_command import certify_agent_impl

    start_time = time.time()
    results = []
    successful = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Certifying agents...", total=len(agents))

        for agent_path in agents:
            try:
                result = certify_agent_impl(
                    agent_id=str(agent_path),
                    level=level,
                    verbose=False,
                )
                if result.certified:
                    results.append({
                        "name": agent_path.name,
                        "status": "certified",
                        "level": result.level.value,
                        "score": result.overall_score,
                    })
                    successful += 1
                else:
                    results.append({
                        "name": agent_path.name,
                        "status": "not_certified",
                        "level": result.level.value,
                        "score": result.overall_score,
                    })
                    failed += 1
            except SystemExit:
                results.append({"name": agent_path.name, "status": "not_certified"})
                failed += 1
            except Exception as e:
                results.append({"name": agent_path.name, "status": "error", "error": str(e)})
                failed += 1

            progress.update(task, advance=1)

    return BatchOperation(
        operation="certify",
        total=len(agents),
        successful=successful,
        failed=failed,
        skipped=0,
        duration_seconds=time.time() - start_time,
        results=results,
    )


# =============================================================================
# Helper Functions
# =============================================================================

def _get_agent_info(agent_dir: Path) -> Optional[Dict[str, Any]]:
    """Get agent information from directory."""
    info = {
        "id": agent_dir.name,
        "name": agent_dir.name.replace("-", " ").replace("_", " ").title(),
        "path": str(agent_dir),
        "version": "0.0.0",
        "status": "draft",
        "category": "custom",
    }

    # Try to load from pack.yaml
    pack_yaml = agent_dir / "pack.yaml"
    if pack_yaml.exists():
        try:
            import yaml
            with open(pack_yaml) as f:
                spec = yaml.safe_load(f)
                if spec:
                    info.update({
                        "id": spec.get("id", info["id"]),
                        "name": spec.get("name", info["name"]),
                        "version": spec.get("version", info["version"]),
                        "category": spec.get("metadata", {}).get("category", info["category"]),
                    })
        except Exception:
            pass

    # Check for __init__.py version
    init_file = agent_dir / "__init__.py"
    if init_file.exists():
        import re
        content = init_file.read_text()
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            info["version"] = match.group(1)

    return info


def _display_batch_result(result: BatchOperation) -> None:
    """Display batch operation result."""
    console.print()

    success_rate = (result.successful / result.total * 100) if result.total > 0 else 0
    color = "green" if success_rate >= 80 else "yellow" if success_rate >= 60 else "red"

    console.print(Panel(
        f"[bold]Operation:[/bold] {result.operation.title()}\n"
        f"[bold]Total:[/bold] {result.total}\n"
        f"[bold]Successful:[/bold] [{color}]{result.successful}[/{color}]\n"
        f"[bold]Failed:[/bold] [red]{result.failed}[/red]\n"
        f"[bold]Skipped:[/bold] {result.skipped}\n"
        f"[bold]Success Rate:[/bold] [{color}]{success_rate:.1f}%[/{color}]\n"
        f"[bold]Duration:[/bold] {result.duration_seconds:.2f}s\n"
        f"[bold]Agents/Hour:[/bold] {int(3600 / (result.duration_seconds / result.total)) if result.total > 0 else 0}",
        title="Batch Operation Result",
        border_style=color,
    ))

    # Show failures
    failures = [r for r in result.results if r.get("status") in ["failed", "error", "invalid", "not_certified"]]
    if failures:
        console.print("\n[bold red]Failures:[/bold red]")
        for f in failures[:10]:  # Show first 10
            console.print(f"  [red]x[/red] {f['name']}: {f.get('error', f.get('status', 'failed'))}")
        if len(failures) > 10:
            console.print(f"  ... and {len(failures) - 10} more")

    console.print()


def _save_batch_report(result: BatchOperation, output: Path) -> None:
    """Save batch operation report."""
    report = result.model_dump()

    with open(output, "w") as f:
        json.dump(report, f, indent=2, default=str)

    console.print(f"[green]Report saved to:[/green] {output}")
