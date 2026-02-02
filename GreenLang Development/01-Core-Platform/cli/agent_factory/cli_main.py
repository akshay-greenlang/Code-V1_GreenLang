# -*- coding: utf-8 -*-
"""
GreenLang Agent Factory CLI - Main Entry Point
===============================================

Enhanced CLI for rapid agent creation, validation, testing, certification, and deployment.
Supports GreenLang's 1-agent-per-hour productivity goal.

Usage:
    gl agent create <template> --name <name> --pack <pack>
    gl agent validate <spec-path>
    gl agent test <agent-id> [--golden] [--integration] [--e2e]
    gl agent certify <agent-id> --level [gold|silver|bronze]
    gl agent deploy <agent-id> --env [dev|staging|prod]
    gl agent list [--pack <pack>] [--status <status>]
    gl agent diff <agent-id> <version1> <version2>

Example:
    >>> from greenlang.cli.agent_factory import app
    >>> app()  # Run CLI
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Create the main Typer app
app = typer.Typer(
    name="agent",
    help="GreenLang Agent Factory: Create, validate, test, certify, and deploy agents",
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode="rich",
)

console = Console()
logger = logging.getLogger(__name__)

# Version info
__version__ = "1.0.0"


# =============================================================================
# Callback and Version
# =============================================================================

@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit",
        is_eager=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose output",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress non-essential output",
    ),
):
    """
    GreenLang Agent Factory CLI - Production-grade agent tooling.

    Build, validate, test, certify, and deploy AI agents at scale.
    Supports GreenLang's 1-agent-per-hour productivity target.
    """
    if version:
        console.print(f"[bold green]GreenLang Agent Factory CLI[/bold green] v{__version__}")
        console.print("[dim]Infrastructure for Climate Intelligence[/dim]")
        raise typer.Exit(0)

    # Store context options
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet


# =============================================================================
# Import and register sub-commands
# =============================================================================

from .create_command import create_app
from .validate_command import validate_app
from .test_command import test_app
from .certify_command import certify_app
from .deploy_command import deploy_app
from .template_command import template_app
from .productivity_helpers import productivity_app

# Add sub-commands as groups
app.add_typer(create_app, name="create", help="Create agents from templates")
app.add_typer(template_app, name="template", help="Manage agent templates")
app.add_typer(productivity_app, name="batch", help="Batch operations and productivity tools")


# =============================================================================
# Direct Commands (not sub-groups)
# =============================================================================

@app.command("new")
def create_agent(
    template: str = typer.Argument(
        ...,
        help="Template to use (calculator, validator, reporter, regulatory, custom)",
    ),
    name: str = typer.Option(
        ...,
        "--name", "-n",
        help="Agent name (e.g., carbon-calculator)",
    ),
    pack: Optional[str] = typer.Option(
        None,
        "--pack", "-p",
        help="Pack to add agent to (e.g., eudr, sb253)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory (default: ./agents/<name>)",
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        help="Enable interactive wizard",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Overwrite existing files",
    ),
):
    """
    Create a new agent from a template.

    Templates available:
    - calculator: Emission calculation agents
    - validator: Data validation agents
    - reporter: Report generation agents
    - regulatory: Compliance checking agents
    - custom: Minimal starter template

    Example:
        gl agent new calculator --name carbon-calc --pack eudr
        gl agent new regulatory --name csrd-validator
    """
    from .create_command import create_agent_from_template

    ctx = typer.Context
    create_agent_from_template(
        template=template,
        name=name,
        pack=pack,
        output_dir=output_dir,
        interactive=interactive,
        force=force,
    )


@app.command("validate")
def validate_agent(
    spec_path: Path = typer.Argument(
        ...,
        help="Path to AgentSpec YAML file or agent directory",
        exists=True,
    ),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        help="Enable strict validation mode",
    ),
    schema_version: str = typer.Option(
        "1.0",
        "--schema-version",
        help="AgentSpec schema version",
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Attempt to auto-fix issues",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output validation report to file",
    ),
):
    """
    Validate an agent specification or implementation.

    Checks:
    - AgentSpec schema compliance
    - Data contract validation
    - Safety constraint verification
    - Explainability coverage
    - Tool dependency resolution
    - Standards reference verification

    Example:
        gl agent validate pack.yaml
        gl agent validate ./agents/eudr --strict
        gl agent validate spec.yaml --fix --output report.json
    """
    from .validate_command import validate_spec

    validate_spec(
        spec_path=spec_path,
        strict=strict,
        schema_version=schema_version,
        fix=fix,
        output=output,
    )


@app.command("test")
def test_agent(
    agent_id: str = typer.Argument(
        ...,
        help="Agent ID or path to agent directory",
    ),
    golden: bool = typer.Option(
        False,
        "--golden",
        help="Run golden tests (determinism checks)",
    ),
    integration: bool = typer.Option(
        False,
        "--integration",
        help="Run integration tests",
    ),
    e2e: bool = typer.Option(
        False,
        "--e2e",
        help="Run end-to-end tests",
    ),
    coverage: bool = typer.Option(
        True,
        "--coverage/--no-coverage",
        help="Generate coverage report",
    ),
    parallel: bool = typer.Option(
        True,
        "--parallel/--serial",
        help="Run tests in parallel",
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast", "-x",
        help="Stop on first failure",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output test report to file",
    ),
):
    """
    Run tests for an agent.

    Test types:
    - Unit tests: Basic functionality
    - Golden tests: Determinism verification (100 runs, byte-identical)
    - Integration tests: External system integration
    - E2E tests: Full workflow testing

    Example:
        gl agent test eudr_compliance
        gl agent test ./agents/carbon --golden --coverage
        gl agent test my-agent --e2e --output results.json
    """
    from .test_command import run_agent_tests

    # Default to all tests if none specified
    if not golden and not integration and not e2e:
        run_all = True
    else:
        run_all = False

    run_agent_tests(
        agent_id=agent_id,
        golden=golden or run_all,
        integration=integration or run_all,
        e2e=e2e,
        coverage=coverage,
        parallel=parallel,
        fail_fast=fail_fast,
        output=output,
    )


@app.command("certify")
def certify_agent(
    agent_id: str = typer.Argument(
        ...,
        help="Agent ID or path to agent directory",
    ),
    level: str = typer.Option(
        "gold",
        "--level", "-l",
        help="Certification level target: gold, silver, bronze",
    ),
    dimensions: Optional[str] = typer.Option(
        None,
        "--dimensions", "-d",
        help="Specific dimensions to evaluate (comma-separated, e.g., D01,D02,D03)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output certification report",
    ),
    upload: bool = typer.Option(
        False,
        "--upload",
        help="Upload to registry on success",
    ),
):
    """
    Certify an agent against the 12-dimension framework.

    Dimensions:
    - D01: Determinism (100 runs, byte-identical)
    - D02: Provenance (SHA-256 hash tracking)
    - D03: Zero-Hallucination (no LLM in calculations)
    - D04: Accuracy (golden test pass rate)
    - D05: Source Verification (traceable factors)
    - D06: Unit Consistency (input/output validation)
    - D07: Regulatory Compliance (GHG Protocol, ISO 14064)
    - D08: Security (secrets, injection prevention)
    - D09: Performance (response time, memory)
    - D10: Documentation (docstrings, API docs)
    - D11: Test Coverage (>90% target)
    - D12: Production Readiness (logging, health checks)

    Levels:
    - GOLD: 100% score, all required dimensions pass
    - SILVER: 95%+ score, all required dimensions pass
    - BRONZE: 85%+ score, all required dimensions pass

    Example:
        gl agent certify eudr_compliance --level gold
        gl agent certify ./agents/carbon --dimensions D01,D02,D03
        gl agent certify my-agent --output cert.pdf --upload
    """
    from .certify_command import certify_agent_impl

    dim_list = None
    if dimensions:
        dim_list = [d.strip().upper() for d in dimensions.split(",")]

    certify_agent_impl(
        agent_id=agent_id,
        level=level.lower(),
        dimensions=dim_list,
        output=output,
        upload=upload,
    )


@app.command("deploy")
def deploy_agent(
    agent_id: str = typer.Argument(
        ...,
        help="Agent ID to deploy",
    ),
    env: str = typer.Option(
        "dev",
        "--env", "-e",
        help="Target environment: dev, staging, prod",
    ),
    version: Optional[str] = typer.Option(
        None,
        "--version", "-V",
        help="Specific version to deploy",
    ),
    canary: bool = typer.Option(
        False,
        "--canary",
        help="Enable canary deployment (10% traffic)",
    ),
    replicas: int = typer.Option(
        1,
        "--replicas", "-r",
        help="Number of replicas",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Generate manifests without deploying",
    ),
    rollback: bool = typer.Option(
        False,
        "--rollback",
        help="Rollback to previous version",
    ),
):
    """
    Deploy an agent to Kubernetes.

    Environments:
    - dev: Development cluster (auto-deploy on commit)
    - staging: Staging cluster (requires tests pass)
    - prod: Production cluster (requires certification)

    Features:
    - Kubernetes manifest generation
    - Helm chart packaging
    - Environment-specific configs
    - Canary deployment support
    - Rollback capability

    Example:
        gl agent deploy eudr_compliance --env staging
        gl agent deploy my-agent --env prod --canary
        gl agent deploy my-agent --env prod --rollback
        gl agent deploy my-agent --env dev --dry-run
    """
    from .deploy_command import deploy_agent_impl

    deploy_agent_impl(
        agent_id=agent_id,
        env=env.lower(),
        version=version,
        canary=canary,
        replicas=replicas,
        dry_run=dry_run,
        rollback=rollback,
    )


@app.command("list")
def list_agents(
    pack: Optional[str] = typer.Option(
        None,
        "--pack", "-p",
        help="Filter by pack (e.g., eudr, sb253)",
    ),
    status: Optional[str] = typer.Option(
        None,
        "--status", "-s",
        help="Filter by status: draft, active, certified, deprecated",
    ),
    category: Optional[str] = typer.Option(
        None,
        "--category", "-c",
        help="Filter by category: calculator, validator, reporter, regulatory",
    ),
    format: str = typer.Option(
        "table",
        "--format", "-f",
        help="Output format: table, json, yaml",
    ),
    limit: int = typer.Option(
        50,
        "--limit", "-l",
        help="Maximum results to show",
    ),
):
    """
    List agents in the registry or local workspace.

    Example:
        gl agent list
        gl agent list --pack eudr --status certified
        gl agent list --category regulatory --format json
    """
    from .productivity_helpers import list_agents_impl

    list_agents_impl(
        pack=pack,
        status=status,
        category=category,
        format=format,
        limit=limit,
    )


@app.command("diff")
def diff_agents(
    agent_id: str = typer.Argument(
        ...,
        help="Agent ID to compare",
    ),
    version1: str = typer.Argument(
        ...,
        help="First version (e.g., 1.0.0, HEAD~1)",
    ),
    version2: str = typer.Argument(
        "HEAD",
        help="Second version (default: HEAD)",
    ),
    show_code: bool = typer.Option(
        True,
        "--code/--no-code",
        help="Show code diff",
    ),
    show_spec: bool = typer.Option(
        True,
        "--spec/--no-spec",
        help="Show spec diff",
    ),
    show_tests: bool = typer.Option(
        False,
        "--tests",
        help="Show test diff",
    ),
):
    """
    Compare two versions of an agent.

    Shows differences in:
    - AgentSpec (YAML)
    - Implementation code
    - Test coverage
    - Golden test results

    Example:
        gl agent diff eudr_compliance 1.0.0 1.1.0
        gl agent diff ./agents/carbon HEAD~1 HEAD
        gl agent diff my-agent v1 v2 --tests
    """
    console.print(Panel(
        f"[bold cyan]Comparing {agent_id}[/bold cyan]\n"
        f"Version {version1} vs {version2}",
        border_style="cyan"
    ))

    # Placeholder - will be implemented in productivity_helpers
    console.print(f"\n[yellow]Diff functionality coming soon...[/yellow]")
    console.print(f"Agent: {agent_id}")
    console.print(f"Versions: {version1} -> {version2}")
    console.print(f"Show code: {show_code}")
    console.print(f"Show spec: {show_spec}")
    console.print(f"Show tests: {show_tests}")


@app.command("info")
def info_agent(
    agent_id: str = typer.Argument(
        ...,
        help="Agent ID or path",
    ),
    format: str = typer.Option(
        "rich",
        "--format", "-f",
        help="Output format: rich, json, yaml",
    ),
):
    """
    Show detailed information about an agent.

    Displays:
    - Metadata (name, version, author)
    - Tools and capabilities
    - Data contracts (inputs/outputs)
    - Test coverage
    - Certification status
    - Deployment history

    Example:
        gl agent info eudr_compliance
        gl agent info ./agents/carbon --format json
    """
    console.print(Panel(
        f"[bold cyan]Agent Information[/bold cyan]\n"
        f"{agent_id}",
        border_style="cyan"
    ))

    # Create info table
    table = Table(title=f"Agent: {agent_id}")
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    # Sample data - will be loaded from actual agent
    info_data = {
        "ID": agent_id,
        "Name": agent_id.replace("_", " ").replace("-", " ").title(),
        "Version": "1.0.0",
        "Status": "active",
        "Category": "regulatory",
        "Pack": "eudr",
        "Author": "GreenLang Team",
        "License": "Apache-2.0",
        "Tools": "5",
        "Golden Tests": "25",
        "Coverage": "94%",
        "Certification": "GOLD",
        "Last Updated": datetime.now().strftime("%Y-%m-%d"),
    }

    for key, value in info_data.items():
        if key == "Certification":
            value = f"[green]{value}[/green]"
        elif key == "Coverage":
            value = f"[green]{value}[/green]"
        table.add_row(key, value)

    console.print(table)


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point for the Agent Factory CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.exception("CLI error")
        raise typer.Exit(1)


if __name__ == "__main__":
    main()
