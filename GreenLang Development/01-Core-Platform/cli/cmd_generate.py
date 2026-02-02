# -*- coding: utf-8 -*-
"""
gl generate - LLM-powered agent code generation

This module provides the CLI interface for generating GreenLang agents
from AgentSpec specifications using the AgentFactory.

Commands:
- gl generate agent <spec.yaml> - Generate agent from spec file

Author: GreenLang Framework Team
Date: October 2025
Spec: FRMW-203 (Agent Factory CLI Integration)
"""

import typer
import asyncio
import logging
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.syntax import Syntax

from greenlang.factory.agent_factory import AgentFactory, GenerationResult
from greenlang.specs import agent_from_yaml, agent_from_json, AgentSpecV2

app = typer.Typer(
    help="Generate agents using LLM-powered code generation",
    no_args_is_help=True
)
console = Console()
logger = logging.getLogger(__name__)


@app.command()
def agent(
    spec_file: Path = typer.Argument(
        ...,
        help="AgentSpec YAML/JSON file to generate from",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output_dir: Path = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for generated code (default: ./generated/<agent-id>)"
    ),
    budget: float = typer.Option(
        5.0,
        "--budget", "-b",
        help="Max cost in USD per agent (default: $5.00)",
        min=0.1,
        max=50.0,
    ),
    max_attempts: int = typer.Option(
        3,
        "--max-attempts",
        help="Max refinement attempts (default: 3)",
        min=1,
        max=10,
    ),
    skip_tests: bool = typer.Option(
        False,
        "--skip-tests",
        help="Skip test generation"
    ),
    skip_docs: bool = typer.Option(
        False,
        "--skip-docs",
        help="Skip documentation generation"
    ),
    skip_demo: bool = typer.Option(
        False,
        "--skip-demo",
        help="Skip demo script generation"
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip code validation (not recommended)"
    ),
    reference_agents: Optional[Path] = typer.Option(
        None,
        "--reference-agents",
        help="Path to reference agents directory (default: greenlang/agents/)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed generation logs"
    ),
):
    """
    Generate a GreenLang agent from AgentSpec using LLM-powered code generation.

    This command uses the AgentFactory to generate complete agent packages including:
    - Tool implementations (deterministic calculations)
    - Agent class (AI orchestration)
    - Test suite (unit + integration)
    - Documentation (README, API reference)
    - Demo script (interactive examples)

    The generation process uses GPT-4 with pattern extraction from reference agents
    and includes comprehensive validation with iterative refinement.

    Examples:
        # Generate agent with defaults
        gl generate agent specs/boiler-efficiency.yaml

        # Generate with custom output directory
        gl generate agent specs/my-agent.yaml -o ./custom-output

        # Generate with higher budget for complex agents
        gl generate agent specs/complex-agent.yaml --budget 10.0

        # Generate without tests (faster development iteration)
        gl generate agent specs/draft-agent.yaml --skip-tests --skip-docs
    """
    # Enable verbose logging if requested
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    console.print()
    console.print(Panel.fit(
        "[bold cyan]GreenLang Agent Factory[/bold cyan]\n"
        "[dim]LLM-Powered Code Generation System[/dim]",
        border_style="cyan"
    ))
    console.print()

    # Load spec file
    console.print(f"[cyan]Loading spec from:[/cyan] {spec_file}")
    try:
        if spec_file.suffix in [".yaml", ".yml"]:
            spec = agent_from_yaml(spec_file)
        elif spec_file.suffix == ".json":
            spec = agent_from_json(spec_file)
        else:
            console.print(f"[red]Error: Unsupported file format '{spec_file.suffix}'[/red]")
            console.print("Supported formats: .yaml, .yml, .json")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to load spec: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]✓[/green] Loaded spec: [bold]{spec.name}[/bold] (v{spec.version})")
    console.print(f"[dim]  Agent ID: {spec.id}[/dim]")
    console.print()

    # Determine output directory
    if output_dir is None:
        output_dir = Path.cwd() / "generated" / spec.id.split("/")[-1]

    console.print(f"[cyan]Output directory:[/cyan] {output_dir}")
    console.print(f"[cyan]Budget:[/cyan] ${budget:.2f}")
    console.print(f"[cyan]Max refinement attempts:[/cyan] {max_attempts}")
    console.print()

    # Show generation plan
    plan_table = Table(title="Generation Plan", show_header=True, header_style="bold cyan")
    plan_table.add_column("Component", style="cyan")
    plan_table.add_column("Status", style="white")

    plan_table.add_row("Tools", "[green]Enabled[/green]")
    plan_table.add_row("Agent", "[green]Enabled[/green]")
    plan_table.add_row("Tests", "[red]Skipped[/red]" if skip_tests else "[green]Enabled[/green]")
    plan_table.add_row("Docs", "[red]Skipped[/red]" if skip_docs else "[green]Enabled[/green]")
    plan_table.add_row("Demo", "[red]Skipped[/red]" if skip_demo else "[green]Enabled[/green]")
    plan_table.add_row("Validation", "[red]Skipped[/red]" if skip_validation else "[green]Enabled[/green]")

    console.print(plan_table)
    console.print()

    # Confirm with user
    if not typer.confirm("Proceed with generation?"):
        console.print("[yellow]Generation cancelled[/yellow]")
        raise typer.Exit(0)

    console.print()

    # Initialize factory
    console.print("[cyan]Initializing Agent Factory...[/cyan]")
    factory = AgentFactory(
        budget_per_agent_usd=budget,
        max_refinement_attempts=max_attempts,
        enable_validation=not skip_validation,
        enable_determinism_check=not skip_validation,
        reference_agents_path=reference_agents,
        output_path=output_dir,
    )
    console.print("[green]✓[/green] Factory initialized")
    console.print()

    # Run generation with progress indicator
    console.print("[bold cyan]Generating agent...[/bold cyan]")
    console.print()

    result: GenerationResult = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating...", total=None)

        # Run async generation
        try:
            result = asyncio.run(
                factory.generate_agent(
                    spec,
                    skip_tests=skip_tests,
                    skip_docs=skip_docs,
                    skip_demo=skip_demo,
                )
            )
        except Exception as e:
            progress.stop()
            console.print()
            console.print(f"[red]Generation failed: {e}[/red]")
            if verbose:
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
            raise typer.Exit(1)

        progress.update(task, completed=100)

    console.print()

    # Display results
    if result.success:
        console.print(Panel.fit(
            "[bold green]✓ Agent Generated Successfully![/bold green]",
            border_style="green"
        ))
        console.print()

        # Stats table
        stats_table = Table(title="Generation Statistics", show_header=True, header_style="bold green")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")

        stats_table.add_row("Duration", f"{result.duration_seconds:.1f}s")
        stats_table.add_row("Cost", f"${result.total_cost_usd:.4f}")
        stats_table.add_row("Refinement Attempts", str(result.attempts))

        if result.validation_result:
            validation_status = "[green]PASSED[/green]" if result.validation_result.passed else "[red]FAILED[/red]"
            stats_table.add_row("Validation", validation_status)

        console.print(stats_table)
        console.print()

        # Show generated files
        console.print("[bold cyan]Generated Files:[/bold cyan]")
        console.print()
        console.print(f"  [cyan]Agent:[/cyan] {output_dir}/agent.py")
        if result.test_code and not skip_tests:
            console.print(f"  [cyan]Tests:[/cyan] {output_dir}/tests/test_agent.py")
        if result.docs and not skip_docs:
            console.print(f"  [cyan]Docs:[/cyan] {output_dir}/README.md")
        if result.demo_script and not skip_demo:
            console.print(f"  [cyan]Demo:[/cyan] {output_dir}/demo.py")
        console.print()

        # Show validation details if available
        if result.validation_result and verbose:
            console.print("[bold cyan]Validation Details:[/bold cyan]")
            console.print()

            if result.validation_result.syntax_errors:
                console.print("[red]Syntax Errors:[/red]")
                for error in result.validation_result.syntax_errors:
                    console.print(f"  • {error}")
                console.print()

            if result.validation_result.type_errors:
                console.print("[yellow]Type Errors:[/yellow]")
                for error in result.validation_result.type_errors:
                    console.print(f"  • {error}")
                console.print()

            if result.validation_result.lint_warnings:
                console.print("[yellow]Lint Warnings:[/yellow]")
                for warning in result.validation_result.lint_warnings[:5]:  # Show first 5
                    console.print(f"  • {warning}")
                if len(result.validation_result.lint_warnings) > 5:
                    console.print(f"  [dim]... and {len(result.validation_result.lint_warnings) - 5} more[/dim]")
                console.print()

        # Show provenance if available
        if result.provenance and verbose:
            console.print("[bold cyan]Provenance:[/bold cyan]")
            console.print()
            console.print(f"  [cyan]Timestamp:[/cyan] {result.provenance.get('timestamp', 'N/A')}")
            console.print(f"  [cyan]LLM Model:[/cyan] {result.provenance.get('llm_model', 'N/A')}")
            console.print(f"  [cyan]Factory Version:[/cyan] {result.provenance.get('factory_version', 'N/A')}")
            console.print()

        # Next steps
        console.print("[bold cyan]Next Steps:[/bold cyan]")
        console.print()
        console.print(f"  1. [yellow]cd {output_dir}[/yellow]")
        console.print(f"  2. [yellow]pip install -e .[/yellow]  # Install agent package")

        if not skip_tests:
            console.print(f"  3. [yellow]pytest[/yellow]  # Run tests")

        console.print(f"  4. [yellow]python demo.py[/yellow]  # Try the demo")
        console.print(f"  5. Review and customize generated code")
        console.print()

        console.print("[green]Agent ready for development![/green]")

    else:
        console.print(Panel.fit(
            "[bold red]✗ Generation Failed[/bold red]",
            border_style="red"
        ))
        console.print()
        console.print(f"[red]Error:[/red] {result.error}")
        console.print()

        if result.validation_result:
            console.print("[yellow]Validation Issues:[/yellow]")
            console.print()

            if result.validation_result.syntax_errors:
                console.print("[red]Syntax Errors:[/red]")
                for error in result.validation_result.syntax_errors:
                    console.print(f"  • {error}")

            if result.validation_result.type_errors:
                console.print("[red]Type Errors:[/red]")
                for error in result.validation_result.type_errors:
                    console.print(f"  • {error}")

            if result.validation_result.test_failures:
                console.print("[red]Test Failures:[/red]")
                for failure in result.validation_result.test_failures:
                    console.print(f"  • {failure}")

        console.print()
        console.print("[yellow]Suggestions:[/yellow]")
        console.print("  • Try increasing the budget (--budget 10.0)")
        console.print("  • Increase max refinement attempts (--max-attempts 5)")
        console.print("  • Simplify the AgentSpec specification")
        console.print("  • Check spec file for errors")

        raise typer.Exit(1)


if __name__ == "__main__":
    app()
