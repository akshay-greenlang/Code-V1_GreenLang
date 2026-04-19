#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gl agent create - Generate agents from AgentSpec YAML.

This module provides the CLI interface for generating GreenLang agents
from AgentSpec v2 YAML specifications using the Agent Generator.

Commands:
    gl agent create --spec <path>  - Generate agent from spec file
    gl agent create --spec <path> --output <dir>  - Generate to specific directory

Example:
    $ gl agent create --spec specs/fuel_analyzer.yaml
    $ gl agent create --spec specs/fuel_analyzer.yaml --output ./my-agents/
    $ gl agent create --spec specs/fuel_analyzer.yaml --no-tests --no-readme

Author: GreenLang Framework Team
Date: December 2025
Spec: Agent Generator CLI
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.syntax import Syntax
from rich.tree import Tree

# Import generator components
from greenlang.generator.spec_parser import AgentSpecParser, ParsedAgentSpec, ValidationError
from greenlang.generator.code_generator import CodeGenerator, GenerationOptions, GeneratedCode

# =============================================================================
# CLI Application
# =============================================================================

app = typer.Typer(
    name="agent",
    help="Agent management commands - create, validate, and manage agents",
    no_args_is_help=True,
)

console = Console()
logger = logging.getLogger(__name__)


# =============================================================================
# Commands
# =============================================================================

@app.command("create")
def create_agent(
    spec: Path = typer.Option(
        ...,
        "--spec", "-s",
        help="Path to AgentSpec YAML file (pack.yaml)",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for generated code (default: ./generated/<agent-id>/)",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing files if output directory exists",
    ),
    generate_tests: bool = typer.Option(
        True,
        "--tests/--no-tests",
        help="Generate test suite (default: yes)",
    ),
    generate_readme: bool = typer.Option(
        True,
        "--readme/--no-readme",
        help="Generate README documentation (default: yes)",
    ),
    use_async: bool = typer.Option(
        True,
        "--async/--sync",
        help="Generate async agent methods (default: async)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed generation logs",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be generated without writing files",
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Preview generated code in console",
    ),
):
    """
    Generate a GreenLang agent from an AgentSpec YAML file.

    This command parses the AgentSpec, validates it, and generates a complete
    agent package including:

    - agent.py: Main agent class with lifecycle hooks
    - tools.py: Tool wrapper classes (zero-hallucination)
    - tests/test_agent.py: Comprehensive test suite
    - README.md: Documentation
    - __init__.py: Package initialization

    Examples:

        # Generate with defaults
        gl agent create --spec specs/fuel_analyzer.yaml

        # Generate to specific directory
        gl agent create --spec specs/my_agent.yaml --output ./agents/my_agent

        # Generate without tests (faster)
        gl agent create --spec specs/my_agent.yaml --no-tests

        # Preview generated code without writing
        gl agent create --spec specs/my_agent.yaml --dry-run --preview

        # Verbose output for debugging
        gl agent create --spec specs/my_agent.yaml -v
    """
    # Setup logging
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        logging.basicConfig(level=logging.WARNING)

    # Print header
    console.print()
    console.print(Panel.fit(
        "[bold cyan]GreenLang Agent Generator[/bold cyan]\n"
        "[dim]Generate production-ready agents from AgentSpec YAML[/dim]",
        border_style="cyan"
    ))
    console.print()

    # ==========================================================================
    # Step 1: Parse AgentSpec
    # ==========================================================================
    console.print("[bold cyan]Step 1:[/bold cyan] Parsing AgentSpec...")
    console.print(f"  [dim]File:[/dim] {spec}")

    try:
        parser = AgentSpecParser(strict=True)
        parsed_spec = parser.parse(spec)

        # Show parse warnings if any
        if parser.warnings:
            console.print()
            console.print("[yellow]Warnings:[/yellow]")
            for warning in parser.warnings:
                console.print(f"  [yellow]![/yellow] {warning}")

        console.print(f"  [green]OK[/green] Parsed: [bold]{parsed_spec.name}[/bold] v{parsed_spec.version}")
        console.print(f"  [dim]ID:[/dim] {parsed_spec.id}")
        console.print(f"  [dim]Tools:[/dim] {len(parsed_spec.tools)}")
        console.print(f"  [dim]Inputs:[/dim] {len(parsed_spec.inputs)}")
        console.print(f"  [dim]Outputs:[/dim] {len(parsed_spec.outputs)}")

        if parsed_spec.tests:
            console.print(f"  [dim]Golden Tests:[/dim] {len(parsed_spec.tests.golden)}")
            console.print(f"  [dim]Property Tests:[/dim] {len(parsed_spec.tests.properties)}")

    except ValidationError as e:
        console.print(f"  [red]FAIL[/red] Validation error: {e.message}")
        if e.errors:
            for error in e.errors:
                console.print(f"    [red]-[/red] {error}")
        raise typer.Exit(1)

    except FileNotFoundError as e:
        console.print(f"  [red]FAIL[/red] File not found: {e}")
        raise typer.Exit(1)

    except Exception as e:
        console.print(f"  [red]FAIL[/red] Parse error: {e}")
        if verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)

    console.print()

    # ==========================================================================
    # Step 2: Determine Output Directory
    # ==========================================================================
    console.print("[bold cyan]Step 2:[/bold cyan] Configuring output...")

    if output is None:
        # Default to ./generated/<module_name>/
        output = Path.cwd() / "generated" / parsed_spec.module_name

    console.print(f"  [dim]Output directory:[/dim] {output}")

    # Check if output exists
    if output.exists() and not overwrite and not dry_run:
        console.print(f"  [red]FAIL[/red] Output directory already exists: {output}")
        console.print("  [dim]Use --overwrite to replace existing files[/dim]")
        raise typer.Exit(1)

    console.print()

    # ==========================================================================
    # Step 3: Show Generation Plan
    # ==========================================================================
    console.print("[bold cyan]Step 3:[/bold cyan] Generation plan...")
    console.print()

    plan_table = Table(title="Generation Plan", show_header=True, header_style="bold cyan")
    plan_table.add_column("Component", style="cyan")
    plan_table.add_column("File", style="white")
    plan_table.add_column("Status", style="white")

    plan_table.add_row("Agent Class", "agent.py", "[green]Enabled[/green]")
    plan_table.add_row("Tools", "tools.py", "[green]Enabled[/green]")
    plan_table.add_row("Tests", "tests/test_agent.py",
                       "[green]Enabled[/green]" if generate_tests else "[dim]Disabled[/dim]")
    plan_table.add_row("README", "README.md",
                       "[green]Enabled[/green]" if generate_readme else "[dim]Disabled[/dim]")
    plan_table.add_row("Package Init", "__init__.py", "[green]Enabled[/green]")

    console.print(plan_table)
    console.print()

    # Show spec summary
    spec_tree = Tree("[bold]AgentSpec Summary[/bold]")

    # Inputs branch
    inputs_branch = spec_tree.add("[cyan]Inputs[/cyan]")
    for inp in parsed_spec.inputs:
        req = "[red]*[/red]" if inp.required else ""
        inputs_branch.add(f"{inp.name}: {inp.python_type} {req}")

    # Outputs branch
    outputs_branch = spec_tree.add("[cyan]Outputs[/cyan]")
    for out in parsed_spec.outputs:
        outputs_branch.add(f"{out.name}: {out.python_type}")

    # Tools branch
    tools_branch = spec_tree.add("[cyan]Tools[/cyan]")
    for tool in parsed_spec.tools:
        safe_icon = "[green]SAFE[/green]" if tool.safe else "[yellow]UNSAFE[/yellow]"
        tools_branch.add(f"{tool.name} {safe_icon}")

    console.print(spec_tree)
    console.print()

    # Confirm if not dry run
    if not dry_run:
        if not typer.confirm("Proceed with generation?"):
            console.print("[yellow]Generation cancelled.[/yellow]")
            raise typer.Exit(0)
        console.print()

    # ==========================================================================
    # Step 4: Generate Code
    # ==========================================================================
    console.print("[bold cyan]Step 4:[/bold cyan] Generating code...")

    # Configure generator
    options = GenerationOptions(
        output_dir=output if not dry_run else None,
        overwrite=overwrite,
        generate_tests=generate_tests,
        generate_readme=generate_readme,
        generate_init=True,
        use_async=use_async,
    )

    generator = CodeGenerator(options=options)

    # Generate with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating...", total=None)

        try:
            result = generator.generate(
                spec=parsed_spec,
                output_dir=output if not dry_run else None,
            )
            progress.update(task, completed=100)

        except Exception as e:
            progress.stop()
            console.print(f"  [red]FAIL[/red] Generation error: {e}")
            if verbose:
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
            raise typer.Exit(1)

    console.print()

    # ==========================================================================
    # Step 5: Show Results
    # ==========================================================================
    if dry_run:
        console.print(Panel.fit(
            "[bold yellow]DRY RUN[/bold yellow] - No files written",
            border_style="yellow"
        ))
    else:
        console.print(Panel.fit(
            "[bold green]Generation Complete![/bold green]",
            border_style="green"
        ))

    console.print()

    # Stats table
    stats_table = Table(title="Generation Statistics", show_header=True, header_style="bold green")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")

    stats_table.add_row("Agent Name", result.spec_name)
    stats_table.add_row("Version", result.spec_version)
    stats_table.add_row("Total Lines", str(result.total_lines))
    stats_table.add_row("Tools Generated", str(result.num_tools))
    stats_table.add_row("Tests Generated", str(result.num_tests))
    stats_table.add_row("Files", str(len(result.files)))

    console.print(stats_table)
    console.print()

    # Show generated files
    if not dry_run:
        console.print("[bold cyan]Generated Files:[/bold cyan]")
        for file in result.files:
            file_path = output / file.relative_path / file.filename if file.relative_path else output / file.filename
            console.print(f"  [green]OK[/green] {file_path}")
        console.print()

    # Preview code if requested
    if preview:
        console.print("[bold cyan]Code Preview:[/bold cyan]")
        console.print()

        # Preview agent code (first 50 lines)
        console.print("[bold]agent.py[/bold]")
        agent_preview = "\n".join(result.agent_code.split("\n")[:50])
        syntax = Syntax(agent_preview, "python", theme="monokai", line_numbers=True)
        console.print(syntax)
        console.print("[dim]... (truncated)[/dim]")
        console.print()

    # Next steps
    if not dry_run:
        console.print("[bold cyan]Next Steps:[/bold cyan]")
        console.print()
        console.print(f"  1. [yellow]cd {output}[/yellow]")
        console.print(f"  2. [yellow]pip install -e .[/yellow]  # If setup.py exists")

        if generate_tests:
            console.print(f"  3. [yellow]pytest tests/[/yellow]  # Run tests")

        console.print(f"  4. Review and customize the generated code")
        console.print(f"  5. Implement tool logic in tools.py")
        console.print()
        console.print("[green]Agent ready for development![/green]")


@app.command("validate")
def validate_spec(
    spec: Path = typer.Option(
        ...,
        "--spec", "-s",
        help="Path to AgentSpec YAML file to validate",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        help="Fail on warnings (default: yes)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed validation results",
    ),
):
    """
    Validate an AgentSpec YAML file without generating code.

    This command parses and validates the AgentSpec, reporting any
    errors or warnings found.

    Examples:

        # Validate a spec file
        gl agent validate --spec specs/fuel_analyzer.yaml

        # Validate with detailed output
        gl agent validate --spec specs/my_agent.yaml -v

        # Validate without failing on warnings
        gl agent validate --spec specs/my_agent.yaml --no-strict
    """
    console.print()
    console.print(Panel.fit(
        "[bold cyan]AgentSpec Validator[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    console.print(f"[cyan]Validating:[/cyan] {spec}")
    console.print()

    try:
        parser = AgentSpecParser(strict=strict)
        parsed_spec = parser.parse(spec)

        # Show validation results
        if parser.errors:
            console.print("[red]Errors:[/red]")
            for error in parser.errors:
                console.print(f"  [red]X[/red] {error}")
            console.print()

        if parser.warnings:
            console.print("[yellow]Warnings:[/yellow]")
            for warning in parser.warnings:
                console.print(f"  [yellow]![/yellow] {warning}")
            console.print()

        if not parser.errors and not parser.warnings:
            console.print("[green]OK[/green] No errors or warnings found.")
            console.print()

        # Show spec details if verbose
        if verbose:
            console.print("[bold]Spec Details:[/bold]")
            console.print(f"  Name: {parsed_spec.name}")
            console.print(f"  Version: {parsed_spec.version}")
            console.print(f"  ID: {parsed_spec.id}")
            console.print(f"  Inputs: {len(parsed_spec.inputs)}")
            console.print(f"  Outputs: {len(parsed_spec.outputs)}")
            console.print(f"  Tools: {len(parsed_spec.tools)}")
            console.print(f"  Has Provenance: {parsed_spec.has_provenance}")
            console.print(f"  Has Tests: {parsed_spec.has_tests}")
            console.print()

        # Exit status
        if parser.errors:
            console.print("[red]FAIL[/red] Validation failed with errors.")
            raise typer.Exit(1)
        elif strict and parser.warnings:
            console.print("[yellow]WARN[/yellow] Validation passed with warnings.")
            raise typer.Exit(1)
        else:
            console.print("[green]PASS[/green] Validation successful!")

    except ValidationError as e:
        console.print(f"[red]FAIL[/red] {e}")
        raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]FAIL[/red] Validation error: {e}")
        raise typer.Exit(1)


@app.command("info")
def show_spec_info(
    spec: Path = typer.Option(
        ...,
        "--spec", "-s",
        help="Path to AgentSpec YAML file",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
):
    """
    Display detailed information about an AgentSpec.

    Examples:

        gl agent info --spec specs/fuel_analyzer.yaml
    """
    console.print()

    try:
        parser = AgentSpecParser(strict=False)
        parsed_spec = parser.parse(spec)

        # Header
        console.print(Panel.fit(
            f"[bold]{parsed_spec.name}[/bold]\n"
            f"[dim]v{parsed_spec.version}[/dim]",
            border_style="cyan"
        ))
        console.print()

        # Basic info
        console.print("[bold cyan]Basic Information[/bold cyan]")
        console.print(f"  ID: {parsed_spec.id}")
        console.print(f"  Kind: {parsed_spec.kind.value}")
        console.print(f"  License: {parsed_spec.license}")

        if parsed_spec.summary:
            console.print(f"  Summary: {parsed_spec.summary}")

        if parsed_spec.author:
            console.print(f"  Author: {parsed_spec.author.name}")
            if parsed_spec.author.email:
                console.print(f"  Email: {parsed_spec.author.email}")

        if parsed_spec.tags:
            console.print(f"  Tags: {', '.join(parsed_spec.tags)}")

        console.print()

        # Inputs
        console.print("[bold cyan]Inputs[/bold cyan]")
        if parsed_spec.inputs:
            for inp in parsed_spec.inputs:
                req = " (required)" if inp.required else " (optional)"
                console.print(f"  - {inp.name}: {inp.python_type}{req}")
                if inp.description:
                    console.print(f"    [dim]{inp.description}[/dim]")
        else:
            console.print("  [dim]No inputs defined[/dim]")
        console.print()

        # Outputs
        console.print("[bold cyan]Outputs[/bold cyan]")
        if parsed_spec.outputs:
            for out in parsed_spec.outputs:
                console.print(f"  - {out.name}: {out.python_type}")
                if out.description:
                    console.print(f"    [dim]{out.description}[/dim]")
        else:
            console.print("  [dim]No outputs defined[/dim]")
        console.print()

        # Tools
        console.print("[bold cyan]Tools[/bold cyan]")
        if parsed_spec.tools:
            for tool in parsed_spec.tools:
                safe = "[green]SAFE[/green]" if tool.safe else "[yellow]UNSAFE[/yellow]"
                console.print(f"  - {tool.name} {safe}")
                console.print(f"    [dim]{tool.description}[/dim]")
                console.print(f"    [dim]Impl: {tool.impl}[/dim]")
        else:
            console.print("  [dim]No tools defined[/dim]")
        console.print()

        # Provenance
        if parsed_spec.provenance:
            console.print("[bold cyan]Provenance[/bold cyan]")
            console.print(f"  GWP Set: {parsed_spec.provenance.gwp_set.value}")
            console.print(f"  Pin EF: {parsed_spec.provenance.pin_ef}")
            console.print(f"  Record: {', '.join(parsed_spec.provenance.record)}")
            console.print()

        # Tests
        if parsed_spec.tests:
            console.print("[bold cyan]Tests[/bold cyan]")
            console.print(f"  Golden Tests: {len(parsed_spec.tests.golden)}")
            for test in parsed_spec.tests.golden:
                console.print(f"    - {test.name}")
            console.print(f"  Property Tests: {len(parsed_spec.tests.properties)}")
            for prop in parsed_spec.tests.properties:
                console.print(f"    - {prop.name}: {prop.rule}")
            console.print()

        # Source info
        console.print("[bold cyan]Source[/bold cyan]")
        console.print(f"  File: {parsed_spec.source_file}")
        console.print(f"  Hash: {parsed_spec.source_hash[:16]}...")
        console.print(f"  Parsed: {parsed_spec.parsed_at.isoformat()}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the agent CLI."""
    app()


if __name__ == "__main__":
    main()
