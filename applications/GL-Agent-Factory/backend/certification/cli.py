"""
Certification CLI Commands

This module provides CLI commands for agent certification.

Usage:
    gl agent certify <agent_path>
    gl agent certify <agent_path> --output report.pdf
    gl agent certify <agent_path> --dimension D01

Example:
    >>> python -m backend.certification.cli certify path/to/agent
"""

import sys
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    typer = None

from .engine import CertificationEngine, CertificationResult, CertificationLevel
from .report import ReportGenerator
from .dimensions.base import DimensionStatus


# Console for rich output
console = Console() if HAS_RICH else None


def create_app():
    """Create Typer app for certification commands."""
    if not HAS_RICH:
        return None

    app = typer.Typer(
        name="certify",
        help="Agent certification commands",
        no_args_is_help=True,
    )

    @app.command()
    def certify(
        agent_path: Path = typer.Argument(
            ...,
            help="Path to agent directory",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
        output: Optional[Path] = typer.Option(
            None,
            "--output",
            "-o",
            help="Output report file (supports .pdf, .html, .json, .md)",
        ),
        dimension: Optional[str] = typer.Option(
            None,
            "--dimension",
            "-d",
            help="Evaluate single dimension (e.g., D01, D02)",
        ),
        verbose: bool = typer.Option(
            True,
            "--verbose/--quiet",
            "-v/-q",
            help="Verbose output",
        ),
        json_output: bool = typer.Option(
            False,
            "--json",
            help="Output results as JSON",
        ),
    ):
        """
        Certify an agent against the 12-dimension framework.

        This command evaluates an agent across all certification dimensions
        and produces a comprehensive certification report.

        Examples:
            gl agent certify agents/my_agent
            gl agent certify agents/my_agent --output report.pdf
            gl agent certify agents/my_agent --dimension D01
        """
        try:
            engine = CertificationEngine()

            # Single dimension evaluation
            if dimension:
                _evaluate_single_dimension(engine, agent_path, dimension, verbose)
                return

            # Full certification
            if verbose and not json_output:
                _print_header(agent_path)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                disable=json_output,
            ) as progress:
                task = progress.add_task("Running certification...", total=None)

                result = engine.evaluate_agent(
                    agent_path,
                    verbose=False,  # We handle our own progress
                )

                progress.update(task, completed=True)

            # Output results
            if json_output:
                import json
                print(json.dumps(result.to_dict(), indent=2, default=str))
            else:
                _print_results(result)

            # Generate report if requested
            if output:
                _generate_report(result, output)

            # Exit with appropriate code
            if not result.certified:
                raise typer.Exit(1)

        except Exception as e:
            if json_output:
                import json
                print(json.dumps({"error": str(e)}))
            else:
                console.print(f"[red]Error: {str(e)}[/red]")
            raise typer.Exit(1)

    @app.command()
    def dimensions():
        """
        List all certification dimensions.

        Displays information about each of the 12 certification dimensions.
        """
        engine = CertificationEngine()
        dims = engine.get_dimension_info()

        table = Table(title="Certification Dimensions")
        table.add_column("ID", style="cyan", width=6)
        table.add_column("Name", style="green", width=25)
        table.add_column("Weight", justify="right", width=8)
        table.add_column("Required", justify="center", width=10)
        table.add_column("Description", width=45)

        for dim in dims:
            required = "[green]Yes[/green]" if dim["required"] else "[yellow]No[/yellow]"
            table.add_row(
                dim["id"],
                dim["name"],
                f"{dim['weight']:.1f}",
                required,
                dim["description"][:45] + "..." if len(dim["description"]) > 45 else dim["description"],
            )

        console.print(table)

    @app.command()
    def report(
        agent_path: Path = typer.Argument(
            ...,
            help="Path to agent directory",
            exists=True,
        ),
        output: Path = typer.Argument(
            ...,
            help="Output report file",
        ),
        format: str = typer.Option(
            "auto",
            "--format",
            "-f",
            help="Report format (pdf, html, json, md, auto)",
        ),
    ):
        """
        Generate a certification report for an agent.

        Runs certification and generates a report in the specified format.

        Examples:
            gl agent certify report agents/my_agent report.pdf
            gl agent certify report agents/my_agent report.html --format html
        """
        engine = CertificationEngine()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running certification...", total=None)
            result = engine.evaluate_agent(agent_path, verbose=False)
            progress.update(task, completed=True)

        _generate_report(result, output, format)

        _print_results(result)

    return app


def _print_header(agent_path: Path) -> None:
    """Print certification header."""
    console.print()
    console.print(Panel.fit(
        "[bold green]GreenLang Agent Certification[/bold green]\n"
        "[dim]12-Dimension Evaluation Framework[/dim]",
        border_style="green",
    ))
    console.print(f"Agent: [cyan]{agent_path}[/cyan]")
    console.print()


def _print_results(result: CertificationResult) -> None:
    """Print certification results."""
    # Summary panel
    status_color = "green" if result.certified else "red"
    status_text = "CERTIFIED" if result.certified else "NOT CERTIFIED"

    summary = f"""
[bold]Agent ID:[/bold] {result.agent_id}
[bold]Version:[/bold] {result.agent_version}
[bold]Certification ID:[/bold] {result.certification_id}

[bold]Status:[/bold] [{status_color}]{status_text}[/{status_color}]
[bold]Level:[/bold] [{status_color}]{result.level.value}[/{status_color}]
[bold]Overall Score:[/bold] {result.overall_score:.1f}/100
[bold]Weighted Score:[/bold] {result.weighted_score:.1f}/100
[bold]Dimensions:[/bold] {result.dimensions_passed}/{result.dimensions_total} passed
"""

    console.print(Panel(summary.strip(), title="Summary", border_style=status_color))

    # Dimension results table
    table = Table(title="Dimension Results")
    table.add_column("ID", style="cyan", width=6)
    table.add_column("Dimension", width=25)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Checks", justify="right", width=10)

    for dim in result.dimension_results:
        if dim.status == DimensionStatus.PASS:
            status = "[green]PASS[/green]"
        elif dim.status == DimensionStatus.FAIL:
            status = "[red]FAIL[/red]"
        elif dim.status == DimensionStatus.WARNING:
            status = "[yellow]WARN[/yellow]"
        else:
            status = "[dim]N/A[/dim]"

        table.add_row(
            dim.dimension_id,
            dim.dimension_name,
            status,
            f"{dim.score:.1f}",
            f"{dim.checks_passed}/{dim.checks_total}",
        )

    console.print(table)

    # Remediation if needed
    if result.remediation_summary:
        console.print()
        console.print("[bold yellow]Remediation Required:[/bold yellow]")
        for dimension, suggestions in result.remediation_summary.items():
            console.print(f"\n  [cyan]{dimension}[/cyan]")
            for suggestion in suggestions[:2]:
                console.print(f"    - {suggestion[:80]}...")

    console.print()


def _evaluate_single_dimension(
    engine: CertificationEngine,
    agent_path: Path,
    dimension_id: str,
    verbose: bool,
) -> None:
    """Evaluate a single dimension."""
    try:
        result = engine.evaluate_single_dimension(agent_path, dimension_id.upper())

        status_color = "green" if result.passed else "red"
        status_text = "PASS" if result.passed else "FAIL"

        console.print()
        console.print(Panel(
            f"[bold]{result.dimension_name}[/bold] ({result.dimension_id})\n\n"
            f"Status: [{status_color}]{status_text}[/{status_color}]\n"
            f"Score: {result.score:.1f}/100\n"
            f"Checks: {result.checks_passed}/{result.checks_total} passed\n"
            f"Time: {result.execution_time_ms:.2f}ms",
            title=f"Dimension {result.dimension_id}",
            border_style=status_color,
        ))

        # Show check details
        if verbose and result.check_results:
            console.print("\n[bold]Check Results:[/bold]")
            for check in result.check_results:
                icon = "[green]v[/green]" if check.passed else "[red]x[/red]"
                console.print(f"  {icon} {check.name}: {check.message}")

        # Show remediation
        if result.remediation:
            console.print("\n[bold yellow]Remediation:[/bold yellow]")
            for suggestion in result.remediation[:3]:
                console.print(f"  - {suggestion[:100]}...")

        console.print()

    except ValueError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


def _generate_report(
    result: CertificationResult,
    output: Path,
    format: str = "auto",
) -> None:
    """Generate certification report."""
    generator = ReportGenerator()

    # Determine format
    if format == "auto":
        suffix = output.suffix.lower()
        format_map = {
            ".pdf": "pdf",
            ".html": "html",
            ".json": "json",
            ".md": "md",
            ".markdown": "md",
        }
        format = format_map.get(suffix, "html")

    # Generate report
    if format == "pdf":
        generator.generate_pdf(result, output)
    elif format == "html":
        generator.generate_html(result, output)
    elif format == "json":
        generator.generate_json(result, output)
    elif format == "md":
        generator.generate_markdown(result, output)
    else:
        console.print(f"[yellow]Unknown format '{format}', using HTML[/yellow]")
        generator.generate_html(result, output)

    console.print(f"[green]Report generated: {output}[/green]")


# Create app for import
app = create_app()


def main():
    """Main entry point for CLI."""
    if app is None:
        print("Error: typer and rich are required for CLI.")
        print("Install with: pip install typer rich")
        sys.exit(1)

    app()


if __name__ == "__main__":
    main()
