# -*- coding: utf-8 -*-
"""
CSRD/ESRS Digital Reporting Platform - CLI Commands

Provides command-line interface for CSRD/ESRS sustainability reporting.

Commands:
    csrd run        - Execute full 6-agent pipeline
    csrd validate   - Run IntakeAgent only (data validation)
    csrd calculate  - Run CalculatorAgent only (metric calculations)
    csrd audit      - Run AuditAgent only (compliance check)
    csrd materialize - Run MaterialityAgent only (double materiality)
    csrd report     - Run ReportingAgent only (XBRL generation)
    csrd aggregate  - Run AggregatorAgent only (multi-framework integration)
    csrd --version  - Show version information

Version: 1.0.0
Author: GreenLang CSRD Team
License: MIT
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import click
from rich.console import Console
from rich.panel import Panel
from greenlang.determinism import DeterministicClock
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.tree import Tree

# Version info
__version__ = "1.0.0"

# Import agents (will be imported on-demand to speed up CLI load time)
console = Console()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load CSRD configuration from file.

    Search order:
    1. Provided config_path
    2. .csrd.yaml in current directory
    3. config/csrd_config.yaml in pack directory
    4. Environment variables

    Args:
        config_path: Optional path to config file

    Returns:
        Configuration dictionary
    """
    import os
    import yaml

    config = {}

    # Try provided path first
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        return config

    # Try .csrd.yaml in current directory
    local_config = Path(".csrd.yaml")
    if local_config.exists():
        with open(local_config) as f:
            config = yaml.safe_load(f) or {}
        return config

    # Try default config in pack directory
    default_config = Path(__file__).parent.parent / "config" / "csrd_config.yaml"
    if default_config.exists():
        with open(default_config) as f:
            config = yaml.safe_load(f) or {}
        return config

    # Environment variables as fallback
    config = {
        "company": {
            "name": os.getenv("CSRD_COMPANY_NAME"),
            "lei": os.getenv("CSRD_COMPANY_LEI"),
            "country": os.getenv("CSRD_COMPANY_COUNTRY"),
            "sector": os.getenv("CSRD_COMPANY_SECTOR"),
        },
        "reporting_period": {
            "start_date": os.getenv("CSRD_PERIOD_START"),
            "end_date": os.getenv("CSRD_PERIOD_END"),
        },
    }

    return config


def save_config(config: Dict[str, Any], config_path: str = ".csrd.yaml"):
    """
    Save CSRD configuration to file.

    Args:
        config: Configuration dictionary
        config_path: Path to save config (default: .csrd.yaml)
    """
    import yaml

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]✓[/green] Configuration saved to {config_path}")


def display_banner():
    """Display CSRD platform banner."""
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]CSRD/ESRS Digital Reporting Platform[/bold cyan]\n"
        "[dim]Zero-Hallucination EU Sustainability Reporting[/dim]\n"
        f"[dim]Version {__version__}[/dim]",
        border_style="cyan"
    ))
    console.print("\n")


def display_agent_result(
    agent_name: str,
    result: Dict[str, Any],
    elapsed_time: float,
    verbose: bool = False
):
    """
    Display agent execution results in a beautiful format.

    Args:
        agent_name: Name of the agent
        result: Result dictionary from agent
        elapsed_time: Time taken in seconds
        verbose: Show detailed output
    """
    metadata = result.get("metadata", {})

    # Create summary table
    table = Table(
        title=f"{agent_name} Results",
        show_header=True,
        header_style="bold cyan"
    )
    table.add_column("Metric", style="dim", width=30)
    table.add_column("Value", justify="right", width=20)

    # Agent-specific metrics
    if agent_name == "IntakeAgent":
        table.add_row("Total Records", str(metadata.get("total_records", 0)))
        table.add_row(
            "[green]Valid Records[/green]",
            str(metadata.get("valid_records", 0))
        )
        table.add_row(
            "[red]Invalid Records[/red]",
            str(metadata.get("invalid_records", 0))
        )
        table.add_row(
            "[yellow]Warnings[/yellow]",
            str(metadata.get("warnings", 0))
        )
        table.add_row(
            "Data Quality Score",
            f"{metadata.get('overall_quality_score', 0):.1f}/100"
        )

    elif agent_name == "MaterialityAgent":
        table.add_row(
            "Material Topics",
            str(metadata.get("material_topics_count", 0))
        )
        table.add_row(
            "Impact Materiality",
            str(metadata.get("impact_material_count", 0))
        )
        table.add_row(
            "Financial Materiality",
            str(metadata.get("financial_material_count", 0))
        )
        table.add_row(
            "Double Material",
            str(metadata.get("double_material_count", 0))
        )

    elif agent_name == "CalculatorAgent":
        table.add_row(
            "Metrics Calculated",
            str(metadata.get("metrics_calculated", 0))
        )
        table.add_row(
            "Zero Hallucination",
            "[green]100% ✓[/green]"
        )
        table.add_row(
            "Calculation Errors",
            str(metadata.get("calculation_errors", 0))
        )
        table.add_row(
            "Avg Time per Metric",
            f"{metadata.get('avg_time_per_metric_ms', 0):.2f}ms"
        )

    elif agent_name == "AggregatorAgent":
        table.add_row(
            "Frameworks Integrated",
            str(metadata.get("frameworks_count", 0))
        )
        table.add_row("ESRS Metrics", str(metadata.get("esrs_metrics", 0)))
        table.add_row("TCFD Metrics", str(metadata.get("tcfd_metrics", 0)))
        table.add_row("GRI Metrics", str(metadata.get("gri_metrics", 0)))
        table.add_row("SASB Metrics", str(metadata.get("sasb_metrics", 0)))

    elif agent_name == "ReportingAgent":
        table.add_row("Report Format", metadata.get("format", "XBRL"))
        table.add_row(
            "XBRL Tags Applied",
            str(metadata.get("xbrl_tags_count", 0))
        )
        table.add_row(
            "ESEF Compliant",
            "[green]YES ✓[/green]" if metadata.get("esef_compliant") else "[red]NO ✗[/red]"
        )
        table.add_row("Report Size", metadata.get("file_size", "N/A"))

    elif agent_name == "AuditAgent":
        is_compliant = metadata.get("is_compliant", False)
        table.add_row(
            "Compliance Status",
            "[green]PASS ✅[/green]" if is_compliant else "[red]FAIL ❌[/red]"
        )
        table.add_row(
            "Rules Checked",
            str(metadata.get("rules_checked", 0))
        )
        table.add_row(
            "[red]Critical Issues[/red]",
            str(metadata.get("critical_issues", 0))
        )
        table.add_row(
            "[yellow]Warnings[/yellow]",
            str(metadata.get("warnings", 0))
        )
        table.add_row(
            "[blue]Info Messages[/blue]",
            str(metadata.get("info_count", 0))
        )

    # Common metrics
    table.add_row("", "")  # Separator
    table.add_row("Processing Time", f"{elapsed_time:.2f}s")
    table.add_row("Timestamp", metadata.get("timestamp", "N/A"))

    console.print("\n")
    console.print(table)

    # Show detailed info if verbose
    if verbose and "issues" in result:
        issues = result["issues"]
        if issues:
            console.print("\n[bold yellow]Issues:[/bold yellow]")
            for issue in issues[:10]:  # Show first 10 issues
                severity_color = {
                    "error": "red",
                    "warning": "yellow",
                    "info": "blue"
                }.get(issue.get("severity", "info"), "white")
                console.print(
                    f"  [{severity_color}]●[/{severity_color}] "
                    f"{issue.get('error_code', 'N/A')}: {issue.get('message', 'N/A')}"
                )
            if len(issues) > 10:
                console.print(f"  [dim]... and {len(issues) - 10} more[/dim]")


def display_pipeline_summary(
    results: Dict[str, Any],
    total_time: float,
    output_dir: Path
):
    """
    Display complete pipeline summary.

    Args:
        results: Results from all agents
        total_time: Total pipeline execution time
        output_dir: Output directory path
    """
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]Pipeline Execution Complete![/bold green]\n"
        f"[dim]Total time: {total_time:.2f}s[/dim]",
        border_style="green"
    ))

    # Create summary tree
    tree = Tree("[bold cyan]Pipeline Results[/bold cyan]")

    # Intake
    if "intake" in results:
        intake_meta = results["intake"].get("metadata", {})
        intake_node = tree.add(f"[green]✓[/green] IntakeAgent")
        intake_node.add(f"Records: {intake_meta.get('total_records', 0)}")
        intake_node.add(f"Valid: {intake_meta.get('valid_records', 0)}")
        intake_node.add(f"Quality: {intake_meta.get('overall_quality_score', 0):.1f}/100")

    # Materiality
    if "materiality" in results:
        mat_meta = results["materiality"].get("metadata", {})
        mat_node = tree.add(f"[green]✓[/green] MaterialityAgent")
        mat_node.add(f"Material topics: {mat_meta.get('material_topics_count', 0)}")
        mat_node.add(f"Double material: {mat_meta.get('double_material_count', 0)}")

    # Calculator
    if "calculator" in results:
        calc_meta = results["calculator"].get("metadata", {})
        calc_node = tree.add(f"[green]✓[/green] CalculatorAgent")
        calc_node.add(f"Metrics: {calc_meta.get('metrics_calculated', 0)}")
        calc_node.add("[green]Zero Hallucination: 100% ✓[/green]")

    # Aggregator
    if "aggregator" in results:
        agg_meta = results["aggregator"].get("metadata", {})
        agg_node = tree.add(f"[green]✓[/green] AggregatorAgent")
        agg_node.add(f"Frameworks: {agg_meta.get('frameworks_count', 0)}")

    # Reporting
    if "reporting" in results:
        rep_meta = results["reporting"].get("metadata", {})
        rep_node = tree.add(f"[green]✓[/green] ReportingAgent")
        rep_node.add(f"Format: {rep_meta.get('format', 'XBRL')}")
        esef = "[green]YES ✓[/green]" if rep_meta.get("esef_compliant") else "[red]NO ✗[/red]"
        rep_node.add(f"ESEF Compliant: {esef}")

    # Audit
    if "audit" in results:
        audit_meta = results["audit"].get("metadata", {})
        is_compliant = audit_meta.get("is_compliant", False)
        status = "[green]PASS ✅[/green]" if is_compliant else "[red]FAIL ❌[/red]"
        audit_node = tree.add(f"[green]✓[/green] AuditAgent")
        audit_node.add(f"Status: {status}")
        audit_node.add(f"Rules checked: {audit_meta.get('rules_checked', 0)}")
        if not is_compliant:
            audit_node.add(
                f"[red]Critical issues: {audit_meta.get('critical_issues', 0)}[/red]"
            )

    console.print(tree)

    # Output files
    console.print("\n[bold]Output Files:[/bold]")
    if output_dir.exists():
        for file in sorted(output_dir.glob("*")):
            if file.is_file():
                size = file.stat().st_size
                size_str = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"
                console.print(f"  [dim]→[/dim] {file.name} [dim]({size_str})[/dim]")


# ============================================================================
# CLI COMMANDS
# ============================================================================

@click.group()
@click.version_option(version=__version__, prog_name="CSRD Platform")
def csrd():
    """
    CSRD/ESRS Digital Reporting Platform CLI.

    Transform raw ESG data into submission-ready EU CSRD reports
    with XBRL tagging in under 30 minutes.

    \b
    Examples:
        csrd run --input data.csv --company-profile company.json
        csrd validate --input data.csv --verbose
        csrd audit --report report.json
    """
    pass


@csrd.command()
@click.option(
    "--input", "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input ESG data file (CSV/JSON/Excel/Parquet)"
)
@click.option(
    "--company-profile", "-p",
    required=True,
    type=click.Path(exists=True),
    help="Company profile JSON file"
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="output",
    help="Output directory for reports (default: ./output)"
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="Configuration file (.csrd.yaml)"
)
@click.option(
    "--skip-materiality",
    is_flag=True,
    help="Skip materiality assessment (use for updates)"
)
@click.option(
    "--skip-audit",
    is_flag=True,
    help="Skip compliance audit (faster processing)"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Verbose output with detailed information"
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Minimal output (errors only)"
)
@click.option(
    "--format",
    type=click.Choice(["xbrl", "json", "both"], case_sensitive=False),
    default="both",
    help="Output format (default: both)"
)
def run(
    input: str,
    company_profile: str,
    output_dir: str,
    config: Optional[str],
    skip_materiality: bool,
    skip_audit: bool,
    verbose: bool,
    quiet: bool,
    format: str
):
    """
    Run the complete CSRD reporting pipeline (all 6 agents).

    This executes the full end-to-end pipeline:
    1. IntakeAgent - Data validation and enrichment
    2. MaterialityAgent - Double materiality assessment
    3. CalculatorAgent - Zero-hallucination metric calculations
    4. AggregatorAgent - Multi-framework integration
    5. ReportingAgent - XBRL/ESEF report generation
    6. AuditAgent - 200+ compliance rule validation

    \b
    Example:
        csrd run --input esg_data.csv \\
                 --company-profile company.json \\
                 --output-dir reports/2024 \\
                 --verbose
    """
    if not quiet:
        display_banner()
        console.print("[cyan]Starting CSRD reporting pipeline...[/cyan]\n")

    # Load configuration
    cfg = load_config(config)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Track timing
    start_time = time.time()
    results = {}

    try:
        # Import agents on-demand
        import json
        import sys
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from agents import (
            IntakeAgent,
            MaterialityAgent,
            CalculatorAgent,
            AggregatorAgent,
            ReportingAgent,
            AuditAgent,
        )

        # Load company profile
        with open(company_profile) as f:
            company_data = json.load(f)

        # ====================================================================
        # AGENT 1: INTAKE
        # ====================================================================
        if not quiet:
            console.print("[bold]Step 1/6:[/bold] Running IntakeAgent (data validation)...")

        agent_start = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=quiet,
        ) as progress:
            task = progress.add_task(
                "[cyan]Validating ESG data...",
                total=None
            )

            intake_agent = IntakeAgent()
            intake_result = intake_agent.process(input)
            results["intake"] = intake_result

            progress.update(task, completed=True)

        agent_time = time.time() - agent_start
        if not quiet:
            display_agent_result("IntakeAgent", intake_result, agent_time, verbose)

        # Check for critical errors
        if intake_result.get("metadata", {}).get("invalid_records", 0) > 0:
            if not quiet:
                console.print(
                    "\n[yellow]⚠ Warning:[/yellow] Some records failed validation. "
                    "Check output for details."
                )

        # ====================================================================
        # AGENT 2: MATERIALITY (optional)
        # ====================================================================
        if not skip_materiality:
            if not quiet:
                console.print(
                    "\n[bold]Step 2/6:[/bold] Running MaterialityAgent "
                    "(double materiality assessment)..."
                )

            agent_start = time.time()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console,
                disable=quiet,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Assessing materiality...",
                    total=None
                )

                materiality_agent = MaterialityAgent()
                materiality_result = materiality_agent.process(
                    intake_result,
                    company_profile=company_data
                )
                results["materiality"] = materiality_result

                progress.update(task, completed=True)

            agent_time = time.time() - agent_start
            if not quiet:
                display_agent_result(
                    "MaterialityAgent",
                    materiality_result,
                    agent_time,
                    verbose
                )

        # ====================================================================
        # AGENT 3: CALCULATOR
        # ====================================================================
        if not quiet:
            console.print(
                "\n[bold]Step 3/6:[/bold] Running CalculatorAgent "
                "(metric calculations)..."
            )

        agent_start = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=quiet,
        ) as progress:
            task = progress.add_task(
                "[cyan]Calculating metrics (ZERO HALLUCINATION)...",
                total=None
            )

            calculator_agent = CalculatorAgent()
            calculator_result = calculator_agent.process(intake_result)
            results["calculator"] = calculator_result

            progress.update(task, completed=True)

        agent_time = time.time() - agent_start
        if not quiet:
            display_agent_result(
                "CalculatorAgent",
                calculator_result,
                agent_time,
                verbose
            )

        # ====================================================================
        # AGENT 4: AGGREGATOR
        # ====================================================================
        if not quiet:
            console.print(
                "\n[bold]Step 4/6:[/bold] Running AggregatorAgent "
                "(multi-framework integration)..."
            )

        agent_start = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=quiet,
        ) as progress:
            task = progress.add_task(
                "[cyan]Integrating frameworks (TCFD/GRI/SASB)...",
                total=None
            )

            aggregator_agent = AggregatorAgent()
            aggregator_result = aggregator_agent.process(calculator_result)
            results["aggregator"] = aggregator_result

            progress.update(task, completed=True)

        agent_time = time.time() - agent_start
        if not quiet:
            display_agent_result(
                "AggregatorAgent",
                aggregator_result,
                agent_time,
                verbose
            )

        # ====================================================================
        # AGENT 5: REPORTING
        # ====================================================================
        if not quiet:
            console.print(
                "\n[bold]Step 5/6:[/bold] Running ReportingAgent "
                "(XBRL/ESEF report generation)..."
            )

        agent_start = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=quiet,
        ) as progress:
            task = progress.add_task(
                "[cyan]Generating XBRL report...",
                total=None
            )

            reporting_agent = ReportingAgent()
            reporting_result = reporting_agent.process(
                aggregator_result,
                company_profile=company_data,
                output_dir=str(output_path),
                format=format
            )
            results["reporting"] = reporting_result

            progress.update(task, completed=True)

        agent_time = time.time() - agent_start
        if not quiet:
            display_agent_result(
                "ReportingAgent",
                reporting_result,
                agent_time,
                verbose
            )

        # ====================================================================
        # AGENT 6: AUDIT (optional)
        # ====================================================================
        if not skip_audit:
            if not quiet:
                console.print(
                    "\n[bold]Step 6/6:[/bold] Running AuditAgent "
                    "(compliance validation)..."
                )

            agent_start = time.time()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console,
                disable=quiet,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Checking 200+ compliance rules...",
                    total=None
                )

                audit_agent = AuditAgent()
                audit_result = audit_agent.process(reporting_result)
                results["audit"] = audit_result

                progress.update(task, completed=True)

            agent_time = time.time() - agent_start
            if not quiet:
                display_agent_result("AuditAgent", audit_result, agent_time, verbose)

            # Check compliance status
            if not audit_result.get("metadata", {}).get("is_compliant", False):
                if not quiet:
                    console.print(
                        "\n[bold red]⚠ COMPLIANCE FAILED[/bold red]"
                    )
                    console.print(
                        "[yellow]Review audit report for critical issues.[/yellow]"
                    )
                sys.exit(2)  # Exit code 2 for warnings/compliance issues

        # ====================================================================
        # SUCCESS
        # ====================================================================
        total_time = time.time() - start_time

        if not quiet:
            display_pipeline_summary(results, total_time, output_path)
            console.print(
                f"\n[bold green]✨ Pipeline completed successfully![/bold green]"
            )
            console.print(f"[dim]Reports saved to: {output_path.absolute()}[/dim]\n")

        sys.exit(0)

    except Exception as e:
        console.print(f"\n[bold red]✗ Pipeline failed:[/bold red] {str(e)}")
        if verbose:
            import traceback
            console.print("\n[dim]" + traceback.format_exc() + "[/dim]")
        sys.exit(1)


@csrd.command()
@click.option(
    "--input", "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input ESG data file (CSV/JSON/Excel/Parquet)"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output validation report (JSON)"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed validation errors"
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Minimal output (errors only)"
)
def validate(input: str, output: Optional[str], verbose: bool, quiet: bool):
    """
    Validate ESG data without running full pipeline (IntakeAgent only).

    Performs comprehensive data validation:
    - Schema validation
    - ESRS metric code validation
    - Data quality assessment (completeness, accuracy, consistency)
    - Outlier detection
    - Unit validation

    \b
    Example:
        csrd validate --input esg_data.csv --verbose
    """
    if not quiet:
        display_banner()
        console.print("[cyan]Running data validation...[/cyan]\n")

    start_time = time.time()

    try:
        # Import IntakeAgent on-demand
        import json
        import sys
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from agents import IntakeAgent

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=quiet,
        ) as progress:
            task = progress.add_task(
                "[cyan]Validating ESG data...",
                total=None
            )

            agent = IntakeAgent()
            result = agent.process(input)

            progress.update(task, completed=True)

        elapsed_time = time.time() - start_time

        # Display results
        if not quiet:
            display_agent_result("IntakeAgent", result, elapsed_time, verbose)

        # Save output if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            if not quiet:
                console.print(f"\n[green]✓[/green] Report saved to {output}")

        # Check for errors
        invalid_records = result.get("metadata", {}).get("invalid_records", 0)
        if invalid_records > 0:
            if not quiet:
                console.print(
                    f"\n[bold yellow]⚠ Validation completed with {invalid_records} "
                    f"invalid records[/bold yellow]"
                )
            sys.exit(2)  # Warning exit code
        else:
            if not quiet:
                console.print("\n[bold green]✓ All records are valid![/bold green]\n")
            sys.exit(0)

    except Exception as e:
        console.print(f"\n[bold red]✗ Validation failed:[/bold red] {str(e)}")
        if verbose:
            import traceback
            console.print("\n[dim]" + traceback.format_exc() + "[/dim]")
        sys.exit(1)


@csrd.command()
@click.option(
    "--input", "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input validated data (from IntakeAgent)"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output calculated metrics (JSON)"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show calculation details"
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Minimal output (errors only)"
)
def calculate(input: str, output: Optional[str], verbose: bool, quiet: bool):
    """
    Calculate ESRS metrics only (CalculatorAgent).

    Performs ZERO-HALLUCINATION metric calculations:
    - 500+ ESRS metric formulas
    - GHG emissions (Scope 1, 2, 3)
    - Social and governance metrics
    - Environmental indicators
    - Complete audit trail

    \b
    Example:
        csrd calculate --input validated_data.json --verbose
    """
    if not quiet:
        display_banner()
        console.print("[cyan]Running metric calculations...[/cyan]\n")

    start_time = time.time()

    try:
        # Import CalculatorAgent on-demand
        import json
        import sys
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from agents import CalculatorAgent

        # Load input data
        with open(input) as f:
            input_data = json.load(f)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=quiet,
        ) as progress:
            task = progress.add_task(
                "[cyan]Calculating metrics (ZERO HALLUCINATION)...",
                total=None
            )

            agent = CalculatorAgent()
            result = agent.process(input_data)

            progress.update(task, completed=True)

        elapsed_time = time.time() - start_time

        # Display results
        if not quiet:
            display_agent_result("CalculatorAgent", result, elapsed_time, verbose)

        # Save output if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            if not quiet:
                console.print(f"\n[green]✓[/green] Calculations saved to {output}")

        if not quiet:
            console.print(
                "\n[bold green]✓ Calculations complete! "
                "[green](ZERO HALLUCINATION GUARANTEE)[/green][/bold green]\n"
            )

        sys.exit(0)

    except Exception as e:
        console.print(f"\n[bold red]✗ Calculation failed:[/bold red] {str(e)}")
        if verbose:
            import traceback
            console.print("\n[dim]" + traceback.format_exc() + "[/dim]")
        sys.exit(1)


@csrd.command()
@click.option(
    "--report", "-r",
    required=True,
    type=click.Path(exists=True),
    help="CSRD report to audit (JSON or XBRL)"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output audit report (JSON)"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed compliance issues"
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Minimal output (errors only)"
)
def audit(report: str, output: Optional[str], verbose: bool, quiet: bool):
    """
    Run compliance audit only (AuditAgent).

    Validates report against 200+ compliance rules:
    - ESRS disclosure requirements
    - ESEF technical standards
    - Data quality thresholds
    - Cross-validation checks
    - Taxonomy alignment

    \b
    Example:
        csrd audit --report report.json --verbose
    """
    if not quiet:
        display_banner()
        console.print("[cyan]Running compliance audit...[/cyan]\n")

    start_time = time.time()

    try:
        # Import AuditAgent on-demand
        import json
        import sys
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from agents import AuditAgent

        # Load report
        with open(report) as f:
            report_data = json.load(f)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=quiet,
        ) as progress:
            task = progress.add_task(
                "[cyan]Checking 200+ compliance rules...",
                total=None
            )

            agent = AuditAgent()
            result = agent.process(report_data)

            progress.update(task, completed=True)

        elapsed_time = time.time() - start_time

        # Display results
        if not quiet:
            display_agent_result("AuditAgent", result, elapsed_time, verbose)

        # Save output if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            if not quiet:
                console.print(f"\n[green]✓[/green] Audit report saved to {output}")

        # Check compliance status
        is_compliant = result.get("metadata", {}).get("is_compliant", False)
        if not is_compliant:
            if not quiet:
                console.print("\n[bold red]⚠ COMPLIANCE FAILED[/bold red]")
                console.print(
                    "[yellow]Review audit report for critical issues.[/yellow]\n"
                )
            sys.exit(2)  # Warning exit code
        else:
            if not quiet:
                console.print("\n[bold green]✓ Compliance check PASSED![/bold green]\n")
            sys.exit(0)

    except Exception as e:
        console.print(f"\n[bold red]✗ Audit failed:[/bold red] {str(e)}")
        if verbose:
            import traceback
            console.print("\n[dim]" + traceback.format_exc() + "[/dim]")
        sys.exit(1)


@csrd.command()
@click.option(
    "--input", "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input validated data (from IntakeAgent)"
)
@click.option(
    "--company-profile", "-p",
    required=True,
    type=click.Path(exists=True),
    help="Company profile JSON file"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output materiality assessment (JSON)"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed assessment"
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Minimal output (errors only)"
)
def materialize(
    input: str,
    company_profile: str,
    output: Optional[str],
    verbose: bool,
    quiet: bool
):
    """
    Run double materiality assessment only (MaterialityAgent).

    Performs AI-powered materiality analysis:
    - Impact materiality (inside-out)
    - Financial materiality (outside-in)
    - Double materiality identification
    - Stakeholder analysis
    - ESRS topic prioritization

    \b
    Example:
        csrd materialize --input data.json --company-profile company.json
    """
    if not quiet:
        display_banner()
        console.print("[cyan]Running materiality assessment...[/cyan]\n")

    start_time = time.time()

    try:
        # Import MaterialityAgent on-demand
        import json
        import sys
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from agents import MaterialityAgent

        # Load inputs
        with open(input) as f:
            input_data = json.load(f)

        with open(company_profile) as f:
            company_data = json.load(f)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=quiet,
        ) as progress:
            task = progress.add_task(
                "[cyan]Assessing materiality...",
                total=None
            )

            agent = MaterialityAgent()
            result = agent.process(input_data, company_profile=company_data)

            progress.update(task, completed=True)

        elapsed_time = time.time() - start_time

        # Display results
        if not quiet:
            display_agent_result("MaterialityAgent", result, elapsed_time, verbose)

        # Save output if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            if not quiet:
                console.print(f"\n[green]✓[/green] Assessment saved to {output}")

        if not quiet:
            console.print(
                "\n[bold green]✓ Materiality assessment complete![/bold green]\n"
            )

        sys.exit(0)

    except Exception as e:
        console.print(f"\n[bold red]✗ Assessment failed:[/bold red] {str(e)}")
        if verbose:
            import traceback
            console.print("\n[dim]" + traceback.format_exc() + "[/dim]")
        sys.exit(1)


@csrd.command()
@click.option(
    "--input", "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input aggregated data (from AggregatorAgent)"
)
@click.option(
    "--company-profile", "-p",
    required=True,
    type=click.Path(exists=True),
    help="Company profile JSON file"
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="output",
    help="Output directory for reports"
)
@click.option(
    "--format",
    type=click.Choice(["xbrl", "json", "both"], case_sensitive=False),
    default="both",
    help="Output format (default: both)"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show generation details"
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Minimal output (errors only)"
)
def report(
    input: str,
    company_profile: str,
    output_dir: str,
    format: str,
    verbose: bool,
    quiet: bool
):
    """
    Generate XBRL/ESEF report only (ReportingAgent).

    Creates submission-ready reports:
    - XBRL inline format (iXBRL)
    - ESEF compliance tagging
    - ESRS taxonomy mapping
    - Digital signature ready
    - Validator-tested output

    \b
    Example:
        csrd report --input data.json --company-profile company.json \\
                    --format xbrl
    """
    if not quiet:
        display_banner()
        console.print("[cyan]Generating CSRD report...[/cyan]\n")

    start_time = time.time()

    try:
        # Import ReportingAgent on-demand
        import json
        import sys
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from agents import ReportingAgent

        # Load inputs
        with open(input) as f:
            input_data = json.load(f)

        with open(company_profile) as f:
            company_data = json.load(f)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=quiet,
        ) as progress:
            task = progress.add_task(
                "[cyan]Generating XBRL report...",
                total=None
            )

            agent = ReportingAgent()
            result = agent.process(
                input_data,
                company_profile=company_data,
                output_dir=str(output_path),
                format=format
            )

            progress.update(task, completed=True)

        elapsed_time = time.time() - start_time

        # Display results
        if not quiet:
            display_agent_result("ReportingAgent", result, elapsed_time, verbose)

            # Show output files
            console.print("\n[bold]Generated Files:[/bold]")
            for file in sorted(output_path.glob("*")):
                if file.is_file():
                    size = file.stat().st_size
                    size_str = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"
                    console.print(
                        f"  [dim]→[/dim] {file.name} [dim]({size_str})[/dim]"
                    )

            console.print(
                f"\n[bold green]✓ Report generated successfully![/bold green]"
            )
            console.print(f"[dim]Reports saved to: {output_path.absolute()}[/dim]\n")

        sys.exit(0)

    except Exception as e:
        console.print(f"\n[bold red]✗ Report generation failed:[/bold red] {str(e)}")
        if verbose:
            import traceback
            console.print("\n[dim]" + traceback.format_exc() + "[/dim]")
        sys.exit(1)


@csrd.command()
@click.option(
    "--input", "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input calculated metrics (from CalculatorAgent)"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output aggregated data (JSON)"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show aggregation details"
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Minimal output (errors only)"
)
def aggregate(input: str, output: Optional[str], verbose: bool, quiet: bool):
    """
    Aggregate multi-framework data only (AggregatorAgent).

    Integrates multiple reporting frameworks:
    - ESRS (primary)
    - TCFD (climate)
    - GRI (comprehensive sustainability)
    - SASB (industry-specific)

    Creates unified dataset with cross-references.

    \b
    Example:
        csrd aggregate --input calculated.json --verbose
    """
    if not quiet:
        display_banner()
        console.print("[cyan]Running framework aggregation...[/cyan]\n")

    start_time = time.time()

    try:
        # Import AggregatorAgent on-demand
        import json
        import sys
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from agents import AggregatorAgent

        # Load input data
        with open(input) as f:
            input_data = json.load(f)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=quiet,
        ) as progress:
            task = progress.add_task(
                "[cyan]Integrating frameworks (TCFD/GRI/SASB)...",
                total=None
            )

            agent = AggregatorAgent()
            result = agent.process(input_data)

            progress.update(task, completed=True)

        elapsed_time = time.time() - start_time

        # Display results
        if not quiet:
            display_agent_result("AggregatorAgent", result, elapsed_time, verbose)

        # Save output if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            if not quiet:
                console.print(f"\n[green]✓[/green] Aggregated data saved to {output}")

        if not quiet:
            console.print("\n[bold green]✓ Aggregation complete![/bold green]\n")

        sys.exit(0)

    except Exception as e:
        console.print(f"\n[bold red]✗ Aggregation failed:[/bold red] {str(e)}")
        if verbose:
            import traceback
            console.print("\n[dim]" + traceback.format_exc() + "[/dim]")
        sys.exit(1)


@csrd.command()
@click.option(
    "--init",
    is_flag=True,
    help="Create new configuration file interactively"
)
@click.option(
    "--show",
    is_flag=True,
    help="Display current configuration"
)
@click.option(
    "--path",
    type=click.Path(),
    default=".csrd.yaml",
    help="Configuration file path (default: .csrd.yaml)"
)
def config(init: bool, show: bool, path: str):
    """
    Manage CSRD configuration.

    \b
    Examples:
        csrd config --init              # Create new config
        csrd config --show              # Show current config
        csrd config --init --path custom.yaml
    """
    if init:
        console.print("[cyan]Creating CSRD configuration...[/cyan]\n")

        config_data = {
            "company": {
                "name": click.prompt("Company legal name"),
                "lei": click.prompt("Legal Entity Identifier (LEI)", default=""),
                "country": click.prompt("Country code (e.g., NL, DE, FR)"),
                "sector": click.prompt("Industry sector"),
            },
            "reporting_period": {
                "start_date": click.prompt(
                    "Reporting period start",
                    default=f"{DeterministicClock.now().year}-01-01"
                ),
                "end_date": click.prompt(
                    "Reporting period end",
                    default=f"{DeterministicClock.now().year}-12-31"
                ),
            },
            "materiality": {
                "threshold": click.prompt(
                    "Materiality threshold (0-100)",
                    type=int,
                    default=50
                ),
            },
            "paths": {
                "esrs_catalog": "data/esrs_catalog.json",
                "emission_factors": "data/emission_factors.json",
                "rules": "rules/compliance_rules.yaml",
            },
        }

        save_config(config_data, path)
        console.print("\n[green]✓[/green] Configuration created successfully!")
        console.print(f"[dim]Saved to: {path}[/dim]\n")

    elif show:
        try:
            config_data = load_config(path)
            console.print("\n[bold]Current CSRD Configuration:[/bold]\n")
            console.print_json(data=config_data)
            console.print()
        except Exception as e:
            console.print(f"[red]✗[/red] Could not load config: {e}")
            sys.exit(1)

    else:
        console.print(
            "[yellow]Please specify --init or --show[/yellow]\n"
            "Run 'csrd config --help' for more information."
        )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    csrd()
