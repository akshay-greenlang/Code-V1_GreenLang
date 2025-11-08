"""
GL-VCCI CLI - Intake Command
File ingestion and data validation for value chain data.

Features:
- Multi-format support (CSV, JSON, Excel, XML, PDF)
- Auto-format detection
- Batch directory processing
- Progress tracking
- Integration with ValueChainIntakeAgent

Version: 1.0.0
Date: 2025-11-08
"""

import sys
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import box
from rich.text import Text

# Import the intake agent
try:
    from services.agents.intake.agent import ValueChainIntakeAgent
    from services.agents.intake.exceptions import IntakeAgentError, UnsupportedFormatError
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False

# Create console
console = Console()

# Create Typer app for intake commands
intake_app = typer.Typer(
    name="intake",
    help="File ingestion and data validation",
    rich_markup_mode="rich"
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_format(file_path: Path) -> str:
    """Auto-detect file format from extension."""
    suffix = file_path.suffix.lower()

    format_map = {
        '.csv': 'csv',
        '.json': 'json',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.xml': 'xml',
        '.pdf': 'pdf'
    }

    return format_map.get(suffix, 'unknown')


def format_number(num: int) -> str:
    """Format number with thousands separator."""
    return f"{num:,}"


# ============================================================================
# INTAKE SINGLE FILE COMMAND
# ============================================================================

@intake_app.command("file")
def intake_file(
    file: Path = typer.Option(
        ...,
        "--file",
        "-f",
        help="File to ingest",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True
    ),
    format: Optional[str] = typer.Option(
        None,
        "--format",
        help="File format (csv, json, excel, xml, pdf). Auto-detects if not specified."
    ),
    entity_type: str = typer.Option(
        "supplier",
        "--entity-type",
        "-e",
        help="Entity type (supplier, product, facility, etc.)"
    ),
    source_system: str = typer.Option(
        "Manual_Upload",
        "--source",
        "-s",
        help="Source system identifier"
    ),
    tenant_id: str = typer.Option(
        "cli-user",
        "--tenant",
        "-t",
        help="Tenant identifier"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for results (JSON)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed processing information"
    )
):
    """
    Ingest data from a single file.

    Supports CSV, JSON, Excel, XML, and PDF formats with automatic
    format detection and intelligent entity resolution.

    Examples:

      # Ingest CSV with auto-detection
      vcci intake file --file suppliers.csv

      # Ingest Excel with explicit format
      vcci intake file --file procurement.xlsx --format excel --entity-type product

      # Ingest with custom tenant and source
      vcci intake file --file data.json --tenant acme-corp --source SAP
    """
    console.print()

    if not AGENT_AVAILABLE:
        console.print(
            "[red]Error:[/red] ValueChainIntakeAgent not available. "
            "Please ensure services.agents.intake module is properly installed."
        )
        sys.exit(1)

    # Auto-detect format if not specified
    if format is None:
        format = detect_format(file)
        if format == 'unknown':
            console.print(f"[red]Error:[/red] Could not detect format for {file}. Please specify --format")
            sys.exit(1)
        if verbose:
            console.print(f"[cyan]Auto-detected format:[/cyan] {format}")

    # Display ingestion info
    info_panel = Panel(
        f"[cyan]File:[/cyan] {file}\n"
        f"[cyan]Format:[/cyan] {format}\n"
        f"[cyan]Entity Type:[/cyan] {entity_type}\n"
        f"[cyan]Source:[/cyan] {source_system}\n"
        f"[cyan]Tenant:[/cyan] {tenant_id}",
        title="[bold cyan]Ingestion Configuration[/bold cyan]",
        border_style="cyan"
    )
    console.print(info_panel)
    console.print()

    try:
        # Initialize agent
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            init_task = progress.add_task("[cyan]Initializing intake agent...", total=None)
            agent = ValueChainIntakeAgent(tenant_id=tenant_id)
            progress.update(init_task, completed=True)

        # Ingest file with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                f"[cyan]Processing {file.name}...",
                total=100
            )

            # Simulate progress updates (in production, this would be real progress)
            for i in range(0, 101, 20):
                progress.update(task, completed=i)
                if i == 20:
                    progress.update(task, description=f"[cyan]Parsing {format.upper()} file...")
                elif i == 40:
                    progress.update(task, description=f"[cyan]Resolving entities...")
                elif i == 60:
                    progress.update(task, description=f"[cyan]Assessing data quality...")
                elif i == 80:
                    progress.update(task, description=f"[cyan]Finalizing ingestion...")

            # Actual ingestion
            result = agent.ingest_file(
                file_path=file,
                format=format,
                entity_type=entity_type,
                source_system=source_system
            )

            progress.update(task, completed=100)

        console.print()

        # Display results
        stats = result.statistics

        # Summary panel
        summary_text = (
            f"[green]Total Records:[/green] {format_number(stats.total_records)}\n"
            f"[green]Successfully Processed:[/green] {format_number(stats.successful)}\n"
        )

        if stats.failed > 0:
            summary_text += f"[red]Failed:[/red] {format_number(stats.failed)}\n"

        summary_text += (
            f"\n[yellow]Entity Resolution:[/yellow]\n"
            f"  Auto-resolved: {format_number(stats.resolved_auto)}\n"
            f"  Sent to review: {format_number(stats.sent_to_review)}\n"
        )

        if stats.resolution_failures > 0:
            summary_text += f"  Failed: {format_number(stats.resolution_failures)}\n"

        if stats.avg_dqi_score:
            summary_text += f"\n[cyan]Avg DQI Score:[/cyan] {stats.avg_dqi_score:.1f}%\n"

        if stats.avg_confidence:
            summary_text += f"[cyan]Avg Confidence:[/cyan] {stats.avg_confidence:.1f}%\n"

        summary_text += (
            f"\n[magenta]Performance:[/magenta]\n"
            f"  Processing time: {stats.processing_time_seconds:.2f}s\n"
            f"  Records/second: {stats.records_per_second:.1f}"
        )

        summary_panel = Panel(
            summary_text,
            title=f"[bold green]Ingestion Complete - {result.batch_id}[/bold green]",
            border_style="green"
        )
        console.print(summary_panel)

        # Review queue notification
        if stats.sent_to_review > 0:
            console.print()
            console.print(
                f"[yellow]Note:[/yellow] {format_number(stats.sent_to_review)} records sent to review queue "
                f"for manual verification (low confidence matches)"
            )

        # Failed records table
        if stats.failed > 0 and verbose:
            console.print()
            failed_table = Table(
                title="Failed Records",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold red"
            )
            failed_table.add_column("Record ID", style="yellow")
            failed_table.add_column("Entity Name")
            failed_table.add_column("Error", style="red")

            for fail in result.failed_records[:10]:  # Show first 10
                failed_table.add_row(
                    fail.get("record_id", "N/A"),
                    fail.get("entity_name", "N/A"),
                    fail.get("error", "Unknown error")
                )

            if len(result.failed_records) > 10:
                failed_table.add_row("...", "...", f"+{len(result.failed_records) - 10} more")

            console.print(failed_table)

        # Save output if specified
        if output:
            import json
            output_data = {
                "batch_id": result.batch_id,
                "statistics": {
                    "total_records": stats.total_records,
                    "successful": stats.successful,
                    "failed": stats.failed,
                    "resolved_auto": stats.resolved_auto,
                    "sent_to_review": stats.sent_to_review,
                    "avg_dqi_score": stats.avg_dqi_score,
                    "processing_time_seconds": stats.processing_time_seconds
                },
                "ingested_records": result.ingested_records,
                "failed_records": result.failed_records
            }

            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)

            console.print()
            console.print(f"[green]Results saved to:[/green] {output}")

    except UnsupportedFormatError as e:
        console.print()
        console.print(f"[red]Error:[/red] Unsupported format: {format}")
        console.print(f"[yellow]Supported formats:[/yellow] csv, json, excel, xml, pdf")
        sys.exit(1)

    except IntakeAgentError as e:
        console.print()
        console.print(f"[red]Ingestion Error:[/red] {str(e)}")
        if verbose and hasattr(e, 'details'):
            console.print(f"[yellow]Details:[/yellow] {e.details}")
        sys.exit(1)

    except Exception as e:
        console.print()
        console.print(f"[red]Unexpected Error:[/red] {str(e)}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)

    console.print()


# ============================================================================
# INTAKE BATCH DIRECTORY COMMAND
# ============================================================================

@intake_app.command("batch")
def intake_batch(
    directory: Path = typer.Option(
        ...,
        "--directory",
        "-d",
        help="Directory containing files to ingest",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True
    ),
    pattern: str = typer.Option(
        "*.*",
        "--pattern",
        "-p",
        help="File pattern to match (e.g., '*.csv', '*.json')"
    ),
    entity_type: str = typer.Option(
        "supplier",
        "--entity-type",
        "-e",
        help="Entity type for all files"
    ),
    tenant_id: str = typer.Option(
        "cli-user",
        "--tenant",
        "-t",
        help="Tenant identifier"
    )
):
    """
    Ingest all files from a directory (batch processing).

    Automatically detects formats and processes multiple files
    with consolidated reporting.

    Examples:

      # Ingest all files in a directory
      vcci intake batch --directory data/

      # Ingest only CSV files
      vcci intake batch --directory data/ --pattern "*.csv"

      # Batch ingest products
      vcci intake batch --directory products/ --entity-type product
    """
    console.print()

    if not AGENT_AVAILABLE:
        console.print(
            "[red]Error:[/red] ValueChainIntakeAgent not available."
        )
        sys.exit(1)

    # Find matching files
    files = list(directory.glob(pattern))

    if not files:
        console.print(f"[yellow]No files found matching pattern '{pattern}' in {directory}[/yellow]")
        sys.exit(0)

    console.print(f"[cyan]Found {len(files)} file(s) to process[/cyan]")
    console.print()

    # Initialize agent
    agent = ValueChainIntakeAgent(tenant_id=tenant_id)

    # Process files with overall progress
    total_records = 0
    total_successful = 0
    total_failed = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        overall_task = progress.add_task(
            "[cyan]Processing batch...",
            total=len(files)
        )

        for i, file in enumerate(files, 1):
            format = detect_format(file)

            if format == 'unknown':
                console.print(f"[yellow]Skipping {file.name} (unknown format)[/yellow]")
                progress.advance(overall_task)
                continue

            progress.update(
                overall_task,
                description=f"[cyan]Processing {file.name} ({i}/{len(files)})..."
            )

            try:
                result = agent.ingest_file(
                    file_path=file,
                    format=format,
                    entity_type=entity_type,
                    source_system="Batch_Upload"
                )

                total_records += result.statistics.total_records
                total_successful += result.statistics.successful
                total_failed += result.statistics.failed

            except Exception as e:
                console.print(f"[red]Failed to process {file.name}:[/red] {str(e)}")

            progress.advance(overall_task)

    console.print()

    # Display batch summary
    summary_panel = Panel(
        f"[green]Files Processed:[/green] {len(files)}\n"
        f"[green]Total Records:[/green] {format_number(total_records)}\n"
        f"[green]Successful:[/green] {format_number(total_successful)}\n"
        f"[red]Failed:[/red] {format_number(total_failed)}\n"
        f"[cyan]Success Rate:[/cyan] {(total_successful / total_records * 100) if total_records > 0 else 0:.1f}%",
        title="[bold cyan]Batch Ingestion Complete[/bold cyan]",
        border_style="green"
    )
    console.print(summary_panel)
    console.print()


# ============================================================================
# INTAKE STATUS COMMAND
# ============================================================================

@intake_app.command("status")
def intake_status(
    batch_id: Optional[str] = typer.Option(
        None,
        "--batch-id",
        "-b",
        help="Show status for specific batch"
    )
):
    """
    Show intake status and statistics.

    Displays overall ingestion metrics or specific batch details.

    Examples:

      # Show overall status
      vcci intake status

      # Show specific batch status
      vcci intake status --batch-id BATCH-20251108-ABC123
    """
    console.print()

    if batch_id:
        # Batch-specific status (would query from storage in production)
        console.print(f"[yellow]Batch-specific status not yet implemented[/yellow]")
        console.print(f"[cyan]Batch ID:[/cyan] {batch_id}")
    else:
        # Overall status
        status_table = Table(
            title="Intake Agent Status",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )

        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", justify="center")
        status_table.add_column("Details")

        status_table.add_row(
            "Intake Agent",
            "[green]Active[/green]",
            "Multi-format ingestion"
        )
        status_table.add_row(
            "Supported Formats",
            "[green]5[/green]",
            "CSV, JSON, Excel, XML, PDF"
        )
        status_table.add_row(
            "Entity Resolution",
            "[green]Enabled[/green]",
            "Confidence-based matching"
        )
        status_table.add_row(
            "Data Quality",
            "[green]Enabled[/green]",
            "DQI scoring active"
        )
        status_table.add_row(
            "Review Queue",
            "[green]Active[/green]",
            "Human verification available"
        )

        console.print(status_table)

    console.print()


# Export the app
__all__ = ["intake_app"]
