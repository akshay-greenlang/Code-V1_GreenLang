"""
CSRD Complete Pipeline Runner

This script runs the complete CSRD reporting pipeline:
- Runs the full 6-agent CSRD pipeline
- Supports configuration files
- Batch processing for multiple companies
- Progress reporting and monitoring
- Error handling and recovery
- Summary report generation
- Performance metrics tracking

Usage:
    python scripts/run_full_pipeline.py --esg-data data.csv --company-profile company.json
    python scripts/run_full_pipeline.py --config pipeline_config.yaml --output output/reports
    python scripts/run_full_pipeline.py --batch companies.json --parallel
    python scripts/run_full_pipeline.py --resume failed_pipeline_abc123

Version: 1.0.0
Author: GreenLang CSRD Team
License: MIT
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from csrd_pipeline import CSRDPipeline, PipelineResult

console = Console()


# ============================================================================
# BATCH PROCESSING MODELS
# ============================================================================

class BatchJob:
    """Represents a batch processing job."""

    def __init__(self, job_id: str, companies: List[Dict[str, Any]]):
        self.job_id = job_id
        self.companies = companies
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.results: List[Dict[str, Any]] = []
        self.successful_count = 0
        self.failed_count = 0

    def add_result(self, company_id: str, status: str, result: Optional[PipelineResult] = None, error: Optional[str] = None):
        """Add a pipeline result."""
        self.results.append({
            "company_id": company_id,
            "status": status,
            "result": result.dict() if result else None,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })

        if status == "success":
            self.successful_count += 1
        else:
            self.failed_count += 1

    def complete(self):
        """Mark batch job as complete."""
        self.end_time = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        return {
            "job_id": self.job_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": duration,
            "total_companies": len(self.companies),
            "successful": self.successful_count,
            "failed": self.failed_count,
            "results": self.results
        }


# ============================================================================
# PIPELINE EXECUTION FUNCTIONS
# ============================================================================

def run_single_pipeline(
    esg_data_file: str,
    company_profile: Dict[str, Any],
    config_path: str,
    output_dir: Optional[str] = None
) -> PipelineResult:
    """
    Run pipeline for a single company.

    Args:
        esg_data_file: Path to ESG data file
        company_profile: Company profile dictionary
        config_path: Path to CSRD config file
        output_dir: Output directory for results

    Returns:
        PipelineResult

    Raises:
        Exception: If pipeline execution fails
    """
    console.print(f"\n[bold cyan]Running CSRD Pipeline[/bold cyan]")
    console.print(f"Company: {company_profile.get('legal_name', 'Unknown')}")
    console.print(f"Data file: {esg_data_file}\n")

    # Initialize pipeline
    pipeline = CSRDPipeline(config_path=config_path)

    # Run pipeline
    result = pipeline.run(
        esg_data_file=esg_data_file,
        company_profile=company_profile,
        output_dir=output_dir
    )

    return result


def run_batch_pipelines(
    batch_config: Dict[str, Any],
    base_config_path: str,
    output_base_dir: str,
    parallel: bool = False
) -> BatchJob:
    """
    Run pipelines for multiple companies.

    Args:
        batch_config: Batch configuration with list of companies
        base_config_path: Base CSRD config path
        output_base_dir: Base output directory
        parallel: Whether to run in parallel (not implemented yet)

    Returns:
        BatchJob with results
    """
    job_id = f"batch_{int(time.time())}"
    companies = batch_config.get("companies", [])

    console.print(f"\n[bold cyan]🚀 Batch Processing Job: {job_id}[/bold cyan]")
    console.print(f"Companies to process: {len(companies)}\n")

    batch_job = BatchJob(job_id, companies)

    if parallel:
        console.print("[yellow]⚠ Parallel processing not yet implemented, running sequentially[/yellow]\n")

    # Process each company
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:

        task = progress.add_task("Processing companies...", total=len(companies))

        for idx, company_config in enumerate(companies, 1):
            company_id = company_config.get("company_id", f"company_{idx}")
            company_name = company_config.get("legal_name", company_id)

            progress.update(task, description=f"Processing {company_name}...")

            try:
                # Setup output directory for this company
                company_output_dir = Path(output_base_dir) / company_id
                company_output_dir.mkdir(parents=True, exist_ok=True)

                # Run pipeline
                result = run_single_pipeline(
                    esg_data_file=company_config["esg_data_file"],
                    company_profile=company_config.get("company_profile", {}),
                    config_path=base_config_path,
                    output_dir=str(company_output_dir)
                )

                batch_job.add_result(company_id, "success", result=result)
                console.print(f"[green]✓ {company_name} completed successfully[/green]")

            except Exception as e:
                batch_job.add_result(company_id, "failed", error=str(e))
                console.print(f"[red]✗ {company_name} failed: {e}[/red]")

            progress.update(task, advance=1)

    batch_job.complete()
    return batch_job


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_summary_report(result: PipelineResult, output_path: Path):
    """Generate a summary report for a pipeline run."""

    lines = [
        "# CSRD Pipeline Execution Report",
        "",
        f"**Pipeline ID:** {result.pipeline_id}",
        f"**Timestamp:** {result.execution_timestamp}",
        f"**Status:** {result.status.upper()}",
        "",
        "## Performance Summary",
        "",
        f"- **Total Time:** {result.performance.total_time_seconds:.2f}s ({result.performance.total_time_seconds / 60:.1f} minutes)",
        f"- **Target Time:** {result.performance.target_time_minutes} minutes",
        f"- **Meets Target:** {'✅ YES' if result.performance.within_target else '❌ NO'}",
        f"- **Records Processed:** {result.performance.records_processed}",
        f"- **Throughput:** {result.performance.records_per_second:.2f} records/second",
        "",
        "## Agent Performance",
        "",
        "| Agent | Duration (s) | % of Total |",
        "|-------|--------------|------------|",
    ]

    perf = result.performance
    total_time = perf.total_time_seconds

    agent_times = [
        ("Stage 1: IntakeAgent", perf.agent_1_intake_seconds),
        ("Stage 2: MaterialityAgent", perf.agent_2_materiality_seconds),
        ("Stage 3: CalculatorAgent", perf.agent_3_calculator_seconds),
        ("Stage 4: AggregatorAgent", perf.agent_4_aggregator_seconds),
        ("Stage 5: ReportingAgent", perf.agent_5_reporting_seconds),
        ("Stage 6: AuditAgent", perf.agent_6_audit_seconds),
    ]

    for agent_name, duration in agent_times:
        percentage = (duration / total_time * 100) if total_time > 0 else 0
        lines.append(f"| {agent_name} | {duration:.2f} | {percentage:.1f}% |")

    lines.extend([
        "",
        "## Data Quality & Compliance",
        "",
        f"- **Data Quality Score:** {result.data_quality_score:.1f}/100",
        f"- **Compliance Status:** {result.compliance_status}",
        f"- **Total Data Points:** {result.total_data_points_processed}",
        f"- **Warnings:** {result.warnings_count}",
        f"- **Errors:** {result.errors_count}",
        "",
        "## Agent Execution Details",
        ""
    ])

    for execution in result.agent_executions:
        lines.extend([
            f"### {execution.agent_name}",
            "",
            f"- **Duration:** {execution.duration_seconds}s",
            f"- **Input Records:** {execution.input_records}",
            f"- **Output Records:** {execution.output_records}",
            f"- **Status:** {execution.status}",
            f"- **Warnings:** {execution.warnings}",
            f"- **Errors:** {execution.errors}",
            ""
        ])

    lines.extend([
        "## Configuration",
        "",
        f"- **Config Version:** {result.configuration_used.get('config_version', 'N/A')}",
        f"- **ESRS Regulation:** {result.configuration_used.get('esrs_regulation', 'N/A')}",
        f"- **ESEF Regulation:** {result.configuration_used.get('esef_regulation', 'N/A')}",
        "",
        "---",
        "",
        f"*Report generated at {datetime.now().isoformat()}*"
    ])

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    console.print(f"[green]✓ Summary report saved to {output_path}[/green]")


def display_pipeline_summary(result: PipelineResult):
    """Display pipeline summary in console."""

    # Status panel
    if result.status == "success":
        status_style = "green"
        status_emoji = "✅"
    elif result.status == "partial_success":
        status_style = "yellow"
        status_emoji = "⚠️"
    else:
        status_style = "red"
        status_emoji = "❌"

    console.print(Panel(
        f"[bold {status_style}]{status_emoji} Pipeline Status: {result.status.upper()}[/bold {status_style}]",
        title="CSRD Pipeline Result",
        border_style=status_style
    ))

    console.print()

    # Performance table
    perf_table = Table(title="Performance Metrics")
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", justify="right", style="green")

    perf_table.add_row("Total Time", f"{result.performance.total_time_seconds:.2f}s")
    perf_table.add_row("Target Time", f"{result.performance.target_time_minutes * 60}s")
    perf_table.add_row("Within Target", "✅ YES" if result.performance.within_target else "❌ NO")
    perf_table.add_row("Records Processed", str(result.performance.records_processed))
    perf_table.add_row("Throughput", f"{result.performance.records_per_second:.2f} rec/s")
    perf_table.add_row("Data Quality", f"{result.data_quality_score:.1f}/100")
    perf_table.add_row("Compliance Status", result.compliance_status)

    console.print(perf_table)
    console.print()

    # Agent breakdown
    agent_table = Table(title="Agent Execution Breakdown")
    agent_table.add_column("Agent", style="cyan")
    agent_table.add_column("Duration (s)", justify="right", style="magenta")
    agent_table.add_column("Records", justify="right", style="green")
    agent_table.add_column("Status", justify="center")

    for execution in result.agent_executions:
        status_icon = "✅" if execution.status == "success" else "❌"
        agent_table.add_row(
            execution.agent_name,
            f"{execution.duration_seconds:.2f}",
            f"{execution.input_records} → {execution.output_records}",
            f"{status_icon}"
        )

    console.print(agent_table)
    console.print()


def display_batch_summary(batch_job: BatchJob):
    """Display batch processing summary."""

    console.print(Panel(
        f"[bold cyan]Batch Job: {batch_job.job_id}[/bold cyan]",
        title="Batch Processing Complete",
        border_style="cyan"
    ))

    console.print()

    # Summary table
    summary_table = Table(title="Batch Processing Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right", style="green")

    duration = (batch_job.end_time - batch_job.start_time).total_seconds() if batch_job.end_time else 0
    summary_table.add_row("Total Companies", str(len(batch_job.companies)))
    summary_table.add_row("Successful", f"✅ {batch_job.successful_count}")
    summary_table.add_row("Failed", f"❌ {batch_job.failed_count}")
    summary_table.add_row("Total Time", f"{duration:.2f}s ({duration / 60:.1f} min)")
    summary_table.add_row("Avg Time per Company", f"{duration / len(batch_job.companies):.2f}s" if batch_job.companies else "N/A")

    console.print(summary_table)
    console.print()


# ============================================================================
# CLI INTERFACE
# ============================================================================

@click.command()
@click.option(
    '--esg-data',
    type=click.Path(exists=True),
    help='ESG data file (CSV/JSON/Excel/Parquet)'
)
@click.option(
    '--company-profile',
    type=click.Path(exists=True),
    help='Company profile JSON file'
)
@click.option(
    '--config',
    type=click.Path(exists=True),
    default='config/csrd_config.yaml',
    help='CSRD configuration file'
)
@click.option(
    '--output',
    type=click.Path(),
    default='output/csrd_report',
    help='Output directory for reports'
)
@click.option(
    '--batch',
    type=click.Path(exists=True),
    help='Batch configuration file (JSON/YAML) with multiple companies'
)
@click.option(
    '--parallel',
    is_flag=True,
    help='Run batch processing in parallel (not yet implemented)'
)
@click.option(
    '--generate-summary',
    is_flag=True,
    default=True,
    help='Generate summary report (Markdown)'
)
@click.option(
    '--resume',
    type=str,
    help='Resume a failed pipeline run (pipeline ID)'
)
def run_pipeline(
    esg_data: Optional[str],
    company_profile: Optional[str],
    config: str,
    output: str,
    batch: Optional[str],
    parallel: bool,
    generate_summary: bool,
    resume: Optional[str]
):
    """
    Run the complete CSRD reporting pipeline.

    Processes ESG data through all 6 agents to generate a submission-ready
    CSRD report. Supports single company or batch processing.
    """
    console.print("\n[bold cyan]🌍 CSRD Complete Pipeline Runner[/bold cyan]\n")

    # Resume functionality
    if resume:
        console.print(f"[yellow]⚠ Resume functionality not yet implemented for {resume}[/yellow]")
        sys.exit(1)

    # Batch processing mode
    if batch:
        console.print(f"[cyan]Mode: Batch Processing[/cyan]")
        console.print(f"[cyan]Batch config: {batch}[/cyan]\n")

        # Load batch configuration
        batch_path = Path(batch)
        if batch_path.suffix in ['.yaml', '.yml']:
            with open(batch_path, 'r') as f:
                batch_config = yaml.safe_load(f)
        else:
            with open(batch_path, 'r') as f:
                batch_config = json.load(f)

        # Run batch
        batch_job = run_batch_pipelines(
            batch_config=batch_config,
            base_config_path=config,
            output_base_dir=output,
            parallel=parallel
        )

        # Display summary
        display_batch_summary(batch_job)

        # Save batch report
        batch_report_path = Path(output) / f"batch_report_{batch_job.job_id}.json"
        with open(batch_report_path, 'w') as f:
            json.dump(batch_job.to_dict(), f, indent=2)
        console.print(f"[green]✓ Batch report saved to {batch_report_path}[/green]\n")

        sys.exit(0 if batch_job.failed_count == 0 else 1)

    # Single company mode
    if not esg_data or not company_profile:
        console.print("[red]Error: --esg-data and --company-profile are required for single company mode[/red]")
        console.print("[yellow]Tip: Use --batch for batch processing mode[/yellow]")
        sys.exit(1)

    console.print(f"[cyan]Mode: Single Company[/cyan]")
    console.print(f"[cyan]ESG data: {esg_data}[/cyan]")
    console.print(f"[cyan]Company profile: {company_profile}[/cyan]")
    console.print(f"[cyan]Config: {config}[/cyan]")
    console.print(f"[cyan]Output: {output}[/cyan]\n")

    # Load company profile
    with open(company_profile, 'r') as f:
        company_profile_data = json.load(f)

    # Run pipeline
    try:
        result = run_single_pipeline(
            esg_data_file=esg_data,
            company_profile=company_profile_data,
            config_path=config,
            output_dir=output
        )

        # Display summary
        console.print("\n")
        display_pipeline_summary(result)

        # Generate summary report
        if generate_summary:
            summary_path = Path(output) / "pipeline_summary.md"
            generate_summary_report(result, summary_path)

        console.print(f"\n[bold green]✅ Pipeline execution complete![/bold green]\n")

        # Exit code based on compliance
        if result.compliance_status == "FAIL":
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        console.print(f"\n[bold red]❌ Pipeline execution failed![/bold red]")
        console.print(f"[red]Error: {e}[/red]\n")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)


if __name__ == '__main__':
    run_pipeline()
