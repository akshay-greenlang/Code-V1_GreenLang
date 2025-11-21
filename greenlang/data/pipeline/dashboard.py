# -*- coding: utf-8 -*-
"""
Data Quality Dashboard and CLI

Interactive dashboard and command-line interface for monitoring
emission factors data quality and pipeline operations.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import json
import logging
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from .monitoring import DataQualityMonitor, CoverageAnalyzer, SourceDiversityAnalyzer, FreshnessTracker
from .pipeline import AutomatedImportPipeline, ScheduledImporter, RollbackManager
from .workflow import UpdateWorkflow, ApprovalManager
from .models import ImportJob, ImportStatus
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)
console = Console()


class DataQualityDashboard:
    """
    Interactive data quality dashboard.

    Displays comprehensive metrics and monitoring information.
    """

    def __init__(self, db_path: str):
        """
        Initialize dashboard.

        Args:
            db_path: Path to database
        """
        self.db_path = db_path
        self.monitor = DataQualityMonitor(db_path)
        self.console = Console()

    def display_overview(self):
        """Display overview dashboard."""
        self.console.clear()

        # Get metrics
        metrics = self.monitor.calculate_quality_metrics()

        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="metrics", size=12),
            Layout(name="details")
        )

        # Header
        layout["header"].update(
            Panel(
                "[bold blue]GreenLang Emission Factors - Data Quality Dashboard[/bold blue]",
                subtitle=f"Generated: {DeterministicClock.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        )

        # Metrics summary
        metrics_table = Table(title="Quality Metrics Summary", box=box.ROUNDED)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Score", justify="right", style="yellow")
        metrics_table.add_column("Status", justify="center")

        metrics_data = [
            ("Overall Quality", metrics.overall_quality_score, self._get_status_emoji(metrics.overall_quality_score)),
            ("Completeness", metrics.completeness_score, self._get_status_emoji(metrics.completeness_score)),
            ("Accuracy", metrics.accuracy_score, self._get_status_emoji(metrics.accuracy_score)),
            ("Freshness", metrics.freshness_score, self._get_status_emoji(metrics.freshness_score)),
            ("Consistency", metrics.consistency_score, self._get_status_emoji(metrics.consistency_score)),
        ]

        for metric_name, score, status in metrics_data:
            metrics_table.add_row(
                metric_name,
                f"{score:.1f}/100",
                status
            )

        layout["metrics"].update(metrics_table)

        # Details table
        details_table = Table(title="Database Statistics", box=box.ROUNDED)
        details_table.add_column("Statistic", style="cyan")
        details_table.add_column("Value", justify="right", style="green")

        details_table.add_row("Total Factors", str(metrics.total_factors))
        details_table.add_row("Unique Sources", str(metrics.unique_sources))
        details_table.add_row("Stale Factors (>3 years)", str(metrics.stale_factors_count))
        details_table.add_row("Average Age (days)", f"{metrics.avg_age_days:.0f}")
        details_table.add_row("Coverage Gaps", str(metrics.coverage_metrics.get('category_gaps', 0)))

        layout["details"].update(details_table)

        self.console.print(layout)

    def display_coverage_report(self):
        """Display coverage analysis report."""
        self.console.clear()

        coverage = self.monitor.coverage_analyzer.generate_coverage_report()

        # Category coverage
        cat_table = Table(title="Category Coverage", box=box.ROUNDED)
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Total Factors", justify="right", style="green")
        cat_table.add_column("Subcategories", justify="right", style="yellow")

        cat_coverage = coverage['category_coverage']['coverage_by_category']
        for category, data in sorted(cat_coverage.items(), key=lambda x: x[1]['total'], reverse=True):
            cat_table.add_row(
                category,
                str(data['total']),
                str(len(data['subcategories']))
            )

        self.console.print(cat_table)

        # Geographic coverage
        if coverage['geographic_coverage']['by_country']:
            self.console.print("\n")
            geo_table = Table(title="Geographic Coverage", box=box.ROUNDED)
            geo_table.add_column("Country", style="cyan")
            geo_table.add_column("Factors", justify="right", style="green")

            geo_data = coverage['geographic_coverage']['by_country']
            for country, scopes in sorted(geo_data.items()):
                total_factors = sum(s['factor_count'] for s in scopes)
                geo_table.add_row(country, str(total_factors))

            self.console.print(geo_table)

        # Recommendations
        if coverage.get('recommendations'):
            self.console.print("\n")
            rec_panel = Panel(
                "\n".join(f"- {rec}" for rec in coverage['recommendations']),
                title="[bold yellow]Recommendations[/bold yellow]",
                border_style="yellow"
            )
            self.console.print(rec_panel)

    def display_source_analysis(self):
        """Display source diversity analysis."""
        self.console.clear()

        source_data = self.monitor.source_analyzer.analyze_source_distribution()

        # Source distribution
        source_table = Table(title="Source Distribution", box=box.ROUNDED)
        source_table.add_column("Source Organization", style="cyan")
        source_table.add_column("Factor Count", justify="right", style="green")
        source_table.add_column("Categories", justify="right", style="yellow")
        source_table.add_column("Latest Update", style="magenta")

        for source, data in sorted(
            source_data['sources'].items(),
            key=lambda x: x[1]['factor_count'],
            reverse=True
        )[:15]:  # Top 15 sources
            source_table.add_row(
                source,
                str(data['factor_count']),
                str(data['category_count']),
                data['newest_factor']
            )

        self.console.print(source_table)

        # Diversity metrics
        self.console.print("\n")
        diversity_panel = Panel(
            f"""
[bold]Source Diversity Metrics[/bold]

Unique Sources: [green]{source_data['unique_sources']}[/green]
Diversity Score: [yellow]{source_data['diversity_score']:.1f}/100[/yellow]
Top 3 Concentration: [cyan]{source_data['top_3_concentration_pct']:.1f}%[/cyan]
            """.strip(),
            border_style="blue"
        )
        self.console.print(diversity_panel)

    def display_freshness_analysis(self):
        """Display data freshness analysis."""
        self.console.clear()

        freshness = self.monitor.freshness_tracker.analyze_freshness()

        # Freshness distribution
        fresh_table = Table(title="Data Freshness Distribution", box=box.ROUNDED)
        fresh_table.add_column("Age Category", style="cyan")
        fresh_table.add_column("Count", justify="right", style="green")
        fresh_table.add_column("Percentage", justify="right", style="yellow")

        total = freshness['total_factors']
        dist = freshness['distribution']

        categories = [
            ("Fresh (<1 year)", dist['fresh_count']),
            ("Recent (1-2 years)", dist['recent_count']),
            ("Aging (2-3 years)", dist['aging_count']),
            ("Stale (>3 years)", dist['stale_count'])
        ]

        for category, count in categories:
            pct = (count / total * 100) if total > 0 else 0
            fresh_table.add_row(category, str(count), f"{pct:.1f}%")

        self.console.print(fresh_table)

        # Stale factors (oldest)
        if freshness['stale_factors']:
            self.console.print("\n")
            stale_table = Table(title="Oldest Factors (Top 10)", box=box.ROUNDED)
            stale_table.add_column("Factor ID", style="cyan")
            stale_table.add_column("Name", style="white")
            stale_table.add_column("Category", style="yellow")
            stale_table.add_column("Age (years)", justify="right", style="red")

            for factor in freshness['stale_factors']:
                stale_table.add_row(
                    factor['factor_id'],
                    factor['name'][:40],
                    factor['category'],
                    str(factor['age_years'])
                )

            self.console.print(stale_table)

    def _get_status_emoji(self, score: float) -> str:
        """Get status emoji based on score."""
        if score >= 90:
            return "[green]✓ Excellent[/green]"
        elif score >= 75:
            return "[yellow]○ Good[/yellow]"
        elif score >= 60:
            return "[orange]△ Fair[/orange]"
        else:
            return "[red]✗ Needs Improvement[/red]"

    def export_report(self, output_path: str, format: str = "json"):
        """
        Export monitoring report.

        Args:
            output_path: Output file path
            format: Export format (json, html)
        """
        report = self.monitor.generate_monitoring_report()

        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.console.print(f"[green]Report exported to: {output_path}[/green]")

        elif format == "html":
            # Generate HTML report (simplified)
            html = self._generate_html_report(report)
            with open(output_path, 'w') as f:
                f.write(html)
            self.console.print(f"[green]HTML report exported to: {output_path}[/green]")

    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report (simplified)."""
        metrics = report['quality_metrics']

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>GreenLang Data Quality Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        .metric {{ background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .score {{ font-size: 24px; font-weight: bold; color: #27ae60; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
    </style>
</head>
<body>
    <h1>GreenLang Emission Factors - Data Quality Report</h1>
    <p>Generated: {DeterministicClock.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="metric">
        <h2>Overall Quality Score</h2>
        <div class="score">{metrics['overall_quality_score']:.1f}/100</div>
    </div>

    <h2>Quality Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Score</th>
        </tr>
        <tr><td>Completeness</td><td>{metrics['completeness_score']:.1f}</td></tr>
        <tr><td>Accuracy</td><td>{metrics['accuracy_score']:.1f}</td></tr>
        <tr><td>Freshness</td><td>{metrics['freshness_score']:.1f}</td></tr>
        <tr><td>Consistency</td><td>{metrics['consistency_score']:.1f}</td></tr>
    </table>

    <h2>Database Statistics</h2>
    <ul>
        <li>Total Factors: {metrics['total_factors']}</li>
        <li>Unique Sources: {metrics['unique_sources']}</li>
        <li>Stale Factors: {metrics['stale_factors_count']}</li>
    </ul>

    <h2>Recommended Actions</h2>
    <ul>
        {"".join(f"<li>{action}</li>" for action in report.get('recommended_actions', []))}
    </ul>
</body>
</html>
        """
        return html


class PipelineCLI:
    """
    Command-line interface for pipeline operations.

    Provides commands for import, validation, monitoring, and workflow.
    """

    def __init__(self, db_path: str, data_dir: str):
        """
        Initialize CLI.

        Args:
            db_path: Path to database
            data_dir: Path to data directory
        """
        self.db_path = db_path
        self.data_dir = Path(data_dir)
        self.console = Console()
        self.dashboard = DataQualityDashboard(db_path)

    def run_import(
        self,
        validate: bool = True,
        rollback_on_failure: bool = True
    ):
        """
        Run import pipeline.

        Args:
            validate: Run pre-import validation
            rollback_on_failure: Rollback on failure
        """
        self.console.print("[bold blue]Starting Import Pipeline[/bold blue]\n")

        # Find YAML files
        yaml_files = [
            str(self.data_dir / 'emission_factors_registry.yaml'),
            str(self.data_dir / 'emission_factors_expansion_phase1.yaml'),
            str(self.data_dir / 'emission_factors_expansion_phase2.yaml')
        ]

        # Create import job
        job = ImportJob(
            job_id=f"import_{DeterministicClock.now().strftime('%Y%m%d_%H%M%S')}",
            job_name=f"Manual Import {DeterministicClock.now().strftime('%Y-%m-%d %H:%M')}",
            source_files=yaml_files,
            target_database=self.db_path,
            validate_before_import=validate,
            triggered_by="cli_user",
            trigger_type="manual"
        )

        # Create pipeline
        pipeline = AutomatedImportPipeline(self.db_path)

        # Run with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Running import...", total=None)

            import asyncio
            result = asyncio.run(
                pipeline.execute_import(job, validate, rollback_on_failure)
            )

            progress.update(task, completed=True)

        # Display results
        self._display_import_results(result)

    def _display_import_results(self, job: ImportJob):
        """Display import job results."""
        self.console.print("\n")

        if job.status == ImportStatus.COMPLETED:
            result_panel = Panel(
                f"""
[bold green]Import Completed Successfully[/bold green]

Total Processed: {job.total_factors_processed}
Successful: [green]{job.successful_imports}[/green]
Failed: [red]{job.failed_imports}[/red]
Duplicates: [yellow]{job.duplicate_factors}[/yellow]
Success Rate: {job.success_rate:.1f}%
Duration: {job.duration_seconds:.2f}s
                """.strip(),
                border_style="green"
            )
        else:
            result_panel = Panel(
                f"""
[bold red]Import Failed[/bold red]

Status: {job.status.value}
Errors: {len(job.errors)}
                """.strip(),
                border_style="red"
            )

        self.console.print(result_panel)

    def validate_data(self):
        """Run validation on current data."""
        self.console.print("[bold blue]Running Data Validation[/bold blue]\n")

        from .validator import EmissionFactorValidator
        import asyncio

        validator = EmissionFactorValidator()

        yaml_files = list(self.data_dir.glob("emission_factors*.yaml"))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task(f"Validating {len(yaml_files)} files...", total=None)

            results = []
            for yaml_file in yaml_files:
                result = asyncio.run(validator.validate_file(str(yaml_file)))
                results.append((yaml_file.name, result))

            progress.update(task, completed=True)

        # Display results
        self.console.print("\n")
        val_table = Table(title="Validation Results", box=box.ROUNDED)
        val_table.add_column("File", style="cyan")
        val_table.add_column("Quality Score", justify="right", style="yellow")
        val_table.add_column("Valid/Total", justify="right", style="green")
        val_table.add_column("Errors", justify="right", style="red")

        for filename, result in results:
            val_table.add_row(
                filename,
                f"{result.quality_score:.1f}",
                f"{result.valid_records}/{result.total_records}",
                str(len(result.errors))
            )

        self.console.print(val_table)

    def show_dashboard(self):
        """Show interactive dashboard."""
        self.dashboard.display_overview()

    def show_coverage(self):
        """Show coverage report."""
        self.dashboard.display_coverage_report()

    def show_sources(self):
        """Show source analysis."""
        self.dashboard.display_source_analysis()

    def show_freshness(self):
        """Show freshness analysis."""
        self.dashboard.display_freshness_analysis()

    def export_report(self, output_path: str, format: str = "json"):
        """
        Export monitoring report.

        Args:
            output_path: Output file path
            format: Export format (json, html)
        """
        self.dashboard.export_report(output_path, format)
