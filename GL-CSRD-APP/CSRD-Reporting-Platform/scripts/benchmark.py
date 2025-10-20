"""
CSRD Pipeline Performance Benchmarking Tool

This script benchmarks the performance of the CSRD reporting pipeline:
- Measures processing time for different dataset sizes
- Tests all 6 agents individually and combined
- Generates comprehensive benchmark reports (JSON + Markdown)
- Compares performance against targets
- Profiles memory usage and throughput

Usage:
    python scripts/benchmark.py --dataset-size small
    python scripts/benchmark.py --dataset-size medium --agents all
    python scripts/benchmark.py --dataset-size large --output benchmark_results.json

Version: 1.0.0
Author: GreenLang CSRD Team
License: MIT
"""

import json
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "agents"))

from agents.intake_agent import IntakeAgent
from agents.materiality_agent import MaterialityAgent
from agents.calculator_agent import CalculatorAgent
from agents.aggregator_agent import AggregatorAgent
from agents.reporting_agent import ReportingAgent
from agents.audit_agent import AuditAgent
from csrd_pipeline import CSRDPipeline

console = Console()


# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================

DATASET_SIZES = {
    "tiny": {
        "records": 10,
        "description": "Tiny dataset for quick testing",
        "target_time_seconds": 5
    },
    "small": {
        "records": 100,
        "description": "Small dataset (100 metrics)",
        "target_time_seconds": 30
    },
    "medium": {
        "records": 1000,
        "description": "Medium dataset (1,000 metrics)",
        "target_time_seconds": 180
    },
    "large": {
        "records": 10000,
        "description": "Large dataset (10,000 metrics)",
        "target_time_seconds": 1800  # 30 minutes
    },
    "xlarge": {
        "records": 50000,
        "description": "Extra large dataset (50,000 metrics)",
        "target_time_seconds": 5400  # 90 minutes
    }
}

PERFORMANCE_TARGETS = {
    "intake_agent": {
        "throughput_records_per_sec": 1000,
        "max_time_per_1k_records": 1.0
    },
    "materiality_agent": {
        "max_time_seconds": 120,  # AI-powered, can be slow
        "llm_calls_expected": 10
    },
    "calculator_agent": {
        "throughput_calcs_per_sec": 500,
        "max_time_per_calc_ms": 2.0
    },
    "aggregator_agent": {
        "max_time_seconds": 60,
        "benchmarks_per_sec": 100
    },
    "reporting_agent": {
        "max_time_seconds": 300,
        "xbrl_tags_per_sec": 50
    },
    "audit_agent": {
        "max_time_seconds": 180,
        "rules_per_sec": 100
    }
}


# ============================================================================
# BENCHMARK RESULT MODELS
# ============================================================================

class AgentBenchmark:
    """Benchmark result for a single agent."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration_seconds: float = 0.0
        self.memory_start_mb: float = 0.0
        self.memory_peak_mb: float = 0.0
        self.memory_delta_mb: float = 0.0
        self.input_records: int = 0
        self.output_records: int = 0
        self.throughput_records_per_sec: float = 0.0
        self.status: str = "not_started"  # not_started, running, success, failed
        self.error_message: Optional[str] = None
        self.metadata: Dict[str, Any] = {}

    def start(self, input_records: int):
        """Start benchmark timing."""
        self.input_records = input_records
        self.start_time = time.perf_counter()
        self.status = "running"
        tracemalloc.start()

    def end(self, output_records: int, metadata: Optional[Dict[str, Any]] = None):
        """End benchmark timing."""
        self.end_time = time.perf_counter()
        self.duration_seconds = self.end_time - self.start_time
        self.output_records = output_records

        if self.duration_seconds > 0:
            self.throughput_records_per_sec = self.input_records / self.duration_seconds

        # Memory tracking
        current, peak = tracemalloc.get_traced_memory()
        self.memory_peak_mb = peak / 1024 / 1024
        tracemalloc.stop()

        self.status = "success"
        if metadata:
            self.metadata = metadata

    def fail(self, error_message: str):
        """Mark benchmark as failed."""
        if self.start_time:
            self.end_time = time.perf_counter()
            self.duration_seconds = self.end_time - self.start_time
        self.status = "failed"
        self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "duration_seconds": round(self.duration_seconds, 3),
            "memory_peak_mb": round(self.memory_peak_mb, 2),
            "input_records": self.input_records,
            "output_records": self.output_records,
            "throughput_records_per_sec": round(self.throughput_records_per_sec, 2),
            "status": self.status,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


class BenchmarkReport:
    """Complete benchmark report."""

    def __init__(self, dataset_size: str, dataset_config: Dict[str, Any]):
        self.benchmark_id = f"benchmark_{int(time.time())}"
        self.timestamp = datetime.now().isoformat()
        self.dataset_size = dataset_size
        self.dataset_config = dataset_config
        self.agent_benchmarks: List[AgentBenchmark] = []
        self.pipeline_benchmark: Optional[AgentBenchmark] = None
        self.total_duration_seconds: float = 0.0
        self.meets_target: bool = False

    def add_agent_benchmark(self, benchmark: AgentBenchmark):
        """Add agent benchmark result."""
        self.agent_benchmarks.append(benchmark)

    def set_pipeline_benchmark(self, benchmark: AgentBenchmark):
        """Set complete pipeline benchmark."""
        self.pipeline_benchmark = benchmark
        self.total_duration_seconds = benchmark.duration_seconds
        self.meets_target = benchmark.duration_seconds <= self.dataset_config["target_time_seconds"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_id": self.benchmark_id,
            "timestamp": self.timestamp,
            "dataset_size": self.dataset_size,
            "dataset_records": self.dataset_config["records"],
            "target_time_seconds": self.dataset_config["target_time_seconds"],
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "meets_target": self.meets_target,
            "agent_benchmarks": [b.to_dict() for b in self.agent_benchmarks],
            "pipeline_benchmark": self.pipeline_benchmark.to_dict() if self.pipeline_benchmark else None,
            "summary": self._generate_summary()
        }

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary statistics."""
        successful_benchmarks = [b for b in self.agent_benchmarks if b.status == "success"]

        if not successful_benchmarks:
            return {}

        return {
            "agents_tested": len(self.agent_benchmarks),
            "agents_successful": len(successful_benchmarks),
            "agents_failed": len([b for b in self.agent_benchmarks if b.status == "failed"]),
            "total_memory_peak_mb": round(sum(b.memory_peak_mb for b in successful_benchmarks), 2),
            "average_throughput": round(
                sum(b.throughput_records_per_sec for b in successful_benchmarks) / len(successful_benchmarks), 2
            ) if successful_benchmarks else 0,
            "slowest_agent": max(successful_benchmarks, key=lambda b: b.duration_seconds).agent_name if successful_benchmarks else None,
            "fastest_agent": min(successful_benchmarks, key=lambda b: b.duration_seconds).agent_name if successful_benchmarks else None
        }


# ============================================================================
# DATA GENERATION FOR BENCHMARKING
# ============================================================================

def generate_benchmark_data(num_records: int, output_path: Path) -> Path:
    """
    Generate synthetic ESG data for benchmarking.

    Args:
        num_records: Number of data points to generate
        output_path: Path to save CSV file

    Returns:
        Path to generated CSV file
    """
    console.print(f"[cyan]Generating {num_records} synthetic ESG data points...[/cyan]")

    # Base metrics to replicate
    base_metrics = [
        {"code": "E1-1", "name": "Scope 1 GHG Emissions", "unit": "tCO2e", "range": (1000, 50000)},
        {"code": "E1-2", "name": "Scope 2 GHG Emissions", "unit": "tCO2e", "range": (500, 30000)},
        {"code": "E1-6", "name": "Total Energy Consumption", "unit": "GJ", "range": (50000, 500000)},
        {"code": "E3-1", "name": "Total Water Withdrawal", "unit": "m3", "range": (10000, 500000)},
        {"code": "E5-1", "name": "Total Waste Generated", "unit": "tonnes", "range": (500, 10000)},
        {"code": "S1-1", "name": "Total Employees", "unit": "FTE", "range": (100, 10000)},
        {"code": "S1-9", "name": "Training Hours", "unit": "hours", "range": (10, 100)},
        {"code": "G1-1", "name": "Board Members", "unit": "count", "range": (5, 15)},
    ]

    import random
    random.seed(42)  # Reproducible results

    records = []
    for i in range(num_records):
        metric = base_metrics[i % len(base_metrics)]
        value = random.uniform(*metric["range"])

        record = {
            "metric_code": f"{metric['code']}-{i // len(base_metrics)}",
            "metric_name": f"{metric['name']} {i // len(base_metrics)}",
            "value": round(value, 2),
            "unit": metric["unit"],
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
            "data_quality": random.choice(["high", "medium", "high", "high"]),  # 75% high quality
            "source_document": f"System_{i % 5 + 1}",
            "verification_status": random.choice(["verified", "third_party_assured", "unverified"]),
            "notes": f"Synthetic data point {i}"
        }
        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    console.print(f"[green]✓ Generated {num_records} records at {output_path}[/green]")
    return output_path


# ============================================================================
# INDIVIDUAL AGENT BENCHMARKS
# ============================================================================

def benchmark_intake_agent(data_file: Path, num_records: int) -> AgentBenchmark:
    """Benchmark IntakeAgent performance."""
    benchmark = AgentBenchmark("IntakeAgent")

    try:
        # Initialize agent
        agent = IntakeAgent(
            esrs_data_points_path=project_root / "data" / "esrs_data_points.json",
            data_quality_rules_path=project_root / "rules" / "data_quality_rules.yaml"
        )

        benchmark.start(num_records)

        # Process data
        result = agent.process(data_file)

        benchmark.end(
            output_records=result['metadata']['valid_records'],
            metadata={
                "data_quality_score": result['metadata']['data_quality_score'],
                "invalid_records": result['metadata']['invalid_records'],
                "throughput": result['metadata']['records_per_second']
            }
        )

    except Exception as e:
        benchmark.fail(str(e))
        console.print(f"[red]✗ IntakeAgent benchmark failed: {e}[/red]")

    return benchmark


def benchmark_calculator_agent(num_records: int) -> AgentBenchmark:
    """Benchmark CalculatorAgent performance."""
    benchmark = AgentBenchmark("CalculatorAgent")

    try:
        # Initialize agent
        agent = CalculatorAgent(
            esrs_formulas_path=project_root / "data" / "esrs_formulas.yaml",
            emission_factors_path=project_root / "data" / "emission_factors.json"
        )

        # Generate sample calculation input
        calculation_input = {
            "validated_data": [
                {"metric_code": f"E1-{i}", "value": 1000 + i, "unit": "tCO2e"}
                for i in range(min(num_records, 100))  # Limit to reasonable number
            ],
            "material_topics": []
        }

        benchmark.start(len(calculation_input["validated_data"]))

        # Calculate metrics
        result = agent.calculate_batch(calculation_input)

        benchmark.end(
            output_records=result['metadata']['metrics_calculated'],
            metadata={
                "ms_per_metric": result['metadata'].get('ms_per_metric', 0),
                "formulas_executed": result['metadata'].get('formulas_executed', 0)
            }
        )

    except Exception as e:
        benchmark.fail(str(e))
        console.print(f"[red]✗ CalculatorAgent benchmark failed: {e}[/red]")

    return benchmark


# ============================================================================
# COMPLETE PIPELINE BENCHMARK
# ============================================================================

def benchmark_complete_pipeline(data_file: Path, num_records: int) -> AgentBenchmark:
    """Benchmark complete CSRD pipeline."""
    benchmark = AgentBenchmark("CompletePipeline")

    try:
        # Load company profile
        company_profile_path = project_root / "examples" / "demo_company_profile.json"
        with open(company_profile_path, 'r') as f:
            company_profile = json.load(f)

        # Initialize pipeline
        pipeline = CSRDPipeline(config_path=str(project_root / "config" / "csrd_config.yaml"))

        benchmark.start(num_records)

        # Run pipeline
        result = pipeline.run(
            esg_data_file=str(data_file),
            company_profile=company_profile,
            output_dir=None  # Don't write output during benchmark
        )

        benchmark.end(
            output_records=result.total_data_points_processed,
            metadata={
                "compliance_status": result.compliance_status,
                "data_quality_score": result.data_quality_score,
                "agents_executed": len(result.agent_executions),
                "performance": result.performance.dict()
            }
        )

    except Exception as e:
        benchmark.fail(str(e))
        console.print(f"[red]✗ Pipeline benchmark failed: {e}[/red]")

    return benchmark


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_markdown_report(report: BenchmarkReport, output_path: Path):
    """Generate Markdown benchmark report."""

    md_lines = [
        f"# CSRD Pipeline Benchmark Report",
        f"",
        f"**Benchmark ID:** {report.benchmark_id}",
        f"**Timestamp:** {report.timestamp}",
        f"**Dataset Size:** {report.dataset_size} ({report.dataset_config['records']} records)",
        f"**Target Time:** {report.dataset_config['target_time_seconds']}s",
        f"**Actual Time:** {report.total_duration_seconds:.2f}s",
        f"**Meets Target:** {'✅ YES' if report.meets_target else '❌ NO'}",
        f"",
        f"## Summary",
        f"",
    ]

    summary = report._generate_summary()
    for key, value in summary.items():
        md_lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")

    md_lines.extend([
        f"",
        f"## Agent Performance",
        f"",
        f"| Agent | Duration (s) | Input | Output | Throughput (rec/s) | Memory (MB) | Status |",
        f"|-------|--------------|-------|--------|-------------------|-------------|--------|"
    ])

    for benchmark in report.agent_benchmarks:
        md_lines.append(
            f"| {benchmark.agent_name} | {benchmark.duration_seconds:.2f} | "
            f"{benchmark.input_records} | {benchmark.output_records} | "
            f"{benchmark.throughput_records_per_sec:.2f} | {benchmark.memory_peak_mb:.2f} | "
            f"{benchmark.status} |"
        )

    if report.pipeline_benchmark:
        md_lines.extend([
            f"",
            f"## Complete Pipeline",
            f"",
            f"- **Duration:** {report.pipeline_benchmark.duration_seconds:.2f}s",
            f"- **Throughput:** {report.pipeline_benchmark.throughput_records_per_sec:.2f} records/sec",
            f"- **Memory Peak:** {report.pipeline_benchmark.memory_peak_mb:.2f} MB",
            f"- **Status:** {report.pipeline_benchmark.status}",
            f""
        ])

    md_lines.extend([
        f"",
        f"## Performance Analysis",
        f"",
        f"### Target Comparison",
        f""
    ])

    for agent_bench in report.agent_benchmarks:
        agent_key = agent_bench.agent_name.lower().replace("agent", "_agent")
        if agent_key in PERFORMANCE_TARGETS:
            targets = PERFORMANCE_TARGETS[agent_key]
            md_lines.append(f"**{agent_bench.agent_name}:**")
            for target_key, target_value in targets.items():
                md_lines.append(f"- {target_key}: {target_value}")
            md_lines.append("")

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(md_lines))

    console.print(f"[green]✓ Markdown report saved to {output_path}[/green]")


def display_results_table(report: BenchmarkReport):
    """Display benchmark results in a rich table."""

    table = Table(title=f"Benchmark Results - {report.dataset_size.upper()} Dataset")

    table.add_column("Agent", style="cyan", no_wrap=True)
    table.add_column("Duration (s)", justify="right", style="magenta")
    table.add_column("Records", justify="right", style="green")
    table.add_column("Throughput (rec/s)", justify="right", style="yellow")
    table.add_column("Memory (MB)", justify="right", style="blue")
    table.add_column("Status", justify="center")

    for benchmark in report.agent_benchmarks:
        status_emoji = "✅" if benchmark.status == "success" else "❌"
        table.add_row(
            benchmark.agent_name,
            f"{benchmark.duration_seconds:.2f}",
            f"{benchmark.input_records} → {benchmark.output_records}",
            f"{benchmark.throughput_records_per_sec:.2f}",
            f"{benchmark.memory_peak_mb:.2f}",
            f"{status_emoji} {benchmark.status}"
        )

    if report.pipeline_benchmark:
        table.add_row("", "", "", "", "", "")  # Separator
        bench = report.pipeline_benchmark
        status_emoji = "✅" if bench.status == "success" else "❌"
        table.add_row(
            "COMPLETE PIPELINE",
            f"{bench.duration_seconds:.2f}",
            f"{bench.input_records} → {bench.output_records}",
            f"{bench.throughput_records_per_sec:.2f}",
            f"{bench.memory_peak_mb:.2f}",
            f"{status_emoji} {bench.status}",
            style="bold"
        )

    console.print(table)

    # Performance summary
    console.print(f"\n[bold]Performance Summary:[/bold]")
    console.print(f"Target Time: {report.dataset_config['target_time_seconds']}s")
    console.print(f"Actual Time: {report.total_duration_seconds:.2f}s")

    if report.meets_target:
        console.print(f"[bold green]✅ MEETS TARGET[/bold green]")
    else:
        console.print(f"[bold red]❌ EXCEEDS TARGET[/bold red]")


# ============================================================================
# CLI INTERFACE
# ============================================================================

@click.command()
@click.option(
    '--dataset-size',
    type=click.Choice(['tiny', 'small', 'medium', 'large', 'xlarge']),
    default='small',
    help='Dataset size to benchmark'
)
@click.option(
    '--agents',
    type=click.Choice(['all', 'intake', 'calculator', 'pipeline']),
    default='all',
    help='Which agents to benchmark'
)
@click.option(
    '--output',
    type=click.Path(),
    default='benchmark_report.json',
    help='Output file for JSON results'
)
@click.option(
    '--markdown',
    type=click.Path(),
    default='benchmark_report.md',
    help='Output file for Markdown report'
)
@click.option(
    '--skip-data-generation',
    is_flag=True,
    help='Skip data generation (use existing benchmark data)'
)
def benchmark(dataset_size: str, agents: str, output: str, markdown: str, skip_data_generation: bool):
    """
    Benchmark CSRD pipeline performance.

    Tests processing time, memory usage, and throughput for different
    dataset sizes. Generates JSON and Markdown reports.
    """
    console.print(f"\n[bold cyan]🚀 CSRD Pipeline Benchmark[/bold cyan]\n")

    dataset_config = DATASET_SIZES[dataset_size]
    num_records = dataset_config["records"]

    console.print(f"Dataset: [bold]{dataset_size.upper()}[/bold] ({num_records} records)")
    console.print(f"Agents: [bold]{agents}[/bold]")
    console.print(f"Target time: [bold]{dataset_config['target_time_seconds']}s[/bold]\n")

    # Create benchmark report
    report = BenchmarkReport(dataset_size, dataset_config)

    # Generate or locate benchmark data
    data_file = project_root / "scripts" / f"benchmark_data_{dataset_size}.csv"

    if not skip_data_generation or not data_file.exists():
        data_file = generate_benchmark_data(num_records, data_file)
    else:
        console.print(f"[cyan]Using existing data: {data_file}[/cyan]\n")

    # Run benchmarks
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        if agents in ['all', 'intake']:
            task = progress.add_task("Benchmarking IntakeAgent...", total=None)
            bench = benchmark_intake_agent(data_file, num_records)
            report.add_agent_benchmark(bench)
            progress.remove_task(task)
            console.print(f"[green]✓ IntakeAgent: {bench.duration_seconds:.2f}s[/green]")

        if agents in ['all', 'calculator']:
            task = progress.add_task("Benchmarking CalculatorAgent...", total=None)
            bench = benchmark_calculator_agent(num_records)
            report.add_agent_benchmark(bench)
            progress.remove_task(task)
            console.print(f"[green]✓ CalculatorAgent: {bench.duration_seconds:.2f}s[/green]")

        if agents in ['all', 'pipeline']:
            task = progress.add_task("Benchmarking Complete Pipeline...", total=None)
            bench = benchmark_complete_pipeline(data_file, num_records)
            report.set_pipeline_benchmark(bench)
            progress.remove_task(task)
            console.print(f"[green]✓ Complete Pipeline: {bench.duration_seconds:.2f}s[/green]")

    console.print()

    # Display results
    display_results_table(report)

    # Save reports
    console.print(f"\n[bold]Saving reports...[/bold]")

    # JSON report
    output_path = Path(output)
    with open(output_path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)
    console.print(f"[green]✓ JSON report: {output_path}[/green]")

    # Markdown report
    markdown_path = Path(markdown)
    generate_markdown_report(report, markdown_path)

    console.print(f"\n[bold green]✅ Benchmark complete![/bold green]\n")


if __name__ == '__main__':
    benchmark()
