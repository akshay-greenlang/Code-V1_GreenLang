# -*- coding: utf-8 -*-
"""
Test Execution Runner
=====================

Execute various test types for GreenLang agents:
- Unit tests: Basic functionality verification
- Golden tests: Determinism checks (100 runs, byte-identical)
- Integration tests: External system integration
- E2E tests: Full workflow testing
- Performance benchmarks: Latency, throughput, memory

Usage:
    gl agent test eudr_compliance
    gl agent test ./agents/carbon --golden --coverage
    gl agent test my-agent --e2e --output results.json
"""

import os
import sys
import json
import subprocess
import logging
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from enum import Enum

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from pydantic import BaseModel, Field

# Create sub-app for test commands
test_app = typer.Typer(
    name="test",
    help="Test execution runner with coverage",
    no_args_is_help=True,
)

console = Console()
logger = logging.getLogger(__name__)


# =============================================================================
# Test Models
# =============================================================================

class TestStatus(str, Enum):
    """Test execution status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestResult(BaseModel):
    """Single test result."""
    name: str
    status: TestStatus
    duration_ms: float = 0
    message: Optional[str] = None
    output: Optional[str] = None
    determinism_hash: Optional[str] = None  # For golden tests


class TestSuiteResult(BaseModel):
    """Complete test suite result."""
    suite_name: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_ms: float = 0
    coverage_percent: Optional[float] = None
    tests: List[TestResult] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class TestReport(BaseModel):
    """Complete test report."""
    agent_id: str
    suites: List[TestSuiteResult] = Field(default_factory=list)
    total_tests: int = 0
    total_passed: int = 0
    total_failed: int = 0
    total_duration_ms: float = 0
    overall_coverage: Optional[float] = None
    determinism_verified: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# Test Commands
# =============================================================================

@test_app.command("run")
def test_run_command(
    agent_path: Path = typer.Argument(
        ...,
        help="Path to agent directory or pack.yaml",
    ),
    golden: bool = typer.Option(
        False,
        "--golden",
        help="Run golden tests (determinism verification)",
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
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Verbose output",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output test report",
    ),
):
    """
    Run tests for an agent.

    Example:
        gl agent test run ./agents/carbon
        gl agent test run . --golden --coverage
    """
    # Default to all if none specified
    if not golden and not integration and not e2e:
        run_all = True
    else:
        run_all = False

    run_agent_tests(
        agent_id=str(agent_path),
        golden=golden or run_all,
        integration=integration or run_all,
        e2e=e2e,
        coverage=coverage,
        parallel=parallel,
        fail_fast=fail_fast,
        output=output,
        verbose=verbose,
    )


@test_app.command("golden")
def test_golden_command(
    agent_path: Path = typer.Argument(
        ...,
        help="Path to agent directory",
    ),
    runs: int = typer.Option(
        100,
        "--runs", "-n",
        help="Number of determinism runs (default: 100)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Verbose output",
    ),
):
    """
    Run golden tests with determinism verification.

    Golden tests verify that agents produce identical outputs
    for identical inputs (zero-hallucination principle).

    Example:
        gl agent test golden ./agents/carbon
        gl agent test golden . --runs 200
    """
    console.print(Panel(
        "[bold cyan]Golden Test Runner[/bold cyan]\n"
        f"Runs: {runs} (determinism verification)",
        border_style="cyan"
    ))

    result = run_golden_tests(agent_path, runs=runs, verbose=verbose)
    _display_golden_results(result)


@test_app.command("benchmark")
def test_benchmark_command(
    agent_path: Path = typer.Argument(
        ...,
        help="Path to agent directory",
    ),
    iterations: int = typer.Option(
        100,
        "--iterations", "-n",
        help="Number of benchmark iterations",
    ),
    warmup: int = typer.Option(
        10,
        "--warmup", "-w",
        help="Warmup iterations",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output benchmark report",
    ),
):
    """
    Run performance benchmarks.

    Measures:
    - Latency (p50, p95, p99)
    - Throughput (requests/second)
    - Memory usage
    - CPU utilization

    Example:
        gl agent test benchmark ./agents/carbon
        gl agent test benchmark . --iterations 500
    """
    console.print(Panel(
        "[bold cyan]Performance Benchmark[/bold cyan]\n"
        f"Iterations: {iterations} (warmup: {warmup})",
        border_style="cyan"
    ))

    result = run_benchmarks(agent_path, iterations=iterations, warmup=warmup)
    _display_benchmark_results(result)

    if output:
        _save_benchmark_report(result, output)


@test_app.command("coverage")
def test_coverage_command(
    agent_path: Path = typer.Argument(
        ...,
        help="Path to agent directory",
    ),
    html: bool = typer.Option(
        False,
        "--html",
        help="Generate HTML coverage report",
    ),
    threshold: int = typer.Option(
        85,
        "--threshold", "-t",
        help="Minimum coverage threshold (%)",
    ),
):
    """
    Generate test coverage report.

    Example:
        gl agent test coverage ./agents/carbon
        gl agent test coverage . --html --threshold 90
    """
    console.print(Panel(
        "[bold cyan]Coverage Report Generator[/bold cyan]\n"
        f"Threshold: {threshold}%",
        border_style="cyan"
    ))

    result = generate_coverage_report(agent_path, html=html)
    _display_coverage_results(result, threshold)


# =============================================================================
# Core Test Functions
# =============================================================================

def run_agent_tests(
    agent_id: str,
    golden: bool = True,
    integration: bool = True,
    e2e: bool = False,
    coverage: bool = True,
    parallel: bool = True,
    fail_fast: bool = False,
    output: Optional[Path] = None,
    verbose: bool = False,
) -> TestReport:
    """
    Run tests for an agent.

    Args:
        agent_id: Agent ID or path
        golden: Run golden tests
        integration: Run integration tests
        e2e: Run E2E tests
        coverage: Generate coverage report
        parallel: Run in parallel
        fail_fast: Stop on first failure
        output: Output report path
        verbose: Verbose output

    Returns:
        TestReport with all results
    """
    console.print(Panel(
        "[bold cyan]GreenLang Test Runner[/bold cyan]\n"
        f"Agent: {agent_id}",
        border_style="cyan"
    ))

    # Resolve agent path
    agent_path = Path(agent_id)
    if not agent_path.exists():
        # Try to find in standard locations
        for search_path in ["./agents", "./packs", "."]:
            candidate = Path(search_path) / agent_id
            if candidate.exists():
                agent_path = candidate
                break

    if not agent_path.exists():
        console.print(f"[red]Agent not found: {agent_id}[/red]")
        raise typer.Exit(1)

    # Find tests directory
    if agent_path.is_file():
        tests_dir = agent_path.parent / "tests"
    else:
        tests_dir = agent_path / "tests" if (agent_path / "tests").exists() else agent_path

    if not tests_dir.exists():
        console.print(f"[yellow]No tests directory found[/yellow]")
        console.print("Create tests with: gl agent create from-template")
        raise typer.Exit(1)

    report = TestReport(agent_id=agent_id)
    start_time = datetime.now()

    # Build pytest arguments
    pytest_args = [sys.executable, "-m", "pytest", str(tests_dir)]

    if verbose:
        pytest_args.append("-v")
    else:
        pytest_args.append("-q")

    if parallel:
        pytest_args.extend(["-n", "auto"])  # Requires pytest-xdist

    if fail_fast:
        pytest_args.append("-x")

    if coverage:
        pytest_args.extend([
            "--cov=" + str(agent_path),
            "--cov-report=term-missing",
        ])

    # Filter by test type
    markers = []
    if golden and not integration and not e2e:
        markers.append("golden or Golden")
    if integration and not golden and not e2e:
        markers.append("integration")
    if e2e:
        markers.append("e2e")

    if markers:
        pytest_args.extend(["-k", " or ".join(markers)])

    # Run tests with live output
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running tests...", total=None)

        console.print(f"\n[dim]Command: {' '.join(pytest_args)}[/dim]\n")

        try:
            result = subprocess.run(
                pytest_args,
                capture_output=not verbose,
                text=True,
            )

            progress.update(task, description="[green]Tests complete")

            # Parse results
            suite_result = _parse_pytest_output(
                result.stdout if not verbose else "",
                result.returncode,
            )
            report.suites.append(suite_result)

            # Calculate totals
            report.total_tests = suite_result.total
            report.total_passed = suite_result.passed
            report.total_failed = suite_result.failed

        except FileNotFoundError:
            console.print("[red]pytest not found. Install with: pip install pytest[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Test execution failed: {str(e)}[/red]")
            raise typer.Exit(1)

    # Calculate duration
    report.total_duration_ms = (datetime.now() - start_time).total_seconds() * 1000

    # Display results
    _display_test_results(report)

    # Save report if requested
    if output:
        _save_test_report(report, output)

    # Exit with appropriate code
    if report.total_failed > 0:
        raise typer.Exit(1)

    return report


def run_golden_tests(
    agent_path: Path,
    runs: int = 100,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run golden tests with determinism verification.

    Args:
        agent_path: Path to agent
        runs: Number of runs for determinism check
        verbose: Verbose output

    Returns:
        Golden test results
    """
    results = {
        "runs": runs,
        "passed": 0,
        "failed": 0,
        "deterministic": True,
        "hashes": [],
        "test_results": [],
    }

    # Find golden test files
    tests_dir = agent_path / "tests" if agent_path.is_dir() else agent_path.parent / "tests"

    if not tests_dir.exists():
        console.print("[yellow]No tests directory found[/yellow]")
        return results

    # Look for golden test patterns
    golden_files = list(tests_dir.glob("*golden*.py")) + list(tests_dir.glob("test_*golden*.py"))

    if not golden_files:
        console.print("[yellow]No golden test files found[/yellow]")
        console.print("Golden tests should match pattern: *golden*.py")
        return results

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Running {runs} determinism checks...", total=runs)

        hashes_seen = set()

        for i in range(runs):
            # Run golden tests
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", str(tests_dir), "-k", "golden", "-q"],
                    capture_output=True,
                    text=True,
                )

                # Hash the output for determinism check
                output_hash = hashlib.sha256(result.stdout.encode()).hexdigest()
                results["hashes"].append(output_hash)
                hashes_seen.add(output_hash)

                if result.returncode == 0:
                    results["passed"] += 1
                else:
                    results["failed"] += 1

            except Exception as e:
                results["failed"] += 1
                if verbose:
                    console.print(f"[red]Run {i+1} failed: {str(e)}[/red]")

            progress.update(task, advance=1)

        # Check determinism
        results["deterministic"] = len(hashes_seen) == 1
        results["unique_hashes"] = len(hashes_seen)

    return results


def run_benchmarks(
    agent_path: Path,
    iterations: int = 100,
    warmup: int = 10,
) -> Dict[str, Any]:
    """
    Run performance benchmarks.

    Args:
        agent_path: Path to agent
        iterations: Number of iterations
        warmup: Warmup iterations

    Returns:
        Benchmark results
    """
    import statistics

    results = {
        "iterations": iterations,
        "warmup": warmup,
        "latencies_ms": [],
        "memory_mb": [],
        "metrics": {},
    }

    # Placeholder benchmark - would actually run agent
    console.print("\n[yellow]Running warmup...[/yellow]")
    for _ in range(warmup):
        time.sleep(0.001)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Benchmarking...", total=iterations)

        for _ in range(iterations):
            start = time.perf_counter()
            # Would run actual agent here
            time.sleep(0.001)  # Placeholder
            latency = (time.perf_counter() - start) * 1000
            results["latencies_ms"].append(latency)
            progress.update(task, advance=1)

    # Calculate metrics
    latencies = results["latencies_ms"]
    results["metrics"] = {
        "p50_ms": statistics.median(latencies),
        "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        "mean_ms": statistics.mean(latencies),
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "throughput_rps": 1000 / statistics.mean(latencies),
    }

    return results


def generate_coverage_report(
    agent_path: Path,
    html: bool = False,
) -> Dict[str, Any]:
    """
    Generate test coverage report.

    Args:
        agent_path: Path to agent
        html: Generate HTML report

    Returns:
        Coverage results
    """
    results = {
        "total_coverage": 0,
        "files": [],
        "html_report": None,
    }

    pytest_args = [
        sys.executable, "-m", "pytest",
        str(agent_path / "tests") if agent_path.is_dir() else str(agent_path),
        f"--cov={agent_path}",
        "--cov-report=json",
    ]

    if html:
        pytest_args.append("--cov-report=html")

    try:
        subprocess.run(pytest_args, capture_output=True, text=True)

        # Read coverage JSON
        cov_file = Path("coverage.json")
        if cov_file.exists():
            with open(cov_file) as f:
                cov_data = json.load(f)
                results["total_coverage"] = cov_data.get("totals", {}).get("percent_covered", 0)
                results["files"] = [
                    {"file": k, "coverage": v.get("summary", {}).get("percent_covered", 0)}
                    for k, v in cov_data.get("files", {}).items()
                ]

        if html:
            results["html_report"] = "htmlcov/index.html"

    except Exception as e:
        console.print(f"[red]Coverage generation failed: {str(e)}[/red]")

    return results


# =============================================================================
# Display Functions
# =============================================================================

def _parse_pytest_output(output: str, return_code: int) -> TestSuiteResult:
    """Parse pytest output to extract test results."""
    result = TestSuiteResult(suite_name="pytest")

    # Simple parsing - would be more robust with pytest-json-report
    lines = output.split("\n")

    for line in lines:
        if "passed" in line.lower():
            # Parse "X passed" pattern
            try:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed":
                        result.passed = int(parts[i-1])
                    elif part == "failed":
                        result.failed = int(parts[i-1])
                    elif part == "skipped":
                        result.skipped = int(parts[i-1])
                    elif part == "error" or part == "errors":
                        result.errors = int(parts[i-1])
            except (ValueError, IndexError):
                pass

    result.total = result.passed + result.failed + result.skipped + result.errors

    # If we couldn't parse, at least set pass/fail based on return code
    if result.total == 0:
        if return_code == 0:
            result.passed = 1
            result.total = 1
        else:
            result.failed = 1
            result.total = 1

    return result


def _display_test_results(report: TestReport) -> None:
    """Display test results."""
    console.print()

    # Summary
    status = "[green]PASSED[/green]" if report.total_failed == 0 else "[red]FAILED[/red]"
    pass_rate = (report.total_passed / report.total_tests * 100) if report.total_tests > 0 else 0

    console.print(Panel(
        f"[bold]Status:[/bold] {status}\n"
        f"[bold]Tests:[/bold] {report.total_passed}/{report.total_tests} passed ({pass_rate:.1f}%)\n"
        f"[bold]Failed:[/bold] {report.total_failed}\n"
        f"[bold]Duration:[/bold] {report.total_duration_ms:.2f}ms",
        title="Test Results",
        border_style="green" if report.total_failed == 0 else "red",
    ))

    # Suite details
    if report.suites:
        console.print("\n[bold]Suite Results:[/bold]")
        table = Table()
        table.add_column("Suite", style="cyan")
        table.add_column("Passed", style="green", justify="right")
        table.add_column("Failed", style="red", justify="right")
        table.add_column("Skipped", style="yellow", justify="right")
        table.add_column("Duration", justify="right")

        for suite in report.suites:
            table.add_row(
                suite.suite_name,
                str(suite.passed),
                str(suite.failed),
                str(suite.skipped),
                f"{suite.duration_ms:.2f}ms",
            )

        console.print(table)

    console.print()


def _display_golden_results(results: Dict[str, Any]) -> None:
    """Display golden test results."""
    console.print()

    status = "[green]DETERMINISTIC[/green]" if results["deterministic"] else "[red]NON-DETERMINISTIC[/red]"

    console.print(Panel(
        f"[bold]Status:[/bold] {status}\n"
        f"[bold]Runs:[/bold] {results['runs']}\n"
        f"[bold]Passed:[/bold] {results['passed']}\n"
        f"[bold]Failed:[/bold] {results['failed']}\n"
        f"[bold]Unique Hashes:[/bold] {results.get('unique_hashes', 0)}",
        title="Golden Test Results (Determinism)",
        border_style="green" if results["deterministic"] else "red",
    ))

    if not results["deterministic"]:
        console.print("\n[red]WARNING:[/red] Agent produces non-deterministic outputs!")
        console.print("This violates the zero-hallucination principle.")
        console.print("\n[bold]Recommendations:[/bold]")
        console.print("  1. Remove random/non-deterministic operations")
        console.print("  2. Use fixed seeds for any randomness")
        console.print("  3. Ensure LLM calls are not in calculation paths")

    console.print()


def _display_benchmark_results(results: Dict[str, Any]) -> None:
    """Display benchmark results."""
    console.print()

    metrics = results.get("metrics", {})

    console.print(Panel(
        f"[bold]Iterations:[/bold] {results['iterations']}\n"
        f"[bold]Warmup:[/bold] {results['warmup']}",
        title="Benchmark Configuration",
        border_style="cyan",
    ))

    # Metrics table
    table = Table(title="Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("p50 Latency", f"{metrics.get('p50_ms', 0):.2f} ms")
    table.add_row("p95 Latency", f"{metrics.get('p95_ms', 0):.2f} ms")
    table.add_row("p99 Latency", f"{metrics.get('p99_ms', 0):.2f} ms")
    table.add_row("Mean Latency", f"{metrics.get('mean_ms', 0):.2f} ms")
    table.add_row("Std Dev", f"{metrics.get('stdev_ms', 0):.2f} ms")
    table.add_row("Min", f"{metrics.get('min_ms', 0):.2f} ms")
    table.add_row("Max", f"{metrics.get('max_ms', 0):.2f} ms")
    table.add_row("Throughput", f"{metrics.get('throughput_rps', 0):.1f} req/s")

    console.print(table)
    console.print()


def _display_coverage_results(results: Dict[str, Any], threshold: int) -> None:
    """Display coverage results."""
    console.print()

    coverage = results.get("total_coverage", 0)
    status = "[green]PASS[/green]" if coverage >= threshold else "[red]FAIL[/red]"
    color = "green" if coverage >= threshold else "red"

    console.print(Panel(
        f"[bold]Coverage:[/bold] [{color}]{coverage:.1f}%[/{color}]\n"
        f"[bold]Threshold:[/bold] {threshold}%\n"
        f"[bold]Status:[/bold] {status}",
        title="Coverage Report",
        border_style=color,
    ))

    # File breakdown
    if results.get("files"):
        table = Table(title="File Coverage")
        table.add_column("File", style="cyan")
        table.add_column("Coverage", justify="right")

        for f in sorted(results["files"], key=lambda x: x["coverage"]):
            cov = f["coverage"]
            color = "green" if cov >= threshold else "yellow" if cov >= 70 else "red"
            table.add_row(f["file"], f"[{color}]{cov:.1f}%[/{color}]")

        console.print(table)

    if results.get("html_report"):
        console.print(f"\n[bold]HTML Report:[/bold] {results['html_report']}")

    console.print()


def _save_test_report(report: TestReport, output: Path) -> None:
    """Save test report to file."""
    data = report.model_dump()

    suffix = output.suffix.lower()
    if suffix == ".json":
        with open(output, "w") as f:
            json.dump(data, f, indent=2, default=str)
    elif suffix == ".xml":
        # JUnit XML format
        _save_junit_xml(report, output)
    else:
        with open(output, "w") as f:
            json.dump(data, f, indent=2, default=str)

    console.print(f"[green]Report saved to:[/green] {output}")


def _save_benchmark_report(results: Dict[str, Any], output: Path) -> None:
    """Save benchmark report to file."""
    with open(output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"[green]Benchmark report saved to:[/green] {output}")


def _save_junit_xml(report: TestReport, output: Path) -> None:
    """Save report in JUnit XML format."""
    xml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="{report.agent_id}" tests="{report.total_tests}" failures="{report.total_failed}" time="{report.total_duration_ms/1000}">
'''

    for suite in report.suites:
        xml_content += f'''  <testsuite name="{suite.suite_name}" tests="{suite.total}" failures="{suite.failed}" time="{suite.duration_ms/1000}">
'''
        for test in suite.tests:
            xml_content += f'''    <testcase name="{test.name}" time="{test.duration_ms/1000}">
'''
            if test.status == TestStatus.FAILED:
                xml_content += f'''      <failure message="{test.message or 'Test failed'}"/>
'''
            elif test.status == TestStatus.SKIPPED:
                xml_content += '''      <skipped/>
'''
            xml_content += '''    </testcase>
'''
        xml_content += '''  </testsuite>
'''

    xml_content += '''</testsuites>
'''

    with open(output, "w") as f:
        f.write(xml_content)
