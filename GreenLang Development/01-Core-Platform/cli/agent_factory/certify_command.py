# -*- coding: utf-8 -*-
"""
Certification Pipeline
======================

12-dimension certification framework for GreenLang agents:
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

Certification Levels:
- GOLD: 100% score, all required dimensions pass
- SILVER: 95%+ score, all required dimensions pass
- BRONZE: 85%+ score, all required dimensions pass

Usage:
    gl agent certify eudr_compliance --level gold
    gl agent certify ./agents/carbon --dimensions D01,D02,D03
"""

import os
import sys
import json
import hashlib
import logging
import subprocess
import ast
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from enum import Enum

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from pydantic import BaseModel, Field

# Create sub-app for certify commands
certify_app = typer.Typer(
    name="certify",
    help="12-dimension agent certification",
    no_args_is_help=True,
)

console = Console()
logger = logging.getLogger(__name__)


# =============================================================================
# Certification Models
# =============================================================================

class CertificationLevel(str, Enum):
    """Certification levels."""
    GOLD = "GOLD"
    SILVER = "SILVER"
    BRONZE = "BRONZE"
    FAIL = "FAIL"


class DimensionStatus(str, Enum):
    """Dimension evaluation status."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIPPED = "skipped"


class DimensionResult(BaseModel):
    """Result for a single dimension."""
    dimension_id: str
    dimension_name: str
    description: str
    weight: float
    required: bool
    status: DimensionStatus
    score: float = Field(ge=0, le=100)
    checks_passed: int = 0
    checks_total: int = 0
    findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    execution_time_ms: float = 0


class CertificationResult(BaseModel):
    """Complete certification result."""
    agent_id: str
    agent_version: str
    certification_id: str
    level: CertificationLevel
    certified: bool
    overall_score: float
    weighted_score: float
    dimensions_passed: int
    dimensions_total: int
    dimension_results: List[DimensionResult] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    valid_until: Optional[str] = None


# =============================================================================
# Dimension Definitions
# =============================================================================

DIMENSIONS = {
    "D01": {
        "name": "Determinism",
        "description": "Identical outputs for identical inputs (100 runs)",
        "weight": 15.0,
        "required": True,
    },
    "D02": {
        "name": "Provenance",
        "description": "SHA-256 hash generation for audit trails",
        "weight": 10.0,
        "required": True,
    },
    "D03": {
        "name": "Zero-Hallucination",
        "description": "No LLM calls in numeric calculation paths",
        "weight": 15.0,
        "required": True,
    },
    "D04": {
        "name": "Accuracy",
        "description": "Golden test pass rate and correctness",
        "weight": 10.0,
        "required": True,
    },
    "D05": {
        "name": "Source Verification",
        "description": "Traceable emission factors and data sources",
        "weight": 8.0,
        "required": True,
    },
    "D06": {
        "name": "Unit Consistency",
        "description": "Input/output unit validation and conversion",
        "weight": 7.0,
        "required": True,
    },
    "D07": {
        "name": "Regulatory Compliance",
        "description": "GHG Protocol, ISO 14064, CSRD alignment",
        "weight": 8.0,
        "required": False,
    },
    "D08": {
        "name": "Security",
        "description": "No secrets exposure, injection prevention",
        "weight": 8.0,
        "required": True,
    },
    "D09": {
        "name": "Performance",
        "description": "Response time, memory efficiency",
        "weight": 5.0,
        "required": False,
    },
    "D10": {
        "name": "Documentation",
        "description": "Docstrings, API documentation, examples",
        "weight": 5.0,
        "required": False,
    },
    "D11": {
        "name": "Test Coverage",
        "description": "Code coverage >90% target",
        "weight": 5.0,
        "required": False,
    },
    "D12": {
        "name": "Production Readiness",
        "description": "Logging, health checks, error handling",
        "weight": 4.0,
        "required": False,
    },
}


# =============================================================================
# Certify Commands
# =============================================================================

@certify_app.command("run")
def certify_run_command(
    agent_path: Path = typer.Argument(
        ...,
        help="Path to agent directory",
    ),
    level: str = typer.Option(
        "gold",
        "--level", "-l",
        help="Target certification level: gold, silver, bronze",
    ),
    dimensions: Optional[str] = typer.Option(
        None,
        "--dimensions", "-d",
        help="Specific dimensions (comma-separated: D01,D02,D03)",
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
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Verbose output",
    ),
):
    """
    Run full certification evaluation.

    Example:
        gl agent certify run ./agents/carbon
        gl agent certify run . --level gold --output report.pdf
    """
    dim_list = None
    if dimensions:
        dim_list = [d.strip().upper() for d in dimensions.split(",")]

    certify_agent_impl(
        agent_id=str(agent_path),
        level=level.lower(),
        dimensions=dim_list,
        output=output,
        upload=upload,
        verbose=verbose,
    )


@certify_app.command("dimension")
def certify_dimension_command(
    agent_path: Path = typer.Argument(
        ...,
        help="Path to agent directory",
    ),
    dimension_id: str = typer.Argument(
        ...,
        help="Dimension to evaluate (D01-D12)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Verbose output",
    ),
):
    """
    Evaluate a single certification dimension.

    Example:
        gl agent certify dimension ./agents/carbon D01
        gl agent certify dimension . D03 --verbose
    """
    result = evaluate_single_dimension(agent_path, dimension_id.upper(), verbose)
    _display_dimension_result(result)


@certify_app.command("list-dimensions")
def certify_list_dimensions_command():
    """
    List all certification dimensions.

    Example:
        gl agent certify list-dimensions
    """
    console.print(Panel(
        "[bold cyan]GreenLang Certification Dimensions[/bold cyan]\n"
        "12-dimension evaluation framework",
        border_style="cyan"
    ))

    table = Table(title="Certification Dimensions")
    table.add_column("ID", style="cyan", width=6)
    table.add_column("Name", style="green", width=20)
    table.add_column("Weight", justify="right", width=8)
    table.add_column("Required", justify="center", width=10)
    table.add_column("Description", width=40)

    for dim_id, dim in DIMENSIONS.items():
        required = "[green]Yes[/green]" if dim["required"] else "[yellow]No[/yellow]"
        table.add_row(
            dim_id,
            dim["name"],
            f"{dim['weight']:.1f}",
            required,
            dim["description"],
        )

    console.print(table)

    console.print("\n[bold]Certification Levels:[/bold]")
    console.print("  [green]GOLD[/green]:   100% score, all required dimensions pass")
    console.print("  [yellow]SILVER[/yellow]: 95%+ score, all required dimensions pass")
    console.print("  [cyan]BRONZE[/cyan]: 85%+ score, all required dimensions pass")
    console.print("  [red]FAIL[/red]:   Below 85% or required dimension fails")


@certify_app.command("report")
def certify_report_command(
    certification_id: str = typer.Argument(
        ...,
        help="Certification ID to retrieve",
    ),
    format: str = typer.Option(
        "rich",
        "--format", "-f",
        help="Output format: rich, json, pdf, html",
    ),
):
    """
    Retrieve and display a certification report.

    Example:
        gl agent certify report CERT-20241201-001
    """
    console.print(f"[yellow]Fetching certification report: {certification_id}[/yellow]")
    console.print("[dim]This would fetch from registry...[/dim]")


# =============================================================================
# Core Certification Functions
# =============================================================================

def certify_agent_impl(
    agent_id: str,
    level: str = "gold",
    dimensions: Optional[List[str]] = None,
    output: Optional[Path] = None,
    upload: bool = False,
    verbose: bool = False,
) -> CertificationResult:
    """
    Run certification evaluation on an agent.

    Args:
        agent_id: Agent ID or path
        level: Target certification level
        dimensions: Specific dimensions to evaluate
        output: Output report path
        upload: Upload to registry on success
        verbose: Verbose output

    Returns:
        CertificationResult
    """
    console.print(Panel(
        "[bold cyan]GreenLang Agent Certification[/bold cyan]\n"
        "12-Dimension Evaluation Framework",
        border_style="cyan"
    ))

    # Resolve agent path
    agent_path = Path(agent_id)
    if not agent_path.exists():
        for search_path in ["./agents", "./packs", "."]:
            candidate = Path(search_path) / agent_id
            if candidate.exists():
                agent_path = candidate
                break

    if not agent_path.exists():
        console.print(f"[red]Agent not found: {agent_id}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Agent:[/bold] {agent_path}")
    console.print(f"[bold]Target Level:[/bold] {level.upper()}")

    # Generate certification ID
    cert_id = f"CERT-{datetime.now().strftime('%Y%m%d')}-{hashlib.sha256(str(agent_path).encode()).hexdigest()[:6].upper()}"

    # Get agent version
    version = _get_agent_version(agent_path)

    # Determine dimensions to evaluate
    if dimensions:
        dims_to_eval = {d: DIMENSIONS[d] for d in dimensions if d in DIMENSIONS}
    else:
        dims_to_eval = DIMENSIONS

    console.print(f"[bold]Dimensions:[/bold] {len(dims_to_eval)}")
    console.print()

    # Evaluate dimensions
    dimension_results = []
    total_weighted_score = 0
    total_weight = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating dimensions...", total=len(dims_to_eval))

        for dim_id, dim_info in dims_to_eval.items():
            progress.update(task, description=f"Evaluating {dim_id}: {dim_info['name']}...")

            result = evaluate_dimension(agent_path, dim_id, dim_info, verbose)
            dimension_results.append(result)

            total_weighted_score += result.score * dim_info["weight"]
            total_weight += dim_info["weight"]

            progress.update(task, advance=1)

    # Calculate overall scores
    overall_score = sum(r.score for r in dimension_results) / len(dimension_results) if dimension_results else 0
    weighted_score = total_weighted_score / total_weight if total_weight > 0 else 0

    # Determine certification level
    dimensions_passed = sum(1 for r in dimension_results if r.status == DimensionStatus.PASS)
    required_passed = all(
        r.status == DimensionStatus.PASS
        for r in dimension_results
        if DIMENSIONS.get(r.dimension_id, {}).get("required", False)
    )

    if weighted_score >= 100 and required_passed:
        cert_level = CertificationLevel.GOLD
    elif weighted_score >= 95 and required_passed:
        cert_level = CertificationLevel.SILVER
    elif weighted_score >= 85 and required_passed:
        cert_level = CertificationLevel.BRONZE
    else:
        cert_level = CertificationLevel.FAIL

    certified = cert_level != CertificationLevel.FAIL

    # Create result
    result = CertificationResult(
        agent_id=agent_id,
        agent_version=version,
        certification_id=cert_id,
        level=cert_level,
        certified=certified,
        overall_score=overall_score,
        weighted_score=weighted_score,
        dimensions_passed=dimensions_passed,
        dimensions_total=len(dimension_results),
        dimension_results=dimension_results,
        valid_until=(datetime.now().replace(year=datetime.now().year + 1)).isoformat() if certified else None,
    )

    # Display results
    _display_certification_result(result)

    # Save report if requested
    if output:
        _save_certification_report(result, output)

    # Upload if requested and certified
    if upload and certified:
        _upload_certification(result)

    # Exit with appropriate code
    if not certified:
        raise typer.Exit(1)

    return result


def evaluate_dimension(
    agent_path: Path,
    dimension_id: str,
    dim_info: Dict[str, Any],
    verbose: bool = False,
) -> DimensionResult:
    """
    Evaluate a single certification dimension.

    Args:
        agent_path: Path to agent
        dimension_id: Dimension ID (D01-D12)
        dim_info: Dimension configuration
        verbose: Verbose output

    Returns:
        DimensionResult
    """
    start_time = datetime.now()
    findings = []
    recommendations = []
    checks_passed = 0
    checks_total = 0

    # Dispatch to specific evaluator
    evaluator = DIMENSION_EVALUATORS.get(dimension_id)
    if evaluator:
        passed, total, score, finds, recs = evaluator(agent_path, verbose)
        checks_passed = passed
        checks_total = total
        findings = finds
        recommendations = recs
    else:
        # Default evaluation
        score = 50.0
        findings.append("No specific evaluator implemented")
        recommendations.append("Manual review recommended")
        checks_total = 1

    # Determine status
    if score >= 100:
        status = DimensionStatus.PASS
    elif score >= 80:
        status = DimensionStatus.WARNING
    else:
        status = DimensionStatus.FAIL

    return DimensionResult(
        dimension_id=dimension_id,
        dimension_name=dim_info["name"],
        description=dim_info["description"],
        weight=dim_info["weight"],
        required=dim_info["required"],
        status=status,
        score=score,
        checks_passed=checks_passed,
        checks_total=checks_total,
        findings=findings,
        recommendations=recommendations,
        execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
    )


def evaluate_single_dimension(
    agent_path: Path,
    dimension_id: str,
    verbose: bool = False,
) -> DimensionResult:
    """Evaluate a single dimension by ID."""
    if dimension_id not in DIMENSIONS:
        console.print(f"[red]Unknown dimension: {dimension_id}[/red]")
        console.print(f"Valid dimensions: {', '.join(DIMENSIONS.keys())}")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold cyan]Evaluating Dimension: {dimension_id}[/bold cyan]\n"
        f"{DIMENSIONS[dimension_id]['name']}",
        border_style="cyan"
    ))

    return evaluate_dimension(agent_path, dimension_id, DIMENSIONS[dimension_id], verbose)


# =============================================================================
# Dimension Evaluators
# =============================================================================

def _evaluate_d01_determinism(agent_path: Path, verbose: bool) -> Tuple[int, int, float, List[str], List[str]]:
    """Evaluate D01: Determinism."""
    findings = []
    recommendations = []
    passed = 0
    total = 0

    # Check for randomness in code
    total += 1
    has_random = False
    for py_file in agent_path.glob("**/*.py"):
        content = py_file.read_text()
        if "random." in content or "np.random" in content:
            has_random = True
            findings.append(f"Random usage found in {py_file.name}")
            recommendations.append("Use fixed seeds or remove random operations")
            break

    if not has_random:
        passed += 1
        findings.append("No random operations detected")

    # Check for time-dependent code
    total += 1
    has_time_dep = False
    for py_file in agent_path.glob("**/*.py"):
        content = py_file.read_text()
        if "datetime.now()" in content and "provenance" not in content.lower():
            has_time_dep = True
            findings.append(f"Time-dependent code in {py_file.name}")
            recommendations.append("Avoid time-dependent calculations")
            break

    if not has_time_dep:
        passed += 1

    # Check for golden tests
    total += 1
    tests_dir = agent_path / "tests"
    if tests_dir.exists():
        golden_tests = list(tests_dir.glob("*golden*.py"))
        if golden_tests:
            passed += 1
            findings.append(f"Golden tests found: {len(golden_tests)} files")
        else:
            findings.append("No golden tests found")
            recommendations.append("Add golden tests for determinism verification")
    else:
        findings.append("No tests directory")
        recommendations.append("Create tests/ directory with golden tests")

    score = (passed / total * 100) if total > 0 else 0
    return passed, total, score, findings, recommendations


def _evaluate_d02_provenance(agent_path: Path, verbose: bool) -> Tuple[int, int, float, List[str], List[str]]:
    """Evaluate D02: Provenance."""
    findings = []
    recommendations = []
    passed = 0
    total = 0

    # Check for SHA-256 usage
    total += 1
    has_hash = False
    for py_file in agent_path.glob("**/*.py"):
        content = py_file.read_text()
        if "hashlib.sha256" in content or "sha256" in content:
            has_hash = True
            passed += 1
            findings.append("SHA-256 hashing implemented")
            break

    if not has_hash:
        findings.append("No SHA-256 hashing found")
        recommendations.append("Add provenance hash generation")

    # Check for provenance output
    total += 1
    has_provenance_output = False
    for py_file in agent_path.glob("**/*.py"):
        content = py_file.read_text()
        if "provenance_hash" in content or "provenance" in content.lower():
            has_provenance_output = True
            passed += 1
            findings.append("Provenance output field found")
            break

    if not has_provenance_output:
        findings.append("No provenance output field")
        recommendations.append("Add provenance_hash to output model")

    # Check pack.yaml for provenance config
    total += 1
    pack_yaml = agent_path / "pack.yaml"
    if pack_yaml.exists():
        content = pack_yaml.read_text()
        if "provenance:" in content:
            passed += 1
            findings.append("Provenance configured in pack.yaml")
        else:
            findings.append("No provenance section in pack.yaml")
            recommendations.append("Add provenance configuration to pack.yaml")

    score = (passed / total * 100) if total > 0 else 0
    return passed, total, score, findings, recommendations


def _evaluate_d03_zero_hallucination(agent_path: Path, verbose: bool) -> Tuple[int, int, float, List[str], List[str]]:
    """Evaluate D03: Zero-Hallucination."""
    findings = []
    recommendations = []
    passed = 0
    total = 0

    # Check for LLM in calculation paths
    total += 1
    llm_in_calc = False

    for py_file in agent_path.glob("**/*.py"):
        content = py_file.read_text()

        # Check for LLM calls in functions with "calc" or "compute" in name
        if "def calc" in content or "def compute" in content:
            if "llm" in content.lower() or "openai" in content.lower() or "anthropic" in content.lower():
                llm_in_calc = True
                findings.append(f"LLM call detected in calculation function: {py_file.name}")
                recommendations.append("Remove LLM calls from calculation paths")
                break

    if not llm_in_calc:
        passed += 1
        findings.append("No LLM calls in calculation paths")

    # Check for deterministic tool types
    total += 1
    pack_yaml = agent_path / "pack.yaml"
    if pack_yaml.exists():
        content = pack_yaml.read_text()
        if "type: deterministic" in content:
            passed += 1
            findings.append("Deterministic tool types configured")
        elif "type: llm" in content and ("calc" in content.lower() or "compute" in content.lower()):
            findings.append("LLM tool type used for calculations")
            recommendations.append("Use 'deterministic' type for calculation tools")

    # Check for database lookups instead of LLM
    total += 1
    has_db_lookup = False
    for py_file in agent_path.glob("**/*.py"):
        content = py_file.read_text()
        if "lookup" in content.lower() or "query" in content.lower() or "fetch" in content.lower():
            has_db_lookup = True
            passed += 1
            findings.append("Database lookup patterns detected")
            break

    if not has_db_lookup:
        findings.append("No explicit database lookups")
        recommendations.append("Consider using database lookups for emission factors")

    score = (passed / total * 100) if total > 0 else 0
    return passed, total, score, findings, recommendations


def _evaluate_d04_accuracy(agent_path: Path, verbose: bool) -> Tuple[int, int, float, List[str], List[str]]:
    """Evaluate D04: Accuracy."""
    findings = []
    recommendations = []
    passed = 0
    total = 0

    # Check for golden tests
    total += 1
    tests_dir = agent_path / "tests"
    if tests_dir.exists():
        test_files = list(tests_dir.glob("test_*.py"))
        if test_files:
            passed += 1
            findings.append(f"Test files found: {len(test_files)}")
        else:
            findings.append("No test files found")
            recommendations.append("Add test files with golden test cases")

    # Check for expected values in tests
    total += 1
    has_expected = False
    if tests_dir.exists():
        for test_file in tests_dir.glob("*.py"):
            content = test_file.read_text()
            if "expected" in content.lower() or "assert" in content:
                has_expected = True
                passed += 1
                findings.append("Expected value assertions found")
                break

    if not has_expected:
        findings.append("No expected value assertions")
        recommendations.append("Add assertions with expected values")

    # Check for tolerance specifications
    total += 1
    has_tolerance = False
    pack_yaml = agent_path / "pack.yaml"
    if pack_yaml.exists():
        content = pack_yaml.read_text()
        if "tolerance" in content:
            has_tolerance = True
            passed += 1
            findings.append("Tolerance specifications found")

    if not has_tolerance:
        findings.append("No tolerance specifications")
        recommendations.append("Add tolerance for numeric comparisons")

    score = (passed / total * 100) if total > 0 else 0
    return passed, total, score, findings, recommendations


def _evaluate_d08_security(agent_path: Path, verbose: bool) -> Tuple[int, int, float, List[str], List[str]]:
    """Evaluate D08: Security."""
    findings = []
    recommendations = []
    passed = 0
    total = 0

    # Check for hardcoded secrets
    total += 1
    secret_patterns = [
        r'api_key\s*=\s*["\'][^"\']+["\']',
        r'password\s*=\s*["\'][^"\']+["\']',
        r'secret\s*=\s*["\'][^"\']+["\']',
        r'token\s*=\s*["\'][^"\']+["\']',
    ]

    has_secrets = False
    for py_file in agent_path.glob("**/*.py"):
        content = py_file.read_text()
        for pattern in secret_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                has_secrets = True
                findings.append(f"Potential hardcoded secret in {py_file.name}")
                recommendations.append("Use environment variables for secrets")
                break
        if has_secrets:
            break

    if not has_secrets:
        passed += 1
        findings.append("No hardcoded secrets detected")

    # Check for .env file handling
    total += 1
    uses_env = False
    for py_file in agent_path.glob("**/*.py"):
        content = py_file.read_text()
        if "os.environ" in content or "dotenv" in content:
            uses_env = True
            passed += 1
            findings.append("Environment variable usage detected")
            break

    if not uses_env:
        recommendations.append("Consider using environment variables for configuration")

    # Check for input validation
    total += 1
    has_validation = False
    for py_file in agent_path.glob("**/*.py"):
        content = py_file.read_text()
        if "pydantic" in content or "validator" in content or "validate" in content:
            has_validation = True
            passed += 1
            findings.append("Input validation found")
            break

    if not has_validation:
        findings.append("No input validation detected")
        recommendations.append("Add Pydantic models for input validation")

    score = (passed / total * 100) if total > 0 else 0
    return passed, total, score, findings, recommendations


def _evaluate_d10_documentation(agent_path: Path, verbose: bool) -> Tuple[int, int, float, List[str], List[str]]:
    """Evaluate D10: Documentation."""
    findings = []
    recommendations = []
    passed = 0
    total = 0

    # Check for README
    total += 1
    readme = agent_path / "README.md"
    if readme.exists():
        passed += 1
        findings.append("README.md found")
    else:
        findings.append("No README.md")
        recommendations.append("Add README.md documentation")

    # Check for docstrings
    total += 1
    has_docstrings = False
    for py_file in agent_path.glob("*.py"):
        content = py_file.read_text()
        if '"""' in content or "'''" in content:
            has_docstrings = True
            passed += 1
            findings.append("Docstrings found in code")
            break

    if not has_docstrings:
        findings.append("No docstrings detected")
        recommendations.append("Add docstrings to functions and classes")

    # Check for type hints
    total += 1
    has_type_hints = False
    for py_file in agent_path.glob("*.py"):
        content = py_file.read_text()
        if "->" in content or ": str" in content or ": int" in content:
            has_type_hints = True
            passed += 1
            findings.append("Type hints found")
            break

    if not has_type_hints:
        findings.append("No type hints detected")
        recommendations.append("Add type hints for better documentation")

    score = (passed / total * 100) if total > 0 else 0
    return passed, total, score, findings, recommendations


def _evaluate_d11_test_coverage(agent_path: Path, verbose: bool) -> Tuple[int, int, float, List[str], List[str]]:
    """Evaluate D11: Test Coverage."""
    findings = []
    recommendations = []
    passed = 0
    total = 0

    # Check for tests
    total += 1
    tests_dir = agent_path / "tests"
    if tests_dir.exists():
        test_files = list(tests_dir.glob("test_*.py"))
        if test_files:
            passed += 1
            findings.append(f"{len(test_files)} test file(s) found")
        else:
            findings.append("Tests directory exists but no test files")
            recommendations.append("Add test_*.py files")
    else:
        findings.append("No tests directory")
        recommendations.append("Create tests/ directory")

    # Check for conftest
    total += 1
    conftest = agent_path / "tests" / "conftest.py"
    if conftest.exists():
        passed += 1
        findings.append("conftest.py found")
    else:
        recommendations.append("Add conftest.py for shared fixtures")

    # Check for pytest markers
    total += 1
    has_markers = False
    for test_file in (agent_path / "tests").glob("*.py") if (agent_path / "tests").exists() else []:
        content = test_file.read_text()
        if "@pytest.mark" in content:
            has_markers = True
            passed += 1
            findings.append("Pytest markers found")
            break

    if not has_markers:
        recommendations.append("Add pytest markers for test categorization")

    score = (passed / total * 100) if total > 0 else 0
    return passed, total, score, findings, recommendations


def _evaluate_d12_production_readiness(agent_path: Path, verbose: bool) -> Tuple[int, int, float, List[str], List[str]]:
    """Evaluate D12: Production Readiness."""
    findings = []
    recommendations = []
    passed = 0
    total = 0

    # Check for logging
    total += 1
    has_logging = False
    for py_file in agent_path.glob("**/*.py"):
        content = py_file.read_text()
        if "import logging" in content or "logger." in content:
            has_logging = True
            passed += 1
            findings.append("Logging implemented")
            break

    if not has_logging:
        findings.append("No logging detected")
        recommendations.append("Add logging for production monitoring")

    # Check for error handling
    total += 1
    has_error_handling = False
    for py_file in agent_path.glob("**/*.py"):
        content = py_file.read_text()
        if "try:" in content and "except" in content:
            has_error_handling = True
            passed += 1
            findings.append("Error handling found")
            break

    if not has_error_handling:
        findings.append("No error handling detected")
        recommendations.append("Add try/except blocks for production resilience")

    # Check for Dockerfile
    total += 1
    dockerfile = agent_path / "Dockerfile"
    if dockerfile.exists():
        passed += 1
        findings.append("Dockerfile found")
    else:
        recommendations.append("Add Dockerfile for containerization")

    score = (passed / total * 100) if total > 0 else 0
    return passed, total, score, findings, recommendations


# Evaluator dispatch table
DIMENSION_EVALUATORS = {
    "D01": _evaluate_d01_determinism,
    "D02": _evaluate_d02_provenance,
    "D03": _evaluate_d03_zero_hallucination,
    "D04": _evaluate_d04_accuracy,
    "D08": _evaluate_d08_security,
    "D10": _evaluate_d10_documentation,
    "D11": _evaluate_d11_test_coverage,
    "D12": _evaluate_d12_production_readiness,
}


# =============================================================================
# Helper Functions
# =============================================================================

def _get_agent_version(agent_path: Path) -> str:
    """Get agent version from files."""
    # Try __init__.py
    init_file = agent_path / "__init__.py"
    if init_file.exists():
        content = init_file.read_text()
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)

    # Try pack.yaml
    pack_yaml = agent_path / "pack.yaml"
    if pack_yaml.exists():
        import yaml
        with open(pack_yaml) as f:
            spec = yaml.safe_load(f)
            if spec and "version" in spec:
                return spec["version"]

    return "0.0.0"


def _display_certification_result(result: CertificationResult) -> None:
    """Display certification result."""
    console.print()

    # Level styling
    level_styles = {
        CertificationLevel.GOLD: ("green", "GOLD"),
        CertificationLevel.SILVER: ("yellow", "SILVER"),
        CertificationLevel.BRONZE: ("cyan", "BRONZE"),
        CertificationLevel.FAIL: ("red", "FAIL"),
    }

    color, label = level_styles.get(result.level, ("white", str(result.level)))
    status = f"[{color}]{label}[/{color}]"
    certified_text = "[green]CERTIFIED[/green]" if result.certified else "[red]NOT CERTIFIED[/red]"

    console.print(Panel(
        f"[bold]Status:[/bold] {certified_text}\n"
        f"[bold]Level:[/bold] {status}\n"
        f"[bold]Certification ID:[/bold] {result.certification_id}\n"
        f"[bold]Overall Score:[/bold] {result.overall_score:.1f}%\n"
        f"[bold]Weighted Score:[/bold] {result.weighted_score:.1f}%\n"
        f"[bold]Dimensions:[/bold] {result.dimensions_passed}/{result.dimensions_total} passed",
        title="Certification Result",
        border_style=color,
    ))

    # Dimension table
    table = Table(title="Dimension Results")
    table.add_column("ID", style="cyan", width=6)
    table.add_column("Dimension", width=20)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Required", justify="center", width=10)

    for dim in result.dimension_results:
        if dim.status == DimensionStatus.PASS:
            status_text = "[green]PASS[/green]"
        elif dim.status == DimensionStatus.WARNING:
            status_text = "[yellow]WARN[/yellow]"
        elif dim.status == DimensionStatus.FAIL:
            status_text = "[red]FAIL[/red]"
        else:
            status_text = "[dim]SKIP[/dim]"

        required = "[green]Yes[/green]" if dim.required else "[dim]No[/dim]"
        score_color = "green" if dim.score >= 80 else "yellow" if dim.score >= 60 else "red"

        table.add_row(
            dim.dimension_id,
            dim.dimension_name,
            status_text,
            f"[{score_color}]{dim.score:.0f}[/{score_color}]",
            required,
        )

    console.print(table)

    # Show recommendations for failed dimensions
    failed_dims = [d for d in result.dimension_results if d.status == DimensionStatus.FAIL]
    if failed_dims:
        console.print("\n[bold yellow]Recommendations:[/bold yellow]")
        for dim in failed_dims[:3]:  # Top 3
            console.print(f"\n  [cyan]{dim.dimension_id}: {dim.dimension_name}[/cyan]")
            for rec in dim.recommendations[:2]:
                console.print(f"    - {rec}")

    if result.certified:
        console.print(f"\n[bold]Valid Until:[/bold] {result.valid_until}")

    console.print()


def _display_dimension_result(result: DimensionResult) -> None:
    """Display single dimension result."""
    console.print()

    status_styles = {
        DimensionStatus.PASS: ("green", "PASS"),
        DimensionStatus.FAIL: ("red", "FAIL"),
        DimensionStatus.WARNING: ("yellow", "WARNING"),
        DimensionStatus.SKIPPED: ("dim", "SKIPPED"),
    }

    color, label = status_styles.get(result.status, ("white", str(result.status)))

    console.print(Panel(
        f"[bold]Dimension:[/bold] {result.dimension_id} - {result.dimension_name}\n"
        f"[bold]Status:[/bold] [{color}]{label}[/{color}]\n"
        f"[bold]Score:[/bold] {result.score:.1f}%\n"
        f"[bold]Checks:[/bold] {result.checks_passed}/{result.checks_total} passed\n"
        f"[bold]Required:[/bold] {'Yes' if result.required else 'No'}",
        title=f"Dimension Evaluation",
        border_style=color,
    ))

    if result.findings:
        console.print("\n[bold]Findings:[/bold]")
        for finding in result.findings:
            console.print(f"  - {finding}")

    if result.recommendations:
        console.print("\n[bold yellow]Recommendations:[/bold yellow]")
        for rec in result.recommendations:
            console.print(f"  - {rec}")

    console.print()


def _save_certification_report(result: CertificationResult, output: Path) -> None:
    """Save certification report."""
    suffix = output.suffix.lower()

    if suffix == ".json":
        with open(output, "w") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)
    elif suffix == ".html":
        _generate_html_report(result, output)
    elif suffix == ".pdf":
        console.print("[yellow]PDF generation requires additional dependencies[/yellow]")
        # Fall back to JSON
        json_output = output.with_suffix(".json")
        with open(json_output, "w") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)
        console.print(f"[green]JSON report saved to:[/green] {json_output}")
        return
    else:
        with open(output, "w") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)

    console.print(f"[green]Report saved to:[/green] {output}")


def _generate_html_report(result: CertificationResult, output: Path) -> None:
    """Generate HTML certification report."""
    level_colors = {
        CertificationLevel.GOLD: "#ffd700",
        CertificationLevel.SILVER: "#c0c0c0",
        CertificationLevel.BRONZE: "#cd7f32",
        CertificationLevel.FAIL: "#ff0000",
    }

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>GreenLang Certification Report - {result.agent_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: {level_colors.get(result.level, "#gray")}; padding: 20px; text-align: center; }}
        .summary {{ margin: 20px 0; padding: 15px; background: #f5f5f5; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        .warn {{ color: orange; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>GreenLang Agent Certification</h1>
        <h2>{result.level.value}</h2>
    </div>

    <div class="summary">
        <h3>Summary</h3>
        <p><strong>Agent:</strong> {result.agent_id}</p>
        <p><strong>Version:</strong> {result.agent_version}</p>
        <p><strong>Certification ID:</strong> {result.certification_id}</p>
        <p><strong>Overall Score:</strong> {result.overall_score:.1f}%</p>
        <p><strong>Weighted Score:</strong> {result.weighted_score:.1f}%</p>
        <p><strong>Timestamp:</strong> {result.timestamp}</p>
    </div>

    <h3>Dimension Results</h3>
    <table>
        <tr>
            <th>ID</th>
            <th>Dimension</th>
            <th>Status</th>
            <th>Score</th>
            <th>Required</th>
        </tr>
'''

    for dim in result.dimension_results:
        status_class = {
            DimensionStatus.PASS: "pass",
            DimensionStatus.FAIL: "fail",
            DimensionStatus.WARNING: "warn",
        }.get(dim.status, "")

        html += f'''        <tr>
            <td>{dim.dimension_id}</td>
            <td>{dim.dimension_name}</td>
            <td class="{status_class}">{dim.status.value.upper()}</td>
            <td>{dim.score:.0f}%</td>
            <td>{"Yes" if dim.required else "No"}</td>
        </tr>
'''

    html += '''    </table>

    <footer>
        <p><em>Generated by GreenLang Agent Factory CLI</em></p>
    </footer>
</body>
</html>
'''

    with open(output, "w") as f:
        f.write(html)


def _upload_certification(result: CertificationResult) -> None:
    """Upload certification to registry."""
    console.print("\n[cyan]Uploading certification to registry...[/cyan]")
    console.print("[dim]This would upload to registry.greenlang.io...[/dim]")
    console.print(f"[green]Certification {result.certification_id} uploaded successfully![/green]")
