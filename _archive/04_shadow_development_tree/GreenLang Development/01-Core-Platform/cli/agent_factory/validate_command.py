# -*- coding: utf-8 -*-
"""
Comprehensive Agent Validation
==============================

Validates AgentSpec and agent implementations against:
- Schema compliance
- Data contract validation
- Safety constraint checking
- Explainability coverage
- Tool dependency resolution
- Standards reference verification

Usage:
    gl agent validate pack.yaml
    gl agent validate ./agents/eudr --strict
    gl agent validate spec.yaml --fix --output report.json
"""

import os
import sys
import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from enum import Enum

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree
from pydantic import BaseModel, Field, ValidationError

# Create sub-app for validate commands
validate_app = typer.Typer(
    name="validate",
    help="Comprehensive agent validation",
    no_args_is_help=True,
)

console = Console()
logger = logging.getLogger(__name__)


# =============================================================================
# Validation Models
# =============================================================================

class ValidationSeverity(str, Enum):
    """Validation message severity."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationMessage(BaseModel):
    """Single validation message."""
    severity: ValidationSeverity
    code: str
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None


class ValidationResult(BaseModel):
    """Complete validation result."""
    valid: bool
    score: float = Field(ge=0, le=100)
    errors: List[ValidationMessage] = Field(default_factory=list)
    warnings: List[ValidationMessage] = Field(default_factory=list)
    info: List[ValidationMessage] = Field(default_factory=list)
    checks_passed: int = 0
    checks_total: int = 0
    duration_ms: float = 0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# Schema Definitions
# =============================================================================

REQUIRED_SPEC_FIELDS = [
    "id",
    "name",
    "version",
]

RECOMMENDED_SPEC_FIELDS = [
    "description",
    "license",
    "metadata",
    "tools",
    "inputs",
    "outputs",
    "tests",
    "provenance",
    "safety",
]

TOOL_REQUIRED_FIELDS = [
    "name",
    "type",
]

INPUT_OUTPUT_REQUIRED_FIELDS = [
    "name",
    "type",
]

VALID_TOOL_TYPES = [
    "deterministic",
    "llm_classification",
    "llm_generation",
    "database_lookup",
    "api_call",
]

VALID_CATEGORIES = [
    "calculator",
    "validator",
    "reporter",
    "regulatory",
    "classifier",
    "aggregator",
    "custom",
]


# =============================================================================
# Validate Commands
# =============================================================================

@validate_app.command("spec")
def validate_spec_command(
    spec_path: Path = typer.Argument(
        ...,
        help="Path to AgentSpec YAML file",
        exists=True,
    ),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        help="Enable strict validation mode",
    ),
    schema_version: str = typer.Option(
        "1.0",
        "--schema-version",
        help="AgentSpec schema version",
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Attempt to auto-fix issues",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output validation report",
    ),
):
    """
    Validate an AgentSpec YAML file.

    Example:
        gl agent validate spec pack.yaml
        gl agent validate spec pack.yaml --strict --fix
    """
    validate_spec(
        spec_path=spec_path,
        strict=strict,
        schema_version=schema_version,
        fix=fix,
        output=output,
    )


@validate_app.command("agent")
def validate_agent_command(
    agent_path: Path = typer.Argument(
        ...,
        help="Path to agent directory",
        exists=True,
    ),
    include_tests: bool = typer.Option(
        True,
        "--tests/--no-tests",
        help="Validate test files",
    ),
    include_docs: bool = typer.Option(
        True,
        "--docs/--no-docs",
        help="Validate documentation",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output validation report",
    ),
):
    """
    Validate a complete agent implementation.

    Example:
        gl agent validate agent ./agents/carbon
        gl agent validate agent ./my-agent --no-tests
    """
    console.print(Panel(
        "[bold cyan]Agent Implementation Validation[/bold cyan]\n"
        f"Path: {agent_path}",
        border_style="cyan"
    ))

    result = validate_agent_implementation(
        agent_path,
        include_tests=include_tests,
        include_docs=include_docs,
    )

    _display_validation_result(result)

    if output:
        _save_validation_report(result, output)

    if not result.valid:
        raise typer.Exit(1)


@validate_app.command("contract")
def validate_contract_command(
    spec_path: Path = typer.Argument(
        ...,
        help="Path to AgentSpec YAML file",
        exists=True,
    ),
    sample_input: Optional[Path] = typer.Option(
        None,
        "--input", "-i",
        help="Sample input data to validate",
    ),
):
    """
    Validate data contracts (inputs/outputs schema).

    Example:
        gl agent validate contract pack.yaml
        gl agent validate contract pack.yaml --input sample.json
    """
    console.print(Panel(
        "[bold cyan]Data Contract Validation[/bold cyan]\n"
        f"Spec: {spec_path}",
        border_style="cyan"
    ))

    result = validate_data_contracts(spec_path, sample_input)
    _display_validation_result(result)

    if not result.valid:
        raise typer.Exit(1)


@validate_app.command("safety")
def validate_safety_command(
    spec_path: Path = typer.Argument(
        ...,
        help="Path to AgentSpec YAML file",
        exists=True,
    ),
):
    """
    Validate safety constraints and guardrails.

    Example:
        gl agent validate safety pack.yaml
    """
    console.print(Panel(
        "[bold cyan]Safety Constraint Validation[/bold cyan]\n"
        f"Spec: {spec_path}",
        border_style="cyan"
    ))

    result = validate_safety_constraints(spec_path)
    _display_validation_result(result)

    if not result.valid:
        raise typer.Exit(1)


# =============================================================================
# Core Validation Functions
# =============================================================================

def validate_spec(
    spec_path: Path,
    strict: bool = True,
    schema_version: str = "1.0",
    fix: bool = False,
    output: Optional[Path] = None,
) -> ValidationResult:
    """
    Validate an AgentSpec YAML file.

    Args:
        spec_path: Path to AgentSpec YAML
        strict: Enable strict validation
        schema_version: Schema version to validate against
        fix: Attempt to auto-fix issues
        output: Output file for validation report

    Returns:
        ValidationResult with all findings
    """
    console.print(Panel(
        "[bold cyan]AgentSpec Validation[/bold cyan]\n"
        f"File: {spec_path}\n"
        f"Mode: {'Strict' if strict else 'Standard'}",
        border_style="cyan"
    ))

    start_time = datetime.now()
    messages: List[ValidationMessage] = []
    checks_passed = 0
    checks_total = 0

    try:
        import yaml

        # Load spec
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = yaml.safe_load(f)

        if spec is None:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="SPEC001",
                message="Spec file is empty",
                location=str(spec_path),
            ))
            return _create_result(messages, checks_passed, checks_total, start_time)

        # Run validation checks
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Validating...", total=None)

            # 1. Required fields
            progress.update(task, description="Checking required fields...")
            passed, total, msgs = _validate_required_fields(spec, strict)
            checks_passed += passed
            checks_total += total
            messages.extend(msgs)

            # 2. Metadata validation
            progress.update(task, description="Validating metadata...")
            passed, total, msgs = _validate_metadata(spec, strict)
            checks_passed += passed
            checks_total += total
            messages.extend(msgs)

            # 3. Tools validation
            progress.update(task, description="Validating tools...")
            passed, total, msgs = _validate_tools(spec, strict)
            checks_passed += passed
            checks_total += total
            messages.extend(msgs)

            # 4. Inputs/Outputs validation
            progress.update(task, description="Validating data contracts...")
            passed, total, msgs = _validate_io_schema(spec, strict)
            checks_passed += passed
            checks_total += total
            messages.extend(msgs)

            # 5. Tests validation
            progress.update(task, description="Validating tests...")
            passed, total, msgs = _validate_tests(spec, strict)
            checks_passed += passed
            checks_total += total
            messages.extend(msgs)

            # 6. Safety constraints
            progress.update(task, description="Validating safety constraints...")
            passed, total, msgs = _validate_safety_spec(spec, strict)
            checks_passed += passed
            checks_total += total
            messages.extend(msgs)

            # 7. Provenance configuration
            progress.update(task, description="Validating provenance...")
            passed, total, msgs = _validate_provenance(spec, strict)
            checks_passed += passed
            checks_total += total
            messages.extend(msgs)

            # 8. Explainability
            progress.update(task, description="Validating explainability...")
            passed, total, msgs = _validate_explainability(spec, strict)
            checks_passed += passed
            checks_total += total
            messages.extend(msgs)

            progress.update(task, description="[green]Validation complete")

    except yaml.YAMLError as e:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="YAML001",
            message=f"YAML parsing error: {str(e)}",
            location=str(spec_path),
        ))
    except Exception as e:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="VAL001",
            message=f"Validation error: {str(e)}",
            location=str(spec_path),
        ))

    result = _create_result(messages, checks_passed, checks_total, start_time)

    # Display results
    _display_validation_result(result)

    # Save report if requested
    if output:
        _save_validation_report(result, output)

    # Auto-fix if requested
    if fix and result.warnings:
        _attempt_auto_fix(spec_path, spec, result)

    return result


def validate_agent_implementation(
    agent_path: Path,
    include_tests: bool = True,
    include_docs: bool = True,
) -> ValidationResult:
    """
    Validate a complete agent implementation.

    Args:
        agent_path: Path to agent directory
        include_tests: Validate test files
        include_docs: Validate documentation

    Returns:
        ValidationResult
    """
    start_time = datetime.now()
    messages: List[ValidationMessage] = []
    checks_passed = 0
    checks_total = 0

    # Check required files
    required_files = ["agent.py", "__init__.py"]
    for req_file in required_files:
        checks_total += 1
        if (agent_path / req_file).exists():
            checks_passed += 1
        else:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="FILE001",
                message=f"Required file missing: {req_file}",
                location=str(agent_path),
            ))

    # Check recommended files
    recommended_files = ["pack.yaml", "tools.py", "README.md"]
    for rec_file in recommended_files:
        checks_total += 1
        if (agent_path / rec_file).exists():
            checks_passed += 1
        else:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="FILE002",
                message=f"Recommended file missing: {rec_file}",
                location=str(agent_path),
                suggestion=f"Consider adding {rec_file}",
            ))

    # Validate Python syntax
    for py_file in agent_path.glob("*.py"):
        checks_total += 1
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                compile(f.read(), py_file, "exec")
            checks_passed += 1
        except SyntaxError as e:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code="PY001",
                message=f"Syntax error: {str(e)}",
                location=str(py_file),
            ))

    # Validate tests if requested
    if include_tests:
        tests_dir = agent_path / "tests"
        if tests_dir.exists():
            test_files = list(tests_dir.glob("test_*.py"))
            checks_total += 1
            if test_files:
                checks_passed += 1
            else:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    code="TEST001",
                    message="No test files found in tests/",
                    location=str(tests_dir),
                    suggestion="Add test_agent.py with golden tests",
                ))
        else:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="TEST002",
                message="Tests directory not found",
                location=str(agent_path),
                suggestion="Create tests/ directory with test files",
            ))

    # Validate docs if requested
    if include_docs:
        checks_total += 1
        if (agent_path / "README.md").exists():
            checks_passed += 1
        else:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.INFO,
                code="DOC001",
                message="README.md not found",
                location=str(agent_path),
                suggestion="Add README.md documentation",
            ))

    return _create_result(messages, checks_passed, checks_total, start_time)


def validate_data_contracts(
    spec_path: Path,
    sample_input: Optional[Path] = None,
) -> ValidationResult:
    """
    Validate data contracts (input/output schemas).

    Args:
        spec_path: Path to AgentSpec YAML
        sample_input: Optional sample input to validate

    Returns:
        ValidationResult
    """
    start_time = datetime.now()
    messages: List[ValidationMessage] = []
    checks_passed = 0
    checks_total = 0

    try:
        import yaml

        with open(spec_path, "r", encoding="utf-8") as f:
            spec = yaml.safe_load(f)

        # Validate inputs
        inputs = spec.get("inputs", [])
        checks_total += 1
        if inputs:
            checks_passed += 1
            for i, inp in enumerate(inputs):
                checks_total += 1
                if all(f in inp for f in INPUT_OUTPUT_REQUIRED_FIELDS):
                    checks_passed += 1
                else:
                    messages.append(ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        code="CONTRACT001",
                        message=f"Input {i} missing required fields",
                        location=f"inputs[{i}]",
                    ))
        else:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="CONTRACT002",
                message="No inputs defined",
                location="inputs",
            ))

        # Validate outputs
        outputs = spec.get("outputs", [])
        checks_total += 1
        if outputs:
            checks_passed += 1
            for i, out in enumerate(outputs):
                checks_total += 1
                if all(f in out for f in INPUT_OUTPUT_REQUIRED_FIELDS):
                    checks_passed += 1
                else:
                    messages.append(ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        code="CONTRACT003",
                        message=f"Output {i} missing required fields",
                        location=f"outputs[{i}]",
                    ))
        else:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="CONTRACT004",
                message="No outputs defined",
                location="outputs",
            ))

        # Validate sample input if provided
        if sample_input:
            checks_total += 1
            try:
                with open(sample_input, "r") as f:
                    sample = json.load(f)
                # Check all required inputs are present
                required_inputs = [i["name"] for i in inputs if i.get("required", True)]
                missing = [r for r in required_inputs if r not in sample]
                if missing:
                    messages.append(ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        code="CONTRACT005",
                        message=f"Sample missing required inputs: {missing}",
                        location=str(sample_input),
                    ))
                else:
                    checks_passed += 1
            except Exception as e:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    code="CONTRACT006",
                    message=f"Failed to validate sample: {str(e)}",
                    location=str(sample_input),
                ))

    except Exception as e:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="CONTRACT007",
            message=f"Contract validation error: {str(e)}",
        ))

    return _create_result(messages, checks_passed, checks_total, start_time)


def validate_safety_constraints(spec_path: Path) -> ValidationResult:
    """
    Validate safety constraints and guardrails.

    Args:
        spec_path: Path to AgentSpec YAML

    Returns:
        ValidationResult
    """
    start_time = datetime.now()
    messages: List[ValidationMessage] = []
    checks_passed = 0
    checks_total = 0

    try:
        import yaml

        with open(spec_path, "r", encoding="utf-8") as f:
            spec = yaml.safe_load(f)

        safety = spec.get("safety", {})

        # Check for safety section
        checks_total += 1
        if safety:
            checks_passed += 1

            # Check max_tokens
            checks_total += 1
            if "max_tokens" in safety:
                checks_passed += 1
                if safety["max_tokens"] > 10000:
                    messages.append(ValidationMessage(
                        severity=ValidationSeverity.WARNING,
                        code="SAFETY001",
                        message="max_tokens is very high (>10000)",
                        location="safety.max_tokens",
                        suggestion="Consider lowering max_tokens for cost control",
                    ))
            else:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    code="SAFETY002",
                    message="max_tokens not specified",
                    location="safety",
                    suggestion="Add max_tokens constraint",
                ))

            # Check rate_limit
            checks_total += 1
            if "rate_limit" in safety:
                checks_passed += 1
            else:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    code="SAFETY003",
                    message="rate_limit not specified",
                    location="safety",
                    suggestion="Add rate_limit for production safety",
                ))

            # Check timeout
            checks_total += 1
            if "timeout_seconds" in safety:
                checks_passed += 1
            else:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    code="SAFETY004",
                    message="timeout_seconds not specified",
                    location="safety",
                    suggestion="Add timeout_seconds for reliability",
                ))

            # Check allowed_tools
            checks_total += 1
            if "allowed_tools" in safety:
                checks_passed += 1
            else:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.INFO,
                    code="SAFETY005",
                    message="allowed_tools not specified",
                    location="safety",
                    suggestion="Consider restricting allowed tools",
                ))
        else:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="SAFETY006",
                message="No safety section defined",
                location="safety",
                suggestion="Add safety constraints for production use",
            ))

    except Exception as e:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.ERROR,
            code="SAFETY007",
            message=f"Safety validation error: {str(e)}",
        ))

    return _create_result(messages, checks_passed, checks_total, start_time)


# =============================================================================
# Validation Helper Functions
# =============================================================================

def _validate_required_fields(spec: dict, strict: bool) -> Tuple[int, int, List[ValidationMessage]]:
    """Validate required spec fields."""
    messages = []
    passed = 0
    total = len(REQUIRED_SPEC_FIELDS)

    for field in REQUIRED_SPEC_FIELDS:
        if field in spec:
            passed += 1
        else:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.ERROR,
                code=f"REQ001",
                message=f"Required field missing: {field}",
                location=field,
            ))

    # Check recommended fields
    if strict:
        for field in RECOMMENDED_SPEC_FIELDS:
            total += 1
            if field in spec:
                passed += 1
            else:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    code=f"REQ002",
                    message=f"Recommended field missing: {field}",
                    location=field,
                    suggestion=f"Consider adding '{field}' section",
                ))

    return passed, total, messages


def _validate_metadata(spec: dict, strict: bool) -> Tuple[int, int, List[ValidationMessage]]:
    """Validate metadata section."""
    messages = []
    passed = 0
    total = 0

    metadata = spec.get("metadata", {})

    # Check author
    total += 1
    if metadata.get("author"):
        passed += 1
    else:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.INFO,
            code="META001",
            message="No author specified",
            location="metadata.author",
        ))

    # Check category
    total += 1
    category = metadata.get("category")
    if category:
        if category in VALID_CATEGORIES:
            passed += 1
        else:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="META002",
                message=f"Unknown category: {category}",
                location="metadata.category",
                suggestion=f"Valid categories: {', '.join(VALID_CATEGORIES)}",
            ))
    else:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.INFO,
            code="META003",
            message="No category specified",
            location="metadata.category",
        ))

    return passed, total, messages


def _validate_tools(spec: dict, strict: bool) -> Tuple[int, int, List[ValidationMessage]]:
    """Validate tools section."""
    messages = []
    passed = 0
    total = 0

    tools = spec.get("tools", [])

    total += 1
    if tools:
        passed += 1
    else:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="TOOL001",
            message="No tools defined",
            location="tools",
        ))
        return passed, total, messages

    for i, tool in enumerate(tools):
        # Check required fields
        for field in TOOL_REQUIRED_FIELDS:
            total += 1
            if field in tool:
                passed += 1
            else:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    code="TOOL002",
                    message=f"Tool {i} missing required field: {field}",
                    location=f"tools[{i}].{field}",
                ))

        # Check tool type
        tool_type = tool.get("type")
        if tool_type:
            total += 1
            if tool_type in VALID_TOOL_TYPES:
                passed += 1
            else:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    code="TOOL003",
                    message=f"Unknown tool type: {tool_type}",
                    location=f"tools[{i}].type",
                    suggestion=f"Valid types: {', '.join(VALID_TOOL_TYPES)}",
                ))

        # Check for LLM in calculation tools
        if strict and tool_type in ["llm_generation", "llm_classification"]:
            tool_name = tool.get("name", f"tool_{i}")
            if "calc" in tool_name.lower() or "compute" in tool_name.lower():
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    code="TOOL004",
                    message=f"Tool '{tool_name}' uses LLM but appears to be a calculation",
                    location=f"tools[{i}]",
                    suggestion="Use 'deterministic' type for calculations (zero-hallucination)",
                ))

    return passed, total, messages


def _validate_io_schema(spec: dict, strict: bool) -> Tuple[int, int, List[ValidationMessage]]:
    """Validate inputs/outputs schema."""
    messages = []
    passed = 0
    total = 0

    # Validate inputs
    inputs = spec.get("inputs", [])
    total += 1
    if inputs:
        passed += 1
    else:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="IO001",
            message="No inputs defined",
            location="inputs",
        ))

    # Validate outputs
    outputs = spec.get("outputs", [])
    total += 1
    if outputs:
        passed += 1
        # Check for provenance output
        if strict:
            total += 1
            has_provenance = any(
                "provenance" in out.get("name", "").lower()
                for out in outputs
            )
            if has_provenance:
                passed += 1
            else:
                messages.append(ValidationMessage(
                    severity=ValidationSeverity.WARNING,
                    code="IO002",
                    message="No provenance output defined",
                    location="outputs",
                    suggestion="Add provenance_hash output for audit trails",
                ))
    else:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="IO003",
            message="No outputs defined",
            location="outputs",
        ))

    return passed, total, messages


def _validate_tests(spec: dict, strict: bool) -> Tuple[int, int, List[ValidationMessage]]:
    """Validate tests section."""
    messages = []
    passed = 0
    total = 0

    tests = spec.get("tests", {})

    total += 1
    if tests:
        passed += 1
    else:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="TEST001",
            message="No tests defined",
            location="tests",
            suggestion="Add golden tests for determinism verification",
        ))
        return passed, total, messages

    # Check golden tests
    golden = tests.get("golden", [])
    total += 1
    if golden:
        passed += 1
        if strict and len(golden) < 5:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.INFO,
                code="TEST002",
                message=f"Only {len(golden)} golden tests defined",
                location="tests.golden",
                suggestion="Consider adding more golden tests (recommended: 10+)",
            ))
    else:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="TEST003",
            message="No golden tests defined",
            location="tests.golden",
            suggestion="Golden tests verify determinism (zero-hallucination)",
        ))

    return passed, total, messages


def _validate_safety_spec(spec: dict, strict: bool) -> Tuple[int, int, List[ValidationMessage]]:
    """Validate safety section."""
    messages = []
    passed = 0
    total = 0

    safety = spec.get("safety", {})

    total += 1
    if safety:
        passed += 1
    else:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="SAFE001",
            message="No safety constraints defined",
            location="safety",
        ))

    return passed, total, messages


def _validate_provenance(spec: dict, strict: bool) -> Tuple[int, int, List[ValidationMessage]]:
    """Validate provenance section."""
    messages = []
    passed = 0
    total = 0

    provenance = spec.get("provenance", {})

    total += 1
    if provenance:
        passed += 1

        # Check algorithm
        total += 1
        algo = provenance.get("algorithm", "sha256")
        if algo in ["sha256", "sha512", "sha3_256"]:
            passed += 1
        else:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.WARNING,
                code="PROV001",
                message=f"Non-standard hash algorithm: {algo}",
                location="provenance.algorithm",
                suggestion="Use sha256 for compatibility",
            ))
    else:
        messages.append(ValidationMessage(
            severity=ValidationSeverity.WARNING,
            code="PROV002",
            message="No provenance configuration",
            location="provenance",
            suggestion="Add provenance tracking for audit trails",
        ))

    return passed, total, messages


def _validate_explainability(spec: dict, strict: bool) -> Tuple[int, int, List[ValidationMessage]]:
    """Validate explainability section."""
    messages = []
    passed = 0
    total = 0

    explainability = spec.get("explainability", {})

    total += 1
    if explainability:
        passed += 1
    else:
        if strict:
            messages.append(ValidationMessage(
                severity=ValidationSeverity.INFO,
                code="EXPL001",
                message="No explainability configuration",
                location="explainability",
                suggestion="Add explainability for transparency",
            ))

    return passed, total, messages


# =============================================================================
# Display and Report Functions
# =============================================================================

def _create_result(
    messages: List[ValidationMessage],
    checks_passed: int,
    checks_total: int,
    start_time: datetime,
) -> ValidationResult:
    """Create ValidationResult from messages."""
    errors = [m for m in messages if m.severity == ValidationSeverity.ERROR]
    warnings = [m for m in messages if m.severity == ValidationSeverity.WARNING]
    info = [m for m in messages if m.severity == ValidationSeverity.INFO]

    valid = len(errors) == 0
    score = (checks_passed / checks_total * 100) if checks_total > 0 else 0

    return ValidationResult(
        valid=valid,
        score=score,
        errors=errors,
        warnings=warnings,
        info=info,
        checks_passed=checks_passed,
        checks_total=checks_total,
        duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
    )


def _display_validation_result(result: ValidationResult) -> None:
    """Display validation result in rich format."""
    console.print()

    # Summary panel
    status = "[green]PASSED[/green]" if result.valid else "[red]FAILED[/red]"
    score_color = "green" if result.score >= 80 else "yellow" if result.score >= 60 else "red"

    console.print(Panel(
        f"[bold]Status:[/bold] {status}\n"
        f"[bold]Score:[/bold] [{score_color}]{result.score:.1f}%[/{score_color}]\n"
        f"[bold]Checks:[/bold] {result.checks_passed}/{result.checks_total} passed\n"
        f"[bold]Duration:[/bold] {result.duration_ms:.2f}ms",
        title="Validation Summary",
        border_style="cyan",
    ))

    # Errors
    if result.errors:
        console.print("\n[bold red]Errors:[/bold red]")
        for err in result.errors:
            console.print(f"  [red]x[/red] [{err.code}] {err.message}")
            if err.location:
                console.print(f"      Location: {err.location}")

    # Warnings
    if result.warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for warn in result.warnings:
            console.print(f"  [yellow]![/yellow] [{warn.code}] {warn.message}")
            if warn.suggestion:
                console.print(f"      Suggestion: {warn.suggestion}")

    # Info
    if result.info:
        console.print("\n[bold blue]Info:[/bold blue]")
        for inf in result.info[:5]:  # Limit to 5
            console.print(f"  [blue]i[/blue] [{inf.code}] {inf.message}")
        if len(result.info) > 5:
            console.print(f"  ... and {len(result.info) - 5} more")

    console.print()


def _save_validation_report(result: ValidationResult, output: Path) -> None:
    """Save validation report to file."""
    report = {
        "valid": result.valid,
        "score": result.score,
        "checks_passed": result.checks_passed,
        "checks_total": result.checks_total,
        "duration_ms": result.duration_ms,
        "timestamp": result.timestamp,
        "errors": [m.model_dump() for m in result.errors],
        "warnings": [m.model_dump() for m in result.warnings],
        "info": [m.model_dump() for m in result.info],
    }

    suffix = output.suffix.lower()
    if suffix == ".json":
        with open(output, "w") as f:
            json.dump(report, f, indent=2)
    elif suffix in [".yaml", ".yml"]:
        import yaml
        with open(output, "w") as f:
            yaml.dump(report, f, default_flow_style=False)
    else:
        # Default to JSON
        with open(output, "w") as f:
            json.dump(report, f, indent=2)

    console.print(f"[green]Report saved to:[/green] {output}")


def _attempt_auto_fix(spec_path: Path, spec: dict, result: ValidationResult) -> None:
    """Attempt to auto-fix validation issues."""
    import yaml

    fixed = False

    # Add missing recommended sections
    if "safety" not in spec:
        spec["safety"] = {
            "max_tokens": 4096,
            "rate_limit": 100,
            "timeout_seconds": 30,
        }
        fixed = True

    if "provenance" not in spec:
        spec["provenance"] = {
            "algorithm": "sha256",
            "tracking": "full",
        }
        fixed = True

    if fixed:
        # Backup original
        backup_path = spec_path.with_suffix(".yaml.bak")
        shutil.copy(spec_path, backup_path)
        console.print(f"[yellow]Backup created:[/yellow] {backup_path}")

        # Write fixed spec
        with open(spec_path, "w") as f:
            yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
        console.print(f"[green]Auto-fixed issues in:[/green] {spec_path}")
