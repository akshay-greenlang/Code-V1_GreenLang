# -*- coding: utf-8 -*-
"""
Lint Command for GL-FOUND-X-002 (GreenLang Schema Compiler & Validator).

This module implements the 'greenlang schema lint' command for linting
schemas for best practices and potential issues.

Features:
    - Schema syntax validation
    - Best practice checks
    - ReDoS vulnerability detection in regex patterns
    - Deprecation warnings
    - Constraint consistency checks
    - Unit specification validation

Exit Codes:
    0 - Success (no issues found)
    1 - Warnings found (non-blocking issues)
    2 - Errors found (schema has problems)

Example:
    $ greenlang schema lint schema.yaml
    $ greenlang schema lint schema.yaml --format json

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 5.1
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from greenlang.schema.compiler.compiler import SchemaCompiler
from greenlang.schema.compiler.parser import parse_payload, ParseError
from greenlang.schema.models.schema_ref import SchemaRef


logger = logging.getLogger(__name__)

# Exit codes
EXIT_SUCCESS = 0
EXIT_WARNINGS = 1
EXIT_ERRORS = 2


@dataclass
class LintFinding:
    """
    A single lint finding.

    Attributes:
        code: Finding code (e.g., 'L001').
        severity: Severity level ('error', 'warning', 'info').
        path: JSON Pointer path to the issue.
        message: Description of the issue.
        suggestion: Optional suggestion for fixing.
    """
    code: str
    severity: str
    path: str
    message: str
    suggestion: Optional[str] = None


@dataclass
class LintResult:
    """
    Result of linting a schema.

    Attributes:
        schema_path: Path to the schema file.
        errors: List of error findings.
        warnings: List of warning findings.
        info: List of informational findings.
        lint_time_ms: Time taken to lint in milliseconds.
    """
    schema_path: str
    errors: List[LintFinding] = field(default_factory=list)
    warnings: List[LintFinding] = field(default_factory=list)
    info: List[LintFinding] = field(default_factory=list)
    lint_time_ms: float = 0.0

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    @property
    def total_findings(self) -> int:
        """Get total number of findings."""
        return len(self.errors) + len(self.warnings) + len(self.info)


def read_schema(schema_path: str) -> Dict[str, Any]:
    """
    Read and parse a schema file.

    Args:
        schema_path: Path to the schema file.

    Returns:
        Parsed schema dictionary.

    Raises:
        click.BadParameter: If file cannot be read or parsed.
    """
    file_path = Path(schema_path)

    if not file_path.exists():
        raise click.BadParameter(f"Schema file not found: {schema_path}")
    if not file_path.is_file():
        raise click.BadParameter(f"Not a file: {schema_path}")

    try:
        content = file_path.read_text(encoding="utf-8")
        result = parse_payload(content)
        return result.data
    except ParseError as e:
        raise click.BadParameter(f"Failed to parse schema: {e.message}")
    except IOError as e:
        raise click.BadParameter(f"Cannot read schema file: {e}")


def lint_schema(schema_dict: Dict[str, Any], schema_path: str) -> LintResult:
    """
    Lint a schema for issues and best practices.

    Args:
        schema_dict: Parsed schema dictionary.
        schema_path: Path to the schema (for reporting).

    Returns:
        LintResult with findings.
    """
    start_time = time.perf_counter()
    result = LintResult(schema_path=schema_path)

    # Check 1: Schema has $schema or type
    if "$schema" not in schema_dict and "type" not in schema_dict:
        result.warnings.append(LintFinding(
            code="L001",
            severity="warning",
            path="",
            message="Schema missing '$schema' dialect declaration",
            suggestion="Add '$schema': 'https://json-schema.org/draft/2020-12/schema'"
        ))

    # Check 2: Compile schema to detect issues
    try:
        compiler = SchemaCompiler()
        compilation_result = compiler.compile(
            schema_source=schema_dict,
            schema_id=Path(schema_path).stem,
            version="1.0.0",
        )

        # Convert compilation warnings to lint findings
        for warning in compilation_result.warnings:
            result.warnings.append(LintFinding(
                code="L100",
                severity="warning",
                path="",
                message=warning,
            ))

        # Convert compilation errors to lint findings
        for error in compilation_result.errors:
            result.errors.append(LintFinding(
                code="L200",
                severity="error",
                path="",
                message=error,
            ))

        if compilation_result.success and compilation_result.ir:
            ir = compilation_result.ir

            # Check 3: Pattern safety
            for path, pattern_info in ir.patterns.items():
                if not pattern_info.is_safe:
                    result.warnings.append(LintFinding(
                        code="L002",
                        severity="warning",
                        path=path,
                        message=f"Potentially unsafe regex pattern: {pattern_info.vulnerability_type}",
                        suggestion=pattern_info.recommendation,
                    ))

                if pattern_info.complexity_score > 0.5:
                    result.info.append(LintFinding(
                        code="L003",
                        severity="info",
                        path=path,
                        message=f"High complexity regex (score: {pattern_info.complexity_score:.2f})",
                        suggestion="Consider simplifying the pattern",
                    ))

                if not pattern_info.is_re2_compatible:
                    result.info.append(LintFinding(
                        code="L004",
                        severity="info",
                        path=path,
                        message="Pattern is not RE2-compatible (uses backreferences or lookaround)",
                        suggestion="Use RE2-compatible patterns for better performance",
                    ))

            # Check 4: Deprecated fields without replacement
            for path, dep_info in ir.deprecated_fields.items():
                if not dep_info.get("replacement"):
                    result.warnings.append(LintFinding(
                        code="L005",
                        severity="warning",
                        path=path,
                        message="Deprecated field without replacement specified",
                        suggestion="Add 'replacement' field to deprecation notice",
                    ))

            # Check 5: Unit specs without dimension
            for path, unit_spec in ir.unit_specs.items():
                if not unit_spec.dimension:
                    result.warnings.append(LintFinding(
                        code="L006",
                        severity="warning",
                        path=path,
                        message="Unit specification missing dimension",
                        suggestion="Add 'dimension' to $unit specification",
                    ))

            # Check 6: Empty required arrays
            lint_nested_schema(schema_dict, "", result)

    except Exception as e:
        logger.error(f"Error during compilation: {e}")
        result.errors.append(LintFinding(
            code="L999",
            severity="error",
            path="",
            message=f"Schema compilation failed: {str(e)}",
        ))

    result.lint_time_ms = (time.perf_counter() - start_time) * 1000
    return result


def lint_nested_schema(
    schema: Dict[str, Any],
    path: str,
    result: LintResult,
) -> None:
    """
    Recursively lint nested schema structures.

    Args:
        schema: Schema or sub-schema dictionary.
        path: Current JSON Pointer path.
        result: LintResult to add findings to.
    """
    # Check for empty required arrays
    required = schema.get("required", [])
    if isinstance(required, list) and len(required) == 0:
        result.info.append(LintFinding(
            code="L007",
            severity="info",
            path=path,
            message="Empty 'required' array",
            suggestion="Remove empty 'required' array or add required fields",
        ))

    # Check for empty properties
    properties = schema.get("properties", {})
    if isinstance(properties, dict) and len(properties) == 0:
        result.info.append(LintFinding(
            code="L008",
            severity="info",
            path=path,
            message="Empty 'properties' object",
            suggestion="Add properties or remove empty object",
        ))

    # Check for overly permissive additionalProperties
    if schema.get("additionalProperties") is True and schema.get("type") == "object":
        if "properties" in schema:
            result.info.append(LintFinding(
                code="L009",
                severity="info",
                path=path,
                message="additionalProperties=true allows any extra fields",
                suggestion="Consider setting additionalProperties to false or a specific schema",
            ))

    # Check for missing type
    if "type" not in schema and "properties" in schema:
        result.warnings.append(LintFinding(
            code="L010",
            severity="warning",
            path=path,
            message="Object schema missing 'type' declaration",
            suggestion="Add 'type': 'object' to schema",
        ))

    # Check for potentially problematic enum values
    enum_values = schema.get("enum", [])
    if isinstance(enum_values, list):
        # Check for mixed types in enum
        types_in_enum = set(type(v).__name__ for v in enum_values)
        if len(types_in_enum) > 1:
            result.info.append(LintFinding(
                code="L011",
                severity="info",
                path=path,
                message=f"Enum contains mixed types: {', '.join(types_in_enum)}",
                suggestion="Consider using consistent types in enum",
            ))

        # Check for very large enums
        if len(enum_values) > 100:
            result.warnings.append(LintFinding(
                code="L012",
                severity="warning",
                path=path,
                message=f"Large enum with {len(enum_values)} values",
                suggestion="Consider using pattern or external reference for large value sets",
            ))

    # Recurse into properties
    if isinstance(properties, dict):
        for prop_name, prop_schema in properties.items():
            if isinstance(prop_schema, dict):
                prop_path = f"{path}/{prop_name}"
                lint_nested_schema(prop_schema, prop_path, result)

    # Recurse into items
    items = schema.get("items")
    if isinstance(items, dict):
        items_path = f"{path}/items"
        lint_nested_schema(items, items_path, result)

    # Recurse into definitions
    for defs_key in ("definitions", "$defs"):
        defs = schema.get(defs_key, {})
        if isinstance(defs, dict):
            for def_name, def_schema in defs.items():
                if isinstance(def_schema, dict):
                    def_path = f"/{defs_key}/{def_name}"
                    lint_nested_schema(def_schema, def_path, result)


def format_lint_result(result: LintResult, format_type: str) -> str:
    """
    Format lint result for output.

    Args:
        result: Lint result.
        format_type: Output format ('pretty', 'json').

    Returns:
        Formatted output string.
    """
    if format_type == "json":
        output = {
            "schema_path": result.schema_path,
            "summary": {
                "errors": len(result.errors),
                "warnings": len(result.warnings),
                "info": len(result.info),
            },
            "findings": [
                {
                    "code": f.code,
                    "severity": f.severity,
                    "path": f.path,
                    "message": f.message,
                    "suggestion": f.suggestion,
                }
                for f in result.errors + result.warnings + result.info
            ],
            "lint_time_ms": result.lint_time_ms,
        }
        return json.dumps(output, indent=2)

    # Pretty format
    lines = []

    # Header
    if result.has_errors:
        status = "ERRORS"
    elif result.has_warnings:
        status = "WARNINGS"
    else:
        status = "OK"

    lines.append(f"\n{status}  {result.schema_path}")
    lines.append(f"  {len(result.errors)} error(s), {len(result.warnings)} warning(s), {len(result.info)} info\n")

    # Findings
    all_findings = (
        [("ERROR", f) for f in result.errors] +
        [("WARNING", f) for f in result.warnings] +
        [("INFO", f) for f in result.info]
    )

    for severity, finding in all_findings:
        path_str = finding.path if finding.path else "(root)"
        lines.append(f"  {severity} {finding.code} at {path_str}")
        lines.append(f"    {finding.message}")
        if finding.suggestion:
            lines.append(f"    Suggestion: {finding.suggestion}")
        lines.append("")

    lines.append(f"Lint completed in {result.lint_time_ms:.2f}ms")

    return "\n".join(lines)


@click.command()
@click.argument(
    "schema",
    type=str,
    required=True,
)
@click.option(
    "--format", "-f",
    "format_type",
    type=click.Choice(["pretty", "json"], case_sensitive=False),
    default="pretty",
    help="Output format.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Treat warnings as errors (exit code 2 for warnings).",
)
@click.option(
    "-v", "--verbose",
    "verbosity",
    count=True,
    help="Increase verbosity. -v shows info findings, -vv shows debug info.",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    default=False,
    help="Suppress all output except exit code.",
)
@click.pass_context
def lint(
    ctx: click.Context,
    schema: str,
    format_type: str,
    strict: bool,
    verbosity: int,
    quiet: bool,
) -> None:
    """
    Lint a schema for best practices and potential issues.

    \b
    Arguments:
        SCHEMA  Path to schema file.

    \b
    Checks Performed:
        - Schema dialect declaration
        - Regex pattern safety (ReDoS detection)
        - Constraint consistency
        - Deprecation notices
        - Unit specification completeness
        - Best practices (type declarations, etc.)

    \b
    Exit Codes:
        0  OK - No errors or warnings
        1  Warnings - Non-blocking issues found
        2  Errors - Schema has problems

    \b
    Examples:
        # Lint a schema
        greenlang schema lint schema.yaml

        # Output as JSON
        greenlang schema lint schema.yaml --format json

        # Strict mode (warnings are errors)
        greenlang schema lint schema.yaml --strict

        # Show all findings including info
        greenlang schema lint schema.yaml -v
    """
    # Configure logging
    if verbosity >= 2:
        logging.basicConfig(level=logging.DEBUG)
    elif verbosity >= 1:
        logging.basicConfig(level=logging.INFO)

    try:
        # Read and parse schema
        schema_dict = read_schema(schema)

        # Lint schema
        result = lint_schema(schema_dict, schema)

        # Filter findings based on verbosity
        if verbosity < 1:
            # Hide info findings unless verbose
            result.info = []

        # Output result
        if not quiet:
            output = format_lint_result(result, format_type)
            click.echo(output)

        # Determine exit code
        if result.has_errors:
            ctx.exit(EXIT_ERRORS)
        elif result.has_warnings:
            if strict:
                ctx.exit(EXIT_ERRORS)
            else:
                ctx.exit(EXIT_WARNINGS)
        else:
            ctx.exit(EXIT_SUCCESS)

    except click.BadParameter as e:
        if not quiet:
            click.echo(f"Error: {e}", err=True)
        ctx.exit(EXIT_ERRORS)
    except Exception as e:
        logger.exception("Lint failed")
        if not quiet:
            click.echo(f"Error: {e}", err=True)
        ctx.exit(EXIT_ERRORS)
