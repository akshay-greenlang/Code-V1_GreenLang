# -*- coding: utf-8 -*-
"""
Validate Command for GL-FOUND-X-002 (GreenLang Schema Compiler & Validator).

This module implements the 'greenlang schema validate' command for validating
payloads against GreenLang schemas.

Features:
    - Single file validation
    - Stdin input support (using '-' as filename)
    - Glob pattern batch validation
    - Multiple output formats (pretty, text, table, json, sarif)
    - Configurable validation profiles (strict, standard, permissive)
    - Fix suggestion generation with safety levels
    - Normalized payload output

Exit Codes:
    0 - Valid (payload passed validation)
    1 - Invalid (payload failed validation)
    2 - Error (system error, missing file, etc.)

Example:
    $ greenlang schema validate data.yaml --schema emissions/activity@1.3.0
    $ greenlang schema validate - --schema test@1.0.0 < data.json
    $ greenlang schema validate --glob "data/*.yaml" --schema test@1.0.0

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 5.1
"""

from __future__ import annotations

import glob as glob_module
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union

import click

from greenlang.schema.models.config import (
    CoercionPolicy,
    PatchLevel,
    UnknownFieldPolicy,
    ValidationOptions,
    ValidationProfile,
)
from greenlang.schema.models.finding import Finding, Severity
from greenlang.schema.models.report import (
    BatchValidationReport,
    ValidationReport,
    ValidationSummary,
)
from greenlang.schema.models.schema_ref import SchemaRef
from greenlang.schema.validator.core import SchemaValidator
from greenlang.schema.cli.formatters import (
    format_json,
    format_pretty,
    format_sarif,
    format_table,
    format_text,
)


logger = logging.getLogger(__name__)

# Exit codes
EXIT_VALID = 0
EXIT_INVALID = 1
EXIT_ERROR = 2

# Default verbosity settings
DEFAULT_MAX_ERRORS_DISPLAY = 5
VERBOSE_MAX_ERRORS_DISPLAY = 100
DEBUG_MAX_ERRORS_DISPLAY = 1000


def parse_schema_ref(schema_str: str) -> SchemaRef:
    """
    Parse a schema reference from string input.

    Supports multiple formats:
        - URI: gl://schemas/emissions/activity@1.3.0
        - Short: emissions/activity@1.3.0
        - Path: ./schemas/activity.yaml (treated as inline)

    Args:
        schema_str: Schema reference string.

    Returns:
        SchemaRef object.

    Raises:
        click.BadParameter: If schema reference is invalid.
    """
    try:
        # Check if it's a GL URI
        if schema_str.startswith("gl://"):
            return SchemaRef.from_uri(schema_str)

        # Check if it's a file path
        if "/" in schema_str and "@" not in schema_str:
            # Looks like a file path, not a schema ref
            raise click.BadParameter(
                f"Schema reference '{schema_str}' looks like a file path. "
                "Use --schema-file for local schema files."
            )

        # Parse as short form: schema_id@version
        if "@" in schema_str:
            schema_id, version = schema_str.rsplit("@", 1)
            return SchemaRef(schema_id=schema_id, version=version)

        raise click.BadParameter(
            f"Invalid schema reference '{schema_str}'. "
            "Expected format: 'schema_id@version' or 'gl://schemas/schema_id@version'"
        )
    except ValueError as e:
        raise click.BadParameter(str(e))


def read_payload(file_arg: str, stdin: TextIO) -> Tuple[str, str]:
    """
    Read payload from file or stdin.

    Args:
        file_arg: File path or '-' for stdin.
        stdin: Stdin stream for reading.

    Returns:
        Tuple of (content, source_name).

    Raises:
        click.BadParameter: If file cannot be read.
    """
    if file_arg == "-":
        # Read from stdin
        content = stdin.read()
        return content, "<stdin>"
    else:
        # Read from file
        file_path = Path(file_arg)
        if not file_path.exists():
            raise click.BadParameter(f"File not found: {file_arg}")
        if not file_path.is_file():
            raise click.BadParameter(f"Not a file: {file_arg}")

        try:
            content = file_path.read_text(encoding="utf-8")
            return content, str(file_path)
        except IOError as e:
            raise click.BadParameter(f"Cannot read file {file_arg}: {e}")


def create_validation_options(
    profile: str,
    patch_level: str,
    return_normalized: bool,
    fail_on_warnings: bool,
    max_errors: int,
) -> ValidationOptions:
    """
    Create ValidationOptions from CLI arguments.

    Args:
        profile: Validation profile name.
        patch_level: Patch safety level.
        return_normalized: Whether to return normalized payload.
        fail_on_warnings: Whether warnings should cause failure.
        max_errors: Maximum errors to report.

    Returns:
        Configured ValidationOptions.
    """
    # Map profile string to enum
    profile_map = {
        "strict": ValidationProfile.STRICT,
        "standard": ValidationProfile.STANDARD,
        "permissive": ValidationProfile.PERMISSIVE,
    }
    profile_enum = profile_map.get(profile, ValidationProfile.STANDARD)

    # Map patch level string to enum
    patch_level_map = {
        "safe": PatchLevel.SAFE,
        "needs_review": PatchLevel.NEEDS_REVIEW,
        "unsafe": PatchLevel.UNSAFE,
    }
    patch_level_enum = patch_level_map.get(patch_level, PatchLevel.SAFE)

    # Set unknown field policy based on profile
    if profile_enum == ValidationProfile.STRICT:
        unknown_policy = UnknownFieldPolicy.ERROR
    elif profile_enum == ValidationProfile.PERMISSIVE:
        unknown_policy = UnknownFieldPolicy.IGNORE
    else:
        unknown_policy = UnknownFieldPolicy.WARN

    return ValidationOptions(
        profile=profile_enum,
        normalize=return_normalized,
        emit_patches=True,
        patch_level=patch_level_enum,
        max_errors=max_errors,
        fail_fast=False,
        unknown_field_policy=unknown_policy,
        coercion_policy=CoercionPolicy.SAFE,
    )


def format_output(
    report: Union[ValidationReport, BatchValidationReport],
    format_type: str,
    verbosity: int,
    quiet: bool,
    source_name: str = "",
) -> str:
    """
    Format validation report for output.

    Args:
        report: Validation report to format.
        format_type: Output format (pretty, text, table, json, sarif).
        verbosity: Verbosity level (0, 1, 2).
        quiet: Whether to suppress extra output.
        source_name: Name of the source file (for display).

    Returns:
        Formatted output string.
    """
    # Determine how many findings to show based on verbosity
    if quiet:
        max_findings = 0
    elif verbosity >= 2:
        max_findings = DEBUG_MAX_ERRORS_DISPLAY
    elif verbosity >= 1:
        max_findings = VERBOSE_MAX_ERRORS_DISPLAY
    else:
        max_findings = DEFAULT_MAX_ERRORS_DISPLAY

    if format_type == "json":
        return format_json(report, max_findings=max_findings)
    elif format_type == "sarif":
        return format_sarif(report, source_name=source_name)
    elif format_type == "table":
        return format_table(report, max_findings=max_findings)
    elif format_type == "text":
        return format_text(report, max_findings=max_findings)
    else:
        # Default to pretty
        return format_pretty(report, max_findings=max_findings, show_hints=(verbosity >= 1))


def validate_single_file(
    payload_content: str,
    source_name: str,
    schema_ref: SchemaRef,
    options: ValidationOptions,
) -> ValidationReport:
    """
    Validate a single payload.

    Args:
        payload_content: Payload content as string.
        source_name: Source file name (for display).
        schema_ref: Schema reference.
        options: Validation options.

    Returns:
        ValidationReport.
    """
    validator = SchemaValidator(options=options)
    return validator.validate(payload=payload_content, schema_ref=schema_ref, options=options)


def validate_glob_pattern(
    pattern: str,
    schema_ref: SchemaRef,
    options: ValidationOptions,
) -> Tuple[List[Tuple[str, ValidationReport]], int, int]:
    """
    Validate multiple files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., "data/*.yaml").
        schema_ref: Schema reference.
        options: Validation options.

    Returns:
        Tuple of (results, valid_count, invalid_count).
        Results is a list of (filename, report) tuples.
    """
    files = glob_module.glob(pattern, recursive=True)

    if not files:
        raise click.BadParameter(f"No files match pattern: {pattern}")

    results: List[Tuple[str, ValidationReport]] = []
    valid_count = 0
    invalid_count = 0

    validator = SchemaValidator(options=options)

    for file_path in sorted(files):
        try:
            content = Path(file_path).read_text(encoding="utf-8")
            report = validator.validate(
                payload=content,
                schema_ref=schema_ref,
                options=options,
            )
            results.append((file_path, report))

            if report.valid:
                valid_count += 1
            else:
                invalid_count += 1
        except Exception as e:
            logger.error(f"Error validating {file_path}: {e}")
            # Create error report for this file
            error_finding = Finding(
                code="GLSCHEMA-E800",
                severity=Severity.ERROR,
                path="",
                message=f"Failed to validate file: {str(e)}",
            )
            error_report = ValidationReport(
                valid=False,
                schema_ref=schema_ref,
                schema_hash="0" * 64,
                summary=ValidationSummary(valid=False, error_count=1),
                findings=[error_finding],
                timings={"total_ms": 0.0},
            )
            results.append((file_path, error_report))
            invalid_count += 1

    return results, valid_count, invalid_count


@click.command()
@click.argument(
    "file",
    type=str,
    required=False,
    default=None,
)
@click.option(
    "--schema", "-s",
    "schema_str",
    required=True,
    help="Schema reference (e.g., 'emissions/activity@1.3.0' or 'gl://schemas/...@1.0.0').",
)
@click.option(
    "--profile", "-p",
    type=click.Choice(["strict", "standard", "permissive"], case_sensitive=False),
    default="standard",
    help="Validation profile. strict=all warnings are errors, permissive=only critical errors.",
)
@click.option(
    "--format", "-f",
    "format_type",
    type=click.Choice(["pretty", "text", "table", "json", "sarif"], case_sensitive=False),
    default="pretty",
    help="Output format. json/sarif for CI, pretty/table for humans.",
)
@click.option(
    "--patch-level",
    type=click.Choice(["safe", "needs_review", "unsafe"], case_sensitive=False),
    default="safe",
    help="Maximum safety level for fix suggestions.",
)
@click.option(
    "--return-normalized", "-n",
    is_flag=True,
    default=False,
    help="Include normalized payload in output.",
)
@click.option(
    "--fail-on-warnings", "-w",
    is_flag=True,
    default=False,
    help="Exit with code 1 if there are any warnings.",
)
@click.option(
    "--max-errors",
    type=int,
    default=100,
    help="Maximum number of errors to report (0=unlimited).",
)
@click.option(
    "--glob", "-g",
    "glob_pattern",
    type=str,
    default=None,
    help="Glob pattern for batch validation (e.g., 'data/*.yaml').",
)
@click.option(
    "-v", "--verbose",
    "verbosity",
    count=True,
    help="Increase verbosity. -v shows all findings, -vv shows debug info.",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    default=False,
    help="Suppress all output except the exit code. Useful for CI.",
)
@click.pass_context
def validate(
    ctx: click.Context,
    file: Optional[str],
    schema_str: str,
    profile: str,
    format_type: str,
    patch_level: str,
    return_normalized: bool,
    fail_on_warnings: bool,
    max_errors: int,
    glob_pattern: Optional[str],
    verbosity: int,
    quiet: bool,
) -> None:
    """
    Validate a payload against a schema.

    \b
    Arguments:
        FILE  Path to payload file, or '-' for stdin.
              Omit if using --glob for batch validation.

    \b
    Exit Codes:
        0  Valid - Payload passed validation
        1  Invalid - Payload failed validation (or has warnings with --fail-on-warnings)
        2  Error - System error, file not found, etc.

    \b
    Examples:
        # Validate a single file
        greenlang schema validate data.yaml --schema emissions/activity@1.3.0

        # Validate from stdin
        cat data.json | greenlang schema validate - -s test@1.0.0

        # Strict validation with JSON output
        greenlang schema validate data.yaml -s test@1.0.0 --profile strict --format json

        # Batch validation with glob
        greenlang schema validate --glob "data/*.yaml" -s test@1.0.0

        # Show all findings
        greenlang schema validate data.yaml -s test@1.0.0 -v

        # CI mode (quiet, JSON output)
        greenlang schema validate data.yaml -s test@1.0.0 --format json --quiet
    """
    start_time = time.perf_counter()

    # Validate arguments
    if file is None and glob_pattern is None:
        raise click.UsageError(
            "Either FILE argument or --glob option is required.\n"
            "Use '-' as FILE to read from stdin."
        )

    if file is not None and glob_pattern is not None:
        raise click.UsageError(
            "Cannot use both FILE argument and --glob option."
        )

    # Parse schema reference
    try:
        schema_ref = parse_schema_ref(schema_str)
    except click.BadParameter as e:
        if not quiet:
            click.echo(f"Error: {e}", err=True)
        ctx.exit(EXIT_ERROR)
        return

    # Create validation options
    options = create_validation_options(
        profile=profile,
        patch_level=patch_level,
        return_normalized=return_normalized,
        fail_on_warnings=fail_on_warnings,
        max_errors=max_errors,
    )

    # Configure logging based on verbosity
    if verbosity >= 2:
        logging.basicConfig(level=logging.DEBUG)
    elif verbosity >= 1:
        logging.basicConfig(level=logging.INFO)

    try:
        if glob_pattern:
            # Batch validation
            results, valid_count, invalid_count = validate_glob_pattern(
                pattern=glob_pattern,
                schema_ref=schema_ref,
                options=options,
            )

            # Output results
            if not quiet:
                # Print summary header
                total = valid_count + invalid_count
                click.echo(f"\nBatch Validation: {valid_count}/{total} valid\n")

                # Print individual results
                for file_path, report in results:
                    if format_type == "json":
                        # For JSON, output each result as a JSON line
                        output = format_output(
                            report, format_type, verbosity, quiet, file_path
                        )
                        click.echo(output)
                    else:
                        # For other formats, show status and details
                        status = "VALID" if report.valid else "INVALID"
                        click.echo(f"[{status}] {file_path}")
                        if not report.valid or verbosity > 0:
                            output = format_output(
                                report, format_type, verbosity, quiet, file_path
                            )
                            click.echo(output)
                            click.echo()

            # Determine exit code
            if invalid_count > 0:
                ctx.exit(EXIT_INVALID)
            elif fail_on_warnings:
                # Check for warnings in any result
                has_warnings = any(
                    r.summary.warning_count > 0 for _, r in results
                )
                if has_warnings:
                    ctx.exit(EXIT_INVALID)

            ctx.exit(EXIT_VALID)

        else:
            # Single file validation
            try:
                payload_content, source_name = read_payload(file, sys.stdin)
            except click.BadParameter as e:
                if not quiet:
                    click.echo(f"Error: {e}", err=True)
                ctx.exit(EXIT_ERROR)
                return

            # Validate
            report = validate_single_file(
                payload_content=payload_content,
                source_name=source_name,
                schema_ref=schema_ref,
                options=options,
            )

            # Output result
            if not quiet:
                output = format_output(
                    report, format_type, verbosity, quiet, source_name
                )
                click.echo(output)

            # Determine exit code
            if not report.valid:
                ctx.exit(EXIT_INVALID)
            elif fail_on_warnings and report.summary.warning_count > 0:
                ctx.exit(EXIT_INVALID)
            else:
                ctx.exit(EXIT_VALID)

    except Exception as e:
        logger.exception("Validation failed")
        if not quiet:
            click.echo(f"Error: {e}", err=True)
        ctx.exit(EXIT_ERROR)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.debug(f"Total CLI time: {elapsed_ms:.2f}ms")
