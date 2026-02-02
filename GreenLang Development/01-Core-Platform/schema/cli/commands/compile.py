# -*- coding: utf-8 -*-
"""
Compile Command for GL-FOUND-X-002 (GreenLang Schema Compiler & Validator).

This module implements the 'greenlang schema compile' command for compiling
schemas to optimized Intermediate Representation (IR).

Features:
    - Compile schemas from file or URI
    - Output to file or stdout
    - JSON output format
    - Schema validation during compilation
    - Compilation warnings and errors

Exit Codes:
    0 - Success (schema compiled successfully)
    2 - Error (compilation failed)

Example:
    $ greenlang schema compile schema.yaml --out ir.json
    $ greenlang schema compile gl://schemas/test@1.0.0 --format json

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 5.1
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import click

from greenlang.schema.compiler.compiler import SchemaCompiler
from greenlang.schema.compiler.ir import CompilationResult, SchemaIR
from greenlang.schema.compiler.parser import parse_payload, ParseError
from greenlang.schema.models.schema_ref import SchemaRef


logger = logging.getLogger(__name__)

# Exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 2


def read_schema_source(schema_arg: str) -> tuple[Dict[str, Any], str, str]:
    """
    Read schema source from file or URI.

    Args:
        schema_arg: Schema file path or URI.

    Returns:
        Tuple of (schema_dict, schema_id, version).

    Raises:
        click.BadParameter: If schema cannot be read.
    """
    # Check if it's a GL URI
    if schema_arg.startswith("gl://"):
        try:
            schema_ref = SchemaRef.from_uri(schema_arg)
            # TODO: Resolve from registry
            raise click.BadParameter(
                f"Schema registry resolution not yet implemented for: {schema_arg}"
            )
        except ValueError as e:
            raise click.BadParameter(str(e))

    # Check if it's a file path
    file_path = Path(schema_arg)
    if not file_path.exists():
        raise click.BadParameter(f"Schema file not found: {schema_arg}")
    if not file_path.is_file():
        raise click.BadParameter(f"Not a file: {schema_arg}")

    try:
        content = file_path.read_text(encoding="utf-8")
        result = parse_payload(content)
        schema_dict = result.data

        # Extract schema_id and version from schema or filename
        schema_id = schema_dict.get("$id", file_path.stem)
        version = schema_dict.get("version", "1.0.0")

        return schema_dict, schema_id, version
    except ParseError as e:
        raise click.BadParameter(f"Failed to parse schema: {e.message}")
    except IOError as e:
        raise click.BadParameter(f"Cannot read schema file: {e}")


def format_ir_json(ir: SchemaIR, pretty: bool = True) -> str:
    """
    Format SchemaIR as JSON.

    Args:
        ir: Compiled schema IR.
        pretty: Whether to format with indentation.

    Returns:
        JSON string representation.
    """
    ir_dict = {
        "schema_id": ir.schema_id,
        "version": ir.version,
        "schema_hash": ir.schema_hash,
        "compiled_at": ir.compiled_at.isoformat(),
        "compiler_version": ir.compiler_version,
        "properties": {
            path: {
                "path": prop.path,
                "type": prop.type,
                "required": prop.required,
                "has_default": prop.has_default,
                "default_value": prop.default_value,
                "gl_extensions": prop.gl_extensions,
            }
            for path, prop in ir.properties.items()
        },
        "required_paths": list(ir.required_paths),
        "numeric_constraints": {
            path: {
                "path": c.path,
                "minimum": c.minimum,
                "maximum": c.maximum,
                "exclusive_minimum": c.exclusive_minimum,
                "exclusive_maximum": c.exclusive_maximum,
                "multiple_of": c.multiple_of,
            }
            for path, c in ir.numeric_constraints.items()
        },
        "string_constraints": {
            path: {
                "path": c.path,
                "min_length": c.min_length,
                "max_length": c.max_length,
                "pattern": c.pattern,
                "format": c.format,
            }
            for path, c in ir.string_constraints.items()
        },
        "array_constraints": {
            path: {
                "path": c.path,
                "min_items": c.min_items,
                "max_items": c.max_items,
                "unique_items": c.unique_items,
            }
            for path, c in ir.array_constraints.items()
        },
        "patterns": {
            path: {
                "pattern": p.pattern,
                "complexity_score": p.complexity_score,
                "is_safe": p.is_safe,
                "is_re2_compatible": p.is_re2_compatible,
                "vulnerability_type": p.vulnerability_type,
            }
            for path, p in ir.patterns.items()
        },
        "unit_specs": {
            path: {
                "path": u.path,
                "dimension": u.dimension,
                "canonical": u.canonical,
                "allowed": u.allowed,
            }
            for path, u in ir.unit_specs.items()
        },
        "rule_bindings": [
            {
                "rule_id": r.rule_id,
                "rule_pack": r.rule_pack,
                "severity": r.severity,
                "applies_to": r.applies_to,
                "when": r.when,
                "check": r.check,
                "message": r.message,
            }
            for r in ir.rule_bindings
        ],
        "deprecated_fields": ir.deprecated_fields,
        "renamed_fields": ir.renamed_fields,
        "enums": ir.enums,
    }

    if pretty:
        return json.dumps(ir_dict, indent=2, default=str)
    else:
        return json.dumps(ir_dict, default=str)


def format_compilation_result(result: CompilationResult, format_type: str) -> str:
    """
    Format compilation result for output.

    Args:
        result: Compilation result.
        format_type: Output format (json).

    Returns:
        Formatted output string.
    """
    if not result.success:
        error_output = {
            "success": False,
            "errors": result.errors,
            "warnings": result.warnings,
            "compile_time_ms": result.compile_time_ms,
        }
        return json.dumps(error_output, indent=2)

    if format_type == "json":
        return format_ir_json(result.ir, pretty=True)

    # Default to JSON
    return format_ir_json(result.ir, pretty=True)


@click.command("compile")
@click.argument(
    "schema",
    type=str,
    required=True,
)
@click.option(
    "--out", "-o",
    "output_file",
    type=click.Path(),
    default=None,
    help="Output file path. If not specified, outputs to stdout.",
)
@click.option(
    "--format", "-f",
    "format_type",
    type=click.Choice(["json"], case_sensitive=False),
    default="json",
    help="Output format for the compiled IR.",
)
@click.option(
    "--schema-id",
    type=str,
    default=None,
    help="Override schema ID (default: extracted from schema or filename).",
)
@click.option(
    "--version",
    "schema_version",
    type=str,
    default=None,
    help="Override schema version (default: extracted from schema or '1.0.0').",
)
@click.option(
    "-v", "--verbose",
    "verbosity",
    count=True,
    help="Increase verbosity. -v shows warnings, -vv shows debug info.",
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    default=False,
    help="Suppress all output except errors.",
)
@click.pass_context
def compile_schema(
    ctx: click.Context,
    schema: str,
    output_file: Optional[str],
    format_type: str,
    schema_id: Optional[str],
    schema_version: Optional[str],
    verbosity: int,
    quiet: bool,
) -> None:
    """
    Compile a schema to Intermediate Representation (IR).

    \b
    Arguments:
        SCHEMA  Path to schema file or schema URI.

    \b
    Exit Codes:
        0  Success - Schema compiled successfully
        2  Error - Compilation failed

    \b
    Examples:
        # Compile a local schema file
        greenlang schema compile schema.yaml --out ir.json

        # Compile and output to stdout
        greenlang schema compile schema.yaml

        # Compile with custom ID and version
        greenlang schema compile schema.yaml --schema-id test/myschema --version 2.0.0

        # Compile from registry URI (not yet implemented)
        greenlang schema compile gl://schemas/emissions/activity@1.3.0
    """
    start_time = time.perf_counter()

    # Configure logging
    if verbosity >= 2:
        logging.basicConfig(level=logging.DEBUG)
    elif verbosity >= 1:
        logging.basicConfig(level=logging.INFO)

    try:
        # Read schema source
        schema_dict, default_id, default_version = read_schema_source(schema)

        # Apply overrides
        final_schema_id = schema_id or default_id
        final_version = schema_version or default_version

        # Compile schema
        compiler = SchemaCompiler()
        result = compiler.compile(
            schema_source=schema_dict,
            schema_id=final_schema_id,
            version=final_version,
        )

        # Handle warnings
        if result.warnings and not quiet:
            for warning in result.warnings:
                click.echo(f"Warning: {warning}", err=True)

        # Handle errors
        if not result.success:
            if not quiet:
                click.echo("Compilation failed:", err=True)
                for error in result.errors:
                    click.echo(f"  Error: {error}", err=True)
            ctx.exit(EXIT_ERROR)
            return

        # Format output
        output = format_compilation_result(result, format_type)

        # Write output
        if output_file:
            output_path = Path(output_file)
            output_path.write_text(output, encoding="utf-8")
            if not quiet:
                click.echo(f"Compiled schema written to: {output_file}")
        else:
            click.echo(output)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if not quiet and verbosity >= 1:
            click.echo(f"\nCompilation completed in {elapsed_ms:.2f}ms", err=True)
            click.echo(f"Schema hash: {result.ir.schema_hash}", err=True)
            click.echo(f"Properties: {len(result.ir.properties)}", err=True)

        ctx.exit(EXIT_SUCCESS)

    except click.BadParameter as e:
        if not quiet:
            click.echo(f"Error: {e}", err=True)
        ctx.exit(EXIT_ERROR)
    except Exception as e:
        logger.exception("Compilation failed")
        if not quiet:
            click.echo(f"Error: {e}", err=True)
        ctx.exit(EXIT_ERROR)
