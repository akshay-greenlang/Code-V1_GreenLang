# -*- coding: utf-8 -*-
"""
CLI Main Entry Point for GL-FOUND-X-002 (GreenLang Schema Compiler & Validator).

This module implements the main CLI entry point using the Click framework.
It provides a hierarchical command structure:

    greenlang schema validate  - Validate payloads against schemas
    greenlang schema compile   - Compile schemas to IR
    greenlang schema lint      - Lint schemas for best practices
    greenlang validate         - Alias for 'greenlang schema validate'

Exit Codes:
    0 - Success (valid payload or successful compilation)
    1 - Validation failed (invalid payload)
    2 - Error (system error, missing file, etc.)

Example:
    $ greenlang schema validate data.yaml --schema emissions/activity@1.3.0
    $ greenlang schema compile schema.yaml --out ir.json
    $ greenlang validate data.yaml -s emissions/activity@1.3.0 --format json

Author: GreenLang Framework Team
Version: 1.0.0
GL-FOUND-X-002: Schema Compiler & Validator - Task 5.1
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import click

from greenlang.schema import __version__ as schema_version


# Configure logging
logger = logging.getLogger(__name__)

# Exit codes
EXIT_SUCCESS = 0
EXIT_INVALID = 1
EXIT_ERROR = 2


def configure_logging(verbosity: int, quiet: bool) -> None:
    """
    Configure logging based on verbosity flags.

    Args:
        verbosity: Verbosity level (0=normal, 1=verbose, 2=debug).
        quiet: If True, suppress all output except errors.
    """
    if quiet:
        level = logging.ERROR
    elif verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class Context:
    """
    CLI context object passed between commands.

    Attributes:
        verbosity: Verbosity level (0, 1, or 2).
        quiet: Whether to suppress output.
        format: Output format (pretty, text, table, json, sarif).
    """

    def __init__(self) -> None:
        """Initialize CLI context."""
        self.verbosity: int = 0
        self.quiet: bool = False
        self.format: str = "pretty"


pass_context = click.make_pass_decorator(Context, ensure=True)


@click.group()
@click.version_option(version=schema_version, prog_name="greenlang-schema")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """
    GreenLang Schema Compiler & Validator (GL-FOUND-X-002).

    A comprehensive CLI for validating payloads against GreenLang schemas,
    compiling schemas to optimized IR, and linting schemas for best practices.

    \b
    Commands:
        schema validate  Validate payloads against schemas
        schema compile   Compile schemas to IR
        schema lint      Lint schemas for best practices
        validate         Alias for 'schema validate'

    \b
    Exit Codes:
        0  Success (valid payload or successful operation)
        1  Validation failed (invalid payload)
        2  Error (system error, missing file, etc.)

    \b
    Examples:
        # Validate a file
        greenlang schema validate data.yaml --schema emissions/activity@1.3.0

        # Validate with strict profile
        greenlang schema validate data.yaml -s emissions/activity@1.3.0 --profile strict

        # Output as JSON for CI
        greenlang validate data.yaml -s emissions/activity@1.3.0 --format json --quiet

        # Compile a schema
        greenlang schema compile schema.yaml --out ir.json
    """
    ctx.ensure_object(Context)


@cli.group()
@click.pass_context
def schema(ctx: click.Context) -> None:
    """
    Schema compiler and validator commands.

    \b
    Subcommands:
        validate  Validate payloads against schemas
        compile   Compile schemas to IR
        lint      Lint schemas for best practices
    """
    pass


# Import and register commands
from greenlang.schema.cli.commands.validate import validate
from greenlang.schema.cli.commands.compile import compile_schema
from greenlang.schema.cli.commands.lint import lint
from greenlang.schema.cli.commands.migrate import migrate

# Register commands under 'schema' group
schema.add_command(validate)
schema.add_command(compile_schema, name="compile")
schema.add_command(lint)
schema.add_command(migrate)

# Register 'validate' as alias at top level
cli.add_command(validate, name="validate")


def main() -> None:
    """
    Main entry point for the CLI.

    This function is called when running 'greenlang' from the command line.
    It invokes the Click CLI and handles exit codes appropriately.
    """
    try:
        cli(standalone_mode=False)
        sys.exit(EXIT_SUCCESS)
    except click.ClickException as e:
        e.show()
        sys.exit(EXIT_ERROR)
    except SystemExit as e:
        # Click may raise SystemExit, preserve exit code
        sys.exit(e.code if e.code is not None else EXIT_SUCCESS)
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(EXIT_ERROR)
    except Exception as e:
        logger.exception("Unexpected error")
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_ERROR)


if __name__ == "__main__":
    main()
