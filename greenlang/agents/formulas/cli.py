"""
Formula Management CLI

Command-line interface for formula versioning operations.

Commands:
    greenlang formula list                  - List all formulas
    greenlang formula show <code>           - Show formula details
    greenlang formula create <code>         - Create new formula
    greenlang formula activate <code> -v N  - Activate version
    greenlang formula rollback <code> -v N  - Rollback to version
    greenlang formula compare <code> -v A B - Compare versions
    greenlang formula migrate <file>        - Migrate from file
    greenlang formula execute <code>        - Execute formula
"""

import click
import json
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import date
import logging

from greenlang.formulas import FormulaManager
from greenlang.formulas.migration import FormulaMigrator
from greenlang.formulas.models import FormulaCategory

logger = logging.getLogger(__name__)


# Default database path
DEFAULT_DB_PATH = str(Path.home() / ".greenlang" / "formulas.db")


@click.group()
@click.option(
    '--db',
    default=DEFAULT_DB_PATH,
    help='Path to formula database',
    envvar='GREENLANG_FORMULA_DB',
)
@click.pass_context
def formula(ctx, db):
    """Formula versioning management commands."""
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Initialize database directory
    db_path = Path(db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Store manager in context
    ctx.obj['manager'] = FormulaManager(str(db_path))
    ctx.obj['db_path'] = str(db_path)


@formula.command()
@click.option('--category', help='Filter by category')
@click.option('--limit', default=50, help='Maximum results')
@click.pass_context
def list(ctx, category, limit):
    """List all formulas."""
    manager = ctx.obj['manager']

    try:
        # Parse category if provided
        cat_enum = FormulaCategory(category) if category else None

        formulas = manager.list_formulas(category=cat_enum)

        if not formulas:
            click.echo("No formulas found.")
            return

        # Display formulas in table format
        click.echo(f"\n{'Code':<20} {'Name':<50} {'Category':<15}")
        click.echo("-" * 85)

        for formula in formulas[:limit]:
            click.echo(
                f"{formula.formula_code:<20} "
                f"{formula.formula_name:<50} "
                f"{formula.category:<15}"
            )

        click.echo(f"\nTotal: {len(formulas)} formulas")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@formula.command()
@click.argument('formula_code')
@click.option('--version', '-v', type=int, help='Specific version (default: active)')
@click.pass_context
def show(ctx, formula_code, version):
    """Show formula details."""
    manager = ctx.obj['manager']

    try:
        # Get formula metadata
        formula = manager.get_formula(formula_code)

        if not formula:
            click.echo(f"Formula {formula_code} not found.", err=True)
            sys.exit(1)

        # Get version
        if version:
            formula_version = manager.get_version(formula_code, version)
        else:
            formula_version = manager.get_active_formula(formula_code)

        if not formula_version:
            click.echo(f"Version not found for {formula_code}.", err=True)
            sys.exit(1)

        # Display details
        click.echo(f"\n{formula.formula_name}")
        click.echo("=" * 80)
        click.echo(f"Code:              {formula.formula_code}")
        click.echo(f"Category:          {formula.category}")
        click.echo(f"Version:           {formula_version.version_number}")
        click.echo(f"Status:            {formula_version.version_status}")
        click.echo(f"\nExpression:        {formula_version.formula_expression}")
        click.echo(f"Calculation Type:  {formula_version.calculation_type}")
        click.echo(f"Required Inputs:   {', '.join(formula_version.required_inputs)}")
        click.echo(f"Output Unit:       {formula_version.output_unit}")
        click.echo(f"\nDeterministic:     {formula_version.deterministic}")
        click.echo(f"Zero Hallucination: {formula_version.zero_hallucination}")

        if formula_version.change_notes:
            click.echo(f"\nChange Notes:      {formula_version.change_notes}")

        if formula_version.example_calculation:
            click.echo(f"\nExample:           {formula_version.example_calculation}")

        click.echo(f"\nExecutions:        {formula_version.execution_count}")
        if formula_version.avg_execution_time_ms:
            click.echo(
                f"Avg Exec Time:     {formula_version.avg_execution_time_ms:.2f} ms"
            )

        click.echo()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@formula.command()
@click.argument('formula_code')
@click.pass_context
def versions(ctx, formula_code):
    """List all versions of a formula."""
    manager = ctx.obj['manager']

    try:
        versions = manager.list_versions(formula_code)

        if not versions:
            click.echo(f"No versions found for {formula_code}.", err=True)
            sys.exit(1)

        # Display versions
        click.echo(f"\nVersions of {formula_code}")
        click.echo("=" * 80)
        click.echo(f"{'Ver':<5} {'Status':<12} {'Created':<20} {'Executions':<12} {'Avg Time':<12}")
        click.echo("-" * 80)

        for v in versions:
            created = v.created_at.strftime("%Y-%m-%d %H:%M")
            exec_count = v.execution_count or 0
            avg_time = f"{v.avg_execution_time_ms:.2f}ms" if v.avg_execution_time_ms else "-"

            click.echo(
                f"{v.version_number:<5} "
                f"{v.version_status:<12} "
                f"{created:<20} "
                f"{exec_count:<12} "
                f"{avg_time:<12}"
            )

        click.echo()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@formula.command()
@click.argument('formula_code')
@click.option('--version', '-v', type=int, required=True, help='Version to activate')
@click.option('--from-date', help='Effective from date (YYYY-MM-DD)')
@click.pass_context
def activate(ctx, formula_code, version, from_date):
    """Activate a formula version."""
    manager = ctx.obj['manager']

    try:
        # Parse date if provided
        effective_from = date.fromisoformat(from_date) if from_date else None

        manager.activate_version(formula_code, version, effective_from)

        click.echo(
            f"✓ Activated {formula_code} version {version}"
            + (f" effective from {effective_from}" if effective_from else "")
        )

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@formula.command()
@click.argument('formula_code')
@click.option('--to-version', '-v', type=int, required=True, help='Version to rollback to')
@click.pass_context
def rollback(ctx, formula_code, to_version):
    """Rollback formula to a previous version."""
    manager = ctx.obj['manager']

    try:
        # Confirm with user
        if not click.confirm(
            f"Rollback {formula_code} to version {to_version}?"
        ):
            click.echo("Cancelled.")
            return

        new_version_id = manager.rollback_to_version(formula_code, to_version)

        click.echo(f"✓ Rolled back {formula_code} to version {to_version}")
        click.echo(f"  New version created with id={new_version_id}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@formula.command()
@click.argument('formula_code')
@click.option('--versions', '-v', required=True, help='Versions to compare (e.g., "1,2")')
@click.pass_context
def compare(ctx, formula_code, versions):
    """Compare two formula versions."""
    manager = ctx.obj['manager']

    try:
        # Parse versions
        version_a, version_b = map(int, versions.split(','))

        comparison = manager.compare_versions(formula_code, version_a, version_b)

        # Display comparison
        click.echo(f"\nComparison: {formula_code} v{version_a} vs v{version_b}")
        click.echo("=" * 80)

        if comparison.expression_changed:
            click.echo("✗ Expression changed:")
            click.echo(f"  {comparison.expression_diff}")

        if comparison.inputs_changed:
            click.echo("✗ Inputs changed:")
            if comparison.added_inputs:
                click.echo(f"  Added: {', '.join(comparison.added_inputs)}")
            if comparison.removed_inputs:
                click.echo(f"  Removed: {', '.join(comparison.removed_inputs)}")

        if comparison.output_unit_changed:
            click.echo("✗ Output unit changed")

        if comparison.validation_rules_changed:
            click.echo("✗ Validation rules changed")

        if not any([
            comparison.expression_changed,
            comparison.inputs_changed,
            comparison.output_unit_changed,
            comparison.validation_rules_changed,
        ]):
            click.echo("✓ No significant changes detected")

        if comparison.avg_time_diff_ms is not None:
            click.echo(f"\nPerformance: {comparison.avg_time_diff_pct:+.1f}% "
                      f"({comparison.avg_time_diff_ms:+.2f}ms)")

        click.echo()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@formula.command()
@click.argument('source_file')
@click.option('--type', type=click.Choice(['yaml', 'python']), required=True)
@click.option('--auto-activate/--no-auto-activate', default=True)
@click.pass_context
def migrate(ctx, source_file, type, auto_activate):
    """Migrate formulas from external file."""
    manager = ctx.obj['manager']

    try:
        migrator = FormulaMigrator(manager)

        click.echo(f"Migrating formulas from {source_file}...")

        if type == 'yaml':
            stats = migrator.migrate_from_yaml(source_file, auto_activate=auto_activate)
        elif type == 'python':
            stats = migrator.migrate_from_python(source_file, auto_activate=auto_activate)
        else:
            click.echo(f"Unsupported type: {type}", err=True)
            sys.exit(1)

        # Display results
        click.echo("\nMigration complete:")
        click.echo(f"  Total:   {stats['total']}")
        click.echo(f"  Success: {stats['success']}")
        click.echo(f"  Failed:  {stats['failed']}")
        click.echo(f"  Skipped: {stats['skipped']}")

        summary = migrator.get_migration_summary()
        click.echo(f"  Success Rate: {summary['success_rate']:.1f}%")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@formula.command()
@click.argument('formula_code')
@click.option('--input', '-i', 'input_json', required=True, help='Input JSON')
@click.option('--version', '-v', type=int, help='Specific version')
@click.pass_context
def execute(ctx, formula_code, input_json, version):
    """Execute a formula with input data."""
    manager = ctx.obj['manager']

    try:
        # Parse input JSON
        input_data = json.loads(input_json)

        # Execute formula
        result = manager.execute_formula_full(
            formula_code=formula_code,
            input_data=input_data,
            version_number=version,
        )

        # Display result
        click.echo(f"\nFormula: {formula_code}")
        if version:
            click.echo(f"Version: {version}")
        click.echo(f"Status:  {result.execution_status}")
        click.echo(f"Output:  {result.output_value}")
        click.echo(f"Time:    {result.execution_time_ms:.2f}ms")
        click.echo(f"\nInput Hash:  {result.input_hash}")
        click.echo(f"Output Hash: {result.output_hash}")

    except json.JSONDecodeError as e:
        click.echo(f"Invalid JSON: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# Main entry point for CLI
if __name__ == '__main__':
    formula()
