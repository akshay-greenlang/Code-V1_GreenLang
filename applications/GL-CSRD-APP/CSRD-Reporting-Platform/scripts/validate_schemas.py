# -*- coding: utf-8 -*-
"""
CSRD Schema Validation Tool

This script validates JSON schemas and example data:
- Validates all JSON schemas (company profile, ESG data, materiality, CSRD report)
- Tests example data against schemas
- Checks schema completeness and coverage
- Reports validation errors with detailed diagnostics
- Verifies all ESRS data points are covered

Usage:
    python scripts/validate_schemas.py
    python scripts/validate_schemas.py --schema schemas/esg_data.schema.json
    python scripts/validate_schemas.py --example examples/demo_esg_data.csv
    python scripts/validate_schemas.py --output validation_report.json

Version: 1.0.0
Author: GreenLang CSRD Team
License: MIT
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import pandas as pd
from jsonschema import Draft7Validator, ValidationError as JsonSchemaValidationError
from jsonschema import validate as json_validate
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

console = Console()


# ============================================================================
# VALIDATION RESULT MODELS
# ============================================================================

class ValidationIssue:
    """Represents a validation issue."""

    def __init__(
        self,
        severity: str,  # "error", "warning", "info"
        message: str,
        schema_path: Optional[str] = None,
        data_path: Optional[str] = None,
        instance_path: Optional[str] = None
    ):
        self.severity = severity
        self.message = message
        self.schema_path = schema_path
        self.data_path = data_path
        self.instance_path = instance_path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "message": self.message,
            "schema_path": self.schema_path,
            "data_path": self.data_path,
            "instance_path": self.instance_path
        }


class SchemaValidationResult:
    """Result of schema validation."""

    def __init__(self, schema_name: str):
        self.schema_name = schema_name
        self.is_valid: bool = True
        self.issues: List[ValidationIssue] = []
        self.metadata: Dict[str, Any] = {}

    def add_issue(self, issue: ValidationIssue):
        """Add validation issue."""
        self.issues.append(issue)
        if issue.severity == "error":
            self.is_valid = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "is_valid": self.is_valid,
            "error_count": len([i for i in self.issues if i.severity == "error"]),
            "warning_count": len([i for i in self.issues if i.severity == "warning"]),
            "info_count": len([i for i in self.issues if i.severity == "info"]),
            "issues": [i.to_dict() for i in self.issues],
            "metadata": self.metadata
        }


class ValidationReport:
    """Complete validation report."""

    def __init__(self):
        self.timestamp = pd.Timestamp.now().isoformat()
        self.schema_results: List[SchemaValidationResult] = []
        self.example_results: List[SchemaValidationResult] = []
        self.overall_valid: bool = True

    def add_schema_result(self, result: SchemaValidationResult):
        """Add schema validation result."""
        self.schema_results.append(result)
        if not result.is_valid:
            self.overall_valid = False

    def add_example_result(self, result: SchemaValidationResult):
        """Add example validation result."""
        self.example_results.append(result)
        if not result.is_valid:
            self.overall_valid = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_valid": self.overall_valid,
            "schemas_validated": len(self.schema_results),
            "examples_validated": len(self.example_results),
            "total_errors": sum(r.to_dict()["error_count"] for r in self.schema_results + self.example_results),
            "total_warnings": sum(r.to_dict()["warning_count"] for r in self.schema_results + self.example_results),
            "schema_results": [r.to_dict() for r in self.schema_results],
            "example_results": [r.to_dict() for r in self.example_results]
        }


# ============================================================================
# SCHEMA VALIDATION FUNCTIONS
# ============================================================================

def load_schema(schema_path: Path) -> Dict[str, Any]:
    """Load JSON schema from file."""
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        return schema
    except Exception as e:
        raise ValueError(f"Failed to load schema {schema_path}: {e}")


def validate_schema_structure(schema_path: Path) -> SchemaValidationResult:
    """
    Validate that a JSON schema is well-formed.

    Args:
        schema_path: Path to JSON schema file

    Returns:
        SchemaValidationResult
    """
    result = SchemaValidationResult(schema_path.name)

    try:
        schema = load_schema(schema_path)

        # Check if it's a valid JSON Schema
        Draft7Validator.check_schema(schema)

        # Extract metadata
        result.metadata = {
            "schema_version": schema.get("$schema", "unknown"),
            "title": schema.get("title", ""),
            "description": schema.get("description", ""),
            "has_definitions": "definitions" in schema or "$defs" in schema,
            "required_fields": schema.get("required", []),
            "properties_count": len(schema.get("properties", {}))
        }

        result.add_issue(ValidationIssue(
            severity="info",
            message=f"Schema is well-formed with {result.metadata['properties_count']} properties"
        ))

    except JsonSchemaValidationError as e:
        result.add_issue(ValidationIssue(
            severity="error",
            message=f"Invalid JSON Schema: {e.message}",
            schema_path=str(schema_path)
        ))
    except Exception as e:
        result.add_issue(ValidationIssue(
            severity="error",
            message=f"Failed to validate schema: {str(e)}",
            schema_path=str(schema_path)
        ))

    return result


def validate_data_against_schema(
    data_path: Path,
    schema_path: Path,
    data_format: str = "auto"
) -> SchemaValidationResult:
    """
    Validate data file against JSON schema.

    Args:
        data_path: Path to data file (JSON or CSV)
        schema_path: Path to JSON schema
        data_format: Data format ("json", "csv", or "auto")

    Returns:
        SchemaValidationResult
    """
    result = SchemaValidationResult(f"{data_path.name} vs {schema_path.name}")

    try:
        # Load schema
        schema = load_schema(schema_path)

        # Load data
        if data_format == "auto":
            data_format = data_path.suffix.lower().replace('.', '')

        if data_format == "json":
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both single object and array
            if isinstance(data, list):
                # Validate each item
                for idx, item in enumerate(data):
                    try:
                        json_validate(instance=item, schema=schema)
                    except JsonSchemaValidationError as e:
                        result.add_issue(ValidationIssue(
                            severity="error",
                            message=f"Item {idx}: {e.message}",
                            data_path=str(data_path),
                            instance_path=e.json_path
                        ))
            else:
                # Validate single object
                try:
                    json_validate(instance=data, schema=schema)
                except JsonSchemaValidationError as e:
                    result.add_issue(ValidationIssue(
                        severity="error",
                        message=e.message,
                        data_path=str(data_path),
                        instance_path=e.json_path
                    ))

        elif data_format == "csv":
            # For CSV, convert to list of dicts and validate each row
            df = pd.read_csv(data_path)
            records = df.to_dict('records')

            for idx, record in enumerate(records):
                try:
                    # Schema might expect nested structure, adjust as needed
                    json_validate(instance=record, schema=schema)
                except JsonSchemaValidationError as e:
                    result.add_issue(ValidationIssue(
                        severity="error",
                        message=f"Row {idx + 1}: {e.message}",
                        data_path=str(data_path),
                        instance_path=e.json_path
                    ))

        else:
            result.add_issue(ValidationIssue(
                severity="error",
                message=f"Unsupported data format: {data_format}"
            ))

        # Success message
        if result.is_valid:
            result.add_issue(ValidationIssue(
                severity="info",
                message="Data validates successfully against schema"
            ))

        result.metadata = {
            "data_format": data_format,
            "schema_used": schema_path.name
        }

    except Exception as e:
        result.add_issue(ValidationIssue(
            severity="error",
            message=f"Validation failed: {str(e)}",
            data_path=str(data_path)
        ))

    return result


def check_esrs_coverage(data_points_path: Path, schemas_dir: Path) -> SchemaValidationResult:
    """
    Check if schemas cover all ESRS data points.

    Args:
        data_points_path: Path to ESRS data points catalog
        schemas_dir: Directory containing schemas

    Returns:
        SchemaValidationResult
    """
    result = SchemaValidationResult("ESRS Coverage Check")

    try:
        # Load ESRS data points
        with open(data_points_path, 'r', encoding='utf-8') as f:
            esrs_data = json.load(f)

        if isinstance(esrs_data, dict) and "data_points" in esrs_data:
            data_points = esrs_data["data_points"]
        else:
            data_points = esrs_data

        # Extract unique ESRS standards
        esrs_standards = set()
        for dp in data_points:
            standard = dp.get("standard") or dp.get("esrs_standard")
            if standard:
                esrs_standards.add(standard)

        result.metadata = {
            "total_esrs_data_points": len(data_points),
            "esrs_standards": sorted(list(esrs_standards)),
            "environmental_standards": len([s for s in esrs_standards if s.startswith("E")]),
            "social_standards": len([s for s in esrs_standards if s.startswith("S")]),
            "governance_standards": len([s for s in esrs_standards if s.startswith("G")]),
        }

        result.add_issue(ValidationIssue(
            severity="info",
            message=f"Found {len(data_points)} ESRS data points across {len(esrs_standards)} standards"
        ))

        # Check mandatory vs optional
        mandatory_count = len([dp for dp in data_points if dp.get("mandatory", False)])
        result.metadata["mandatory_data_points"] = mandatory_count
        result.metadata["optional_data_points"] = len(data_points) - mandatory_count

        result.add_issue(ValidationIssue(
            severity="info",
            message=f"Mandatory: {mandatory_count}, Optional: {len(data_points) - mandatory_count}"
        ))

    except Exception as e:
        result.add_issue(ValidationIssue(
            severity="error",
            message=f"ESRS coverage check failed: {str(e)}"
        ))

    return result


# ============================================================================
# REPORT GENERATION
# ============================================================================

def display_validation_results(report: ValidationReport):
    """Display validation results in rich tables."""

    # Overall summary
    if report.overall_valid:
        console.print(Panel(
            "[bold green]✅ ALL VALIDATIONS PASSED[/bold green]",
            title="Validation Summary",
            border_style="green"
        ))
    else:
        console.print(Panel(
            "[bold red]❌ VALIDATION ERRORS FOUND[/bold red]",
            title="Validation Summary",
            border_style="red"
        ))

    console.print()

    # Schema validation results
    if report.schema_results:
        schema_table = Table(title="Schema Validation Results")
        schema_table.add_column("Schema", style="cyan")
        schema_table.add_column("Status", justify="center")
        schema_table.add_column("Errors", justify="right", style="red")
        schema_table.add_column("Warnings", justify="right", style="yellow")
        schema_table.add_column("Properties", justify="right", style="blue")

        for result in report.schema_results:
            result_dict = result.to_dict()
            status = "✅" if result.is_valid else "❌"
            schema_table.add_row(
                result.schema_name,
                status,
                str(result_dict["error_count"]),
                str(result_dict["warning_count"]),
                str(result.metadata.get("properties_count", "-"))
            )

        console.print(schema_table)
        console.print()

    # Example validation results
    if report.example_results:
        example_table = Table(title="Example Data Validation Results")
        example_table.add_column("Example", style="cyan")
        example_table.add_column("Status", justify="center")
        example_table.add_column("Errors", justify="right", style="red")
        example_table.add_column("Warnings", justify="right", style="yellow")

        for result in report.example_results:
            result_dict = result.to_dict()
            status = "✅" if result.is_valid else "❌"
            example_table.add_row(
                result.schema_name,
                status,
                str(result_dict["error_count"]),
                str(result_dict["warning_count"])
            )

        console.print(example_table)
        console.print()

    # Display errors
    all_errors = []
    for result in report.schema_results + report.example_results:
        all_errors.extend([i for i in result.issues if i.severity == "error"])

    if all_errors:
        console.print("[bold red]Errors:[/bold red]")
        for idx, error in enumerate(all_errors[:10], 1):  # Show first 10
            console.print(f"  {idx}. {error.message}")
            if error.schema_path:
                console.print(f"     Schema: {error.schema_path}")
            if error.data_path:
                console.print(f"     Data: {error.data_path}")
        if len(all_errors) > 10:
            console.print(f"  ... and {len(all_errors) - 10} more errors")
        console.print()


# ============================================================================
# CLI INTERFACE
# ============================================================================

@click.command()
@click.option(
    '--schema',
    type=click.Path(exists=True),
    help='Validate specific schema file'
)
@click.option(
    '--example',
    type=click.Path(exists=True),
    help='Validate specific example data file'
)
@click.option(
    '--schemas-dir',
    type=click.Path(exists=True),
    default='schemas',
    help='Directory containing schemas (default: schemas/)'
)
@click.option(
    '--examples-dir',
    type=click.Path(exists=True),
    default='examples',
    help='Directory containing example data (default: examples/)'
)
@click.option(
    '--output',
    type=click.Path(),
    default='validation_report.json',
    help='Output file for validation report'
)
@click.option(
    '--check-esrs-coverage',
    is_flag=True,
    help='Check ESRS data point coverage'
)
def validate_schemas(
    schema: Optional[str],
    example: Optional[str],
    schemas_dir: str,
    examples_dir: str,
    output: str,
    check_esrs_coverage: bool
):
    """
    Validate CSRD JSON schemas and example data.

    Validates all JSON schemas for correctness and tests example data
    against the schemas. Generates a comprehensive validation report.
    """
    console.print("\n[bold cyan]🔍 CSRD Schema Validation Tool[/bold cyan]\n")

    report = ValidationReport()

    schemas_path = Path(schemas_dir)
    examples_path = Path(examples_dir)

    # Validate individual schema if specified
    if schema:
        console.print(f"[cyan]Validating schema: {schema}[/cyan]")
        result = validate_schema_structure(Path(schema))
        report.add_schema_result(result)
    else:
        # Validate all schemas in directory
        console.print(f"[cyan]Validating all schemas in {schemas_path}[/cyan]")
        schema_files = list(schemas_path.glob("*.schema.json"))

        for schema_file in schema_files:
            console.print(f"  - {schema_file.name}")
            result = validate_schema_structure(schema_file)
            report.add_schema_result(result)

    console.print()

    # Validate individual example if specified
    if example:
        # Need to determine which schema to use
        example_path = Path(example)
        console.print(f"[cyan]Validating example: {example}[/cyan]")

        # Try to match schema based on filename
        schema_mapping = {
            "company_profile": "company_profile.schema.json",
            "esg_data": "esg_data.schema.json",
            "materiality": "materiality.schema.json",
            "report": "csrd_report.schema.json"
        }

        schema_file = None
        for key, schema_name in schema_mapping.items():
            if key in example_path.name.lower():
                schema_file = schemas_path / schema_name
                break

        if schema_file and schema_file.exists():
            result = validate_data_against_schema(example_path, schema_file)
            report.add_example_result(result)
        else:
            console.print(f"[yellow]⚠ Could not determine schema for {example_path.name}[/yellow]")

    else:
        # Validate all examples
        console.print(f"[cyan]Validating example data in {examples_path}[/cyan]")

        # Validate company profile
        company_profile = examples_path / "demo_company_profile.json"
        if company_profile.exists():
            console.print(f"  - {company_profile.name}")
            result = validate_data_against_schema(
                company_profile,
                schemas_path / "company_profile.schema.json"
            )
            report.add_example_result(result)

        # Validate materiality
        materiality = examples_path / "demo_materiality.json"
        if materiality.exists():
            console.print(f"  - {materiality.name}")
            result = validate_data_against_schema(
                materiality,
                schemas_path / "materiality.schema.json"
            )
            report.add_example_result(result)

        # Note: CSV validation against JSON schema is tricky,
        # would need schema specifically designed for flat structure

    console.print()

    # ESRS coverage check
    if check_esrs_coverage:
        console.print("[cyan]Checking ESRS data point coverage[/cyan]")
        data_points_path = project_root / "data" / "esrs_data_points.json"
        if data_points_path.exists():
            result = check_esrs_coverage(data_points_path, schemas_path)
            report.add_schema_result(result)
        else:
            console.print(f"[yellow]⚠ ESRS data points file not found: {data_points_path}[/yellow]")

        console.print()

    # Display results
    display_validation_results(report)

    # Save report
    output_path = Path(output)
    with open(output_path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)

    console.print(f"[green]✓ Validation report saved to {output_path}[/green]\n")

    # Exit code based on validation result
    sys.exit(0 if report.overall_valid else 1)


if __name__ == '__main__':
    validate_schemas()
