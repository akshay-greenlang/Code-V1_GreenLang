"""
Schema management commands for GreenLang CLI
"""

import click
import json
import yaml
import sys
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax

console = Console()

# Schema directory - look for schemas in the project
SCHEMA_DIR = Path(__file__).parent.parent.parent / "schemas"

# Built-in schemas that should be available
BUILTIN_SCHEMAS = {
    "pipeline": "gl_pipeline.schema.v1.json",
    "pack": "pack.schema.v1.json",
    "agent": "agent.schema.v1.json",
    "dataset": "dataset.schema.v1.json",
}


@click.group()
def schema():
    """Schema management commands

    Manage and validate against GreenLang schemas.
    """


@schema.command(name="list")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
def list_schemas(format: str):
    """List available schemas

    Examples:
        gl schema list              # List schemas in table format
        gl schema list --format json    # List schemas in JSON format
    """
    schemas = []

    # Check for built-in schemas
    for schema_name, schema_file in BUILTIN_SCHEMAS.items():
        schema_path = SCHEMA_DIR / schema_file
        if schema_path.exists():
            try:
                with open(schema_path, "r") as f:
                    schema_data = json.load(f)
                    schemas.append(
                        {
                            "name": schema_name,
                            "file": schema_file,
                            "title": schema_data.get("title", schema_name),
                            "description": schema_data.get("description", ""),
                            "version": schema_data.get("version", "1.0"),
                            "type": "built-in",
                            "path": str(schema_path),
                        }
                    )
            except Exception as e:
                schemas.append(
                    {
                        "name": schema_name,
                        "file": schema_file,
                        "title": schema_name,
                        "description": f"Error loading: {e}",
                        "version": "unknown",
                        "type": "built-in",
                        "path": str(schema_path),
                    }
                )

    # Check for custom schemas in current directory
    current_dir = Path.cwd()
    for schema_file in current_dir.glob("**/*.schema.json"):
        if schema_file.is_file():
            try:
                with open(schema_file, "r") as f:
                    schema_data = json.load(f)
                    name = schema_file.stem.replace(".schema", "")
                    schemas.append(
                        {
                            "name": name,
                            "file": schema_file.name,
                            "title": schema_data.get("title", name),
                            "description": schema_data.get("description", ""),
                            "version": schema_data.get("version", "1.0"),
                            "type": "custom",
                            "path": str(schema_file),
                        }
                    )
            except Exception:
                # Skip invalid schema files
                continue

    if not schemas:
        console.print("[yellow]No schemas found[/yellow]")
        console.print(f"Expected schema directory: {SCHEMA_DIR}")
        return

    if format == "json":
        console.print(json.dumps(schemas, indent=2))
    else:
        table = Table(title="Available Schemas")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Title", style="green")
        table.add_column("Version", style="yellow")
        table.add_column("Type", style="magenta")
        table.add_column("Description", style="dim")

        for schema in sorted(schemas, key=lambda s: s["name"]):
            table.add_row(
                schema["name"],
                schema["title"],
                schema["version"],
                schema["type"],
                (
                    schema["description"][:50] + "..."
                    if len(schema["description"]) > 50
                    else schema["description"]
                ),
            )

        console.print(table)


@schema.command(name="print")
@click.argument("schema_name")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "yaml"]),
    default="json",
    help="Output format",
)
@click.option("--pretty", is_flag=True, help="Pretty print with syntax highlighting")
def print_schema(schema_name: str, format: str, pretty: bool):
    """Print a schema definition

    Examples:
        gl schema print pipeline           # Print pipeline schema
        gl schema print pack --format yaml    # Print pack schema as YAML
        gl schema print agent --pretty        # Pretty print with syntax highlighting
    """
    schema_path = None

    # Check built-in schemas first
    if schema_name in BUILTIN_SCHEMAS:
        schema_path = SCHEMA_DIR / BUILTIN_SCHEMAS[schema_name]
    else:
        # Check for custom schema files
        current_dir = Path.cwd()
        candidates = [
            current_dir / f"{schema_name}.schema.json",
            current_dir / f"schemas/{schema_name}.schema.json",
            Path(schema_name) if Path(schema_name).exists() else None,
        ]

        for candidate in candidates:
            if candidate and candidate.exists():
                schema_path = candidate
                break

    if not schema_path or not schema_path.exists():
        console.print(f"[red]Schema '{schema_name}' not found[/red]")
        console.print("Available schemas:")
        # Show available schemas
        ctx = click.get_current_context()
        ctx.invoke(list_schemas, format="table")
        sys.exit(1)

    try:
        with open(schema_path, "r") as f:
            schema_data = json.load(f)

        if format == "yaml":
            content = yaml.dump(
                schema_data, default_flow_style=False, sort_keys=False, indent=2
            )
            lang = "yaml"
        else:
            content = json.dumps(schema_data, indent=2)
            lang = "json"

        if pretty:
            syntax = Syntax(content, lang, theme="monokai", line_numbers=True)
            console.print(syntax)
        else:
            console.print(content)

    except Exception as e:
        console.print(f"[red]Error reading schema: {e}[/red]")
        sys.exit(1)


@schema.command(name="validate")
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--schema", "-s", required=True, help="Schema name or path to validate against"
)
@click.option("--json", "output_json", is_flag=True, help="Output results as JSON")
@click.option("--strict", is_flag=True, help="Fail on warnings")
def validate_file(file_path: str, schema: str, output_json: bool, strict: bool):
    """Validate a file against a schema

    Examples:
        gl schema validate pipeline.yaml --schema pipeline
        gl schema validate pack.yaml --schema pack --json
        gl schema validate data.json --schema /path/to/custom.schema.json
    """
    file_path_obj = Path(file_path)

    # Determine schema path
    schema_path = None
    if schema in BUILTIN_SCHEMAS:
        schema_path = SCHEMA_DIR / BUILTIN_SCHEMAS[schema]
    elif Path(schema).exists():
        schema_path = Path(schema)
    else:
        # Try to find schema file
        current_dir = Path.cwd()
        candidates = [
            current_dir / f"{schema}.schema.json",
            current_dir / f"schemas/{schema}.schema.json",
        ]

        for candidate in candidates:
            if candidate.exists():
                schema_path = candidate
                break

    if not schema_path or not schema_path.exists():
        error_msg = f"Schema '{schema}' not found"
        if output_json:
            result = {"ok": False, "errors": [error_msg], "warnings": []}
            console.print(json.dumps(result, indent=2))
        else:
            console.print(f"[red]Error: {error_msg}[/red]")
        sys.exit(1)

    errors = []
    warnings = []

    try:
        # Load schema
        with open(schema_path, "r") as f:
            schema_data = json.load(f)

        # Load file to validate
        with open(file_path, "r") as f:
            if file_path_obj.suffix in [".yaml", ".yml"]:
                file_data = yaml.safe_load(f)
            else:
                file_data = json.load(f)

        # Validate using jsonschema
        try:
            import jsonschema

            validator = jsonschema.Draft7Validator(schema_data)

            # Collect validation errors
            for error in validator.iter_errors(file_data):
                error_path = (
                    ".".join(str(p) for p in error.path) if error.path else "root"
                )
                error_msg = f"{error_path}: {error.message}"

                # Classify as error or warning (basic heuristic)
                if error.validator in ["required", "type", "enum"]:
                    errors.append(error_msg)
                else:
                    warnings.append(error_msg)

            # Check for additional warnings (custom logic)
            if isinstance(file_data, dict):
                # Check for deprecated fields, etc.
                if "deprecated_field" in file_data:
                    warnings.append(
                        "deprecated_field is deprecated and will be removed in future versions"
                    )

        except ImportError:
            errors.append(
                "jsonschema library not installed. Install with: pip install jsonschema"
            )

    except Exception as e:
        errors.append(f"Validation error: {str(e)}")

    # Determine success
    success = len(errors) == 0 and (not strict or len(warnings) == 0)

    if output_json:
        result = {
            "ok": success,
            "file": str(file_path),
            "schema": str(schema_path),
            "errors": errors,
            "warnings": warnings,
        }
        console.print(json.dumps(result, indent=2))
    else:
        if success:
            console.print(f"[green]+ Validation passed[/green] - {file_path_obj.name}")
        else:
            console.print(f"[red]x Validation failed[/red] - {file_path_obj.name}")

        if errors:
            console.print(f"\n[bold red]Errors ({len(errors)}):[/bold red]")
            for error in errors:
                console.print(f"  • {error}")

        if warnings:
            console.print(f"\n[bold yellow]Warnings ({len(warnings)}):[/bold yellow]")
            for warning in warnings:
                console.print(f"  • {warning}")

        console.print(f"\nSchema: {schema_path}")

    # Exit with appropriate code
    if not success:
        sys.exit(1)


@schema.command(name="init")
@click.argument("schema_name")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option(
    "--type",
    "schema_type",
    type=click.Choice(["object", "array", "string", "number", "boolean"]),
    default="object",
    help="Root schema type",
)
def init_schema(schema_name: str, output: Optional[str], schema_type: str):
    """Create a new schema template

    Examples:
        gl schema init my-schema                    # Create my-schema.schema.json
        gl schema init custom --output custom.json    # Create custom.json
        gl schema init data --type array              # Create array schema
    """
    # Determine output file
    if output:
        output_path = Path(output)
    else:
        output_path = Path(f"{schema_name}.schema.json")

    if output_path.exists():
        console.print(f"[red]File '{output_path}' already exists[/red]")
        sys.exit(1)

    # Create basic schema template
    template = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": f"#{schema_name}",
        "title": schema_name.replace("-", " ").replace("_", " ").title(),
        "description": f"Schema for {schema_name}",
        "type": schema_type,
        "version": "1.0",
    }

    if schema_type == "object":
        template.update(
            {
                "properties": {
                    "name": {"type": "string", "description": "Name of the resource"},
                    "version": {
                        "type": "string",
                        "pattern": "^\\d+\\.\\d+\\.\\d+$",
                        "description": "Semantic version",
                    },
                },
                "required": ["name", "version"],
                "additionalProperties": False,
            }
        )
    elif schema_type == "array":
        template.update(
            {
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Unique identifier"}
                    },
                    "required": ["id"],
                }
            }
        )

    # Write schema file
    with open(output_path, "w") as f:
        json.dump(template, f, indent=2)

    console.print(f"[green]+ Created schema template: {output_path}[/green]")
    console.print(f"Schema type: {schema_type}")
    console.print("\nNext steps:")
    console.print(f"  1. Edit {output_path} to define your schema")
    console.print(
        f"  2. Use 'gl schema validate <file> --schema {schema_name}' to validate files"
    )
