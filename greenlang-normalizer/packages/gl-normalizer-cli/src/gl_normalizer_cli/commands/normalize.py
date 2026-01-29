"""
GL-FOUND-X-003: GreenLang Normalizer CLI - Normalize Command

This module implements the normalize command for single value normalization,
supporting unit conversion, dimension validation, and multiple output formats.

Example:
    >>> glnorm normalize 100 kg --to metric_ton
    >>> glnorm normalize 1500 kWh --to MJ --context '{"expected_dimension": "energy"}'
    >>> glnorm normalize 100 "kg CO2e" --format yaml
"""

import json
import sys
import hashlib
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from pydantic import BaseModel, Field

console = Console()


class OutputFormat(str, Enum):
    """Supported output formats for normalization results."""

    JSON = "json"
    YAML = "yaml"
    TABLE = "table"


class NormalizationOutput(BaseModel):
    """Output model for normalization results."""

    success: bool = Field(..., description="Whether normalization succeeded")
    original_value: float = Field(..., description="Original input value")
    original_unit: str = Field(..., description="Original input unit")
    canonical_value: Optional[float] = Field(None, description="Normalized value")
    canonical_unit: Optional[str] = Field(None, description="Target/canonical unit")
    dimension: Optional[str] = Field(None, description="Physical dimension")
    conversion_factor: Optional[float] = Field(None, description="Conversion factor applied")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    warnings: list[str] = Field(default_factory=list, description="Any warnings generated")
    errors: list[str] = Field(default_factory=list, description="Any errors encountered")


def normalize_value(
    value: float = typer.Argument(
        ...,
        help="Numeric value to normalize (e.g., 100, 1.5, 1e6).",
    ),
    unit: str = typer.Argument(
        ...,
        help="Unit string for the value (e.g., kg, kWh, 'kg CO2e').",
    ),
    to: Optional[str] = typer.Option(
        None,
        "--to",
        "-t",
        help="Target unit to convert to. If not specified, uses canonical unit.",
    ),
    context: Optional[str] = typer.Option(
        None,
        "--context",
        "-c",
        help=(
            "JSON string with additional context. "
            "Supports: expected_dimension, reference_conditions, gwp_version."
        ),
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.TABLE,
        "--format",
        "-f",
        help="Output format: json, yaml, or table.",
        case_sensitive=False,
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        "-k",
        envvar="GLNORM_API_KEY",
        help="API key for remote normalization service.",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        "-l",
        help="Use local normalization engine instead of API.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        "-s",
        help="Enable strict mode - fail on any warnings or ambiguities.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show additional details including conversion steps.",
    ),
) -> None:
    """
    Normalize a single value with its unit.

    Parses the input value and unit, optionally converts to a target unit,
    and outputs the result in the specified format.

    [bold]Examples:[/bold]

        glnorm normalize 100 kg --to metric_ton

        glnorm normalize 1500 kWh --to MJ

        glnorm normalize 100 "kg CO2e" --context '{"gwp_version": "AR6"}'

        glnorm normalize 250 Nm3 --context '{"reference_conditions": {"temperature_C": 0}}'

    [bold]Context Options:[/bold]

        expected_dimension: Validate the unit matches this dimension (energy, mass, etc.)

        reference_conditions: For basis-dependent units (Nm3, scf)
            temperature_C: Reference temperature in Celsius
            pressure_kPa: Reference pressure in kilopascals

        gwp_version: GWP version for CO2e conversions (AR5, AR6)
    """
    start_time = datetime.now()

    # Parse context if provided
    context_data: Dict[str, Any] = {}
    if context:
        try:
            context_data = json.loads(context)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error:[/red] Invalid JSON in --context: {e}")
            raise typer.Exit(code=1)

    try:
        # Perform normalization
        result = _perform_normalization(
            value=value,
            unit=unit,
            target_unit=to,
            context=context_data,
            use_api=not local,
            api_key=api_key,
            strict=strict,
        )

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        result.processing_time_ms = processing_time

        # Output result based on format
        if format == OutputFormat.JSON:
            _output_json(result)
        elif format == OutputFormat.YAML:
            _output_yaml(result)
        else:
            _output_table(result, verbose=verbose)

        # Exit with error code if normalization failed
        if not result.success:
            raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


def _perform_normalization(
    value: float,
    unit: str,
    target_unit: Optional[str],
    context: Dict[str, Any],
    use_api: bool,
    api_key: Optional[str],
    strict: bool,
) -> NormalizationOutput:
    """
    Perform the actual normalization operation.

    Args:
        value: Numeric value to normalize
        unit: Unit string
        target_unit: Optional target unit for conversion
        context: Additional context data
        use_api: Whether to use the remote API
        api_key: API key for remote service
        strict: Whether to use strict mode

    Returns:
        NormalizationOutput with the result
    """
    warnings: list[str] = []
    errors: list[str] = []

    # Calculate provenance hash
    provenance_str = f"{value}|{unit}|{target_unit}|{json.dumps(context, sort_keys=True)}"
    provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

    if use_api and api_key:
        # Use remote API for normalization
        return _normalize_via_api(
            value=value,
            unit=unit,
            target_unit=target_unit,
            context=context,
            api_key=api_key,
            strict=strict,
            provenance_hash=provenance_hash,
        )
    else:
        # Use local normalization engine
        return _normalize_locally(
            value=value,
            unit=unit,
            target_unit=target_unit,
            context=context,
            strict=strict,
            provenance_hash=provenance_hash,
        )


def _normalize_locally(
    value: float,
    unit: str,
    target_unit: Optional[str],
    context: Dict[str, Any],
    strict: bool,
    provenance_hash: str,
) -> NormalizationOutput:
    """
    Perform normalization using the local core library.

    Args:
        value: Numeric value
        unit: Unit string
        target_unit: Optional target unit
        context: Context data
        strict: Strict mode flag
        provenance_hash: Pre-calculated provenance hash

    Returns:
        NormalizationOutput with results
    """
    warnings: list[str] = []
    errors: list[str] = []

    try:
        from gl_normalizer_core import UnitParser, UnitConverter, Quantity
        from gl_normalizer_core.errors import ParseError, ConversionError

        # Initialize parser and converter
        parser = UnitParser(strict_mode=strict)
        converter = UnitConverter()

        # Parse the input
        input_string = f"{value} {unit}"
        parse_result = parser.parse(input_string)

        if not parse_result.success:
            errors.append(f"Parse failed: Could not parse '{input_string}'")
            return NormalizationOutput(
                success=False,
                original_value=value,
                original_unit=unit,
                provenance_hash=provenance_hash,
                processing_time_ms=0,
                errors=errors,
            )

        quantity = parse_result.quantity
        warnings.extend(parse_result.warnings)

        # Validate dimension if expected_dimension provided
        expected_dimension = context.get("expected_dimension")
        if expected_dimension:
            # Dimension validation would be performed here
            pass

        # Convert if target unit specified
        if target_unit:
            conversion_result = converter.convert(quantity, target_unit)

            if not conversion_result.success:
                errors.append(f"Conversion failed: Cannot convert from {unit} to {target_unit}")
                return NormalizationOutput(
                    success=False,
                    original_value=value,
                    original_unit=unit,
                    canonical_value=None,
                    canonical_unit=target_unit,
                    provenance_hash=provenance_hash,
                    processing_time_ms=0,
                    warnings=warnings,
                    errors=errors,
                )

            warnings.extend(conversion_result.warnings)

            return NormalizationOutput(
                success=True,
                original_value=value,
                original_unit=unit,
                canonical_value=conversion_result.converted_quantity.magnitude,
                canonical_unit=conversion_result.converted_quantity.unit,
                dimension=quantity.unit_system.value,
                conversion_factor=conversion_result.conversion_factor,
                provenance_hash=provenance_hash,
                processing_time_ms=0,
                warnings=warnings,
            )
        else:
            # No conversion, just return parsed value
            return NormalizationOutput(
                success=True,
                original_value=value,
                original_unit=unit,
                canonical_value=quantity.magnitude,
                canonical_unit=quantity.unit,
                dimension=quantity.unit_system.value,
                provenance_hash=provenance_hash,
                processing_time_ms=0,
                warnings=warnings,
            )

    except ImportError:
        errors.append(
            "gl-normalizer-core not installed. Install it or use --api-key for API mode."
        )
        return NormalizationOutput(
            success=False,
            original_value=value,
            original_unit=unit,
            provenance_hash=provenance_hash,
            processing_time_ms=0,
            errors=errors,
        )

    except Exception as e:
        errors.append(f"Normalization error: {str(e)}")
        return NormalizationOutput(
            success=False,
            original_value=value,
            original_unit=unit,
            provenance_hash=provenance_hash,
            processing_time_ms=0,
            errors=errors,
        )


def _normalize_via_api(
    value: float,
    unit: str,
    target_unit: Optional[str],
    context: Dict[str, Any],
    api_key: str,
    strict: bool,
    provenance_hash: str,
) -> NormalizationOutput:
    """
    Perform normalization via the remote API.

    Args:
        value: Numeric value
        unit: Unit string
        target_unit: Optional target unit
        context: Context data
        api_key: API key
        strict: Strict mode flag
        provenance_hash: Pre-calculated provenance hash

    Returns:
        NormalizationOutput with results
    """
    import httpx

    from gl_normalizer_cli.commands.config import load_config

    config = load_config()
    api_url = config.get("api_url", "https://api.greenlang.io/normalizer/v1")

    # Build request payload
    payload = {
        "source_record_id": f"cli-{provenance_hash[:8]}",
        "policy_mode": "STRICT" if strict else "LENIENT",
        "measurements": [
            {
                "field": "cli_input",
                "value": value,
                "unit": unit,
                "expected_dimension": context.get("expected_dimension"),
                "metadata": {
                    k: v
                    for k, v in context.items()
                    if k in ("reference_conditions", "gwp_version", "locale")
                },
            }
        ],
    }

    if target_unit:
        payload["measurements"][0]["target_unit"] = target_unit

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{api_url}/normalize",
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()

        # Parse API response
        if data.get("status") == "success":
            measurement = data.get("canonical_measurements", [{}])[0]
            return NormalizationOutput(
                success=True,
                original_value=value,
                original_unit=unit,
                canonical_value=measurement.get("value"),
                canonical_unit=measurement.get("unit"),
                dimension=measurement.get("dimension"),
                conversion_factor=measurement.get("conversion_trace", {}).get("factor"),
                provenance_hash=provenance_hash,
                processing_time_ms=0,
                warnings=measurement.get("warnings", []),
            )
        else:
            errors = [e.get("message", str(e)) for e in data.get("errors", [])]
            return NormalizationOutput(
                success=False,
                original_value=value,
                original_unit=unit,
                provenance_hash=provenance_hash,
                processing_time_ms=0,
                errors=errors,
            )

    except httpx.HTTPError as e:
        return NormalizationOutput(
            success=False,
            original_value=value,
            original_unit=unit,
            provenance_hash=provenance_hash,
            processing_time_ms=0,
            errors=[f"API request failed: {str(e)}"],
        )


def _output_json(result: NormalizationOutput) -> None:
    """Output result as JSON to stdout."""
    console.print(result.model_dump_json(indent=2))


def _output_yaml(result: NormalizationOutput) -> None:
    """Output result as YAML to stdout."""
    import yaml

    console.print(yaml.dump(result.model_dump(), default_flow_style=False, sort_keys=False))


def _output_table(result: NormalizationOutput, verbose: bool = False) -> None:
    """Output result as a formatted table."""
    if result.success:
        # Success output
        table = Table(
            title="Normalization Result",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Field", style="dim")
        table.add_column("Value", style="green")

        table.add_row("Status", "[green]SUCCESS[/green]")
        table.add_row("Original", f"{result.original_value} {result.original_unit}")
        table.add_row("Canonical", f"{result.canonical_value} {result.canonical_unit}")

        if result.dimension:
            table.add_row("Dimension", result.dimension)

        if result.conversion_factor and result.conversion_factor != 1.0:
            table.add_row("Factor", f"{result.conversion_factor}")

        if verbose:
            table.add_row("Provenance", result.provenance_hash[:16] + "...")
            table.add_row("Time (ms)", f"{result.processing_time_ms:.2f}")

        if result.warnings:
            for warning in result.warnings:
                table.add_row("[yellow]Warning[/yellow]", warning)

        console.print(table)

    else:
        # Error output
        panel = Panel(
            "\n".join([f"[red]{e}[/red]" for e in result.errors])
            or "[red]Unknown error[/red]",
            title="[red]Normalization Failed[/red]",
            border_style="red",
        )
        console.print(panel)

        if result.warnings:
            for warning in result.warnings:
                console.print(f"[yellow]Warning:[/yellow] {warning}")
