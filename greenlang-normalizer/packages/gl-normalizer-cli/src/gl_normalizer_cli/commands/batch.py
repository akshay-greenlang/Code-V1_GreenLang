"""
GL-FOUND-X-003: GreenLang Normalizer CLI - Batch Command

This module implements the batch processing command for normalizing
multiple records from CSV, JSON, or JSONL files with progress tracking
and multiple output formats.

Example:
    >>> glnorm batch input.csv --output output.json
    >>> glnorm batch data.jsonl --mode partial --format csv
    >>> cat input.json | glnorm batch - --output -
"""

import csv
import json
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TextIO
from enum import Enum

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich import box
from pydantic import BaseModel, Field

console = Console()


class BatchMode(str, Enum):
    """Processing modes for batch operations."""

    FAIL_FAST = "fail_fast"  # Stop on first error
    PARTIAL = "partial"  # Continue and return partial results
    THRESHOLD = "threshold"  # Stop if error rate exceeds threshold


class OutputFormat(str, Enum):
    """Supported output formats for batch results."""

    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    YAML = "yaml"


class BatchSummary(BaseModel):
    """Summary statistics for batch processing."""

    total_records: int = Field(..., description="Total records processed")
    success_count: int = Field(..., description="Successfully normalized records")
    error_count: int = Field(..., description="Records with errors")
    warning_count: int = Field(..., description="Records with warnings")
    processing_time_ms: float = Field(..., description="Total processing time")
    records_per_second: float = Field(..., description="Processing throughput")


class BatchResult(BaseModel):
    """Result model for a single record in batch processing."""

    source_record_id: str = Field(..., description="Record identifier")
    success: bool = Field(..., description="Whether normalization succeeded")
    original_value: Optional[float] = Field(None, description="Original value")
    original_unit: Optional[str] = Field(None, description="Original unit")
    canonical_value: Optional[float] = Field(None, description="Normalized value")
    canonical_unit: Optional[str] = Field(None, description="Canonical unit")
    dimension: Optional[str] = Field(None, description="Physical dimension")
    provenance_hash: str = Field(..., description="SHA-256 hash")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    errors: List[str] = Field(default_factory=list, description="Errors")


def process_batch(
    input_file: str = typer.Argument(
        ...,
        help="Input file path (CSV, JSON, JSONL) or '-' for stdin.",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path or '-' for stdout. If not specified, prints summary only.",
    ),
    mode: BatchMode = typer.Option(
        BatchMode.PARTIAL,
        "--mode",
        "-m",
        help="Processing mode: fail_fast, partial, or threshold.",
        case_sensitive=False,
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.JSON,
        "--format",
        "-f",
        help="Output format: json, jsonl, csv, or yaml.",
        case_sensitive=False,
    ),
    target_unit: Optional[str] = typer.Option(
        None,
        "--to",
        "-t",
        help="Target unit for conversion (applies to all records).",
    ),
    value_column: str = typer.Option(
        "value",
        "--value-col",
        help="Column name for numeric values (CSV/JSON).",
    ),
    unit_column: str = typer.Option(
        "unit",
        "--unit-col",
        help="Column name for unit strings (CSV/JSON).",
    ),
    id_column: Optional[str] = typer.Option(
        None,
        "--id-col",
        help="Column name for record IDs. If not specified, uses row index.",
    ),
    error_threshold: float = typer.Option(
        0.1,
        "--error-threshold",
        help="Error rate threshold for 'threshold' mode (0.0-1.0).",
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
        help="Enable strict mode for all normalizations.",
    ),
    batch_size: int = typer.Option(
        1000,
        "--batch-size",
        "-b",
        help="Number of records per batch for API mode.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress progress output.",
    ),
) -> None:
    """
    Process a batch of records from a file.

    Reads records from CSV, JSON, or JSONL files, normalizes each record,
    and outputs results in the specified format.

    [bold]Input Formats:[/bold]

        CSV: Requires columns for value and unit (configurable via --value-col, --unit-col)

        JSON: Array of objects with value/unit fields

        JSONL: One JSON object per line

    [bold]Examples:[/bold]

        glnorm batch data.csv --output results.json

        glnorm batch records.jsonl --mode fail_fast --output -

        cat input.json | glnorm batch - --format csv --output output.csv

        glnorm batch data.csv --to MJ --value-col "energy" --unit-col "energy_unit"

    [bold]Processing Modes:[/bold]

        fail_fast: Stop processing on first error

        partial: Continue processing, output both successes and failures

        threshold: Stop if error rate exceeds --error-threshold
    """
    start_time = datetime.now()

    try:
        # Load input records
        records = _load_input(
            input_file=input_file,
            value_column=value_column,
            unit_column=unit_column,
            id_column=id_column,
        )

        total_records = len(records)
        if total_records == 0:
            console.print("[yellow]Warning:[/yellow] No records found in input file.")
            raise typer.Exit(code=0)

        if not quiet:
            console.print(f"[cyan]Processing {total_records} records...[/cyan]")

        # Process records with progress tracking
        results = _process_records(
            records=records,
            target_unit=target_unit,
            mode=mode,
            error_threshold=error_threshold,
            use_api=not local,
            api_key=api_key,
            strict=strict,
            batch_size=batch_size,
            quiet=quiet,
        )

        # Calculate summary
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        success_count = sum(1 for r in results if r.success)
        error_count = sum(1 for r in results if not r.success)
        warning_count = sum(1 for r in results if r.warnings)

        summary = BatchSummary(
            total_records=len(results),
            success_count=success_count,
            error_count=error_count,
            warning_count=warning_count,
            processing_time_ms=processing_time,
            records_per_second=(
                len(results) / (processing_time / 1000) if processing_time > 0 else 0
            ),
        )

        # Output results
        if output:
            _write_output(
                results=results,
                output_path=output,
                format=format,
                summary=summary,
            )
            if not quiet:
                console.print(f"[green]Results written to: {output}[/green]")

        # Print summary
        if not quiet:
            _print_summary(summary)

        # Exit with error code if any failures
        if error_count > 0:
            raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(code=1)


def _load_input(
    input_file: str,
    value_column: str,
    unit_column: str,
    id_column: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Load records from input file.

    Args:
        input_file: Path to input file or '-' for stdin
        value_column: Column name for values
        unit_column: Column name for units
        id_column: Optional column name for record IDs

    Returns:
        List of record dictionaries
    """
    records: List[Dict[str, Any]] = []

    # Handle stdin
    if input_file == "-":
        content = sys.stdin.read()
        return _parse_content(content, value_column, unit_column, id_column)

    # Read from file
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(path, "r", encoding="utf-8") as f:
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return _load_csv(f, value_column, unit_column, id_column)
        elif suffix == ".json":
            return _load_json(f, value_column, unit_column, id_column)
        elif suffix in (".jsonl", ".ndjson"):
            return _load_jsonl(f, value_column, unit_column, id_column)
        else:
            # Try to auto-detect format
            content = f.read()
            return _parse_content(content, value_column, unit_column, id_column)


def _parse_content(
    content: str,
    value_column: str,
    unit_column: str,
    id_column: Optional[str],
) -> List[Dict[str, Any]]:
    """Auto-detect and parse content format."""
    content = content.strip()

    # Try JSON array first
    if content.startswith("["):
        data = json.loads(content)
        return _normalize_records(data, value_column, unit_column, id_column)

    # Try JSONL
    if content.startswith("{"):
        records = []
        for i, line in enumerate(content.split("\n")):
            line = line.strip()
            if line:
                record = json.loads(line)
                record["_index"] = i
                records.append(record)
        return _normalize_records(records, value_column, unit_column, id_column)

    # Assume CSV
    import io

    return _load_csv(io.StringIO(content), value_column, unit_column, id_column)


def _load_csv(
    f: TextIO,
    value_column: str,
    unit_column: str,
    id_column: Optional[str],
) -> List[Dict[str, Any]]:
    """Load records from CSV file."""
    reader = csv.DictReader(f)
    records = []

    for i, row in enumerate(reader):
        if value_column not in row:
            raise ValueError(f"Value column '{value_column}' not found in CSV")
        if unit_column not in row:
            raise ValueError(f"Unit column '{unit_column}' not found in CSV")

        record_id = row.get(id_column, str(i)) if id_column else str(i)

        records.append({
            "source_record_id": record_id,
            "value": float(row[value_column]),
            "unit": row[unit_column],
            "_original": row,
        })

    return records


def _load_json(
    f: TextIO,
    value_column: str,
    unit_column: str,
    id_column: Optional[str],
) -> List[Dict[str, Any]]:
    """Load records from JSON array file."""
    data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    return _normalize_records(data, value_column, unit_column, id_column)


def _load_jsonl(
    f: TextIO,
    value_column: str,
    unit_column: str,
    id_column: Optional[str],
) -> List[Dict[str, Any]]:
    """Load records from JSONL file."""
    records = []

    for i, line in enumerate(f):
        line = line.strip()
        if line:
            record = json.loads(line)
            record["_index"] = i
            records.append(record)

    return _normalize_records(records, value_column, unit_column, id_column)


def _normalize_records(
    data: List[Dict[str, Any]],
    value_column: str,
    unit_column: str,
    id_column: Optional[str],
) -> List[Dict[str, Any]]:
    """Normalize record structure for processing."""
    records = []

    for i, item in enumerate(data):
        record_id = item.get(id_column, str(i)) if id_column else str(item.get("_index", i))

        # Support nested value/unit fields
        value = item.get(value_column) or item.get("measurements", [{}])[0].get(value_column)
        unit = item.get(unit_column) or item.get("measurements", [{}])[0].get(unit_column)

        if value is None or unit is None:
            raise ValueError(
                f"Record {i}: Missing '{value_column}' or '{unit_column}' field"
            )

        records.append({
            "source_record_id": record_id,
            "value": float(value),
            "unit": str(unit),
            "_original": item,
        })

    return records


def _process_records(
    records: List[Dict[str, Any]],
    target_unit: Optional[str],
    mode: BatchMode,
    error_threshold: float,
    use_api: bool,
    api_key: Optional[str],
    strict: bool,
    batch_size: int,
    quiet: bool,
) -> List[BatchResult]:
    """
    Process all records with the appropriate mode.

    Args:
        records: Input records
        target_unit: Optional target unit
        mode: Processing mode
        error_threshold: Error threshold for threshold mode
        use_api: Whether to use API
        api_key: API key
        strict: Strict mode flag
        batch_size: Batch size for API calls
        quiet: Suppress progress output

    Returns:
        List of BatchResult objects
    """
    results: List[BatchResult] = []
    error_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        disable=quiet,
    ) as progress:
        task = progress.add_task("Normalizing...", total=len(records))

        for i, record in enumerate(records):
            # Check threshold mode
            if mode == BatchMode.THRESHOLD:
                current_error_rate = error_count / (i + 1) if i > 0 else 0
                if current_error_rate > error_threshold:
                    console.print(
                        f"[red]Error rate ({current_error_rate:.2%}) exceeds "
                        f"threshold ({error_threshold:.2%}). Stopping.[/red]"
                    )
                    break

            # Process record
            result = _normalize_record(
                record=record,
                target_unit=target_unit,
                use_api=use_api,
                api_key=api_key,
                strict=strict,
            )

            results.append(result)

            if not result.success:
                error_count += 1

                # Check fail_fast mode
                if mode == BatchMode.FAIL_FAST:
                    console.print(
                        f"[red]Error in record {record['source_record_id']}: "
                        f"{result.errors[0] if result.errors else 'Unknown error'}[/red]"
                    )
                    console.print("[red]Stopping due to fail_fast mode.[/red]")
                    break

            progress.update(task, advance=1)

    return results


def _normalize_record(
    record: Dict[str, Any],
    target_unit: Optional[str],
    use_api: bool,
    api_key: Optional[str],
    strict: bool,
) -> BatchResult:
    """
    Normalize a single record.

    Args:
        record: Record dictionary with value, unit, source_record_id
        target_unit: Optional target unit
        use_api: Whether to use API
        api_key: API key
        strict: Strict mode flag

    Returns:
        BatchResult for the record
    """
    value = record["value"]
    unit = record["unit"]
    source_record_id = record["source_record_id"]

    # Calculate provenance hash
    provenance_str = f"{source_record_id}|{value}|{unit}|{target_unit}"
    provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

    try:
        from gl_normalizer_core import UnitParser, UnitConverter

        parser = UnitParser(strict_mode=strict)
        converter = UnitConverter()

        # Parse
        parse_result = parser.parse(f"{value} {unit}")

        if not parse_result.success:
            return BatchResult(
                source_record_id=source_record_id,
                success=False,
                original_value=value,
                original_unit=unit,
                provenance_hash=provenance_hash,
                errors=[f"Parse failed: Could not parse '{value} {unit}'"],
            )

        quantity = parse_result.quantity
        warnings = list(parse_result.warnings)

        # Convert if target unit specified
        if target_unit:
            conversion_result = converter.convert(quantity, target_unit)

            if not conversion_result.success:
                return BatchResult(
                    source_record_id=source_record_id,
                    success=False,
                    original_value=value,
                    original_unit=unit,
                    provenance_hash=provenance_hash,
                    warnings=warnings,
                    errors=[f"Cannot convert from {unit} to {target_unit}"],
                )

            warnings.extend(conversion_result.warnings)

            return BatchResult(
                source_record_id=source_record_id,
                success=True,
                original_value=value,
                original_unit=unit,
                canonical_value=conversion_result.converted_quantity.magnitude,
                canonical_unit=conversion_result.converted_quantity.unit,
                dimension=quantity.unit_system.value,
                provenance_hash=provenance_hash,
                warnings=warnings,
            )
        else:
            return BatchResult(
                source_record_id=source_record_id,
                success=True,
                original_value=value,
                original_unit=unit,
                canonical_value=quantity.magnitude,
                canonical_unit=quantity.unit,
                dimension=quantity.unit_system.value,
                provenance_hash=provenance_hash,
                warnings=warnings,
            )

    except ImportError:
        return BatchResult(
            source_record_id=source_record_id,
            success=False,
            original_value=value,
            original_unit=unit,
            provenance_hash=provenance_hash,
            errors=["gl-normalizer-core not installed"],
        )

    except Exception as e:
        return BatchResult(
            source_record_id=source_record_id,
            success=False,
            original_value=value,
            original_unit=unit,
            provenance_hash=provenance_hash,
            errors=[str(e)],
        )


def _write_output(
    results: List[BatchResult],
    output_path: str,
    format: OutputFormat,
    summary: BatchSummary,
) -> None:
    """
    Write results to output file.

    Args:
        results: List of batch results
        output_path: Output file path or '-' for stdout
        format: Output format
        summary: Processing summary
    """
    # Determine output target
    if output_path == "-":
        output_file = sys.stdout
        close_file = False
    else:
        output_file = open(output_path, "w", encoding="utf-8")
        close_file = True

    try:
        if format == OutputFormat.JSON:
            output_data = {
                "summary": summary.model_dump(),
                "results": [r.model_dump() for r in results],
            }
            json.dump(output_data, output_file, indent=2)
            output_file.write("\n")

        elif format == OutputFormat.JSONL:
            for result in results:
                output_file.write(result.model_dump_json() + "\n")

        elif format == OutputFormat.CSV:
            if results:
                writer = csv.DictWriter(
                    output_file,
                    fieldnames=[
                        "source_record_id",
                        "success",
                        "original_value",
                        "original_unit",
                        "canonical_value",
                        "canonical_unit",
                        "dimension",
                        "provenance_hash",
                        "warnings",
                        "errors",
                    ],
                )
                writer.writeheader()
                for result in results:
                    row = result.model_dump()
                    row["warnings"] = "; ".join(row["warnings"])
                    row["errors"] = "; ".join(row["errors"])
                    writer.writerow(row)

        elif format == OutputFormat.YAML:
            import yaml

            output_data = {
                "summary": summary.model_dump(),
                "results": [r.model_dump() for r in results],
            }
            yaml.dump(output_data, output_file, default_flow_style=False, sort_keys=False)

    finally:
        if close_file:
            output_file.close()


def _print_summary(summary: BatchSummary) -> None:
    """Print processing summary to console."""
    table = Table(
        title="Batch Processing Summary",
        box=box.ROUNDED,
        show_header=False,
    )
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Total Records", str(summary.total_records))

    success_style = "green" if summary.success_count > 0 else "dim"
    table.add_row("Successful", f"[{success_style}]{summary.success_count}[/{success_style}]")

    error_style = "red" if summary.error_count > 0 else "green"
    table.add_row("Errors", f"[{error_style}]{summary.error_count}[/{error_style}]")

    warning_style = "yellow" if summary.warning_count > 0 else "dim"
    table.add_row("With Warnings", f"[{warning_style}]{summary.warning_count}[/{warning_style}]")

    table.add_row("Processing Time", f"{summary.processing_time_ms:.2f} ms")
    table.add_row("Throughput", f"{summary.records_per_second:.1f} records/sec")

    console.print()
    console.print(table)
