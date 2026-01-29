# -*- coding: utf-8 -*-
"""
CLI Output Formatters for GL-FOUND-X-002.

This module provides output formatters for validation reports in various formats:
- pretty: Colorized terminal output with context
- json: JSON output for machine processing
- table: Compact tabular output
- sarif: SARIF 2.1.0 format for IDE/CI integration
- text: Plain text output (no colors)

Example:
    >>> from greenlang.schema.cli.formatters import format_pretty, format_json
    >>> print(format_pretty(report, use_color=True))
    >>> print(format_json(report, indent=2))

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator
"""

from greenlang.schema.cli.formatters.pretty import (
    PrettyFormatter,
    format_pretty,
)

from greenlang.schema.cli.formatters.json_fmt import (
    JSONFormatter,
    CompactJSONFormatter,
    NDJSONFormatter,
    format_json,
)

from greenlang.schema.cli.formatters.table import (
    TableFormatter,
    CSVFormatter,
    format_table,
)

from greenlang.schema.cli.formatters.sarif import (
    SARIFFormatter,
    SARIFFixFormatter,
    format_sarif,
    SARIF_VERSION,
    SARIF_SCHEMA,
)

from greenlang.schema.cli.formatters.text import (
    TextFormatter,
    format_text,
)


# Format type to formatter function mapping
FORMAT_FUNCTIONS = {
    "pretty": format_pretty,
    "json": format_json,
    "table": format_table,
    "sarif": format_sarif,
    "text": format_text,
}

# Format type to formatter class mapping
FORMAT_CLASSES = {
    "pretty": PrettyFormatter,
    "json": JSONFormatter,
    "compact_json": CompactJSONFormatter,
    "ndjson": NDJSONFormatter,
    "table": TableFormatter,
    "csv": CSVFormatter,
    "sarif": SARIFFormatter,
    "sarif_fix": SARIFFixFormatter,
    "text": TextFormatter,
}


def get_formatter(format_type: str, **kwargs):
    """
    Get a formatter instance by format type.

    Args:
        format_type: Format type name (pretty, json, table, sarif, etc.).
        **kwargs: Additional arguments to pass to formatter constructor.

    Returns:
        Formatter instance.

    Raises:
        ValueError: If format type is not recognized.

    Example:
        >>> formatter = get_formatter("pretty", use_color=True)
        >>> output = formatter.format(report)
    """
    if format_type not in FORMAT_CLASSES:
        raise ValueError(
            f"Unknown format type '{format_type}'. "
            f"Available formats: {', '.join(FORMAT_CLASSES.keys())}"
        )

    return FORMAT_CLASSES[format_type](**kwargs)


def format_report(report, format_type: str = "pretty", **kwargs) -> str:
    """
    Format a validation report using the specified format.

    Args:
        report: ValidationReport or BatchValidationReport to format.
        format_type: Format type (pretty, json, table, sarif, text).
        **kwargs: Additional arguments for the formatter.

    Returns:
        Formatted string.

    Raises:
        ValueError: If format type is not recognized.

    Example:
        >>> output = format_report(report, "json", indent=2)
        >>> print(output)
    """
    if format_type not in FORMAT_FUNCTIONS:
        raise ValueError(
            f"Unknown format type '{format_type}'. "
            f"Available formats: {', '.join(FORMAT_FUNCTIONS.keys())}"
        )

    return FORMAT_FUNCTIONS[format_type](report, **kwargs)


__all__ = [
    # Pretty formatter
    "PrettyFormatter",
    "format_pretty",
    # JSON formatters
    "JSONFormatter",
    "CompactJSONFormatter",
    "NDJSONFormatter",
    "format_json",
    # Table formatters
    "TableFormatter",
    "CSVFormatter",
    "format_table",
    # SARIF formatters
    "SARIFFormatter",
    "SARIFFixFormatter",
    "format_sarif",
    "SARIF_VERSION",
    "SARIF_SCHEMA",
    # Text formatter
    "TextFormatter",
    "format_text",
    # Utility functions
    "get_formatter",
    "format_report",
    # Format mappings
    "FORMAT_FUNCTIONS",
    "FORMAT_CLASSES",
]
