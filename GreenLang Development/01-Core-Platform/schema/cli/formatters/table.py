# -*- coding: utf-8 -*-
"""
Table Output Formatter for GL-FOUND-X-002.

This module provides a compact tabular output formatter for validation reports.
Suitable for terminal display with limited width and for quick scanning
of multiple findings.

Features:
- Compact columnar display
- Configurable column widths
- Unicode box-drawing characters (optional)
- Color support for severity levels
- Batch report summary tables

Example:
    >>> from greenlang.schema.cli.formatters.table import TableFormatter
    >>> formatter = TableFormatter(use_color=True, max_width=120)
    >>> output = formatter.format(validation_report)
    >>> print(output)

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from greenlang.schema.models.report import ValidationReport, BatchValidationReport
from greenlang.schema.models.finding import Finding, Severity


class TableFormatter:
    """
    Compact tabular output formatter for validation reports.

    Produces a compact table view of validation findings suitable
    for terminal display and quick scanning.

    Attributes:
        use_color: Whether to use ANSI color codes.
        max_width: Maximum table width in characters.
        use_unicode: Whether to use Unicode box-drawing characters.
        show_hints: Whether to show hint column.

    Example:
        >>> formatter = TableFormatter(use_color=True, max_width=120)
        >>> print(formatter.format(report))
        STATUS   SCHEMA                         ERRORS  WARNINGS
        INVALID  emissions/activity@1.3.0       3       1

        SEV     CODE           PATH                  MESSAGE
        ERROR   GLSCHEMA-E100  /energy_consumption   Missing required field
        ...
    """

    # ANSI color codes
    COLORS: Dict[str, str] = {
        "error": "\033[91m",
        "warning": "\033[93m",
        "info": "\033[94m",
        "success": "\033[92m",
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "header": "\033[1;37m",
    }

    # Unicode box-drawing characters
    BOX_CHARS: Dict[str, str] = {
        "h": "\u2500",      # Horizontal line
        "v": "\u2502",      # Vertical line
        "tl": "\u250c",     # Top-left corner
        "tr": "\u2510",     # Top-right corner
        "bl": "\u2514",     # Bottom-left corner
        "br": "\u2518",     # Bottom-right corner
        "lt": "\u251c",     # Left tee
        "rt": "\u2524",     # Right tee
        "tt": "\u252c",     # Top tee
        "bt": "\u2534",     # Bottom tee
        "x": "\u253c",      # Cross
    }

    # ASCII fallback characters
    ASCII_CHARS: Dict[str, str] = {
        "h": "-",
        "v": "|",
        "tl": "+",
        "tr": "+",
        "bl": "+",
        "br": "+",
        "lt": "+",
        "rt": "+",
        "tt": "+",
        "bt": "+",
        "x": "+",
    }

    # Severity abbreviations
    SEVERITY_ABBREV: Dict[Severity, str] = {
        Severity.ERROR: "ERR",
        Severity.WARNING: "WARN",
        Severity.INFO: "INFO",
    }

    def __init__(
        self,
        use_color: bool = True,
        max_width: int = 120,
        use_unicode: bool = True,
        show_hints: bool = False,
        output_stream: Any = None,
    ) -> None:
        """
        Initialize TableFormatter.

        Args:
            use_color: Whether to use ANSI color codes.
            max_width: Maximum table width in characters.
            use_unicode: Whether to use Unicode box-drawing characters.
            show_hints: Whether to show hint column for findings.
            output_stream: Output stream for TTY detection.
        """
        self.max_width = max_width
        self.show_hints = show_hints

        # Auto-detect capabilities
        stream = output_stream or sys.stdout
        if hasattr(stream, 'isatty'):
            is_tty = stream.isatty()
            self.use_color = use_color and is_tty
            self.use_unicode = use_unicode and is_tty
        else:
            self.use_color = use_color
            self.use_unicode = use_unicode

        # Select box characters
        self.box = self.BOX_CHARS if self.use_unicode else self.ASCII_CHARS

    def format(self, report: Union[ValidationReport, BatchValidationReport]) -> str:
        """
        Format validation report as a table.

        Args:
            report: ValidationReport or BatchValidationReport to format.

        Returns:
            Formatted table string.

        Example:
            >>> output = formatter.format(report)
            >>> print(output)
        """
        if isinstance(report, BatchValidationReport):
            return self._format_batch_report(report)
        return self._format_single_report(report)

    def _format_single_report(self, report: ValidationReport) -> str:
        """Format a single validation report as a table."""
        lines: List[str] = []

        # Status header
        lines.append(self._format_status_header(report))
        lines.append("")

        # Findings table
        if report.findings:
            lines.append(self._format_findings_table(report.findings))
        else:
            lines.append(self._colorize("No findings.", "success"))

        # Footer with timing
        lines.append("")
        lines.append(
            self._colorize(
                f"Validation completed in {report.timings.total_ms:.1f}ms",
                "dim"
            )
        )

        return "\n".join(lines)

    def _format_batch_report(self, report: BatchValidationReport) -> str:
        """Format a batch validation report as a table."""
        lines: List[str] = []

        # Summary header
        lines.append(self._format_batch_header(report))
        lines.append("")

        # Results table
        lines.append(self._format_batch_results_table(report))

        return "\n".join(lines)

    def _format_status_header(self, report: ValidationReport) -> str:
        """Format the status header row."""
        if report.valid:
            status = self._colorize("VALID", "success", bold=True)
        else:
            status = self._colorize("INVALID", "error", bold=True)

        schema_ref = str(report.schema_ref)

        error_count = report.summary.error_count
        warning_count = report.summary.warning_count

        counts: List[str] = []
        if error_count > 0:
            counts.append(self._colorize(f"{error_count} error(s)", "error"))
        if warning_count > 0:
            counts.append(self._colorize(f"{warning_count} warning(s)", "warning"))

        counts_str = ", ".join(counts) if counts else "no issues"

        return f"{status}  {schema_ref}  [{counts_str}]"

    def _format_batch_header(self, report: BatchValidationReport) -> str:
        """Format the batch status header."""
        success_rate = report.summary.success_rate()

        if report.summary.error_count == 0:
            status = self._colorize("ALL VALID", "success", bold=True)
        else:
            status = self._colorize("ERRORS FOUND", "error", bold=True)

        return (
            f"{status}  {report.schema_ref}  "
            f"[{report.summary.valid_count}/{report.summary.total_items} valid, "
            f"{success_rate:.1f}%]"
        )

    def _format_findings_table(self, findings: List[Finding]) -> str:
        """Format findings as a table."""
        # Define columns
        columns = [
            ("SEV", 5),
            ("CODE", 15),
            ("PATH", 30),
            ("MESSAGE", 50),
        ]

        if self.show_hints:
            columns.append(("HINT", 30))

        # Adjust widths to fit max_width
        columns = self._adjust_column_widths(columns)

        lines: List[str] = []

        # Header row
        header_cells = [
            self._pad_cell(col[0], col[1])
            for col in columns
        ]
        header = self._colorize("  ".join(header_cells), "header")
        lines.append(header)

        # Separator
        separator_cells = [self.box["h"] * col[1] for col in columns]
        lines.append("  ".join(separator_cells))

        # Data rows
        for finding in findings:
            row = self._format_finding_row(finding, columns)
            lines.append(row)

        return "\n".join(lines)

    def _format_finding_row(
        self,
        finding: Finding,
        columns: List[Tuple[str, int]]
    ) -> str:
        """Format a single finding as a table row."""
        col_widths = {col[0]: col[1] for col in columns}

        # Severity cell (colored)
        sev_abbrev = self.SEVERITY_ABBREV.get(finding.severity, "???")
        sev_color = {
            Severity.ERROR: "error",
            Severity.WARNING: "warning",
            Severity.INFO: "info",
        }.get(finding.severity, "reset")
        sev_cell = self._colorize(
            self._pad_cell(sev_abbrev, col_widths.get("SEV", 5)),
            sev_color
        )

        # Code cell
        code_cell = self._pad_cell(
            finding.code,
            col_widths.get("CODE", 15)
        )

        # Path cell
        path_cell = self._truncate_cell(
            finding.path or "(root)",
            col_widths.get("PATH", 30)
        )

        # Message cell
        message_cell = self._truncate_cell(
            finding.message,
            col_widths.get("MESSAGE", 50)
        )

        cells = [sev_cell, code_cell, path_cell, message_cell]

        # Hint cell (if enabled)
        if self.show_hints and "HINT" in col_widths:
            hint_text = ""
            if finding.hint and finding.hint.suggested_values:
                hint_text = ", ".join(
                    str(v) for v in finding.hint.suggested_values[:2]
                )
            hint_cell = self._truncate_cell(hint_text, col_widths.get("HINT", 30))
            cells.append(hint_cell)

        return "  ".join(cells)

    def _format_batch_results_table(self, report: BatchValidationReport) -> str:
        """Format batch results as a table."""
        # Define columns
        columns = [
            ("INDEX", 7),
            ("ID", 20),
            ("STATUS", 8),
            ("ERRORS", 7),
            ("WARNINGS", 9),
        ]

        columns = self._adjust_column_widths(columns)

        lines: List[str] = []

        # Header row
        header_cells = [
            self._pad_cell(col[0], col[1])
            for col in columns
        ]
        header = self._colorize("  ".join(header_cells), "header")
        lines.append(header)

        # Separator
        separator_cells = [self.box["h"] * col[1] for col in columns]
        lines.append("  ".join(separator_cells))

        # Data rows (show failed items first, then summary)
        failed_items = report.failed_items()
        items_to_show = failed_items[:20]  # Limit to 20 rows

        for item in items_to_show:
            row = self._format_item_row(item, columns)
            lines.append(row)

        # Show count of remaining items
        remaining = len(report.results) - len(items_to_show)
        if remaining > 0:
            lines.append(
                self._colorize(
                    f"  ... and {remaining} more items ({report.summary.valid_count} valid)",
                    "dim"
                )
            )

        return "\n".join(lines)

    def _format_item_row(
        self,
        item: Any,
        columns: List[Tuple[str, int]]
    ) -> str:
        """Format a single batch item as a table row."""
        col_widths = {col[0]: col[1] for col in columns}

        # Index cell
        index_cell = self._pad_cell(str(item.index), col_widths.get("INDEX", 7))

        # ID cell
        id_text = item.id or "-"
        id_cell = self._truncate_cell(id_text, col_widths.get("ID", 20))

        # Status cell (colored)
        if item.valid:
            status_cell = self._colorize(
                self._pad_cell("VALID", col_widths.get("STATUS", 8)),
                "success"
            )
        else:
            status_cell = self._colorize(
                self._pad_cell("INVALID", col_widths.get("STATUS", 8)),
                "error"
            )

        # Error count cell
        error_count = item.error_count()
        errors_cell = self._pad_cell(str(error_count), col_widths.get("ERRORS", 7))
        if error_count > 0:
            errors_cell = self._colorize(errors_cell, "error")

        # Warning count cell
        warning_count = item.warning_count()
        warnings_cell = self._pad_cell(str(warning_count), col_widths.get("WARNINGS", 9))
        if warning_count > 0:
            warnings_cell = self._colorize(warnings_cell, "warning")

        return "  ".join([index_cell, id_cell, status_cell, errors_cell, warnings_cell])

    def _adjust_column_widths(
        self,
        columns: List[Tuple[str, int]]
    ) -> List[Tuple[str, int]]:
        """Adjust column widths to fit within max_width."""
        total_width = sum(col[1] for col in columns) + (len(columns) - 1) * 2

        if total_width <= self.max_width:
            return columns

        # Calculate how much we need to reduce
        excess = total_width - self.max_width

        # Reduce from largest columns first
        adjusted = list(columns)
        while excess > 0:
            # Find largest column (excluding small fixed columns)
            max_idx = -1
            max_width = 0
            for i, (name, width) in enumerate(adjusted):
                if width > 10 and width > max_width:
                    max_idx = i
                    max_width = width

            if max_idx == -1:
                break  # Cannot reduce further

            # Reduce by 5 or remaining excess
            reduction = min(5, excess, adjusted[max_idx][1] - 10)
            adjusted[max_idx] = (adjusted[max_idx][0], adjusted[max_idx][1] - reduction)
            excess -= reduction

        return adjusted

    def _pad_cell(self, text: str, width: int) -> str:
        """Pad text to fixed width."""
        if len(text) >= width:
            return text[:width]
        return text + " " * (width - len(text))

    def _truncate_cell(self, text: str, width: int) -> str:
        """Truncate text to fit width, adding ellipsis if needed."""
        if len(text) <= width:
            return self._pad_cell(text, width)
        if width <= 3:
            return text[:width]
        return text[:width - 3] + "..."

    def _colorize(
        self,
        text: str,
        color: str,
        bold: bool = False
    ) -> str:
        """Apply ANSI color to text if colors are enabled."""
        if not self.use_color:
            return text

        codes: List[str] = []
        if bold:
            codes.append(self.COLORS.get("bold", ""))
        codes.append(self.COLORS.get(color, ""))

        prefix = "".join(codes)
        suffix = self.COLORS.get("reset", "")

        return f"{prefix}{text}{suffix}"


class CSVFormatter:
    """
    CSV output formatter for validation findings.

    Produces comma-separated output suitable for spreadsheet import
    and data analysis tools.

    Example:
        >>> formatter = CSVFormatter()
        >>> output = formatter.format(report)
        >>> # Can be imported into Excel, Google Sheets, etc.
    """

    def __init__(
        self,
        delimiter: str = ",",
        include_header: bool = True,
        quote_char: str = '"',
    ) -> None:
        """
        Initialize CSVFormatter.

        Args:
            delimiter: Field delimiter character.
            include_header: Whether to include header row.
            quote_char: Character to use for quoting fields.
        """
        self.delimiter = delimiter
        self.include_header = include_header
        self.quote_char = quote_char

    def format(self, report: Union[ValidationReport, BatchValidationReport]) -> str:
        """
        Format validation report as CSV.

        Args:
            report: ValidationReport or BatchValidationReport to format.

        Returns:
            CSV string with findings data.
        """
        if isinstance(report, BatchValidationReport):
            return self._format_batch_report(report)
        return self._format_single_report(report)

    def _format_single_report(self, report: ValidationReport) -> str:
        """Format a single report as CSV."""
        lines: List[str] = []

        # Header row
        if self.include_header:
            headers = ["severity", "code", "path", "message", "expected", "actual"]
            lines.append(self._format_row(headers))

        # Data rows
        for finding in report.findings:
            row = [
                finding.severity.value,
                finding.code,
                finding.path,
                finding.message,
                str(finding.expected) if finding.expected else "",
                str(finding.actual) if finding.actual is not None else "",
            ]
            lines.append(self._format_row(row))

        return "\n".join(lines)

    def _format_batch_report(self, report: BatchValidationReport) -> str:
        """Format a batch report as CSV."""
        lines: List[str] = []

        # Header row
        if self.include_header:
            headers = ["index", "id", "valid", "severity", "code", "path", "message"]
            lines.append(self._format_row(headers))

        # Data rows
        for item in report.results:
            for finding in item.findings:
                row = [
                    str(item.index),
                    item.id or "",
                    str(item.valid).lower(),
                    finding.severity.value,
                    finding.code,
                    finding.path,
                    finding.message,
                ]
                lines.append(self._format_row(row))

        return "\n".join(lines)

    def _format_row(self, cells: List[str]) -> str:
        """Format a row of cells as CSV."""
        quoted_cells = [self._quote_cell(cell) for cell in cells]
        return self.delimiter.join(quoted_cells)

    def _quote_cell(self, cell: str) -> str:
        """Quote a cell if necessary."""
        needs_quote = (
            self.delimiter in cell or
            self.quote_char in cell or
            "\n" in cell or
            "\r" in cell
        )

        if needs_quote:
            # Escape quote characters by doubling them
            escaped = cell.replace(
                self.quote_char,
                self.quote_char + self.quote_char
            )
            return f"{self.quote_char}{escaped}{self.quote_char}"

        return cell


def format_table(
    report: Union[ValidationReport, BatchValidationReport],
    use_color: bool = True,
    max_width: int = 120,
    max_findings: int = 100,
) -> str:
    """
    Format validation report as a table.

    Convenience function for creating a TableFormatter and formatting
    a report in one call.

    Args:
        report: ValidationReport or BatchValidationReport to format.
        use_color: Whether to use ANSI color codes.
        max_width: Maximum table width in characters.
        max_findings: Maximum number of findings to display.

    Returns:
        Formatted table string.

    Example:
        >>> output = format_table(report, use_color=True)
        >>> print(output)
    """
    formatter = TableFormatter(use_color=use_color, max_width=max_width)
    return formatter.format(report)


__all__ = [
    "TableFormatter",
    "CSVFormatter",
    "format_table",
]
