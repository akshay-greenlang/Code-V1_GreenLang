# -*- coding: utf-8 -*-
"""
Plain Text Output Formatter for GL-FOUND-X-002.

This module provides a plain text output formatter for validation reports.
Unlike the pretty formatter, this produces output without ANSI color codes,
making it suitable for log files, CI systems without color support, and
piped output.

Example:
    >>> from greenlang.schema.cli.formatters.text import TextFormatter
    >>> formatter = TextFormatter(verbosity=1)
    >>> output = formatter.format(validation_report)
    >>> print(output)

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

from greenlang.schema.models.report import ValidationReport, BatchValidationReport
from greenlang.schema.models.finding import Finding, Severity


class TextFormatter:
    """
    Plain text output formatter for validation reports.

    Produces clean, uncolored text output suitable for log files,
    CI systems, and piped output.

    Attributes:
        verbosity: Verbosity level (0=summary, 1=all findings, 2=detailed).
        max_findings: Maximum findings to show at verbosity 0.
        include_hints: Whether to include hints in output.

    Example:
        >>> formatter = TextFormatter(verbosity=1)
        >>> print(formatter.format(report))
        INVALID  emissions/activity@1.3.0  (3 errors, 1 warning)
        ...
    """

    def __init__(
        self,
        verbosity: int = 0,
        max_findings: int = 5,
        include_hints: bool = False,
    ) -> None:
        """
        Initialize TextFormatter.

        Args:
            verbosity: Output verbosity level:
                - 0: Summary + first max_findings errors
                - 1: All findings
                - 2: Detailed findings with hints
            max_findings: Maximum findings to show at verbosity 0.
            include_hints: Whether to include hints in output.
        """
        self.verbosity = verbosity
        self.max_findings = max_findings
        self.include_hints = include_hints

    def format(self, report: Union[ValidationReport, BatchValidationReport]) -> str:
        """
        Format validation report as plain text.

        Args:
            report: ValidationReport or BatchValidationReport to format.

        Returns:
            Plain text string.

        Example:
            >>> output = formatter.format(report)
            >>> print(output)
        """
        if isinstance(report, BatchValidationReport):
            return self._format_batch_report(report)
        return self._format_single_report(report)

    def _format_single_report(self, report: ValidationReport) -> str:
        """Format a single validation report."""
        lines: List[str] = []

        # Status header
        lines.append(self._format_header(report))
        lines.append("")

        # Findings
        if report.findings:
            lines.extend(self._format_findings(report))
        else:
            lines.append("No findings.")

        # Summary footer
        lines.append("")
        lines.append(f"Completed in {report.timings.total_ms:.1f}ms")

        return "\n".join(lines)

    def _format_batch_report(self, report: BatchValidationReport) -> str:
        """Format a batch validation report."""
        lines: List[str] = []

        # Header
        status = "ALL VALID" if report.summary.error_count == 0 else "ERRORS FOUND"
        lines.append(
            f"{status}  {report.schema_ref}  "
            f"({report.summary.total_items} items: "
            f"{report.summary.valid_count} valid, {report.summary.error_count} with errors)"
        )
        lines.append("")

        # Failed items
        failed_items = report.failed_items()
        items_to_show = failed_items[:self.max_findings] if self.verbosity == 0 else failed_items

        for item in items_to_show:
            item_id = item.id or f"Item #{item.index}"
            lines.append(f"  INVALID {item_id} ({item.error_count()} errors)")
            for finding in item.findings[:3] if self.verbosity == 0 else item.findings:
                lines.append(f"    {finding.code} at {finding.path}: {finding.message}")

        # Show count of remaining items
        remaining = len(failed_items) - len(items_to_show)
        if remaining > 0:
            lines.append("")
            lines.append(f"  ... and {remaining} more failed items")

        # Summary
        lines.append("")
        lines.append(
            f"Success rate: {report.summary.success_rate():.1f}% "
            f"({report.summary.valid_count}/{report.summary.total_items})"
        )

        return "\n".join(lines)

    def _format_header(self, report: ValidationReport) -> str:
        """Format the status header."""
        status = "VALID" if report.valid else "INVALID"

        counts: List[str] = []
        if report.summary.error_count > 0:
            counts.append(f"{report.summary.error_count} error(s)")
        if report.summary.warning_count > 0:
            counts.append(f"{report.summary.warning_count} warning(s)")
        if report.summary.info_count > 0 and self.verbosity >= 1:
            counts.append(f"{report.summary.info_count} info")

        counts_str = ", ".join(counts) if counts else "no issues"

        return f"{status}  {report.schema_ref}  ({counts_str})"

    def _format_findings(self, report: ValidationReport) -> List[str]:
        """Format findings as text lines."""
        lines: List[str] = []

        # Determine which findings to show
        findings_to_show = report.findings
        if self.verbosity == 0:
            errors = [f for f in report.findings if f.is_error()]
            findings_to_show = errors[:self.max_findings]

        for finding in findings_to_show:
            lines.extend(self._format_finding(finding))
            lines.append("")  # Empty line between findings

        # Show count of hidden findings
        total_errors = report.summary.error_count
        shown_errors = sum(1 for f in findings_to_show if f.is_error())

        if total_errors > shown_errors:
            remaining = total_errors - shown_errors
            lines.append(f"... and {remaining} more error(s)")
            lines.append("")

        return lines

    def _format_finding(self, finding: Finding) -> List[str]:
        """Format a single finding as text lines."""
        lines: List[str] = []

        # First line: severity, code, path
        severity_str = finding.severity.value.upper()
        path_str = finding.path or "(root)"

        lines.append(f"{severity_str} {finding.code} at {path_str}")
        lines.append(f"  {finding.message}")

        # Expected/actual values
        if self.verbosity >= 1:
            if finding.expected is not None:
                lines.append(f"  Expected: {finding.expected}")
            if finding.actual is not None:
                lines.append(f"  Actual: {finding.actual}")

        # Hints
        if self.include_hints and finding.hint and self.verbosity >= 2:
            if finding.hint.suggested_values:
                suggestions = ", ".join(str(v) for v in finding.hint.suggested_values[:5])
                lines.append(f"  Suggestions: {suggestions}")
            if finding.hint.docs_url:
                lines.append(f"  Docs: {finding.hint.docs_url}")

        return lines


def format_text(
    report: Union[ValidationReport, BatchValidationReport],
    verbosity: int = 0,
    max_findings: int = 5,
) -> str:
    """
    Format validation report as plain text.

    Convenience function for creating a TextFormatter and formatting
    a report in one call.

    Args:
        report: ValidationReport or BatchValidationReport to format.
        verbosity: Output verbosity level (0-2).
        max_findings: Maximum findings to show at verbosity 0.

    Returns:
        Plain text string without color codes.

    Example:
        >>> output = format_text(report, verbosity=1)
        >>> print(output)
    """
    formatter = TextFormatter(verbosity=verbosity, max_findings=max_findings)
    return formatter.format(report)


__all__ = [
    "TextFormatter",
    "format_text",
]
