# -*- coding: utf-8 -*-
"""
Pretty (Colorized) Output Formatter for GL-FOUND-X-002.

This module provides a colorized terminal output formatter for validation
reports. It supports ANSI colors for enhanced readability and includes
context-aware formatting for different finding types.

Features:
- Colorized output based on severity (red=error, yellow=warning, blue=info)
- Contextual information display (expected vs actual values)
- Summary header with status and counts
- Verbosity levels for controlling detail
- Color disable option for pipe/file output

Example:
    >>> from greenlang.schema.cli.formatters.pretty import PrettyFormatter
    >>> formatter = PrettyFormatter(use_color=True, verbosity=1)
    >>> output = formatter.format(validation_report)
    >>> print(output)

Author: GreenLang Framework Team
Version: 0.1.0
GL-FOUND-X-002: Schema Compiler & Validator
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional, Union

from greenlang.schema.models.report import ValidationReport, BatchValidationReport
from greenlang.schema.models.finding import Finding, Severity


class PrettyFormatter:
    """
    Colorized terminal output formatter.

    Produces human-readable, colorized output for validation reports
    suitable for terminal display.

    Attributes:
        use_color: Whether to use ANSI color codes.
        verbosity: Verbosity level (0=summary, 1=all findings, 2=detailed).
        max_findings: Maximum findings to show at verbosity 0.

    Example:
        >>> formatter = PrettyFormatter(use_color=True, verbosity=1)
        >>> print(formatter.format(report))
        INVALID  emissions/activity@1.3.0  (3 errors, 1 warning)
        ...
    """

    # ANSI color codes for terminal output
    COLORS: Dict[str, str] = {
        "error": "\033[91m",      # Bright red
        "warning": "\033[93m",    # Bright yellow
        "info": "\033[94m",       # Bright blue
        "success": "\033[92m",    # Bright green
        "reset": "\033[0m",       # Reset all formatting
        "bold": "\033[1m",        # Bold text
        "dim": "\033[2m",         # Dim text
        "underline": "\033[4m",   # Underlined text
        "cyan": "\033[96m",       # Bright cyan for paths
        "magenta": "\033[95m",    # Bright magenta for codes
    }

    # Severity to color mapping
    SEVERITY_COLORS: Dict[Severity, str] = {
        Severity.ERROR: "error",
        Severity.WARNING: "warning",
        Severity.INFO: "info",
    }

    def __init__(
        self,
        use_color: bool = True,
        verbosity: int = 0,
        max_findings: int = 5,
        output_stream: Any = None,
    ) -> None:
        """
        Initialize PrettyFormatter.

        Args:
            use_color: Whether to use ANSI color codes. Auto-detects
                if output is a TTY when not specified.
            verbosity: Output verbosity level:
                - 0: Summary + first max_findings errors
                - 1: All findings
                - 2: Detailed findings with hints and suggestions
            max_findings: Maximum findings to show at verbosity 0.
            output_stream: Output stream for TTY detection (default: stdout).
        """
        self.verbosity = verbosity
        self.max_findings = max_findings

        # Auto-detect color support if output is a TTY
        stream = output_stream or sys.stdout
        if use_color and hasattr(stream, 'isatty'):
            self.use_color = stream.isatty()
        else:
            self.use_color = use_color

    def format(self, report: Union[ValidationReport, BatchValidationReport]) -> str:
        """
        Format validation report for terminal display.

        Args:
            report: ValidationReport or BatchValidationReport to format.

        Returns:
            Formatted string with optional ANSI color codes.

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

        # Header with status
        lines.append(self.format_header(report))
        lines.append("")  # Empty line after header

        # Findings
        findings_output = self._format_findings(report)
        if findings_output:
            lines.append(findings_output)

        # Summary footer
        lines.append(self.format_summary(report))

        return "\n".join(lines)

    def _format_batch_report(self, report: BatchValidationReport) -> str:
        """Format a batch validation report."""
        lines: List[str] = []

        # Batch header
        status_color = "success" if report.summary.error_count == 0 else "error"
        status_text = "ALL VALID" if report.summary.error_count == 0 else "ERRORS FOUND"

        lines.append(
            f"{self._colorize(status_text, status_color, bold=True)}  "
            f"{self._colorize(str(report.schema_ref), 'cyan')}  "
            f"({report.summary.total_items} items: "
            f"{report.summary.valid_count} valid, {report.summary.error_count} with errors)"
        )
        lines.append("")

        # Show failed items
        failed_items = report.failed_items()
        items_to_show = failed_items
        if self.verbosity == 0:
            items_to_show = failed_items[:self.max_findings]

        for item in items_to_show:
            item_id = item.id or f"Item #{item.index}"
            lines.append(
                f"  {self._colorize('INVALID', 'error')} {item_id}"
            )
            for finding in item.findings[:3] if self.verbosity == 0 else item.findings:
                lines.append(f"    {self._format_finding_brief(finding)}")

        # Show truncation notice
        remaining = len(failed_items) - len(items_to_show)
        if remaining > 0:
            lines.append("")
            lines.append(
                self._colorize(f"  ... and {remaining} more failed items", "dim")
            )

        # Footer
        lines.append("")
        lines.append(
            f"Success rate: {report.summary.success_rate():.1f}% "
            f"({report.summary.valid_count}/{report.summary.total_items})"
        )

        return "\n".join(lines)

    def format_header(self, report: ValidationReport) -> str:
        """
        Format status header with validation status and counts.

        Args:
            report: ValidationReport to format header for.

        Returns:
            Formatted header string.

        Example:
            >>> header = formatter.format_header(report)
            >>> print(header)
            INVALID  emissions/activity@1.3.0  (3 errors, 1 warning)
        """
        # Determine status
        if report.valid:
            status = self._colorize("VALID", "success", bold=True)
        else:
            status = self._colorize("INVALID", "error", bold=True)

        # Schema reference
        schema_ref_str = self._colorize(str(report.schema_ref), "cyan")

        # Counts
        counts: List[str] = []
        if report.summary.error_count > 0:
            count_str = f"{report.summary.error_count} error"
            if report.summary.error_count != 1:
                count_str += "s"
            counts.append(self._colorize(count_str, "error"))

        if report.summary.warning_count > 0:
            count_str = f"{report.summary.warning_count} warning"
            if report.summary.warning_count != 1:
                count_str += "s"
            counts.append(self._colorize(count_str, "warning"))

        if report.summary.info_count > 0 and self.verbosity >= 1:
            count_str = f"{report.summary.info_count} info"
            counts.append(self._colorize(count_str, "info"))

        counts_str = ", ".join(counts) if counts else "no issues"

        return f"{status}  {schema_ref_str}  ({counts_str})"

    def format_finding(self, finding: Finding) -> str:
        """
        Format a single finding with color and context.

        Args:
            finding: Finding to format.

        Returns:
            Multi-line formatted finding string.

        Example:
            >>> output = formatter.format_finding(finding)
            >>> print(output)
            ERROR GLSCHEMA-E100 at /energy_consumption
              Missing required field
              Expected: { value: number, unit: string }
        """
        lines: List[str] = []

        # First line: severity, code, path
        severity_str = finding.severity.value.upper()
        severity_color = self.SEVERITY_COLORS.get(finding.severity, "reset")

        code_str = self._colorize(finding.code, "magenta")
        path_str = self._colorize(finding.path or "(root)", "cyan")

        lines.append(
            f"{self._colorize(severity_str, severity_color, bold=True)} "
            f"{code_str} at {path_str}"
        )

        # Message (indented)
        lines.append(f"  {finding.message}")

        # Expected value (if available and verbosity allows)
        if finding.expected and self.verbosity >= 1:
            expected_str = self._format_value(finding.expected)
            lines.append(f"  {self._colorize('Expected:', 'dim')} {expected_str}")

        # Actual value (if available and verbosity allows)
        if finding.actual is not None and self.verbosity >= 1:
            actual_str = self._format_value(finding.actual)
            lines.append(f"  {self._colorize('Actual:', 'dim')} {actual_str}")

        # Hints (if available and verbosity is high)
        if finding.hint and self.verbosity >= 2:
            if finding.hint.suggested_values:
                suggestions = ", ".join(
                    str(v) for v in finding.hint.suggested_values[:5]
                )
                lines.append(
                    f"  {self._colorize('Suggestions:', 'dim')} {suggestions}"
                )
            if finding.hint.docs_url:
                lines.append(
                    f"  {self._colorize('Docs:', 'dim')} {finding.hint.docs_url}"
                )

        return "\n".join(lines)

    def _format_finding_brief(self, finding: Finding) -> str:
        """Format a finding as a brief one-liner."""
        severity_str = finding.severity.value.upper()
        severity_color = self.SEVERITY_COLORS.get(finding.severity, "reset")

        return (
            f"{self._colorize(severity_str, severity_color)} "
            f"{self._colorize(finding.code, 'magenta')} "
            f"at {self._colorize(finding.path or '(root)', 'cyan')}: "
            f"{finding.message[:60]}{'...' if len(finding.message) > 60 else ''}"
        )

    def _format_findings(self, report: ValidationReport) -> str:
        """Format all findings based on verbosity."""
        if not report.findings:
            return ""

        lines: List[str] = []

        # Determine which findings to show
        findings_to_show = report.findings
        if self.verbosity == 0:
            # At verbosity 0, show only errors up to max_findings
            errors = [f for f in report.findings if f.is_error()]
            findings_to_show = errors[:self.max_findings]

        for finding in findings_to_show:
            lines.append(self.format_finding(finding))
            lines.append("")  # Empty line between findings

        # Show count of hidden findings
        total_errors = report.summary.error_count
        shown_errors = sum(1 for f in findings_to_show if f.is_error())

        if total_errors > shown_errors:
            remaining = total_errors - shown_errors
            lines.append(
                self._colorize(
                    f"... and {remaining} more error{'s' if remaining != 1 else ''}",
                    "dim"
                )
            )
            lines.append("")

        return "\n".join(lines)

    def format_summary(self, report: ValidationReport) -> str:
        """
        Format summary footer with hints for next steps.

        Args:
            report: ValidationReport to format summary for.

        Returns:
            Formatted summary string.

        Example:
            >>> summary = formatter.format_summary(report)
            >>> print(summary)
            Run with -v to see all findings
            Use --format json for machine output
        """
        lines: List[str] = []

        # Timing info
        timing_str = f"[{report.timings.total_ms:.1f}ms]"
        lines.append(self._colorize(timing_str, "dim"))

        # Hints based on current state
        if self.verbosity == 0 and report.summary.total_findings() > self.max_findings:
            lines.append(
                self._colorize("Run with -v to see all findings", "dim")
            )

        if not report.valid:
            lines.append(
                self._colorize("Use --format json for machine output", "dim")
            )

        # Fix suggestions hint
        if report.fix_suggestions and report.has_safe_fixes():
            safe_count = sum(1 for s in report.fix_suggestions if s.is_safe())
            lines.append(
                self._colorize(
                    f"{safe_count} safe fix suggestion{'s' if safe_count != 1 else ''} available",
                    "info"
                )
            )

        return "\n".join(lines)

    def _colorize(
        self,
        text: str,
        color: str,
        bold: bool = False
    ) -> str:
        """
        Apply ANSI color to text if colors are enabled.

        Args:
            text: Text to colorize.
            color: Color name from COLORS dict.
            bold: Whether to also apply bold formatting.

        Returns:
            Colorized string or original text if colors disabled.
        """
        if not self.use_color:
            return text

        codes: List[str] = []
        if bold:
            codes.append(self.COLORS.get("bold", ""))
        codes.append(self.COLORS.get(color, ""))

        prefix = "".join(codes)
        suffix = self.COLORS.get("reset", "")

        return f"{prefix}{text}{suffix}"

    def _format_value(self, value: Any) -> str:
        """
        Format a value for display, handling complex types.

        Args:
            value: Value to format.

        Returns:
            Formatted string representation.
        """
        if isinstance(value, dict):
            # Compact JSON for small dicts
            try:
                json_str = json.dumps(value, ensure_ascii=False)
                if len(json_str) <= 80:
                    return json_str
                # Pretty print for larger dicts
                return json.dumps(value, indent=2, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(value)
        elif isinstance(value, list):
            try:
                json_str = json.dumps(value, ensure_ascii=False)
                if len(json_str) <= 60:
                    return json_str
                return f"[{len(value)} items]"
            except (TypeError, ValueError):
                return str(value)
        elif isinstance(value, str):
            if len(value) > 60:
                return f"'{value[:57]}...'"
            return f"'{value}'"
        else:
            return str(value)


def format_pretty(
    report: Union[ValidationReport, BatchValidationReport],
    use_color: bool = True,
    verbosity: int = 0,
    max_findings: int = 5,
    show_hints: bool = False,
) -> str:
    """
    Format validation report with colors.

    Convenience function for creating a PrettyFormatter and formatting
    a report in one call.

    Args:
        report: ValidationReport or BatchValidationReport to format.
        use_color: Whether to use ANSI color codes.
        verbosity: Output verbosity level (0-2).
        max_findings: Maximum findings to show at verbosity 0.
        show_hints: Whether to show detailed hints (sets verbosity=2 if True).

    Returns:
        Formatted string with optional ANSI color codes.

    Example:
        >>> output = format_pretty(report, use_color=True, verbosity=1)
        >>> print(output)
    """
    # If show_hints is True, increase verbosity to show hints
    effective_verbosity = 2 if show_hints else verbosity
    formatter = PrettyFormatter(
        use_color=use_color,
        verbosity=effective_verbosity,
        max_findings=max_findings,
    )
    return formatter.format(report)


__all__ = [
    "PrettyFormatter",
    "format_pretty",
]
