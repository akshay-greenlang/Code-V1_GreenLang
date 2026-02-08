# -*- coding: utf-8 -*-
"""
Test Report Generation Engine for QA Test Harness - AGENT-FOUND-009

Provides report generation in multiple formats (text, JSON, markdown, HTML)
for test suite results and aggregated QA statistics.

Zero-Hallucination Guarantees:
    - All report content derived from deterministic data
    - No LLM calls for report generation
    - Format templates are static and verifiable
    - Complete provenance hash included in all reports

Example:
    >>> from greenlang.qa_test_harness.report_generator import ReportGenerator
    >>> generator = ReportGenerator(config)
    >>> report = generator.generate(suite_result, format="markdown")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-009 QA Test Harness
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from greenlang.qa_test_harness.config import QATestHarnessConfig
from greenlang.qa_test_harness.models import (
    TestSuiteResult,
    TestCaseResult,
    TestStatus,
    QAStatistics,
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Test report generation engine.

    Generates formatted test reports from suite results and statistics
    in text, JSON, markdown, and HTML formats.

    Attributes:
        config: QA test harness configuration.

    Example:
        >>> generator = ReportGenerator(config)
        >>> report = generator.generate(suite_result)
    """

    def __init__(self, config: QATestHarnessConfig) -> None:
        """Initialize ReportGenerator.

        Args:
            config: QA test harness configuration.
        """
        self.config = config
        logger.info(
            "ReportGenerator initialized: default_format=%s",
            config.report_format,
        )

    def generate(
        self,
        suite_result: TestSuiteResult,
        format: Optional[str] = None,
    ) -> str:
        """Generate a test report in the specified format.

        Args:
            suite_result: Test suite result to report on.
            format: Report format (text, json, markdown, html).
                Defaults to config report_format.

        Returns:
            Formatted report string.
        """
        fmt = (format or self.config.report_format).lower()

        if fmt == "json":
            return self.generate_json(suite_result)
        elif fmt == "html":
            return self.generate_html(suite_result)
        elif fmt == "text":
            return self.generate_text(suite_result)
        else:
            return self.generate_markdown(suite_result)

    def generate_text(self, suite_result: TestSuiteResult) -> str:
        """Generate a plain text test report.

        Args:
            suite_result: Test suite result.

        Returns:
            Plain text formatted report.
        """
        lines = [
            "=" * 80,
            f"TEST SUITE: {suite_result.name}",
            "=" * 80,
            f"Status: {self._format_status(suite_result.status)}",
            f"Duration: {self._format_duration(suite_result.duration_ms)}",
            f"Pass Rate: {suite_result.pass_rate}%",
            "",
            f"Total: {suite_result.total_tests}",
            f"Passed: {suite_result.passed}",
            f"Failed: {suite_result.failed}",
            f"Skipped: {suite_result.skipped}",
            f"Errors: {suite_result.errors}",
            "",
            "-" * 80,
            "TEST RESULTS:",
            "-" * 80,
        ]

        for test in suite_result.test_results:
            symbol = self._get_status_symbol(test.status)
            duration = self._format_duration(test.duration_ms)
            lines.append(f"{symbol} {test.name} ({duration})")

            if test.status == TestStatus.FAILED:
                for assertion in test.assertions:
                    if not assertion.passed:
                        lines.append(
                            f"       - {assertion.name}: {assertion.message}"
                        )

            if test.error_message:
                lines.append(f"       Error: {test.error_message}")

        lines.extend([
            "",
            "-" * 80,
            f"Provenance Hash: {suite_result.provenance_hash}",
            "=" * 80,
        ])

        return "\n".join(lines)

    def generate_json(self, suite_result: TestSuiteResult) -> str:
        """Generate a JSON test report.

        Args:
            suite_result: Test suite result.

        Returns:
            JSON formatted report string.
        """
        return json.dumps(
            suite_result.model_dump(), indent=2, default=str,
        )

    def generate_markdown(self, suite_result: TestSuiteResult) -> str:
        """Generate a markdown test report.

        Args:
            suite_result: Test suite result.

        Returns:
            Markdown formatted report string.
        """
        lines = [
            f"# Test Suite: {suite_result.name}",
            "",
            f"**Status:** {self._format_status(suite_result.status)}",
            f"**Duration:** {self._format_duration(suite_result.duration_ms)}",
            f"**Pass Rate:** {suite_result.pass_rate}%",
            "",
            "## Summary",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| Total | {suite_result.total_tests} |",
            f"| Passed | {suite_result.passed} |",
            f"| Failed | {suite_result.failed} |",
            f"| Skipped | {suite_result.skipped} |",
            f"| Errors | {suite_result.errors} |",
            "",
            "## Test Results",
            "",
            "| Status | Test | Category | Duration |",
            "|--------|------|----------|----------|",
        ]

        for test in suite_result.test_results:
            status_label = self._format_status(test.status)
            duration = self._format_duration(test.duration_ms)
            category = test.category.value if test.category else "unknown"
            lines.append(
                f"| {status_label} | {test.name} | {category} | {duration} |"
            )

        # Failed test details
        failed_tests = [
            t for t in suite_result.test_results
            if t.status in (TestStatus.FAILED, TestStatus.ERROR)
        ]
        if failed_tests:
            lines.extend([
                "",
                "## Failures",
                "",
            ])
            for test in failed_tests:
                lines.append(f"### {test.name}")
                lines.append("")
                if test.error_message:
                    lines.append(f"**Error:** {test.error_message}")
                    lines.append("")
                failed_assertions = [
                    a for a in test.assertions if not a.passed
                ]
                if failed_assertions:
                    lines.append("| Assertion | Expected | Actual | Severity |")
                    lines.append("|-----------|----------|--------|----------|")
                    for a in failed_assertions:
                        lines.append(
                            f"| {a.name} | {a.expected} | {a.actual} "
                            f"| {a.severity.value} |"
                        )
                    lines.append("")

        lines.extend([
            "",
            "## Provenance",
            "",
            f"Hash: `{suite_result.provenance_hash}`",
        ])

        return "\n".join(lines)

    def generate_html(self, suite_result: TestSuiteResult) -> str:
        """Generate an HTML test report.

        Args:
            suite_result: Test suite result.

        Returns:
            HTML formatted report string.
        """
        status_color = self._get_status_color(suite_result.status)

        rows = []
        for test in suite_result.test_results:
            color = self._get_status_color(test.status)
            duration = self._format_duration(test.duration_ms)
            error_info = ""
            if test.error_message:
                error_info = f"<br><small>{test.error_message}</small>"
            rows.append(
                f"<tr>"
                f"<td style='color:{color}'>{self._format_status(test.status)}</td>"
                f"<td>{test.name}{error_info}</td>"
                f"<td>{test.category.value if test.category else 'unknown'}</td>"
                f"<td>{duration}</td>"
                f"</tr>"
            )

        table_rows = "\n".join(rows)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Test Report: {suite_result.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ margin: 10px 0; }}
        .status {{ color: {status_color}; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .provenance {{ font-family: monospace; color: #666; margin-top: 20px; }}
    </style>
</head>
<body>
    <h1>Test Suite: {suite_result.name}</h1>
    <div class="summary">
        <p><strong>Status:</strong> <span class="status">{self._format_status(suite_result.status)}</span></p>
        <p><strong>Duration:</strong> {self._format_duration(suite_result.duration_ms)}</p>
        <p><strong>Pass Rate:</strong> {suite_result.pass_rate}%</p>
    </div>
    <h2>Summary</h2>
    <table>
        <tr><th>Metric</th><th>Count</th></tr>
        <tr><td>Total</td><td>{suite_result.total_tests}</td></tr>
        <tr><td>Passed</td><td>{suite_result.passed}</td></tr>
        <tr><td>Failed</td><td>{suite_result.failed}</td></tr>
        <tr><td>Skipped</td><td>{suite_result.skipped}</td></tr>
        <tr><td>Errors</td><td>{suite_result.errors}</td></tr>
    </table>
    <h2>Test Results</h2>
    <table>
        <tr><th>Status</th><th>Test</th><th>Category</th><th>Duration</th></tr>
        {table_rows}
    </table>
    <div class="provenance">
        <p>Provenance Hash: <code>{suite_result.provenance_hash}</code></p>
    </div>
</body>
</html>"""

        return html

    def generate_summary(self, statistics: QAStatistics) -> str:
        """Generate a summary report from aggregated statistics.

        Args:
            statistics: Aggregated QA statistics.

        Returns:
            Markdown formatted summary string.
        """
        pass_rate_label = f"{statistics.pass_rate:.1f}%"
        avg_duration = self._format_duration(statistics.avg_duration_ms)

        lines = [
            "# QA Test Harness Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Runs | {statistics.total_runs} |",
            f"| Passed | {statistics.passed} |",
            f"| Failed | {statistics.failed} |",
            f"| Skipped | {statistics.skipped} |",
            f"| Errors | {statistics.errors} |",
            f"| Pass Rate | {pass_rate_label} |",
            f"| Avg Duration | {avg_duration} |",
            f"| Regressions | {statistics.regressions_detected} |",
            f"| Golden File Mismatches | {statistics.golden_file_mismatches} |",
            f"| Coverage | {statistics.coverage_percent:.1f}% |",
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_duration(self, ms: float) -> str:
        """Format a duration in milliseconds to a human-readable string.

        Args:
            ms: Duration in milliseconds.

        Returns:
            Formatted duration string.
        """
        if ms < 1.0:
            return f"{ms * 1000:.0f}us"
        elif ms < 1000.0:
            return f"{ms:.2f}ms"
        else:
            return f"{ms / 1000:.2f}s"

    def _format_status(self, status: TestStatus) -> str:
        """Format a test status to a display string.

        Args:
            status: Test status enum value.

        Returns:
            Uppercase status string.
        """
        return status.value.upper()

    def _get_status_symbol(self, status: TestStatus) -> str:
        """Get a text symbol for a test status.

        Args:
            status: Test status enum value.

        Returns:
            Status symbol string (e.g. "[PASS]").
        """
        symbols = {
            TestStatus.PASSED: "[PASS]",
            TestStatus.FAILED: "[FAIL]",
            TestStatus.SKIPPED: "[SKIP]",
            TestStatus.ERROR: "[ERR ]",
            TestStatus.TIMEOUT: "[TIME]",
            TestStatus.RUNNING: "[RUN ]",
            TestStatus.PENDING: "[PEND]",
        }
        return symbols.get(status, "[????]")

    def _get_status_color(self, status: TestStatus) -> str:
        """Get an HTML color for a test status.

        Args:
            status: Test status enum value.

        Returns:
            CSS color string.
        """
        colors = {
            TestStatus.PASSED: "#28a745",
            TestStatus.FAILED: "#dc3545",
            TestStatus.SKIPPED: "#6c757d",
            TestStatus.ERROR: "#dc3545",
            TestStatus.TIMEOUT: "#fd7e14",
            TestStatus.RUNNING: "#007bff",
            TestStatus.PENDING: "#6c757d",
        }
        return colors.get(status, "#333333")


__all__ = [
    "ReportGenerator",
]
