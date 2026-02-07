# -*- coding: utf-8 -*-
"""
Test Reporter - SEC-009 Phase 4

Report generation for SOC 2 control testing results. Provides multiple output
formats including text summaries, detailed reports, exception reports, and
PDF exports for auditor review.

Features:
    - Executive summary with pass/fail counts and compliance rates
    - Detailed reports with evidence and exception documentation
    - Exception-only reports for remediation tracking
    - PDF export for formal auditor deliverables
    - Markdown and JSON formats for integration

Example:
    >>> reporter = TestReporter()
    >>> summary = reporter.generate_summary(results)
    >>> detailed = reporter.generate_detailed_report(results)
    >>> pdf_bytes = reporter.export_for_auditor(results)
"""

from __future__ import annotations

import base64
import io
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.infrastructure.soc2_preparation.control_testing.test_framework import (
    Evidence,
    Severity,
    TestResult,
    TestRun,
    TestStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report Configuration
# ---------------------------------------------------------------------------


class ReportConfig(BaseModel):
    """Configuration for report generation.

    Attributes:
        company_name: Company name for report header.
        report_title: Title for the report.
        include_evidence: Whether to include evidence details.
        include_exceptions: Whether to include exception details.
        include_passed_tests: Whether to include passed test details.
        max_evidence_length: Maximum length for inline evidence.
    """

    company_name: str = Field(default="GreenLang Inc.")
    report_title: str = Field(default="SOC 2 Type II Control Testing Report")
    include_evidence: bool = Field(default=True)
    include_exceptions: bool = Field(default=True)
    include_passed_tests: bool = Field(default=False)
    max_evidence_length: int = Field(default=2000)


# ---------------------------------------------------------------------------
# Report Data Models
# ---------------------------------------------------------------------------


class TestSummary(BaseModel):
    """Summary statistics for test results.

    Attributes:
        total_tests: Total number of tests executed.
        passed: Number of passed tests.
        failed: Number of failed tests.
        errors: Number of tests with errors.
        skipped: Number of skipped tests.
        pass_rate: Percentage of tests that passed.
        by_criterion: Breakdown by SOC 2 criterion.
        by_severity: Breakdown of failures by severity.
    """

    total_tests: int = Field(default=0)
    passed: int = Field(default=0)
    failed: int = Field(default=0)
    errors: int = Field(default=0)
    skipped: int = Field(default=0)
    pass_rate: float = Field(default=0.0)
    by_criterion: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    by_severity: Dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Test Reporter
# ---------------------------------------------------------------------------


class TestReporter:
    """Generate reports from SOC 2 control testing results.

    Provides multiple output formats for different audiences:
    - Summary: High-level executive overview
    - Detailed: Complete results with evidence for auditors
    - Exceptions: Focus on failures for remediation teams
    - PDF: Formal deliverable for audit evidence

    Attributes:
        _config: Report configuration settings.
    """

    def __init__(self, config: Optional[ReportConfig] = None) -> None:
        """Initialize the test reporter.

        Args:
            config: Optional report configuration.
        """
        self._config = config or ReportConfig()
        logger.debug("TestReporter initialized")

    # ------------------------------------------------------------------
    # Summary Generation
    # ------------------------------------------------------------------

    def generate_summary(self, results: List[TestResult]) -> str:
        """Generate an executive summary of test results.

        Args:
            results: List of test results to summarize.

        Returns:
            Formatted summary string.
        """
        summary = self._calculate_summary(results)
        report_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines = [
            "=" * 70,
            f"  {self._config.report_title}",
            f"  {self._config.company_name}",
            f"  Generated: {report_time}",
            "=" * 70,
            "",
            "EXECUTIVE SUMMARY",
            "-" * 70,
            "",
            f"  Total Tests Executed: {summary.total_tests}",
            f"  Passed:               {summary.passed}",
            f"  Failed:               {summary.failed}",
            f"  Errors:               {summary.errors}",
            f"  Skipped:              {summary.skipped}",
            "",
            f"  Overall Pass Rate:    {summary.pass_rate:.1f}%",
            "",
        ]

        # Compliance status
        if summary.pass_rate == 100.0:
            lines.append("  STATUS: COMPLIANT - All controls operating effectively")
        elif summary.pass_rate >= 95.0:
            lines.append("  STATUS: SUBSTANTIALLY COMPLIANT - Minor findings to address")
        elif summary.pass_rate >= 80.0:
            lines.append("  STATUS: PARTIALLY COMPLIANT - Multiple findings require remediation")
        else:
            lines.append("  STATUS: NON-COMPLIANT - Significant control deficiencies")

        lines.append("")

        # By criterion breakdown
        if summary.by_criterion:
            lines.append("RESULTS BY CRITERION")
            lines.append("-" * 70)
            lines.append("")
            lines.append("  Criterion    Passed    Failed    Errors    Pass Rate")
            lines.append("  " + "-" * 55)

            for criterion, counts in sorted(summary.by_criterion.items()):
                passed = counts.get("passed", 0)
                failed = counts.get("failed", 0)
                errors = counts.get("errors", 0)
                total = passed + failed + errors
                rate = (passed / total * 100) if total > 0 else 0
                lines.append(
                    f"  {criterion:12} {passed:8} {failed:9} {errors:9} {rate:8.1f}%"
                )

            lines.append("")

        # By severity breakdown
        if summary.by_severity:
            lines.append("FINDINGS BY SEVERITY")
            lines.append("-" * 70)
            lines.append("")

            severity_order = ["critical", "high", "medium", "low"]
            for sev in severity_order:
                count = summary.by_severity.get(sev, 0)
                if count > 0:
                    lines.append(f"  {sev.upper():12} {count}")

            lines.append("")

        lines.append("=" * 70)
        lines.append("  End of Summary Report")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _calculate_summary(self, results: List[TestResult]) -> TestSummary:
        """Calculate summary statistics from results.

        Args:
            results: List of test results.

        Returns:
            TestSummary with calculated statistics.
        """
        summary = TestSummary()
        summary.total_tests = len(results)

        for result in results:
            # Count by status
            if result.status == TestStatus.PASSED:
                summary.passed += 1
            elif result.status == TestStatus.FAILED:
                summary.failed += 1
            elif result.status == TestStatus.ERROR:
                summary.errors += 1
            elif result.status == TestStatus.SKIPPED:
                summary.skipped += 1

            # Count by criterion
            criterion = result.test_id.split(".")[0]
            if criterion not in summary.by_criterion:
                summary.by_criterion[criterion] = {"passed": 0, "failed": 0, "errors": 0}

            if result.status == TestStatus.PASSED:
                summary.by_criterion[criterion]["passed"] += 1
            elif result.status == TestStatus.FAILED:
                summary.by_criterion[criterion]["failed"] += 1
            elif result.status == TestStatus.ERROR:
                summary.by_criterion[criterion]["errors"] += 1

            # Count by severity
            if result.status == TestStatus.FAILED and result.severity:
                sev = result.severity.value
                summary.by_severity[sev] = summary.by_severity.get(sev, 0) + 1

        # Calculate pass rate
        executed = summary.passed + summary.failed
        summary.pass_rate = (summary.passed / executed * 100) if executed > 0 else 0

        return summary

    # ------------------------------------------------------------------
    # Detailed Report Generation
    # ------------------------------------------------------------------

    def generate_detailed_report(self, results: List[TestResult]) -> str:
        """Generate a detailed report with all test results and evidence.

        Args:
            results: List of test results.

        Returns:
            Detailed report as formatted string.
        """
        report_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines = [
            "=" * 80,
            f"  {self._config.report_title} - DETAILED",
            f"  {self._config.company_name}",
            f"  Generated: {report_time}",
            "=" * 80,
            "",
        ]

        # Add summary first
        summary = self._calculate_summary(results)
        lines.append(f"Total: {summary.total_tests} | Passed: {summary.passed} | "
                     f"Failed: {summary.failed} | Errors: {summary.errors}")
        lines.append(f"Pass Rate: {summary.pass_rate:.1f}%")
        lines.append("")
        lines.append("-" * 80)
        lines.append("")

        # Group by criterion
        by_criterion: Dict[str, List[TestResult]] = {}
        for result in results:
            criterion = result.test_id.split(".")[0]
            if criterion not in by_criterion:
                by_criterion[criterion] = []
            by_criterion[criterion].append(result)

        # Generate detail for each criterion
        for criterion in sorted(by_criterion.keys()):
            criterion_results = by_criterion[criterion]
            lines.append(f"CRITERION: {criterion}")
            lines.append("=" * 80)
            lines.append("")

            for result in criterion_results:
                # Skip passed tests if configured
                if result.status == TestStatus.PASSED and not self._config.include_passed_tests:
                    continue

                lines.extend(self._format_test_result(result))
                lines.append("")

        lines.append("=" * 80)
        lines.append("  End of Detailed Report")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _format_test_result(self, result: TestResult) -> List[str]:
        """Format a single test result for the detailed report.

        Args:
            result: Test result to format.

        Returns:
            List of formatted lines.
        """
        lines = [
            f"Test ID: {result.test_id}",
            "-" * 40,
            f"  Status:     {result.status.value.upper()}",
        ]

        if result.severity:
            lines.append(f"  Severity:   {result.severity.value.upper()}")

        if result.actual_result:
            lines.append(f"  Result:     {result.actual_result}")

        if result.duration_ms > 0:
            lines.append(f"  Duration:   {result.duration_ms}ms")

        if result.executed_by:
            lines.append(f"  Executed:   {result.executed_by}")

        if result.started_at and result.completed_at:
            lines.append(f"  Started:    {result.started_at.isoformat()}")
            lines.append(f"  Completed:  {result.completed_at.isoformat()}")

        # Include exceptions
        if self._config.include_exceptions and result.exceptions:
            lines.append("")
            lines.append("  Exceptions/Findings:")
            for i, exc in enumerate(result.exceptions, 1):
                lines.append(f"    {i}. {exc}")

        # Include evidence
        if self._config.include_evidence and result.evidence:
            lines.append("")
            lines.append("  Evidence Collected:")
            for evidence in result.evidence:
                lines.append(f"    - Type: {evidence.evidence_type}")
                lines.append(f"      Description: {evidence.description}")
                lines.append(f"      Hash: {evidence.hash[:16]}...")
                if len(evidence.content) <= self._config.max_evidence_length:
                    # Truncate and indent content
                    content_lines = evidence.content.split("\n")[:10]
                    for line in content_lines:
                        lines.append(f"        {line[:100]}")
                else:
                    lines.append(f"        [Content truncated - {len(evidence.content)} chars]")

        if result.error_message:
            lines.append("")
            lines.append(f"  Error: {result.error_message}")

        if result.notes:
            lines.append("")
            lines.append(f"  Notes: {result.notes}")

        return lines

    # ------------------------------------------------------------------
    # Exceptions Report
    # ------------------------------------------------------------------

    def generate_exceptions_report(self, results: List[TestResult]) -> str:
        """Generate a report focused on test failures and exceptions.

        Args:
            results: List of test results.

        Returns:
            Exceptions report as formatted string.
        """
        report_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Filter to failed/error results only
        failed_results = [
            r for r in results
            if r.status in (TestStatus.FAILED, TestStatus.ERROR)
        ]

        lines = [
            "=" * 80,
            f"  SOC 2 CONTROL TESTING - EXCEPTIONS REPORT",
            f"  {self._config.company_name}",
            f"  Generated: {report_time}",
            "=" * 80,
            "",
            f"Total Exceptions: {len(failed_results)}",
            "",
        ]

        # Group by severity
        by_severity: Dict[str, List[TestResult]] = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "unknown": [],
        }

        for result in failed_results:
            if result.severity:
                by_severity[result.severity.value].append(result)
            else:
                by_severity["unknown"].append(result)

        # Report by severity (highest first)
        for severity in ["critical", "high", "medium", "low", "unknown"]:
            severity_results = by_severity[severity]
            if not severity_results:
                continue

            lines.append("-" * 80)
            lines.append(f"SEVERITY: {severity.upper()} ({len(severity_results)} findings)")
            lines.append("-" * 80)
            lines.append("")

            for result in severity_results:
                lines.append(f"  [{result.test_id}] {result.status.value.upper()}")
                lines.append(f"  Result: {result.actual_result}")

                if result.exceptions:
                    lines.append("  Exceptions:")
                    for exc in result.exceptions:
                        lines.append(f"    - {exc}")

                if result.error_message:
                    lines.append(f"  Error: {result.error_message}")

                lines.append("")

        # Remediation guidance
        lines.append("=" * 80)
        lines.append("REMEDIATION GUIDANCE")
        lines.append("-" * 80)
        lines.append("")
        lines.append("Critical findings must be remediated within 30 days.")
        lines.append("High findings must be remediated within 60 days.")
        lines.append("Medium findings must be remediated within 90 days.")
        lines.append("Low findings should be remediated within 120 days.")
        lines.append("")
        lines.append("=" * 80)
        lines.append("  End of Exceptions Report")
        lines.append("=" * 80)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # PDF Export
    # ------------------------------------------------------------------

    def export_for_auditor(self, results: List[TestResult]) -> bytes:
        """Export test results as PDF for auditor review.

        Generates a formatted PDF document suitable for audit evidence.
        Uses a simple text-to-PDF conversion if reportlab is not available.

        Args:
            results: List of test results.

        Returns:
            PDF file as bytes.
        """
        try:
            return self._generate_pdf_reportlab(results)
        except ImportError:
            # Fall back to simple text format wrapped as "PDF"
            logger.warning("reportlab not available, generating text-based export")
            return self._generate_text_pdf_fallback(results)

    def _generate_pdf_reportlab(self, results: List[TestResult]) -> bytes:
        """Generate PDF using reportlab library.

        Args:
            results: List of test results.

        Returns:
            PDF bytes.
        """
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=18,
            spaceAfter=20,
        )
        heading_style = ParagraphStyle(
            "CustomHeading",
            parent=styles["Heading2"],
            fontSize=14,
            spaceAfter=10,
        )
        body_style = styles["Normal"]

        elements: List[Any] = []

        # Title
        elements.append(Paragraph(self._config.report_title, title_style))
        elements.append(Paragraph(self._config.company_name, body_style))
        elements.append(
            Paragraph(
                f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
                body_style,
            )
        )
        elements.append(Spacer(1, 20))

        # Executive Summary
        summary = self._calculate_summary(results)
        elements.append(Paragraph("Executive Summary", heading_style))

        summary_data = [
            ["Metric", "Value"],
            ["Total Tests", str(summary.total_tests)],
            ["Passed", str(summary.passed)],
            ["Failed", str(summary.failed)],
            ["Errors", str(summary.errors)],
            ["Pass Rate", f"{summary.pass_rate:.1f}%"],
        ]

        summary_table = Table(summary_data, colWidths=[2 * inch, 2 * inch])
        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        elements.append(summary_table)
        elements.append(Spacer(1, 20))

        # Results by Criterion
        elements.append(Paragraph("Results by Criterion", heading_style))

        criterion_data = [["Criterion", "Passed", "Failed", "Errors", "Pass Rate"]]
        for criterion, counts in sorted(summary.by_criterion.items()):
            passed = counts.get("passed", 0)
            failed = counts.get("failed", 0)
            errors = counts.get("errors", 0)
            total = passed + failed + errors
            rate = (passed / total * 100) if total > 0 else 0
            criterion_data.append([criterion, str(passed), str(failed), str(errors), f"{rate:.1f}%"])

        criterion_table = Table(criterion_data, colWidths=[1.5 * inch] * 5)
        criterion_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        elements.append(criterion_table)
        elements.append(Spacer(1, 20))

        # Failed Tests Detail
        failed_results = [r for r in results if r.status == TestStatus.FAILED]
        if failed_results:
            elements.append(Paragraph("Failed Tests - Detail", heading_style))

            for result in failed_results:
                elements.append(
                    Paragraph(
                        f"<b>{result.test_id}</b> - {result.severity.value.upper() if result.severity else 'N/A'}",
                        body_style,
                    )
                )
                elements.append(Paragraph(f"Result: {result.actual_result}", body_style))

                if result.exceptions:
                    for exc in result.exceptions[:5]:
                        elements.append(Paragraph(f"  - {exc}", body_style))

                elements.append(Spacer(1, 10))

        # Build PDF
        doc.build(elements)
        return buffer.getvalue()

    def _generate_text_pdf_fallback(self, results: List[TestResult]) -> bytes:
        """Generate a text-based fallback when reportlab is not available.

        Args:
            results: List of test results.

        Returns:
            Text content as bytes (with PDF-like header comment).
        """
        detailed_report = self.generate_detailed_report(results)
        header = "# PDF Export (Text Fallback - install reportlab for proper PDF)\n\n"
        content = header + detailed_report
        return content.encode("utf-8")

    # ------------------------------------------------------------------
    # JSON Export
    # ------------------------------------------------------------------

    def export_json(self, results: List[TestResult]) -> str:
        """Export test results as JSON.

        Args:
            results: List of test results.

        Returns:
            JSON string of results.
        """
        summary = self._calculate_summary(results)

        export_data = {
            "report_metadata": {
                "title": self._config.report_title,
                "company": self._config.company_name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            "summary": summary.model_dump(),
            "results": [r.model_dump(mode="json") for r in results],
        }

        return json.dumps(export_data, indent=2, default=str)

    # ------------------------------------------------------------------
    # Markdown Export
    # ------------------------------------------------------------------

    def export_markdown(self, results: List[TestResult]) -> str:
        """Export test results as Markdown.

        Args:
            results: List of test results.

        Returns:
            Markdown formatted string.
        """
        summary = self._calculate_summary(results)
        report_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines = [
            f"# {self._config.report_title}",
            "",
            f"**Company:** {self._config.company_name}",
            f"**Generated:** {report_time}",
            "",
            "## Executive Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Tests | {summary.total_tests} |",
            f"| Passed | {summary.passed} |",
            f"| Failed | {summary.failed} |",
            f"| Errors | {summary.errors} |",
            f"| Pass Rate | {summary.pass_rate:.1f}% |",
            "",
            "## Results by Criterion",
            "",
            "| Criterion | Passed | Failed | Errors | Pass Rate |",
            "|-----------|--------|--------|--------|-----------|",
        ]

        for criterion, counts in sorted(summary.by_criterion.items()):
            passed = counts.get("passed", 0)
            failed = counts.get("failed", 0)
            errors = counts.get("errors", 0)
            total = passed + failed + errors
            rate = (passed / total * 100) if total > 0 else 0
            lines.append(f"| {criterion} | {passed} | {failed} | {errors} | {rate:.1f}% |")

        lines.append("")

        # Failed tests
        failed_results = [r for r in results if r.status == TestStatus.FAILED]
        if failed_results:
            lines.append("## Failed Tests")
            lines.append("")

            for result in failed_results:
                severity = result.severity.value.upper() if result.severity else "N/A"
                lines.append(f"### {result.test_id} ({severity})")
                lines.append("")
                lines.append(f"**Result:** {result.actual_result}")
                lines.append("")

                if result.exceptions:
                    lines.append("**Exceptions:**")
                    for exc in result.exceptions:
                        lines.append(f"- {exc}")
                    lines.append("")

        lines.append("---")
        lines.append("*End of Report*")

        return "\n".join(lines)


__all__ = [
    "TestReporter",
    "ReportConfig",
    "TestSummary",
]
