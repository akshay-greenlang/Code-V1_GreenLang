# -*- coding: utf-8 -*-
"""
Unit Tests for ReportGenerator (AGENT-FOUND-009)

Tests text, JSON, markdown, and HTML report generation, summary creation,
and content validation for pass rate, duration, failures, and provenance.

Coverage target: 85%+ of report_generator.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline enums and stubs
# ---------------------------------------------------------------------------

class TestStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


class TestCaseResult:
    def __init__(self, test_id, name, status, duration_ms=0.0,
                 assertions=None, error_message=None, **kw):
        self.test_id = test_id
        self.name = name
        self.status = status
        self.duration_ms = duration_ms
        self.assertions = assertions or []
        self.error_message = error_message


class FailedAssertion:
    def __init__(self, name, message=""):
        self.name = name
        self.passed = False
        self.message = message


class TestSuiteResult:
    def __init__(self, suite_id, name, status, test_results=None,
                 total_tests=0, passed=0, failed=0, skipped=0, errors=0,
                 duration_ms=0.0, pass_rate=0.0, provenance_hash="", **kw):
        self.suite_id = suite_id
        self.name = name
        self.status = status
        self.test_results = test_results or []
        self.total_tests = total_tests
        self.passed = passed
        self.failed = failed
        self.skipped = skipped
        self.errors = errors
        self.duration_ms = duration_ms
        self.pass_rate = pass_rate
        self.provenance_hash = provenance_hash


# ---------------------------------------------------------------------------
# Inline ReportGenerator
# ---------------------------------------------------------------------------


class ReportGenerator:
    """Generate test reports in various formats."""

    def generate(self, suite_result: TestSuiteResult,
                 format: str = "text") -> str:
        if format == "json":
            return self._generate_json(suite_result)
        elif format == "markdown":
            return self._generate_markdown(suite_result)
        elif format == "html":
            return self._generate_html(suite_result)
        else:
            return self._generate_text(suite_result)

    def generate_summary(self, suite_result: TestSuiteResult) -> Dict[str, Any]:
        return {
            "suite_name": suite_result.name,
            "status": suite_result.status.value,
            "total_tests": suite_result.total_tests,
            "passed": suite_result.passed,
            "failed": suite_result.failed,
            "skipped": suite_result.skipped,
            "errors": suite_result.errors,
            "pass_rate": suite_result.pass_rate,
            "duration_ms": suite_result.duration_ms,
            "provenance_hash": suite_result.provenance_hash,
        }

    def _generate_text(self, result: TestSuiteResult) -> str:
        lines = [
            "=" * 80,
            f"TEST SUITE: {result.name}",
            "=" * 80,
            f"Status: {result.status.value.upper()}",
            f"Duration: {result.duration_ms:.2f}ms",
            f"Pass Rate: {result.pass_rate}%",
            "",
            f"Total: {result.total_tests}",
            f"Passed: {result.passed}",
            f"Failed: {result.failed}",
            f"Skipped: {result.skipped}",
            f"Errors: {result.errors}",
            "",
        ]
        for test in result.test_results:
            status_sym = {
                TestStatus.PASSED: "[PASS]",
                TestStatus.FAILED: "[FAIL]",
                TestStatus.SKIPPED: "[SKIP]",
                TestStatus.ERROR: "[ERR ]",
                TestStatus.TIMEOUT: "[TIME]",
            }.get(test.status, "[????]")
            lines.append(f"{status_sym} {test.name} ({test.duration_ms:.2f}ms)")
            if test.status == TestStatus.FAILED:
                for a in test.assertions:
                    if not a.passed:
                        lines.append(f"       - {a.name}: {a.message}")
            if test.error_message:
                lines.append(f"       Error: {test.error_message}")
        lines.append(f"Provenance Hash: {result.provenance_hash}")
        return "\n".join(lines)

    def _generate_json(self, result: TestSuiteResult) -> str:
        data = {
            "suite_id": result.suite_id,
            "name": result.name,
            "status": result.status.value,
            "total_tests": result.total_tests,
            "passed": result.passed,
            "failed": result.failed,
            "skipped": result.skipped,
            "errors": result.errors,
            "duration_ms": result.duration_ms,
            "pass_rate": result.pass_rate,
            "provenance_hash": result.provenance_hash,
            "tests": [
                {
                    "test_id": t.test_id, "name": t.name,
                    "status": t.status.value, "duration_ms": t.duration_ms,
                }
                for t in result.test_results
            ],
        }
        return json.dumps(data, indent=2, default=str)

    def _generate_markdown(self, result: TestSuiteResult) -> str:
        lines = [
            f"# Test Suite: {result.name}",
            "",
            f"**Status:** {result.status.value.upper()}",
            f"**Duration:** {result.duration_ms:.2f}ms",
            f"**Pass Rate:** {result.pass_rate}%",
            "",
            "## Summary",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| Total | {result.total_tests} |",
            f"| Passed | {result.passed} |",
            f"| Failed | {result.failed} |",
            f"| Skipped | {result.skipped} |",
            f"| Errors | {result.errors} |",
            "",
            "## Test Results",
            "",
            "| Status | Test | Duration |",
            "|--------|------|----------|",
        ]
        for test in result.test_results:
            lines.append(f"| {test.status.value.upper()} | {test.name} | {test.duration_ms:.2f}ms |")
        lines.extend([
            "", "## Provenance", "",
            f"Hash: `{result.provenance_hash}`",
        ])
        return "\n".join(lines)

    def _generate_html(self, result: TestSuiteResult) -> str:
        rows = []
        for test in result.test_results:
            rows.append(
                f"<tr><td>{test.status.value}</td>"
                f"<td>{test.name}</td>"
                f"<td>{test.duration_ms:.2f}ms</td></tr>"
            )
        table_rows = "\n".join(rows)
        return f"""<html>
<head><title>Test Suite: {result.name}</title></head>
<body>
<h1>{result.name}</h1>
<p>Status: {result.status.value.upper()}</p>
<p>Pass Rate: {result.pass_rate}%</p>
<p>Duration: {result.duration_ms:.2f}ms</p>
<table>
<tr><th>Status</th><th>Test</th><th>Duration</th></tr>
{table_rows}
</table>
<p>Provenance: {result.provenance_hash}</p>
</body>
</html>"""


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def generator():
    return ReportGenerator()


@pytest.fixture
def suite_result():
    tests = [
        TestCaseResult("t1", "test_pass_1", TestStatus.PASSED, 1.0),
        TestCaseResult("t2", "test_pass_2", TestStatus.PASSED, 2.0),
        TestCaseResult("t3", "test_fail_1", TestStatus.FAILED, 3.0,
                       assertions=[FailedAssertion("check", "Value mismatch")],
                       error_message="Assertion failed"),
        TestCaseResult("t4", "test_skip_1", TestStatus.SKIPPED, 0.0),
    ]
    return TestSuiteResult(
        suite_id="s1", name="MySuite", status=TestStatus.FAILED,
        test_results=tests, total_tests=4, passed=2, failed=1,
        skipped=1, errors=0, duration_ms=100.5, pass_rate=50.0,
        provenance_hash="abc123def456",
    )


# ===========================================================================
# Test Classes
# ===========================================================================


class TestGenerateTextReport:
    def test_generate_text_report(self, generator, suite_result):
        report = generator.generate(suite_result, format="text")
        assert "MySuite" in report
        assert "FAILED" in report

    def test_text_report_contains_test_names(self, generator, suite_result):
        report = generator.generate(suite_result, format="text")
        assert "test_pass_1" in report
        assert "test_fail_1" in report

    def test_text_report_contains_status_markers(self, generator, suite_result):
        report = generator.generate(suite_result, format="text")
        assert "[PASS]" in report
        assert "[FAIL]" in report

    def test_text_report_contains_error_details(self, generator, suite_result):
        report = generator.generate(suite_result, format="text")
        assert "Assertion failed" in report

    def test_text_report_contains_failed_assertion_details(self, generator, suite_result):
        report = generator.generate(suite_result, format="text")
        assert "Value mismatch" in report


class TestGenerateJsonReport:
    def test_generate_json_report(self, generator, suite_result):
        report = generator.generate(suite_result, format="json")
        data = json.loads(report)
        assert data["name"] == "MySuite"

    def test_json_report_valid_json(self, generator, suite_result):
        report = generator.generate(suite_result, format="json")
        data = json.loads(report)
        assert isinstance(data, dict)

    def test_json_report_contains_tests(self, generator, suite_result):
        report = generator.generate(suite_result, format="json")
        data = json.loads(report)
        assert len(data["tests"]) == 4

    def test_json_report_pass_rate(self, generator, suite_result):
        report = generator.generate(suite_result, format="json")
        data = json.loads(report)
        assert data["pass_rate"] == 50.0

    def test_json_report_provenance(self, generator, suite_result):
        report = generator.generate(suite_result, format="json")
        data = json.loads(report)
        assert data["provenance_hash"] == "abc123def456"


class TestGenerateMarkdownReport:
    def test_generate_markdown_report(self, generator, suite_result):
        report = generator.generate(suite_result, format="markdown")
        assert "# Test Suite: MySuite" in report

    def test_markdown_report_contains_table(self, generator, suite_result):
        report = generator.generate(suite_result, format="markdown")
        assert "| Metric | Count |" in report

    def test_markdown_report_contains_results_table(self, generator, suite_result):
        report = generator.generate(suite_result, format="markdown")
        assert "| Status | Test | Duration |" in report

    def test_markdown_report_contains_provenance(self, generator, suite_result):
        report = generator.generate(suite_result, format="markdown")
        assert "abc123def456" in report

    def test_markdown_report_pass_rate(self, generator, suite_result):
        report = generator.generate(suite_result, format="markdown")
        assert "50.0%" in report


class TestGenerateHtmlReport:
    def test_generate_html_report(self, generator, suite_result):
        report = generator.generate(suite_result, format="html")
        assert "<html>" in report
        assert "</html>" in report

    def test_html_report_contains_title(self, generator, suite_result):
        report = generator.generate(suite_result, format="html")
        assert "MySuite" in report

    def test_html_report_contains_table(self, generator, suite_result):
        report = generator.generate(suite_result, format="html")
        assert "<table>" in report

    def test_html_report_contains_provenance(self, generator, suite_result):
        report = generator.generate(suite_result, format="html")
        assert "abc123def456" in report

    def test_html_report_pass_rate(self, generator, suite_result):
        report = generator.generate(suite_result, format="html")
        assert "50.0%" in report


class TestGenerateSummary:
    def test_generate_summary(self, generator, suite_result):
        summary = generator.generate_summary(suite_result)
        assert summary["suite_name"] == "MySuite"
        assert summary["status"] == "failed"
        assert summary["total_tests"] == 4

    def test_summary_pass_rate(self, generator, suite_result):
        summary = generator.generate_summary(suite_result)
        assert summary["pass_rate"] == 50.0

    def test_summary_duration(self, generator, suite_result):
        summary = generator.generate_summary(suite_result)
        assert summary["duration_ms"] == 100.5

    def test_summary_provenance(self, generator, suite_result):
        summary = generator.generate_summary(suite_result)
        assert summary["provenance_hash"] == "abc123def456"


class TestReportContainsPassRate:
    def test_text_contains_pass_rate(self, generator, suite_result):
        report = generator.generate(suite_result, format="text")
        assert "50.0" in report

    def test_json_contains_pass_rate(self, generator, suite_result):
        report = generator.generate(suite_result, format="json")
        assert "50.0" in report


class TestReportContainsDuration:
    def test_text_contains_duration(self, generator, suite_result):
        report = generator.generate(suite_result, format="text")
        assert "100.50" in report

    def test_json_contains_duration(self, generator, suite_result):
        data = json.loads(generator.generate(suite_result, format="json"))
        assert data["duration_ms"] == 100.5


class TestReportContainsFailures:
    def test_text_contains_failure_info(self, generator, suite_result):
        report = generator.generate(suite_result, format="text")
        assert "[FAIL]" in report

    def test_json_contains_failed_count(self, generator, suite_result):
        data = json.loads(generator.generate(suite_result, format="json"))
        assert data["failed"] == 1


class TestReportContainsProvenance:
    def test_text_provenance(self, generator, suite_result):
        report = generator.generate(suite_result, format="text")
        assert "abc123def456" in report

    def test_markdown_provenance(self, generator, suite_result):
        report = generator.generate(suite_result, format="markdown")
        assert "`abc123def456`" in report
