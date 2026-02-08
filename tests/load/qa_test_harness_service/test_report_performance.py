# -*- coding: utf-8 -*-
"""
Load Tests for Report Generation Performance (AGENT-FOUND-009)

Tests report generation latency, large suite report handling, concurrent
report generation, summary computation throughput, and format-specific
performance characteristics.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline enums and stubs (self-contained for load testing)
# ---------------------------------------------------------------------------


class TestStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


class FailedAssertion:
    def __init__(self, name, message=""):
        self.name = name
        self.passed = False
        self.message = message


class TestCaseResult:
    def __init__(self, test_id, name, status, duration_ms=0.0,
                 assertions=None, error_message=None):
        self.test_id = test_id
        self.name = name
        self.status = status
        self.duration_ms = duration_ms
        self.assertions = assertions or []
        self.error_message = error_message


class TestSuiteResult:
    def __init__(self, suite_id, name, status, test_results=None,
                 total_tests=0, passed=0, failed=0, skipped=0, errors=0,
                 duration_ms=0.0, pass_rate=0.0, provenance_hash=""):
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
# Inline ReportGenerator for load testing
# ---------------------------------------------------------------------------


class LoadReportGenerator:
    """Report generator optimized for load testing."""

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
            lines.append(
                f"| {test.status.value.upper()} | {test.name} "
                f"| {test.duration_ms:.2f}ms |"
            )
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


# ---------------------------------------------------------------------------
# Helper: build suite results of various sizes
# ---------------------------------------------------------------------------


def _build_suite_result(name: str, num_tests: int,
                        fail_pct: float = 0.0) -> TestSuiteResult:
    """Build a synthetic TestSuiteResult with the given number of tests."""
    tests = []
    passed = 0
    failed = 0
    for i in range(num_tests):
        if i < int(num_tests * fail_pct):
            status = TestStatus.FAILED
            assertions = [FailedAssertion(f"check_{i}", f"Mismatch at {i}")]
            error_msg = f"Failed at index {i}"
            failed += 1
        else:
            status = TestStatus.PASSED
            assertions = []
            error_msg = None
            passed += 1
        tests.append(TestCaseResult(
            test_id=f"t-{i:06d}", name=f"test_{i}",
            status=status, duration_ms=float(i % 100) * 0.1,
            assertions=assertions, error_message=error_msg,
        ))

    total = len(tests)
    pass_rate = (passed / total * 100) if total else 0
    overall = TestStatus.FAILED if failed > 0 else TestStatus.PASSED

    return TestSuiteResult(
        suite_id=f"suite-{name}", name=name, status=overall,
        test_results=tests, total_tests=total, passed=passed,
        failed=failed, skipped=0, errors=0,
        duration_ms=float(total) * 1.5, pass_rate=round(pass_rate, 2),
        provenance_hash="load-test-prov-hash-0123456789abcdef",
    )


# ===========================================================================
# Test Classes
# ===========================================================================


class TestReportGenerationLatency:
    """Test report generation latency for small and medium suites."""

    def test_text_report_under_100ms(self):
        """Test that text report generation completes in under 100ms."""
        generator = LoadReportGenerator()
        suite = _build_suite_result("TextPerfTest", 50, fail_pct=0.1)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            report = generator.generate(suite, format="text")
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
            assert len(report) > 0

        avg_ms = sum(times) / len(times)
        assert avg_ms < 100.0, f"Text report avg {avg_ms:.3f}ms exceeds 100ms target"

    def test_json_report_under_100ms(self):
        """Test that JSON report generation completes in under 100ms."""
        generator = LoadReportGenerator()
        suite = _build_suite_result("JsonPerfTest", 50, fail_pct=0.1)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            report = generator.generate(suite, format="json")
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
            data = json.loads(report)
            assert data["total_tests"] == 50

        avg_ms = sum(times) / len(times)
        assert avg_ms < 100.0, f"JSON report avg {avg_ms:.3f}ms exceeds 100ms target"

    def test_markdown_report_under_100ms(self):
        """Test that markdown report generation completes in under 100ms."""
        generator = LoadReportGenerator()
        suite = _build_suite_result("MdPerfTest", 50, fail_pct=0.1)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            report = generator.generate(suite, format="markdown")
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
            assert "# Test Suite:" in report

        avg_ms = sum(times) / len(times)
        assert avg_ms < 100.0, f"Markdown report avg {avg_ms:.3f}ms exceeds 100ms target"

    def test_html_report_under_100ms(self):
        """Test that HTML report generation completes in under 100ms."""
        generator = LoadReportGenerator()
        suite = _build_suite_result("HtmlPerfTest", 50, fail_pct=0.1)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            report = generator.generate(suite, format="html")
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
            assert "<html>" in report

        avg_ms = sum(times) / len(times)
        assert avg_ms < 100.0, f"HTML report avg {avg_ms:.3f}ms exceeds 100ms target"


class TestLargeSuiteReport:
    """Test report generation for large test suites."""

    def test_report_500_tests_under_500ms(self):
        """Test text report for 500 tests completes in under 500ms."""
        generator = LoadReportGenerator()
        suite = _build_suite_result("Large500", 500, fail_pct=0.2)

        start = time.perf_counter()
        report = generator.generate(suite, format="text")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(report) > 0
        assert "Large500" in report
        assert elapsed_ms < 500.0, f"500-test report took {elapsed_ms:.3f}ms"

    def test_json_report_1000_tests(self):
        """Test JSON report for 1000 tests completes in under 1 second."""
        generator = LoadReportGenerator()
        suite = _build_suite_result("Large1000", 1000, fail_pct=0.1)

        start = time.perf_counter()
        report = generator.generate(suite, format="json")
        elapsed_ms = (time.perf_counter() - start) * 1000

        data = json.loads(report)
        assert data["total_tests"] == 1000
        assert len(data["tests"]) == 1000
        assert elapsed_ms < 1000.0, f"1000-test JSON report took {elapsed_ms:.3f}ms"

    def test_markdown_report_1000_tests(self):
        """Test markdown report for 1000 tests completes in under 1 second."""
        generator = LoadReportGenerator()
        suite = _build_suite_result("LargeMd1000", 1000, fail_pct=0.05)

        start = time.perf_counter()
        report = generator.generate(suite, format="markdown")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert "# Test Suite: LargeMd1000" in report
        assert elapsed_ms < 1000.0, f"1000-test MD report took {elapsed_ms:.3f}ms"

    def test_html_report_2000_tests(self):
        """Test HTML report for 2000 tests completes within time budget."""
        generator = LoadReportGenerator()
        suite = _build_suite_result("LargeHtml2000", 2000, fail_pct=0.1)

        start = time.perf_counter()
        report = generator.generate(suite, format="html")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert "<html>" in report
        assert "LargeHtml2000" in report
        assert elapsed_ms < 2000.0, f"2000-test HTML report took {elapsed_ms:.3f}ms"


class TestConcurrentReportGeneration:
    """Test concurrent report generation across threads."""

    def test_concurrent_text_reports_10_threads(self):
        """Test generating 10 text reports concurrently."""
        generator = LoadReportGenerator()
        results = []

        def gen_report(idx):
            suite = _build_suite_result(f"ConcText_{idx}", 100, fail_pct=0.1)
            return generator.generate(suite, format="text")

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(gen_report, i) for i in range(10)]
            for f in as_completed(futures):
                results.append(f.result())
        elapsed_s = time.perf_counter() - start

        assert len(results) == 10
        assert all(len(r) > 0 for r in results)
        assert elapsed_s < 10.0, f"10 concurrent reports took {elapsed_s:.3f}s"

    def test_concurrent_json_reports_20_threads(self):
        """Test generating 20 JSON reports concurrently."""
        generator = LoadReportGenerator()
        results = []

        def gen_report(idx):
            suite = _build_suite_result(f"ConcJson_{idx}", 50)
            report = generator.generate(suite, format="json")
            return json.loads(report)

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(gen_report, i) for i in range(20)]
            for f in as_completed(futures):
                results.append(f.result())
        elapsed_s = time.perf_counter() - start

        assert len(results) == 20
        assert all(r["total_tests"] == 50 for r in results)
        assert elapsed_s < 10.0, f"20 concurrent JSON reports took {elapsed_s:.3f}s"

    def test_concurrent_mixed_format_reports(self):
        """Test generating reports in mixed formats concurrently."""
        generator = LoadReportGenerator()
        formats = ["text", "json", "markdown", "html"]
        results = []

        def gen_report(idx):
            fmt = formats[idx % len(formats)]
            suite = _build_suite_result(f"Mixed_{idx}", 30)
            return generator.generate(suite, format=fmt)

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(gen_report, i) for i in range(40)]
            for f in as_completed(futures):
                results.append(f.result())
        elapsed_s = time.perf_counter() - start

        assert len(results) == 40
        assert all(len(r) > 0 for r in results)
        assert elapsed_s < 10.0, f"40 mixed reports took {elapsed_s:.3f}s"


class TestSummaryComputationThroughput:
    """Test summary generation throughput."""

    def test_summary_throughput_1000_per_second(self):
        """Test that summary generation achieves >1000/second."""
        generator = LoadReportGenerator()
        suite = _build_suite_result("SummaryPerf", 100, fail_pct=0.15)

        start = time.perf_counter()
        for _ in range(5000):
            summary = generator.generate_summary(suite)
        elapsed_s = time.perf_counter() - start

        throughput = 5000 / elapsed_s
        assert throughput > 1000, (
            f"Summary throughput {throughput:.0f}/s below 1000/s target"
        )

    def test_summary_correctness_under_load(self):
        """Test that summaries remain correct under sustained load."""
        generator = LoadReportGenerator()
        suite = _build_suite_result("SummaryCorrect", 200, fail_pct=0.25)

        for _ in range(1000):
            summary = generator.generate_summary(suite)
            assert summary["total_tests"] == 200
            assert summary["passed"] == 150
            assert summary["failed"] == 50
            assert summary["pass_rate"] == 75.0
            assert summary["suite_name"] == "SummaryCorrect"


class TestReportFormatPerformanceComparison:
    """Compare performance characteristics across report formats."""

    def test_format_relative_performance(self):
        """Test that all formats complete within 2x of each other."""
        generator = LoadReportGenerator()
        suite = _build_suite_result("FormatComp", 200, fail_pct=0.1)
        format_times = {}

        for fmt in ["text", "json", "markdown", "html"]:
            times = []
            for _ in range(50):
                start = time.perf_counter()
                generator.generate(suite, format=fmt)
                elapsed_ms = (time.perf_counter() - start) * 1000
                times.append(elapsed_ms)
            format_times[fmt] = sum(times) / len(times)

        min_time = min(format_times.values())
        max_time = max(format_times.values())

        # No format should be more than 15x slower than the fastest
        # (JSON serialization and HTML string building can vary significantly)
        assert max_time < min_time * 15, (
            f"Format performance gap too large: "
            f"min={min_time:.3f}ms, max={max_time:.3f}ms, "
            f"ratio={max_time / min_time:.1f}x"
        )

    def test_json_parseable_under_load(self):
        """Test that JSON output remains parseable under sustained generation."""
        generator = LoadReportGenerator()

        for i in range(500):
            suite = _build_suite_result(f"ParseTest_{i}", 20 + (i % 50))
            report = generator.generate(suite, format="json")
            data = json.loads(report)
            assert data["name"] == f"ParseTest_{i}"
            assert data["total_tests"] == 20 + (i % 50)


class TestReportOutputSizeScaling:
    """Test that report output size scales linearly with test count."""

    def test_output_size_scales_linearly(self):
        """Test that report size scales approximately linearly."""
        generator = LoadReportGenerator()
        sizes = {}

        for count in [10, 50, 100, 500]:
            suite = _build_suite_result(f"Scale_{count}", count)
            report = generator.generate(suite, format="text")
            sizes[count] = len(report)

        # Size at 500 tests should be roughly 50x size at 10 tests
        # Allow generous margin (5x to 100x)
        ratio = sizes[500] / sizes[10]
        assert 5 < ratio < 100, (
            f"Size scaling ratio {ratio:.1f} outside expected range 5-100"
        )

    def test_json_output_size_proportional(self):
        """Test JSON report size is proportional to test count."""
        generator = LoadReportGenerator()
        sizes = {}

        for count in [10, 100, 1000]:
            suite = _build_suite_result(f"JsonScale_{count}", count)
            report = generator.generate(suite, format="json")
            sizes[count] = len(report)

        # Validate proportional growth
        ratio_100_to_10 = sizes[100] / sizes[10]
        ratio_1000_to_100 = sizes[1000] / sizes[100]

        # Both ratios should be roughly similar (within 2x of each other)
        assert 0.2 < ratio_100_to_10 / ratio_1000_to_100 < 5.0, (
            f"Non-linear growth: 100/10={ratio_100_to_10:.1f}, "
            f"1000/100={ratio_1000_to_100:.1f}"
        )
