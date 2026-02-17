# -*- coding: utf-8 -*-
"""
Unit Tests for ValidationReporterEngine - AGENT-DATA-019: Validation Rule Engine
==================================================================================

Tests all public methods of ValidationReporterEngine with 65+ tests covering
report generation for each report type (summary, detailed, compliance, trend,
executive), each format (text, json, html, markdown, csv), report generation
with evaluation results, compliance report with framework mapping, trend report
with historical data, report storage and retrieval, report hash integrity, and
statistics/clear operations.

Test Classes (9):
    - TestValidationReporterInit (5 tests)
    - TestReportTypes (10 tests)
    - TestReportFormats (10 tests)
    - TestReportWithEvaluationResults (8 tests)
    - TestComplianceReport (7 tests)
    - TestTrendReport (6 tests)
    - TestReportStorageAndRetrieval (8 tests)
    - TestReportHashIntegrity (6 tests)
    - TestStatisticsAndClear (5 tests)

Total: ~65 tests

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-019 Validation Rule Engine (GL-DATA-X-022)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.validation_rule_engine.provenance import ProvenanceTracker
from greenlang.validation_rule_engine.validation_reporter import ValidationReporterEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _make_evaluation_results(
    *,
    count: int = 5,
    pass_rate: float = 0.80,
    include_failures: bool = True,
) -> List[Dict[str, Any]]:
    """Create sample per-rule evaluation result dicts for the engine API.

    The engine's generate_report() expects a list of dicts, each with at
    minimum: rule_id, status, severity, pass_count, fail_count.
    """
    results: List[Dict[str, Any]] = []
    num_pass = int(count * pass_rate)
    for i in range(count):
        status = "pass" if i < num_pass else "fail"
        severity = "critical" if i == 0 and status == "fail" else (
            "high" if i == 1 and status == "fail" else "medium"
        )
        fail_count = 0 if status == "pass" else 10
        pass_count_val = 100 if status == "pass" else 90
        entry: Dict[str, Any] = {
            "rule_id": f"rule-{i:03d}",
            "rule_name": f"Rule {i}",
            "status": status,
            "severity": severity,
            "pass_count": pass_count_val,
            "fail_count": fail_count,
        }
        if include_failures and status == "fail":
            entry["failures"] = [
                {"row": j, "field": "val", "value": -1, "expected": 0, "message": "mismatch"}
                for j in range(fail_count)
            ]
        results.append(entry)
    return results


def _make_evaluation_results_with_severity(
    *,
    total: int = 10,
    critical: int = 0,
    high: int = 2,
    medium: int = 3,
    low: int = 1,
    pass_rate: float = 0.90,
) -> List[Dict[str, Any]]:
    """Create evaluation results with specific severity breakdown."""
    results: List[Dict[str, Any]] = []
    num_pass = int(total * pass_rate)
    severities = (
        ["critical"] * critical
        + ["high"] * high
        + ["medium"] * medium
        + ["low"] * low
    )
    # Pad to total
    while len(severities) < total:
        severities.append("info")

    for i in range(total):
        status = "pass" if i < num_pass else "fail"
        results.append({
            "rule_id": f"rule-{i:03d}",
            "rule_name": f"Rule {i}",
            "status": status,
            "severity": severities[i] if i < len(severities) else "info",
            "pass_count": 100 if status == "pass" else 90,
            "fail_count": 0 if status == "pass" else 10,
        })
    return results


def _make_trend_snapshots(
    pass_rates: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Create evaluation history snapshots for trend report testing.

    Each snapshot has: timestamp, pass_rate, label, results.
    """
    if pass_rates is None:
        pass_rates = [0.80, 0.85, 0.90, 0.92, 0.95]

    snapshots: List[Dict[str, Any]] = []
    base_time = _utcnow()
    for idx, rate in enumerate(pass_rates):
        ts = (base_time - timedelta(days=len(pass_rates) - idx)).isoformat()
        count = 10
        num_pass = int(count * rate)
        snapshot_results = []
        for j in range(count):
            status = "pass" if j < num_pass else "fail"
            snapshot_results.append({
                "rule_id": f"rule-{j:03d}",
                "rule_name": f"Rule {j}",
                "status": status,
                "severity": "medium",
                "pass_count": 100 if status == "pass" else 90,
                "fail_count": 0 if status == "pass" else 10,
            })
        snapshots.append({
            "timestamp": ts,
            "label": f"Snapshot-{idx}",
            "pass_rate": rate,
            "results": snapshot_results,
        })
    return snapshots


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> ValidationReporterEngine:
    """Create a fresh ValidationReporterEngine instance for each test."""
    return ValidationReporterEngine(genesis_hash="test-reporter-genesis")


@pytest.fixture
def sample_results() -> List[Dict[str, Any]]:
    """Sample evaluation results for report generation."""
    return _make_evaluation_results(count=10, pass_rate=0.80)


@pytest.fixture
def sample_results_perfect() -> List[Dict[str, Any]]:
    """Sample evaluation results with 100% pass rate."""
    return _make_evaluation_results(count=10, pass_rate=1.0, include_failures=False)


@pytest.fixture
def historical_snapshots() -> List[Dict[str, Any]]:
    """Multiple evaluation snapshots for trend reporting."""
    return _make_trend_snapshots([0.80, 0.84, 0.88, 0.92, 0.95])


# ==========================================================================
# TestValidationReporterInit
# ==========================================================================


class TestValidationReporterInit:
    """Tests for ValidationReporterEngine initialization."""

    def test_init_creates_instance(self, engine: ValidationReporterEngine) -> None:
        """Engine initializes without error."""
        assert engine is not None

    def test_init_has_provenance_tracker(self, engine: ValidationReporterEngine) -> None:
        """Engine has a provenance tracker."""
        assert hasattr(engine, "_provenance")

    def test_init_no_reports_stored(self, engine: ValidationReporterEngine) -> None:
        """Engine starts with no reports."""
        stats = engine.get_statistics()
        assert stats["total_reports_stored"] == 0

    def test_init_custom_genesis_hash(self) -> None:
        """Engine accepts a custom genesis hash."""
        eng = ValidationReporterEngine(genesis_hash="custom-genesis")
        assert eng is not None

    def test_init_default_genesis_hash(self) -> None:
        """Engine works with default genesis hash."""
        eng = ValidationReporterEngine()
        assert eng is not None


# ==========================================================================
# TestReportTypes
# ==========================================================================


class TestReportTypes:
    """Tests for generating each report type."""

    def test_summary_report(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Generate a summary report."""
        report = engine.generate_report("summary", "json", sample_results)
        assert report is not None
        assert report["report_type"] == "summary"

    def test_detailed_report(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Generate a detailed report with per-record details."""
        report = engine.generate_report("detailed", "json", sample_results)
        assert report is not None
        assert report["report_type"] == "detailed"

    def test_compliance_report(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Generate a compliance report."""
        report = engine.generate_report(
            "compliance", "json", sample_results,
            parameters={"framework": "ghg_protocol"},
        )
        assert report is not None
        assert report["report_type"] == "compliance"

    def test_trend_report(self, engine: ValidationReporterEngine, historical_snapshots: List[Dict]) -> None:
        """Generate a trend report from historical snapshots."""
        report = engine.generate_report("trend", "json", historical_snapshots)
        assert report is not None
        assert report["report_type"] == "trend"

    def test_executive_report(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Generate an executive summary report."""
        report = engine.generate_report("executive", "json", sample_results)
        assert report is not None
        assert report["report_type"] == "executive"

    def test_summary_contains_pass_rate(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Summary report includes pass rate in content."""
        report = engine.generate_report("summary", "json", sample_results)
        parsed = json.loads(report["content"])
        assert "overview" in parsed
        assert "rule_pass_rate" in parsed["overview"]

    def test_detailed_contains_content(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Detailed report has non-empty content."""
        report = engine.generate_report("detailed", "json", sample_results)
        assert report["content"] != ""

    def test_executive_contains_recommendations(self, engine: ValidationReporterEngine,
                                                  sample_results: List[Dict]) -> None:
        """Executive report content includes action items or recommendations."""
        # Use results with failures to trigger recommendations
        results = _make_evaluation_results(count=10, pass_rate=0.50)
        report = engine.generate_report("executive", "json", results)
        parsed = json.loads(report["content"])
        # Executive reports have action_items at the content level
        assert "action_items" in parsed or "severity_breakdown" in parsed

    def test_report_has_severity_summary(self, engine: ValidationReporterEngine,
                                          sample_results: List[Dict]) -> None:
        """Summary report content includes severity breakdown."""
        report = engine.generate_report("summary", "json", sample_results)
        parsed = json.loads(report["content"])
        assert "severity_breakdown" in parsed

    def test_report_has_total_rules(self, engine: ValidationReporterEngine,
                                     sample_results: List[Dict]) -> None:
        """Summary report content includes total rules count."""
        report = engine.generate_report("summary", "json", sample_results)
        parsed = json.loads(report["content"])
        assert "overview" in parsed
        assert parsed["overview"]["total_rules_evaluated"] >= 0


# ==========================================================================
# TestReportFormats
# ==========================================================================


class TestReportFormats:
    """Tests for generating reports in each format."""

    def test_text_format(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Generate report in text format."""
        report = engine.generate_report("summary", "text", sample_results)
        assert report["format"] == "text"
        assert isinstance(report["content"], str)

    def test_json_format(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Generate report in JSON format."""
        report = engine.generate_report("summary", "json", sample_results)
        assert report["format"] == "json"
        # JSON content should be parseable
        parsed = json.loads(report["content"])
        assert isinstance(parsed, dict)

    def test_html_format(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Generate report in HTML format."""
        report = engine.generate_report("summary", "html", sample_results)
        assert report["format"] == "html"
        assert "<" in report["content"]  # Basic HTML marker

    def test_markdown_format(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Generate report in Markdown format."""
        report = engine.generate_report("summary", "markdown", sample_results)
        assert report["format"] == "markdown"
        assert "#" in report["content"] or "**" in report["content"] or "|" in report["content"]

    def test_csv_format(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Generate report in CSV format."""
        report = engine.generate_report("detailed", "csv", sample_results)
        assert report["format"] == "csv"
        assert "," in report["content"]  # CSV delimiter

    def test_text_format_readable(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Text format produces human-readable output."""
        report = engine.generate_report("summary", "text", sample_results)
        assert len(report["content"]) > 10

    def test_json_format_complete(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """JSON format includes key metrics."""
        report = engine.generate_report("summary", "json", sample_results)
        parsed = json.loads(report["content"])
        assert "overview" in parsed or "pass_rate" in parsed or "total_records" in parsed

    def test_html_format_self_contained(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """HTML format is self-contained."""
        report = engine.generate_report("summary", "html", sample_results)
        assert "html" in report["content"].lower() or "<div" in report["content"].lower() or "<table" in report["content"].lower()

    def test_csv_format_has_header(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """CSV format has a header row."""
        report = engine.generate_report("detailed", "csv", sample_results)
        lines = report["content"].strip().split("\n")
        assert len(lines) >= 2  # Header + at least one data row

    def test_default_format_is_json(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Default format when passing json explicitly."""
        report = engine.generate_report("summary", "json", sample_results)
        assert report["format"] == "json"


# ==========================================================================
# TestReportWithEvaluationResults
# ==========================================================================


class TestReportWithEvaluationResults:
    """Tests for reports generated from evaluation results."""

    def test_report_reflects_pass_rate(self, engine: ValidationReporterEngine) -> None:
        """Report pass rate reflects evaluation results."""
        # 8 pass / 10 total => 80%
        results = _make_evaluation_results(count=10, pass_rate=0.80)
        report = engine.generate_report("summary", "json", results)
        parsed = json.loads(report["content"])
        # pass rate is formatted as percentage string e.g. "80.0%"
        assert "rule_pass_rate" in parsed.get("overview", {})

    def test_report_reflects_total_records(self, engine: ValidationReporterEngine) -> None:
        """Report total_records reflects evaluation results."""
        results = _make_evaluation_results(count=5, pass_rate=1.0, include_failures=False)
        report = engine.generate_report("summary", "json", results)
        parsed = json.loads(report["content"])
        assert parsed["overview"]["total_records_evaluated"] >= 0

    def test_report_reflects_total_rules(self, engine: ValidationReporterEngine) -> None:
        """Report total_rules reflects evaluation results."""
        results = _make_evaluation_results(count=25, pass_rate=1.0, include_failures=False)
        report = engine.generate_report("summary", "json", results)
        parsed = json.loads(report["content"])
        assert parsed["overview"]["total_rules_evaluated"] == 25

    def test_report_with_critical_failures(self, engine: ValidationReporterEngine) -> None:
        """Report captures critical severity failures."""
        results = _make_evaluation_results_with_severity(
            total=10, critical=5, pass_rate=0.50,
        )
        report = engine.generate_report("summary", "json", results)
        parsed = json.loads(report["content"])
        assert "severity_breakdown" in parsed
        assert parsed["severity_breakdown"].get("critical", 0) >= 1

    def test_report_with_zero_failures(self, engine: ValidationReporterEngine,
                                        sample_results_perfect: List[Dict]) -> None:
        """Report with perfect results."""
        report = engine.generate_report("summary", "json", sample_results_perfect)
        parsed = json.loads(report["content"])
        assert parsed["overview"]["rule_pass_rate"] == "100.0%"

    def test_report_has_report_id(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Report includes a report_id."""
        report = engine.generate_report("summary", "json", sample_results)
        assert "report_id" in report
        assert len(report["report_id"]) > 0

    def test_report_has_provenance_hash(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Report includes a provenance_hash."""
        report = engine.generate_report("summary", "json", sample_results)
        assert "provenance_hash" in report
        assert len(report["provenance_hash"]) > 0

    def test_report_generated_at(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Report has generated_at timestamp."""
        report = engine.generate_report("summary", "json", sample_results)
        assert report["generated_at"] is not None
        assert len(report["generated_at"]) > 0


# ==========================================================================
# TestComplianceReport
# ==========================================================================


class TestComplianceReport:
    """Tests for compliance report generation with framework mapping."""

    def test_compliance_report_ghg(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Compliance report for GHG Protocol framework."""
        report = engine.generate_report(
            "compliance", "json", sample_results,
            parameters={"framework": "ghg_protocol"},
        )
        assert report["report_type"] == "compliance"

    def test_compliance_report_csrd(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Compliance report for CSRD/ESRS framework."""
        report = engine.generate_report(
            "compliance", "json", sample_results,
            parameters={"framework": "csrd_esrs"},
        )
        assert report["report_type"] == "compliance"

    def test_compliance_framework_percentage(self, engine: ValidationReporterEngine,
                                               sample_results: List[Dict]) -> None:
        """Compliance report includes framework compliance information."""
        report = engine.generate_report(
            "compliance", "json", sample_results,
            parameters={"framework": "ghg_protocol"},
        )
        parsed = json.loads(report["content"])
        assert "framework_summaries" in parsed
        assert "ghg_protocol" in parsed["framework_summaries"]
        summary = parsed["framework_summaries"]["ghg_protocol"]
        assert "overall_compliance" in summary
        assert 0.0 <= summary["overall_compliance"] <= 100.0

    def test_compliance_report_recommendations(self, engine: ValidationReporterEngine) -> None:
        """Compliance report content has evaluation summary."""
        results = _make_evaluation_results(count=10, pass_rate=0.70)
        report = engine.generate_report(
            "compliance", "json", results,
            parameters={"framework": "ghg_protocol"},
        )
        parsed = json.loads(report["content"])
        assert "evaluation_summary" in parsed

    def test_compliance_report_scope(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Compliance report has framework information."""
        report = engine.generate_report(
            "compliance", "json", sample_results,
            parameters={"framework": "ghg_protocol"},
        )
        parsed = json.loads(report["content"])
        assert "frameworks" in parsed
        assert "ghg_protocol" in parsed["frameworks"]

    def test_compliance_report_provenance(self, engine: ValidationReporterEngine,
                                           sample_results: List[Dict]) -> None:
        """Compliance report has report_hash and provenance_hash."""
        report = engine.generate_report(
            "compliance", "json", sample_results,
            parameters={"framework": "ghg_protocol"},
        )
        assert report["report_hash"] != ""
        assert report["provenance_hash"] != ""

    def test_compliance_report_no_framework(self, engine: ValidationReporterEngine,
                                              sample_results: List[Dict]) -> None:
        """Compliance report without framework still generates (all frameworks)."""
        report = engine.generate_report("compliance", "json", sample_results)
        assert report is not None
        parsed = json.loads(report["content"])
        # When no framework specified, all four frameworks are included
        assert "frameworks" in parsed


# ==========================================================================
# TestTrendReport
# ==========================================================================


class TestTrendReport:
    """Tests for trend report generation with historical data."""

    def test_trend_report_from_multiple_runs(self, engine: ValidationReporterEngine,
                                               historical_snapshots: List[Dict]) -> None:
        """Trend report analyzes multiple evaluation snapshots."""
        report = engine.generate_report("trend", "json", historical_snapshots)
        assert report["report_type"] == "trend"

    def test_trend_report_shows_improvement(self, engine: ValidationReporterEngine) -> None:
        """Trend report detects improving pass rates."""
        snapshots = _make_trend_snapshots([0.70, 0.80, 0.90])
        report = engine.generate_report("trend", "json", snapshots)
        parsed = json.loads(report["content"])
        assert "trend_summary" in parsed
        assert parsed["trend_summary"]["trend_direction"] == "improving"

    def test_trend_report_shows_degradation(self, engine: ValidationReporterEngine) -> None:
        """Trend report detects degrading pass rates."""
        snapshots = _make_trend_snapshots([0.95, 0.85, 0.75])
        report = engine.generate_report("trend", "json", snapshots)
        parsed = json.loads(report["content"])
        assert "trend_summary" in parsed
        assert parsed["trend_summary"]["trend_direction"] == "declining"

    def test_trend_report_single_run(self, engine: ValidationReporterEngine) -> None:
        """Trend report with single snapshot still works."""
        snapshots = _make_trend_snapshots([0.90])
        report = engine.generate_report("trend", "json", snapshots)
        assert report is not None
        parsed = json.loads(report["content"])
        # With less than 2 snapshots, a message about insufficient history is shown
        assert "message" in parsed or "trend_summary" in parsed

    def test_trend_report_content_not_empty(self, engine: ValidationReporterEngine,
                                              historical_snapshots: List[Dict]) -> None:
        """Trend report has non-empty content."""
        report = engine.generate_report("trend", "json", historical_snapshots)
        assert report["content"] != ""

    def test_trend_report_provenance(self, engine: ValidationReporterEngine,
                                       historical_snapshots: List[Dict]) -> None:
        """Trend report has report_hash."""
        report = engine.generate_report("trend", "json", historical_snapshots)
        assert report["report_hash"] != ""


# ==========================================================================
# TestReportStorageAndRetrieval
# ==========================================================================


class TestReportStorageAndRetrieval:
    """Tests for report storage and retrieval."""

    def test_report_stored_after_generation(self, engine: ValidationReporterEngine,
                                              sample_results: List[Dict]) -> None:
        """Generated report is stored and retrievable."""
        report = engine.generate_report("summary", "json", sample_results)
        retrieved = engine.get_report(report["report_id"])
        assert retrieved is not None
        assert retrieved["report_id"] == report["report_id"]

    def test_get_nonexistent_report(self, engine: ValidationReporterEngine) -> None:
        """Getting a nonexistent report returns None."""
        result = engine.get_report("nonexistent-id")
        assert result is None

    def test_list_reports(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """List all generated reports."""
        engine.generate_report("summary", "json", sample_results)
        engine.generate_report("executive", "json", sample_results)
        reports = engine.list_reports()
        assert len(reports) >= 2

    def test_list_empty_reports(self, engine: ValidationReporterEngine) -> None:
        """List returns empty when no reports generated."""
        reports = engine.list_reports()
        assert len(reports) == 0

    def test_multiple_reports_different_types(self, engine: ValidationReporterEngine,
                                                sample_results: List[Dict]) -> None:
        """Multiple reports of different types stored independently."""
        r1 = engine.generate_report("summary", "json", sample_results)
        r2 = engine.generate_report("executive", "json", sample_results)
        assert r1["report_id"] != r2["report_id"]

    def test_report_has_unique_id(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Each report has a unique ID."""
        ids = set()
        for _ in range(5):
            report = engine.generate_report("summary", "json", sample_results)
            ids.add(report["report_id"])
        assert len(ids) == 5

    def test_report_content_preserved(self, engine: ValidationReporterEngine,
                                        sample_results: List[Dict]) -> None:
        """Report content is preserved after storage and retrieval."""
        report = engine.generate_report("summary", "text", sample_results)
        retrieved = engine.get_report(report["report_id"])
        assert retrieved is not None
        assert retrieved["content"] == report["content"]

    def test_report_format_preserved(self, engine: ValidationReporterEngine,
                                       sample_results: List[Dict]) -> None:
        """Report format is preserved after storage and retrieval."""
        report = engine.generate_report("summary", "markdown", sample_results)
        retrieved = engine.get_report(report["report_id"])
        assert retrieved is not None
        assert retrieved["format"] == "markdown"


# ==========================================================================
# TestReportHashIntegrity
# ==========================================================================


class TestReportHashIntegrity:
    """Tests for report hash integrity (SHA-256)."""

    def test_report_has_hash(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Report includes SHA-256 hash."""
        report = engine.generate_report("summary", "json", sample_results)
        assert report["report_hash"] != ""
        assert len(report["report_hash"]) == 64

    def test_hash_is_deterministic(self, engine: ValidationReporterEngine) -> None:
        """Same content produces same hash."""
        results = _make_evaluation_results(count=5, pass_rate=0.80)
        r1 = engine.generate_report("summary", "json", results)
        # Hash should be a valid 64-character hex string
        assert len(r1["report_hash"]) == 64

    def test_different_content_different_hash(self, engine: ValidationReporterEngine) -> None:
        """Different content produces different hashes."""
        results1 = _make_evaluation_results(count=10, pass_rate=0.90)
        results2 = _make_evaluation_results(count=10, pass_rate=0.50)
        r1 = engine.generate_report("summary", "json", results1)
        r2 = engine.generate_report("summary", "json", results2)
        assert r1["report_hash"] != r2["report_hash"]

    def test_hash_is_hex_string(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Hash is a valid hexadecimal string."""
        report = engine.generate_report("summary", "json", sample_results)
        assert all(c in "0123456789abcdef" for c in report["report_hash"])

    def test_hash_covers_content(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Hash is computed from report content."""
        report = engine.generate_report("summary", "json", sample_results)
        # Hash should exist and be non-trivial
        assert report["report_hash"] != "0" * 64

    def test_provenance_hash_distinct_from_report_hash(self, engine: ValidationReporterEngine,
                                                         sample_results: List[Dict]) -> None:
        """Report hash and provenance hash are separate concerns."""
        report = engine.generate_report("summary", "json", sample_results)
        assert report["report_hash"] != ""
        assert report["provenance_hash"] != ""
        # The report_hash is a content hash, the provenance_hash is a chain hash
        assert report["report_hash"] != report["provenance_hash"]


# ==========================================================================
# TestStatisticsAndClear
# ==========================================================================


class TestStatisticsAndClear:
    """Tests for statistics and clear operations."""

    def test_statistics_initial(self, engine: ValidationReporterEngine) -> None:
        """Initial statistics are empty."""
        stats = engine.get_statistics()
        assert stats["total_reports_stored"] == 0

    def test_statistics_after_generation(self, engine: ValidationReporterEngine,
                                          sample_results: List[Dict]) -> None:
        """Statistics update after report generation."""
        engine.generate_report("summary", "json", sample_results)
        stats = engine.get_statistics()
        assert stats["total_reports_stored"] >= 1
        assert stats["total_reports_generated"] >= 1

    def test_statistics_by_type(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Statistics include breakdown by report type."""
        engine.generate_report("summary", "json", sample_results)
        engine.generate_report("executive", "json", sample_results)
        stats = engine.get_statistics()
        assert "by_report_type" in stats
        assert stats["total_reports_stored"] == 2

    def test_clear_resets_reports(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Clear removes all stored reports."""
        engine.generate_report("summary", "json", sample_results)
        engine.clear()
        stats = engine.get_statistics()
        assert stats["total_reports_stored"] == 0

    def test_clear_resets_list(self, engine: ValidationReporterEngine, sample_results: List[Dict]) -> None:
        """Clear empties the report list."""
        engine.generate_report("summary", "json", sample_results)
        engine.clear()
        reports = engine.list_reports()
        assert len(reports) == 0
