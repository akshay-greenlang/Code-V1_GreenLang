# -*- coding: utf-8 -*-
"""
Unit tests for AuditTrailEngine - AGENT-DATA-015 (Engine 6 of 7)

Tests event recording, compliance report generation, discrepancy logs,
event querying, export, integrity verification, and clear operations
with 50+ tests targeting 85%+ code coverage.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation (GL-DATA-X-018)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from greenlang.cross_source_reconciliation.audit_trail import (
    AuditTrailEngine,
    _safe_float,
    _safe_str,
    _extract_source_id,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> AuditTrailEngine:
    """Create a fresh AuditTrailEngine instance."""
    return AuditTrailEngine()


@pytest.fixture
def engine_with_events(engine) -> AuditTrailEngine:
    """Engine pre-populated with a mix of event types for job-001."""
    engine.record_event("job-001", "match", {"confidence": 0.95, "strategy": "exact"})
    engine.record_event("job-001", "match", {"confidence": 0.85, "strategy": "fuzzy"})
    engine.record_event("job-001", "comparison", {"field_name": "co2", "result": "mismatch"})
    engine.record_event(
        "job-001",
        "discrepancy_detected",
        {
            "discrepancy_id": "disc-001",
            "discrepancy_type": "value_mismatch",
            "severity": "high",
            "field_name": "co2",
            "source_a_value": "100",
            "source_b_value": "110",
            "deviation_pct": 10.0,
        },
    )
    engine.record_event(
        "job-001",
        "resolution_applied",
        {
            "resolution_id": "res-001",
            "discrepancy_id": "disc-001",
            "strategy": "priority_wins",
            "winning_source_id": "erp",
            "resolved_value": "100",
            "justification": "Highest credibility source",
            "confidence": 0.92,
        },
    )
    engine.record_event(
        "job-001",
        "golden_record_created",
        {
            "record_id": "gr-001",
            "entity_id": "facility-1",
            "period": "2025-Q1",
            "field_sources": {"co2": "erp"},
            "total_confidence": 0.9,
        },
    )
    # Events for a different job
    engine.record_event("job-002", "custom", {"note": "separate job"})
    return engine


@pytest.fixture
def mock_match_result():
    """Build a mock MatchResult dataclass."""
    mr = MagicMock()
    mr.match_id = "match-001"
    mr.confidence = 0.92
    mr.strategy = "exact"
    mr.status = "matched"
    mr.source_a_key = MagicMock(source_id="src-a")
    mr.source_b_key = MagicMock(source_id="src-b")
    mr.matched_fields = ["entity_id", "period"]
    return mr


@pytest.fixture
def mock_field_comparison():
    """Build a mock FieldComparison dataclass."""
    fc = MagicMock()
    fc.field_name = "co2_tonnes"
    fc.field_type = "numeric"
    fc.result = "mismatch"
    fc.source_a_value = 100.0
    fc.source_b_value = 110.0
    fc.absolute_diff = 10.0
    fc.relative_diff_pct = 10.0
    fc.tolerance_abs = 5.0
    fc.tolerance_pct = 5.0
    return fc


@pytest.fixture
def mock_discrepancy():
    """Build a mock Discrepancy dataclass."""
    d = MagicMock()
    d.discrepancy_id = "disc-001"
    d.match_id = "match-001"
    d.discrepancy_type = "value_mismatch"
    d.severity = "high"
    d.field_name = "co2_tonnes"
    d.source_a_value = 100.0
    d.source_b_value = 110.0
    d.deviation_pct = 10.0
    d.description = "Value mismatch detected"
    return d


@pytest.fixture
def mock_resolution_decision():
    """Build a mock ResolutionDecision dataclass."""
    rd = MagicMock()
    rd.resolution_id = "res-001"
    rd.discrepancy_id = "disc-001"
    rd.strategy = "priority_wins"
    rd.winning_source_id = "erp"
    rd.resolved_value = 100.0
    rd.justification = "Highest credibility source"
    rd.confidence = 0.92
    rd.reviewer = None
    return rd


@pytest.fixture
def mock_golden_record():
    """Build a mock GoldenRecord dataclass."""
    gr = MagicMock()
    gr.record_id = "gr-001"
    gr.entity_id = "facility-1"
    gr.period = "2025-Q1"
    gr.fields = {"co2_tonnes": 100.0, "region": "EU"}
    gr.field_sources = {"co2_tonnes": "erp", "region": "erp"}
    gr.field_confidences = {"co2_tonnes": 0.95, "region": 0.90}
    gr.total_confidence = 0.925
    return gr


# ===========================================================================
# Test Module-Level Helpers
# ===========================================================================


class TestModuleHelpers:
    """Tests for module-level helper functions."""

    def test_safe_str_none(self):
        assert _safe_str(None) == "null"

    def test_safe_str_dict(self):
        result = _safe_str({"a": 1})
        assert '"a"' in result

    def test_safe_str_list(self):
        result = _safe_str([1, 2, 3])
        assert "[1, 2, 3]" in result

    def test_safe_str_plain(self):
        assert _safe_str("hello") == "hello"

    def test_safe_float_valid(self):
        assert _safe_float(3.14) == 3.14

    def test_safe_float_string(self):
        assert _safe_float("2.5") == 2.5

    def test_safe_float_none(self):
        assert _safe_float(None, default=0.0) == 0.0

    def test_safe_float_invalid(self):
        assert _safe_float("abc", default=-1.0) == -1.0

    def test_extract_source_id_with_attr(self):
        obj = MagicMock(source_id="src-a")
        assert _extract_source_id(obj) == "src-a"

    def test_extract_source_id_none(self):
        assert _extract_source_id(None) == "unknown"

    def test_extract_source_id_string(self):
        assert _extract_source_id("my-source") == "my-source"


# ===========================================================================
# Test record_event
# ===========================================================================


class TestRecordEvent:
    """Tests for the record_event method."""

    def test_creates_event_with_uuid(self, engine):
        event = engine.record_event("job-001", "custom", {"note": "test"})
        assert event.event_id != ""
        assert len(event.event_id) > 0

    def test_creates_event_with_timestamp(self, engine):
        event = engine.record_event("job-001", "custom")
        assert event.timestamp is not None

    def test_stores_event_in_log(self, engine):
        assert engine.event_count == 0
        engine.record_event("job-001", "custom")
        assert engine.event_count == 1

    def test_provenance_hash_present(self, engine):
        event = engine.record_event("job-001", "custom")
        assert event.provenance_hash != ""
        assert len(event.provenance_hash) == 64

    def test_empty_job_id_raises(self, engine):
        with pytest.raises(ValueError, match="job_id"):
            engine.record_event("", "custom")

    def test_whitespace_job_id_raises(self, engine):
        with pytest.raises(ValueError, match="job_id"):
            engine.record_event("  ", "custom")

    def test_invalid_event_type_raises(self, engine):
        with pytest.raises(ValueError, match="Invalid event_type"):
            engine.record_event("job-001", "invalid_type")

    def test_all_valid_event_types(self, engine):
        """All valid event types can be recorded without error."""
        for etype in AuditTrailEngine.VALID_EVENT_TYPES:
            event = engine.record_event(f"job-{etype}", etype)
            assert event.event_type == etype

    def test_details_default_to_empty(self, engine):
        event = engine.record_event("job-001", "custom")
        assert event.details == {}

    def test_details_preserved(self, engine):
        event = engine.record_event("job-001", "custom", {"key": "value"})
        assert event.details["key"] == "value"

    def test_job_id_preserved(self, engine):
        event = engine.record_event("my-job", "custom")
        assert event.job_id == "my-job"


# ===========================================================================
# Test record_match_event
# ===========================================================================


class TestRecordMatchEvent:
    """Tests for record_match_event."""

    def test_extracts_match_details(self, engine, mock_match_result):
        event = engine.record_match_event("job-001", mock_match_result)
        assert event.event_type == "match"
        assert event.details["confidence"] == 0.92
        assert event.details["strategy"] == "exact"
        assert event.details["source_a_id"] == "src-a"
        assert event.details["source_b_id"] == "src-b"
        assert event.details["matched_fields"] == ["entity_id", "period"]

    def test_match_event_stored(self, engine, mock_match_result):
        engine.record_match_event("job-001", mock_match_result)
        assert engine.event_count == 1


# ===========================================================================
# Test record_comparison_event
# ===========================================================================


class TestRecordComparisonEvent:
    """Tests for record_comparison_event."""

    def test_stores_field_comparison(self, engine, mock_field_comparison):
        event = engine.record_comparison_event("job-001", mock_field_comparison)
        assert event.event_type == "comparison"
        assert event.details["field_name"] == "co2_tonnes"
        assert event.details["result"] == "mismatch"
        assert event.details["absolute_diff"] == 10.0
        assert event.details["relative_diff_pct"] == 10.0


# ===========================================================================
# Test record_discrepancy_event
# ===========================================================================


class TestRecordDiscrepancyEvent:
    """Tests for record_discrepancy_event."""

    def test_stores_type_and_severity(self, engine, mock_discrepancy):
        event = engine.record_discrepancy_event("job-001", mock_discrepancy)
        assert event.event_type == "discrepancy_detected"
        assert event.details["discrepancy_type"] == "value_mismatch"
        assert event.details["severity"] == "high"
        assert event.details["field_name"] == "co2_tonnes"
        assert event.details["deviation_pct"] == 10.0


# ===========================================================================
# Test record_resolution_event
# ===========================================================================


class TestRecordResolutionEvent:
    """Tests for record_resolution_event."""

    def test_stores_strategy_and_justification(self, engine, mock_resolution_decision):
        event = engine.record_resolution_event("job-001", mock_resolution_decision)
        assert event.event_type == "resolution_applied"
        assert event.details["strategy"] == "priority_wins"
        assert event.details["winning_source_id"] == "erp"
        assert event.details["justification"] == "Highest credibility source"
        assert event.details["confidence"] == 0.92


# ===========================================================================
# Test record_golden_record_event
# ===========================================================================


class TestRecordGoldenRecordEvent:
    """Tests for record_golden_record_event."""

    def test_stores_field_sources(self, engine, mock_golden_record):
        event = engine.record_golden_record_event("job-001", mock_golden_record)
        assert event.event_type == "golden_record_created"
        assert event.details["entity_id"] == "facility-1"
        assert event.details["period"] == "2025-Q1"
        assert "co2_tonnes" in event.details["field_sources"]
        assert event.details["total_confidence"] == pytest.approx(0.925)


# ===========================================================================
# Test generate_report
# ===========================================================================


class TestGenerateReport:
    """Tests for reconciliation summary report generation."""

    def test_correct_counts_and_summary(self, engine_with_events):
        report = engine_with_events.generate_report(
            job_id="job-001",
            total_records=100,
            matched_records=90,
            discrepancies_found=15,
            discrepancies_resolved=12,
            golden_records_created=90,
            unresolved_count=3,
        )
        assert report.job_id == "job-001"
        assert report.total_records == 100
        assert report.matched_records == 90
        assert report.discrepancies_found == 15
        assert report.discrepancies_resolved == 12
        assert report.golden_records_created == 90
        assert report.unresolved_count == 3

    def test_provenance_hash_present(self, engine):
        report = engine.generate_report(
            job_id="job-001",
            total_records=50,
            matched_records=50,
            discrepancies_found=0,
            discrepancies_resolved=0,
            golden_records_created=50,
            unresolved_count=0,
        )
        assert report.provenance_hash != ""
        assert len(report.provenance_hash) == 64

    def test_summary_contains_statistics(self, engine):
        report = engine.generate_report(
            job_id="job-001",
            total_records=100,
            matched_records=80,
            discrepancies_found=5,
            discrepancies_resolved=4,
            golden_records_created=80,
            unresolved_count=1,
        )
        assert "80/100" in report.summary
        assert "partial" in report.summary.lower() or "80.0%" in report.summary

    def test_empty_job_id_raises(self, engine):
        with pytest.raises(ValueError, match="job_id"):
            engine.generate_report("", 10, 10, 0, 0, 10, 0)

    def test_negative_records_raises(self, engine):
        with pytest.raises(ValueError, match="non-negative"):
            engine.generate_report("job-001", -1, 10, 0, 0, 10, 0)

    def test_zero_records_gives_completed(self, engine):
        report = engine.generate_report("job-001", 0, 0, 0, 0, 0, 0)
        assert "completed" in report.summary.lower()

    def test_no_matches_gives_failed(self, engine):
        report = engine.generate_report("job-001", 100, 0, 0, 0, 0, 0)
        assert "failed" in report.summary.lower()

    def test_all_resolved_gives_completed(self, engine):
        report = engine.generate_report("job-001", 100, 100, 10, 10, 100, 0)
        assert "completed" in report.summary.lower()


# ===========================================================================
# Test generate_compliance_report
# ===========================================================================


class TestGenerateComplianceReport:
    """Tests for framework-specific compliance reports."""

    def test_ghg_protocol_format(self, engine_with_events):
        report = engine_with_events.generate_compliance_report(
            "job-001", framework="ghg_protocol"
        )
        assert report["framework"] == "ghg_protocol"
        assert "data_quality_indicators" in report
        assert "gap_analysis" in report
        assert "estimation_methodology" in report
        assert report["provenance_hash"] != ""

    def test_csrd_esrs_format(self, engine_with_events):
        report = engine_with_events.generate_compliance_report(
            "job-001", framework="csrd_esrs"
        )
        assert report["framework"] == "csrd_esrs"
        assert "data_quality_score" in report
        assert "materiality_linkage" in report
        assert "double_materiality_indicators" in report
        assert report["provenance_hash"] != ""

    def test_unsupported_framework_raises(self, engine):
        with pytest.raises(ValueError, match="Unsupported framework"):
            engine.generate_compliance_report("job-001", framework="iso_14064")

    def test_empty_job_id_raises(self, engine):
        with pytest.raises(ValueError, match="job_id"):
            engine.generate_compliance_report("", framework="ghg_protocol")

    def test_ghg_protocol_counts(self, engine_with_events):
        report = engine_with_events.generate_compliance_report(
            "job-001", framework="ghg_protocol"
        )
        assert report["match_count"] == 2
        assert report["discrepancy_count"] == 1
        assert report["resolution_count"] == 1
        assert report["golden_record_count"] == 1

    def test_csrd_quality_score_structure(self, engine_with_events):
        report = engine_with_events.generate_compliance_report(
            "job-001", framework="csrd_esrs"
        )
        qs = report["data_quality_score"]
        assert "composite_score" in qs
        assert "completeness_score" in qs
        assert "accuracy_score" in qs
        assert "reliability_score" in qs
        assert "quality_level" in qs


# ===========================================================================
# Test generate_discrepancy_log
# ===========================================================================


class TestGenerateDiscrepancyLog:
    """Tests for chronological discrepancy log generation."""

    def test_chronological_order(self, engine_with_events):
        log = engine_with_events.generate_discrepancy_log("job-001")
        assert len(log) == 1
        assert log[0]["discrepancy_id"] == "disc-001"
        assert log[0]["severity"] == "high"

    def test_resolved_flag(self, engine_with_events):
        """Discrepancy that has a resolution is marked as resolved."""
        log = engine_with_events.generate_discrepancy_log("job-001")
        assert log[0]["resolved"] is True
        assert log[0]["resolution"]["strategy"] == "priority_wins"

    def test_unresolved_discrepancy(self, engine):
        """Discrepancy without resolution has resolved=False."""
        engine.record_event(
            "job-x",
            "discrepancy_detected",
            {"discrepancy_id": "disc-u", "severity": "medium"},
        )
        log = engine.generate_discrepancy_log("job-x")
        assert len(log) == 1
        assert log[0]["resolved"] is False
        assert log[0]["resolution"] is None

    def test_empty_job_returns_empty_list(self, engine):
        log = engine.generate_discrepancy_log("nonexistent-job")
        assert log == []

    def test_empty_job_id_raises(self, engine):
        with pytest.raises(ValueError, match="job_id"):
            engine.generate_discrepancy_log("")


# ===========================================================================
# Test get_events
# ===========================================================================


class TestGetEvents:
    """Tests for event querying with filters."""

    def test_filter_by_job(self, engine_with_events):
        events = engine_with_events.get_events(job_id="job-001")
        assert len(events) == 6
        for e in events:
            assert e.job_id == "job-001"

    def test_filter_by_type(self, engine_with_events):
        events = engine_with_events.get_events(event_type="match")
        # 2 from job-001
        assert len(events) == 2
        for e in events:
            assert e.event_type == "match"

    def test_filter_by_job_and_type(self, engine_with_events):
        events = engine_with_events.get_events(
            job_id="job-001", event_type="match"
        )
        assert len(events) == 2

    def test_filter_by_timestamp(self, engine_with_events):
        """Since filter returns events with timestamp >= since."""
        far_future = datetime(2099, 1, 1, tzinfo=timezone.utc)
        events = engine_with_events.get_events(since=far_future)
        assert len(events) == 0

    def test_no_filters_returns_all(self, engine_with_events):
        events = engine_with_events.get_events()
        assert len(events) == 7  # 6 for job-001 + 1 for job-002

    def test_invalid_event_type_raises(self, engine_with_events):
        with pytest.raises(ValueError, match="Invalid event_type"):
            engine_with_events.get_events(event_type="bogus")


# ===========================================================================
# Test get_event_count
# ===========================================================================


class TestGetEventCount:
    """Tests for event count aggregation."""

    def test_correct_total(self, engine_with_events):
        counts = engine_with_events.get_event_count()
        assert counts["total"] == 7

    def test_correct_per_type(self, engine_with_events):
        counts = engine_with_events.get_event_count(job_id="job-001")
        assert counts["match"] == 2
        assert counts["comparison"] == 1
        assert counts["discrepancy_detected"] == 1
        assert counts["resolution_applied"] == 1
        assert counts["golden_record_created"] == 1
        assert counts["total"] == 6

    def test_job_filter(self, engine_with_events):
        counts = engine_with_events.get_event_count(job_id="job-002")
        assert counts["total"] == 1
        assert counts["custom"] == 1


# ===========================================================================
# Test export_audit_trail
# ===========================================================================


class TestExportAuditTrail:
    """Tests for audit trail export."""

    def test_json_format(self, engine_with_events):
        result = engine_with_events.export_audit_trail("job-001", format="json")
        data = json.loads(result)
        assert data["job_id"] == "job-001"
        assert data["event_count"] == 6
        assert len(data["events"]) == 6

    def test_json_events_structure(self, engine_with_events):
        result = engine_with_events.export_audit_trail("job-001", format="json")
        data = json.loads(result)
        event = data["events"][0]
        assert "event_id" in event
        assert "job_id" in event
        assert "event_type" in event
        assert "timestamp" in event
        assert "details" in event
        assert "provenance_hash" in event

    def test_csv_format(self, engine_with_events):
        result = engine_with_events.export_audit_trail("job-001", format="csv")
        lines = result.strip().split("\n")
        assert len(lines) == 7  # 1 header + 6 data rows
        header = lines[0]
        assert "event_id" in header
        assert "event_type" in header

    def test_empty_job_id_raises(self, engine):
        with pytest.raises(ValueError, match="job_id"):
            engine.export_audit_trail("", format="json")

    def test_unsupported_format_raises(self, engine):
        with pytest.raises(ValueError, match="Unsupported export format"):
            engine.export_audit_trail("job-001", format="xml")

    def test_no_events_returns_empty_json(self, engine):
        result = engine.export_audit_trail("empty-job", format="json")
        data = json.loads(result)
        assert data["event_count"] == 0
        assert data["events"] == []


# ===========================================================================
# Test verify_audit_integrity
# ===========================================================================


class TestVerifyAuditIntegrity:
    """Tests for provenance chain integrity verification."""

    def test_passes_for_valid_chain(self, engine_with_events):
        result = engine_with_events.verify_audit_integrity("job-001")
        assert result is True

    def test_passes_for_empty_job(self, engine):
        """No events is vacuously valid."""
        result = engine.verify_audit_integrity("nonexistent")
        assert result is True

    def test_empty_job_id_raises(self, engine):
        with pytest.raises(ValueError, match="job_id"):
            engine.verify_audit_integrity("")

    def test_single_event_valid(self, engine):
        engine.record_event("job-x", "custom", {"a": 1})
        assert engine.verify_audit_integrity("job-x") is True


# ===========================================================================
# Test clear_events
# ===========================================================================


class TestClearEvents:
    """Tests for clearing stored events."""

    def test_clear_all(self, engine_with_events):
        count = engine_with_events.clear_events()
        assert count == 7
        assert engine_with_events.event_count == 0

    def test_clear_by_job(self, engine_with_events):
        count = engine_with_events.clear_events(job_id="job-001")
        assert count == 6
        assert engine_with_events.event_count == 1  # job-002 remains

    def test_clear_nonexistent_job(self, engine_with_events):
        count = engine_with_events.clear_events(job_id="nonexistent")
        assert count == 0
        assert engine_with_events.event_count == 7


# ===========================================================================
# Test Properties
# ===========================================================================


class TestProperties:
    """Tests for engine properties."""

    def test_event_count(self, engine):
        assert engine.event_count == 0
        engine.record_event("j1", "custom")
        assert engine.event_count == 1

    def test_provenance_chain_length(self, engine):
        initial = engine.provenance_chain_length
        engine.record_event("j1", "custom")
        assert engine.provenance_chain_length > initial


# ===========================================================================
# Test generate_resolution_justification
# ===========================================================================


class TestGenerateResolutionJustification:
    """Tests for per-discrepancy resolution justification."""

    def test_justification_structure(self, engine_with_events):
        result = engine_with_events.generate_resolution_justification("job-001")
        assert result["job_id"] == "job-001"
        assert result["total_resolutions"] == 1
        assert "justifications" in result
        assert "strategy_distribution" in result
        assert "provenance_hash" in result

    def test_justification_contains_strategy_rationale(self, engine_with_events):
        result = engine_with_events.generate_resolution_justification("job-001")
        j = result["justifications"][0]
        assert j["strategy"] == "priority_wins"
        assert "strategy_rationale" in j
        assert len(j["strategy_rationale"]) > 0

    def test_empty_job_returns_zero_resolutions(self, engine):
        result = engine.generate_resolution_justification("empty-job")
        assert result["total_resolutions"] == 0
        assert result["justifications"] == []

    def test_empty_job_id_raises(self, engine):
        with pytest.raises(ValueError, match="job_id"):
            engine.generate_resolution_justification("")


# ===========================================================================
# Test Multiple Jobs Isolation
# ===========================================================================


class TestMultipleJobsIsolation:
    """Tests that events from different jobs are properly isolated."""

    def test_events_isolated_by_job(self, engine_with_events):
        j1_events = engine_with_events.get_events(job_id="job-001")
        j2_events = engine_with_events.get_events(job_id="job-002")
        assert len(j1_events) == 6
        assert len(j2_events) == 1

    def test_report_only_counts_job_events(self, engine_with_events):
        counts = engine_with_events.get_event_count(job_id="job-002")
        assert counts["total"] == 1

    def test_clear_one_job_preserves_other(self, engine_with_events):
        engine_with_events.clear_events(job_id="job-002")
        assert engine_with_events.event_count == 6
        remaining = engine_with_events.get_events(job_id="job-001")
        assert len(remaining) == 6
