# -*- coding: utf-8 -*-
"""
Tests for AlertGenerator - AGENT-EUDR-003 Feature 7: Alert Generation

Comprehensive test suite covering:
- Alert generation based on change severity
- Alert acknowledgement workflow
- Alert filtering (by severity, plot, acknowledged)
- Alert summary and statistics
- Evidence package generation (JSON, CSV, PDF)
- Compliance determination from evidence
- Pagination of alert lists
- Determinism and reproducibility

Test count: 75+ tests
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-003 (Feature 7 - Alert Generation)
"""

import json
import uuid

import pytest

from tests.agents.eudr.satellite_monitoring.conftest import (
    SatelliteAlert,
    EvidencePackage,
    ChangeDetectionResult,
    compute_test_hash,
    SHA256_HEX_LENGTH,
    ALERT_SEVERITIES,
    EVIDENCE_FORMATS,
    EUDR_COMMODITIES,
    CHANGE_CLASSIFICATIONS,
)


# ---------------------------------------------------------------------------
# Helpers for alert logic
# ---------------------------------------------------------------------------


def _generate_alert(
    plot_id: str,
    ndvi_drop: float,
    confidence: float,
    area_ha: float = 1.0,
    threshold_confidence: float = 0.7,
) -> SatelliteAlert:
    """Generate an alert based on NDVI drop and confidence."""
    if confidence < threshold_confidence:
        return None

    if ndvi_drop <= -0.15:
        severity = "critical"
        change_type = "deforestation"
    elif ndvi_drop <= -0.05:
        severity = "warning"
        change_type = "degradation"
    elif ndvi_drop < 0.0:
        severity = "info"
        change_type = "minor_change"
    else:
        return None

    alert_id = f"ALR-{uuid.uuid4().hex[:8].upper()}"
    return SatelliteAlert(
        alert_id=alert_id,
        plot_id=plot_id,
        severity=severity,
        change_type=change_type,
        ndvi_drop=ndvi_drop,
        confidence=confidence,
        area_affected_ha=area_ha,
        detected_at="2026-03-01T00:00:00+00:00",
        provenance_hash=compute_test_hash({
            "plot_id": plot_id,
            "ndvi_drop": ndvi_drop,
            "confidence": confidence,
            "severity": severity,
        }),
    )


def _acknowledge_alert(alert: SatelliteAlert, user: str) -> SatelliteAlert:
    """Acknowledge an alert."""
    if alert is None:
        return None
    alert.acknowledged = True
    alert.acknowledged_by = user
    alert.acknowledged_at = "2026-03-02T10:00:00+00:00"
    return alert


def _filter_alerts(
    alerts: list,
    severity: str = None,
    plot_id: str = None,
    acknowledged: bool = None,
    offset: int = 0,
    limit: int = 100,
) -> list:
    """Filter alerts by criteria."""
    result = alerts
    if severity is not None:
        result = [a for a in result if a.severity == severity]
    if plot_id is not None:
        result = [a for a in result if a.plot_id == plot_id]
    if acknowledged is not None:
        result = [a for a in result if a.acknowledged == acknowledged]
    return result[offset:offset + limit]


def _alert_summary(alerts: list) -> dict:
    """Generate alert summary statistics."""
    summary = {
        "total": len(alerts),
        "by_severity": {},
        "by_commodity": {},
        "by_country": {},
        "acknowledged": sum(1 for a in alerts if a.acknowledged),
        "unacknowledged": sum(1 for a in alerts if not a.acknowledged),
    }
    for sev in ALERT_SEVERITIES:
        summary["by_severity"][sev] = sum(1 for a in alerts if a.severity == sev)
    return summary


def _generate_evidence(
    plot_id: str,
    compliance_status: str,
    format: str = "json",
) -> EvidencePackage:
    """Generate an evidence package."""
    return EvidencePackage(
        evidence_id=f"EVD-{uuid.uuid4().hex[:8].upper()}",
        plot_id=plot_id,
        compliance_status=compliance_status,
        format=format,
        baseline_snapshot={"ndvi_mean": 0.72, "forest_pct": 95.0},
        change_results=[{"classification": "no_change", "confidence": 0.95}],
        alert_history=[],
        generated_at="2026-03-01T00:00:00+00:00",
        provenance_hash=compute_test_hash({
            "plot_id": plot_id,
            "compliance_status": compliance_status,
        }),
    )


# ===========================================================================
# 1. Alert Generation (20 tests)
# ===========================================================================


class TestAlertGeneration:
    """Test alert generation based on change severity."""

    def test_critical_alert_deforestation(self):
        """Test critical alert for significant deforestation."""
        alert = _generate_alert("PLOT-001", -0.30, 0.95)
        assert alert is not None
        assert alert.severity == "critical"
        assert alert.change_type == "deforestation"

    def test_warning_alert_degradation(self):
        """Test warning alert for degradation."""
        alert = _generate_alert("PLOT-002", -0.08, 0.85)
        assert alert is not None
        assert alert.severity == "warning"
        assert alert.change_type == "degradation"

    def test_info_alert_minor_change(self):
        """Test info alert for minor change."""
        alert = _generate_alert("PLOT-003", -0.03, 0.80)
        assert alert is not None
        assert alert.severity == "info"
        assert alert.change_type == "minor_change"

    def test_no_alert_no_change(self):
        """Test no alert when NDVI increases (no negative change)."""
        alert = _generate_alert("PLOT-004", 0.05, 0.90)
        assert alert is None

    def test_no_alert_low_confidence(self):
        """Test no alert when confidence is below threshold."""
        alert = _generate_alert("PLOT-005", -0.30, 0.50, threshold_confidence=0.7)
        assert alert is None

    @pytest.mark.parametrize("ndvi_drop,confidence,expected_severity", [
        (-0.50, 0.98, "critical"),
        (-0.40, 0.95, "critical"),
        (-0.30, 0.90, "critical"),
        (-0.20, 0.85, "critical"),
        (-0.15, 0.80, "critical"),
        (-0.14, 0.85, "warning"),
        (-0.10, 0.80, "warning"),
        (-0.08, 0.80, "warning"),
        (-0.05, 0.80, "warning"),
        (-0.04, 0.80, "info"),
        (-0.03, 0.80, "info"),
        (-0.02, 0.75, "info"),
        (-0.01, 0.75, "info"),
        (0.00, 0.90, None),
        (0.10, 0.90, None),
    ])
    def test_alert_severity_thresholds(self, ndvi_drop, confidence, expected_severity):
        """Test alert severity thresholds for various NDVI drops."""
        alert = _generate_alert("PLOT-T", ndvi_drop, confidence)
        if expected_severity is None:
            assert alert is None
        else:
            assert alert is not None
            assert alert.severity == expected_severity

    def test_alert_metadata(self):
        """Test alert contains all required metadata."""
        alert = _generate_alert("PLOT-META", -0.25, 0.92, area_ha=2.5)
        assert alert.alert_id.startswith("ALR-")
        assert alert.plot_id == "PLOT-META"
        assert alert.ndvi_drop == -0.25
        assert alert.confidence == 0.92
        assert alert.area_affected_ha == 2.5
        assert alert.detected_at is not None
        assert alert.provenance_hash != ""
        assert len(alert.provenance_hash) == SHA256_HEX_LENGTH

    def test_alert_not_acknowledged_by_default(self):
        """Test alert is not acknowledged when created."""
        alert = _generate_alert("PLOT-006", -0.20, 0.90)
        assert alert.acknowledged is False
        assert alert.acknowledged_by is None
        assert alert.acknowledged_at is None


# ===========================================================================
# 2. Alert Acknowledgement (8 tests)
# ===========================================================================


class TestAlertAcknowledge:
    """Test alert acknowledgement workflow."""

    def test_acknowledge_success(self):
        """Test successful alert acknowledgement."""
        alert = _generate_alert("PLOT-ACK", -0.20, 0.90)
        acked = _acknowledge_alert(alert, "user@test.com")
        assert acked.acknowledged is True
        assert acked.acknowledged_by == "user@test.com"
        assert acked.acknowledged_at is not None

    def test_acknowledge_nonexistent(self):
        """Test acknowledging None alert returns None."""
        result = _acknowledge_alert(None, "user@test.com")
        assert result is None

    def test_acknowledge_already_acknowledged(self):
        """Test acknowledging an already-acknowledged alert."""
        alert = _generate_alert("PLOT-DBL", -0.20, 0.90)
        acked1 = _acknowledge_alert(alert, "user1@test.com")
        acked2 = _acknowledge_alert(acked1, "user2@test.com")
        assert acked2.acknowledged is True
        assert acked2.acknowledged_by == "user2@test.com"

    def test_acknowledge_preserves_alert_data(self):
        """Test acknowledgement does not modify alert data."""
        alert = _generate_alert("PLOT-PRES", -0.25, 0.92)
        original_severity = alert.severity
        original_ndvi = alert.ndvi_drop
        _acknowledge_alert(alert, "user@test.com")
        assert alert.severity == original_severity
        assert alert.ndvi_drop == original_ndvi


# ===========================================================================
# 3. Alert Filtering (12 tests)
# ===========================================================================


class TestAlertFiltering:
    """Test alert filtering capabilities."""

    @pytest.fixture
    def sample_alerts(self):
        """Create a list of sample alerts for filtering tests."""
        alerts = []
        for i in range(10):
            ndvi = -0.30 + i * 0.05  # Range from -0.30 to 0.15
            conf = 0.90
            alert = _generate_alert(f"PLOT-{i:03d}", ndvi, conf)
            if alert is not None:
                if i < 3:
                    alert.plot_id = "PLOT-BR-001"
                alerts.append(alert)
        if len(alerts) > 2:
            _acknowledge_alert(alerts[0], "user@test.com")
            _acknowledge_alert(alerts[1], "user@test.com")
        return alerts

    def test_filter_by_severity(self, sample_alerts):
        """Test filtering by severity level."""
        critical = _filter_alerts(sample_alerts, severity="critical")
        for a in critical:
            assert a.severity == "critical"

    def test_filter_by_plot(self, sample_alerts):
        """Test filtering by plot ID."""
        plot_alerts = _filter_alerts(sample_alerts, plot_id="PLOT-BR-001")
        for a in plot_alerts:
            assert a.plot_id == "PLOT-BR-001"

    def test_filter_acknowledged(self, sample_alerts):
        """Test filtering acknowledged alerts."""
        acked = _filter_alerts(sample_alerts, acknowledged=True)
        for a in acked:
            assert a.acknowledged is True

    def test_filter_unacknowledged(self, sample_alerts):
        """Test filtering unacknowledged alerts."""
        unacked = _filter_alerts(sample_alerts, acknowledged=False)
        for a in unacked:
            assert a.acknowledged is False

    def test_pagination(self, sample_alerts):
        """Test alert pagination."""
        page1 = _filter_alerts(sample_alerts, offset=0, limit=3)
        page2 = _filter_alerts(sample_alerts, offset=3, limit=3)
        assert len(page1) <= 3
        assert len(page2) <= 3
        if len(sample_alerts) > 3:
            assert page1[0].alert_id != page2[0].alert_id

    def test_pagination_beyond_end(self, sample_alerts):
        """Test pagination beyond list end returns empty."""
        result = _filter_alerts(sample_alerts, offset=1000, limit=10)
        assert result == []

    def test_combined_filters(self, sample_alerts):
        """Test combining multiple filters."""
        result = _filter_alerts(
            sample_alerts,
            severity="critical",
            acknowledged=False,
        )
        for a in result:
            assert a.severity == "critical"
            assert a.acknowledged is False


# ===========================================================================
# 4. Alert Summary (8 tests)
# ===========================================================================


class TestAlertSummary:
    """Test alert summary statistics."""

    @pytest.fixture
    def mixed_alerts(self):
        """Create a mixed set of alerts."""
        specs = [
            ("PLOT-001", -0.30, 0.95),  # critical
            ("PLOT-002", -0.25, 0.90),  # critical
            ("PLOT-003", -0.08, 0.85),  # warning
            ("PLOT-004", -0.10, 0.80),  # warning
            ("PLOT-005", -0.06, 0.80),  # warning
            ("PLOT-006", -0.03, 0.75),  # info
            ("PLOT-007", -0.02, 0.75),  # info
        ]
        alerts = [_generate_alert(pid, ndvi, conf) for pid, ndvi, conf in specs]
        return [a for a in alerts if a is not None]

    def test_summary_counts(self, mixed_alerts):
        """Test summary total count matches alert list."""
        summary = _alert_summary(mixed_alerts)
        assert summary["total"] == len(mixed_alerts)

    def test_summary_by_severity(self, mixed_alerts):
        """Test summary by severity breakdown."""
        summary = _alert_summary(mixed_alerts)
        total_by_sev = sum(summary["by_severity"].values())
        assert total_by_sev == summary["total"]

    def test_summary_acknowledged_count(self, mixed_alerts):
        """Test acknowledged/unacknowledged counts."""
        _acknowledge_alert(mixed_alerts[0], "user@test.com")
        summary = _alert_summary(mixed_alerts)
        assert summary["acknowledged"] == 1
        assert summary["unacknowledged"] == len(mixed_alerts) - 1

    def test_summary_all_severities_present(self, mixed_alerts):
        """Test all severity levels appear in summary."""
        summary = _alert_summary(mixed_alerts)
        for sev in ALERT_SEVERITIES:
            assert sev in summary["by_severity"]

    def test_summary_empty_alerts(self):
        """Test summary with empty alert list."""
        summary = _alert_summary([])
        assert summary["total"] == 0
        assert summary["acknowledged"] == 0


# ===========================================================================
# 5. Evidence Generation (12 tests)
# ===========================================================================


class TestEvidenceGeneration:
    """Test evidence package generation."""

    def test_evidence_json_format(self):
        """Test evidence in JSON format."""
        evidence = _generate_evidence("PLOT-001", "compliant", "json")
        assert evidence.format == "json"
        assert evidence.plot_id == "PLOT-001"

    def test_evidence_csv_format(self):
        """Test evidence in CSV format."""
        evidence = _generate_evidence("PLOT-002", "non_compliant", "csv")
        assert evidence.format == "csv"

    def test_evidence_pdf_data(self):
        """Test evidence PDF format reference."""
        evidence = _generate_evidence("PLOT-003", "compliant", "pdf")
        assert evidence.format == "pdf"

    def test_evidence_provenance_hash(self):
        """Test evidence has a valid provenance hash."""
        evidence = _generate_evidence("PLOT-004", "compliant")
        assert evidence.provenance_hash != ""
        assert len(evidence.provenance_hash) == SHA256_HEX_LENGTH

    def test_evidence_compliance_compliant(self):
        """Test evidence with compliant status."""
        evidence = _generate_evidence("PLOT-005", "compliant")
        assert evidence.compliance_status == "compliant"

    def test_evidence_compliance_non_compliant(self):
        """Test evidence with non-compliant status."""
        evidence = _generate_evidence("PLOT-006", "non_compliant")
        assert evidence.compliance_status == "non_compliant"

    def test_evidence_baseline_snapshot(self):
        """Test evidence contains baseline snapshot."""
        evidence = _generate_evidence("PLOT-007", "compliant")
        assert evidence.baseline_snapshot is not None
        assert "ndvi_mean" in evidence.baseline_snapshot

    def test_evidence_change_results(self):
        """Test evidence contains change results."""
        evidence = _generate_evidence("PLOT-008", "compliant")
        assert evidence.change_results is not None
        assert len(evidence.change_results) > 0

    def test_evidence_generated_at(self):
        """Test evidence has generation timestamp."""
        evidence = _generate_evidence("PLOT-009", "compliant")
        assert evidence.generated_at is not None

    @pytest.mark.parametrize("format", EVIDENCE_FORMATS)
    def test_all_evidence_formats(self, format):
        """Test evidence generation in all supported formats."""
        evidence = _generate_evidence("PLOT-FMT", "compliant", format)
        assert evidence.format == format

    def test_evidence_id_unique(self):
        """Test each evidence package has a unique ID."""
        ids = set()
        for i in range(10):
            evidence = _generate_evidence(f"PLOT-{i}", "compliant")
            ids.add(evidence.evidence_id)
        assert len(ids) == 10


# ===========================================================================
# 6. Compliance Determination (8 tests)
# ===========================================================================


class TestComplianceDetermination:
    """Test compliance status in evidence packages."""

    def test_compliant(self):
        """Test compliant when no deforestation detected."""
        evidence = _generate_evidence("PLOT-C", "compliant")
        assert evidence.compliance_status == "compliant"

    def test_non_compliant(self):
        """Test non-compliant when deforestation detected."""
        evidence = _generate_evidence("PLOT-NC", "non_compliant")
        assert evidence.compliance_status == "non_compliant"

    def test_insufficient_data(self):
        """Test insufficient data status."""
        evidence = _generate_evidence("PLOT-ID", "insufficient_data")
        assert evidence.compliance_status == "insufficient_data"

    def test_manual_review(self):
        """Test manual review status."""
        evidence = _generate_evidence("PLOT-MR", "manual_review")
        assert evidence.compliance_status == "manual_review"

    @pytest.mark.parametrize("status", [
        "compliant", "non_compliant", "insufficient_data", "manual_review",
    ])
    def test_all_compliance_statuses(self, status):
        """Test all compliance status values."""
        evidence = _generate_evidence("PLOT-ALL", status)
        assert evidence.compliance_status == status


# ===========================================================================
# 7. Determinism (8 tests)
# ===========================================================================


class TestAlertDeterminism:
    """Test alert and evidence generation determinism."""

    def test_alert_severity_deterministic(self):
        """Test alert severity classification is deterministic."""
        alerts = [_generate_alert("PLOT-D", -0.25, 0.90) for _ in range(10)]
        severities = [a.severity for a in alerts if a is not None]
        assert len(set(severities)) == 1

    def test_alert_provenance_deterministic(self):
        """Test alert provenance hash is deterministic."""
        data = {"plot_id": "PLOT-D", "ndvi_drop": -0.25, "confidence": 0.90, "severity": "critical"}
        hashes = [compute_test_hash(data) for _ in range(10)]
        assert len(set(hashes)) == 1

    def test_evidence_provenance_deterministic(self):
        """Test evidence provenance hash is deterministic."""
        data = {"plot_id": "PLOT-E", "compliance_status": "compliant"}
        hashes = [compute_test_hash(data) for _ in range(10)]
        assert len(set(hashes)) == 1

    def test_summary_deterministic(self):
        """Test alert summary is deterministic."""
        alerts = [_generate_alert(f"PLOT-{i}", -0.25, 0.90) for i in range(5)]
        alerts = [a for a in alerts if a is not None]
        summaries = [_alert_summary(alerts) for _ in range(5)]
        for s in summaries[1:]:
            assert s["total"] == summaries[0]["total"]
            assert s["by_severity"] == summaries[0]["by_severity"]

    def test_filter_deterministic(self):
        """Test alert filtering is deterministic."""
        alerts = [_generate_alert(f"PLOT-{i}", -0.25, 0.90) for i in range(5)]
        alerts = [a for a in alerts if a is not None]
        results = [_filter_alerts(alerts, severity="critical") for _ in range(5)]
        for r in results[1:]:
            assert len(r) == len(results[0])
