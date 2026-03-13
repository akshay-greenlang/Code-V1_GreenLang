# -*- coding: utf-8 -*-
"""
Unit tests for Continuous Monitoring Agent models - AGENT-EUDR-033

Tests all Pydantic v2 models, enums, constants, and model validation
for the Continuous Monitoring Agent's domain objects.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.continuous_monitoring.models import (
    AGENT_ID,
    AGENT_VERSION,
    EUDR_ARTICLES_MONITORED,
    ActionRecommendation,
    AlertSeverity,
    AlertStatus,
    AuditAction,
    AuditEntry,
    CertificationCheck,
    CertificationStatus,
    ChangeDetectionRecord,
    ChangeImpact,
    ChangeType,
    ComplianceAuditRecord,
    ComplianceCheckItem,
    ComplianceStatus,
    DataFreshnessRecord,
    DeforestationCorrelation,
    DeforestationMonitorRecord,
    FreshnessStatus,
    GeolocationShift,
    HealthStatus,
    InvestigationRecord,
    InvestigationStatus,
    MonitoringAlert,
    MonitoringScope,
    MonitoringSummary,
    RegulatoryImpact,
    RegulatoryTrackingRecord,
    RegulatoryUpdate,
    RiskLevel,
    RiskScoreMonitorRecord,
    RiskScoreSnapshot,
    ScanStatus,
    StaleEntity,
    SupplierChange,
    SupplyChainScanRecord,
    TrendDirection,
)


class TestConstants:
    """Test agent-level constants."""

    def test_agent_id(self):
        assert AGENT_ID == "GL-EUDR-CM-033"

    def test_agent_version(self):
        assert AGENT_VERSION == "1.0.0"

    def test_eudr_articles_monitored_count(self):
        assert len(EUDR_ARTICLES_MONITORED) == 8

    def test_eudr_articles_monitored_includes_article_4(self):
        assert "Article 4" in EUDR_ARTICLES_MONITORED

    def test_eudr_articles_monitored_includes_article_8(self):
        assert "Article 8" in EUDR_ARTICLES_MONITORED

    def test_eudr_articles_monitored_includes_article_10(self):
        assert "Article 10" in EUDR_ARTICLES_MONITORED

    def test_eudr_articles_monitored_includes_article_29(self):
        assert "Article 29" in EUDR_ARTICLES_MONITORED


class TestEnums:
    """Test all enum definitions."""

    def test_monitoring_scope_values(self):
        assert len(MonitoringScope) == 7
        assert MonitoringScope.SUPPLY_CHAIN.value == "supply_chain"
        assert MonitoringScope.DEFORESTATION.value == "deforestation"
        assert MonitoringScope.COMPLIANCE.value == "compliance"
        assert MonitoringScope.RISK.value == "risk"
        assert MonitoringScope.DATA_FRESHNESS.value == "data_freshness"
        assert MonitoringScope.REGULATORY.value == "regulatory"
        assert MonitoringScope.CHANGE_DETECTION.value == "change_detection"

    def test_scan_status_values(self):
        assert len(ScanStatus) == 5
        assert ScanStatus.PENDING.value == "pending"
        assert ScanStatus.RUNNING.value == "running"
        assert ScanStatus.COMPLETED.value == "completed"
        assert ScanStatus.FAILED.value == "failed"
        assert ScanStatus.PARTIAL.value == "partial"

    def test_alert_severity_values(self):
        assert len(AlertSeverity) == 4
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.HIGH.value == "high"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_alert_status_values(self):
        assert len(AlertStatus) == 5
        assert AlertStatus.OPEN.value == "open"
        assert AlertStatus.ACKNOWLEDGED.value == "acknowledged"
        assert AlertStatus.INVESTIGATING.value == "investigating"
        assert AlertStatus.RESOLVED.value == "resolved"
        assert AlertStatus.DISMISSED.value == "dismissed"

    def test_certification_status_values(self):
        assert len(CertificationStatus) == 5
        assert CertificationStatus.VALID.value == "valid"
        assert CertificationStatus.EXPIRING_SOON.value == "expiring_soon"
        assert CertificationStatus.EXPIRED.value == "expired"
        assert CertificationStatus.REVOKED.value == "revoked"
        assert CertificationStatus.PENDING_RENEWAL.value == "pending_renewal"

    def test_change_type_values(self):
        assert len(ChangeType) == 8
        assert ChangeType.SUPPLIER_STATUS.value == "supplier_status"
        assert ChangeType.CERTIFICATION.value == "certification"
        assert ChangeType.GEOLOCATION.value == "geolocation"
        assert ChangeType.OWNERSHIP.value == "ownership"

    def test_change_impact_values(self):
        assert len(ChangeImpact) == 5
        assert ChangeImpact.NEGLIGIBLE.value == "negligible"
        assert ChangeImpact.LOW.value == "low"
        assert ChangeImpact.MODERATE.value == "moderate"
        assert ChangeImpact.HIGH.value == "high"
        assert ChangeImpact.CRITICAL.value == "critical"

    def test_compliance_status_values(self):
        assert len(ComplianceStatus) == 5
        assert ComplianceStatus.COMPLIANT.value == "compliant"
        assert ComplianceStatus.NON_COMPLIANT.value == "non_compliant"
        assert ComplianceStatus.PARTIALLY_COMPLIANT.value == "partially_compliant"
        assert ComplianceStatus.PENDING_REVIEW.value == "pending_review"
        assert ComplianceStatus.EXPIRED.value == "expired"

    def test_trend_direction_values(self):
        assert len(TrendDirection) == 3
        assert TrendDirection.IMPROVING.value == "improving"
        assert TrendDirection.STABLE.value == "stable"
        assert TrendDirection.WORSENING.value == "worsening"

    def test_freshness_status_values(self):
        assert len(FreshnessStatus) == 4
        assert FreshnessStatus.FRESH.value == "fresh"
        assert FreshnessStatus.STALE_WARNING.value == "stale_warning"
        assert FreshnessStatus.STALE_CRITICAL.value == "stale_critical"
        assert FreshnessStatus.UNKNOWN.value == "unknown"

    def test_regulatory_impact_values(self):
        assert len(RegulatoryImpact) == 5
        assert RegulatoryImpact.NONE.value == "none"
        assert RegulatoryImpact.LOW.value == "low"
        assert RegulatoryImpact.MODERATE.value == "moderate"
        assert RegulatoryImpact.HIGH.value == "high"
        assert RegulatoryImpact.BREAKING.value == "breaking"

    def test_risk_level_values(self):
        assert len(RiskLevel) == 5
        assert RiskLevel.NEGLIGIBLE.value == "negligible"
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MODERATE.value == "moderate"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_investigation_status_values(self):
        assert len(InvestigationStatus) == 5
        assert InvestigationStatus.PENDING.value == "pending"
        assert InvestigationStatus.IN_PROGRESS.value == "in_progress"
        assert InvestigationStatus.COMPLETED.value == "completed"
        assert InvestigationStatus.ESCALATED.value == "escalated"
        assert InvestigationStatus.CLOSED.value == "closed"

    def test_audit_action_values(self):
        assert len(AuditAction) == 11
        assert AuditAction.SCAN.value == "scan"
        assert AuditAction.DETECT.value == "detect"
        assert AuditAction.ALERT.value == "alert"
        assert AuditAction.INVESTIGATE.value == "investigate"
        assert AuditAction.VERIFY.value == "verify"
        assert AuditAction.ASSESS.value == "assess"
        assert AuditAction.TRACK.value == "track"
        assert AuditAction.NOTIFY.value == "notify"
        assert AuditAction.REFRESH.value == "refresh"
        assert AuditAction.CORRELATE.value == "correlate"
        assert AuditAction.AUDIT.value == "audit"


class TestSupplyChainScanRecord:
    """Test SupplyChainScanRecord model."""

    def test_create(self):
        record = SupplyChainScanRecord(
            scan_id="SCM-001",
            operator_id="OP-001",
        )
        assert record.scan_id == "SCM-001"
        assert record.operator_id == "OP-001"

    def test_defaults(self):
        record = SupplyChainScanRecord(
            scan_id="SCM-X", operator_id="OP-X",
        )
        assert record.suppliers_scanned == 0
        assert record.changes_detected == 0
        assert record.scan_status == ScanStatus.PENDING
        assert record.provenance_hash == ""

    def test_supplier_changes_default_empty(self):
        record = SupplyChainScanRecord(scan_id="SCM-X", operator_id="OP-X")
        assert record.supplier_changes == []
        assert record.certification_checks == []
        assert record.geolocation_shifts == []


class TestDeforestationMonitorRecord:
    """Test DeforestationMonitorRecord model."""

    def test_create(self):
        record = DeforestationMonitorRecord(
            monitor_id="DM-001",
            operator_id="OP-001",
        )
        assert record.monitor_id == "DM-001"

    def test_defaults(self):
        record = DeforestationMonitorRecord(
            monitor_id="DM-X", operator_id="OP-X",
        )
        assert record.alerts_checked == 0
        assert record.correlations_found == 0
        assert record.investigations_triggered == 0
        assert record.total_area_affected_hectares == Decimal("0")


class TestComplianceAuditRecord:
    """Test ComplianceAuditRecord model."""

    def test_create(self, sample_compliance_audit):
        assert sample_compliance_audit.audit_id == "CA-001"
        assert sample_compliance_audit.compliance_status == ComplianceStatus.PARTIALLY_COMPLIANT

    def test_defaults(self):
        audit = ComplianceAuditRecord(
            audit_id="CA-X", operator_id="OP-X",
        )
        assert audit.compliance_status == ComplianceStatus.PENDING_REVIEW
        assert audit.overall_score == Decimal("0")
        assert audit.checks_total == 0

    def test_check_items_default_empty(self):
        audit = ComplianceAuditRecord(audit_id="CA-X", operator_id="OP-X")
        assert audit.check_items == []
        assert audit.recommendations == []


class TestChangeDetectionRecord:
    """Test ChangeDetectionRecord model."""

    def test_create(self):
        record = ChangeDetectionRecord(
            detection_id="CHG-001",
            operator_id="OP-001",
            entity_id="SUP-001",
        )
        assert record.detection_id == "CHG-001"

    def test_defaults(self):
        record = ChangeDetectionRecord(
            detection_id="CHG-X",
            operator_id="OP-X",
            entity_id="E-X",
        )
        assert record.change_type == ChangeType.SUPPLIER_STATUS
        assert record.change_impact == ChangeImpact.LOW
        assert record.old_state == {}
        assert record.new_state == {}

    def test_entity_type_default(self):
        record = ChangeDetectionRecord(
            detection_id="CHG-X", operator_id="OP-X", entity_id="E-X",
        )
        assert record.entity_type == "supplier"


class TestRiskScoreMonitorRecord:
    """Test RiskScoreMonitorRecord model."""

    def test_create(self):
        record = RiskScoreMonitorRecord(
            monitor_id="RSM-001",
            operator_id="OP-001",
            entity_id="SUP-001",
        )
        assert record.monitor_id == "RSM-001"

    def test_defaults(self):
        record = RiskScoreMonitorRecord(
            monitor_id="RSM-X",
            operator_id="OP-X",
            entity_id="E-X",
        )
        assert record.current_score == Decimal("0")
        assert record.previous_score == Decimal("0")
        assert record.trend_direction == TrendDirection.STABLE
        assert record.degradation_detected is False
        assert record.risk_level == RiskLevel.LOW

    def test_trend_snapshots_default_empty(self):
        record = RiskScoreMonitorRecord(
            monitor_id="RSM-X", operator_id="OP-X", entity_id="E-X",
        )
        assert record.trend_snapshots == []
        assert record.correlated_incidents == []


class TestDataFreshnessRecord:
    """Test DataFreshnessRecord model."""

    def test_create(self):
        record = DataFreshnessRecord(
            freshness_id="DF-001",
            operator_id="OP-001",
        )
        assert record.freshness_id == "DF-001"

    def test_defaults(self):
        record = DataFreshnessRecord(
            freshness_id="DF-X", operator_id="OP-X",
        )
        assert record.entities_checked == 0
        assert record.fresh_count == 0
        assert record.freshness_percentage == Decimal("0")
        assert record.meets_target is False
        assert record.stale_entities == []


class TestRegulatoryTrackingRecord:
    """Test RegulatoryTrackingRecord model."""

    def test_create(self):
        record = RegulatoryTrackingRecord(
            tracking_id="RT-001",
            operator_id="OP-001",
        )
        assert record.tracking_id == "RT-001"

    def test_defaults(self):
        record = RegulatoryTrackingRecord(
            tracking_id="RT-X", operator_id="OP-X",
        )
        assert record.updates_found == 0
        assert record.high_impact_count == 0
        assert record.regulatory_updates == []
        assert record.notifications_sent == 0


class TestMonitoringAlert:
    """Test MonitoringAlert model."""

    def test_create(self):
        alert = MonitoringAlert(
            alert_id="ALT-001",
            operator_id="OP-001",
        )
        assert alert.alert_id == "ALT-001"
        assert alert.severity == AlertSeverity.INFO
        assert alert.alert_status == AlertStatus.OPEN


class TestAuditEntry:
    """Test AuditEntry model."""

    def test_create(self):
        entry = AuditEntry(
            entry_id="AUD-001",
            entity_type="supply_chain_scan",
            entity_id="SCE-001",
            actor="AGENT-EUDR-033",
        )
        assert entry.action == AuditAction.SCAN

    def test_action_override(self):
        entry = AuditEntry(
            entry_id="AUD-002",
            entity_type="compliance_audit",
            entity_id="CA-001",
            actor="user:admin",
            action=AuditAction.AUDIT,
        )
        assert entry.action == AuditAction.AUDIT


class TestHealthStatus:
    """Test HealthStatus model."""

    def test_defaults(self):
        health = HealthStatus()
        assert health.agent_id == AGENT_ID
        assert health.status == "healthy"
        assert health.version == AGENT_VERSION


class TestSubModels:
    """Test sub-models used within core records."""

    def test_supplier_change(self):
        change = SupplierChange(
            supplier_id="SUP-001",
            field_changed="status",
            old_value="active",
            new_value="suspended",
        )
        assert change.supplier_id == "SUP-001"
        assert change.field_changed == "status"

    def test_certification_check(self):
        check = CertificationCheck(
            certification_id="CERT-001",
            supplier_id="SUP-001",
        )
        assert check.status == CertificationStatus.VALID

    def test_geolocation_shift(self):
        shift = GeolocationShift(entity_id="PLOT-001")
        assert shift.is_stable is True
        assert shift.drift_km == Decimal("0")

    def test_deforestation_correlation(self):
        corr = DeforestationCorrelation(
            alert_id="A-001",
            entity_id="P-001",
        )
        assert corr.confidence == Decimal("0")

    def test_stale_entity(self):
        stale = StaleEntity(entity_id="SUP-001")
        assert stale.freshness_status == FreshnessStatus.UNKNOWN

    def test_regulatory_update(self):
        update = RegulatoryUpdate(update_id="U-001")
        assert update.impact_level == RegulatoryImpact.NONE

    def test_risk_score_snapshot(self):
        snap = RiskScoreSnapshot(entity_id="E-001")
        assert snap.score == Decimal("0")
        assert snap.risk_level == RiskLevel.LOW

    def test_compliance_check_item(self):
        item = ComplianceCheckItem(
            check_id="CHK-001",
            article_reference="Article 8",
        )
        assert item.status == ComplianceStatus.PENDING_REVIEW

    def test_action_recommendation(self):
        rec = ActionRecommendation(action="Review supplier status")
        assert rec.priority == "medium"
        assert rec.deadline_days == 30

    def test_investigation_record(self):
        inv = InvestigationRecord(
            investigation_id="INV-001",
            operator_id="OP-001",
            trigger_alert_id="ALT-001",
        )
        assert inv.investigation_status == InvestigationStatus.PENDING

    def test_monitoring_summary(self):
        summary = MonitoringSummary(
            summary_id="SUM-001",
            operator_id="OP-001",
        )
        assert summary.total_alerts == 0
        assert summary.overall_risk_level == RiskLevel.LOW
