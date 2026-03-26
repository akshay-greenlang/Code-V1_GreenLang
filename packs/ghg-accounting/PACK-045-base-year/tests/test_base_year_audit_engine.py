# -*- coding: utf-8 -*-
"""
Tests for BaseYearAuditEngine (Engine 9).

Covers audit entries, verification packages, ISAE 3410 compliance,
audit trail export, and approval records.
Target: ~40 tests.
"""

import pytest
from decimal import Decimal
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from engines.base_year_audit_engine import (
    BaseYearAuditEngine,
    AuditEntry,
    AuditTrail,
    AuditTrailFilter,
    VerificationPackage,
    ApprovalRecord,
    AuditEventType,
    AuditSeverity,
    VerificationLevel,
    ApprovalStatus,
    ExportFormat,
)


# ============================================================================
# Engine Init
# ============================================================================

class TestBaseYearAuditEngineInit:
    def test_engine_creation(self, audit_engine):
        assert audit_engine is not None

    def test_engine_is_instance(self, audit_engine):
        assert isinstance(audit_engine, BaseYearAuditEngine)

    def test_initial_entry_count(self, audit_engine):
        count = audit_engine.get_entry_count(organization_id="ORG-EMPTY", base_year=2022)
        assert count >= 0


# ============================================================================
# Create Audit Entry (requires actor param)
# ============================================================================

class TestCreateAuditEntry:
    def test_create_entry(self, audit_engine):
        entry = audit_engine.create_audit_entry(
            event_type=AuditEventType.BASE_YEAR_ESTABLISHED,
            actor="system@greenlang",
            description="Established base year inventory for 2022",
            organization_id="ORG-AUDIT-001",
            base_year=2022,
            severity=AuditSeverity.INFO,
        )
        assert isinstance(entry, AuditEntry)
        assert entry.event_type == AuditEventType.BASE_YEAR_ESTABLISHED

    def test_entry_has_timestamp(self, audit_engine):
        entry = audit_engine.create_audit_entry(
            event_type=AuditEventType.BASE_YEAR_ESTABLISHED,
            actor="test@greenlang",
            description="Test entry",
            organization_id="ORG-AUDIT-002",
            base_year=2022,
        )
        assert entry.timestamp is not None

    def test_entry_has_id(self, audit_engine):
        entry = audit_engine.create_audit_entry(
            event_type=AuditEventType.BASE_YEAR_ESTABLISHED,
            actor="test@greenlang",
            description="Test entry ID",
            organization_id="ORG-AUDIT-003",
            base_year=2022,
        )
        assert entry.entry_id != ""

    def test_create_trigger_detected_entry(self, audit_engine):
        entry = audit_engine.create_audit_entry(
            event_type=AuditEventType.TRIGGER_DETECTED,
            actor="detection@greenlang",
            description="Acquisition trigger detected",
            organization_id="ORG-AUDIT-004",
            base_year=2022,
            severity=AuditSeverity.MEDIUM,
        )
        assert entry.severity == AuditSeverity.MEDIUM

    def test_create_recalculation_approved_entry(self, audit_engine):
        entry = audit_engine.create_audit_entry(
            event_type=AuditEventType.RECALCULATION_APPROVED,
            actor="approver@greenlang",
            description="Base year recalculation approved",
            organization_id="ORG-AUDIT-005",
            base_year=2022,
            severity=AuditSeverity.INFO,
        )
        assert entry.event_type == AuditEventType.RECALCULATION_APPROVED

    def test_entry_has_provenance_hash(self, audit_engine):
        entry = audit_engine.create_audit_entry(
            event_type=AuditEventType.BASE_YEAR_ESTABLISHED,
            actor="test@greenlang",
            description="Hash test",
            organization_id="ORG-AUDIT-006",
            base_year=2022,
        )
        assert entry.provenance_hash != ""
        assert len(entry.provenance_hash) == 64

    def test_entry_with_before_after_values(self, audit_engine):
        entry = audit_engine.create_audit_entry(
            event_type=AuditEventType.RECALCULATION_APPLIED,
            actor="system@greenlang",
            description="Adjustment applied",
            organization_id="ORG-AUDIT-007",
            base_year=2022,
            before_value="100000 tCO2e",
            after_value="108000 tCO2e",
        )
        assert entry.before_value == "100000 tCO2e"
        assert entry.after_value == "108000 tCO2e"


# ============================================================================
# Get Audit Trail (requires organization_id and base_year)
# ============================================================================

class TestGetAuditTrail:
    def test_get_audit_trail(self, audit_engine):
        audit_engine.create_audit_entry(
            event_type=AuditEventType.BASE_YEAR_ESTABLISHED,
            actor="test@greenlang",
            description="Entry 1",
            organization_id="ORG-TRAIL-001",
            base_year=2022,
        )
        audit_engine.create_audit_entry(
            event_type=AuditEventType.TRIGGER_DETECTED,
            actor="test@greenlang",
            description="Entry 2",
            organization_id="ORG-TRAIL-001",
            base_year=2022,
        )
        trail = audit_engine.get_audit_trail(
            organization_id="ORG-TRAIL-001", base_year=2022
        )
        assert isinstance(trail, AuditTrail)
        assert len(trail.entries) >= 2

    def test_get_audit_trail_empty(self, audit_engine):
        trail = audit_engine.get_audit_trail(
            organization_id="ORG-TRAIL-EMPTY", base_year=2019
        )
        assert isinstance(trail, AuditTrail)

    def test_get_latest_entry(self, audit_engine):
        audit_engine.create_audit_entry(
            event_type=AuditEventType.BASE_YEAR_ESTABLISHED,
            actor="test@greenlang",
            description="Latest test",
            organization_id="ORG-TRAIL-002",
            base_year=2022,
        )
        latest = audit_engine.get_latest_entry(
            organization_id="ORG-TRAIL-002", base_year=2022
        )
        assert latest is not None

    def test_get_entry_count(self, audit_engine):
        audit_engine.create_audit_entry(
            event_type=AuditEventType.BASE_YEAR_ESTABLISHED,
            actor="test@greenlang",
            description="Count test",
            organization_id="ORG-TRAIL-003",
            base_year=2022,
        )
        count = audit_engine.get_entry_count(
            organization_id="ORG-TRAIL-003", base_year=2022
        )
        assert count >= 1


# ============================================================================
# Record Approval
# ============================================================================

class TestRecordApproval:
    def test_record_approval(self, audit_engine):
        record = audit_engine.record_approval(
            subject="Base year recalculation package PKG-001",
            requested_by="analyst@greenlang",
            approver="manager@greenlang",
            status=ApprovalStatus.APPROVED,
            organization_id="ORG-APPR-001",
            base_year=2022,
        )
        assert isinstance(record, ApprovalRecord)

    def test_record_rejection(self, audit_engine):
        record = audit_engine.record_approval(
            subject="Recalculation package PKG-002",
            requested_by="analyst@greenlang",
            approver="reviewer@greenlang",
            status=ApprovalStatus.REJECTED,
            organization_id="ORG-APPR-002",
            base_year=2022,
            comments="Insufficient evidence",
        )
        assert record.approval_status == ApprovalStatus.REJECTED


# ============================================================================
# Verification Package
# ============================================================================

class TestVerificationPackage:
    def test_create_verification_package(self, audit_engine):
        package = audit_engine.create_verification_package(
            base_year=2022,
            verifier_name="Independent Verifiers Ltd",
            verification_level=VerificationLevel.LIMITED_ASSURANCE,
            organization_id="ORG-VERIF-001",
        )
        assert isinstance(package, VerificationPackage)

    def test_verification_package_has_hash(self, audit_engine):
        package = audit_engine.create_verification_package(
            base_year=2022,
            verifier_name="Green Assurance Corp",
            verification_level=VerificationLevel.REASONABLE_ASSURANCE,
            organization_id="ORG-VERIF-002",
        )
        assert package.provenance_hash != ""


# ============================================================================
# ISAE 3410 (takes audit_trail)
# ============================================================================

class TestISAE3410:
    def test_generate_isae3410_package(self, audit_engine):
        audit_engine.create_audit_entry(
            event_type=AuditEventType.BASE_YEAR_ESTABLISHED,
            actor="test@greenlang",
            description="ISAE test entry",
            organization_id="ORG-ISAE-001",
            base_year=2022,
        )
        trail = audit_engine.get_audit_trail(
            organization_id="ORG-ISAE-001", base_year=2022
        )
        package = audit_engine.generate_isae3410_package(trail)
        assert package is not None
        assert isinstance(package, dict)


# ============================================================================
# Validate Completeness (takes audit_trail)
# ============================================================================

class TestValidateAuditCompleteness:
    def test_validate_completeness(self, audit_engine):
        audit_engine.create_audit_entry(
            event_type=AuditEventType.BASE_YEAR_ESTABLISHED,
            actor="test@greenlang",
            description="Completeness test",
            organization_id="ORG-COMPL-001",
            base_year=2022,
        )
        trail = audit_engine.get_audit_trail(
            organization_id="ORG-COMPL-001", base_year=2022
        )
        gaps = audit_engine.validate_audit_completeness(trail)
        assert isinstance(gaps, list)


# ============================================================================
# Export Audit Log (takes trail and output_format)
# ============================================================================

class TestExportAuditLog:
    def test_export_markdown(self, audit_engine):
        audit_engine.create_audit_entry(
            event_type=AuditEventType.BASE_YEAR_ESTABLISHED,
            actor="test@greenlang",
            description="Export test",
            organization_id="ORG-EXPORT-001",
            base_year=2022,
        )
        trail = audit_engine.get_audit_trail(
            organization_id="ORG-EXPORT-001", base_year=2022
        )
        export = audit_engine.export_audit_log(trail, output_format=ExportFormat.MARKDOWN)
        assert isinstance(export, str)
        assert len(export) > 0

    def test_export_json(self, audit_engine):
        audit_engine.create_audit_entry(
            event_type=AuditEventType.BASE_YEAR_ESTABLISHED,
            actor="test@greenlang",
            description="JSON export test",
            organization_id="ORG-EXPORT-002",
            base_year=2022,
        )
        trail = audit_engine.get_audit_trail(
            organization_id="ORG-EXPORT-002", base_year=2022
        )
        export = audit_engine.export_audit_log(trail, output_format=ExportFormat.JSON)
        assert isinstance(export, str)


# ============================================================================
# Clear Trail
# ============================================================================

class TestClearTrail:
    def test_clear_trail(self, audit_engine):
        audit_engine.create_audit_entry(
            event_type=AuditEventType.BASE_YEAR_ESTABLISHED,
            actor="test@greenlang",
            description="To be cleared",
            organization_id="ORG-CLEAR-001",
            base_year=2022,
        )
        audit_engine.clear_trail(organization_id="ORG-CLEAR-001", base_year=2022)
        trail = audit_engine.get_audit_trail(
            organization_id="ORG-CLEAR-001", base_year=2022
        )
        assert len(trail.entries) == 0


# ============================================================================
# Enums
# ============================================================================

class TestAuditEnums:
    def test_audit_event_types(self):
        assert AuditEventType.BASE_YEAR_ESTABLISHED is not None
        assert AuditEventType.TRIGGER_DETECTED is not None
        assert AuditEventType.SIGNIFICANCE_ASSESSED is not None
        assert AuditEventType.RECALCULATION_APPROVED is not None
        assert AuditEventType.RECALCULATION_APPLIED is not None
        assert len(AuditEventType) >= 5

    def test_audit_severity(self):
        assert AuditSeverity.INFO is not None
        assert AuditSeverity.LOW is not None
        assert AuditSeverity.MEDIUM is not None
        assert AuditSeverity.HIGH is not None
        assert AuditSeverity.CRITICAL is not None
        assert len(AuditSeverity) == 5

    def test_verification_level(self):
        assert VerificationLevel.INTERNAL_REVIEW is not None
        assert VerificationLevel.LIMITED_ASSURANCE is not None
        assert VerificationLevel.REASONABLE_ASSURANCE is not None
        assert len(VerificationLevel) == 3

    def test_approval_status(self):
        assert ApprovalStatus.PENDING is not None
        assert ApprovalStatus.APPROVED is not None
        assert ApprovalStatus.REJECTED is not None
        assert len(ApprovalStatus) >= 3

    def test_export_format(self):
        assert ExportFormat.JSON is not None
        assert ExportFormat.CSV is not None
        assert ExportFormat.MARKDOWN is not None
        assert len(ExportFormat) == 3
