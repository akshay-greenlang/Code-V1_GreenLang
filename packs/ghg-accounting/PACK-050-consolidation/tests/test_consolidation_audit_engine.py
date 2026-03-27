# -*- coding: utf-8 -*-
"""
PACK-050 GHG Consolidation Pack - Consolidation Audit Engine Tests

Tests audit trail recording, reconciliation, completeness checks,
sign-off tracking, audit findings, assurance package generation,
and provenance tracking.

Target: 50-70 tests.
"""

import pytest
from decimal import Decimal
from datetime import date

from engines.consolidation_audit_engine import (
    ConsolidationAuditEngine,
    AuditEntry,
    ReconciliationResult,
    CompletenessCheck,
    SignOff,
    AuditFinding,
    AssurancePackage,
    AuditStepType,
    FindingSeverity,
    FindingStatus,
    SignOffLevel,
    DEFAULT_MATERIALITY_THRESHOLD_PCT,
    DEFAULT_COMPLETENESS_TARGET_PCT,
    _round2,
)


@pytest.fixture
def engine():
    """Fresh ConsolidationAuditEngine."""
    return ConsolidationAuditEngine()


class TestAuditTrailRecording:
    """Test audit trail step recording."""

    def test_record_data_receipt(self, engine):
        entry = engine.record_step(
            reporting_year=2025,
            step_type="DATA_RECEIPT",
            description="Received emission data from Entity A",
            entity_id="ENT-A",
            performed_by="system",
        )
        assert isinstance(entry, AuditEntry)
        assert entry.step_type == AuditStepType.DATA_RECEIPT.value
        assert entry.entity_id == "ENT-A"

    def test_record_equity_adjustment(self, engine):
        entry = engine.record_step(
            reporting_year=2025,
            step_type="EQUITY_ADJUSTMENT",
            description="Applied 80% equity share to Entity B",
            entity_id="ENT-B",
            before_value="10000",
            after_value="8000",
            impact_tco2e=Decimal("-2000"),
        )
        assert entry.impact_tco2e == Decimal("-2000")

    def test_record_elimination(self, engine):
        entry = engine.record_step(
            reporting_year=2025,
            step_type="INTERCOMPANY_ELIMINATION",
            description="Eliminated intra-group electricity transfer",
            impact_tco2e=Decimal("-500"),
        )
        assert entry.step_type == AuditStepType.INTERCOMPANY_ELIMINATION.value

    def test_record_manual_adjustment(self, engine):
        entry = engine.record_step(
            reporting_year=2025,
            step_type="MANUAL_ADJUSTMENT",
            description="Error correction for natural gas",
            entity_id="ENT-C",
            impact_tco2e=Decimal("-300"),
            evidence_reference="ADJ-001",
        )
        assert entry.evidence_reference == "ADJ-001"

    def test_record_with_metadata(self, engine):
        entry = engine.record_step(
            reporting_year=2025,
            step_type="DATA_VALIDATION",
            description="Validated entity data quality",
            metadata={"quality_score": "0.95", "checks_passed": 12},
        )
        assert entry.metadata["quality_score"] == "0.95"

    def test_entry_provenance_hash(self, engine):
        entry = engine.record_step(
            reporting_year=2025,
            step_type="DATA_RECEIPT",
            description="Test entry",
        )
        assert len(entry.provenance_hash) == 64

    @pytest.mark.parametrize("step_type", [
        "DATA_RECEIPT", "DATA_VALIDATION", "EQUITY_ADJUSTMENT",
        "INTERCOMPANY_ELIMINATION", "MANUAL_ADJUSTMENT",
        "SCOPE_RECLASSIFICATION", "BASE_YEAR_RESTATEMENT",
        "RECONCILIATION", "COMPLETENESS_CHECK", "SIGN_OFF",
        "REPORT_GENERATION", "ASSURANCE_PACKAGE",
    ])
    def test_all_step_types(self, engine, step_type):
        entry = engine.record_step(
            reporting_year=2025,
            step_type=step_type,
            description=f"Test {step_type}",
        )
        assert entry.step_type == step_type


class TestAuditTrailRetrieval:
    """Test audit trail retrieval and filtering."""

    def test_get_trail_by_year(self, engine):
        engine.record_step(2025, "DATA_RECEIPT", "Entry 1")
        engine.record_step(2024, "DATA_RECEIPT", "Entry 2")
        trail = engine.get_audit_trail(reporting_year=2025)
        assert len(trail) == 1

    def test_get_trail_by_entity(self, engine):
        engine.record_step(2025, "DATA_RECEIPT", "A data", entity_id="A")
        engine.record_step(2025, "DATA_RECEIPT", "B data", entity_id="B")
        trail = engine.get_audit_trail(entity_id="A")
        assert len(trail) == 1

    def test_get_trail_by_step_type(self, engine):
        engine.record_step(2025, "DATA_RECEIPT", "Entry 1")
        engine.record_step(2025, "EQUITY_ADJUSTMENT", "Entry 2")
        trail = engine.get_audit_trail(step_type="DATA_RECEIPT")
        assert len(trail) == 1

    def test_get_all_trail(self, engine):
        engine.record_step(2025, "DATA_RECEIPT", "Entry 1")
        engine.record_step(2025, "DATA_RECEIPT", "Entry 2")
        trail = engine.get_audit_trail()
        assert len(trail) == 2


class TestReconciliation:
    """Test bottom-up vs top-down reconciliation."""

    def test_reconciled_within_1pct(self, engine):
        result = engine.reconcile(
            reporting_year=2025,
            bottom_up=Decimal("50000"),
            top_down=Decimal("50000"),
        )
        assert isinstance(result, ReconciliationResult)
        assert result.status == "RECONCILED"
        assert result.variance == Decimal("0.00")
        assert result.is_material is False

    def test_variance_calculation(self, engine):
        result = engine.reconcile(
            reporting_year=2025,
            bottom_up=Decimal("51000"),
            top_down=Decimal("50000"),
        )
        assert result.variance == Decimal("1000.00")
        assert result.variance_pct == Decimal("2.00")

    def test_material_variance(self, engine):
        result = engine.reconcile(
            reporting_year=2025,
            bottom_up=Decimal("55000"),
            top_down=Decimal("50000"),
        )
        # 5000/50000 = 10% > 5% threshold
        assert result.is_material is True
        assert result.variance_pct == Decimal("10.00")

    def test_immaterial_variance(self, engine):
        result = engine.reconcile(
            reporting_year=2025,
            bottom_up=Decimal("50500"),
            top_down=Decimal("50000"),
        )
        # 500/50000 = 1.00% < 5% threshold
        assert result.is_material is False

    def test_custom_materiality_threshold(self, engine):
        result = engine.reconcile(
            reporting_year=2025,
            bottom_up=Decimal("51000"),
            top_down=Decimal("50000"),
            materiality_threshold_pct=Decimal("1"),
        )
        # 2% > 1% threshold
        assert result.is_material is True

    def test_reconciling_items_reduce_unexplained(self, engine):
        result = engine.reconcile(
            reporting_year=2025,
            bottom_up=Decimal("51000"),
            top_down=Decimal("50000"),
            reconciling_items=[
                {"description": "Timing difference", "amount": "800"},
            ],
        )
        assert result.unexplained_variance == Decimal("200.00")

    def test_reconciliation_provenance_hash(self, engine):
        result = engine.reconcile(
            reporting_year=2025,
            bottom_up=Decimal("50000"),
            top_down=Decimal("50000"),
        )
        assert len(result.provenance_hash) == 64

    def test_reconciliation_recorded_in_audit_trail(self, engine):
        engine.reconcile(
            reporting_year=2025,
            bottom_up=Decimal("50000"),
            top_down=Decimal("50000"),
        )
        trail = engine.get_audit_trail(step_type="RECONCILIATION")
        assert len(trail) == 1

    def test_get_reconciliation_by_id(self, engine):
        result = engine.reconcile(
            reporting_year=2025,
            bottom_up=Decimal("50000"),
            top_down=Decimal("49500"),
        )
        retrieved = engine.get_reconciliation(result.reconciliation_id)
        assert retrieved.variance == result.variance

    def test_get_reconciliation_not_found(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.get_reconciliation("NONEXISTENT")

    def test_negative_variance(self, engine):
        result = engine.reconcile(
            reporting_year=2025,
            bottom_up=Decimal("48000"),
            top_down=Decimal("50000"),
        )
        assert result.variance == Decimal("-2000.00")


class TestCompletenessCheck:
    """Test reporting completeness checks."""

    def test_full_completeness(self, engine):
        check = engine.check_completeness(
            reporting_year=2025,
            entities_in_boundary=["A", "B", "C"],
            entities_reported=["A", "B", "C"],
        )
        assert isinstance(check, CompletenessCheck)
        assert check.completeness_pct == Decimal("100.00")
        assert check.meets_target is True
        assert len(check.missing_entities) == 0

    def test_partial_completeness(self, engine):
        check = engine.check_completeness(
            reporting_year=2025,
            entities_in_boundary=["A", "B", "C", "D", "E"],
            entities_reported=["A", "B", "C"],
        )
        assert check.completeness_pct == Decimal("60.00")
        assert check.meets_target is False
        assert len(check.missing_entities) == 2
        assert "D" in check.missing_entities
        assert "E" in check.missing_entities

    def test_custom_target_pct(self, engine):
        check = engine.check_completeness(
            reporting_year=2025,
            entities_in_boundary=["A", "B", "C", "D"],
            entities_reported=["A", "B", "C"],
            target_pct=Decimal("70"),
        )
        # 75% >= 70%
        assert check.meets_target is True

    def test_scope_coverage_tracking(self, engine):
        check = engine.check_completeness(
            reporting_year=2025,
            entities_in_boundary=["A"],
            entities_reported=["A"],
            scope_coverage={"scope1": True, "scope2": True, "scope3": False},
        )
        assert check.scope_coverage["scope3"] is False
        assert "scope3" in check.scopes_missing

    def test_completeness_provenance_hash(self, engine):
        check = engine.check_completeness(
            reporting_year=2025,
            entities_in_boundary=["A"],
            entities_reported=["A"],
        )
        assert len(check.provenance_hash) == 64

    def test_completeness_recorded_in_audit_trail(self, engine):
        engine.check_completeness(
            reporting_year=2025,
            entities_in_boundary=["A"],
            entities_reported=["A"],
        )
        trail = engine.get_audit_trail(step_type="COMPLETENESS_CHECK")
        assert len(trail) == 1

    def test_zero_entities(self, engine):
        check = engine.check_completeness(
            reporting_year=2025,
            entities_in_boundary=[],
            entities_reported=[],
        )
        assert check.completeness_pct == Decimal("0")


class TestSignOff:
    """Test sign-off tracking."""

    def test_entity_signoff(self, engine):
        signoff = engine.record_signoff(
            reporting_year=2025,
            level="ENTITY",
            signer="cfo@sub.com",
            entity_id="ENT-SUB-001",
            role="CFO",
            comments="Data verified and approved",
        )
        assert isinstance(signoff, SignOff)
        assert signoff.level == SignOffLevel.ENTITY.value
        assert signoff.entity_id == "ENT-SUB-001"

    def test_group_signoff(self, engine):
        signoff = engine.record_signoff(
            reporting_year=2025,
            level="GROUP",
            signer="group-cfo@corp.com",
            role="Group CFO",
        )
        assert signoff.level == SignOffLevel.GROUP.value

    def test_external_assurance_signoff(self, engine):
        signoff = engine.record_signoff(
            reporting_year=2025,
            level="EXTERNAL_ASSURANCE",
            signer="auditor@firm.com",
            role="Lead Assurance Partner",
        )
        assert signoff.level == SignOffLevel.EXTERNAL_ASSURANCE.value

    def test_signoff_provenance_hash(self, engine):
        signoff = engine.record_signoff(
            reporting_year=2025,
            level="GROUP",
            signer="signer@corp.com",
        )
        assert len(signoff.provenance_hash) == 64

    def test_signoff_recorded_in_audit_trail(self, engine):
        engine.record_signoff(
            reporting_year=2025,
            level="ENTITY",
            signer="signer@corp.com",
            entity_id="ENT-A",
        )
        trail = engine.get_audit_trail(step_type="SIGN_OFF")
        assert len(trail) == 1

    def test_get_signoffs_by_year(self, engine):
        engine.record_signoff(2025, "ENTITY", "s1", entity_id="A")
        engine.record_signoff(2024, "ENTITY", "s2", entity_id="B")
        signoffs = engine.get_signoffs(reporting_year=2025)
        assert len(signoffs) == 1

    def test_get_signoffs_by_level(self, engine):
        engine.record_signoff(2025, "ENTITY", "s1", entity_id="A")
        engine.record_signoff(2025, "GROUP", "s2")
        entity_signoffs = engine.get_signoffs(level="ENTITY")
        assert len(entity_signoffs) == 1


class TestAuditFindings:
    """Test audit finding management."""

    def test_record_finding(self, engine):
        finding = engine.record_finding(
            reporting_year=2025,
            severity="MAJOR",
            title="Missing data for Entity B",
            description="Entity B has not submitted Scope 3 data",
            entity_id="ENT-B",
            impact_tco2e=Decimal("2000"),
            recommendation="Request data submission",
        )
        assert isinstance(finding, AuditFinding)
        assert finding.severity == FindingSeverity.MAJOR.value
        assert finding.status == FindingStatus.OPEN.value

    def test_finding_severity_levels(self, engine):
        for severity in ["CRITICAL", "MAJOR", "MINOR", "OBSERVATION", "IMPROVEMENT"]:
            finding = engine.record_finding(
                reporting_year=2025,
                severity=severity,
                title=f"Test {severity}",
                description=f"Test description for {severity}",
            )
            assert finding.severity == severity

    def test_invalid_severity_raises(self, engine):
        with pytest.raises(ValueError, match="Invalid severity"):
            engine.record_finding(
                reporting_year=2025,
                severity="INVALID",
                title="Test",
                description="Test",
            )

    def test_update_finding_status(self, engine):
        finding = engine.record_finding(
            reporting_year=2025,
            severity="MINOR",
            title="Test finding",
            description="Description",
        )
        updated = engine.update_finding_status(
            finding.finding_id,
            "RESOLVED",
            resolution_notes="Fixed by data resubmission",
        )
        assert updated.status == FindingStatus.RESOLVED.value
        assert updated.resolution_notes == "Fixed by data resubmission"

    def test_update_finding_not_found(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.update_finding_status("NONEXISTENT", "RESOLVED")

    def test_update_finding_invalid_status(self, engine):
        finding = engine.record_finding(
            reporting_year=2025, severity="MINOR",
            title="T", description="D",
        )
        with pytest.raises(ValueError, match="Invalid status"):
            engine.update_finding_status(finding.finding_id, "INVALID_STATUS")

    def test_get_findings_by_severity(self, engine):
        engine.record_finding(2025, "CRITICAL", "T1", "D1")
        engine.record_finding(2025, "MINOR", "T2", "D2")
        critical = engine.get_findings(severity="CRITICAL")
        assert len(critical) == 1

    def test_get_findings_by_status(self, engine):
        finding = engine.record_finding(2025, "MINOR", "T", "D")
        engine.update_finding_status(finding.finding_id, "RESOLVED")
        open_findings = engine.get_findings(status="OPEN")
        resolved_findings = engine.get_findings(status="RESOLVED")
        assert len(open_findings) == 0
        assert len(resolved_findings) == 1

    def test_finding_provenance_hash(self, engine):
        finding = engine.record_finding(
            reporting_year=2025, severity="MINOR",
            title="T", description="D",
        )
        assert len(finding.provenance_hash) == 64


class TestAssurancePackage:
    """Test assurance package generation."""

    def _build_complete_audit(self, engine):
        """Build a complete audit trail for testing."""
        engine.record_step(2025, "DATA_RECEIPT", "Data received from A", entity_id="A")
        engine.record_step(2025, "DATA_RECEIPT", "Data received from B", entity_id="B")
        engine.record_step(2025, "EQUITY_ADJUSTMENT", "Adjusted B", entity_id="B", impact_tco2e="-1000")
        engine.reconcile(2025, bottom_up=Decimal("50000"), top_down=Decimal("50200"))
        engine.check_completeness(
            2025,
            entities_in_boundary=["A", "B"],
            entities_reported=["A", "B"],
            scope_coverage={"scope1": True, "scope2": True, "scope3": True},
        )
        engine.record_signoff(2025, "ENTITY", "cfo_a@corp.com", entity_id="A")
        engine.record_signoff(2025, "ENTITY", "cfo_b@corp.com", entity_id="B")
        engine.record_signoff(2025, "GROUP", "group_cfo@corp.com")

    def test_generate_assurance_package(self, engine):
        self._build_complete_audit(engine)
        package = engine.generate_assurance_package(
            reporting_year=2025,
            organisation_name="Test Corp",
            consolidated_total_tco2e=Decimal("50000"),
        )
        assert isinstance(package, AssurancePackage)
        assert package.reporting_year == 2025
        assert package.organisation_name == "Test Corp"

    def test_assurance_package_counts(self, engine):
        self._build_complete_audit(engine)
        package = engine.generate_assurance_package(2025)
        assert package.total_audit_entries > 0
        assert package.total_signoffs == 3
        assert package.entity_signoffs == 2
        assert package.group_signoffs == 1

    def test_assurance_readiness_complete(self, engine):
        self._build_complete_audit(engine)
        package = engine.generate_assurance_package(2025)
        assert package.is_assurance_ready is True
        assert all(package.assurance_readiness_checks.values())

    def test_assurance_not_ready_no_group_signoff(self, engine):
        engine.record_step(2025, "DATA_RECEIPT", "Data")
        engine.reconcile(2025, Decimal("50000"), Decimal("50000"))
        engine.check_completeness(2025, ["A"], ["A"])
        # No group signoff
        package = engine.generate_assurance_package(2025)
        assert package.is_assurance_ready is False
        assert package.assurance_readiness_checks["group_signoff_done"] is False

    def test_assurance_not_ready_critical_findings(self, engine):
        self._build_complete_audit(engine)
        engine.record_finding(2025, "CRITICAL", "Critical issue", "Details")
        package = engine.generate_assurance_package(2025)
        assert package.is_assurance_ready is False
        assert package.critical_findings == 1

    def test_assurance_not_ready_open_findings(self, engine):
        self._build_complete_audit(engine)
        engine.record_finding(2025, "MAJOR", "Open issue", "Details")
        package = engine.generate_assurance_package(2025)
        assert package.is_assurance_ready is False
        assert package.open_findings == 1

    def test_assurance_package_provenance_hash(self, engine):
        self._build_complete_audit(engine)
        package = engine.generate_assurance_package(2025)
        assert len(package.provenance_hash) == 64

    def test_get_package_by_id(self, engine):
        engine.record_step(2025, "DATA_RECEIPT", "Data")
        package = engine.generate_assurance_package(2025)
        retrieved = engine.get_package(package.package_id)
        assert retrieved.package_id == package.package_id

    def test_get_package_not_found(self, engine):
        with pytest.raises(KeyError, match="not found"):
            engine.get_package("NONEXISTENT")

    def test_assurance_package_recorded_in_trail(self, engine):
        engine.record_step(2025, "DATA_RECEIPT", "Data")
        engine.generate_assurance_package(2025)
        trail = engine.get_audit_trail(step_type="ASSURANCE_PACKAGE")
        assert len(trail) == 1
