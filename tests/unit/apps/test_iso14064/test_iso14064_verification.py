# -*- coding: utf-8 -*-
"""
Unit tests for VerificationWorkflow -- ISO 14064-1:2018 Clause 9 / ISO 14064-3.

Tests the verification state machine, internal review, external verification,
finding lifecycle, materiality assessment, verification statement generation,
and status queries with 30+ tests.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    FindingSeverity,
    FindingStatus,
    ISOCategory,
    VerificationLevel,
    VerificationStage,
)
from services.models import ISOInventory, _new_id
from services.verification_workflow import (
    VerificationWorkflow,
    _VALID_TRANSITIONS,
)


# ---------------------------------------------------------------------------
# Fixtures specific to verification tests
# ---------------------------------------------------------------------------

@pytest.fixture
def inv_store():
    """Inventory store with one inventory for workflow testing."""
    inv = ISOInventory(org_id="org-1", year=2025)
    return {inv.id: inv}


@pytest.fixture
def inv_id(inv_store):
    """Convenience: single inventory ID."""
    return list(inv_store.keys())[0]


@pytest.fixture
def workflow(default_config, inv_store):
    """VerificationWorkflow wired to a populated inventory store."""
    return VerificationWorkflow(config=default_config, inventory_store=inv_store)


@pytest.fixture
def reviewed_record(workflow, inv_id):
    """A record that has been internally reviewed (INTERNAL_REVIEW stage)."""
    return workflow.start_internal_review(inv_id, "reviewer-1")


@pytest.fixture
def approved_record(workflow, inv_id, reviewed_record):
    """A record that has passed internal review (APPROVED stage)."""
    return workflow.approve_inventory(reviewed_record.id, "approver-1")


# ===========================================================================
# Tests
# ===========================================================================


class TestStateMachineConstants:
    """Test the valid transitions map."""

    def test_draft_can_go_to_internal_review(self):
        assert VerificationStage.INTERNAL_REVIEW in _VALID_TRANSITIONS[VerificationStage.DRAFT]

    def test_internal_review_can_go_to_approved_or_draft(self):
        targets = _VALID_TRANSITIONS[VerificationStage.INTERNAL_REVIEW]
        assert VerificationStage.APPROVED in targets
        assert VerificationStage.DRAFT in targets

    def test_approved_can_go_to_external_verification(self):
        assert VerificationStage.EXTERNAL_VERIFICATION in _VALID_TRANSITIONS[VerificationStage.APPROVED]

    def test_verified_is_terminal(self):
        assert _VALID_TRANSITIONS[VerificationStage.VERIFIED] == []


class TestInternalReview:
    """Test starting and managing internal review."""

    def test_start_internal_review(self, workflow, inv_id):
        record = workflow.start_internal_review(inv_id, "reviewer-1")
        assert record.stage == VerificationStage.INTERNAL_REVIEW
        assert record.inventory_id == inv_id
        assert len(record.id) == 36

    def test_start_review_nonexistent_inventory_raises(self, workflow):
        with pytest.raises(ValueError, match="not found"):
            workflow.start_internal_review("bad-inv", "reviewer-1")

    def test_duplicate_active_review_raises(self, workflow, inv_id):
        workflow.start_internal_review(inv_id, "reviewer-1")
        with pytest.raises(ValueError, match="Active verification"):
            workflow.start_internal_review(inv_id, "reviewer-2")

    def test_approve_inventory(self, workflow, inv_id, reviewed_record):
        approved = workflow.approve_inventory(reviewed_record.id, "approver-1")
        assert approved.stage == VerificationStage.APPROVED

    def test_approve_with_open_critical_finding_raises(self, workflow, inv_id, reviewed_record):
        workflow.add_finding(
            reviewed_record.id, "Material misstatement in Scope 1",
            severity=FindingSeverity.CRITICAL,
        )
        with pytest.raises(ValueError, match="critical/high"):
            workflow.approve_inventory(reviewed_record.id, "approver-1")

    def test_approve_with_open_high_finding_raises(self, workflow, inv_id, reviewed_record):
        workflow.add_finding(
            reviewed_record.id, "Incorrect emission factors used",
            severity=FindingSeverity.HIGH,
        )
        with pytest.raises(ValueError, match="critical/high"):
            workflow.approve_inventory(reviewed_record.id, "approver-1")

    def test_approve_after_resolving_findings(self, workflow, inv_id, reviewed_record):
        finding = workflow.add_finding(
            reviewed_record.id, "Incorrect emission factors used",
            severity=FindingSeverity.HIGH,
        )
        workflow.resolve_finding(reviewed_record.id, finding.id, "Corrected EF values")
        approved = workflow.approve_inventory(reviewed_record.id, "approver-1")
        assert approved.stage == VerificationStage.APPROVED


class TestRejectToDraft:
    """Test rejection back to draft."""

    def test_reject_to_draft(self, workflow, inv_id, reviewed_record):
        rejected = workflow.reject_to_draft(reviewed_record.id, "Data quality issues")
        assert rejected.stage == VerificationStage.DRAFT
        # A finding should have been added for the rejection
        assert len(rejected.findings) >= 1


class TestExternalVerification:
    """Test external verifier assignment and completion."""

    def test_assign_external_verifier(self, workflow, inv_id, approved_record):
        record = workflow.assign_external_verifier(
            inventory_id=inv_id,
            verifier_name="John Smith",
            verifier_organization="Big4 Audit",
            assurance_level=VerificationLevel.LIMITED,
            verifier_accreditation="ISO 14065",
        )
        assert record.stage == VerificationStage.EXTERNAL_VERIFICATION
        assert record.verifier_name == "John Smith"
        assert record.verifier_organization == "Big4 Audit"
        assert record.level == VerificationLevel.LIMITED

    def test_assign_verifier_without_approval_raises(self, workflow, inv_id, reviewed_record):
        with pytest.raises(ValueError, match="approved"):
            workflow.assign_external_verifier(
                inv_id, "John Smith", "Big4",
            )

    def test_complete_verification(self, workflow, inv_id, approved_record):
        ext_record = workflow.assign_external_verifier(
            inv_id, "John Smith", "Big4",
        )
        verified = workflow.complete_verification(
            ext_record.id,
            opinion="unqualified",
            statement="The GHG inventory is fairly stated.",
        )
        assert verified.stage == VerificationStage.VERIFIED
        assert verified.opinion == "unqualified"
        assert verified.statement == "The GHG inventory is fairly stated."
        assert verified.completed_at is not None

    def test_verified_is_terminal(self, workflow, inv_id, approved_record):
        ext_record = workflow.assign_external_verifier(inv_id, "JS", "Big4")
        verified = workflow.complete_verification(
            ext_record.id, "unqualified", "Fair statement",
        )
        with pytest.raises(ValueError, match="Invalid transition"):
            workflow.approve_inventory(verified.id, "someone")


class TestFindingManagement:
    """Test finding lifecycle."""

    def test_add_finding(self, workflow, inv_id, reviewed_record):
        finding = workflow.add_finding(
            reviewed_record.id,
            "Missing emission factor documentation",
            severity=FindingSeverity.MEDIUM,
            iso_category=ISOCategory.CATEGORY_1_DIRECT,
            clause_reference="6.2.1",
        )
        assert finding.severity == FindingSeverity.MEDIUM
        assert finding.status == FindingStatus.OPEN
        assert finding.iso_category == ISOCategory.CATEGORY_1_DIRECT

    def test_add_finding_to_verified_raises(self, workflow, inv_id, approved_record):
        ext_record = workflow.assign_external_verifier(inv_id, "JS", "Big4")
        workflow.complete_verification(ext_record.id, "unqualified", "OK")
        with pytest.raises(ValueError, match="completed verification"):
            workflow.add_finding(ext_record.id, "Late finding")

    def test_resolve_finding(self, workflow, inv_id, reviewed_record):
        finding = workflow.add_finding(
            reviewed_record.id, "Issue found", FindingSeverity.LOW,
        )
        resolved = workflow.resolve_finding(
            reviewed_record.id, finding.id, "Issue corrected",
        )
        assert resolved.status == FindingStatus.RESOLVED
        assert resolved.resolution == "Issue corrected"

    def test_resolve_nonexistent_finding_raises(self, workflow, inv_id, reviewed_record):
        with pytest.raises(ValueError, match="not found"):
            workflow.resolve_finding(
                reviewed_record.id, "bad-finding-id", "Resolution",
            )

    def test_get_findings_all(self, workflow, inv_id, reviewed_record):
        workflow.add_finding(reviewed_record.id, "F1", FindingSeverity.LOW)
        workflow.add_finding(reviewed_record.id, "F2", FindingSeverity.MEDIUM)
        findings = workflow.get_findings(reviewed_record.id)
        assert len(findings) == 2

    def test_get_findings_filter_by_severity(self, workflow, inv_id, reviewed_record):
        workflow.add_finding(reviewed_record.id, "F1", FindingSeverity.LOW)
        workflow.add_finding(reviewed_record.id, "F2", FindingSeverity.CRITICAL)
        critical = workflow.get_findings(
            reviewed_record.id, severity=FindingSeverity.CRITICAL,
        )
        assert len(critical) == 1
        assert critical[0].severity == FindingSeverity.CRITICAL

    def test_get_findings_filter_by_status(self, workflow, inv_id, reviewed_record):
        f1 = workflow.add_finding(reviewed_record.id, "F1", FindingSeverity.LOW)
        workflow.add_finding(reviewed_record.id, "F2", FindingSeverity.MEDIUM)
        workflow.resolve_finding(reviewed_record.id, f1.id, "Fixed")
        open_findings = workflow.get_findings(
            reviewed_record.id, status=FindingStatus.OPEN,
        )
        assert len(open_findings) == 1


class TestMaterialityAssessment:
    """Test materiality assessment logic."""

    def test_material_error(self, workflow, inv_id, reviewed_record):
        result = workflow.assess_materiality(
            reviewed_record.id,
            total_emissions=10000.0,
            error_amount=600.0,
        )
        # 600/10000 = 6% >= 5% default threshold
        assert result["is_material"] is True
        assert result["error_pct"] == 6.0
        assert result["severity"] == "critical"

    def test_immaterial_error(self, workflow, inv_id, reviewed_record):
        result = workflow.assess_materiality(
            reviewed_record.id,
            total_emissions=10000.0,
            error_amount=100.0,
        )
        # 100/10000 = 1% < 5%
        assert result["is_material"] is False
        assert result["severity"] == "low"

    def test_zero_emissions_not_material(self, workflow, inv_id, reviewed_record):
        result = workflow.assess_materiality(
            reviewed_record.id,
            total_emissions=0.0,
            error_amount=500.0,
        )
        assert result["is_material"] is False
        assert "zero or negative" in result["note"]


class TestVerificationStatement:
    """Test verification statement generation."""

    def test_statement_internal_review(self, workflow, inv_id, reviewed_record):
        stmt = workflow.generate_verification_statement(reviewed_record.id)
        assert stmt["stage"] == "internal_review"
        assert "in progress" in stmt["statement_text"].lower()

    def test_statement_no_findings(self, workflow, inv_id, approved_record):
        ext_record = workflow.assign_external_verifier(inv_id, "JS", "Big4")
        workflow.complete_verification(ext_record.id, "unqualified", "")
        stmt = workflow.generate_verification_statement(ext_record.id)
        assert stmt["stage"] == "verified"
        assert stmt["findings_summary"]["total"] == 0

    def test_statement_with_resolved_findings(self, workflow, inv_id, reviewed_record):
        f = workflow.add_finding(reviewed_record.id, "Issue X", FindingSeverity.LOW)
        workflow.resolve_finding(reviewed_record.id, f.id, "Fixed")
        stmt = workflow.generate_verification_statement(reviewed_record.id)
        assert stmt["findings_summary"]["resolved"] == 1

    def test_statement_has_provenance_hash(self, workflow, inv_id, reviewed_record):
        stmt = workflow.generate_verification_statement(reviewed_record.id)
        assert len(stmt["provenance_hash"]) == 64


class TestStatusQueries:
    """Test verification status queries."""

    def test_get_verification_status(self, workflow, inv_id, reviewed_record):
        status = workflow.get_verification_status(inv_id)
        assert status is not None
        assert status.stage == VerificationStage.INTERNAL_REVIEW

    def test_get_status_no_review(self, workflow, inv_store):
        # Create a second inventory with no review
        inv2 = ISOInventory(org_id="org-2", year=2025)
        inv_store[inv2.id] = inv2
        assert workflow.get_verification_status(inv2.id) is None

    def test_get_verification_history(self, workflow, inv_id):
        workflow.start_internal_review(inv_id, "reviewer-1")
        history = workflow.get_verification_history(inv_id)
        assert len(history) == 1

    def test_get_record_by_id(self, workflow, inv_id, reviewed_record):
        record = workflow.get_record(reviewed_record.id)
        assert record is not None
        assert record.id == reviewed_record.id

    def test_get_nonexistent_record(self, workflow):
        assert workflow.get_record("bad-id") is None
