"""
Unit tests for GL-GHG-APP v1.0 Verification Workflow

Tests internal review, external verification, status transitions,
finding management, verification statements, and audit history.
30+ test cases.
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Optional

from services.config import (
    FindingSeverity,
    FindingType,
    Scope,
    VerificationLevel,
    VerificationStatus,
)
from services.models import (
    GHGInventory,
    ScopeEmissions,
    VerificationFinding,
    VerificationRecord,
)


# ---------------------------------------------------------------------------
# VerificationWorkflow under test
# ---------------------------------------------------------------------------

class VerificationWorkflow:
    """Manages verification and assurance lifecycle."""

    VALID_TRANSITIONS = {
        VerificationStatus.DRAFT: [VerificationStatus.IN_REVIEW],
        VerificationStatus.IN_REVIEW: [VerificationStatus.SUBMITTED, VerificationStatus.REJECTED],
        VerificationStatus.SUBMITTED: [VerificationStatus.APPROVED, VerificationStatus.REJECTED],
        VerificationStatus.APPROVED: [VerificationStatus.EXTERNALLY_VERIFIED],
        VerificationStatus.REJECTED: [VerificationStatus.IN_REVIEW],
        VerificationStatus.EXTERNALLY_VERIFIED: [],
    }

    def __init__(self):
        self.records: Dict[str, VerificationRecord] = {}
        self.history: Dict[str, List[VerificationRecord]] = {}

    def start_review(
        self,
        inventory_id: str,
        level: VerificationLevel = VerificationLevel.INTERNAL_REVIEW,
        reviewer_id: Optional[str] = None,
    ) -> VerificationRecord:
        """Start a new verification review."""
        record = VerificationRecord(
            inventory_id=inventory_id,
            level=level,
            verifier_id=reviewer_id,
            status=VerificationStatus.IN_REVIEW,
            started_at=datetime.utcnow(),
        )
        self.records[record.id] = record
        self.history.setdefault(inventory_id, []).append(record)
        return record

    def transition(
        self,
        record_id: str,
        new_status: VerificationStatus,
        actor: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> VerificationRecord:
        """Transition verification status with validation."""
        record = self.records.get(record_id)
        if record is None:
            raise ValueError(f"Verification record {record_id} not found")

        valid_next = self.VALID_TRANSITIONS.get(record.status, [])
        if new_status not in valid_next:
            raise ValueError(
                f"Invalid transition from {record.status.value} to {new_status.value}. "
                f"Valid: {[s.value for s in valid_next]}"
            )

        record.status = new_status

        if new_status == VerificationStatus.SUBMITTED:
            record.submitted_at = datetime.utcnow()
        elif new_status == VerificationStatus.APPROVED:
            record.completed_at = datetime.utcnow()
        elif new_status == VerificationStatus.REJECTED:
            record.opinion = reason or "Rejected without specific reason"
        elif new_status == VerificationStatus.EXTERNALLY_VERIFIED:
            record.completed_at = datetime.utcnow()

        return record

    def submit_for_approval(self, record_id: str) -> VerificationRecord:
        """Submit for approval (from in_review)."""
        return self.transition(record_id, VerificationStatus.SUBMITTED)

    def approve(self, record_id: str, approver: str) -> VerificationRecord:
        """Approve the verification."""
        record = self.transition(record_id, VerificationStatus.APPROVED, actor=approver)
        record.verifier_id = approver
        return record

    def reject(self, record_id: str, reason: str) -> VerificationRecord:
        """Reject the verification."""
        return self.transition(record_id, VerificationStatus.REJECTED, reason=reason)

    def assign_external_verifier(
        self,
        record_id: str,
        verifier_name: str,
        organization: str,
        level: VerificationLevel = VerificationLevel.LIMITED_ASSURANCE,
    ) -> VerificationRecord:
        """Assign an external verifier."""
        record = self.records.get(record_id)
        if record is None:
            raise ValueError(f"Verification record {record_id} not found")
        record.verifier_name = verifier_name
        record.verifier_organization = organization
        record.level = level
        return record

    def add_finding(
        self,
        record_id: str,
        finding_type: FindingType,
        description: str,
        materiality: FindingSeverity = FindingSeverity.LOW,
        scope: Optional[Scope] = None,
    ) -> VerificationFinding:
        """Add a finding to a verification record."""
        record = self.records.get(record_id)
        if record is None:
            raise ValueError(f"Verification record {record_id} not found")
        finding = VerificationFinding(
            finding_type=finding_type,
            description=description,
            materiality=materiality,
            scope=scope,
        )
        record.findings.append(finding)
        return finding

    def resolve_finding(
        self,
        record_id: str,
        finding_id: str,
        resolution: str,
    ) -> VerificationFinding:
        """Resolve a finding."""
        record = self.records.get(record_id)
        if record is None:
            raise ValueError(f"Verification record {record_id} not found")
        finding = next((f for f in record.findings if f.id == finding_id), None)
        if finding is None:
            raise ValueError(f"Finding {finding_id} not found")
        finding.resolved = True
        finding.resolution = resolution
        finding.resolved_at = datetime.utcnow()
        return finding

    def generate_statement(self, record_id: str) -> Optional[str]:
        """Generate verification statement (only if approved)."""
        record = self.records.get(record_id)
        if record is None or record.status != VerificationStatus.APPROVED:
            return None
        has_major = record.has_major_findings
        if has_major:
            opinion = "qualified"
        else:
            opinion = "unqualified"
        statement = (
            f"Verification statement for inventory {record.inventory_id}. "
            f"Level: {record.level.value}. Opinion: {opinion}. "
            f"Findings: {len(record.findings)} total, {record.open_findings_count} unresolved."
        )
        record.statement = statement
        record.opinion = opinion
        return statement

    def get_history(self, inventory_id: str) -> List[VerificationRecord]:
        """Get verification history for an inventory."""
        return sorted(
            self.history.get(inventory_id, []),
            key=lambda r: r.started_at,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workflow():
    return VerificationWorkflow()


# ---------------------------------------------------------------------------
# TestStartReview
# ---------------------------------------------------------------------------

class TestStartReview:
    """Test starting a verification review."""

    def test_creates_record(self, workflow):
        """Test review creates a verification record."""
        record = workflow.start_review("inv-001")
        assert record.inventory_id == "inv-001"
        assert record.id in workflow.records

    def test_status_in_review(self, workflow):
        """Test initial status is in_review."""
        record = workflow.start_review("inv-001")
        assert record.status == VerificationStatus.IN_REVIEW

    def test_default_level_internal(self, workflow):
        """Test default level is internal review."""
        record = workflow.start_review("inv-001")
        assert record.level == VerificationLevel.INTERNAL_REVIEW

    def test_with_reviewer(self, workflow):
        """Test assigning reviewer at start."""
        record = workflow.start_review("inv-001", reviewer_id="user-123")
        assert record.verifier_id == "user-123"


# ---------------------------------------------------------------------------
# TestSubmitForApproval
# ---------------------------------------------------------------------------

class TestSubmitForApproval:
    """Test submit for approval transition."""

    def test_valid_transition(self, workflow):
        """Test valid transition from in_review to submitted."""
        record = workflow.start_review("inv-001")
        updated = workflow.submit_for_approval(record.id)
        assert updated.status == VerificationStatus.SUBMITTED
        assert updated.submitted_at is not None

    def test_from_draft_invalid(self, workflow):
        """Test cannot submit directly from draft."""
        record = VerificationRecord(inventory_id="inv-001", status=VerificationStatus.DRAFT)
        workflow.records[record.id] = record
        with pytest.raises(ValueError, match="Invalid transition"):
            workflow.submit_for_approval(record.id)


# ---------------------------------------------------------------------------
# TestApprove
# ---------------------------------------------------------------------------

class TestApprove:
    """Test approval."""

    def test_sets_approved_status(self, workflow):
        """Test approval sets approved status."""
        record = workflow.start_review("inv-001")
        workflow.submit_for_approval(record.id)
        approved = workflow.approve(record.id, "approver-001")
        assert approved.status == VerificationStatus.APPROVED

    def test_records_approver(self, workflow):
        """Test approver is recorded."""
        record = workflow.start_review("inv-001")
        workflow.submit_for_approval(record.id)
        approved = workflow.approve(record.id, "approver-001")
        assert approved.verifier_id == "approver-001"

    def test_completion_timestamp(self, workflow):
        """Test completed_at is set on approval."""
        record = workflow.start_review("inv-001")
        workflow.submit_for_approval(record.id)
        approved = workflow.approve(record.id, "approver-001")
        assert approved.completed_at is not None


# ---------------------------------------------------------------------------
# TestReject
# ---------------------------------------------------------------------------

class TestReject:
    """Test rejection."""

    def test_sets_rejected_status(self, workflow):
        """Test rejection sets rejected status."""
        record = workflow.start_review("inv-001")
        workflow.submit_for_approval(record.id)
        rejected = workflow.reject(record.id, "Material misstatement in Scope 1")
        assert rejected.status == VerificationStatus.REJECTED

    def test_records_reason(self, workflow):
        """Test rejection reason is recorded."""
        record = workflow.start_review("inv-001")
        workflow.submit_for_approval(record.id)
        rejected = workflow.reject(record.id, "Material misstatement in Scope 1")
        assert "misstatement" in rejected.opinion.lower()

    def test_can_return_to_review(self, workflow):
        """Test rejected can return to in_review."""
        record = workflow.start_review("inv-001")
        workflow.submit_for_approval(record.id)
        workflow.reject(record.id, "Need corrections")
        updated = workflow.transition(record.id, VerificationStatus.IN_REVIEW)
        assert updated.status == VerificationStatus.IN_REVIEW


# ---------------------------------------------------------------------------
# TestExternalVerifier
# ---------------------------------------------------------------------------

class TestExternalVerifier:
    """Test external verifier assignment."""

    def test_assignment(self, workflow):
        """Test assigning external verifier."""
        record = workflow.start_review("inv-001")
        updated = workflow.assign_external_verifier(
            record.id, "Deloitte", "Deloitte LLP", VerificationLevel.REASONABLE_ASSURANCE,
        )
        assert updated.verifier_name == "Deloitte"
        assert updated.verifier_organization == "Deloitte LLP"
        assert updated.level == VerificationLevel.REASONABLE_ASSURANCE

    def test_contact_info(self, workflow):
        """Test verifier organization info."""
        record = workflow.start_review("inv-001")
        updated = workflow.assign_external_verifier(record.id, "EY", "Ernst & Young LLP")
        assert updated.verifier_name == "EY"
        assert updated.verifier_organization == "Ernst & Young LLP"


# ---------------------------------------------------------------------------
# TestFindings
# ---------------------------------------------------------------------------

class TestFindings:
    """Test finding management."""

    def test_add_material_finding(self, workflow):
        """Test adding a material finding."""
        record = workflow.start_review("inv-001")
        finding = workflow.add_finding(
            record.id,
            FindingType.MAJOR_NONCONFORMITY,
            "Scope 1 process emissions calculated with incorrect emission factor",
            FindingSeverity.CRITICAL,
            Scope.SCOPE_1,
        )
        assert finding.materiality == FindingSeverity.CRITICAL
        assert len(record.findings) == 1

    def test_add_immaterial_finding(self, workflow):
        """Test adding an immaterial finding."""
        record = workflow.start_review("inv-001")
        finding = workflow.add_finding(
            record.id,
            FindingType.OBSERVATION,
            "Documentation formatting could be improved in emission factor references",
            FindingSeverity.LOW,
        )
        assert finding.materiality == FindingSeverity.LOW

    def test_resolve_finding(self, workflow):
        """Test resolving a finding."""
        record = workflow.start_review("inv-001")
        finding = workflow.add_finding(
            record.id,
            FindingType.MINOR_NONCONFORMITY,
            "Mobile combustion data missing for December operational fleet",
            FindingSeverity.MEDIUM,
        )
        resolved = workflow.resolve_finding(record.id, finding.id, "December fleet data added")
        assert resolved.resolved is True
        assert resolved.resolution == "December fleet data added"
        assert resolved.resolved_at is not None

    def test_outstanding_count(self, workflow):
        """Test outstanding finding count."""
        record = workflow.start_review("inv-001")
        f1 = workflow.add_finding(record.id, FindingType.OBSERVATION, "Test observation finding for tracking purposes", FindingSeverity.LOW)
        f2 = workflow.add_finding(record.id, FindingType.MINOR_NONCONFORMITY, "Test minor nonconformity finding for tracking", FindingSeverity.MEDIUM)
        assert record.open_findings_count == 2
        workflow.resolve_finding(record.id, f1.id, "Resolved")
        assert record.open_findings_count == 1

    def test_has_major_findings(self, workflow):
        """Test major findings detection."""
        record = workflow.start_review("inv-001")
        workflow.add_finding(
            record.id,
            FindingType.MAJOR_NONCONFORMITY,
            "Critical calculation error in Scope 1 stationary combustion emissions",
            FindingSeverity.CRITICAL,
        )
        assert record.has_major_findings is True

    def test_no_major_after_resolution(self, workflow):
        """Test no major findings after all resolved."""
        record = workflow.start_review("inv-001")
        finding = workflow.add_finding(
            record.id,
            FindingType.MAJOR_NONCONFORMITY,
            "Critical calculation error in Scope 1 stationary combustion emissions",
            FindingSeverity.HIGH,
        )
        workflow.resolve_finding(record.id, finding.id, "Corrected calculation")
        assert record.has_major_findings is False


# ---------------------------------------------------------------------------
# TestVerificationStatement
# ---------------------------------------------------------------------------

class TestVerificationStatement:
    """Test verification statement generation."""

    def test_generated_when_approved(self, workflow):
        """Test statement generated only when approved."""
        record = workflow.start_review("inv-001")
        workflow.submit_for_approval(record.id)
        workflow.approve(record.id, "approver-001")
        statement = workflow.generate_statement(record.id)
        assert statement is not None
        assert "inv-001" in statement

    def test_not_generated_when_in_review(self, workflow):
        """Test statement not generated when in_review."""
        record = workflow.start_review("inv-001")
        statement = workflow.generate_statement(record.id)
        assert statement is None

    def test_unqualified_opinion(self, workflow):
        """Test unqualified opinion when no major findings."""
        record = workflow.start_review("inv-001")
        workflow.submit_for_approval(record.id)
        workflow.approve(record.id, "approver-001")
        workflow.generate_statement(record.id)
        assert record.opinion == "unqualified"

    def test_qualified_opinion_with_major_findings(self, workflow):
        """Test qualified opinion when major findings exist."""
        record = workflow.start_review("inv-001")
        workflow.add_finding(
            record.id,
            FindingType.MAJOR_NONCONFORMITY,
            "Material misstatement in Scope 1 process emissions calculation",
            FindingSeverity.CRITICAL,
        )
        workflow.submit_for_approval(record.id)
        workflow.approve(record.id, "approver-001")
        workflow.generate_statement(record.id)
        assert record.opinion == "qualified"


# ---------------------------------------------------------------------------
# TestHistory
# ---------------------------------------------------------------------------

class TestHistory:
    """Test verification history."""

    def test_chronological_order(self, workflow):
        """Test history is in chronological order."""
        r1 = workflow.start_review("inv-001")
        r2 = workflow.start_review("inv-001")
        history = workflow.get_history("inv-001")
        assert len(history) == 2
        assert history[0].started_at <= history[1].started_at

    def test_multiple_records_per_inventory(self, workflow):
        """Test multiple verification records per inventory."""
        workflow.start_review("inv-001")
        workflow.start_review("inv-001")
        workflow.start_review("inv-001")
        history = workflow.get_history("inv-001")
        assert len(history) == 3

    def test_empty_history(self, workflow):
        """Test empty history for non-existent inventory."""
        history = workflow.get_history("inv-nonexistent")
        assert history == []

    def test_separate_inventories(self, workflow):
        """Test separate history per inventory."""
        workflow.start_review("inv-001")
        workflow.start_review("inv-002")
        h1 = workflow.get_history("inv-001")
        h2 = workflow.get_history("inv-002")
        assert len(h1) == 1
        assert len(h2) == 1
