"""
Verification Workflow -- Internal Review and External Assurance

Manages the verification lifecycle for GHG inventories per the
GHG Protocol Corporate Standard guidance on verification:

  1. Internal review (draft -> in_review -> approved/rejected)
  2. External verification (limited or reasonable assurance)
  3. Finding management (observations, nonconformities)
  4. Verification statement generation

Workflow states:
  DRAFT -> IN_REVIEW -> SUBMITTED -> APPROVED | REJECTED
  APPROVED -> EXTERNALLY_VERIFIED (optional)

Example:
    >>> wf = VerificationWorkflow(config)
    >>> record = wf.start_internal_review(inventory_id, reviewer_id)
    >>> wf.add_finding(record.id, AddFindingRequest(...))
    >>> wf.approve_inventory(record.id, approver_id)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import (
    FindingSeverity,
    FindingType,
    GHGAppConfig,
    VerificationLevel,
    VerificationStatus,
)
from .models import (
    AddFindingRequest,
    GHGInventory,
    VerificationFinding,
    VerificationRecord,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# Valid state transitions
_VALID_TRANSITIONS: Dict[VerificationStatus, List[VerificationStatus]] = {
    VerificationStatus.DRAFT: [VerificationStatus.IN_REVIEW],
    VerificationStatus.IN_REVIEW: [
        VerificationStatus.SUBMITTED,
        VerificationStatus.REJECTED,
    ],
    VerificationStatus.SUBMITTED: [
        VerificationStatus.APPROVED,
        VerificationStatus.REJECTED,
    ],
    VerificationStatus.APPROVED: [VerificationStatus.EXTERNALLY_VERIFIED],
    VerificationStatus.REJECTED: [VerificationStatus.DRAFT],
    VerificationStatus.EXTERNALLY_VERIFIED: [],
}


class VerificationWorkflow:
    """
    Manages internal review and external assurance for GHG inventories.

    Implements a state-machine workflow with finding management,
    approval gates, and verification statement generation.
    """

    def __init__(
        self,
        config: Optional[GHGAppConfig] = None,
        inventory_store: Optional[Dict[str, GHGInventory]] = None,
    ) -> None:
        """
        Initialize VerificationWorkflow.

        Args:
            config: Application configuration.
            inventory_store: Shared reference to inventory storage.
        """
        self.config = config or GHGAppConfig()
        self._inventory_store = inventory_store if inventory_store is not None else {}
        self._records: Dict[str, VerificationRecord] = {}
        logger.info("VerificationWorkflow initialized")

    # ------------------------------------------------------------------
    # Internal Review
    # ------------------------------------------------------------------

    def start_internal_review(
        self,
        inventory_id: str,
        reviewer_id: str,
    ) -> VerificationRecord:
        """
        Start an internal review process for an inventory.

        Transitions: creates new record in DRAFT, then moves to IN_REVIEW.

        Args:
            inventory_id: Inventory ID to verify.
            reviewer_id: ID of the assigned reviewer.

        Returns:
            Created VerificationRecord in IN_REVIEW status.

        Raises:
            ValueError: If inventory not found or already under review.
        """
        self._validate_inventory_exists(inventory_id)
        self._check_no_active_verification(inventory_id)

        record = VerificationRecord(
            inventory_id=inventory_id,
            level=VerificationLevel.INTERNAL_REVIEW,
            verifier_id=reviewer_id,
            verifier_name=f"Internal Reviewer ({reviewer_id})",
            status=VerificationStatus.IN_REVIEW,
        )
        self._records[record.id] = record

        # Link to inventory
        self._update_inventory_verification(inventory_id, record)

        logger.info(
            "Started internal review for inventory %s by reviewer %s (record=%s)",
            inventory_id,
            reviewer_id,
            record.id,
        )
        return record

    def submit_for_approval(
        self,
        record_id: str,
    ) -> VerificationRecord:
        """
        Submit a reviewed inventory for approval.

        All major findings must be resolved before submission.

        Args:
            record_id: Verification record ID.

        Returns:
            Updated VerificationRecord.

        Raises:
            ValueError: If major findings are unresolved.
        """
        record = self._get_record_or_raise(record_id)
        self._validate_transition(record.status, VerificationStatus.SUBMITTED)

        if record.has_major_findings:
            unresolved_major = [
                f for f in record.findings
                if not f.resolved
                and f.materiality in (FindingSeverity.HIGH, FindingSeverity.CRITICAL)
            ]
            raise ValueError(
                f"Cannot submit: {len(unresolved_major)} major/critical "
                f"findings remain unresolved"
            )

        record.status = VerificationStatus.SUBMITTED
        record.submitted_at = _now()

        logger.info("Submitted verification record %s for approval", record_id)
        return record

    def approve_inventory(
        self,
        record_id: str,
        approver_id: str,
    ) -> VerificationRecord:
        """
        Approve a submitted inventory.

        Args:
            record_id: Verification record ID.
            approver_id: ID of the approver.

        Returns:
            Approved VerificationRecord.

        Raises:
            ValueError: If record not in SUBMITTED state.
        """
        record = self._get_record_or_raise(record_id)
        self._validate_transition(record.status, VerificationStatus.APPROVED)

        record.status = VerificationStatus.APPROVED
        record.completed_at = _now()

        # Update inventory status
        inventory = self._inventory_store.get(record.inventory_id)
        if inventory:
            inventory.status = "final"
            inventory.updated_at = _now()

        logger.info(
            "Inventory %s approved by %s (record=%s)",
            record.inventory_id,
            approver_id,
            record_id,
        )
        return record

    def reject_inventory(
        self,
        record_id: str,
        approver_id: str,
        reason: str,
    ) -> VerificationRecord:
        """
        Reject an inventory (send back for revisions).

        Args:
            record_id: Verification record ID.
            approver_id: ID of the rejecting party.
            reason: Reason for rejection.

        Returns:
            Rejected VerificationRecord.
        """
        record = self._get_record_or_raise(record_id)

        if record.status not in (
            VerificationStatus.IN_REVIEW,
            VerificationStatus.SUBMITTED,
        ):
            raise ValueError(
                f"Cannot reject from status '{record.status.value}'"
            )

        record.status = VerificationStatus.REJECTED

        # Add rejection as a finding
        rejection_finding = VerificationFinding(
            finding_type=FindingType.MAJOR_NONCONFORMITY,
            description=f"Inventory rejected by {approver_id}: {reason}",
            materiality=FindingSeverity.HIGH,
        )
        record.findings.append(rejection_finding)

        logger.info(
            "Inventory %s rejected by %s: %s (record=%s)",
            record.inventory_id,
            approver_id,
            reason,
            record_id,
        )
        return record

    def reopen_for_revision(
        self,
        record_id: str,
    ) -> VerificationRecord:
        """
        Reopen a rejected record for revision.

        Args:
            record_id: Verification record ID.

        Returns:
            Record moved back to DRAFT status.
        """
        record = self._get_record_or_raise(record_id)
        self._validate_transition(record.status, VerificationStatus.DRAFT)

        record.status = VerificationStatus.DRAFT

        logger.info("Reopened verification record %s for revision", record_id)
        return record

    # ------------------------------------------------------------------
    # External Verification
    # ------------------------------------------------------------------

    def assign_external_verifier(
        self,
        inventory_id: str,
        verifier_name: str,
        verifier_organization: str,
        level: VerificationLevel = VerificationLevel.LIMITED_ASSURANCE,
    ) -> VerificationRecord:
        """
        Assign an external verifier for third-party assurance.

        Args:
            inventory_id: Inventory ID.
            verifier_name: Name of the external verifier.
            verifier_organization: Verifier's organization.
            level: Assurance level (limited or reasonable).

        Returns:
            Created VerificationRecord.
        """
        self._validate_inventory_exists(inventory_id)

        # Check that internal review is complete (approved)
        existing = self._find_active_record(inventory_id)
        if existing and existing.status != VerificationStatus.APPROVED:
            raise ValueError(
                "Internal review must be approved before external verification"
            )

        record = VerificationRecord(
            inventory_id=inventory_id,
            level=level,
            verifier_name=verifier_name,
            verifier_organization=verifier_organization,
            status=VerificationStatus.IN_REVIEW,
        )
        self._records[record.id] = record
        self._update_inventory_verification(inventory_id, record)

        logger.info(
            "Assigned external verifier '%s' (%s) for inventory %s at %s level",
            verifier_name,
            verifier_organization,
            inventory_id,
            level.value,
        )
        return record

    def complete_external_verification(
        self,
        record_id: str,
        opinion: str,
        statement: str,
    ) -> VerificationRecord:
        """
        Complete external verification with opinion and statement.

        Args:
            record_id: Verification record ID.
            opinion: Verifier opinion (unqualified/qualified/adverse/disclaimer).
            statement: Full verification statement text.

        Returns:
            Completed VerificationRecord.
        """
        record = self._get_record_or_raise(record_id)

        if record.level not in (
            VerificationLevel.LIMITED_ASSURANCE,
            VerificationLevel.REASONABLE_ASSURANCE,
        ):
            raise ValueError("Only external verification records can be completed this way")

        record.status = VerificationStatus.EXTERNALLY_VERIFIED
        record.opinion = opinion
        record.statement = statement
        record.completed_at = _now()

        # Update inventory
        inventory = self._inventory_store.get(record.inventory_id)
        if inventory:
            inventory.status = "verified"
            inventory.updated_at = _now()

        logger.info(
            "External verification complete for inventory %s: opinion=%s",
            record.inventory_id,
            opinion,
        )
        return record

    # ------------------------------------------------------------------
    # Finding Management
    # ------------------------------------------------------------------

    def add_finding(
        self,
        record_id: str,
        request: AddFindingRequest,
    ) -> VerificationFinding:
        """
        Add a finding to a verification record.

        Args:
            record_id: Verification record ID.
            request: Finding details.

        Returns:
            Created VerificationFinding.

        Raises:
            ValueError: If record is already completed.
        """
        record = self._get_record_or_raise(record_id)

        if record.status in (
            VerificationStatus.APPROVED,
            VerificationStatus.EXTERNALLY_VERIFIED,
        ):
            raise ValueError("Cannot add findings to a completed verification")

        finding = VerificationFinding(
            finding_type=request.finding_type,
            description=request.description,
            scope=request.scope,
            materiality=request.materiality,
        )
        record.findings.append(finding)

        logger.info(
            "Added %s finding to record %s: %s (severity=%s)",
            request.finding_type.value,
            record_id,
            request.description[:50],
            request.materiality.value,
        )
        return finding

    def resolve_finding(
        self,
        record_id: str,
        finding_id: str,
        resolution: str,
    ) -> VerificationFinding:
        """
        Resolve a finding.

        Args:
            record_id: Verification record ID.
            finding_id: Finding ID to resolve.
            resolution: Resolution description.

        Returns:
            Resolved VerificationFinding.

        Raises:
            ValueError: If finding not found.
        """
        record = self._get_record_or_raise(record_id)
        finding = self._get_finding_or_raise(record, finding_id)

        finding.resolved = True
        finding.resolution = resolution
        finding.resolved_at = _now()

        logger.info(
            "Resolved finding %s in record %s: %s",
            finding_id,
            record_id,
            resolution[:50],
        )
        return finding

    def get_findings(
        self,
        record_id: str,
        resolved: Optional[bool] = None,
    ) -> List[VerificationFinding]:
        """
        Get findings for a verification record, optionally filtered.

        Args:
            record_id: Verification record ID.
            resolved: Filter by resolution status (None = all).

        Returns:
            List of VerificationFindings.
        """
        record = self._get_record_or_raise(record_id)

        if resolved is None:
            return record.findings

        return [f for f in record.findings if f.resolved == resolved]

    # ------------------------------------------------------------------
    # Verification Statement
    # ------------------------------------------------------------------

    def generate_verification_statement(
        self,
        record_id: str,
    ) -> Dict[str, Any]:
        """
        Generate a verification statement for a completed review.

        Args:
            record_id: Verification record ID.

        Returns:
            Dict with statement content.
        """
        record = self._get_record_or_raise(record_id)

        total_findings = len(record.findings)
        open_findings = record.open_findings_count
        resolved_findings = total_findings - open_findings

        by_type: Dict[str, int] = {}
        for f in record.findings:
            key = f.finding_type.value
            by_type[key] = by_type.get(key, 0) + 1

        by_severity: Dict[str, int] = {}
        for f in record.findings:
            key = f.materiality.value
            by_severity[key] = by_severity.get(key, 0) + 1

        statement: Dict[str, Any] = {
            "record_id": record.id,
            "inventory_id": record.inventory_id,
            "verification_level": record.level.value,
            "status": record.status.value,
            "verifier": record.verifier_name or "Internal",
            "organization": record.verifier_organization or "",
            "started_at": record.started_at.isoformat(),
            "completed_at": record.completed_at.isoformat() if record.completed_at else None,
            "findings_summary": {
                "total": total_findings,
                "open": open_findings,
                "resolved": resolved_findings,
                "by_type": by_type,
                "by_severity": by_severity,
            },
        }

        if record.opinion:
            statement["opinion"] = record.opinion

        if record.statement:
            statement["statement_text"] = record.statement
        else:
            statement["statement_text"] = self._auto_generate_statement(record)

        statement["provenance_hash"] = _sha256(
            f"{record.id}:{record.inventory_id}:{record.status.value}"
        )

        return statement

    # ------------------------------------------------------------------
    # Status Queries
    # ------------------------------------------------------------------

    def get_verification_status(
        self,
        inventory_id: str,
    ) -> Optional[VerificationRecord]:
        """
        Get the current verification record for an inventory.

        Returns the most recent active record.

        Args:
            inventory_id: Inventory ID.

        Returns:
            VerificationRecord or None.
        """
        return self._find_active_record(inventory_id)

    def get_verification_history(
        self,
        inventory_id: str,
    ) -> List[VerificationRecord]:
        """
        Get all verification records for an inventory.

        Args:
            inventory_id: Inventory ID.

        Returns:
            List of VerificationRecords ordered by start date.
        """
        records = [
            r for r in self._records.values()
            if r.inventory_id == inventory_id
        ]
        return sorted(records, key=lambda r: r.started_at)

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_record_or_raise(self, record_id: str) -> VerificationRecord:
        """Retrieve record or raise ValueError."""
        record = self._records.get(record_id)
        if record is None:
            raise ValueError(f"Verification record not found: {record_id}")
        return record

    def _get_finding_or_raise(
        self,
        record: VerificationRecord,
        finding_id: str,
    ) -> VerificationFinding:
        """Retrieve finding within a record or raise ValueError."""
        for finding in record.findings:
            if finding.id == finding_id:
                return finding
        raise ValueError(f"Finding not found: {finding_id}")

    def _validate_inventory_exists(self, inventory_id: str) -> None:
        """Validate that the inventory exists."""
        if inventory_id not in self._inventory_store:
            raise ValueError(f"Inventory not found: {inventory_id}")

    def _check_no_active_verification(self, inventory_id: str) -> None:
        """Check that no active verification exists."""
        active = self._find_active_record(inventory_id)
        if active and active.status not in (
            VerificationStatus.APPROVED,
            VerificationStatus.REJECTED,
            VerificationStatus.EXTERNALLY_VERIFIED,
        ):
            raise ValueError(
                f"Active verification already exists for inventory "
                f"{inventory_id} (status={active.status.value})"
            )

    def _find_active_record(
        self,
        inventory_id: str,
    ) -> Optional[VerificationRecord]:
        """Find the most recent verification record for an inventory."""
        records = [
            r for r in self._records.values()
            if r.inventory_id == inventory_id
        ]
        if not records:
            return None
        return max(records, key=lambda r: r.started_at)

    @staticmethod
    def _validate_transition(
        current: VerificationStatus,
        target: VerificationStatus,
    ) -> None:
        """Validate that a state transition is allowed."""
        valid = _VALID_TRANSITIONS.get(current, [])
        if target not in valid:
            raise ValueError(
                f"Invalid transition: {current.value} -> {target.value}. "
                f"Valid targets: {[v.value for v in valid]}"
            )

    def _update_inventory_verification(
        self,
        inventory_id: str,
        record: VerificationRecord,
    ) -> None:
        """Link verification record to inventory."""
        inventory = self._inventory_store.get(inventory_id)
        if inventory:
            inventory.verification = record
            inventory.updated_at = _now()

    @staticmethod
    def _auto_generate_statement(record: VerificationRecord) -> str:
        """Auto-generate a verification statement based on findings."""
        total = len(record.findings)
        open_count = record.open_findings_count
        level = record.level.value.replace("_", " ").title()

        if total == 0:
            return (
                f"This {level} of the GHG inventory has been completed. "
                f"No findings were identified during the review process. "
                f"The inventory is considered to be a fair representation "
                f"of the organization's GHG emissions."
            )

        if open_count == 0:
            return (
                f"This {level} of the GHG inventory has been completed. "
                f"{total} finding(s) were identified, all of which have been "
                f"satisfactorily resolved. The inventory, as corrected, is "
                f"considered to be a fair representation of the organization's "
                f"GHG emissions."
            )

        return (
            f"This {level} of the GHG inventory has been completed. "
            f"{total} finding(s) were identified, of which {open_count} "
            f"remain(s) unresolved. The verifier recommends resolving "
            f"outstanding findings before publication."
        )
