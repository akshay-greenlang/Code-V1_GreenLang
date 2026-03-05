"""
Verification Workflow -- ISO 14064-1:2018 Clause 9 / ISO 14064-3 Implementation

Manages the verification lifecycle for ISO 14064-1 GHG inventories:

  Workflow Stages:
    DRAFT -> INTERNAL_REVIEW -> APPROVED -> EXTERNAL_VERIFICATION -> VERIFIED

  Finding Severities (ISO 14064-3):
    - low: Minor observations for future improvement
    - medium: Issues requiring attention but not blocking
    - high: Significant issues requiring resolution before approval
    - critical: Material misstatements exceeding materiality threshold

  Assurance Levels (ISO 14064-3):
    - Limited: Negative form of conclusion ("nothing has come to our attention...")
    - Reasonable: Positive form of conclusion ("in our opinion...")

Key features:
  - Stage management with state machine transitions
  - Finding tracking with severity and resolution lifecycle
  - Materiality assessment (default 5% threshold)
  - Verification statement generation
  - Limited vs reasonable assurance support
  - Complete audit trail via provenance hashes

Example:
    >>> wf = VerificationWorkflow(config)
    >>> record = wf.start_internal_review("inv-1", "reviewer-1")
    >>> finding = wf.add_finding(record.id, "Scope 2 calculation error", ...)
    >>> wf.resolve_finding(record.id, finding.id, "Corrected calculation")
    >>> wf.approve_inventory(record.id, "approver-1")
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .config import (
    FindingSeverity,
    FindingStatus,
    ISO14064AppConfig,
    ISOCategory,
    VerificationLevel,
    VerificationStage,
)
from .models import (
    Finding,
    FindingsSummary,
    ISOInventory,
    VerificationRecord,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Valid State Transitions
# ---------------------------------------------------------------------------

_VALID_TRANSITIONS: Dict[VerificationStage, List[VerificationStage]] = {
    VerificationStage.DRAFT: [VerificationStage.INTERNAL_REVIEW],
    VerificationStage.INTERNAL_REVIEW: [
        VerificationStage.APPROVED,
        VerificationStage.DRAFT,  # Reject back to draft
    ],
    VerificationStage.APPROVED: [VerificationStage.EXTERNAL_VERIFICATION],
    VerificationStage.EXTERNAL_VERIFICATION: [
        VerificationStage.VERIFIED,
        VerificationStage.APPROVED,  # Findings requiring rework
    ],
    VerificationStage.VERIFIED: [],  # Terminal state
}


class VerificationWorkflow:
    """
    Manages internal review and external verification for ISO 14064-1 inventories.

    Implements a state-machine workflow with finding management,
    materiality assessment, and verification statement generation
    per ISO 14064-3.

    Attributes:
        config: Application configuration.
        _inventory_store: Shared reference to inventory storage.
        _records: In-memory verification record store.
    """

    def __init__(
        self,
        config: Optional[ISO14064AppConfig] = None,
        inventory_store: Optional[Dict[str, ISOInventory]] = None,
    ) -> None:
        """
        Initialize VerificationWorkflow.

        Args:
            config: Application configuration.
            inventory_store: Shared reference to inventory storage.
        """
        self.config = config or ISO14064AppConfig()
        self._inventory_store = inventory_store or {}
        self._records: Dict[str, VerificationRecord] = {}
        self._materiality_threshold = self.config.recalculation_threshold_percent
        logger.info(
            "VerificationWorkflow initialized (materiality=%.1f%%)",
            self._materiality_threshold,
        )

    # ------------------------------------------------------------------
    # Internal Review
    # ------------------------------------------------------------------

    def start_internal_review(
        self,
        inventory_id: str,
        reviewer_id: str,
    ) -> VerificationRecord:
        """
        Start an internal review process.

        Creates a new record in INTERNAL_REVIEW stage.

        Args:
            inventory_id: Inventory ID to verify.
            reviewer_id: Assigned reviewer ID.

        Returns:
            VerificationRecord in INTERNAL_REVIEW stage.

        Raises:
            ValueError: If inventory not found or already under review.
        """
        self._validate_inventory_exists(inventory_id)
        self._check_no_active_review(inventory_id)

        record = VerificationRecord(
            inventory_id=inventory_id,
            stage=VerificationStage.INTERNAL_REVIEW,
            level=VerificationLevel.NOT_VERIFIED,
            verifier_name=f"Internal Reviewer ({reviewer_id})",
        )
        self._records[record.id] = record

        logger.info(
            "Started internal review for inventory %s by reviewer %s (record=%s)",
            inventory_id,
            reviewer_id,
            record.id,
        )
        return record

    def approve_inventory(
        self,
        record_id: str,
        approver_id: str,
    ) -> VerificationRecord:
        """
        Approve an internally reviewed inventory.

        All critical/high findings must be resolved before approval.

        Args:
            record_id: Verification record ID.
            approver_id: Approver ID.

        Returns:
            Approved VerificationRecord.

        Raises:
            ValueError: If open critical findings exist.
        """
        record = self._get_record_or_raise(record_id)
        self._validate_transition(record.stage, VerificationStage.APPROVED)

        # Check for blocking findings
        blocking = [
            f for f in record.findings
            if f.severity in (FindingSeverity.CRITICAL, FindingSeverity.HIGH)
            and f.status in (FindingStatus.OPEN, FindingStatus.IN_PROGRESS)
        ]
        if blocking:
            raise ValueError(
                f"Cannot approve: {len(blocking)} critical/high finding(s) "
                f"remain unresolved."
            )

        record.stage = VerificationStage.APPROVED

        logger.info(
            "Inventory %s approved by %s (record=%s)",
            record.inventory_id,
            approver_id,
            record_id,
        )
        return record

    def reject_to_draft(
        self,
        record_id: str,
        reason: str,
    ) -> VerificationRecord:
        """
        Reject back to DRAFT for rework.

        Args:
            record_id: Verification record ID.
            reason: Rejection reason.

        Returns:
            Record moved back to DRAFT.
        """
        record = self._get_record_or_raise(record_id)
        self._validate_transition(record.stage, VerificationStage.DRAFT)

        record.stage = VerificationStage.DRAFT

        # Log rejection as a finding
        finding = Finding(
            description=f"Rejected back to draft: {reason}",
            severity=FindingSeverity.MEDIUM,
            status=FindingStatus.OPEN,
        )
        record.findings.append(finding)

        logger.info(
            "Rejected record %s back to draft: %s",
            record_id,
            reason[:80],
        )
        return record

    # ------------------------------------------------------------------
    # External Verification
    # ------------------------------------------------------------------

    def assign_external_verifier(
        self,
        inventory_id: str,
        verifier_name: str,
        verifier_organization: str,
        assurance_level: VerificationLevel = VerificationLevel.LIMITED,
        verifier_accreditation: Optional[str] = None,
        scope_of_verification: Optional[str] = None,
    ) -> VerificationRecord:
        """
        Assign an external verifier for third-party assurance.

        Requires that internal review is approved first.

        Args:
            inventory_id: Inventory ID.
            verifier_name: External verifier name.
            verifier_organization: Verifier organization.
            assurance_level: Limited or reasonable assurance.
            verifier_accreditation: Accreditation reference.
            scope_of_verification: Scope description.

        Returns:
            VerificationRecord in EXTERNAL_VERIFICATION stage.
        """
        self._validate_inventory_exists(inventory_id)

        # Find approved internal review
        existing = self._find_active_record(inventory_id)
        if existing is None or existing.stage != VerificationStage.APPROVED:
            raise ValueError(
                "Internal review must be approved before external verification."
            )

        # Transition the existing record
        self._validate_transition(existing.stage, VerificationStage.EXTERNAL_VERIFICATION)

        existing.stage = VerificationStage.EXTERNAL_VERIFICATION
        existing.level = assurance_level
        existing.verifier_name = verifier_name
        existing.verifier_organization = verifier_organization
        existing.verifier_accreditation = verifier_accreditation
        existing.scope_of_verification = scope_of_verification

        logger.info(
            "Assigned external verifier '%s' (%s) for inventory %s at %s assurance",
            verifier_name,
            verifier_organization,
            inventory_id,
            assurance_level.value,
        )
        return existing

    def complete_verification(
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
            Completed VerificationRecord in VERIFIED stage.
        """
        record = self._get_record_or_raise(record_id)
        self._validate_transition(record.stage, VerificationStage.VERIFIED)

        record.stage = VerificationStage.VERIFIED
        record.opinion = opinion
        record.statement = statement
        record.completed_at = _now()

        # Update findings summary
        record.findings_summary = self._build_findings_summary(record)

        logger.info(
            "External verification complete for inventory %s: opinion=%s (record=%s)",
            record.inventory_id,
            opinion,
            record_id,
        )
        return record

    # ------------------------------------------------------------------
    # Finding Management
    # ------------------------------------------------------------------

    def add_finding(
        self,
        record_id: str,
        description: str,
        severity: FindingSeverity = FindingSeverity.LOW,
        iso_category: Optional[ISOCategory] = None,
        clause_reference: Optional[str] = None,
    ) -> Finding:
        """
        Add a finding to a verification record.

        Args:
            record_id: Verification record ID.
            description: Finding description (min 10 chars).
            severity: Severity level.
            iso_category: Affected ISO category.
            clause_reference: ISO 14064-1 clause reference.

        Returns:
            Created Finding.

        Raises:
            ValueError: If record is in terminal state.
        """
        record = self._get_record_or_raise(record_id)

        if record.stage == VerificationStage.VERIFIED:
            raise ValueError("Cannot add findings to a completed verification.")

        finding = Finding(
            description=description,
            severity=severity,
            iso_category=iso_category,
            clause_reference=clause_reference,
            status=FindingStatus.OPEN,
        )
        record.findings.append(finding)

        logger.info(
            "Added finding to record %s: %s (severity=%s)",
            record_id,
            description[:60],
            severity.value,
        )
        return finding

    def resolve_finding(
        self,
        record_id: str,
        finding_id: str,
        resolution: str,
        corrective_action_id: Optional[str] = None,
    ) -> Finding:
        """
        Resolve a verification finding.

        Args:
            record_id: Verification record ID.
            finding_id: Finding ID.
            resolution: Resolution description.
            corrective_action_id: Linked corrective action ID.

        Returns:
            Resolved Finding.
        """
        record = self._get_record_or_raise(record_id)
        finding = self._find_finding(record, finding_id)

        finding.status = FindingStatus.RESOLVED
        finding.resolution = resolution
        finding.corrective_action_id = corrective_action_id
        finding.resolved_at = _now()

        logger.info(
            "Resolved finding %s in record %s: %s",
            finding_id,
            record_id,
            resolution[:60],
        )
        return finding

    def get_findings(
        self,
        record_id: str,
        severity: Optional[FindingSeverity] = None,
        status: Optional[FindingStatus] = None,
    ) -> List[Finding]:
        """
        Get findings, optionally filtered.

        Args:
            record_id: Verification record ID.
            severity: Filter by severity.
            status: Filter by status.

        Returns:
            Filtered list of Finding objects.
        """
        record = self._get_record_or_raise(record_id)
        findings = record.findings

        if severity:
            findings = [f for f in findings if f.severity == severity]

        if status:
            findings = [f for f in findings if f.status == status]

        return findings

    # ------------------------------------------------------------------
    # Materiality Assessment
    # ------------------------------------------------------------------

    def assess_materiality(
        self,
        record_id: str,
        total_emissions: float,
        error_amount: float,
    ) -> Dict[str, Any]:
        """
        Assess whether an error or omission is material.

        Per ISO 14064-3, materiality is assessed relative to the total
        emissions and the stated materiality threshold.

        Args:
            record_id: Verification record ID.
            total_emissions: Total inventory emissions (tCO2e).
            error_amount: Amount of error or omission (tCO2e).

        Returns:
            Dict with materiality assessment.
        """
        record = self._get_record_or_raise(record_id)
        threshold = float(record.materiality_threshold_pct)

        if total_emissions <= 0:
            return {
                "is_material": False,
                "error_pct": 0.0,
                "threshold_pct": threshold,
                "note": "Total emissions is zero or negative",
            }

        error_pct = abs(error_amount) / total_emissions * 100
        is_material = error_pct >= threshold

        return {
            "is_material": is_material,
            "error_amount_tco2e": error_amount,
            "error_pct": round(error_pct, 2),
            "threshold_pct": threshold,
            "severity": "critical" if is_material else "low",
        }

    # ------------------------------------------------------------------
    # Verification Statement
    # ------------------------------------------------------------------

    def generate_verification_statement(
        self,
        record_id: str,
    ) -> Dict[str, Any]:
        """
        Generate a verification statement.

        Args:
            record_id: Verification record ID.

        Returns:
            Dict with statement content.
        """
        record = self._get_record_or_raise(record_id)

        summary = self._build_findings_summary(record)

        statement: Dict[str, Any] = {
            "record_id": record.id,
            "inventory_id": record.inventory_id,
            "stage": record.stage.value,
            "assurance_level": record.level.value,
            "verifier": record.verifier_name or "Internal",
            "organization": record.verifier_organization or "",
            "accreditation": record.verifier_accreditation or "",
            "scope": record.scope_of_verification or "All categories",
            "materiality_threshold_pct": str(record.materiality_threshold_pct),
            "started_at": record.started_at.isoformat(),
            "completed_at": (
                record.completed_at.isoformat() if record.completed_at else None
            ),
            "findings_summary": {
                "total": summary.total_findings,
                "open": summary.open_count,
                "in_progress": summary.in_progress_count,
                "resolved": summary.resolved_count,
                "by_severity": summary.by_severity,
                "by_category": summary.by_category,
                "critical_open": summary.critical_open,
            },
        }

        if record.opinion:
            statement["opinion"] = record.opinion

        if record.statement:
            statement["statement_text"] = record.statement
        else:
            statement["statement_text"] = self._auto_generate_statement(record, summary)

        statement["provenance_hash"] = _sha256(
            f"{record.id}:{record.inventory_id}:{record.stage.value}"
        )

        return statement

    # ------------------------------------------------------------------
    # Status Queries
    # ------------------------------------------------------------------

    def get_verification_status(
        self,
        inventory_id: str,
    ) -> Optional[VerificationRecord]:
        """Get the current verification record for an inventory."""
        return self._find_active_record(inventory_id)

    def get_verification_history(
        self,
        inventory_id: str,
    ) -> List[VerificationRecord]:
        """Get all verification records for an inventory."""
        records = [
            r for r in self._records.values()
            if r.inventory_id == inventory_id
        ]
        return sorted(records, key=lambda r: r.started_at)

    def get_record(self, record_id: str) -> Optional[VerificationRecord]:
        """Get a verification record by ID."""
        return self._records.get(record_id)

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_record_or_raise(self, record_id: str) -> VerificationRecord:
        """Retrieve record or raise ValueError."""
        record = self._records.get(record_id)
        if record is None:
            raise ValueError(f"Verification record not found: {record_id}")
        return record

    def _validate_inventory_exists(self, inventory_id: str) -> None:
        """Validate that the inventory exists."""
        if inventory_id not in self._inventory_store:
            raise ValueError(f"Inventory not found: {inventory_id}")

    def _check_no_active_review(self, inventory_id: str) -> None:
        """Check that no active review is in progress."""
        active = self._find_active_record(inventory_id)
        if active and active.stage not in (
            VerificationStage.VERIFIED,
            VerificationStage.DRAFT,
        ):
            raise ValueError(
                f"Active verification already exists for inventory "
                f"{inventory_id} (stage={active.stage.value})"
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
        current: VerificationStage,
        target: VerificationStage,
    ) -> None:
        """Validate that a state transition is allowed."""
        valid = _VALID_TRANSITIONS.get(current, [])
        if target not in valid:
            raise ValueError(
                f"Invalid transition: {current.value} -> {target.value}. "
                f"Valid targets: {[v.value for v in valid]}"
            )

    @staticmethod
    def _find_finding(
        record: VerificationRecord,
        finding_id: str,
    ) -> Finding:
        """Find a finding within a record."""
        for finding in record.findings:
            if finding.id == finding_id:
                return finding
        raise ValueError(f"Finding not found: {finding_id}")

    @staticmethod
    def _build_findings_summary(record: VerificationRecord) -> FindingsSummary:
        """Build a FindingsSummary from the record's findings."""
        total = len(record.findings)
        open_count = sum(
            1 for f in record.findings if f.status == FindingStatus.OPEN
        )
        in_progress_count = sum(
            1 for f in record.findings if f.status == FindingStatus.IN_PROGRESS
        )
        resolved_count = sum(
            1 for f in record.findings
            if f.status in (FindingStatus.RESOLVED, FindingStatus.ACCEPTED)
        )

        by_severity: Dict[str, int] = {}
        for f in record.findings:
            by_severity[f.severity.value] = by_severity.get(f.severity.value, 0) + 1

        by_category: Dict[str, int] = {}
        for f in record.findings:
            cat_key = f.iso_category.value if f.iso_category else "unspecified"
            by_category[cat_key] = by_category.get(cat_key, 0) + 1

        critical_open = any(
            f.severity == FindingSeverity.CRITICAL
            and f.status in (FindingStatus.OPEN, FindingStatus.IN_PROGRESS)
            for f in record.findings
        )

        return FindingsSummary(
            total_findings=total,
            open_count=open_count,
            in_progress_count=in_progress_count,
            resolved_count=resolved_count,
            by_severity=by_severity,
            by_category=by_category,
            critical_open=critical_open,
        )

    @staticmethod
    def _auto_generate_statement(
        record: VerificationRecord,
        summary: FindingsSummary,
    ) -> str:
        """Auto-generate a verification statement based on findings."""
        total = summary.total_findings
        open_count = summary.open_count + summary.in_progress_count
        critical_open = summary.critical_open
        level = record.level.value.replace("_", " ").title()

        if record.stage == VerificationStage.INTERNAL_REVIEW:
            return (
                "Internal review is in progress. Verification statement "
                "will be generated upon completion."
            )

        if total == 0:
            return (
                f"Based on this {level} engagement of the GHG inventory "
                f"prepared in accordance with ISO 14064-1:2018, no findings "
                f"were identified. The GHG inventory appears to be a fair "
                f"representation of the organization's GHG emissions and removals."
            )

        if open_count == 0:
            return (
                f"Based on this {level} engagement of the GHG inventory "
                f"prepared in accordance with ISO 14064-1:2018, {total} "
                f"finding(s) were identified, all of which have been "
                f"satisfactorily resolved. The GHG inventory, as corrected, "
                f"appears to be a fair representation of the organization's "
                f"GHG emissions and removals."
            )

        if critical_open:
            return (
                f"Based on this {level} engagement of the GHG inventory "
                f"prepared in accordance with ISO 14064-1:2018, {total} "
                f"finding(s) were identified, including critical finding(s). "
                f"{open_count} finding(s) remain unresolved. The verifier is "
                f"unable to provide an unqualified opinion until all critical "
                f"findings are resolved."
            )

        return (
            f"Based on this {level} engagement of the GHG inventory "
            f"prepared in accordance with ISO 14064-1:2018, {total} "
            f"finding(s) were identified, of which {open_count} remain "
            f"unresolved. The verifier recommends resolving outstanding "
            f"findings before publication."
        )
