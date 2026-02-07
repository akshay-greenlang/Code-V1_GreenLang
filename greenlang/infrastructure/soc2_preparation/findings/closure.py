# -*- coding: utf-8 -*-
"""
Finding Closure - SEC-009 Phase 6

Verify and close remediated findings. Provides closure request workflow,
verification of remediation effectiveness, and documentation of closure
with full audit trail.

Closure Process:
    1. Remediation owner requests closure with evidence
    2. Independent verifier reviews remediation
    3. Verifier confirms effectiveness or rejects
    4. Finding is closed or sent back for additional work

Example:
    >>> closure = FindingClosure(tracker, workflow)
    >>> await closure.request_closure(
    ...     finding_id=finding_uuid,
    ...     evidence=[evidence_uuid_1, evidence_uuid_2],
    ...     notes="MFA enabled for all 5 admin accounts",
    ... )
    >>> success = await closure.verify_remediation(
    ...     finding_id=finding_uuid,
    ...     verified_by=verifier_uuid,
    ... )
    >>> if success:
    ...     await closure.close_finding(finding_uuid)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from greenlang.infrastructure.soc2_preparation.findings.tracker import (
    Finding,
    FindingStatus,
    FindingTracker,
)
from greenlang.infrastructure.soc2_preparation.findings.remediation import (
    RemediationState,
    RemediationWorkflow,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class VerificationStatus(str, Enum):
    """Status of closure verification."""

    PENDING = "pending"
    """Verification request pending review."""

    IN_PROGRESS = "in_progress"
    """Verification actively in progress."""

    APPROVED = "approved"
    """Remediation verified as effective."""

    REJECTED = "rejected"
    """Remediation found ineffective or incomplete."""

    ADDITIONAL_INFO_REQUIRED = "additional_info_required"
    """More information needed from remediation owner."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class ClosureRequest(BaseModel):
    """Request to close a remediated finding.

    Attributes:
        request_id: Unique request identifier.
        finding_id: Associated finding ID.
        evidence_ids: Evidence supporting closure.
        notes: Closure request notes.
        requested_by: Who requested closure.
        requested_at: Request timestamp.
        verification_status: Current verification status.
        verifier_id: Assigned verifier.
        verification_notes: Notes from verification.
        verified_at: Verification timestamp.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier.",
    )
    finding_id: str = Field(
        ...,
        description="Associated finding ID.",
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Evidence IDs supporting closure.",
    )
    notes: str = Field(
        default="",
        max_length=4096,
        description="Closure request notes.",
    )
    requested_by: str = Field(
        default="",
        description="Who requested closure.",
    )
    requested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request timestamp.",
    )
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.PENDING,
        description="Current verification status.",
    )
    verifier_id: Optional[str] = Field(
        default=None,
        description="Assigned verifier ID.",
    )
    verification_notes: str = Field(
        default="",
        max_length=4096,
        description="Notes from verification.",
    )
    verified_at: Optional[datetime] = Field(
        default=None,
        description="Verification timestamp.",
    )


class ClosureVerification(BaseModel):
    """Verification record for finding closure.

    Attributes:
        verification_id: Unique verification ID.
        finding_id: Associated finding ID.
        closure_request_id: Associated closure request.
        verifier_id: Verifier user ID.
        verifier_name: Verifier display name.
        verification_type: Type of verification performed.
        steps_performed: Verification steps taken.
        result: Verification result (approved/rejected).
        findings: Issues found during verification.
        recommendations: Recommendations if rejected.
        verified_at: Verification timestamp.
    """

    model_config = ConfigDict(extra="forbid")

    verification_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique verification ID.",
    )
    finding_id: str = Field(
        ...,
        description="Associated finding ID.",
    )
    closure_request_id: str = Field(
        ...,
        description="Associated closure request.",
    )
    verifier_id: str = Field(
        ...,
        description="Verifier user ID.",
    )
    verifier_name: str = Field(
        default="",
        description="Verifier display name.",
    )
    verification_type: str = Field(
        default="manual",
        description="Verification type: manual, automated, walkthrough.",
    )
    steps_performed: List[str] = Field(
        default_factory=list,
        description="Verification steps performed.",
    )
    result: str = Field(
        default="pending",
        description="Result: approved, rejected, pending.",
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Issues found during verification.",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations if rejected.",
    )
    verified_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Verification timestamp.",
    )


class ClosureReport(BaseModel):
    """Report generated when a finding is closed.

    Attributes:
        report_id: Unique report ID.
        finding_id: Associated finding ID.
        finding_title: Finding title.
        finding_classification: Finding classification.
        identified_at: When finding was identified.
        closed_at: When finding was closed.
        days_to_close: Days from identification to closure.
        root_cause: Root cause summary.
        remediation_summary: Remediation summary.
        verification_summary: Verification summary.
        evidence_count: Number of evidence items.
        generated_at: Report generation time.
    """

    model_config = ConfigDict(extra="forbid")

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique report ID.",
    )
    finding_id: str
    finding_title: str
    finding_classification: str
    identified_at: datetime
    closed_at: datetime
    days_to_close: int
    root_cause: str
    remediation_summary: str
    verification_summary: str
    evidence_count: int
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Finding Closure
# ---------------------------------------------------------------------------


class FindingClosure:
    """Manage finding closure verification and documentation.

    Coordinates between finding tracker and remediation workflow to
    ensure proper verification and documentation of closed findings.

    Attributes:
        _tracker: Finding tracker instance.
        _workflow: Remediation workflow instance.
        _closure_requests: Closure requests by finding_id.
        _verifications: Verification records by finding_id.
        _closure_reports: Closure reports by finding_id.
    """

    def __init__(
        self,
        tracker: FindingTracker,
        workflow: RemediationWorkflow,
    ) -> None:
        """Initialize finding closure manager.

        Args:
            tracker: Finding tracker instance.
            workflow: Remediation workflow instance.
        """
        self._tracker = tracker
        self._workflow = workflow
        self._closure_requests: Dict[str, ClosureRequest] = {}
        self._verifications: Dict[str, List[ClosureVerification]] = {}
        self._closure_reports: Dict[str, ClosureReport] = {}
        logger.info("FindingClosure initialized")

    # ------------------------------------------------------------------
    # Closure Request
    # ------------------------------------------------------------------

    async def request_closure(
        self,
        finding_id: uuid.UUID,
        evidence: List[uuid.UUID],
        notes: str,
        requested_by: str = "",
    ) -> ClosureRequest:
        """Request closure of a remediated finding.

        Args:
            finding_id: Finding identifier.
            evidence: Evidence IDs supporting closure.
            notes: Closure request notes.
            requested_by: Who is requesting closure.

        Returns:
            Created ClosureRequest.

        Raises:
            ValueError: If finding not found or not ready for closure.
        """
        finding_id_str = str(finding_id)

        # Get finding
        finding = await self._tracker.get_finding(finding_id)
        if finding is None:
            raise ValueError(f"Finding {finding_id_str} not found")

        # Check remediation state
        state = await self._workflow.get_state(finding_id)
        if state not in (RemediationState.IMPLEMENTED, RemediationState.TESTED):
            raise ValueError(
                f"Finding must be IMPLEMENTED or TESTED before closure. "
                f"Current state: {state.value}"
            )

        # Create closure request
        request = ClosureRequest(
            finding_id=finding_id_str,
            evidence_ids=[str(e) for e in evidence],
            notes=notes,
            requested_by=requested_by,
        )

        self._closure_requests[finding_id_str] = request

        # Update finding status
        await self._tracker.update_status(finding_id, "pending_closure")

        logger.info(
            "Closure requested for finding %s with %d evidence items",
            finding_id_str[:8],
            len(evidence),
        )

        return request

    async def get_closure_request(
        self,
        finding_id: uuid.UUID,
    ) -> Optional[ClosureRequest]:
        """Get closure request for a finding.

        Args:
            finding_id: Finding identifier.

        Returns:
            ClosureRequest if exists.
        """
        return self._closure_requests.get(str(finding_id))

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    async def verify_remediation(
        self,
        finding_id: uuid.UUID,
        verified_by: uuid.UUID,
        verifier_name: str = "",
        verification_type: str = "manual",
        steps_performed: Optional[List[str]] = None,
    ) -> bool:
        """Verify remediation effectiveness.

        Performs verification of the remediation based on:
        - Evidence review
        - Retesting of control (if applicable)
        - Comparison to original finding conditions

        Args:
            finding_id: Finding identifier.
            verified_by: Verifier user ID.
            verifier_name: Verifier display name.
            verification_type: Type of verification.
            steps_performed: Verification steps taken.

        Returns:
            True if verification passes, False otherwise.

        Raises:
            ValueError: If finding not found or no closure request exists.
        """
        finding_id_str = str(finding_id)

        # Get closure request
        request = self._closure_requests.get(finding_id_str)
        if request is None:
            raise ValueError(f"No closure request for finding {finding_id_str}")

        # Get finding
        finding = await self._tracker.get_finding(finding_id)
        if finding is None:
            raise ValueError(f"Finding {finding_id_str} not found")

        # Perform verification checks
        verification_passed = True
        findings: List[str] = []
        recommendations: List[str] = []

        # Check evidence exists
        if not request.evidence_ids:
            verification_passed = False
            findings.append("No evidence provided")
            recommendations.append("Provide evidence of remediation")

        # Check remediation evidence
        remediation_evidence = await self._workflow.get_evidence(finding_id)
        if not remediation_evidence:
            verification_passed = False
            findings.append("No remediation evidence uploaded")
            recommendations.append("Upload evidence showing remediation implementation")

        # Check progress completed
        progress = await self._workflow.get_current_progress(finding_id)
        if progress and progress.progress_percentage < 100:
            verification_passed = False
            findings.append(f"Remediation only {progress.progress_percentage}% complete")
            recommendations.append("Complete all remediation steps")

        # Create verification record
        verification = ClosureVerification(
            finding_id=finding_id_str,
            closure_request_id=request.request_id,
            verifier_id=str(verified_by),
            verifier_name=verifier_name,
            verification_type=verification_type,
            steps_performed=steps_performed or ["Evidence review", "Status check"],
            result="approved" if verification_passed else "rejected",
            findings=findings,
            recommendations=recommendations,
        )

        if finding_id_str not in self._verifications:
            self._verifications[finding_id_str] = []
        self._verifications[finding_id_str].append(verification)

        # Update request status
        request.verification_status = (
            VerificationStatus.APPROVED if verification_passed
            else VerificationStatus.REJECTED
        )
        request.verifier_id = str(verified_by)
        request.verified_at = datetime.now(timezone.utc)
        request.verification_notes = "; ".join(findings) if findings else "Verification passed"

        # Update workflow state if approved
        if verification_passed:
            await self._workflow.transition_state(finding_id, RemediationState.TESTED)

        logger.info(
            "Verification %s for finding %s (by %s)",
            "approved" if verification_passed else "rejected",
            finding_id_str[:8],
            verifier_name or str(verified_by)[:8],
        )

        return verification_passed

    async def get_verifications(
        self,
        finding_id: uuid.UUID,
    ) -> List[ClosureVerification]:
        """Get all verification records for a finding.

        Args:
            finding_id: Finding identifier.

        Returns:
            List of verification records.
        """
        return self._verifications.get(str(finding_id), [])

    # ------------------------------------------------------------------
    # Closure
    # ------------------------------------------------------------------

    async def close_finding(
        self,
        finding_id: uuid.UUID,
        closed_by: str = "",
    ) -> Finding:
        """Close a verified finding.

        Args:
            finding_id: Finding identifier.
            closed_by: Who is closing the finding.

        Returns:
            Closed Finding.

        Raises:
            ValueError: If finding not verified or already closed.
        """
        finding_id_str = str(finding_id)

        # Check verification status
        request = self._closure_requests.get(finding_id_str)
        if request is None:
            raise ValueError(f"No closure request for finding {finding_id_str}")

        if request.verification_status != VerificationStatus.APPROVED:
            raise ValueError(
                f"Finding must be verified before closure. "
                f"Current status: {request.verification_status.value}"
            )

        # Get finding
        finding = await self._tracker.get_finding(finding_id)
        if finding is None:
            raise ValueError(f"Finding {finding_id_str} not found")

        if finding.status == FindingStatus.CLOSED:
            raise ValueError("Finding is already closed")

        # Close finding
        await self._tracker.update_status(
            finding_id,
            "closed",
            notes=f"Closed by {closed_by}. Verification approved."
        )

        # Update workflow state
        await self._workflow.transition_state(finding_id, RemediationState.CLOSED)

        # Generate closure report
        await self._generate_closure_report(finding_id)

        # Get updated finding
        finding = await self._tracker.get_finding(finding_id)

        logger.info("Closed finding %s (by %s)", finding_id_str[:8], closed_by)

        return finding  # type: ignore

    async def reopen_finding(
        self,
        finding_id: uuid.UUID,
        reason: str,
        reopened_by: str = "",
    ) -> Finding:
        """Reopen a previously closed finding.

        Args:
            finding_id: Finding identifier.
            reason: Reason for reopening.
            reopened_by: Who is reopening.

        Returns:
            Reopened Finding.

        Raises:
            ValueError: If finding not found or not closed.
        """
        finding_id_str = str(finding_id)

        # Get finding
        finding = await self._tracker.get_finding(finding_id)
        if finding is None:
            raise ValueError(f"Finding {finding_id_str} not found")

        if finding.status != FindingStatus.CLOSED:
            raise ValueError("Only closed findings can be reopened")

        # Reopen finding
        await self._tracker.update_status(
            finding_id,
            "reopened",
            notes=f"Reopened by {reopened_by}: {reason}"
        )

        # Reset remediation state
        self._workflow._finding_states[finding_id_str] = RemediationState.IN_PROGRESS

        # Clear closure request
        if finding_id_str in self._closure_requests:
            del self._closure_requests[finding_id_str]

        # Get updated finding
        finding = await self._tracker.get_finding(finding_id)

        logger.warning(
            "Reopened finding %s: %s (by %s)",
            finding_id_str[:8],
            reason,
            reopened_by,
        )

        return finding  # type: ignore

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    async def _generate_closure_report(
        self,
        finding_id: uuid.UUID,
    ) -> ClosureReport:
        """Generate closure report for a finding.

        Args:
            finding_id: Finding identifier.

        Returns:
            Generated ClosureReport.
        """
        finding_id_str = str(finding_id)

        # Get finding
        finding = await self._tracker.get_finding(finding_id)
        if finding is None:
            raise ValueError(f"Finding {finding_id_str} not found")

        # Get plan
        plan = await self._workflow.get_plan(finding_id)

        # Get evidence count
        evidence = await self._workflow.get_evidence(finding_id)
        evidence_count = len(evidence)

        # Get verification summary
        verifications = await self.get_verifications(finding_id)
        verification_summary = ""
        if verifications:
            latest = verifications[-1]
            verification_summary = (
                f"Verified by {latest.verifier_name or latest.verifier_id[:8]} "
                f"on {latest.verified_at.isoformat()} "
                f"via {latest.verification_type}"
            )

        closed_at = finding.closed_at or datetime.now(timezone.utc)
        days_to_close = (closed_at - finding.created_at).days

        report = ClosureReport(
            finding_id=finding_id_str,
            finding_title=finding.title,
            finding_classification=finding.classification.value if finding.classification else "unknown",
            identified_at=finding.created_at,
            closed_at=closed_at,
            days_to_close=days_to_close,
            root_cause=finding.root_cause,
            remediation_summary=plan.description if plan else "No plan documented",
            verification_summary=verification_summary,
            evidence_count=evidence_count,
        )

        self._closure_reports[finding_id_str] = report

        logger.debug("Generated closure report for finding %s", finding_id_str[:8])
        return report

    async def generate_closure_report(
        self,
        finding_id: uuid.UUID,
    ) -> str:
        """Generate formatted closure report text.

        Args:
            finding_id: Finding identifier.

        Returns:
            Formatted report string.
        """
        finding_id_str = str(finding_id)

        # Get or generate report
        report = self._closure_reports.get(finding_id_str)
        if report is None:
            report = await self._generate_closure_report(finding_id)

        lines = [
            "=" * 70,
            "  FINDING CLOSURE REPORT",
            "=" * 70,
            "",
            f"  Finding ID:        {report.finding_id}",
            f"  Title:             {report.finding_title}",
            f"  Classification:    {report.finding_classification.upper()}",
            "",
            f"  Identified:        {report.identified_at.strftime('%Y-%m-%d %H:%M UTC')}",
            f"  Closed:            {report.closed_at.strftime('%Y-%m-%d %H:%M UTC')}",
            f"  Days to Close:     {report.days_to_close}",
            "",
            "  ROOT CAUSE",
            "-" * 70,
            f"  {report.root_cause or 'Not documented'}",
            "",
            "  REMEDIATION",
            "-" * 70,
            f"  {report.remediation_summary}",
            "",
            "  VERIFICATION",
            "-" * 70,
            f"  {report.verification_summary}",
            f"  Evidence Items:    {report.evidence_count}",
            "",
            "=" * 70,
            f"  Report Generated:  {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 70,
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Query Methods
    # ------------------------------------------------------------------

    async def list_pending_verification(self) -> List[ClosureRequest]:
        """Get all closure requests pending verification.

        Returns:
            List of pending closure requests.
        """
        return [
            r for r in self._closure_requests.values()
            if r.verification_status == VerificationStatus.PENDING
        ]

    async def list_rejected(self) -> List[ClosureRequest]:
        """Get all rejected closure requests.

        Returns:
            List of rejected requests.
        """
        return [
            r for r in self._closure_requests.values()
            if r.verification_status == VerificationStatus.REJECTED
        ]

    async def get_statistics(self) -> Dict[str, Any]:
        """Get closure statistics.

        Returns:
            Dictionary with closure metrics.
        """
        total_requests = len(self._closure_requests)
        by_status: Dict[str, int] = {}

        for request in self._closure_requests.values():
            status = request.verification_status.value
            by_status[status] = by_status.get(status, 0) + 1

        total_closed = len(self._closure_reports)

        # Calculate average days to close
        avg_days = 0.0
        if self._closure_reports:
            total_days = sum(r.days_to_close for r in self._closure_reports.values())
            avg_days = total_days / len(self._closure_reports)

        return {
            "total_closure_requests": total_requests,
            "by_verification_status": by_status,
            "total_closed": total_closed,
            "avg_days_to_close": round(avg_days, 1),
            "pending_verification": by_status.get(VerificationStatus.PENDING.value, 0),
            "rejected": by_status.get(VerificationStatus.REJECTED.value, 0),
        }


__all__ = [
    "FindingClosure",
    "ClosureRequest",
    "ClosureVerification",
    "ClosureReport",
    "VerificationStatus",
]
