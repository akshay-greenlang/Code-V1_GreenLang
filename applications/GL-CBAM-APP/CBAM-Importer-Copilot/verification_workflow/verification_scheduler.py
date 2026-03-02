# -*- coding: utf-8 -*-
"""
VerificationSchedulerEngine - Site visit scheduling and outcome management.

Manages the scheduling of CBAM verification site visits, tracks visit status
through its lifecycle (scheduled -> completed/cancelled), and records outcomes
with findings. Implements the annual (2026) and biennial (2027+) visit
frequency rules per the Omnibus Simplification Package.

Example:
    >>> registry = VerifierRegistryEngine()
    >>> scheduler = VerificationSchedulerEngine(registry)
    >>> visit = scheduler.schedule_site_visit(
    ...     installation_id="INST-001",
    ...     verifier_id="VER-001",
    ...     visit_date=date(2026, 6, 15),
    ...     visit_type=VisitType.ON_SITE,
    ... )

Version: 1.1.0
Author: GreenLang CBAM Team
License: Proprietary
"""

import hashlib
import logging
import threading
import uuid
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from verification_workflow.verifier_registry import (
    COIResult,
    VerifierRegistryEngine,
    VerifierStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANNUAL_VISIT_REQUIRED_UNTIL = 2026  # Annual in 2026, biennial from 2027+


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VisitType(str, Enum):
    """Type of verification site visit."""
    ON_SITE = "on_site"
    REMOTE = "remote"
    HYBRID = "hybrid"


class VisitStatus(str, Enum):
    """Lifecycle status of a site visit."""
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    RESCHEDULED = "rescheduled"


class VisitOutcome(str, Enum):
    """Verification outcome of a completed visit."""
    PASS = "pass"
    FAIL = "fail"
    CONDITIONAL = "conditional"
    PENDING_REVIEW = "pending_review"


class FindingSeverity(str, Enum):
    """Severity of a verification finding."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    OBSERVATION = "observation"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class VisitFinding(BaseModel):
    """Individual finding raised during a verification visit."""

    finding_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    cn_code: str = Field(default="", description="CN code the finding relates to")
    category: str = Field(default="general", description="Finding category")
    severity: FindingSeverity = Field(default=FindingSeverity.MINOR)
    description: str = Field(default="")
    corrective_action: str = Field(default="", description="Required corrective action")
    due_date: Optional[date] = Field(default=None, description="Deadline for corrective action")
    resolved: bool = Field(default=False)
    resolved_at: Optional[datetime] = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}


class SiteVisit(BaseModel):
    """A CBAM verification site visit record."""

    visit_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    installation_id: str = Field(..., description="Installation being verified")
    verifier_id: str = Field(..., description="Assigned verifier")
    verifier_name: str = Field(default="", description="Verifier company name")
    scheduled_date: date = Field(..., description="Planned visit date")
    actual_date: Optional[date] = Field(default=None, description="Actual visit date")
    visit_type: VisitType = Field(default=VisitType.ON_SITE)
    status: VisitStatus = Field(default=VisitStatus.SCHEDULED)
    outcome: Optional[VisitOutcome] = Field(default=None)
    findings: List[VisitFinding] = Field(default_factory=list)
    report_reference: str = Field(default="", description="Verification report document reference")
    report_issued_date: Optional[date] = Field(default=None)
    year: int = Field(default=0, description="CBAM reporting year being verified")
    sector: str = Field(default="", description="CBAM sector of the installation")
    country: str = Field(default="", description="Country of the installation")
    notes: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(default="")

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Engine Implementation
# ---------------------------------------------------------------------------

class VerificationSchedulerEngine:
    """
    Manages the scheduling and lifecycle of CBAM verification site visits.

    Collaborates with VerifierRegistryEngine to validate verifier eligibility
    and COI before assignment. Enforces the annual/biennial visit frequency
    rules from the Omnibus Simplification Package.

    Args:
        registry: VerifierRegistryEngine instance for verifier lookups.
    """

    def __init__(self, registry: Optional[VerifierRegistryEngine] = None) -> None:
        """Initialise with a verifier registry."""
        self._registry = registry or VerifierRegistryEngine()
        self._lock = threading.RLock()
        self._visits: Dict[str, SiteVisit] = {}
        self._installation_visits: Dict[str, List[str]] = {}
        logger.info("VerificationSchedulerEngine initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule_site_visit(
        self,
        installation_id: str,
        verifier_id: str,
        visit_date: date,
        visit_type: VisitType = VisitType.ON_SITE,
        year: Optional[int] = None,
        sector: str = "",
        country: str = "",
        notes: str = "",
    ) -> SiteVisit:
        """
        Schedule a verification site visit.

        Validates that the verifier is active, accredited, and free of COI
        before creating the visit record.

        Args:
            installation_id: Installation to verify.
            verifier_id: Assigned verifier ID.
            visit_date: Planned date for the visit.
            visit_type: On-site, remote, or hybrid.
            year: CBAM reporting year. Defaults to visit_date year.
            sector: CBAM sector of the installation.
            country: Installation country.
            notes: Additional notes.

        Returns:
            Scheduled SiteVisit record.

        Raises:
            ValueError: If verifier is invalid, expired, or has COI.
        """
        with self._lock:
            # Validate verifier
            verifier = self._registry.get_verifier(verifier_id)
            if verifier.status != VerifierStatus.ACTIVE:
                raise ValueError(
                    f"Verifier {verifier_id} is not active (status={verifier.status.value})"
                )

            if not self._registry.check_accreditation_validity(verifier_id):
                raise ValueError(
                    f"Verifier {verifier_id} accreditation is not valid"
                )

            # COI check
            coi = self._registry.check_conflict_of_interest(verifier_id, installation_id)
            if coi.result == COIResult.CONFLICT_DETECTED:
                raise ValueError(
                    f"Conflict of interest detected between verifier {verifier_id} "
                    f"and installation {installation_id}: {coi.reasons}"
                )

            reporting_year = year or visit_date.year

            visit = SiteVisit(
                installation_id=installation_id,
                verifier_id=verifier_id,
                verifier_name=verifier.company_name,
                scheduled_date=visit_date,
                visit_type=visit_type,
                status=VisitStatus.SCHEDULED,
                year=reporting_year,
                sector=sector,
                country=country,
                notes=notes,
            )
            visit.provenance_hash = self._hash_visit(visit)

            self._visits[visit.visit_id] = visit
            if installation_id not in self._installation_visits:
                self._installation_visits[installation_id] = []
            self._installation_visits[installation_id].append(visit.visit_id)

            logger.info(
                "Visit scheduled: id=%s installation=%s verifier=%s date=%s type=%s",
                visit.visit_id, installation_id, verifier_id, visit_date, visit_type.value,
            )
            return visit

    def get_visit(self, visit_id: str) -> SiteVisit:
        """
        Retrieve a visit by ID.

        Args:
            visit_id: Visit identifier.

        Returns:
            SiteVisit record.

        Raises:
            KeyError: If visit not found.
        """
        with self._lock:
            if visit_id not in self._visits:
                raise KeyError(f"Visit {visit_id} not found")
            return self._visits[visit_id]

    def get_visit_schedule(self, installation_id: str) -> List[SiteVisit]:
        """
        Return all visits for an installation, ordered by date.

        Args:
            installation_id: Installation identifier.

        Returns:
            List of SiteVisit records sorted by scheduled_date.
        """
        with self._lock:
            visit_ids = self._installation_visits.get(installation_id, [])
            visits = [self._visits[vid] for vid in visit_ids if vid in self._visits]
            return sorted(visits, key=lambda v: v.scheduled_date)

    def get_upcoming_visits(self, days_ahead: int = 30) -> List[SiteVisit]:
        """
        Return visits scheduled within the specified window.

        Args:
            days_ahead: Number of days to look ahead (default 30).

        Returns:
            List of upcoming SiteVisit records.
        """
        with self._lock:
            cutoff = date.today() + timedelta(days=days_ahead)
            today = date.today()
            upcoming: List[SiteVisit] = []
            for v in self._visits.values():
                if v.status in (VisitStatus.SCHEDULED, VisitStatus.CONFIRMED):
                    if today <= v.scheduled_date <= cutoff:
                        upcoming.append(v)
            return sorted(upcoming, key=lambda x: x.scheduled_date)

    def record_visit_outcome(
        self,
        visit_id: str,
        outcome: VisitOutcome,
        findings: Optional[List[VisitFinding]] = None,
        report_reference: str = "",
        actual_date: Optional[date] = None,
    ) -> SiteVisit:
        """
        Record the outcome of a completed verification visit.

        Args:
            visit_id: Visit identifier.
            outcome: Pass, fail, conditional, or pending review.
            findings: List of findings raised during the visit.
            report_reference: Document reference for the verification report.
            actual_date: Actual date the visit took place (if different from scheduled).

        Returns:
            Updated SiteVisit with outcome recorded.

        Raises:
            KeyError: If visit not found.
            ValueError: If visit is already completed or cancelled.
        """
        with self._lock:
            visit = self.get_visit(visit_id)
            if visit.status in (VisitStatus.COMPLETED, VisitStatus.CANCELLED):
                raise ValueError(
                    f"Cannot record outcome for visit {visit_id} "
                    f"with status {visit.status.value}"
                )

            visit.status = VisitStatus.COMPLETED
            visit.outcome = outcome
            visit.findings = findings or []
            visit.report_reference = report_reference
            visit.actual_date = actual_date or visit.scheduled_date
            visit.report_issued_date = date.today()
            visit.updated_at = datetime.utcnow()
            visit.provenance_hash = self._hash_visit(visit)

            # Record in registry for performance tracking
            self._registry.record_visit_outcome(
                verifier_id=visit.verifier_id,
                installation_id=visit.installation_id,
                outcome=outcome.value,
                finding_count=len(visit.findings),
                days_to_complete=(date.today() - visit.actual_date).days,
                sector=visit.sector,
                country=visit.country,
            )

            logger.info(
                "Visit outcome recorded: id=%s outcome=%s findings=%d",
                visit_id, outcome.value, len(visit.findings),
            )
            return visit

    def cancel_visit(self, visit_id: str, reason: str = "") -> SiteVisit:
        """
        Cancel a scheduled visit.

        Args:
            visit_id: Visit identifier.
            reason: Cancellation reason.

        Returns:
            Updated SiteVisit with cancelled status.
        """
        with self._lock:
            visit = self.get_visit(visit_id)
            if visit.status == VisitStatus.COMPLETED:
                raise ValueError(f"Cannot cancel completed visit {visit_id}")
            visit.status = VisitStatus.CANCELLED
            visit.notes = f"{visit.notes}\nCancelled: {reason}".strip()
            visit.updated_at = datetime.utcnow()
            visit.provenance_hash = self._hash_visit(visit)

            logger.info("Visit cancelled: id=%s reason=%s", visit_id, reason)
            return visit

    def is_visit_required(self, installation_id: str, year: int) -> bool:
        """
        Determine if a verification visit is required for the given year.

        Rules per Omnibus Simplification:
            - 2026: Annual visit required for all installations.
            - 2027+: Biennial (every 2 years). Visit required if no visit
              was completed in the previous year.

        Args:
            installation_id: Installation identifier.
            year: Year to check.

        Returns:
            True if a visit is required.
        """
        with self._lock:
            if year <= ANNUAL_VISIT_REQUIRED_UNTIL:
                # Annual: check if there is a completed visit for this year
                return not self._has_completed_visit(installation_id, year)

            # Biennial from 2027: required if no visit in current or previous year
            has_current = self._has_completed_visit(installation_id, year)
            has_previous = self._has_completed_visit(installation_id, year - 1)
            return not (has_current or has_previous)

    def get_next_required_visit(self, installation_id: str) -> date:
        """
        Calculate the next date by which a verification visit must occur.

        Args:
            installation_id: Installation identifier.

        Returns:
            The latest date by which the next visit should be completed.
        """
        with self._lock:
            current_year = date.today().year
            check_year = current_year

            # Look up to 3 years ahead
            for offset in range(4):
                yr = current_year + offset
                if self.is_visit_required(installation_id, yr):
                    check_year = yr
                    break

            # Default: visit must occur by end of Q3 of the required year
            return date(check_year, 9, 30)

    def assign_verifier(
        self, installation_id: str, sector: str
    ) -> Optional[Dict[str, Any]]:
        """
        Auto-assign a verifier based on sector expertise and COI clearance.

        Selection criteria:
            1. Active and accredited verifiers with matching sector expertise.
            2. No conflict of interest with the installation.
            3. Prefer verifier with lowest current workload (fewest upcoming visits).

        Args:
            installation_id: Installation to assign verifier for.
            sector: CBAM sector of the installation.

        Returns:
            Dict with verifier details and assignment rationale, or None if
            no suitable verifier found.
        """
        with self._lock:
            candidates = self._registry.search_verifiers(
                sector_expertise=sector,
                accreditation_status=VerifierStatus.ACTIVE,
            )

            eligible: List[Dict[str, Any]] = []
            for v in candidates:
                # Check accreditation validity
                if not self._registry.check_accreditation_validity(v.verifier_id):
                    continue

                # COI check
                coi = self._registry.check_conflict_of_interest(
                    v.verifier_id, installation_id
                )
                if coi.result == COIResult.CONFLICT_DETECTED:
                    continue

                # Count upcoming visits (workload)
                workload = self._count_upcoming_visits(v.verifier_id)

                eligible.append({
                    "verifier": v,
                    "workload": workload,
                    "coi_status": coi.result.value,
                })

            if not eligible:
                logger.warning(
                    "No eligible verifier found for installation=%s sector=%s",
                    installation_id, sector,
                )
                return None

            # Sort by workload (ascending)
            eligible.sort(key=lambda x: x["workload"])
            best = eligible[0]

            logger.info(
                "Verifier assigned: installation=%s verifier=%s workload=%d",
                installation_id, best["verifier"].verifier_id, best["workload"],
            )
            return {
                "verifier_id": best["verifier"].verifier_id,
                "company_name": best["verifier"].company_name,
                "sector_expertise": best["verifier"].sector_expertise,
                "workload": best["workload"],
                "coi_status": best["coi_status"],
                "assignment_rationale": "Lowest workload among eligible verifiers with matching sector expertise and no COI.",
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_completed_visit(self, installation_id: str, year: int) -> bool:
        """Check if a completed visit exists for the installation-year pair."""
        visit_ids = self._installation_visits.get(installation_id, [])
        for vid in visit_ids:
            v = self._visits.get(vid)
            if v and v.status == VisitStatus.COMPLETED and v.year == year:
                return True
        return False

    def _count_upcoming_visits(self, verifier_id: str) -> int:
        """Count scheduled/confirmed visits for a verifier."""
        count = 0
        for v in self._visits.values():
            if v.verifier_id == verifier_id and v.status in (
                VisitStatus.SCHEDULED, VisitStatus.CONFIRMED
            ):
                count += 1
        return count

    # ------------------------------------------------------------------
    # Provenance hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_visit(visit: SiteVisit) -> str:
        """Compute SHA-256 provenance hash for a SiteVisit."""
        payload = (
            f"{visit.visit_id}|{visit.installation_id}|{visit.verifier_id}|"
            f"{visit.scheduled_date}|{visit.visit_type.value}|{visit.status.value}|"
            f"{visit.outcome}|{len(visit.findings)}|{visit.report_reference}|"
            f"{visit.updated_at.isoformat()}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
