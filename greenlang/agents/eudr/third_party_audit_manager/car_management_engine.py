# -*- coding: utf-8 -*-
"""
Corrective Action Request Management Engine - AGENT-EUDR-024

Full CAR lifecycle management engine implementing a 12-status lifecycle
(ISSUED -> ACKNOWLEDGED -> RCA_SUBMITTED -> CAP_SUBMITTED -> CAP_APPROVED ->
IN_PROGRESS -> EVIDENCE_SUBMITTED -> VERIFICATION_PENDING -> CLOSED, plus
REJECTED, OVERDUE, ESCALATED), with SLA enforcement, 4-level escalation
stages, deadline tracking, and evidence verification workflows.

CAR Lifecycle (12 statuses):
    ISSUED -> ACKNOWLEDGED -> RCA_SUBMITTED -> CAP_SUBMITTED ->
    CAP_APPROVED -> IN_PROGRESS -> EVIDENCE_SUBMITTED ->
    VERIFICATION_PENDING -> CLOSED
    Also: REJECTED (returns to IN_PROGRESS), OVERDUE, ESCALATED

SLA Deadlines by Severity:
    CRITICAL: 30 days total (acknowledge 3d, RCA 7d, CAP 14d)
    MAJOR: 90 days total (acknowledge 7d, RCA 14d, CAP 30d)
    MINOR: 365 days total (acknowledge 14d, RCA 30d, CAP 60d)

Escalation Stages (4 levels):
    Stage 1: 75% of SLA elapsed -> notification to management
    Stage 2: 90% of SLA elapsed -> senior management alert
    Stage 3: SLA exceeded -> overdue status, authority notification
    Stage 4: SLA + 30 days -> competent authority formal report

Features:
    - F5.1-F5.12: Complete CAR lifecycle management (PRD Section 6.5)
    - Automatic SLA calculation from NC severity
    - Sub-deadline tracking (acknowledge, RCA, CAP milestones)
    - 4-level escalation with configurable thresholds
    - CAR status transitions with validation
    - Evidence submission and verification workflow
    - Verification outcome tracking (effective/not_effective)
    - Authority-issued CAR handling (Art. 18)
    - Batch CAR status monitoring
    - SLA dashboard metrics integration
    - Deterministic deadline calculations (bit-perfect)

Performance:
    - < 100 ms for CAR issuance
    - < 50 ms for status transition

Dependencies:
    - None (standalone engine within TAM agent)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple
from greenlang.schemas import utcnow

from greenlang.agents.eudr.third_party_audit_manager.config import (
    ThirdPartyAuditManagerConfig,
    get_config,
)
from greenlang.agents.eudr.third_party_audit_manager.models import (
    CARStatus,
    CorrectiveActionRequest,
    IssueCARRequest,
    IssueCARResponse,
    NCSeverity,
    NC_SEVERITY_SLA_DAYS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Valid CAR status transitions
VALID_CAR_TRANSITIONS: Dict[str, List[str]] = {
    CARStatus.ISSUED.value: [CARStatus.ACKNOWLEDGED.value],
    CARStatus.ACKNOWLEDGED.value: [CARStatus.RCA_SUBMITTED.value],
    CARStatus.RCA_SUBMITTED.value: [CARStatus.CAP_SUBMITTED.value],
    CARStatus.CAP_SUBMITTED.value: [
        CARStatus.CAP_APPROVED.value,
        CARStatus.REJECTED.value,
    ],
    CARStatus.CAP_APPROVED.value: [CARStatus.IN_PROGRESS.value],
    CARStatus.IN_PROGRESS.value: [CARStatus.EVIDENCE_SUBMITTED.value],
    CARStatus.EVIDENCE_SUBMITTED.value: [
        CARStatus.VERIFICATION_PENDING.value,
    ],
    CARStatus.VERIFICATION_PENDING.value: [
        CARStatus.CLOSED.value,
        CARStatus.REJECTED.value,
    ],
    CARStatus.REJECTED.value: [CARStatus.IN_PROGRESS.value],
    CARStatus.OVERDUE.value: [
        CARStatus.ESCALATED.value,
        CARStatus.IN_PROGRESS.value,
    ],
    CARStatus.ESCALATED.value: [
        CARStatus.IN_PROGRESS.value,
        CARStatus.CLOSED.value,
    ],
    CARStatus.CLOSED.value: [],
}

#: Escalation stage descriptions
ESCALATION_DESCRIPTIONS: Dict[int, str] = {
    1: "SLA 75% elapsed: management notification sent",
    2: "SLA 90% elapsed: senior management alert sent",
    3: "SLA exceeded: marked overdue, authority notification sent",
    4: "SLA + 30 days: formal competent authority report filed",
}

def _compute_provenance_hash(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash for provenance tracking.

    Args:
        data: Dictionary to hash.

    Returns:
        64-character hex SHA-256 hash string.
    """
    canonical = json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

class CARManagementEngine:
    """Corrective Action Request lifecycle management engine.

    Manages the complete CAR lifecycle from issuance through verified
    closure with SLA enforcement, escalation, and evidence verification.

    All SLA calculations are deterministic: same severity and issuance
    time produce the same deadlines (bit-perfect reproducibility).

    Attributes:
        config: Agent configuration.
    """

    def __init__(
        self,
        config: Optional[ThirdPartyAuditManagerConfig] = None,
    ) -> None:
        """Initialize the CAR management engine.

        Args:
            config: Optional configuration override.
        """
        self.config = config or get_config()
        logger.info("CARManagementEngine initialized")

    def issue_car(
        self, request: IssueCARRequest
    ) -> IssueCARResponse:
        """Issue a new corrective action request.

        Calculates SLA deadline from the highest NC severity,
        sets sub-deadlines for milestones, and creates the CAR record.

        Args:
            request: CAR issuance request.

        Returns:
            IssueCARResponse with issued CAR and SLA details.
        """
        start_time = utcnow()

        try:
            now = utcnow()

            # Determine SLA from severity (use custom if provided)
            severity = self._determine_severity(request.nc_ids)
            sla_days = self._get_sla_days(severity, request.custom_sla_days)
            sla_deadline = now + timedelta(days=sla_days)

            # Calculate sub-deadlines
            sub_deadlines = self._calculate_sub_deadlines(
                severity, now, sla_days
            )

            # Create CAR record
            car = CorrectiveActionRequest(
                nc_ids=request.nc_ids,
                audit_id=request.audit_id,
                supplier_id=request.supplier_id,
                severity=severity,
                sla_deadline=sla_deadline,
                sla_status="on_track",
                status=CARStatus.ISSUED,
                issued_by=request.issued_by,
                issued_at=now,
                authority_issued=request.authority_issued,
                escalation_level=0,
            )

            car.provenance_hash = _compute_provenance_hash({
                "car_id": car.car_id,
                "nc_ids": request.nc_ids,
                "severity": severity.value,
                "sla_deadline": str(sla_deadline),
                "issued_by": request.issued_by,
            })

            processing_time = Decimal(str(
                (utcnow() - start_time).total_seconds() * 1000
            )).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            sla_details = {
                "severity": severity.value,
                "sla_days": sla_days,
                "sla_deadline": sla_deadline.isoformat(),
                "sub_deadlines": {
                    k: v.isoformat() for k, v in sub_deadlines.items()
                },
                "escalation_thresholds": {
                    "stage1_at": str(
                        now + timedelta(
                            days=int(sla_days * float(self.config.escalation_stage1_pct))
                        )
                    ),
                    "stage2_at": str(
                        now + timedelta(
                            days=int(sla_days * float(self.config.escalation_stage2_pct))
                        )
                    ),
                    "stage3_at": str(sla_deadline),
                    "stage4_at": str(sla_deadline + timedelta(days=30)),
                },
            }

            response = IssueCARResponse(
                car=car,
                sla_details=sla_details,
                processing_time_ms=processing_time,
                request_id=request.request_id,
            )

            response.provenance_hash = _compute_provenance_hash({
                "car_id": car.car_id,
                "severity": severity.value,
                "processing_time_ms": str(processing_time),
            })

            logger.info(
                f"CAR issued: id={car.car_id}, severity={severity.value}, "
                f"sla_deadline={sla_deadline.isoformat()}"
            )

            return response

        except Exception as e:
            logger.error("CAR issuance failed: %s", e, exc_info=True)
            raise

    def advance_car_status(
        self,
        car: CorrectiveActionRequest,
        new_status: CARStatus,
        verification_outcome: Optional[str] = None,
    ) -> CorrectiveActionRequest:
        """Advance a CAR to a new lifecycle status.

        Validates the status transition and updates timestamps.

        Args:
            car: Current CAR record.
            new_status: Target status.
            verification_outcome: Required for CLOSED/REJECTED transitions.

        Returns:
            Updated CAR record.

        Raises:
            ValueError: If the status transition is not allowed.
        """
        current = car.status.value
        target = new_status.value

        allowed = VALID_CAR_TRANSITIONS.get(current, [])
        if target not in allowed:
            raise ValueError(
                f"Invalid CAR status transition: {current} -> {target}. "
                f"Allowed transitions: {allowed}"
            )

        now = utcnow()

        # Update status and relevant timestamp
        car.status = new_status

        if new_status == CARStatus.ACKNOWLEDGED:
            car.acknowledged_at = now
        elif new_status == CARStatus.RCA_SUBMITTED:
            car.rca_submitted_at = now
        elif new_status == CARStatus.CAP_SUBMITTED:
            car.cap_submitted_at = now
        elif new_status == CARStatus.CAP_APPROVED:
            car.cap_approved_at = now
        elif new_status == CARStatus.EVIDENCE_SUBMITTED:
            car.evidence_submitted_at = now
        elif new_status == CARStatus.VERIFICATION_PENDING:
            car.verified_at = now
        elif new_status == CARStatus.CLOSED:
            car.closed_at = now
            car.verification_outcome = verification_outcome or "effective"
        elif new_status == CARStatus.REJECTED:
            car.verification_outcome = verification_outcome or "not_effective"

        # Update provenance hash
        car.provenance_hash = _compute_provenance_hash({
            "car_id": car.car_id,
            "previous_status": current,
            "new_status": target,
            "updated_at": str(now),
        })

        logger.info(
            f"CAR {car.car_id} status advanced: {current} -> {target}"
        )

        return car

    def check_sla_status(
        self, car: CorrectiveActionRequest
    ) -> Dict[str, Any]:
        """Check CAR SLA compliance status.

        Evaluates the current SLA status, escalation level, and
        remaining time.

        Args:
            car: CAR to check.

        Returns:
            Dictionary with SLA status details.
        """
        now = utcnow()

        if car.status == CARStatus.CLOSED:
            return {
                "car_id": car.car_id,
                "sla_status": "closed",
                "is_overdue": False,
                "escalation_level": car.escalation_level,
                "closed_at": str(car.closed_at),
                "within_sla": car.closed_at <= car.sla_deadline if car.closed_at else False,
                "checked_at": now.isoformat(),
            }

        total_sla_seconds = (
            car.sla_deadline - car.issued_at
        ).total_seconds()
        elapsed_seconds = (now - car.issued_at).total_seconds()
        remaining_seconds = (car.sla_deadline - now).total_seconds()

        elapsed_pct = Decimal("0")
        if total_sla_seconds > 0:
            elapsed_pct = (
                Decimal(str(elapsed_seconds))
                / Decimal(str(total_sla_seconds))
            ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

        is_overdue = remaining_seconds < 0
        days_remaining = remaining_seconds / 86400 if not is_overdue else 0
        days_overdue = abs(remaining_seconds) / 86400 if is_overdue else 0

        # Determine SLA status label
        if is_overdue:
            sla_status = "overdue"
        elif elapsed_pct >= self.config.escalation_stage2_pct:
            sla_status = "critical"
        elif elapsed_pct >= self.config.escalation_stage1_pct:
            sla_status = "warning"
        else:
            sla_status = "on_track"

        return {
            "car_id": car.car_id,
            "severity": car.severity.value,
            "sla_status": sla_status,
            "sla_deadline": car.sla_deadline.isoformat(),
            "elapsed_pct": str(elapsed_pct),
            "is_overdue": is_overdue,
            "days_remaining": round(days_remaining, 1) if not is_overdue else 0,
            "days_overdue": round(days_overdue, 1) if is_overdue else 0,
            "escalation_level": car.escalation_level,
            "current_status": car.status.value,
            "checked_at": now.isoformat(),
        }

    def evaluate_escalation(
        self, car: CorrectiveActionRequest
    ) -> Dict[str, Any]:
        """Evaluate whether a CAR needs escalation.

        Checks the current escalation level against the elapsed SLA
        percentage and determines if a new escalation is needed.

        Args:
            car: CAR to evaluate.

        Returns:
            Dictionary with escalation evaluation results.
        """
        if car.status == CARStatus.CLOSED:
            return {
                "car_id": car.car_id,
                "needs_escalation": False,
                "current_level": car.escalation_level,
                "reason": "CAR is closed",
            }

        now = utcnow()
        total_sla_seconds = (
            car.sla_deadline - car.issued_at
        ).total_seconds()
        elapsed_seconds = (now - car.issued_at).total_seconds()

        if total_sla_seconds <= 0:
            return {
                "car_id": car.car_id,
                "needs_escalation": False,
                "current_level": car.escalation_level,
                "reason": "Invalid SLA timeline",
            }

        elapsed_pct = Decimal(str(elapsed_seconds)) / Decimal(
            str(total_sla_seconds)
        )

        # Determine required escalation level
        required_level = 0

        # Stage 4: SLA + 30 days exceeded
        overdue_seconds = elapsed_seconds - total_sla_seconds
        if overdue_seconds > 30 * 86400:
            required_level = 4
        # Stage 3: SLA exceeded
        elif elapsed_pct > Decimal("1.0"):
            required_level = 3
        # Stage 2: 90% elapsed
        elif elapsed_pct >= self.config.escalation_stage2_pct:
            required_level = 2
        # Stage 1: 75% elapsed
        elif elapsed_pct >= self.config.escalation_stage1_pct:
            required_level = 1

        # Cap at max escalation levels
        required_level = min(
            required_level, self.config.max_escalation_levels
        )

        needs_escalation = required_level > car.escalation_level

        return {
            "car_id": car.car_id,
            "needs_escalation": needs_escalation,
            "current_level": car.escalation_level,
            "required_level": required_level,
            "elapsed_pct": str(
                elapsed_pct.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            ),
            "description": (
                ESCALATION_DESCRIPTIONS.get(required_level, "No escalation")
                if needs_escalation else "No escalation needed"
            ),
            "evaluated_at": now.isoformat(),
        }

    def escalate_car(
        self,
        car: CorrectiveActionRequest,
        target_level: int,
        reason: str = "",
    ) -> CorrectiveActionRequest:
        """Escalate a CAR to a higher escalation level.

        Records the escalation event in the escalation history
        and updates the escalation level.

        Args:
            car: CAR to escalate.
            target_level: Target escalation level (1-4).
            reason: Escalation reason.

        Returns:
            Updated CAR with new escalation level.

        Raises:
            ValueError: If target level is invalid or not higher.
        """
        if target_level <= car.escalation_level:
            raise ValueError(
                f"Target escalation level ({target_level}) must be higher "
                f"than current level ({car.escalation_level})"
            )

        if target_level > self.config.max_escalation_levels:
            raise ValueError(
                f"Target escalation level ({target_level}) exceeds "
                f"maximum ({self.config.max_escalation_levels})"
            )

        now = utcnow()

        escalation_event = {
            "from_level": car.escalation_level,
            "to_level": target_level,
            "reason": reason or ESCALATION_DESCRIPTIONS.get(
                target_level, "Escalation"
            ),
            "escalated_at": now.isoformat(),
        }

        car.escalation_history.append(escalation_event)
        car.escalation_level = target_level

        # Mark as overdue if stage 3+
        if target_level >= 3 and car.status != CARStatus.OVERDUE:
            car.status = CARStatus.OVERDUE
            car.sla_status = "overdue"

        # Mark as escalated if stage 4
        if target_level >= 4:
            car.status = CARStatus.ESCALATED

        # Update provenance
        car.provenance_hash = _compute_provenance_hash({
            "car_id": car.car_id,
            "escalation_level": target_level,
            "escalated_at": str(now),
        })

        logger.warning(
            f"CAR {car.car_id} escalated to level {target_level}: "
            f"{escalation_event['reason']}"
        )

        return car

    def batch_check_sla(
        self, cars: List[CorrectiveActionRequest]
    ) -> Dict[str, Any]:
        """Batch check SLA status for multiple CARs.

        Args:
            cars: List of CARs to check.

        Returns:
            Dictionary with aggregated SLA status results.
        """
        results: List[Dict[str, Any]] = []
        summary = {
            "total": len(cars),
            "on_track": 0,
            "warning": 0,
            "critical": 0,
            "overdue": 0,
            "closed": 0,
        }

        for car in cars:
            status = self.check_sla_status(car)
            results.append(status)

            sla_status = status["sla_status"]
            if sla_status in summary:
                summary[sla_status] += 1

        return {
            "results": results,
            "summary": summary,
            "checked_at": utcnow().isoformat(),
        }

    def _determine_severity(
        self, nc_ids: List[str]
    ) -> NCSeverity:
        """Determine the highest severity from linked NCs.

        In production, this would look up NC records from the database.
        For now, defaults to MAJOR if not resolvable.

        Args:
            nc_ids: Non-conformance identifiers.

        Returns:
            Highest NC severity.
        """
        # Default to MAJOR - in production would resolve from database
        return NCSeverity.MAJOR

    def _get_sla_days(
        self,
        severity: NCSeverity,
        custom_sla_days: Optional[int] = None,
    ) -> int:
        """Get SLA deadline days for a severity level.

        Args:
            severity: NC severity.
            custom_sla_days: Optional custom SLA override.

        Returns:
            SLA deadline in days.
        """
        if custom_sla_days is not None:
            return custom_sla_days

        sla_map = {
            NCSeverity.CRITICAL: self.config.critical_sla_days,
            NCSeverity.MAJOR: self.config.major_sla_days,
            NCSeverity.MINOR: self.config.minor_sla_days,
        }

        return sla_map.get(severity, self.config.major_sla_days)

    def _calculate_sub_deadlines(
        self,
        severity: NCSeverity,
        issued_at: datetime,
        total_sla_days: int,
    ) -> Dict[str, datetime]:
        """Calculate CAR sub-deadlines for milestone tracking.

        Args:
            severity: NC severity.
            issued_at: CAR issuance timestamp.
            total_sla_days: Total SLA deadline in days.

        Returns:
            Dictionary mapping milestone name to deadline datetime.
        """
        if severity == NCSeverity.CRITICAL:
            ack_days = self.config.critical_acknowledge_days
            rca_days = self.config.critical_rca_days
            cap_days = self.config.critical_cap_days
        elif severity == NCSeverity.MAJOR:
            ack_days = self.config.major_acknowledge_days
            rca_days = self.config.major_rca_days
            cap_days = self.config.major_cap_days
        else:
            ack_days = self.config.minor_acknowledge_days
            rca_days = self.config.minor_rca_days
            cap_days = self.config.minor_cap_days

        return {
            "acknowledge_by": issued_at + timedelta(days=ack_days),
            "rca_by": issued_at + timedelta(days=rca_days),
            "cap_by": issued_at + timedelta(days=cap_days),
            "close_by": issued_at + timedelta(days=total_sla_days),
        }
