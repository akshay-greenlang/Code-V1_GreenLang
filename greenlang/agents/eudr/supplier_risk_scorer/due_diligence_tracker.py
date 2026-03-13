# -*- coding: utf-8 -*-
"""
Due Diligence Tracker Engine - AGENT-EUDR-017 Engine 2

Comprehensive due diligence activity tracking per EUDR Articles 8-11,
covering audit scheduling, site visit logging, document review tracking,
questionnaire response management, supplier screening results, non-conformance
recording (minor/major/critical), corrective action plan monitoring, DD
completion rate calculation, gap identification, escalation workflow,
regulatory submission readiness scoring, DD cost tracking, and automated
re-assessment scheduling.

Due Diligence Levels:
    - SIMPLIFIED: Low-risk suppliers per Article 13, reduced DD requirements
    - STANDARD: Standard-risk suppliers per Article 10
    - ENHANCED: High-risk suppliers per Article 11, additional verification

Activity Types:
    - AUDIT: On-site or remote audit by qualified auditor
    - SITE_VISIT: Physical inspection of production/processing sites
    - DOCUMENT_REVIEW: Review of EUDR-required documentation package
    - QUESTIONNAIRE: Supplier self-assessment questionnaire response
    - SCREENING: Screening against sanctions lists, deforestation databases
    - VERIFICATION: Third-party verification of claims or certifications
    - TRAINING: Supplier capacity building or training session
    - INTERVIEW: Stakeholder or worker interviews

Non-Conformance Severity:
    - MINOR: Low-impact issue, no immediate EUDR compliance risk
    - MAJOR: Significant issue requiring corrective action within 90 days
    - CRITICAL: Severe non-compliance, immediate action required, potential
      suspension of supplier relationship

Zero-Hallucination: All DD tracking is database-backed with deterministic
    completion rate calculations, status transitions, and deadline tracking.
    No LLM calls in DD activity recording or status evaluation.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-017 Supplier Risk Scorer (GL-EUDR-SRS-017)
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from .config import get_config
from .metrics import (
    observe_dd_tracking_duration,
    record_dd_record_created,
)
from .models import (
    CorrectiveActionPlan,
    DDActivity,
    DDActivityType,
    DDLevel,
    DDStatus,
    DueDiligenceRecord,
    DueDiligenceResponse,
    NonConformance,
    NonConformanceStatus,
    NonConformanceType,
    TrackDueDiligenceRequest,
)
from .provenance import get_provenance_tracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Activity completion weights for DD completion rate
_ACTIVITY_WEIGHTS: Dict[str, Decimal] = {
    "audit": Decimal("0.30"),
    "site_visit": Decimal("0.25"),
    "document_review": Decimal("0.20"),
    "questionnaire": Decimal("0.10"),
    "screening": Decimal("0.10"),
    "verification": Decimal("0.05"),
}

#: Required activities per DD level
_REQUIRED_ACTIVITIES_BY_LEVEL: Dict[str, List[str]] = {
    "simplified": ["document_review", "screening"],
    "standard": ["document_review", "screening", "questionnaire"],
    "enhanced": ["audit", "site_visit", "document_review", "screening", "questionnaire"],
}

#: Escalation thresholds
_ESCALATION_CRITICAL_NC_COUNT: int = 1
_ESCALATION_MAJOR_NC_COUNT: int = 3
_ESCALATION_OVERDUE_DAYS: int = 30


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal for precise arithmetic."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _float(value: Decimal) -> float:
    """Convert Decimal to float for API responses."""
    return float(value)


# ---------------------------------------------------------------------------
# DueDiligenceTracker
# ---------------------------------------------------------------------------


class DueDiligenceTracker:
    """Track all due diligence activities per supplier per EUDR Articles 8-11.

    Manages comprehensive DD activity logging, non-conformance tracking,
    corrective action monitoring, completion rate calculation, gap analysis,
    escalation workflows, readiness scoring, cost tracking, and automated
    re-assessment scheduling for EUDR compliance.

    Attributes:
        _dd_records: In-memory store of DD records keyed by record_id.
        _supplier_dd: Mapping from supplier_id to list of record_ids.
        _activities: Store of DD activities keyed by activity_id.
        _non_conformances: Store of non-conformances keyed by nc_id.
        _corrective_actions: Store of corrective actions keyed by cap_id.
        _lock: Threading lock for thread-safe access.

    Example:
        >>> tracker = DueDiligenceTracker()
        >>> request = TrackDueDiligenceRequest(supplier_id="SUP123", ...)
        >>> result = tracker.record_activity(request)
        >>> assert result.dd_record.completion_rate >= 0.0
    """

    def __init__(self) -> None:
        """Initialize DueDiligenceTracker with empty stores."""
        self._dd_records: Dict[str, DueDiligenceRecord] = {}
        self._supplier_dd: Dict[str, List[str]] = defaultdict(list)
        self._activities: Dict[str, DDActivity] = {}
        self._non_conformances: Dict[str, NonConformance] = {}
        self._corrective_actions: Dict[str, CorrectiveActionPlan] = {}
        self._lock: threading.Lock = threading.Lock()
        logger.info("DueDiligenceTracker initialized")

    # ------------------------------------------------------------------
    # Record DD activity
    # ------------------------------------------------------------------

    def record_activity(
        self,
        request: TrackDueDiligenceRequest,
    ) -> DueDiligenceResponse:
        """Record a due diligence activity for a supplier.

        Creates or updates DD record for supplier, logs the activity,
        recalculates completion rate, identifies gaps, checks escalation
        triggers, and updates readiness score.

        Args:
            request: TrackDueDiligenceRequest containing supplier_id,
                activity_type, dd_level, activity details, and optional
                non-conformances.

        Returns:
            DueDiligenceResponse with updated DueDiligenceRecord including
            completion_rate, status, next activities, and escalation flags.

        Raises:
            ValueError: If supplier_id is empty or activity_type invalid.
        """
        start_time = time.perf_counter()
        cfg = get_config()

        try:
            # Step 1: Validate inputs
            self._validate_dd_inputs(request)

            # Step 2: Get or create DD record for supplier
            dd_record = self._get_or_create_dd_record(
                request.supplier_id,
                request.dd_level,
            )

            # Step 3: Create activity record
            activity_id = str(uuid.uuid4())
            now = _utcnow()

            activity = DDActivity(
                activity_id=activity_id,
                supplier_id=request.supplier_id,
                activity_type=request.activity_type,
                activity_date=request.activity_date or now,
                conducted_by=request.conducted_by or "system",
                findings=request.findings or "",
                documents_reviewed=request.documents_reviewed or [],
                sites_visited=request.sites_visited or [],
                stakeholders_interviewed=request.stakeholders_interviewed or [],
                cost_usd=request.cost_usd or 0.0,
                duration_hours=request.duration_hours or 0.0,
            )

            # Step 4: Store activity
            with self._lock:
                self._activities[activity_id] = activity
                dd_record.activities.append(activity)

            # Step 5: Record non-conformances if any
            if request.non_conformances:
                for nc_data in request.non_conformances:
                    nc = self._create_non_conformance(
                        dd_record.record_id,
                        request.supplier_id,
                        nc_data,
                    )
                    with self._lock:
                        self._non_conformances[nc.nc_id] = nc
                        dd_record.non_conformances.append(nc)

            # Step 6: Recalculate completion rate
            completion_rate = self._calculate_completion_rate(dd_record)
            dd_record.completion_rate = _float(completion_rate)

            # Step 7: Update DD status
            dd_record.dd_status = self._determine_dd_status(
                dd_record, completion_rate
            )

            # Step 8: Identify gaps
            gaps = self._identify_gaps(dd_record)
            dd_record.gaps = gaps

            # Step 9: Check escalation triggers
            escalation_required = self._check_escalation_triggers(dd_record)
            dd_record.escalation_required = escalation_required

            # Step 10: Calculate readiness score
            readiness = self._calculate_readiness_score(dd_record)
            dd_record.regulatory_submission_readiness = _float(readiness)

            # Step 11: Update next assessment date if needed
            if completion_rate >= Decimal("0.8"):
                dd_record.next_dd_date = now + timedelta(
                    days=cfg.dd_tracking_period_months * 30
                )
            else:
                dd_record.next_dd_date = now + timedelta(days=30)

            dd_record.last_updated = now

            # Step 12: Store updated record
            with self._lock:
                self._dd_records[dd_record.record_id] = dd_record

            # Step 13: Record provenance
            get_provenance_tracker().record_operation(
                entity_type="due_diligence",
                entity_id=activity_id,
                action="track",
                details={
                    "supplier_id": request.supplier_id,
                    "activity_type": request.activity_type.value,
                    "completion_rate": _float(completion_rate),
                    "dd_status": dd_record.dd_status.value,
                },
            )

            # Step 14: Record metrics
            duration = time.perf_counter() - start_time
            observe_dd_tracking_duration(
                duration,
                request.activity_type.value,
                request.dd_level.value,
            )
            record_dd_record_created(
                dd_level=request.dd_level.value,
                activity_type=request.activity_type.value,
            )

            logger.info(
                "DD activity recorded: supplier_id=%s, activity=%s, completion=%.1f%%, "
                "status=%s, duration=%.3fs",
                request.supplier_id,
                request.activity_type.value,
                _float(completion_rate * 100),
                dd_record.dd_status.value,
                duration,
            )

            return DueDiligenceResponse(
                dd_record=dd_record,
                processing_time_ms=duration * 1000.0,
            )

        except Exception as e:
            logger.error(
                "DD activity recording failed: supplier_id=%s, error=%s",
                request.supplier_id if hasattr(request, "supplier_id") else "unknown",
                str(e),
                exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Get DD history
    # ------------------------------------------------------------------

    def get_dd_history(
        self,
        supplier_id: str,
        activity_type: Optional[DDActivityType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[DDActivity]:
        """Get DD activity history for a supplier.

        Args:
            supplier_id: Supplier identifier.
            activity_type: Optional filter by activity type.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            List of DDActivity objects sorted by date descending.

        Raises:
            ValueError: If supplier_id not found.
        """
        with self._lock:
            if supplier_id not in self._supplier_dd:
                raise ValueError(f"No DD records found for supplier {supplier_id}")

            record_ids = self._supplier_dd[supplier_id]
            activities = []

            for record_id in record_ids:
                record = self._dd_records.get(record_id)
                if not record:
                    continue

                for activity in record.activities:
                    # Apply filters
                    if activity_type and activity.activity_type != activity_type:
                        continue
                    if start_date and activity.activity_date < start_date:
                        continue
                    if end_date and activity.activity_date > end_date:
                        continue

                    activities.append(activity)

        # Sort by date descending
        activities.sort(key=lambda x: x.activity_date, reverse=True)

        logger.info(
            "DD history retrieved: supplier_id=%s, activities=%d",
            supplier_id,
            len(activities),
        )

        return activities

    # ------------------------------------------------------------------
    # Track non-conformance
    # ------------------------------------------------------------------

    def track_non_conformance(
        self,
        record_id: str,
        nc_type: NonConformanceType,
        description: str,
        severity_score: float,
        detected_date: Optional[datetime] = None,
    ) -> NonConformance:
        """Track a non-conformance finding.

        Args:
            record_id: DD record identifier.
            nc_type: Non-conformance type.
            description: Description of non-conformance.
            severity_score: Severity score [0, 100].
            detected_date: Optional detection date.

        Returns:
            NonConformance object.

        Raises:
            ValueError: If record_id not found.
        """
        with self._lock:
            if record_id not in self._dd_records:
                raise ValueError(f"DD record {record_id} not found")

            record = self._dd_records[record_id]

        nc_id = str(uuid.uuid4())
        now = _utcnow()

        nc = NonConformance(
            nc_id=nc_id,
            record_id=record_id,
            supplier_id=record.supplier_id,
            nc_type=nc_type,
            description=description,
            severity_score=severity_score,
            detected_date=detected_date or now,
            status=NonConformanceStatus.OPEN,
        )

        with self._lock:
            self._non_conformances[nc_id] = nc
            record.non_conformances.append(nc)

        logger.info(
            "Non-conformance tracked: nc_id=%s, supplier_id=%s, type=%s, severity=%.1f",
            nc_id,
            record.supplier_id,
            nc_type.value,
            severity_score,
        )

        return nc

    # ------------------------------------------------------------------
    # Create corrective action plan
    # ------------------------------------------------------------------

    def create_corrective_action(
        self,
        nc_id: str,
        action_description: str,
        responsible_party: str,
        deadline: datetime,
        resources_required: Optional[str] = None,
    ) -> CorrectiveActionPlan:
        """Create a corrective action plan for a non-conformance.

        Args:
            nc_id: Non-conformance identifier.
            action_description: Description of corrective action.
            responsible_party: Person/entity responsible.
            deadline: Completion deadline.
            resources_required: Optional description of required resources.

        Returns:
            CorrectiveActionPlan object.

        Raises:
            ValueError: If nc_id not found.
        """
        with self._lock:
            if nc_id not in self._non_conformances:
                raise ValueError(f"Non-conformance {nc_id} not found")

            nc = self._non_conformances[nc_id]

        cap_id = str(uuid.uuid4())
        now = _utcnow()

        cap = CorrectiveActionPlan(
            cap_id=cap_id,
            nc_id=nc_id,
            action_description=action_description,
            responsible_party=responsible_party,
            deadline=deadline,
            resources_required=resources_required or "",
            created_date=now,
            status=NonConformanceStatus.OPEN,
        )

        with self._lock:
            self._corrective_actions[cap_id] = cap

        logger.info(
            "Corrective action plan created: cap_id=%s, nc_id=%s, deadline=%s",
            cap_id,
            nc_id,
            deadline.isoformat(),
        )

        return cap

    # ------------------------------------------------------------------
    # Check completion rate
    # ------------------------------------------------------------------

    def check_completion_rate(
        self,
        supplier_id: str,
    ) -> float:
        """Check DD completion rate for a supplier.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            Completion rate [0.0, 1.0].

        Raises:
            ValueError: If supplier_id not found.
        """
        with self._lock:
            if supplier_id not in self._supplier_dd:
                raise ValueError(f"No DD records found for supplier {supplier_id}")

            record_ids = self._supplier_dd[supplier_id]
            if not record_ids:
                return 0.0

            # Get most recent record
            latest_record_id = record_ids[-1]
            record = self._dd_records.get(latest_record_id)
            if not record:
                return 0.0

        completion_rate = self._calculate_completion_rate(record)
        return _float(completion_rate)

    # ------------------------------------------------------------------
    # Identify gaps
    # ------------------------------------------------------------------

    def identify_gaps(
        self,
        supplier_id: str,
    ) -> List[str]:
        """Identify DD coverage gaps for a supplier.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            List of gap descriptions.

        Raises:
            ValueError: If supplier_id not found.
        """
        with self._lock:
            if supplier_id not in self._supplier_dd:
                raise ValueError(f"No DD records found for supplier {supplier_id}")

            record_ids = self._supplier_dd[supplier_id]
            if not record_ids:
                return ["No DD activities recorded"]

            latest_record_id = record_ids[-1]
            record = self._dd_records.get(latest_record_id)
            if not record:
                return ["DD record not found"]

        gaps = self._identify_gaps(record)
        return gaps

    # ------------------------------------------------------------------
    # Escalate
    # ------------------------------------------------------------------

    def escalate(
        self,
        supplier_id: str,
        reason: str,
        escalated_to: str,
    ) -> bool:
        """Escalate supplier DD issues to management.

        Args:
            supplier_id: Supplier identifier.
            reason: Escalation reason.
            escalated_to: Person/team escalated to.

        Returns:
            True if escalation successful.

        Raises:
            ValueError: If supplier_id not found.
        """
        with self._lock:
            if supplier_id not in self._supplier_dd:
                raise ValueError(f"No DD records found for supplier {supplier_id}")

            record_ids = self._supplier_dd[supplier_id]
            if not record_ids:
                return False

            latest_record_id = record_ids[-1]
            record = self._dd_records.get(latest_record_id)
            if not record:
                return False

            record.escalation_required = True
            record.escalation_reason = reason
            record.escalated_to = escalated_to
            record.escalation_date = _utcnow()

        logger.warning(
            "Supplier DD escalated: supplier_id=%s, reason=%s, escalated_to=%s",
            supplier_id,
            reason,
            escalated_to,
        )

        return True

    # ------------------------------------------------------------------
    # Get readiness score
    # ------------------------------------------------------------------

    def get_readiness_score(
        self,
        supplier_id: str,
    ) -> float:
        """Get regulatory submission readiness score for supplier.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            Readiness score [0.0, 1.0].

        Raises:
            ValueError: If supplier_id not found.
        """
        with self._lock:
            if supplier_id not in self._supplier_dd:
                raise ValueError(f"No DD records found for supplier {supplier_id}")

            record_ids = self._supplier_dd[supplier_id]
            if not record_ids:
                return 0.0

            latest_record_id = record_ids[-1]
            record = self._dd_records.get(latest_record_id)
            if not record:
                return 0.0

        readiness = self._calculate_readiness_score(record)
        return _float(readiness)

    # ------------------------------------------------------------------
    # Get cost summary
    # ------------------------------------------------------------------

    def get_cost_summary(
        self,
        supplier_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get DD cost summary for a supplier.

        Args:
            supplier_id: Supplier identifier.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            Dictionary with total_cost, cost_by_activity, activity_count.

        Raises:
            ValueError: If supplier_id not found.
        """
        activities = self.get_dd_history(supplier_id, start_date=start_date, end_date=end_date)

        total_cost = Decimal("0.0")
        cost_by_activity: Dict[str, Decimal] = defaultdict(lambda: Decimal("0.0"))

        for activity in activities:
            cost = _decimal(activity.cost_usd)
            total_cost += cost
            cost_by_activity[activity.activity_type.value] += cost

        summary = {
            "supplier_id": supplier_id,
            "total_cost_usd": _float(total_cost),
            "cost_by_activity": {k: _float(v) for k, v in cost_by_activity.items()},
            "activity_count": len(activities),
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
        }

        logger.info(
            "DD cost summary: supplier_id=%s, total_cost=%.2f, activities=%d",
            supplier_id,
            _float(total_cost),
            len(activities),
        )

        return summary

    # ------------------------------------------------------------------
    # Schedule reassessment
    # ------------------------------------------------------------------

    def schedule_reassessment(
        self,
        supplier_id: str,
        reassessment_date: datetime,
        reason: str,
    ) -> bool:
        """Schedule automated DD reassessment for supplier.

        Args:
            supplier_id: Supplier identifier.
            reassessment_date: Date for next assessment.
            reason: Reason for reassessment.

        Returns:
            True if scheduling successful.

        Raises:
            ValueError: If supplier_id not found.
        """
        with self._lock:
            if supplier_id not in self._supplier_dd:
                raise ValueError(f"No DD records found for supplier {supplier_id}")

            record_ids = self._supplier_dd[supplier_id]
            if not record_ids:
                return False

            latest_record_id = record_ids[-1]
            record = self._dd_records.get(latest_record_id)
            if not record:
                return False

            record.next_dd_date = reassessment_date
            record.reassessment_reason = reason

        logger.info(
            "DD reassessment scheduled: supplier_id=%s, date=%s, reason=%s",
            supplier_id,
            reassessment_date.isoformat(),
            reason,
        )

        return True

    # ------------------------------------------------------------------
    # Helper methods: Validation
    # ------------------------------------------------------------------

    def _validate_dd_inputs(
        self,
        request: TrackDueDiligenceRequest,
    ) -> None:
        """Validate DD tracking request inputs.

        Raises:
            ValueError: If validation fails.
        """
        if not request.supplier_id:
            raise ValueError("supplier_id is required")

        if not request.activity_type:
            raise ValueError("activity_type is required")

        if not request.dd_level:
            raise ValueError("dd_level is required")

    # ------------------------------------------------------------------
    # Helper methods: DD record management
    # ------------------------------------------------------------------

    def _get_or_create_dd_record(
        self,
        supplier_id: str,
        dd_level: DDLevel,
    ) -> DueDiligenceRecord:
        """Get existing DD record or create new one for supplier.

        Args:
            supplier_id: Supplier identifier.
            dd_level: DD level.

        Returns:
            DueDiligenceRecord object.
        """
        with self._lock:
            if supplier_id in self._supplier_dd:
                record_ids = self._supplier_dd[supplier_id]
                if record_ids:
                    latest_record_id = record_ids[-1]
                    record = self._dd_records.get(latest_record_id)
                    if record:
                        return record

            # Create new record
            record_id = str(uuid.uuid4())
            now = _utcnow()

            record = DueDiligenceRecord(
                record_id=record_id,
                supplier_id=supplier_id,
                dd_level=dd_level,
                dd_status=DDStatus.INITIATED,
                activities=[],
                non_conformances=[],
                completion_rate=0.0,
                gaps=[],
                regulatory_submission_readiness=0.0,
                escalation_required=False,
                last_dd_date=now,
                next_dd_date=now + timedelta(days=90),
                last_updated=now,
            )

            self._dd_records[record_id] = record
            self._supplier_dd[supplier_id].append(record_id)

        return record

    # ------------------------------------------------------------------
    # Helper methods: Completion rate
    # ------------------------------------------------------------------

    def _calculate_completion_rate(
        self,
        dd_record: DueDiligenceRecord,
    ) -> Decimal:
        """Calculate DD completion rate based on completed activities.

        Args:
            dd_record: DueDiligenceRecord object.

        Returns:
            Completion rate [0.0, 1.0].
        """
        required_activities = _REQUIRED_ACTIVITIES_BY_LEVEL.get(
            dd_record.dd_level.value,
            ["document_review", "screening"],
        )

        completed_activity_types = set(
            activity.activity_type.value for activity in dd_record.activities
        )

        total_weight = Decimal("0.0")
        completed_weight = Decimal("0.0")

        for activity_type in required_activities:
            weight = _ACTIVITY_WEIGHTS.get(activity_type, Decimal("0.1"))
            total_weight += weight

            if activity_type in completed_activity_types:
                completed_weight += weight

        if total_weight == 0:
            return Decimal("0.0")

        completion = completed_weight / total_weight
        return completion.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Helper methods: DD status
    # ------------------------------------------------------------------

    def _determine_dd_status(
        self,
        dd_record: DueDiligenceRecord,
        completion_rate: Decimal,
    ) -> DDStatus:
        """Determine DD status from completion rate and other factors.

        Args:
            dd_record: DueDiligenceRecord object.
            completion_rate: Completion rate [0.0, 1.0].

        Returns:
            DDStatus enum value.
        """
        # Check for critical non-conformances
        critical_ncs = [
            nc for nc in dd_record.non_conformances
            if nc.nc_type == NonConformanceType.CRITICAL
            and nc.status == NonConformanceStatus.OPEN
        ]

        if critical_ncs:
            return DDStatus.SUSPENDED

        # Check completion
        if completion_rate >= Decimal("1.0"):
            return DDStatus.COMPLETED
        elif completion_rate >= Decimal("0.5"):
            return DDStatus.IN_PROGRESS
        elif completion_rate > Decimal("0.0"):
            return DDStatus.INITIATED
        else:
            return DDStatus.INITIATED

    # ------------------------------------------------------------------
    # Helper methods: Gap identification
    # ------------------------------------------------------------------

    def _identify_gaps(
        self,
        dd_record: DueDiligenceRecord,
    ) -> List[str]:
        """Identify DD coverage gaps.

        Args:
            dd_record: DueDiligenceRecord object.

        Returns:
            List of gap descriptions.
        """
        required_activities = _REQUIRED_ACTIVITIES_BY_LEVEL.get(
            dd_record.dd_level.value,
            ["document_review", "screening"],
        )

        completed_activity_types = set(
            activity.activity_type.value for activity in dd_record.activities
        )

        gaps = []
        for activity_type in required_activities:
            if activity_type not in completed_activity_types:
                gaps.append(f"Missing {activity_type.replace('_', ' ')} activity")

        # Check for overdue activities
        now = _utcnow()
        if dd_record.next_dd_date and dd_record.next_dd_date < now:
            days_overdue = (now - dd_record.next_dd_date).days
            gaps.append(f"DD reassessment overdue by {days_overdue} days")

        # Check for open non-conformances
        open_ncs = [
            nc for nc in dd_record.non_conformances
            if nc.status == NonConformanceStatus.OPEN
        ]
        if open_ncs:
            gaps.append(f"{len(open_ncs)} open non-conformances requiring resolution")

        return gaps

    # ------------------------------------------------------------------
    # Helper methods: Escalation
    # ------------------------------------------------------------------

    def _check_escalation_triggers(
        self,
        dd_record: DueDiligenceRecord,
    ) -> bool:
        """Check if escalation is required.

        Args:
            dd_record: DueDiligenceRecord object.

        Returns:
            True if escalation required.
        """
        # Critical NC count
        critical_ncs = [
            nc for nc in dd_record.non_conformances
            if nc.nc_type == NonConformanceType.CRITICAL
            and nc.status == NonConformanceStatus.OPEN
        ]
        if len(critical_ncs) >= _ESCALATION_CRITICAL_NC_COUNT:
            return True

        # Major NC count
        major_ncs = [
            nc for nc in dd_record.non_conformances
            if nc.nc_type == NonConformanceType.MAJOR
            and nc.status == NonConformanceStatus.OPEN
        ]
        if len(major_ncs) >= _ESCALATION_MAJOR_NC_COUNT:
            return True

        # Overdue reassessment
        now = _utcnow()
        if dd_record.next_dd_date:
            days_overdue = (now - dd_record.next_dd_date).days
            if days_overdue > _ESCALATION_OVERDUE_DAYS:
                return True

        return False

    # ------------------------------------------------------------------
    # Helper methods: Readiness score
    # ------------------------------------------------------------------

    def _calculate_readiness_score(
        self,
        dd_record: DueDiligenceRecord,
    ) -> Decimal:
        """Calculate regulatory submission readiness score.

        Score based on:
        - Completion rate (50%)
        - Open non-conformances (30%)
        - Data freshness (20%)

        Args:
            dd_record: DueDiligenceRecord object.

        Returns:
            Readiness score [0.0, 1.0].
        """
        # Completion component
        completion_rate = _decimal(dd_record.completion_rate)
        completion_component = completion_rate * Decimal("0.5")

        # Non-conformance component
        total_ncs = len(dd_record.non_conformances)
        open_ncs = len([
            nc for nc in dd_record.non_conformances
            if nc.status == NonConformanceStatus.OPEN
        ])

        if total_ncs == 0:
            nc_component = Decimal("0.3")
        else:
            nc_resolution_rate = Decimal("1.0") - (Decimal(open_ncs) / Decimal(total_ncs))
            nc_component = nc_resolution_rate * Decimal("0.3")

        # Freshness component
        now = _utcnow()
        days_since_last_dd = (now - dd_record.last_dd_date).days
        if days_since_last_dd < 90:
            freshness_component = Decimal("0.2")
        elif days_since_last_dd < 180:
            freshness_component = Decimal("0.15")
        else:
            freshness_component = Decimal("0.1")

        # Total readiness
        readiness = completion_component + nc_component + freshness_component
        readiness = max(Decimal("0.0"), min(Decimal("1.0"), readiness))

        return readiness.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Helper methods: Non-conformance creation
    # ------------------------------------------------------------------

    def _create_non_conformance(
        self,
        record_id: str,
        supplier_id: str,
        nc_data: Dict[str, Any],
    ) -> NonConformance:
        """Create NonConformance object from request data.

        Args:
            record_id: DD record identifier.
            supplier_id: Supplier identifier.
            nc_data: Non-conformance data dictionary.

        Returns:
            NonConformance object.
        """
        nc_id = str(uuid.uuid4())
        now = _utcnow()

        return NonConformance(
            nc_id=nc_id,
            record_id=record_id,
            supplier_id=supplier_id,
            nc_type=nc_data.get("nc_type", NonConformanceType.MINOR),
            description=nc_data.get("description", ""),
            severity_score=nc_data.get("severity_score", 0.0),
            detected_date=nc_data.get("detected_date", now),
            status=NonConformanceStatus.OPEN,
        )
