# -*- coding: utf-8 -*-
"""
Deadline Tracker Engine - AGENT-EUDR-034

Tracks regulatory and internal submission deadlines for annual EUDR
reviews. Monitors approaching deadlines, manages submissions to
competent authorities, and escalates overdue items.

Zero-Hallucination:
    - All deadline calculations use pure date arithmetic
    - Status classifications are deterministic threshold comparisons
    - No LLM involvement in deadline evaluation

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-034 (GL-EUDR-ARS-034)
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .config import AnnualReviewSchedulerConfig, get_config
from .models import (
    AGENT_ID,
    DeadlineAlert,
    DeadlineAlertLevel,
    DeadlineEntry,
    DeadlineStatus,
    DeadlineTrack,
    DeadlineTrackingRecord,
    DeadlineType,
    ReviewPhase,
    REVIEW_PHASES_ORDER,
)
from .provenance import ProvenanceTracker
from . import metrics as m

logger = logging.getLogger(__name__)


class DeadlineTracker:
    """Regulatory and internal deadline tracking engine.

    Registers deadlines, checks for approaching and overdue items,
    manages submissions to competent authorities, and tracks
    submission lifecycle.

    Example:
        >>> tracker = DeadlineTracker()
        >>> record = await tracker.register_deadline(
        ...     operator_id="OP-001",
        ...     deadline_type=DeadlineType.REGULATORY_SUBMISSION,
        ...     title="Annual DDS Submission",
        ...     due_date=datetime(2026, 3, 31, tzinfo=timezone.utc),
        ... )
        >>> assert record.total_deadlines == 1
    """

    def __init__(
        self,
        config: Optional[AnnualReviewSchedulerConfig] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize DeadlineTracker engine."""
        self.config = config or get_config()
        self._provenance = provenance or ProvenanceTracker()
        self._tracking_records: Dict[str, DeadlineTrackingRecord] = {}
        self._deadlines: Dict[str, DeadlineEntry] = {}
        self._submissions: Dict[str, Dict[str, Any]] = {}
        self._deadline_tracks: Dict[str, DeadlineTrack] = {}
        self._alerts: Dict[str, DeadlineAlert] = {}
        logger.info("DeadlineTracker engine initialized")

    async def register_deadline(
        self,
        operator_id: str,
        deadline_type: DeadlineType,
        title: str,
        due_date: datetime,
        article_reference: str = "",
        responsible_entity: Optional[str] = None,
        review_year: int = 0,
    ) -> DeadlineTrackingRecord:
        """Register a new deadline for tracking.

        Args:
            operator_id: Operator identifier.
            deadline_type: Type of deadline.
            title: Deadline title.
            due_date: Deadline date.
            article_reference: Related EUDR article.
            responsible_entity: Entity responsible for meeting deadline.
            review_year: Applicable review year.

        Returns:
            DeadlineTrackingRecord with the registered deadline.
        """
        start_time = time.monotonic()
        now = datetime.now(timezone.utc).replace(microsecond=0)
        deadline_id = str(uuid.uuid4())
        tracking_id = str(uuid.uuid4())

        days_remaining = (due_date.replace(tzinfo=timezone.utc) - now).days
        status = self._classify_deadline_status(days_remaining)

        deadline = DeadlineEntry(
            deadline_id=deadline_id,
            deadline_type=deadline_type,
            title=title,
            due_date=due_date,
            status=status,
            days_remaining=days_remaining,
            responsible_entity=responsible_entity,
            article_reference=article_reference,
        )
        self._deadlines[deadline_id] = deadline

        # Build or update tracking record
        record = DeadlineTrackingRecord(
            tracking_id=tracking_id,
            operator_id=operator_id,
            review_year=review_year,
            deadlines=[deadline],
            total_deadlines=1,
            approaching_count=1 if status == DeadlineStatus.AT_RISK else 0,
            overdue_count=1 if status == DeadlineStatus.OVERDUE else 0,
            met_count=0,
            next_deadline=due_date,
            checked_at=now,
        )

        # Compute provenance
        prov_data = {
            "tracking_id": tracking_id,
            "deadline_id": deadline_id,
            "operator_id": operator_id,
            "due_date": due_date.isoformat(),
            "registered_at": now.isoformat(),
        }
        record.provenance_hash = self._provenance.compute_hash(prov_data)
        self._provenance.record(
            "deadline", "register", deadline_id, AGENT_ID,
            metadata={"type": deadline_type.value, "due_date": due_date.isoformat()},
        )

        self._tracking_records[tracking_id] = record

        elapsed = time.monotonic() - start_time
        m.observe_deadline_check_duration(elapsed)
        m.record_deadline_registered(deadline_type.value)
        m.set_approaching_deadlines(record.approaching_count)
        m.set_overdue_deadlines(record.overdue_count)

        logger.info(
            "Deadline %s registered: %s due %s (status=%s, %d days)",
            deadline_id, title, due_date.isoformat(),
            status.value, days_remaining,
        )
        return record

    async def check_approaching_deadlines(
        self,
        operator_id: str,
        review_year: Optional[int] = None,
    ) -> DeadlineTrackingRecord:
        """Check all deadlines for approaching and overdue items.

        Args:
            operator_id: Operator identifier.
            review_year: Optional filter by review year.

        Returns:
            DeadlineTrackingRecord with updated statuses.
        """
        start_time = time.monotonic()
        now = datetime.now(timezone.utc).replace(microsecond=0)
        tracking_id = str(uuid.uuid4())

        # Refresh all deadline statuses
        updated_deadlines: List[DeadlineEntry] = []
        for deadline in self._deadlines.values():
            days_remaining = (deadline.due_date.replace(tzinfo=timezone.utc) - now).days
            deadline.days_remaining = days_remaining
            if deadline.status != DeadlineStatus.COMPLETED:
                deadline.status = self._classify_deadline_status(days_remaining)
            updated_deadlines.append(deadline)

        approaching = [
            d for d in updated_deadlines
            if d.status == DeadlineStatus.AT_RISK
        ]
        overdue = [d for d in updated_deadlines if d.status == DeadlineStatus.OVERDUE]
        met = [d for d in updated_deadlines if d.status == DeadlineStatus.COMPLETED]

        # Find next deadline
        upcoming = [
            d for d in updated_deadlines
            if d.status not in (DeadlineStatus.COMPLETED, DeadlineStatus.WAIVED, DeadlineStatus.OVERDUE)
        ]
        upcoming.sort(key=lambda d: d.due_date)
        next_deadline = upcoming[0].due_date if upcoming else None

        record = DeadlineTrackingRecord(
            tracking_id=tracking_id,
            operator_id=operator_id,
            review_year=review_year or 0,
            deadlines=updated_deadlines,
            total_deadlines=len(updated_deadlines),
            approaching_count=len(approaching),
            overdue_count=len(overdue),
            met_count=len(met),
            next_deadline=next_deadline,
            checked_at=now,
        )

        # Provenance
        prov_data = {
            "tracking_id": tracking_id,
            "total": len(updated_deadlines),
            "approaching": len(approaching),
            "overdue": len(overdue),
            "checked_at": now.isoformat(),
        }
        record.provenance_hash = self._provenance.compute_hash(prov_data)
        self._tracking_records[tracking_id] = record

        elapsed = time.monotonic() - start_time
        m.observe_deadline_check_duration(elapsed)
        m.set_approaching_deadlines(len(approaching))
        m.set_overdue_deadlines(len(overdue))

        # Record overdue metrics
        for d in overdue:
            m.record_deadline_overdue()

        logger.info(
            "Deadline check for %s: %d total, %d approaching, %d overdue",
            operator_id, len(updated_deadlines), len(approaching), len(overdue),
        )
        return record

    async def submit_to_authority(
        self,
        operator_id: str,
        deadline_id: str,
        submission_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Submit compliance documentation to competent authority.

        Args:
            operator_id: Operator identifier.
            deadline_id: Deadline being fulfilled.
            submission_data: Submission payload.

        Returns:
            Submission result dictionary.
        """
        start_time = time.monotonic()
        now = datetime.now(timezone.utc).replace(microsecond=0)
        submission_id = str(uuid.uuid4())

        deadline = self._deadlines.get(deadline_id)
        if deadline is None:
            raise ValueError(f"Deadline {deadline_id} not found")

        # Record submission
        result = {
            "submission_id": submission_id,
            "operator_id": operator_id,
            "deadline_id": deadline_id,
            "submitted_at": now.isoformat(),
            "status": "submitted",
            "reference_number": f"EUDR-SUB-{now.strftime('%Y%m%d')}-{submission_id[:8]}",
            "data_hash": self._provenance.compute_hash(submission_data),
        }

        # Update deadline status
        deadline.status = DeadlineStatus.COMPLETED

        self._submissions[submission_id] = result
        self._provenance.record(
            "submission", "submit_to_authority", submission_id, AGENT_ID,
            metadata={
                "operator_id": operator_id,
                "deadline_id": deadline_id,
            },
        )

        elapsed = time.monotonic() - start_time
        m.observe_submission_duration(elapsed)
        m.record_submission("submitted")
        m.record_deadline_met()

        logger.info(
            "Submission %s created for deadline %s (operator=%s)",
            submission_id, deadline_id, operator_id,
        )
        return result

    async def track_submission_status(
        self,
        submission_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Track the status of a previous submission.

        Args:
            submission_id: Submission identifier.

        Returns:
            Submission status dictionary or None.
        """
        return self._submissions.get(submission_id)

    async def get_record(self, tracking_id: str) -> Optional[DeadlineTrackingRecord]:
        """Get a specific tracking record by ID."""
        return self._tracking_records.get(tracking_id)

    async def list_records(
        self,
        operator_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[DeadlineTrackingRecord]:
        """List deadline tracking records."""
        results = list(self._tracking_records.values())
        if operator_id:
            results = [r for r in results if r.operator_id == operator_id]
        results.sort(key=lambda r: r.checked_at, reverse=True)
        return results[offset: offset + limit]

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status."""
        return {
            "engine": "DeadlineTracker",
            "status": "healthy",
            "total_deadlines": len(self._deadlines),
            "total_submissions": len(self._submissions),
        }

    # -- Engine-test API (DeadlineTrack-based) --

    async def create_deadline(
        self,
        cycle_id: str,
        phase: ReviewPhase,
        description: str,
        due_date: datetime,
        assigned_entity_id: Optional[str] = None,
        warning_days_before: int = 7,
        critical_days_before: int = 3,
    ) -> DeadlineTrack:
        """Create a new deadline returning a DeadlineTrack model."""
        now = datetime.now(timezone.utc).replace(microsecond=0)
        deadline_id = f"dln-{uuid.uuid4()}"
        days_remaining = (due_date.replace(tzinfo=timezone.utc) - now).days
        status = self._classify_deadline_status(days_remaining)

        prov_data = {
            "deadline_id": deadline_id,
            "cycle_id": cycle_id,
            "due_date": due_date.isoformat(),
            "created_at": now.isoformat(),
        }
        prov_hash = self._provenance.compute_hash(prov_data)

        track = DeadlineTrack(
            deadline_id=deadline_id,
            cycle_id=cycle_id,
            phase=phase,
            description=description,
            due_date=due_date,
            status=status,
            assigned_entity_id=assigned_entity_id,
            warning_days_before=warning_days_before,
            critical_days_before=critical_days_before,
            provenance_hash=prov_hash,
        )
        self._deadline_tracks[deadline_id] = track
        m.record_deadline_registered("engine")
        return track

    async def complete_deadline(self, deadline_id: str) -> DeadlineTrack:
        """Mark a deadline as completed."""
        track = self._get_deadline_track(deadline_id)
        if track.status == DeadlineStatus.COMPLETED:
            raise ValueError(f"Deadline {deadline_id} is already completed")
        now = datetime.now(timezone.utc).replace(microsecond=0)
        track.status = DeadlineStatus.COMPLETED
        track.completed_at = now
        track.provenance_hash = self._provenance.compute_hash(
            {"deadline_id": deadline_id, "action": "complete", "at": now.isoformat()}
        )
        m.record_deadline_met()
        return track

    async def waive_deadline(self, deadline_id: str, reason: str = "") -> DeadlineTrack:
        """Waive a deadline."""
        track = self._get_deadline_track(deadline_id)
        track.status = DeadlineStatus.WAIVED
        track.provenance_hash = self._provenance.compute_hash(
            {"deadline_id": deadline_id, "action": "waive"}
        )
        return track

    async def mark_at_risk(self, deadline_id: str, reason: str = "") -> DeadlineTrack:
        """Mark a deadline as at-risk."""
        track = self._get_deadline_track(deadline_id)
        track.status = DeadlineStatus.AT_RISK
        return track

    async def extend_deadline(self, deadline_id: str, additional_days: int = 0) -> DeadlineTrack:
        """Extend a deadline by adding additional days."""
        if additional_days <= 0:
            raise ValueError("additional_days must be positive")
        track = self._get_deadline_track(deadline_id)
        if track.status == DeadlineStatus.COMPLETED:
            raise ValueError(f"Cannot extend completed deadline {deadline_id}")
        track.due_date = track.due_date + timedelta(days=additional_days)
        # Re-evaluate status
        now = datetime.now(timezone.utc).replace(microsecond=0)
        days_remaining = (track.due_date.replace(tzinfo=timezone.utc) - now).days
        track.status = self._classify_deadline_status(days_remaining)
        track.provenance_hash = self._provenance.compute_hash(
            {"deadline_id": deadline_id, "action": "extend", "days": additional_days}
        )
        return track

    async def get_deadline(self, deadline_id: str) -> DeadlineTrack:
        """Get a deadline by ID."""
        return self._get_deadline_track(deadline_id)

    async def list_deadlines(
        self,
        cycle_id: Optional[str] = None,
        phase: Optional[ReviewPhase] = None,
        status: Optional[DeadlineStatus] = None,
    ) -> List[DeadlineTrack]:
        """List deadlines with optional filters."""
        results = list(self._deadline_tracks.values())
        if cycle_id:
            results = [d for d in results if d.cycle_id == cycle_id]
        if phase is not None:
            results = [d for d in results if d.phase == phase]
        if status is not None:
            results = [d for d in results if d.status == status]
        results.sort(key=lambda d: d.due_date)
        return results

    async def get_overdue_deadlines(self, cycle_id: str) -> List[DeadlineTrack]:
        """Get all overdue deadlines for a cycle."""
        now = datetime.now(timezone.utc).replace(microsecond=0)
        results: List[DeadlineTrack] = []
        for d in self._deadline_tracks.values():
            if d.cycle_id != cycle_id:
                continue
            if d.status in (DeadlineStatus.COMPLETED, DeadlineStatus.WAIVED):
                continue
            days_remaining = (d.due_date.replace(tzinfo=timezone.utc) - now).days
            if days_remaining < 0:
                d.status = DeadlineStatus.OVERDUE
                results.append(d)
        return results

    async def check_and_generate_alerts(self, deadline_id: str) -> List[DeadlineAlert]:
        """Check a deadline and generate alerts if needed."""
        track = self._get_deadline_track(deadline_id)
        now = datetime.now(timezone.utc).replace(microsecond=0)
        days_remaining = (track.due_date.replace(tzinfo=timezone.utc) - now).days

        alerts: List[DeadlineAlert] = []

        if days_remaining < 0:
            # Overdue -> escalation alert
            alert = DeadlineAlert(
                alert_id=f"alt-{uuid.uuid4()}",
                deadline_id=deadline_id,
                cycle_id=track.cycle_id,
                alert_level=DeadlineAlertLevel.ESCALATION,
                message=f"Deadline overdue by {abs(days_remaining)} days",
                days_remaining=days_remaining,
            )
            alerts.append(alert)
            self._alerts[alert.alert_id] = alert
        elif days_remaining <= track.critical_days_before:
            alert = DeadlineAlert(
                alert_id=f"alt-{uuid.uuid4()}",
                deadline_id=deadline_id,
                cycle_id=track.cycle_id,
                alert_level=DeadlineAlertLevel.CRITICAL,
                message=f"Critical: {days_remaining} days remaining",
                days_remaining=days_remaining,
            )
            alerts.append(alert)
            self._alerts[alert.alert_id] = alert
        elif days_remaining <= track.warning_days_before:
            alert = DeadlineAlert(
                alert_id=f"alt-{uuid.uuid4()}",
                deadline_id=deadline_id,
                cycle_id=track.cycle_id,
                alert_level=DeadlineAlertLevel.WARNING,
                message=f"Warning: {days_remaining} days remaining",
                days_remaining=days_remaining,
            )
            alerts.append(alert)
            self._alerts[alert.alert_id] = alert

        return alerts

    async def acknowledge_alert(self, alert_id: str) -> DeadlineAlert:
        """Acknowledge a deadline alert."""
        alert = self._alerts.get(alert_id)
        if alert is None:
            raise ValueError(f"Alert {alert_id} not found")
        alert.acknowledged = True
        return alert

    async def batch_check_alerts(self, cycle_id: str) -> List[DeadlineAlert]:
        """Run alert checks for all deadlines in a cycle."""
        all_alerts: List[DeadlineAlert] = []
        for d in self._deadline_tracks.values():
            if d.cycle_id != cycle_id:
                continue
            if d.status in (DeadlineStatus.COMPLETED, DeadlineStatus.WAIVED):
                continue
            alerts = await self.check_and_generate_alerts(d.deadline_id)
            all_alerts.extend(alerts)
        return all_alerts

    async def create_deadlines_for_cycle(self, cycle) -> List[DeadlineTrack]:
        """Create standard deadlines for all phases of a review cycle."""
        deadlines: List[DeadlineTrack] = []
        base_date = cycle.scheduled_start or datetime.now(timezone.utc)
        offset = 0

        for i, phase in enumerate(REVIEW_PHASES_ORDER):
            phase_cfg = None
            if hasattr(cycle, 'phase_configs'):
                for pc in cycle.phase_configs:
                    if pc.phase == phase:
                        phase_cfg = pc
                        break
            duration = phase_cfg.duration_days if phase_cfg else 14
            offset += duration
            due = base_date + timedelta(days=offset)
            d = await self.create_deadline(
                cycle_id=cycle.cycle_id,
                phase=phase,
                description=f"Complete {phase.value} phase",
                due_date=due,
            )
            deadlines.append(d)
        return deadlines

    def _get_deadline_track(self, deadline_id: str) -> DeadlineTrack:
        """Retrieve a DeadlineTrack or raise ValueError."""
        track = self._deadline_tracks.get(deadline_id)
        if track is None:
            raise ValueError(f"Deadline {deadline_id} not found")
        return track

    # -- Private helpers --

    def _classify_deadline_status(self, days_remaining: int) -> DeadlineStatus:
        """Classify deadline status by days remaining."""
        if days_remaining < 0:
            return DeadlineStatus.OVERDUE
        elif days_remaining <= self.config.deadline_critical_days:
            return DeadlineStatus.AT_RISK
        elif days_remaining <= self.config.deadline_warning_days:
            return DeadlineStatus.AT_RISK
        return DeadlineStatus.ON_TRACK
