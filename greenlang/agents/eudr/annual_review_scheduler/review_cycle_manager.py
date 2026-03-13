# -*- coding: utf-8 -*-
"""
Review Cycle Manager Engine - AGENT-EUDR-034

Creates, schedules, and manages annual review cycles for EUDR compliance.
Handles cycle lifecycle from draft through completion, including task
generation, status transitions, and completion tracking.

Zero-Hallucination:
    - All completion percentages are deterministic Decimal arithmetic
    - Task scheduling uses pure date arithmetic (no ML/LLM)
    - Status transitions follow a deterministic state machine

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
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import AnnualReviewSchedulerConfig, get_config
from .models import (
    AGENT_ID,
    ChecklistItemStatus,
    ChecklistPriority,
    ReviewCycleRecord,
    ReviewCycleStatus,
    ReviewTask,
    ReviewCycle,
    ReviewType,
    ReviewPhase,
    ReviewPhaseConfig,
    CommodityScope,
    REVIEW_PHASES_ORDER,
)
from .provenance import ProvenanceTracker
from . import metrics as m

logger = logging.getLogger(__name__)

# Valid status transitions for review cycles
_VALID_TRANSITIONS: Dict[ReviewCycleStatus, List[ReviewCycleStatus]] = {
    ReviewCycleStatus.DRAFT: [
        ReviewCycleStatus.SCHEDULED,
        ReviewCycleStatus.CANCELLED,
    ],
    ReviewCycleStatus.SCHEDULED: [
        ReviewCycleStatus.IN_PROGRESS,
        ReviewCycleStatus.CANCELLED,
    ],
    ReviewCycleStatus.IN_PROGRESS: [
        ReviewCycleStatus.PAUSED,
        ReviewCycleStatus.COMPLETED,
        ReviewCycleStatus.OVERDUE,
        ReviewCycleStatus.CANCELLED,
    ],
    ReviewCycleStatus.PAUSED: [
        ReviewCycleStatus.IN_PROGRESS,
        ReviewCycleStatus.CANCELLED,
    ],
    ReviewCycleStatus.OVERDUE: [
        ReviewCycleStatus.IN_PROGRESS,
        ReviewCycleStatus.COMPLETED,
        ReviewCycleStatus.CANCELLED,
    ],
    ReviewCycleStatus.COMPLETED: [],
    ReviewCycleStatus.CANCELLED: [],
}

# Standard review tasks generated for each cycle
_STANDARD_TASK_TEMPLATES: List[Dict[str, Any]] = [
    {
        "title": "Review supply chain mapping updates",
        "section": "supply_chain",
        "priority": ChecklistPriority.MANDATORY,
        "article": "Article 8",
    },
    {
        "title": "Verify geolocation data accuracy",
        "section": "geolocation",
        "priority": ChecklistPriority.MANDATORY,
        "article": "Article 9",
    },
    {
        "title": "Update risk assessment results",
        "section": "risk_assessment",
        "priority": ChecklistPriority.MANDATORY,
        "article": "Article 10",
    },
    {
        "title": "Review due diligence statements",
        "section": "due_diligence",
        "priority": ChecklistPriority.MANDATORY,
        "article": "Article 4",
    },
    {
        "title": "Validate satellite monitoring coverage",
        "section": "monitoring",
        "priority": ChecklistPriority.HIGH,
        "article": "Article 10",
    },
    {
        "title": "Check certification renewals",
        "section": "certifications",
        "priority": ChecklistPriority.HIGH,
        "article": "Article 12",
    },
    {
        "title": "Review deforestation alert responses",
        "section": "deforestation",
        "priority": ChecklistPriority.HIGH,
        "article": "Article 10",
    },
    {
        "title": "Update supplier risk scores",
        "section": "risk_scoring",
        "priority": ChecklistPriority.MEDIUM,
        "article": "Article 10",
    },
    {
        "title": "Validate document authentication records",
        "section": "documentation",
        "priority": ChecklistPriority.MEDIUM,
        "article": "Article 12",
    },
    {
        "title": "Review regulatory compliance changes",
        "section": "regulatory",
        "priority": ChecklistPriority.MEDIUM,
        "article": "Article 29",
    },
    {
        "title": "Confirm data retention compliance",
        "section": "retention",
        "priority": ChecklistPriority.LOW,
        "article": "Article 31",
    },
    {
        "title": "Generate year-over-year comparison report",
        "section": "reporting",
        "priority": ChecklistPriority.LOW,
        "article": "Article 14",
    },
]


class ReviewCycleManager:
    """Review cycle creation and management engine.

    Creates annual review cycles, generates tasks, manages lifecycle
    transitions, and tracks completion across all review activities.

    Example:
        >>> manager = ReviewCycleManager()
        >>> cycle = await manager.create_review_cycle(
        ...     operator_id="OP-001", review_year=2026
        ... )
        >>> assert cycle.cycle_status == ReviewCycleStatus.DRAFT
    """

    def __init__(
        self,
        config: Optional[AnnualReviewSchedulerConfig] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize ReviewCycleManager engine."""
        self.config = config or get_config()
        self._provenance = provenance or ProvenanceTracker()
        self._cycles: Dict[str, ReviewCycleRecord] = {}
        self._review_cycles: Dict[str, ReviewCycle] = {}
        logger.info("ReviewCycleManager engine initialized")

    async def create_review_cycle(
        self,
        operator_id: str,
        review_year: int,
        commodities: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
    ) -> ReviewCycleRecord:
        """Create a new annual review cycle.

        Args:
            operator_id: Operator identifier.
            review_year: Year under review.
            commodities: Commodities covered by this cycle.
            start_date: Optional explicit start date.

        Returns:
            ReviewCycleRecord with cycle details.
        """
        start_time = time.monotonic()
        now = datetime.now(timezone.utc).replace(microsecond=0)
        cycle_id = str(uuid.uuid4())

        # Validate active cycle limit
        active = self._count_active_cycles(operator_id)
        if active >= self.config.review_cycle_max_active:
            logger.warning(
                "Operator %s has %d active cycles (max %d)",
                operator_id, active, self.config.review_cycle_max_active,
            )

        # Determine cycle dates
        cycle_start = start_date or self._compute_default_start(review_year)
        cycle_end = cycle_start + timedelta(days=self.config.review_cycle_duration_days)
        grace_deadline = cycle_end + timedelta(days=self.config.review_cycle_grace_period_days)

        record = ReviewCycleRecord(
            cycle_id=cycle_id,
            operator_id=operator_id,
            review_year=review_year,
            cycle_status=ReviewCycleStatus.DRAFT,
            start_date=cycle_start,
            end_date=cycle_end,
            grace_deadline=grace_deadline,
            commodities=commodities or [],
            created_at=now,
        )

        # Compute provenance hash
        prov_data = {
            "cycle_id": cycle_id,
            "operator_id": operator_id,
            "review_year": review_year,
            "created_at": now.isoformat(),
        }
        record.provenance_hash = self._provenance.compute_hash(prov_data)
        self._provenance.record(
            "review_cycle", "create", cycle_id, AGENT_ID,
            metadata={"operator_id": operator_id, "review_year": review_year},
        )

        self._cycles[cycle_id] = record

        elapsed = time.monotonic() - start_time
        m.observe_cycle_creation_duration(elapsed)
        m.record_review_cycle_created(record.cycle_status.value)
        m.set_active_review_cycles(self._count_active_cycles(operator_id))

        logger.info(
            "Review cycle %s created for operator %s year %d",
            cycle_id, operator_id, review_year,
        )
        return record

    async def schedule_tasks(
        self,
        cycle_id: str,
        additional_tasks: Optional[List[Dict[str, Any]]] = None,
    ) -> ReviewCycleRecord:
        """Generate and schedule review tasks for a cycle.

        Args:
            cycle_id: Cycle identifier.
            additional_tasks: Extra task definitions to include.

        Returns:
            Updated ReviewCycleRecord with tasks.

        Raises:
            ValueError: If cycle not found.
        """
        start_time = time.monotonic()
        record = self._get_cycle(cycle_id)
        now = datetime.now(timezone.utc).replace(microsecond=0)

        tasks: List[ReviewTask] = []

        # Generate standard tasks
        for template in _STANDARD_TASK_TEMPLATES:
            task = ReviewTask(
                task_id=str(uuid.uuid4()),
                title=template["title"],
                description=f"EUDR {template['article']} - {template['title']}",
                status=ChecklistItemStatus.PENDING,
                priority=template["priority"],
                due_date=record.end_date,
            )
            tasks.append(task)

        # Add custom tasks
        if additional_tasks:
            for task_def in additional_tasks[: self.config.review_cycle_task_batch_size]:
                task = ReviewTask(
                    task_id=str(uuid.uuid4()),
                    title=task_def.get("title", "Custom review task"),
                    description=task_def.get("description", ""),
                    assignee=task_def.get("assignee"),
                    status=ChecklistItemStatus.PENDING,
                    priority=ChecklistPriority(
                        task_def.get("priority", "medium")
                    ),
                    due_date=record.end_date,
                )
                tasks.append(task)

        record.tasks = tasks
        record.tasks_total = len(tasks)
        record.tasks_completed = 0
        record.completion_percent = Decimal("0")
        record.updated_at = now

        # Update provenance
        prov_data = {
            "cycle_id": cycle_id, "tasks_count": len(tasks),
            "scheduled_at": now.isoformat(),
        }
        record.provenance_hash = self._provenance.compute_hash(prov_data)

        elapsed = time.monotonic() - start_time
        m.observe_task_scheduling_duration(elapsed)
        for task in tasks:
            m.record_review_task_scheduled(task.priority.value)
        m.set_pending_tasks(record.tasks_total)

        logger.info(
            "Scheduled %d tasks for cycle %s", len(tasks), cycle_id,
        )
        return record

    async def update_cycle_status(
        self,
        cycle_id: str,
        new_status: ReviewCycleStatus,
    ) -> ReviewCycleRecord:
        """Transition a review cycle to a new status.

        Args:
            cycle_id: Cycle identifier.
            new_status: Target status.

        Returns:
            Updated ReviewCycleRecord.

        Raises:
            ValueError: If transition is invalid or cycle not found.
        """
        record = self._get_cycle(cycle_id)
        current = record.cycle_status

        allowed = _VALID_TRANSITIONS.get(current, [])
        if new_status not in allowed:
            raise ValueError(
                f"Invalid transition from {current.value} to {new_status.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )

        now = datetime.now(timezone.utc).replace(microsecond=0)
        record.cycle_status = new_status
        record.updated_at = now

        if new_status == ReviewCycleStatus.COMPLETED:
            record.completed_at = now
            m.record_review_completed()

        # Update provenance
        prov_data = {
            "cycle_id": cycle_id, "old_status": current.value,
            "new_status": new_status.value, "updated_at": now.isoformat(),
        }
        record.provenance_hash = self._provenance.compute_hash(prov_data)
        self._provenance.record(
            "review_cycle", "status_change", cycle_id, AGENT_ID,
            metadata={"from": current.value, "to": new_status.value},
        )

        m.set_active_review_cycles(
            self._count_active_cycles(record.operator_id)
        )

        logger.info(
            "Cycle %s transitioned %s -> %s",
            cycle_id, current.value, new_status.value,
        )
        return record

    async def get_active_cycles(
        self,
        operator_id: Optional[str] = None,
    ) -> List[ReviewCycleRecord]:
        """Get all active review cycles.

        Args:
            operator_id: Optional filter by operator.

        Returns:
            List of active ReviewCycleRecords.
        """
        active_statuses = {
            ReviewCycleStatus.DRAFT,
            ReviewCycleStatus.SCHEDULED,
            ReviewCycleStatus.IN_PROGRESS,
            ReviewCycleStatus.PAUSED,
            ReviewCycleStatus.OVERDUE,
        }
        results = []
        for record in self._cycles.values():
            if record.cycle_status not in active_statuses:
                continue
            if operator_id and record.operator_id != operator_id:
                continue
            results.append(record)
        return sorted(results, key=lambda r: r.created_at, reverse=True)

    async def get_cycle(self, cycle_id: str) -> ReviewCycle:
        """Get a specific review cycle by ID.

        Args:
            cycle_id: Cycle identifier.

        Returns:
            ReviewCycle.

        Raises:
            ValueError: If cycle not found.
        """
        cycle = self._review_cycles.get(cycle_id)
        if cycle is not None:
            return cycle
        # Fall back to old-style records for backward compat
        record = self._cycles.get(cycle_id)
        if record is not None:
            # Wrap in ReviewCycle for API consistency
            return ReviewCycle(
                cycle_id=record.cycle_id,
                operator_id=record.operator_id,
                review_year=record.review_year,
                status=record.cycle_status,
                created_at=record.created_at,
                provenance_hash=record.provenance_hash,
            )
        raise ValueError(f"Review cycle {cycle_id} not found")

    async def list_cycles(
        self,
        operator_id: Optional[str] = None,
        review_year: Optional[int] = None,
        status: Optional[Any] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ReviewCycle]:
        """List review cycles with optional filters.

        Args:
            operator_id: Filter by operator.
            review_year: Filter by review year.
            status: Filter by status (str or ReviewCycleStatus enum).
            limit: Maximum results.
            offset: Skip first N results.

        Returns:
            Filtered list of ReviewCycles.
        """
        results: List[ReviewCycle] = list(self._review_cycles.values())
        if operator_id:
            results = [r for r in results if r.operator_id == operator_id]
        if review_year:
            results = [r for r in results if r.review_year == review_year]
        if status is not None:
            status_val = status.value if isinstance(status, ReviewCycleStatus) else str(status)
            results = [r for r in results if r.status.value == status_val]
        results.sort(key=lambda r: r.created_at, reverse=True)
        return results[offset: offset + limit]

    async def update_task_status(
        self,
        cycle_id: str,
        task_id: str,
        new_status: ChecklistItemStatus,
    ) -> ReviewCycleRecord:
        """Update status of a specific task within a cycle.

        Args:
            cycle_id: Cycle identifier.
            task_id: Task identifier.
            new_status: New task status.

        Returns:
            Updated ReviewCycleRecord.

        Raises:
            ValueError: If cycle or task not found.
        """
        record = self._get_cycle(cycle_id)
        now = datetime.now(timezone.utc).replace(microsecond=0)

        task_found = False
        for task in record.tasks:
            if task.task_id == task_id:
                task.status = new_status
                if new_status == ChecklistItemStatus.COMPLETED:
                    task.completed_at = now
                task_found = True
                break

        if not task_found:
            raise ValueError(f"Task {task_id} not found in cycle {cycle_id}")

        # Recalculate completion
        record.tasks_completed = sum(
            1 for t in record.tasks
            if t.status == ChecklistItemStatus.COMPLETED
        )
        if record.tasks_total > 0:
            record.completion_percent = (
                Decimal(str(record.tasks_completed))
                / Decimal(str(record.tasks_total))
                * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        record.updated_at = now

        m.set_pending_tasks(record.tasks_total - record.tasks_completed)
        m.set_overall_review_progress(float(record.completion_percent))

        logger.info(
            "Task %s in cycle %s updated to %s (%.1f%% complete)",
            task_id, cycle_id, new_status.value, record.completion_percent,
        )
        return record

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status."""
        return {
            "engine": "ReviewCycleManager",
            "status": "healthy",
            "total_cycles": len(self._cycles),
            "active_cycles": sum(
                1 for c in self._cycles.values()
                if c.cycle_status not in (
                    ReviewCycleStatus.COMPLETED,
                    ReviewCycleStatus.CANCELLED,
                )
            ),
        }

    # -- Engine-test API (ReviewCycle-based) --

    async def create_cycle(
        self,
        operator_id: str,
        review_year: int,
        review_type: ReviewType = ReviewType.ANNUAL,
        commodity_scope: Optional[List[CommodityScope]] = None,
    ) -> ReviewCycle:
        """Create a new review cycle returning a ReviewCycle model."""
        now = datetime.now(timezone.utc).replace(microsecond=0)
        cycle_id = f"cyc-{uuid.uuid4()}"
        cycle_start = now + timedelta(days=7)
        cycle_end = cycle_start + timedelta(days=self.config.review_cycle_duration_days)

        phase_configs = [
            ReviewPhaseConfig(phase=p, duration_days=getattr(
                self.config,
                f"{p.value}_phase_days",
                self.config.review_cycle_duration_days // 6,
            ))
            for p in REVIEW_PHASES_ORDER
        ]

        prov_data = {
            "cycle_id": cycle_id,
            "operator_id": operator_id,
            "review_year": review_year,
            "review_type": review_type.value,
            "created_at": now.isoformat(),
        }
        prov_hash = self._provenance.compute_hash(prov_data)

        cycle = ReviewCycle(
            cycle_id=cycle_id,
            operator_id=operator_id,
            review_year=review_year,
            review_type=review_type,
            commodity_scope=commodity_scope or [],
            status=ReviewCycleStatus.DRAFT,
            current_phase=ReviewPhase.PREPARATION,
            phase_configs=phase_configs,
            scheduled_start=cycle_start,
            scheduled_end=cycle_end,
            created_by="AGENT-EUDR-034",
            created_at=now,
            provenance_hash=prov_hash,
        )
        self._review_cycles[cycle_id] = cycle
        m.record_review_cycle_created("draft")
        return cycle

    async def schedule_cycle(
        self, cycle_id: str, start_date: Optional[datetime] = None,
    ) -> ReviewCycle:
        """Schedule a draft cycle."""
        cycle = self._get_review_cycle(cycle_id)
        if cycle.status != ReviewCycleStatus.DRAFT:
            raise ValueError(f"Cannot schedule cycle in {cycle.status.value} state")
        cycle.status = ReviewCycleStatus.SCHEDULED
        if start_date:
            cycle.scheduled_start = start_date
            cycle.scheduled_end = start_date + timedelta(days=self.config.review_cycle_duration_days)
        cycle.provenance_hash = self._provenance.compute_hash(
            {"cycle_id": cycle_id, "action": "schedule"}
        )
        return cycle

    async def start_cycle(self, cycle_id: str) -> ReviewCycle:
        """Start a scheduled cycle."""
        cycle = self._get_review_cycle(cycle_id)
        if cycle.status != ReviewCycleStatus.SCHEDULED:
            raise ValueError(f"Cannot start cycle in {cycle.status.value} state")
        cycle.status = ReviewCycleStatus.IN_PROGRESS
        cycle.actual_start = datetime.now(timezone.utc).replace(microsecond=0)
        cycle.provenance_hash = self._provenance.compute_hash(
            {"cycle_id": cycle_id, "action": "start"}
        )
        return cycle

    async def advance_phase(self, cycle_id: str) -> ReviewCycle:
        """Advance cycle to the next phase."""
        cycle = self._get_review_cycle(cycle_id)
        if cycle.status != ReviewCycleStatus.IN_PROGRESS:
            raise ValueError(f"Cannot advance phase for cycle in {cycle.status.value} state")

        current_idx = REVIEW_PHASES_ORDER.index(cycle.current_phase)
        if current_idx + 1 < len(REVIEW_PHASES_ORDER):
            cycle.current_phase = REVIEW_PHASES_ORDER[current_idx + 1]
        else:
            # Last phase complete -> complete cycle
            cycle.status = ReviewCycleStatus.COMPLETED
            cycle.actual_end = datetime.now(timezone.utc).replace(microsecond=0)
            m.record_review_completed()

        cycle.provenance_hash = self._provenance.compute_hash(
            {"cycle_id": cycle_id, "action": "advance_phase", "phase": cycle.current_phase.value}
        )
        return cycle

    async def pause_cycle(self, cycle_id: str, reason: str = "") -> ReviewCycle:
        """Pause an active cycle."""
        cycle = self._get_review_cycle(cycle_id)
        if cycle.status != ReviewCycleStatus.IN_PROGRESS:
            raise ValueError(f"Cannot pause cycle in {cycle.status.value} state")
        cycle.status = ReviewCycleStatus.PAUSED
        return cycle

    async def resume_cycle(self, cycle_id: str) -> ReviewCycle:
        """Resume a paused cycle."""
        cycle = self._get_review_cycle(cycle_id)
        if cycle.status != ReviewCycleStatus.PAUSED:
            raise ValueError(f"Cannot resume cycle in {cycle.status.value} state")
        cycle.status = ReviewCycleStatus.IN_PROGRESS
        return cycle

    async def cancel_cycle(self, cycle_id: str, reason: str = "") -> ReviewCycle:
        """Cancel a cycle."""
        cycle = self._get_review_cycle(cycle_id)
        if cycle.status in (ReviewCycleStatus.COMPLETED, ReviewCycleStatus.CANCELLED):
            raise ValueError(f"Cannot cancel cycle in {cycle.status.value} state")
        cycle.status = ReviewCycleStatus.CANCELLED
        return cycle

    async def auto_schedule_next(self, cycle_id: str) -> ReviewCycle:
        """Auto-schedule the next cycle after a completed one."""
        cycle = self._get_review_cycle(cycle_id)
        if cycle.status != ReviewCycleStatus.COMPLETED:
            raise ValueError(f"Cycle {cycle_id} is not completed")
        return await self.create_cycle(
            operator_id=cycle.operator_id,
            review_year=cycle.review_year + 1,
            review_type=cycle.review_type,
            commodity_scope=cycle.commodity_scope,
        )

    # -- Private helpers --

    def _get_review_cycle(self, cycle_id: str) -> ReviewCycle:
        """Retrieve a ReviewCycle or raise ValueError."""
        cycle = self._review_cycles.get(cycle_id)
        if cycle is None:
            raise ValueError(f"Review cycle {cycle_id} not found")
        return cycle

    def _get_cycle(self, cycle_id: str) -> ReviewCycleRecord:
        """Retrieve a cycle or raise ValueError."""
        record = self._cycles.get(cycle_id)
        if record is None:
            raise ValueError(f"Review cycle {cycle_id} not found")
        return record

    def _count_active_cycles(self, operator_id: str) -> int:
        """Count active cycles for an operator."""
        terminal = {ReviewCycleStatus.COMPLETED, ReviewCycleStatus.CANCELLED}
        return sum(
            1 for c in self._cycles.values()
            if c.operator_id == operator_id and c.cycle_status not in terminal
        )

    def _compute_default_start(self, review_year: int) -> datetime:
        """Compute the default start date for a review cycle.

        The default is January 1st of the year following the review year,
        minus the creation lead days.
        """
        base = datetime(
            review_year + 1,
            self.config.review_cycle_default_month,
            min(self.config.review_cycle_default_day, 28),
            tzinfo=timezone.utc,
        )
        lead = timedelta(days=self.config.review_cycle_creation_lead_days)
        return (base - lead).replace(microsecond=0)
