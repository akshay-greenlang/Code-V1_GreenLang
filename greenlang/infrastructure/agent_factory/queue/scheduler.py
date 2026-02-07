# -*- coding: utf-8 -*-
"""
Task Scheduler - Cron-based and event-driven task scheduling.

Provides periodic scheduling of tasks using cron expressions and
event-driven triggering through message bus subscriptions.  A schedule
registry with CRUD operations tracks all schedules, and a background loop
computes next-run times and enqueues tasks when they are due.

Cron expression parsing is intentionally lightweight (no external deps).
Supported fields: minute, hour, day-of-month, month, day-of-week with
basic wildcards (``*``, ``*/N``, ``N``).

Example:
    >>> scheduler = TaskScheduler(task_queue=queue)
    >>> schedule_id = await scheduler.create_schedule(
    ...     name="hourly-carbon-calc",
    ...     cron_expression="0 * * * *",
    ...     agent_key="carbon-agent",
    ...     payload={"scope": 1},
    ... )
    >>> await scheduler.start()

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

from greenlang.infrastructure.agent_factory.queue.task_queue import (
    DistributedTaskQueue,
    TaskItem,
    TaskPriority,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schedule model
# ---------------------------------------------------------------------------

class ScheduleStatus(str, Enum):
    """Runtime status of a schedule entry."""

    ACTIVE = "active"
    PAUSED = "paused"
    DELETED = "deleted"


@dataclass
class ScheduleEntry:
    """Definition of a scheduled recurring task.

    Attributes:
        id: Unique schedule identifier.
        name: Human-readable schedule name.
        cron_expression: Cron expression (min hour dom month dow).
        agent_key: Target agent for the spawned task.
        payload: Task payload.
        priority: Task priority.
        status: Active, paused, or deleted.
        created_at: UTC creation timestamp.
        last_run_at: UTC timestamp of the last execution.
        next_run_at: UTC timestamp of the next scheduled execution.
        run_count: Number of times this schedule has fired.
        metadata: Extra context.
        ttl: Task TTL in seconds (0 = no expiry).
        max_retries: Task retry limit.
    """

    name: str
    cron_expression: str
    agent_key: str
    payload: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: TaskPriority = TaskPriority.NORMAL
    status: ScheduleStatus = ScheduleStatus.ACTIVE
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_run_at: Optional[str] = None
    next_run_at: Optional[str] = None
    run_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    ttl: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "cron_expression": self.cron_expression,
            "agent_key": self.agent_key,
            "payload": self.payload,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "last_run_at": self.last_run_at,
            "next_run_at": self.next_run_at,
            "run_count": self.run_count,
            "metadata": self.metadata,
            "ttl": self.ttl,
            "max_retries": self.max_retries,
        }


# ---------------------------------------------------------------------------
# Lightweight cron parser
# ---------------------------------------------------------------------------

class CronExpression:
    """Minimal cron expression evaluator.

    Supports five fields:  minute  hour  dom  month  dow
    Each field can be:
    - ``*``    : every value
    - ``N``    : exact value
    - ``*/N``  : every N-th value
    - ``N,M``  : specific values

    This implementation is intentionally simple to avoid external
    dependencies.  For production use with complex expressions, replace
    with ``croniter``.
    """

    def __init__(self, expression: str) -> None:
        parts = expression.strip().split()
        if len(parts) != 5:
            raise ValueError(
                f"Cron expression must have 5 fields, got {len(parts)}: "
                f"'{expression}'"
            )
        self._minute = self._parse_field(parts[0], 0, 59)
        self._hour = self._parse_field(parts[1], 0, 23)
        self._dom = self._parse_field(parts[2], 1, 31)
        self._month = self._parse_field(parts[3], 1, 12)
        self._dow = self._parse_field(parts[4], 0, 6)

    def matches(self, dt: datetime) -> bool:
        """Return True if *dt* matches the cron expression."""
        return (
            dt.minute in self._minute
            and dt.hour in self._hour
            and dt.day in self._dom
            and dt.month in self._month
            and dt.weekday() in self._dow  # Monday=0
        )

    def next_run(self, after: datetime) -> datetime:
        """Compute the next datetime after *after* that matches.

        Scans forward minute-by-minute up to 366 days.  Returns the
        first matching minute boundary.
        """
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        limit = after + timedelta(days=366)
        while candidate < limit:
            if self.matches(candidate):
                return candidate
            candidate += timedelta(minutes=1)
        # Fallback: should not be reached for valid cron expressions
        return after + timedelta(hours=1)

    # ------------------------------------------------------------------
    # Field parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_field(token: str, lo: int, hi: int) -> set[int]:
        """Parse a single cron field into a set of valid integers."""
        values: set[int] = set()

        for part in token.split(","):
            part = part.strip()
            if part == "*":
                values.update(range(lo, hi + 1))
            elif part.startswith("*/"):
                step = int(part[2:])
                values.update(range(lo, hi + 1, step))
            elif "-" in part:
                a, b = part.split("-", 1)
                values.update(range(int(a), int(b) + 1))
            else:
                values.add(int(part))

        return values


# ---------------------------------------------------------------------------
# Event trigger
# ---------------------------------------------------------------------------

# Async callable that receives an event payload and returns a task payload.
EventTransformer = Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]


@dataclass
class EventTrigger:
    """Links a message-bus topic subscription to task creation.

    Attributes:
        id: Unique trigger identifier.
        topic: Message bus topic to subscribe to.
        agent_key: Target agent for the spawned task.
        priority: Task priority.
        transformer: Optional async callable that maps the event payload
            to the task payload.
        active: Whether the trigger is enabled.
    """

    topic: str
    agent_key: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: TaskPriority = TaskPriority.NORMAL
    transformer: Optional[EventTransformer] = None
    active: bool = True


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------

class TaskScheduler:
    """Cron-based and event-driven task scheduling engine.

    The scheduler maintains an in-memory registry of schedule entries.
    A background loop evaluates which schedules are due each minute and
    enqueues tasks into the DistributedTaskQueue.

    Event-driven triggers allow external event sources (message bus) to
    spawn tasks on demand.
    """

    def __init__(
        self,
        task_queue: DistributedTaskQueue,
        check_interval: float = 30.0,
    ) -> None:
        """Initialize the scheduler.

        Args:
            task_queue: Queue to enqueue scheduled tasks into.
            check_interval: Seconds between schedule evaluation passes.
        """
        self._queue = task_queue
        self._check_interval = check_interval
        self._schedules: Dict[str, ScheduleEntry] = {}
        self._cron_cache: Dict[str, CronExpression] = {}
        self._event_triggers: Dict[str, EventTrigger] = {}
        self._running: bool = False
        self._loop_task: Optional[asyncio.Task[None]] = None
        logger.debug(
            "TaskScheduler initialized (check_interval=%.1fs)",
            check_interval,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the scheduling loop."""
        if self._running:
            return
        self._running = True
        # Pre-compute next_run for all active schedules
        for entry in self._schedules.values():
            if entry.status == ScheduleStatus.ACTIVE:
                self._update_next_run(entry)
        self._loop_task = asyncio.create_task(self._scheduling_loop())
        logger.info(
            "TaskScheduler started (%d schedules, %d event triggers)",
            len(self._schedules),
            len(self._event_triggers),
        )

    async def stop(self) -> None:
        """Stop the scheduling loop."""
        self._running = False
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("TaskScheduler stopped")

    # ------------------------------------------------------------------
    # CRUD for cron schedules
    # ------------------------------------------------------------------

    async def create_schedule(
        self,
        name: str,
        cron_expression: str,
        agent_key: str,
        payload: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        ttl: int = 0,
        max_retries: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new cron schedule.

        Args:
            name: Human-readable name.
            cron_expression: Five-field cron expression.
            agent_key: Target agent.
            payload: Task payload.
            priority: Task priority.
            ttl: Task TTL in seconds.
            max_retries: Task retry budget.
            metadata: Extra metadata.

        Returns:
            Schedule ID.
        """
        # Validate cron expression
        cron = CronExpression(cron_expression)

        entry = ScheduleEntry(
            name=name,
            cron_expression=cron_expression,
            agent_key=agent_key,
            payload=payload or {},
            priority=priority,
            ttl=ttl,
            max_retries=max_retries,
            metadata=metadata or {},
        )

        self._schedules[entry.id] = entry
        self._cron_cache[entry.id] = cron
        self._update_next_run(entry)

        logger.info(
            "Schedule created: %s ('%s', cron='%s', next=%s)",
            entry.id,
            name,
            cron_expression,
            entry.next_run_at,
        )
        return entry.id

    async def update_schedule(
        self,
        schedule_id: str,
        *,
        cron_expression: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        priority: Optional[TaskPriority] = None,
        status: Optional[ScheduleStatus] = None,
    ) -> bool:
        """Update fields on an existing schedule.

        Args:
            schedule_id: Schedule to update.
            cron_expression: New cron expression.
            payload: New payload.
            priority: New priority.
            status: New status.

        Returns:
            True if the schedule was found and updated.
        """
        entry = self._schedules.get(schedule_id)
        if entry is None:
            return False

        if cron_expression is not None:
            cron = CronExpression(cron_expression)
            entry.cron_expression = cron_expression
            self._cron_cache[schedule_id] = cron

        if payload is not None:
            entry.payload = payload
        if priority is not None:
            entry.priority = priority
        if status is not None:
            entry.status = status

        self._update_next_run(entry)
        logger.info("Schedule %s updated", schedule_id)
        return True

    async def delete_schedule(self, schedule_id: str) -> bool:
        """Soft-delete a schedule.

        Args:
            schedule_id: Schedule to delete.

        Returns:
            True if found and deleted.
        """
        entry = self._schedules.get(schedule_id)
        if entry is None:
            return False
        entry.status = ScheduleStatus.DELETED
        self._cron_cache.pop(schedule_id, None)
        logger.info("Schedule %s deleted", schedule_id)
        return True

    async def get_schedule(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a schedule by ID.

        Args:
            schedule_id: Schedule to look up.

        Returns:
            Schedule dict or None.
        """
        entry = self._schedules.get(schedule_id)
        return entry.to_dict() if entry else None

    async def list_schedules(
        self,
        status_filter: Optional[ScheduleStatus] = None,
    ) -> List[Dict[str, Any]]:
        """List all schedules, optionally filtered by status.

        Args:
            status_filter: Optional status filter.

        Returns:
            List of schedule dicts.
        """
        results: List[Dict[str, Any]] = []
        for entry in self._schedules.values():
            if status_filter is not None and entry.status != status_filter:
                continue
            results.append(entry.to_dict())
        return results

    # ------------------------------------------------------------------
    # Event-driven triggers
    # ------------------------------------------------------------------

    async def register_event_trigger(
        self,
        topic: str,
        agent_key: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        transformer: Optional[EventTransformer] = None,
    ) -> str:
        """Register an event-driven trigger.

        When ``handle_event`` is called with a matching topic, a task is
        enqueued for the specified agent.

        Args:
            topic: Topic string to match.
            agent_key: Target agent.
            priority: Task priority.
            transformer: Optional payload transformer.

        Returns:
            Trigger ID.
        """
        trigger = EventTrigger(
            topic=topic,
            agent_key=agent_key,
            priority=priority,
            transformer=transformer,
        )
        self._event_triggers[trigger.id] = trigger
        logger.info(
            "Event trigger registered: %s (topic=%s, agent=%s)",
            trigger.id,
            topic,
            agent_key,
        )
        return trigger.id

    async def remove_event_trigger(self, trigger_id: str) -> bool:
        """Remove an event trigger.

        Args:
            trigger_id: Trigger to remove.

        Returns:
            True if found and removed.
        """
        return self._event_triggers.pop(trigger_id, None) is not None

    async def handle_event(
        self,
        topic: str,
        event_payload: Dict[str, Any],
    ) -> List[str]:
        """Process an incoming event, enqueuing tasks for matching triggers.

        Args:
            topic: Event topic.
            event_payload: Event data.

        Returns:
            List of enqueued task IDs.
        """
        task_ids: List[str] = []
        for trigger in self._event_triggers.values():
            if not trigger.active:
                continue
            if trigger.topic != topic:
                continue

            # Build payload
            if trigger.transformer is not None:
                payload = await trigger.transformer(event_payload)
            else:
                payload = dict(event_payload)

            task = TaskItem(
                agent_key=trigger.agent_key,
                payload=payload,
                priority=trigger.priority,
                metadata={"trigger_id": trigger.id, "event_topic": topic},
            )
            task_id = await self._queue.enqueue(task)
            task_ids.append(task_id)

        if task_ids:
            logger.info(
                "Event on '%s' spawned %d task(s)", topic, len(task_ids)
            )
        return task_ids

    # ------------------------------------------------------------------
    # Internal scheduling loop
    # ------------------------------------------------------------------

    async def _scheduling_loop(self) -> None:
        """Background loop that evaluates cron schedules."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                await self._evaluate_schedules(now)
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Scheduling loop error")
                await asyncio.sleep(self._check_interval)

    async def _evaluate_schedules(self, now: datetime) -> None:
        """Check all active schedules and enqueue due tasks."""
        for entry in list(self._schedules.values()):
            if entry.status != ScheduleStatus.ACTIVE:
                continue
            if entry.next_run_at is None:
                self._update_next_run(entry)
                continue

            next_dt = datetime.fromisoformat(entry.next_run_at)
            if now >= next_dt:
                await self._fire_schedule(entry)
                self._update_next_run(entry)

    async def _fire_schedule(self, entry: ScheduleEntry) -> None:
        """Enqueue a task for a due schedule entry."""
        task = TaskItem(
            agent_key=entry.agent_key,
            payload=dict(entry.payload),
            priority=entry.priority,
            ttl=entry.ttl,
            max_retries=entry.max_retries,
            metadata={
                "schedule_id": entry.id,
                "schedule_name": entry.name,
            },
        )
        try:
            await self._queue.enqueue(task)
            entry.run_count += 1
            entry.last_run_at = datetime.now(timezone.utc).isoformat()
            logger.debug(
                "Schedule '%s' fired (run #%d)", entry.name, entry.run_count
            )
        except Exception:
            logger.exception(
                "Failed to enqueue task for schedule '%s'", entry.name
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_next_run(self, entry: ScheduleEntry) -> None:
        """Recompute and set next_run_at on a schedule entry."""
        cron = self._cron_cache.get(entry.id)
        if cron is None:
            try:
                cron = CronExpression(entry.cron_expression)
                self._cron_cache[entry.id] = cron
            except ValueError:
                logger.error(
                    "Invalid cron expression for schedule %s: %s",
                    entry.id,
                    entry.cron_expression,
                )
                return

        after = datetime.now(timezone.utc)
        next_dt = cron.next_run(after)
        entry.next_run_at = next_dt.isoformat()

    def detect_missed_schedules(self) -> List[Dict[str, Any]]:
        """Return schedules whose next_run is in the past.

        This is useful on startup to detect schedules that were missed
        while the scheduler was offline.

        Returns:
            List of schedule dicts that are overdue.
        """
        now = datetime.now(timezone.utc)
        missed: List[Dict[str, Any]] = []
        for entry in self._schedules.values():
            if entry.status != ScheduleStatus.ACTIVE:
                continue
            if entry.next_run_at is None:
                continue
            next_dt = datetime.fromisoformat(entry.next_run_at)
            if next_dt < now:
                missed.append(entry.to_dict())
        return missed


__all__ = [
    "CronExpression",
    "EventTrigger",
    "ScheduleEntry",
    "ScheduleStatus",
    "TaskScheduler",
]
