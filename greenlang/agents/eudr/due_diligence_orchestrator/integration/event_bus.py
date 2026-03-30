# -*- coding: utf-8 -*-
"""
Event Bus Integration - AGENT-EUDR-026

In-process event bus for publishing and subscribing to workflow lifecycle
events. Supports synchronous and asynchronous event delivery, event
filtering by type, replay for late subscribers, and provenance tracking
for all published events.

Event Categories:
    Workflow Events:
        - workflow.created, workflow.started, workflow.completed
        - workflow.failed, workflow.cancelled, workflow.paused
        - workflow.resumed, workflow.terminated

    Agent Events:
        - agent.started, agent.completed, agent.failed
        - agent.retrying, agent.skipped, agent.timed_out
        - agent.circuit_broken

    Quality Gate Events:
        - quality_gate.evaluating, quality_gate.passed
        - quality_gate.failed, quality_gate.overridden

    Phase Events:
        - phase.started, phase.completed, phase.failed

    Package Events:
        - package.generating, package.completed, package.failed

Design:
    The event bus uses an in-process publish-subscribe pattern with
    optional persistence via callback hooks. For production deployments,
    the bus can be extended with Redis Pub/Sub or Kafka backends by
    registering external transport handlers.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set
from greenlang.schemas import utcnow

from greenlang.agents.eudr.due_diligence_orchestrator.config import (
    DueDiligenceOrchestratorConfig,
    get_config,
)
from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    _new_uuid,
    _utcnow,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------

# Workflow lifecycle events
WORKFLOW_CREATED = "workflow.created"
WORKFLOW_STARTED = "workflow.started"
WORKFLOW_COMPLETED = "workflow.completed"
WORKFLOW_FAILED = "workflow.failed"
WORKFLOW_CANCELLED = "workflow.cancelled"
WORKFLOW_PAUSED = "workflow.paused"
WORKFLOW_RESUMED = "workflow.resumed"
WORKFLOW_TERMINATED = "workflow.terminated"

# Agent execution events
AGENT_STARTED = "agent.started"
AGENT_COMPLETED = "agent.completed"
AGENT_FAILED = "agent.failed"
AGENT_RETRYING = "agent.retrying"
AGENT_SKIPPED = "agent.skipped"
AGENT_TIMED_OUT = "agent.timed_out"
AGENT_CIRCUIT_BROKEN = "agent.circuit_broken"

# Quality gate events
QUALITY_GATE_EVALUATING = "quality_gate.evaluating"
QUALITY_GATE_PASSED = "quality_gate.passed"
QUALITY_GATE_FAILED = "quality_gate.failed"
QUALITY_GATE_OVERRIDDEN = "quality_gate.overridden"

# Phase events
PHASE_STARTED = "phase.started"
PHASE_COMPLETED = "phase.completed"
PHASE_FAILED = "phase.failed"

# Package events
PACKAGE_GENERATING = "package.generating"
PACKAGE_COMPLETED = "package.completed"
PACKAGE_FAILED = "package.failed"

#: All recognized event types.
ALL_EVENT_TYPES: List[str] = [
    WORKFLOW_CREATED, WORKFLOW_STARTED, WORKFLOW_COMPLETED,
    WORKFLOW_FAILED, WORKFLOW_CANCELLED, WORKFLOW_PAUSED,
    WORKFLOW_RESUMED, WORKFLOW_TERMINATED,
    AGENT_STARTED, AGENT_COMPLETED, AGENT_FAILED,
    AGENT_RETRYING, AGENT_SKIPPED, AGENT_TIMED_OUT,
    AGENT_CIRCUIT_BROKEN,
    QUALITY_GATE_EVALUATING, QUALITY_GATE_PASSED,
    QUALITY_GATE_FAILED, QUALITY_GATE_OVERRIDDEN,
    PHASE_STARTED, PHASE_COMPLETED, PHASE_FAILED,
    PACKAGE_GENERATING, PACKAGE_COMPLETED, PACKAGE_FAILED,
]

#: Event type category prefixes.
EVENT_CATEGORIES: Dict[str, List[str]] = {
    "workflow": [
        WORKFLOW_CREATED, WORKFLOW_STARTED, WORKFLOW_COMPLETED,
        WORKFLOW_FAILED, WORKFLOW_CANCELLED, WORKFLOW_PAUSED,
        WORKFLOW_RESUMED, WORKFLOW_TERMINATED,
    ],
    "agent": [
        AGENT_STARTED, AGENT_COMPLETED, AGENT_FAILED,
        AGENT_RETRYING, AGENT_SKIPPED, AGENT_TIMED_OUT,
        AGENT_CIRCUIT_BROKEN,
    ],
    "quality_gate": [
        QUALITY_GATE_EVALUATING, QUALITY_GATE_PASSED,
        QUALITY_GATE_FAILED, QUALITY_GATE_OVERRIDDEN,
    ],
    "phase": [
        PHASE_STARTED, PHASE_COMPLETED, PHASE_FAILED,
    ],
    "package": [
        PACKAGE_GENERATING, PACKAGE_COMPLETED, PACKAGE_FAILED,
    ],
}


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------


class WorkflowEvent:
    """Represents a single workflow lifecycle event.

    Attributes:
        event_id: Unique event identifier (UUID4).
        event_type: Event type string (e.g., "agent.completed").
        workflow_id: Associated workflow identifier.
        timestamp: UTC timestamp of event creation.
        data: Event-specific payload data.
        provenance_hash: SHA-256 hash of the event content.
    """

    __slots__ = (
        "event_id", "event_type", "workflow_id", "timestamp",
        "data", "provenance_hash",
    )

    def __init__(
        self,
        event_type: str,
        workflow_id: str,
        data: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Initialize a WorkflowEvent.

        Args:
            event_type: Event type string.
            workflow_id: Associated workflow identifier.
            data: Optional event payload data.
            event_id: Optional explicit event ID.
            timestamp: Optional explicit timestamp.
        """
        self.event_id = event_id or _new_uuid()
        self.event_type = event_type
        self.workflow_id = workflow_id
        self.timestamp = timestamp or utcnow()
        self.data = data or {}
        self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of the event content.

        Returns:
            64-character hex SHA-256 hash.
        """
        payload = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "workflow_id": self.workflow_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary.

        Returns:
            Dictionary representation of the event.
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "workflow_id": self.workflow_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "provenance_hash": self.provenance_hash,
        }

    def __repr__(self) -> str:
        """Return string representation of the event."""
        return (
            f"WorkflowEvent("
            f"event_type={self.event_type!r}, "
            f"workflow_id={self.workflow_id!r}, "
            f"event_id={self.event_id!r})"
        )


# ---------------------------------------------------------------------------
# Subscriber type alias
# ---------------------------------------------------------------------------

#: Type alias for event handler callback functions.
EventHandler = Callable[[WorkflowEvent], None]


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class EventBus:
    """In-process event bus for workflow lifecycle events.

    Supports publish-subscribe pattern with event type filtering,
    workflow-scoped subscriptions, event history replay, and
    thread-safe operation.

    Attributes:
        _config: Orchestrator configuration.
        _subscribers: Map of event_type to list of handlers.
        _wildcard_subscribers: Handlers subscribed to all events.
        _workflow_subscribers: Map of (workflow_id, event_type) to handlers.
        _history: Ordered list of published events for replay.
        _max_history: Maximum events to retain in history.
        _lock: Thread lock for concurrent access safety.

    Example:
        >>> bus = EventBus()
        >>> bus.subscribe("agent.completed", my_handler)
        >>> bus.publish(WorkflowEvent(
        ...     event_type="agent.completed",
        ...     workflow_id="WF-001",
        ...     data={"agent_id": "EUDR-016"}
        ... ))
    """

    def __init__(
        self,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
        max_history: int = 10000,
    ) -> None:
        """Initialize the EventBus.

        Args:
            config: Optional configuration override.
            max_history: Maximum events to retain in history buffer.
        """
        self._config = config or get_config()
        self._subscribers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._wildcard_subscribers: List[EventHandler] = []
        self._workflow_subscribers: Dict[
            str, Dict[str, List[EventHandler]]
        ] = defaultdict(lambda: defaultdict(list))
        self._history: List[WorkflowEvent] = []
        self._max_history = max_history
        self._lock = threading.Lock()

        logger.info(
            f"EventBus initialized (max_history={max_history})"
        )

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def publish(self, event: WorkflowEvent) -> int:
        """Publish an event to all matching subscribers.

        Args:
            event: WorkflowEvent to publish.

        Returns:
            Number of handlers that received the event.
        """
        handler_count = 0

        with self._lock:
            # Store in history
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            # Collect all applicable handlers
            handlers: List[EventHandler] = []

            # Type-specific subscribers
            handlers.extend(self._subscribers.get(event.event_type, []))

            # Wildcard subscribers
            handlers.extend(self._wildcard_subscribers)

            # Workflow-scoped subscribers
            wf_subs = self._workflow_subscribers.get(event.workflow_id, {})
            handlers.extend(wf_subs.get(event.event_type, []))
            handlers.extend(wf_subs.get("*", []))

        # Dispatch outside the lock to prevent deadlocks
        for handler in handlers:
            try:
                handler(event)
                handler_count += 1
            except Exception as e:
                logger.error(
                    f"Event handler error for {event.event_type}: "
                    f"{type(e).__name__}: {str(e)[:200]}"
                )

        logger.debug(
            f"Published {event.event_type} for workflow "
            f"{event.workflow_id} to {handler_count} handlers"
        )

        return handler_count

    def publish_event(
        self,
        event_type: str,
        workflow_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> WorkflowEvent:
        """Create and publish an event in one call.

        Args:
            event_type: Event type string.
            workflow_id: Associated workflow identifier.
            data: Optional event payload data.

        Returns:
            The created and published WorkflowEvent.
        """
        event = WorkflowEvent(
            event_type=event_type,
            workflow_id=workflow_id,
            data=data,
        )
        self.publish(event)
        return event

    # ------------------------------------------------------------------
    # Subscribe
    # ------------------------------------------------------------------

    def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
    ) -> None:
        """Subscribe a handler to a specific event type.

        Args:
            event_type: Event type to subscribe to, or "*" for all.
            handler: Callback function to invoke on event.
        """
        with self._lock:
            if event_type == "*":
                self._wildcard_subscribers.append(handler)
            else:
                self._subscribers[event_type].append(handler)

        logger.debug(f"Subscribed handler to {event_type}")

    def subscribe_workflow(
        self,
        workflow_id: str,
        event_type: str,
        handler: EventHandler,
    ) -> None:
        """Subscribe a handler scoped to a specific workflow.

        Args:
            workflow_id: Workflow to scope the subscription to.
            event_type: Event type to subscribe to, or "*" for all.
            handler: Callback function to invoke on event.
        """
        with self._lock:
            self._workflow_subscribers[workflow_id][event_type].append(
                handler
            )

        logger.debug(
            f"Subscribed workflow-scoped handler to {event_type} "
            f"for workflow {workflow_id}"
        )

    def subscribe_category(
        self,
        category: str,
        handler: EventHandler,
    ) -> None:
        """Subscribe a handler to all events in a category.

        Args:
            category: Event category (workflow, agent, quality_gate,
                      phase, package).
            handler: Callback function to invoke on event.

        Raises:
            ValueError: If category is not recognized.
        """
        event_types = EVENT_CATEGORIES.get(category)
        if event_types is None:
            raise ValueError(
                f"Unknown event category '{category}'. "
                f"Valid: {sorted(EVENT_CATEGORIES.keys())}"
            )
        for event_type in event_types:
            self.subscribe(event_type, handler)

    # ------------------------------------------------------------------
    # Unsubscribe
    # ------------------------------------------------------------------

    def unsubscribe(
        self,
        event_type: str,
        handler: EventHandler,
    ) -> bool:
        """Unsubscribe a handler from an event type.

        Args:
            event_type: Event type to unsubscribe from.
            handler: Handler to remove.

        Returns:
            True if the handler was found and removed.
        """
        with self._lock:
            if event_type == "*":
                try:
                    self._wildcard_subscribers.remove(handler)
                    return True
                except ValueError:
                    return False
            else:
                handlers = self._subscribers.get(event_type, [])
                try:
                    handlers.remove(handler)
                    return True
                except ValueError:
                    return False

    def unsubscribe_workflow(self, workflow_id: str) -> int:
        """Remove all subscriptions for a specific workflow.

        Args:
            workflow_id: Workflow to clear subscriptions for.

        Returns:
            Number of subscription entries removed.
        """
        with self._lock:
            if workflow_id in self._workflow_subscribers:
                count = sum(
                    len(handlers)
                    for handlers in
                    self._workflow_subscribers[workflow_id].values()
                )
                del self._workflow_subscribers[workflow_id]
                return count
            return 0

    # ------------------------------------------------------------------
    # History and replay
    # ------------------------------------------------------------------

    def get_history(
        self,
        workflow_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[WorkflowEvent]:
        """Get event history with optional filtering.

        Args:
            workflow_id: Optional workflow filter.
            event_type: Optional event type filter.
            limit: Maximum number of events to return.

        Returns:
            List of matching events, most recent first.
        """
        with self._lock:
            events = list(self._history)

        # Apply filters
        if workflow_id:
            events = [e for e in events if e.workflow_id == workflow_id]
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # Return most recent first, limited
        return list(reversed(events[-limit:]))

    def replay(
        self,
        handler: EventHandler,
        workflow_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> int:
        """Replay historical events to a handler.

        Useful for late subscribers who need to catch up on past events.

        Args:
            handler: Handler to replay events to.
            workflow_id: Optional workflow filter.
            event_type: Optional event type filter.

        Returns:
            Number of events replayed.
        """
        events = self.get_history(
            workflow_id=workflow_id,
            event_type=event_type,
            limit=self._max_history,
        )

        # Replay in chronological order (reverse the most-recent-first)
        replayed = 0
        for event in reversed(events):
            try:
                handler(event)
                replayed += 1
            except Exception as e:
                logger.error(
                    f"Replay handler error for {event.event_type}: "
                    f"{type(e).__name__}: {str(e)[:200]}"
                )

        logger.info(
            f"Replayed {replayed} events "
            f"(workflow={workflow_id}, type={event_type})"
        )
        return replayed

    # ------------------------------------------------------------------
    # Workflow event counts
    # ------------------------------------------------------------------

    def get_event_counts(
        self, workflow_id: str
    ) -> Dict[str, int]:
        """Get event type counts for a specific workflow.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            Dictionary mapping event_type to count.
        """
        counts: Dict[str, int] = defaultdict(int)
        with self._lock:
            for event in self._history:
                if event.workflow_id == workflow_id:
                    counts[event.event_type] += 1
        return dict(counts)

    def get_workflow_timeline(
        self, workflow_id: str
    ) -> List[Dict[str, Any]]:
        """Get chronological event timeline for a workflow.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            Ordered list of event summary dictionaries.
        """
        with self._lock:
            events = [
                e for e in self._history
                if e.workflow_id == workflow_id
            ]

        return [
            {
                "event_id": e.event_id,
                "event_type": e.event_type,
                "timestamp": e.timestamp.isoformat(),
                "agent_id": e.data.get("agent_id", ""),
                "status": e.data.get("status", ""),
            }
            for e in events
        ]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear_history(
        self, workflow_id: Optional[str] = None
    ) -> int:
        """Clear event history.

        Args:
            workflow_id: If provided, clear only this workflow's events.

        Returns:
            Number of events removed.
        """
        with self._lock:
            if workflow_id:
                before = len(self._history)
                self._history = [
                    e for e in self._history
                    if e.workflow_id != workflow_id
                ]
                removed = before - len(self._history)
            else:
                removed = len(self._history)
                self._history.clear()

        logger.info(f"Cleared {removed} events from history")
        return removed

    def clear_all(self) -> None:
        """Clear all subscriptions and history.

        Resets the event bus to its initial state.
        """
        with self._lock:
            self._subscribers.clear()
            self._wildcard_subscribers.clear()
            self._workflow_subscribers.clear()
            self._history.clear()

        logger.info("EventBus cleared: all subscriptions and history removed")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics.

        Returns:
            Dictionary with subscriber and history stats.
        """
        with self._lock:
            type_sub_count = sum(
                len(handlers)
                for handlers in self._subscribers.values()
            )
            wildcard_count = len(self._wildcard_subscribers)
            wf_sub_count = sum(
                len(handlers)
                for wf_subs in self._workflow_subscribers.values()
                for handlers in wf_subs.values()
            )
            history_count = len(self._history)

        return {
            "type_subscribers": type_sub_count,
            "wildcard_subscribers": wildcard_count,
            "workflow_subscribers": wf_sub_count,
            "total_subscribers": (
                type_sub_count + wildcard_count + wf_sub_count
            ),
            "history_count": history_count,
            "max_history": self._max_history,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_event_bus_instance: Optional[EventBus] = None
_event_bus_lock = threading.Lock()


def get_event_bus(
    config: Optional[DueDiligenceOrchestratorConfig] = None,
) -> EventBus:
    """Get or create the singleton EventBus instance.

    Args:
        config: Optional configuration override.

    Returns:
        Singleton EventBus instance.
    """
    global _event_bus_instance

    if _event_bus_instance is None:
        with _event_bus_lock:
            if _event_bus_instance is None:
                _event_bus_instance = EventBus(config)
    return _event_bus_instance


def reset_event_bus() -> None:
    """Reset the singleton EventBus instance.

    Creates a fresh instance on next access.
    """
    global _event_bus_instance

    with _event_bus_lock:
        if _event_bus_instance is not None:
            _event_bus_instance.clear_all()
        _event_bus_instance = None

    logger.info("EventBus singleton reset")
