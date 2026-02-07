# -*- coding: utf-8 -*-
"""
Agent-specific span types for the GreenLang Agent Factory.

Each span type carries domain-specific attributes and convenience methods
for recording events that are meaningful to agent operations (executions,
lifecycle transitions, queue processing, inter-agent messages).

SpanFactory provides a clean factory API so callers do not need to import
individual span classes directly.

Example:
    >>> span = SpanFactory.execution_span("carbon-calc", "2.1.0", "tenant-42")
    >>> span.record_input_hash("abc123")
    >>> span.end()

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base span mixin
# ---------------------------------------------------------------------------

@dataclass
class _BaseAgentSpan:
    """Common attributes shared by all agent span types.

    Attributes:
        span_name: Logical name for the span.
        start_time: UTC ISO-8601 start timestamp.
        attributes: Key-value attributes set on the span.
        events: List of recorded span events.
    """

    span_name: str = ""
    start_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: list[Dict[str, Any]] = field(default_factory=list)
    _end_time: Optional[str] = field(default=None, repr=False)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        """Record a timestamped event."""
        self.events.append({
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attributes": attrs or {},
        })

    def end(self) -> None:
        """Mark the span as ended."""
        self._end_time = datetime.now(timezone.utc).isoformat()

    @property
    def duration_ms(self) -> float:
        """Elapsed wall-clock time in milliseconds (0 if not ended)."""
        if self._end_time is None:
            return 0.0
        start = datetime.fromisoformat(self.start_time)
        end = datetime.fromisoformat(self._end_time)
        return (end - start).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the span to a plain dictionary."""
        return {
            "span_name": self.span_name,
            "start_time": self.start_time,
            "end_time": self._end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
        }


# ---------------------------------------------------------------------------
# Concrete span types
# ---------------------------------------------------------------------------

@dataclass
class AgentExecutionSpan(_BaseAgentSpan):
    """Span for a single agent execution invocation.

    Attributes:
        agent_key: Unique agent identifier.
        version: Agent version string.
        tenant_id: Tenant making the request.
        input_hash: SHA-256 of the input payload (for provenance).
    """

    agent_key: str = ""
    version: str = ""
    tenant_id: str = ""
    input_hash: str = ""

    def __post_init__(self) -> None:
        self.span_name = self.span_name or f"agent.execute.{self.agent_key}"
        self.set_attribute("agent.key", self.agent_key)
        self.set_attribute("agent.version", self.version)
        if self.tenant_id:
            self.set_attribute("tenant.id", self.tenant_id)

    def record_input_hash(self, raw_input: str) -> None:
        """Hash the raw input and store in the span."""
        self.input_hash = hashlib.sha256(raw_input.encode("utf-8")).hexdigest()
        self.set_attribute("agent.input_hash", self.input_hash)
        self.add_event("input_hashed", {"hash": self.input_hash})

    def record_output_size(self, size_bytes: int) -> None:
        """Record the output payload size."""
        self.set_attribute("agent.output_size_bytes", size_bytes)

    def record_cost(self, cost_usd: float) -> None:
        """Record the execution cost."""
        self.set_attribute("agent.cost_usd", cost_usd)


@dataclass
class LifecycleSpan(_BaseAgentSpan):
    """Span for an agent lifecycle state transition.

    Attributes:
        state_from: Previous lifecycle state.
        state_to: New lifecycle state.
        reason: Human-readable reason for the transition.
    """

    state_from: str = ""
    state_to: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        self.span_name = self.span_name or f"agent.lifecycle.{self.state_from}_to_{self.state_to}"
        self.set_attribute("lifecycle.state_from", self.state_from)
        self.set_attribute("lifecycle.state_to", self.state_to)
        self.set_attribute("lifecycle.reason", self.reason)

    def record_actor(self, actor: str) -> None:
        """Record who triggered the transition."""
        self.set_attribute("lifecycle.actor", actor)
        self.add_event("transition_triggered", {"actor": actor})


@dataclass
class QueueSpan(_BaseAgentSpan):
    """Span for a task's journey through the execution queue.

    Attributes:
        task_id: Unique task identifier.
        priority: Queue priority (0 = highest).
        wait_time_ms: Time spent waiting in the queue.
    """

    task_id: str = ""
    priority: int = 0
    wait_time_ms: float = 0.0

    def __post_init__(self) -> None:
        self.span_name = self.span_name or f"agent.queue.{self.task_id}"
        self.set_attribute("queue.task_id", self.task_id)
        self.set_attribute("queue.priority", self.priority)

    def record_enqueue(self) -> None:
        """Record the task being enqueued."""
        self.add_event("enqueued", {"priority": self.priority})

    def record_dequeue(self, wait_ms: float) -> None:
        """Record the task being dequeued with wait time."""
        self.wait_time_ms = wait_ms
        self.set_attribute("queue.wait_time_ms", wait_ms)
        self.add_event("dequeued", {"wait_time_ms": wait_ms})

    def record_dlq(self, reason: str) -> None:
        """Record the task being sent to the dead-letter queue."""
        self.set_attribute("queue.dlq", True)
        self.set_attribute("queue.dlq_reason", reason)
        self.add_event("sent_to_dlq", {"reason": reason})


@dataclass
class MessageSpan(_BaseAgentSpan):
    """Span for an inter-agent message.

    Attributes:
        message_id: Unique message identifier.
        source: Source agent key.
        target: Target agent key.
        message_type: Type of message (e.g. request, response, event).
    """

    message_id: str = ""
    source: str = ""
    target: str = ""
    message_type: str = ""

    def __post_init__(self) -> None:
        self.span_name = self.span_name or f"agent.message.{self.source}_to_{self.target}"
        self.set_attribute("message.id", self.message_id)
        self.set_attribute("message.source", self.source)
        self.set_attribute("message.target", self.target)
        self.set_attribute("message.type", self.message_type)

    def record_payload_size(self, size_bytes: int) -> None:
        """Record the message payload size."""
        self.set_attribute("message.payload_size_bytes", size_bytes)

    def record_delivery(self, success: bool) -> None:
        """Record whether the message was delivered successfully."""
        self.set_attribute("message.delivered", success)
        self.add_event("delivery_result", {"success": success})


# ---------------------------------------------------------------------------
# SpanFactory
# ---------------------------------------------------------------------------

class SpanFactory:
    """Factory for creating typed agent spans.

    Provides static methods so callers do not need to import individual
    span dataclasses.

    Example:
        >>> span = SpanFactory.execution_span("carbon-calc", "2.1.0")
    """

    @staticmethod
    def execution_span(
        agent_key: str,
        version: str,
        tenant_id: str = "",
    ) -> AgentExecutionSpan:
        """Create an execution span.

        Args:
            agent_key: Agent identifier.
            version: Agent version.
            tenant_id: Optional tenant.

        Returns:
            Initialized AgentExecutionSpan.
        """
        return AgentExecutionSpan(
            agent_key=agent_key,
            version=version,
            tenant_id=tenant_id,
        )

    @staticmethod
    def lifecycle_span(
        state_from: str,
        state_to: str,
        reason: str = "",
    ) -> LifecycleSpan:
        """Create a lifecycle transition span.

        Args:
            state_from: Previous state.
            state_to: Target state.
            reason: Transition reason.

        Returns:
            Initialized LifecycleSpan.
        """
        return LifecycleSpan(
            state_from=state_from,
            state_to=state_to,
            reason=reason,
        )

    @staticmethod
    def queue_span(
        task_id: str,
        priority: int = 0,
    ) -> QueueSpan:
        """Create a queue-processing span.

        Args:
            task_id: Task identifier.
            priority: Queue priority.

        Returns:
            Initialized QueueSpan.
        """
        return QueueSpan(task_id=task_id, priority=priority)

    @staticmethod
    def message_span(
        message_id: str,
        source: str,
        target: str,
        message_type: str = "request",
    ) -> MessageSpan:
        """Create an inter-agent message span.

        Args:
            message_id: Message identifier.
            source: Source agent.
            target: Target agent.
            message_type: Message type.

        Returns:
            Initialized MessageSpan.
        """
        return MessageSpan(
            message_id=message_id,
            source=source,
            target=target,
            message_type=message_type,
        )


__all__ = [
    "AgentExecutionSpan",
    "LifecycleSpan",
    "MessageSpan",
    "QueueSpan",
    "SpanFactory",
]
