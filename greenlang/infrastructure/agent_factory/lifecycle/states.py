# -*- coding: utf-8 -*-
"""
Agent State Machine - Lifecycle state management for GreenLang agents.

This module implements a deterministic finite state machine for tracking
agent lifecycle states, validating transitions, maintaining transition
history, and dispatching event callbacks on state changes.

States:
    CREATED -> VALIDATING -> VALIDATED -> DEPLOYING -> WARMING_UP ->
    RUNNING -> DRAINING -> RETIRED
    Any state -> FAILED | FORCE_STOPPED
    RUNNING -> DEGRADED -> RUNNING | DRAINING

Example:
    >>> machine = AgentStateMachine(initial_state=AgentState.CREATED)
    >>> machine.on_transition(my_callback)
    >>> machine.transition(AgentState.VALIDATING, reason="startup", actor="system")
    >>> machine.current_state
    AgentState.VALIDATING

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """Enumeration of all valid agent lifecycle states."""

    CREATED = "created"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYING = "deploying"
    WARMING_UP = "warming_up"
    RUNNING = "running"
    DEGRADED = "degraded"
    DRAINING = "draining"
    RETIRED = "retired"
    FAILED = "failed"
    FORCE_STOPPED = "force_stopped"


# ---------------------------------------------------------------------------
# Valid state transitions map
# ---------------------------------------------------------------------------

VALID_TRANSITIONS: Dict[AgentState, FrozenSet[AgentState]] = {
    AgentState.CREATED: frozenset({
        AgentState.VALIDATING,
        AgentState.FAILED,
        AgentState.FORCE_STOPPED,
    }),
    AgentState.VALIDATING: frozenset({
        AgentState.VALIDATED,
        AgentState.FAILED,
        AgentState.FORCE_STOPPED,
    }),
    AgentState.VALIDATED: frozenset({
        AgentState.DEPLOYING,
        AgentState.FAILED,
        AgentState.FORCE_STOPPED,
    }),
    AgentState.DEPLOYING: frozenset({
        AgentState.WARMING_UP,
        AgentState.FAILED,
        AgentState.FORCE_STOPPED,
    }),
    AgentState.WARMING_UP: frozenset({
        AgentState.RUNNING,
        AgentState.FAILED,
        AgentState.FORCE_STOPPED,
    }),
    AgentState.RUNNING: frozenset({
        AgentState.DEGRADED,
        AgentState.DRAINING,
        AgentState.FAILED,
        AgentState.FORCE_STOPPED,
    }),
    AgentState.DEGRADED: frozenset({
        AgentState.RUNNING,
        AgentState.DRAINING,
        AgentState.FAILED,
        AgentState.FORCE_STOPPED,
    }),
    AgentState.DRAINING: frozenset({
        AgentState.RETIRED,
        AgentState.FAILED,
        AgentState.FORCE_STOPPED,
    }),
    AgentState.RETIRED: frozenset(),
    AgentState.FAILED: frozenset({
        AgentState.CREATED,
        AgentState.FORCE_STOPPED,
    }),
    AgentState.FORCE_STOPPED: frozenset(),
}


@dataclass(frozen=True)
class AgentStateTransition:
    """Immutable record of a single state transition.

    Attributes:
        from_state: State the agent was in before the transition.
        to_state: State the agent moved to after the transition.
        timestamp: UTC ISO-8601 timestamp of the transition.
        reason: Human-readable reason for the transition.
        actor: Identifier of the entity that triggered the change.
        metadata: Optional extra context about the transition.
    """

    from_state: AgentState
    to_state: AgentState
    timestamp: str
    reason: str
    actor: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(
        self,
        from_state: AgentState,
        to_state: AgentState,
        message: Optional[str] = None,
    ) -> None:
        self.from_state = from_state
        self.to_state = to_state
        default_msg = (
            f"Invalid transition from {from_state.value} to {to_state.value}"
        )
        super().__init__(message or default_msg)


# Type alias for transition callbacks
TransitionCallback = Callable[[AgentStateTransition], Any]


class AgentStateMachine:
    """Deterministic finite state machine for agent lifecycle management.

    Maintains the current state, validates transitions against the
    allowed-transition map, persists a full history of transitions, and
    notifies registered callbacks whenever a transition occurs.

    Attributes:
        current_state: The current lifecycle state of the agent.
        history: Ordered list of all transitions that have occurred.

    Example:
        >>> sm = AgentStateMachine()
        >>> sm.transition(AgentState.VALIDATING, reason="boot", actor="mgr")
        >>> sm.current_state
        AgentState.VALIDATING
        >>> len(sm.history)
        1
    """

    def __init__(
        self,
        initial_state: AgentState = AgentState.CREATED,
        max_history: int = 500,
    ) -> None:
        """Initialize the state machine.

        Args:
            initial_state: Starting state (default CREATED).
            max_history: Maximum number of transition records to keep.
        """
        self._state: AgentState = initial_state
        self._callbacks: List[TransitionCallback] = []
        self._history: List[AgentStateTransition] = []
        self._max_history = max_history
        self._created_at: str = datetime.now(timezone.utc).isoformat()
        logger.debug("AgentStateMachine created in state %s", initial_state.value)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def current_state(self) -> AgentState:
        """Return the current state."""
        return self._state

    @property
    def history(self) -> List[AgentStateTransition]:
        """Return a copy of the full transition history."""
        return list(self._history)

    @property
    def created_at(self) -> str:
        """Return the UTC ISO-8601 timestamp of creation."""
        return self._created_at

    # ------------------------------------------------------------------
    # Transition validation helpers
    # ------------------------------------------------------------------

    def can_transition(self, target: AgentState) -> bool:
        """Check whether transitioning to *target* is valid.

        Args:
            target: Candidate target state.

        Returns:
            True if the transition is allowed.
        """
        allowed = VALID_TRANSITIONS.get(self._state, frozenset())
        return target in allowed

    def get_valid_transitions(self) -> FrozenSet[AgentState]:
        """Return the set of states reachable from the current state."""
        return VALID_TRANSITIONS.get(self._state, frozenset())

    # ------------------------------------------------------------------
    # State transition
    # ------------------------------------------------------------------

    def transition(
        self,
        target: AgentState,
        *,
        reason: str = "",
        actor: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentStateTransition:
        """Execute a state transition.

        Args:
            target: Desired new state.
            reason: Human-readable reason for the change.
            actor: Identifier of the caller triggering the change.
            metadata: Optional context dictionary attached to the record.

        Returns:
            The AgentStateTransition record created by this change.

        Raises:
            InvalidTransitionError: If the transition is not valid.
        """
        if not self.can_transition(target):
            raise InvalidTransitionError(self._state, target)

        record = AgentStateTransition(
            from_state=self._state,
            to_state=target,
            timestamp=datetime.now(timezone.utc).isoformat(),
            reason=reason,
            actor=actor,
            metadata=metadata or {},
        )

        previous = self._state
        self._state = target

        # Store history (with bounded size)
        self._history.append(record)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        logger.info(
            "State transition: %s -> %s (reason=%s, actor=%s)",
            previous.value,
            target.value,
            reason,
            actor,
        )

        # Dispatch callbacks (best-effort, never block the transition)
        self._dispatch_callbacks(record)

        return record

    # ------------------------------------------------------------------
    # Callback management
    # ------------------------------------------------------------------

    def on_transition(self, callback: TransitionCallback) -> None:
        """Register a callback invoked after every state transition.

        Args:
            callback: A callable accepting an AgentStateTransition.
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: TransitionCallback) -> bool:
        """Remove a previously registered callback.

        Args:
            callback: The callback to remove.

        Returns:
            True if the callback was found and removed.
        """
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def is_terminal(self) -> bool:
        """Return True if the current state has no valid outgoing transitions."""
        return len(self.get_valid_transitions()) == 0

    def is_active(self) -> bool:
        """Return True if the agent is in an operational state."""
        return self._state in {
            AgentState.RUNNING,
            AgentState.DEGRADED,
            AgentState.WARMING_UP,
        }

    def time_in_current_state_seconds(self) -> float:
        """Return wall-clock seconds spent in the current state."""
        if not self._history:
            created = datetime.fromisoformat(self._created_at)
            return (datetime.now(timezone.utc) - created).total_seconds()
        last_ts = datetime.fromisoformat(self._history[-1].timestamp)
        return (datetime.now(timezone.utc) - last_ts).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the state machine to a plain dictionary."""
        return {
            "current_state": self._state.value,
            "created_at": self._created_at,
            "transition_count": len(self._history),
            "is_terminal": self.is_terminal(),
            "valid_transitions": [s.value for s in self.get_valid_transitions()],
            "history": [
                {
                    "from": t.from_state.value,
                    "to": t.to_state.value,
                    "timestamp": t.timestamp,
                    "reason": t.reason,
                    "actor": t.actor,
                }
                for t in self._history[-10:]  # last 10 for summary
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dispatch_callbacks(self, record: AgentStateTransition) -> None:
        """Invoke registered callbacks with best-effort error handling."""
        for cb in self._callbacks:
            try:
                cb(record)
            except Exception:
                logger.exception(
                    "Transition callback %s raised an exception", cb
                )


__all__ = [
    "AgentState",
    "AgentStateMachine",
    "AgentStateTransition",
    "InvalidTransitionError",
    "TransitionCallback",
    "VALID_TRANSITIONS",
]
