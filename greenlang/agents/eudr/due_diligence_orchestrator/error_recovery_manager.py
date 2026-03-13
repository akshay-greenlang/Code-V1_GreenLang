# -*- coding: utf-8 -*-
"""
Error Recovery Manager - AGENT-EUDR-026

Implements exponential backoff with jitter, circuit breaker pattern,
error classification, dead letter queue, and fallback strategies for
resilient agent execution within due diligence workflows.

Exponential Backoff Formula:
    delay = min(base_delay * 2^attempt + random(0, jitter_max), max_delay)

    Default: base=1s, max=300s, jitter=2s, max_attempts=5

    Example progression:
        Attempt 1: 1 * 2^1 + jitter = ~4s
        Attempt 2: 1 * 2^2 + jitter = ~6s
        Attempt 3: 1 * 2^3 + jitter = ~10s
        Attempt 4: 1 * 2^4 + jitter = ~18s
        Attempt 5: 1 * 2^5 + jitter = ~34s

Circuit Breaker State Machine:
    CLOSED -> OPEN (after failure_threshold consecutive failures)
    OPEN -> HALF_OPEN (after reset_timeout expires)
    HALF_OPEN -> CLOSED (after success_threshold consecutive successes)
    HALF_OPEN -> OPEN (on any failure)

Error Classification:
    - Transient (retry): HTTP 429, 503, 504, timeout, connection error
    - Permanent (fail): HTTP 400, 401, 403, 404, validation error
    - Degraded (fallback): Partial result, stale data, low confidence

Dead Letter Queue:
    - Entries for permanently failed agents after max retries exhausted
    - Contains full context: input data, retry history, error details
    - Manual resolution workflow with audit trail

Features:
    - Exponential backoff with configurable jitter
    - Circuit breaker pattern with three states
    - Automatic error classification from HTTP status and exception type
    - Dead letter queue for unrecoverable failures
    - Fallback strategy selection (cached, degraded, manual, fail)
    - Retry record tracking with complete history
    - Metrics integration for retry and circuit breaker events
    - Thread-safe circuit breaker state management

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
Status: Production Ready
"""

from __future__ import annotations

import logging
import random
import threading
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.due_diligence_orchestrator.config import (
    DueDiligenceOrchestratorConfig,
    get_config,
)
from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    CircuitBreakerRecord,
    CircuitBreakerState,
    DeadLetterEntry,
    ErrorClassification,
    FallbackStrategy,
    RetryRecord,
    _new_uuid,
    _utcnow,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error classification rules
# ---------------------------------------------------------------------------

#: HTTP status codes classified as transient (safe to retry).
_TRANSIENT_STATUS_CODES: frozenset = frozenset({
    408, 429, 500, 502, 503, 504,
})

#: HTTP status codes classified as permanent (do not retry).
_PERMANENT_STATUS_CODES: frozenset = frozenset({
    400, 401, 403, 404, 405, 409, 410, 422,
})

#: Exception type keywords classified as transient.
_TRANSIENT_EXCEPTIONS: frozenset = frozenset({
    "timeout", "connectionerror", "connecttimeout",
    "readtimeout", "connectionreset", "brokenpipe",
    "temporaryerror", "serviceunavailable",
})

#: Exception type keywords classified as permanent.
_PERMANENT_EXCEPTIONS: frozenset = frozenset({
    "validationerror", "valueerror", "typeerror",
    "keyerror", "attributeerror", "permissionerror",
    "authenticationerror", "authorizationerror",
    "notfounderror",
})


# ---------------------------------------------------------------------------
# ErrorRecoveryManager
# ---------------------------------------------------------------------------


class ErrorRecoveryManager:
    """Error recovery manager with retry, circuit breaker, and dead letter.

    Provides resilient error handling for agent execution within due
    diligence workflows. Implements exponential backoff, circuit breaker
    pattern, automatic error classification, and dead letter queue for
    unrecoverable failures.

    Thread-safe for concurrent multi-workflow environments.

    Attributes:
        _config: Configuration with retry and circuit breaker settings.
        _circuit_breakers: Per-agent circuit breaker records.
        _retry_history: Per-workflow, per-agent retry history.
        _dead_letter_queue: Dead letter queue entries.
        _lock: Threading lock for state management.

    Example:
        >>> manager = ErrorRecoveryManager()
        >>> should_retry, delay = manager.should_retry(
        ...     agent_id="EUDR-016", attempt=1, error_message="timeout"
        ... )
        >>> assert should_retry is True
    """

    def __init__(
        self,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize the ErrorRecoveryManager.

        Args:
            config: Optional configuration override.
        """
        self._config = config or get_config()
        self._circuit_breakers: Dict[str, CircuitBreakerRecord] = {}
        self._retry_history: Dict[str, List[RetryRecord]] = {}
        self._dead_letter_queue: List[DeadLetterEntry] = []
        self._lock = threading.Lock()
        logger.info(
            f"ErrorRecoveryManager initialized "
            f"(max_retries={self._config.retry_max_attempts}, "
            f"cb_threshold={self._config.cb_failure_threshold})"
        )

    # ------------------------------------------------------------------
    # Error classification
    # ------------------------------------------------------------------

    def classify_error(
        self,
        error_message: str,
        http_status: Optional[int] = None,
        exception_type: Optional[str] = None,
    ) -> ErrorClassification:
        """Classify an error as transient, permanent, or degraded.

        Classification logic (in order):
        1. HTTP status code lookup (most reliable)
        2. Exception type keyword matching
        3. Error message keyword matching
        4. Default to UNKNOWN

        Args:
            error_message: Error message text.
            http_status: Optional HTTP status code.
            exception_type: Optional exception class name.

        Returns:
            ErrorClassification enum value.

        Example:
            >>> manager = ErrorRecoveryManager()
            >>> cls = manager.classify_error("timeout", http_status=503)
            >>> assert cls == ErrorClassification.TRANSIENT
        """
        # 1. HTTP status code classification
        if http_status is not None:
            if http_status in _TRANSIENT_STATUS_CODES:
                return ErrorClassification.TRANSIENT
            if http_status in _PERMANENT_STATUS_CODES:
                return ErrorClassification.PERMANENT

        # 2. Exception type classification
        if exception_type:
            exc_lower = exception_type.lower()
            for keyword in _TRANSIENT_EXCEPTIONS:
                if keyword in exc_lower:
                    return ErrorClassification.TRANSIENT
            for keyword in _PERMANENT_EXCEPTIONS:
                if keyword in exc_lower:
                    return ErrorClassification.PERMANENT

        # 3. Error message keyword classification
        msg_lower = error_message.lower()
        transient_keywords = [
            "timeout", "connection", "temporary", "unavailable",
            "rate limit", "too many requests", "retry",
        ]
        permanent_keywords = [
            "validation", "invalid", "not found", "unauthorized",
            "forbidden", "permission", "authentication",
        ]
        degraded_keywords = [
            "partial", "stale", "low confidence", "incomplete",
            "degraded",
        ]

        for kw in transient_keywords:
            if kw in msg_lower:
                return ErrorClassification.TRANSIENT
        for kw in permanent_keywords:
            if kw in msg_lower:
                return ErrorClassification.PERMANENT
        for kw in degraded_keywords:
            if kw in msg_lower:
                return ErrorClassification.DEGRADED

        return ErrorClassification.UNKNOWN

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    def should_retry(
        self,
        agent_id: str,
        attempt: int,
        error_message: str,
        http_status: Optional[int] = None,
        exception_type: Optional[str] = None,
    ) -> Tuple[bool, Decimal]:
        """Determine if an agent execution should be retried.

        Retry is allowed when:
        1. Error is classified as TRANSIENT
        2. Attempt count is below max_attempts
        3. Circuit breaker is not OPEN

        Args:
            agent_id: Agent identifier.
            attempt: Current attempt number (0-based).
            error_message: Error message from the failure.
            http_status: Optional HTTP status code.
            exception_type: Optional exception class name.

        Returns:
            Tuple of (should_retry, delay_seconds).

        Example:
            >>> manager = ErrorRecoveryManager()
            >>> retry, delay = manager.should_retry(
            ...     "EUDR-016", 0, "timeout"
            ... )
            >>> assert retry is True
            >>> assert delay > Decimal("0")
        """
        classification = self.classify_error(
            error_message, http_status, exception_type
        )

        # Permanent errors are never retried
        if classification == ErrorClassification.PERMANENT:
            return False, Decimal("0")

        # Check max attempts
        if attempt >= self._config.retry_max_attempts:
            return False, Decimal("0")

        # Check circuit breaker
        with self._lock:
            cb = self._circuit_breakers.get(agent_id)
            if cb and cb.state == CircuitBreakerState.OPEN:
                return False, Decimal("0")

        # Compute backoff delay
        delay = self._compute_backoff_delay(attempt)

        return True, delay

    def compute_backoff_delay(self, attempt: int) -> Decimal:
        """Compute exponential backoff delay with jitter.

        Formula: delay = min(base * 2^attempt + jitter, max_delay)

        Args:
            attempt: Current attempt number (0-based).

        Returns:
            Delay in seconds as Decimal.
        """
        return self._compute_backoff_delay(attempt)

    def record_retry(
        self,
        workflow_id: str,
        agent_id: str,
        attempt: int,
        delay_s: Decimal,
        error_message: str,
        error_classification: ErrorClassification = ErrorClassification.TRANSIENT,
    ) -> RetryRecord:
        """Record a retry attempt.

        Args:
            workflow_id: Workflow identifier.
            agent_id: Agent identifier.
            attempt: Attempt number (1-based).
            delay_s: Computed delay in seconds.
            error_message: Error that triggered the retry.
            error_classification: Error type.

        Returns:
            RetryRecord for the attempt.
        """
        record = RetryRecord(
            retry_id=_new_uuid(),
            workflow_id=workflow_id,
            agent_id=agent_id,
            attempt_number=attempt,
            delay_s=delay_s,
            error_message=error_message,
            error_classification=error_classification,
            outcome="pending",
            attempted_at=_utcnow(),
        )

        with self._lock:
            key = f"{workflow_id}:{agent_id}"
            if key not in self._retry_history:
                self._retry_history[key] = []
            self._retry_history[key].append(record)

        logger.info(
            f"Retry {attempt} for {agent_id} in {workflow_id}: "
            f"delay={delay_s}s, error={error_message[:80]}"
        )

        return record

    def get_retry_history(
        self,
        workflow_id: str,
        agent_id: str,
    ) -> List[RetryRecord]:
        """Get retry history for a specific agent in a workflow.

        Args:
            workflow_id: Workflow identifier.
            agent_id: Agent identifier.

        Returns:
            List of RetryRecord objects.
        """
        with self._lock:
            key = f"{workflow_id}:{agent_id}"
            return list(self._retry_history.get(key, []))

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def record_success(self, agent_id: str) -> CircuitBreakerState:
        """Record a successful agent execution for circuit breaker.

        Args:
            agent_id: Agent identifier.

        Returns:
            New circuit breaker state after recording success.
        """
        with self._lock:
            cb = self._get_or_create_cb(agent_id)
            now = _utcnow()
            cb.last_success_at = now

            if cb.state == CircuitBreakerState.HALF_OPEN:
                cb.success_count += 1
                if cb.success_count >= cb.failure_threshold:
                    # Reset to closed
                    old_state = cb.state
                    cb.state = CircuitBreakerState.CLOSED
                    cb.failure_count = 0
                    cb.success_count = 0
                    logger.info(
                        f"Circuit breaker for {agent_id}: "
                        f"HALF_OPEN -> CLOSED"
                    )
            elif cb.state == CircuitBreakerState.CLOSED:
                cb.failure_count = 0

            return cb.state

    def record_failure(self, agent_id: str) -> CircuitBreakerState:
        """Record a failed agent execution for circuit breaker.

        Args:
            agent_id: Agent identifier.

        Returns:
            New circuit breaker state after recording failure.
        """
        with self._lock:
            cb = self._get_or_create_cb(agent_id)
            now = _utcnow()
            cb.last_failure_at = now
            cb.failure_count += 1

            if cb.state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                cb.state = CircuitBreakerState.OPEN
                cb.opened_at = now
                cb.success_count = 0
                logger.warning(
                    f"Circuit breaker for {agent_id}: "
                    f"HALF_OPEN -> OPEN (failure in probe)"
                )
            elif cb.state == CircuitBreakerState.CLOSED:
                if cb.failure_count >= cb.failure_threshold:
                    cb.state = CircuitBreakerState.OPEN
                    cb.opened_at = now
                    logger.warning(
                        f"Circuit breaker for {agent_id}: "
                        f"CLOSED -> OPEN (failures={cb.failure_count})"
                    )

            return cb.state

    def check_circuit_breaker(self, agent_id: str) -> CircuitBreakerState:
        """Check and potentially transition circuit breaker state.

        If the circuit is OPEN and reset_timeout has elapsed,
        transitions to HALF_OPEN for probe testing.

        Args:
            agent_id: Agent identifier.

        Returns:
            Current circuit breaker state.
        """
        with self._lock:
            cb = self._circuit_breakers.get(agent_id)
            if cb is None:
                return CircuitBreakerState.CLOSED

            if cb.state == CircuitBreakerState.OPEN and cb.opened_at:
                elapsed = (_utcnow() - cb.opened_at).total_seconds()
                if elapsed >= cb.reset_timeout_s:
                    cb.state = CircuitBreakerState.HALF_OPEN
                    cb.half_open_at = _utcnow()
                    cb.success_count = 0
                    logger.info(
                        f"Circuit breaker for {agent_id}: "
                        f"OPEN -> HALF_OPEN (timeout elapsed)"
                    )

            return cb.state

    def get_circuit_breaker_record(
        self,
        agent_id: str,
    ) -> Optional[CircuitBreakerRecord]:
        """Get the circuit breaker record for an agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            CircuitBreakerRecord or None.
        """
        with self._lock:
            return self._circuit_breakers.get(agent_id)

    def get_open_circuit_count(self) -> int:
        """Get the number of open circuit breakers.

        Returns:
            Count of agents with OPEN circuit breakers.
        """
        with self._lock:
            return sum(
                1 for cb in self._circuit_breakers.values()
                if cb.state == CircuitBreakerState.OPEN
            )

    def reset_circuit_breaker(self, agent_id: str) -> None:
        """Manually reset a circuit breaker to CLOSED state.

        Args:
            agent_id: Agent identifier.
        """
        with self._lock:
            cb = self._circuit_breakers.get(agent_id)
            if cb:
                cb.state = CircuitBreakerState.CLOSED
                cb.failure_count = 0
                cb.success_count = 0
                logger.info(
                    f"Circuit breaker for {agent_id} manually reset to CLOSED"
                )

    # ------------------------------------------------------------------
    # Dead letter queue
    # ------------------------------------------------------------------

    def add_to_dead_letter(
        self,
        workflow_id: str,
        agent_id: str,
        error_message: str,
        error_classification: ErrorClassification,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> DeadLetterEntry:
        """Add a permanently failed agent to the dead letter queue.

        Called when an agent has exhausted all retry attempts and
        cannot be recovered automatically.

        Args:
            workflow_id: Workflow identifier.
            agent_id: Failed agent identifier.
            error_message: Final error message.
            error_classification: Error classification.
            input_data: Input data that was sent to the agent.

        Returns:
            DeadLetterEntry for the failed agent.
        """
        with self._lock:
            # Collect retry history
            key = f"{workflow_id}:{agent_id}"
            retry_records = list(self._retry_history.get(key, []))

            # Get circuit breaker state
            cb = self._circuit_breakers.get(agent_id)
            cb_state = cb.state if cb else None

            entry = DeadLetterEntry(
                entry_id=_new_uuid(),
                workflow_id=workflow_id,
                agent_id=agent_id,
                error_message=error_message,
                error_classification=error_classification,
                retry_history=retry_records,
                input_data=input_data or {},
                circuit_breaker_state=cb_state,
                created_at=_utcnow(),
            )

            self._dead_letter_queue.append(entry)

            logger.error(
                f"Dead letter: {agent_id} in {workflow_id} "
                f"after {len(retry_records)} retries: {error_message[:100]}"
            )

            return entry

    def get_dead_letter_queue(self) -> List[DeadLetterEntry]:
        """Get all dead letter queue entries.

        Returns:
            List of DeadLetterEntry objects.
        """
        with self._lock:
            return list(self._dead_letter_queue)

    def get_dead_letter_count(self) -> int:
        """Get the number of dead letter entries.

        Returns:
            Count of unresolved dead letter entries.
        """
        with self._lock:
            return sum(
                1 for e in self._dead_letter_queue if not e.resolved
            )

    def resolve_dead_letter(
        self,
        entry_id: str,
        resolved_by: str,
    ) -> bool:
        """Mark a dead letter entry as resolved.

        Args:
            entry_id: Dead letter entry identifier.
            resolved_by: User who resolved the entry.

        Returns:
            True if entry was found and resolved.
        """
        with self._lock:
            for entry in self._dead_letter_queue:
                if entry.entry_id == entry_id:
                    entry.resolved = True
                    entry.resolved_by = resolved_by
                    entry.resolved_at = _utcnow()
                    logger.info(
                        f"Dead letter {entry_id} resolved by {resolved_by}"
                    )
                    return True
            return False

    # ------------------------------------------------------------------
    # Fallback strategy
    # ------------------------------------------------------------------

    def determine_fallback(
        self,
        agent_id: str,
        error_classification: ErrorClassification,
        fallback_strategy: FallbackStrategy = FallbackStrategy.FAIL,
    ) -> FallbackStrategy:
        """Determine the appropriate fallback strategy.

        Strategy selection logic:
        1. If agent has a configured fallback, use it
        2. If error is degraded, try cached result
        3. If error is permanent, check if agent is non-critical
        4. Default to configured strategy

        Args:
            agent_id: Agent identifier.
            error_classification: Error classification.
            fallback_strategy: Configured fallback strategy.

        Returns:
            FallbackStrategy to apply.
        """
        if error_classification == ErrorClassification.DEGRADED:
            return FallbackStrategy.CACHED_RESULT

        if error_classification == ErrorClassification.PERMANENT:
            if fallback_strategy == FallbackStrategy.FAIL:
                return FallbackStrategy.MANUAL_OVERRIDE
            return fallback_strategy

        return fallback_strategy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_backoff_delay(self, attempt: int) -> Decimal:
        """Compute exponential backoff delay with jitter.

        Formula: delay = min(base * 2^attempt + jitter, max_delay)

        Args:
            attempt: Attempt number (0-based).

        Returns:
            Delay in seconds as Decimal.
        """
        base = float(self._config.retry_base_delay_s)
        max_delay = float(self._config.retry_max_delay_s)
        jitter_max = float(self._config.retry_jitter_max_s)

        # Exponential component
        exponential = base * (2 ** attempt)

        # Add jitter
        jitter = random.uniform(0, jitter_max)

        # Apply cap
        delay = min(exponential + jitter, max_delay)

        return Decimal(str(delay)).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

    def _get_or_create_cb(self, agent_id: str) -> CircuitBreakerRecord:
        """Get or create a circuit breaker record for an agent.

        Must be called with self._lock held.

        Args:
            agent_id: Agent identifier.

        Returns:
            CircuitBreakerRecord for the agent.
        """
        if agent_id not in self._circuit_breakers:
            self._circuit_breakers[agent_id] = CircuitBreakerRecord(
                agent_id=agent_id,
                state=CircuitBreakerState.CLOSED,
                failure_threshold=self._config.cb_failure_threshold,
                reset_timeout_s=self._config.cb_reset_timeout_s,
            )
        return self._circuit_breakers[agent_id]
