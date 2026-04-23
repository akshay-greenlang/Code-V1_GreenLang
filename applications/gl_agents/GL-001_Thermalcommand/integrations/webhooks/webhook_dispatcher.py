"""
GL-001 ThermalCommand - Webhook Dispatcher

This module provides asynchronous webhook dispatch with:
- Async HTTP delivery using aiohttp
- Exponential backoff retry logic
- Circuit breaker pattern for fault tolerance
- Delivery tracking and reporting
- Concurrent delivery to multiple endpoints
- Graceful degradation

The dispatcher is designed for high-throughput, reliable event delivery
with minimal latency and resource consumption.

Example:
    >>> dispatcher = WebhookDispatcher(manager)
    >>> await dispatcher.start()
    >>> results = await dispatcher.dispatch_event(event)
    >>> await dispatcher.shutdown()

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple
import hashlib
import logging
import random
import time
import uuid

from pydantic import BaseModel, Field

from .webhook_events import WebhookEvent, WebhookEventType
from .webhook_config import (
    WebhookConfig,
    WebhookEndpoint,
    EndpointStatus,
    RetryConfig,
)
from .webhook_manager import (
    WebhookManager,
    DeliveryResult,
    DeliveryStatus,
)


logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerState:
    """State for a circuit breaker."""

    endpoint_id: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    opened_at: Optional[float] = None
    half_open_at: Optional[float] = None

    # Configuration
    failure_threshold: int = 5
    success_threshold: int = 3
    reset_timeout_seconds: float = 60.0
    half_open_max_requests: int = 3


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    Implements the circuit breaker pattern:
    - CLOSED: Normal operation, track failures
    - OPEN: Block requests, wait for reset timeout
    - HALF_OPEN: Allow limited requests to test recovery

    Example:
        >>> breaker = CircuitBreaker()
        >>> if breaker.can_execute(endpoint_id):
        ...     result = await deliver()
        ...     breaker.record_result(endpoint_id, success=True)
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        reset_timeout_seconds: float = 60.0
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            success_threshold: Successes to close half-open circuit
            reset_timeout_seconds: Time before attempting half-open
        """
        self._failure_threshold = failure_threshold
        self._success_threshold = success_threshold
        self._reset_timeout = reset_timeout_seconds
        self._states: Dict[str, CircuitBreakerState] = {}

    def _get_state(self, endpoint_id: str) -> CircuitBreakerState:
        """Get or create circuit state for endpoint."""
        if endpoint_id not in self._states:
            self._states[endpoint_id] = CircuitBreakerState(
                endpoint_id=endpoint_id,
                failure_threshold=self._failure_threshold,
                success_threshold=self._success_threshold,
                reset_timeout_seconds=self._reset_timeout
            )
        return self._states[endpoint_id]

    def can_execute(self, endpoint_id: str) -> Tuple[bool, str]:
        """
        Check if request can be executed.

        Args:
            endpoint_id: Target endpoint

        Returns:
            Tuple of (can_execute, reason)
        """
        state = self._get_state(endpoint_id)
        current_time = time.time()

        if state.state == CircuitState.CLOSED:
            return True, "circuit_closed"

        elif state.state == CircuitState.OPEN:
            # Check if reset timeout has passed
            if state.opened_at and current_time - state.opened_at >= state.reset_timeout_seconds:
                # Transition to half-open
                state.state = CircuitState.HALF_OPEN
                state.half_open_at = current_time
                state.success_count = 0
                logger.info(f"Circuit half-open for endpoint {endpoint_id}")
                return True, "circuit_half_open"

            return False, "circuit_open"

        elif state.state == CircuitState.HALF_OPEN:
            # Allow limited requests in half-open state
            if state.success_count < state.half_open_max_requests:
                return True, "circuit_half_open"
            return False, "circuit_half_open_limit"

        return False, "unknown_state"

    def record_success(self, endpoint_id: str) -> None:
        """
        Record a successful request.

        Args:
            endpoint_id: Target endpoint
        """
        state = self._get_state(endpoint_id)
        state.last_success_time = time.time()
        state.success_count += 1

        if state.state == CircuitState.HALF_OPEN:
            if state.success_count >= state.success_threshold:
                # Close the circuit
                state.state = CircuitState.CLOSED
                state.failure_count = 0
                state.opened_at = None
                state.half_open_at = None
                logger.info(f"Circuit closed for endpoint {endpoint_id}")

        elif state.state == CircuitState.CLOSED:
            # Reset failure count on success
            state.failure_count = 0

    def record_failure(self, endpoint_id: str) -> None:
        """
        Record a failed request.

        Args:
            endpoint_id: Target endpoint
        """
        state = self._get_state(endpoint_id)
        state.last_failure_time = time.time()
        state.failure_count += 1

        if state.state == CircuitState.HALF_OPEN:
            # Immediately open circuit on failure in half-open state
            state.state = CircuitState.OPEN
            state.opened_at = time.time()
            state.success_count = 0
            logger.warning(f"Circuit reopened for endpoint {endpoint_id}")

        elif state.state == CircuitState.CLOSED:
            if state.failure_count >= state.failure_threshold:
                # Open the circuit
                state.state = CircuitState.OPEN
                state.opened_at = time.time()
                logger.warning(
                    f"Circuit opened for endpoint {endpoint_id} "
                    f"after {state.failure_count} failures"
                )

    def get_state(self, endpoint_id: str) -> CircuitState:
        """Get current circuit state."""
        return self._get_state(endpoint_id).state

    def reset(self, endpoint_id: str) -> None:
        """Reset circuit breaker for endpoint."""
        if endpoint_id in self._states:
            del self._states[endpoint_id]

    def get_all_states(self) -> Dict[str, CircuitState]:
        """Get states for all endpoints."""
        return {
            ep_id: state.state
            for ep_id, state in self._states.items()
        }


class DispatchResult(BaseModel):
    """
    Result of dispatching an event to all endpoints.

    Attributes:
        event_id: Event identifier
        event_type: Type of event
        dispatched_at: Dispatch start timestamp
        completed_at: Dispatch completion timestamp
        total_endpoints: Number of target endpoints
        successful_deliveries: Number of successful deliveries
        failed_deliveries: Number of failed deliveries
        skipped_deliveries: Number of skipped (circuit open, duplicate)
        delivery_results: Results per endpoint
    """

    event_id: str = Field(..., description="Event identifier")
    event_type: str = Field(..., description="Event type")
    dispatched_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Dispatch start time"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Dispatch completion time"
    )
    total_endpoints: int = Field(default=0, ge=0, description="Total endpoints")
    successful_deliveries: int = Field(default=0, ge=0, description="Successful count")
    failed_deliveries: int = Field(default=0, ge=0, description="Failed count")
    skipped_deliveries: int = Field(default=0, ge=0, description="Skipped count")
    delivery_results: Dict[str, DeliveryResult] = Field(
        default_factory=dict,
        description="Results per endpoint"
    )
    total_duration_ms: float = Field(default=0.0, ge=0.0, description="Total duration")


@dataclass
class RetryTask:
    """Task for retry queue."""

    task_id: str
    event: WebhookEvent
    endpoint: WebhookEndpoint
    attempt: int
    scheduled_at: float
    delivery_result: DeliveryResult


class WebhookDispatcher:
    """
    Asynchronous webhook dispatcher with retry and circuit breaker.

    Handles the actual delivery of webhook events to endpoints with:
    - Concurrent async delivery
    - Exponential backoff retries
    - Circuit breaker protection
    - Rate limit handling
    - Delivery tracking

    Attributes:
        manager: Webhook manager for configuration and tracking
        circuit_breaker: Circuit breaker for fault tolerance

    Example:
        >>> dispatcher = WebhookDispatcher(manager)
        >>> await dispatcher.start()
        >>> result = await dispatcher.dispatch_event(event)
        >>> await dispatcher.shutdown()
    """

    def __init__(
        self,
        manager: WebhookManager,
        max_concurrent_deliveries: int = 100,
        circuit_failure_threshold: int = 5,
        circuit_reset_timeout: float = 60.0
    ):
        """
        Initialize webhook dispatcher.

        Args:
            manager: Webhook manager
            max_concurrent_deliveries: Maximum concurrent HTTP requests
            circuit_failure_threshold: Failures before opening circuit
            circuit_reset_timeout: Seconds before half-open attempt
        """
        self._manager = manager
        self._max_concurrent = max_concurrent_deliveries
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            reset_timeout_seconds=circuit_reset_timeout
        )
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._retry_queue: Deque[RetryTask] = deque()
        self._running = False
        self._retry_task: Optional[asyncio.Task] = None
        self._http_session: Optional[Any] = None  # aiohttp.ClientSession

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Get circuit breaker."""
        return self._circuit_breaker

    async def start(self) -> None:
        """
        Start the dispatcher.

        Initializes HTTP session and starts background tasks.
        """
        if self._running:
            return

        self._running = True
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._retry_task = asyncio.create_task(self._retry_loop())

        # Initialize HTTP session (lazy import for optional dependency)
        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=60)
            self._http_session = aiohttp.ClientSession(timeout=timeout)
        except ImportError:
            logger.warning("aiohttp not installed, using mock HTTP client")
            self._http_session = None

        logger.info("Webhook dispatcher started")

    async def shutdown(self) -> None:
        """
        Shutdown the dispatcher.

        Closes HTTP session and stops background tasks.
        """
        if not self._running:
            return

        self._running = False

        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass

        if self._http_session:
            await self._http_session.close()

        logger.info("Webhook dispatcher shutdown complete")

    async def _retry_loop(self) -> None:
        """Background task for processing retry queue."""
        while self._running:
            try:
                await asyncio.sleep(1)  # Check every second
                await self._process_retry_queue()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Retry loop error: {e}")

    async def _process_retry_queue(self) -> None:
        """Process pending retry tasks."""
        current_time = time.time()
        tasks_to_process = []

        # Collect tasks that are due
        while self._retry_queue:
            task = self._retry_queue[0]
            if task.scheduled_at <= current_time:
                tasks_to_process.append(self._retry_queue.popleft())
            else:
                break  # Queue is ordered by scheduled time

        # Process tasks
        for task in tasks_to_process:
            try:
                await self._deliver_with_retry(
                    task.event,
                    task.endpoint,
                    task.attempt,
                    task.delivery_result
                )
            except Exception as e:
                logger.error(f"Retry task failed: {e}")

    async def dispatch_event(
        self,
        event: WebhookEvent,
        endpoint_ids: Optional[List[str]] = None
    ) -> DispatchResult:
        """
        Dispatch event to all subscribed endpoints.

        Args:
            event: Event to dispatch
            endpoint_ids: Optional list of specific endpoint IDs

        Returns:
            DispatchResult with delivery outcomes
        """
        start_time = datetime.now(timezone.utc)

        # Get target endpoints
        if endpoint_ids:
            endpoints = [
                self._manager.registry.get_endpoint(eid)
                for eid in endpoint_ids
            ]
            endpoints = [e for e in endpoints if e is not None and e.is_active()]
        else:
            endpoints = self._manager.get_endpoints_for_event(event.event_type)

        result = DispatchResult(
            event_id=event.event_id,
            event_type=str(event.event_type),
            dispatched_at=start_time,
            total_endpoints=len(endpoints)
        )

        if not endpoints:
            result.completed_at = datetime.now(timezone.utc)
            return result

        # Dispatch to all endpoints concurrently
        delivery_tasks = [
            self._deliver_to_endpoint(event, endpoint)
            for endpoint in endpoints
        ]

        delivery_results = await asyncio.gather(*delivery_tasks, return_exceptions=True)

        # Process results
        for endpoint, delivery_outcome in zip(endpoints, delivery_results):
            if isinstance(delivery_outcome, Exception):
                logger.error(f"Delivery exception for {endpoint.endpoint_id}: {delivery_outcome}")
                delivery_result = DeliveryResult(
                    event_id=event.event_id,
                    endpoint_id=endpoint.endpoint_id,
                    status=DeliveryStatus.FAILED,
                    error_message=str(delivery_outcome)
                )
                result.failed_deliveries += 1
            else:
                delivery_result = delivery_outcome

                if delivery_result.status == DeliveryStatus.DELIVERED:
                    result.successful_deliveries += 1
                elif delivery_result.status in {
                    DeliveryStatus.CIRCUIT_OPEN,
                    DeliveryStatus.RATE_LIMITED
                }:
                    result.skipped_deliveries += 1
                else:
                    result.failed_deliveries += 1

            result.delivery_results[endpoint.endpoint_id] = delivery_result

        result.completed_at = datetime.now(timezone.utc)
        result.total_duration_ms = (
            result.completed_at - result.dispatched_at
        ).total_seconds() * 1000

        return result

    async def _deliver_to_endpoint(
        self,
        event: WebhookEvent,
        endpoint: WebhookEndpoint
    ) -> DeliveryResult:
        """
        Deliver event to a single endpoint with all protections.

        Args:
            event: Event to deliver
            endpoint: Target endpoint

        Returns:
            DeliveryResult
        """
        # Check for duplicate
        if self._manager.is_duplicate(event, endpoint):
            cached = self._manager.get_cached_result(event, endpoint)
            if cached:
                logger.debug(f"Returning cached result for {event.event_id}")
                return cached

        # Check circuit breaker
        can_execute, reason = self._circuit_breaker.can_execute(endpoint.endpoint_id)
        if not can_execute:
            result = DeliveryResult(
                event_id=event.event_id,
                endpoint_id=endpoint.endpoint_id,
                status=DeliveryStatus.CIRCUIT_OPEN,
                error_message=f"Circuit breaker: {reason}"
            )
            self._manager.record_delivery_result(event, endpoint, result)
            return result

        # Check rate limit
        can_proceed, wait_time = self._manager.check_rate_limit(endpoint)
        if not can_proceed:
            result = DeliveryResult(
                event_id=event.event_id,
                endpoint_id=endpoint.endpoint_id,
                status=DeliveryStatus.RATE_LIMITED,
                error_message=f"Rate limited, retry after {wait_time:.1f}s"
            )
            self._manager.record_delivery_result(event, endpoint, result)
            # Schedule retry
            self._schedule_retry(event, endpoint, 1, result, wait_time)
            return result

        # Deliver with retry
        return await self._deliver_with_retry(event, endpoint, 1)

    async def _deliver_with_retry(
        self,
        event: WebhookEvent,
        endpoint: WebhookEndpoint,
        attempt: int,
        previous_result: Optional[DeliveryResult] = None
    ) -> DeliveryResult:
        """
        Deliver with retry logic.

        Args:
            event: Event to deliver
            endpoint: Target endpoint
            attempt: Current attempt number
            previous_result: Previous delivery result if retrying

        Returns:
            DeliveryResult
        """
        retry_config = endpoint.retry_config
        max_attempts = retry_config.max_retries + 1

        result = await self._execute_delivery(event, endpoint, attempt)

        # Record in circuit breaker
        if result.status == DeliveryStatus.DELIVERED:
            self._circuit_breaker.record_success(endpoint.endpoint_id)
        else:
            self._circuit_breaker.record_failure(endpoint.endpoint_id)

        # Handle retry
        if result.status == DeliveryStatus.FAILED and attempt < max_attempts:
            if result.http_status_code in retry_config.retry_on_status_codes or result.http_status_code is None:
                delay_ms = retry_config.calculate_delay(attempt - 1)
                result.status = DeliveryStatus.RETRYING
                result.next_retry_at = datetime.now(timezone.utc) + timedelta(
                    milliseconds=delay_ms
                )

                self._schedule_retry(
                    event,
                    endpoint,
                    attempt + 1,
                    result,
                    delay_ms / 1000.0
                )

                logger.info(
                    f"Scheduling retry {attempt + 1}/{max_attempts} for "
                    f"{event.event_id} to {endpoint.endpoint_id} in {delay_ms}ms"
                )
        elif result.status == DeliveryStatus.FAILED:
            # Max retries exceeded, add to DLQ
            self._manager.add_to_dlq(
                event,
                endpoint,
                result,
                result.error_message or "Max retries exceeded"
            )
            result.status = DeliveryStatus.DLQ

            # Update endpoint status if consistently failing
            circuit_state = self._circuit_breaker.get_state(endpoint.endpoint_id)
            if circuit_state == CircuitState.OPEN:
                self._manager.update_endpoint_status(
                    endpoint.endpoint_id,
                    EndpointStatus.FAILING
                )

        self._manager.record_delivery_result(event, endpoint, result)
        return result

    async def _execute_delivery(
        self,
        event: WebhookEvent,
        endpoint: WebhookEndpoint,
        attempt: int
    ) -> DeliveryResult:
        """
        Execute HTTP delivery to endpoint.

        Args:
            event: Event to deliver
            endpoint: Target endpoint
            attempt: Attempt number

        Returns:
            DeliveryResult
        """
        start_time = datetime.now(timezone.utc)

        # Prepare delivery
        headers, payload, delivery_id = self._manager.prepare_delivery(event, endpoint)

        result = DeliveryResult(
            delivery_id=delivery_id,
            event_id=event.event_id,
            endpoint_id=endpoint.endpoint_id,
            status=DeliveryStatus.IN_PROGRESS,
            attempt_count=attempt,
            started_at=start_time,
            signature=headers.get(self._manager.config.signature_header)
        )

        try:
            async with self._semaphore:
                if self._http_session:
                    # Real HTTP delivery
                    async with self._http_session.post(
                        endpoint.url,
                        data=payload,
                        headers=headers,
                        timeout=endpoint.timeout_ms / 1000.0
                    ) as response:
                        result.http_status_code = response.status
                        result.response_body = await response.text()

                        if 200 <= response.status < 300:
                            result.status = DeliveryStatus.DELIVERED
                        else:
                            result.status = DeliveryStatus.FAILED
                            result.error_message = f"HTTP {response.status}"
                else:
                    # Mock delivery for testing
                    result = await self._mock_delivery(result, endpoint, payload, headers)

        except asyncio.TimeoutError:
            result.status = DeliveryStatus.FAILED
            result.error_message = f"Timeout after {endpoint.timeout_ms}ms"

        except Exception as e:
            result.status = DeliveryStatus.FAILED
            result.error_message = str(e)
            logger.error(f"Delivery error: {e}")

        finally:
            result.completed_at = datetime.now(timezone.utc)
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000

        return result

    async def _mock_delivery(
        self,
        result: DeliveryResult,
        endpoint: WebhookEndpoint,
        payload: str,
        headers: Dict[str, str]
    ) -> DeliveryResult:
        """
        Mock delivery for testing without aiohttp.

        Args:
            result: Delivery result to update
            endpoint: Target endpoint
            payload: JSON payload
            headers: Request headers

        Returns:
            Updated DeliveryResult
        """
        # Simulate network latency
        await asyncio.sleep(random.uniform(0.01, 0.1))

        # Simulate various responses
        success_rate = 0.95
        if random.random() < success_rate:
            result.status = DeliveryStatus.DELIVERED
            result.http_status_code = 200
            result.response_body = '{"status": "ok"}'
        else:
            result.status = DeliveryStatus.FAILED
            result.http_status_code = random.choice([500, 502, 503])
            result.error_message = f"Mock failure: HTTP {result.http_status_code}"

        return result

    def _schedule_retry(
        self,
        event: WebhookEvent,
        endpoint: WebhookEndpoint,
        attempt: int,
        result: DeliveryResult,
        delay_seconds: float
    ) -> None:
        """
        Schedule a retry task.

        Args:
            event: Event to retry
            endpoint: Target endpoint
            attempt: Next attempt number
            result: Current delivery result
            delay_seconds: Delay before retry
        """
        task = RetryTask(
            task_id=str(uuid.uuid4()),
            event=event,
            endpoint=endpoint,
            attempt=attempt,
            scheduled_at=time.time() + delay_seconds,
            delivery_result=result
        )

        # Insert in order
        inserted = False
        for i, existing in enumerate(self._retry_queue):
            if task.scheduled_at < existing.scheduled_at:
                self._retry_queue.insert(i, task)
                inserted = True
                break

        if not inserted:
            self._retry_queue.append(task)

    def get_retry_queue_size(self) -> int:
        """Get number of pending retries."""
        return len(self._retry_queue)

    def get_circuit_states(self) -> Dict[str, CircuitState]:
        """Get circuit breaker states for all endpoints."""
        return self._circuit_breaker.get_all_states()

    async def retry_from_dlq(
        self,
        entry_id: str
    ) -> Optional[DeliveryResult]:
        """
        Retry a delivery from the dead letter queue.

        Args:
            entry_id: DLQ entry ID

        Returns:
            DeliveryResult if retry attempted
        """
        entry = self._manager.dlq.get_entry(entry_id)
        if entry is None:
            return None

        endpoint = self._manager.registry.get_endpoint(entry.endpoint_id)
        if endpoint is None:
            return None

        # Remove from DLQ before retry
        self._manager.dlq.remove_entry(entry_id)

        # Retry delivery
        result = await self._deliver_with_retry(
            entry.event,
            endpoint,
            1  # Start fresh retry count
        )

        return result


class WebhookDeliveryTracker:
    """
    Tracks webhook delivery statistics and history.

    Provides methods for querying delivery history, calculating
    success rates, and identifying problematic endpoints.

    Example:
        >>> tracker = WebhookDeliveryTracker()
        >>> tracker.record(result)
        >>> stats = tracker.get_endpoint_stats(endpoint_id)
    """

    def __init__(self, history_size: int = 10000):
        """
        Initialize delivery tracker.

        Args:
            history_size: Maximum number of deliveries to track
        """
        self._history_size = history_size
        self._history: Deque[DeliveryResult] = deque(maxlen=history_size)
        self._by_endpoint: Dict[str, Deque[DeliveryResult]] = {}
        self._endpoint_history_size = 1000

    def record(self, result: DeliveryResult) -> None:
        """
        Record a delivery result.

        Args:
            result: Delivery result to record
        """
        self._history.append(result)

        if result.endpoint_id not in self._by_endpoint:
            self._by_endpoint[result.endpoint_id] = deque(
                maxlen=self._endpoint_history_size
            )
        self._by_endpoint[result.endpoint_id].append(result)

    def get_endpoint_stats(
        self,
        endpoint_id: str,
        window_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Get statistics for an endpoint.

        Args:
            endpoint_id: Endpoint identifier
            window_minutes: Time window for statistics

        Returns:
            Dictionary of statistics
        """
        history = self._by_endpoint.get(endpoint_id, deque())
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)

        recent = [
            r for r in history
            if r.completed_at and r.completed_at >= cutoff
        ]

        if not recent:
            return {
                "total_deliveries": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
                "error_rate": 0.0,
            }

        successful = sum(1 for r in recent if r.status == DeliveryStatus.DELIVERED)
        failed = sum(
            1 for r in recent
            if r.status in {DeliveryStatus.FAILED, DeliveryStatus.DLQ}
        )
        durations = [r.duration_ms for r in recent if r.duration_ms > 0]

        return {
            "total_deliveries": len(recent),
            "successful_deliveries": successful,
            "failed_deliveries": failed,
            "success_rate": successful / len(recent) if recent else 0.0,
            "error_rate": failed / len(recent) if recent else 0.0,
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0.0,
            "min_duration_ms": min(durations) if durations else 0.0,
            "max_duration_ms": max(durations) if durations else 0.0,
        }

    def get_recent_failures(
        self,
        limit: int = 100
    ) -> List[DeliveryResult]:
        """
        Get recent failed deliveries.

        Args:
            limit: Maximum number to return

        Returns:
            List of failed delivery results
        """
        failures = [
            r for r in self._history
            if r.status in {DeliveryStatus.FAILED, DeliveryStatus.DLQ}
        ]
        return list(failures)[-limit:]

    def get_global_stats(self, window_minutes: int = 60) -> Dict[str, Any]:
        """
        Get global delivery statistics.

        Args:
            window_minutes: Time window

        Returns:
            Dictionary of global statistics
        """
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        recent = [
            r for r in self._history
            if r.completed_at and r.completed_at >= cutoff
        ]

        if not recent:
            return {
                "total_deliveries": 0,
                "success_rate": 0.0,
                "endpoints_with_failures": 0,
            }

        successful = sum(1 for r in recent if r.status == DeliveryStatus.DELIVERED)
        failed_endpoints = set(
            r.endpoint_id for r in recent
            if r.status in {DeliveryStatus.FAILED, DeliveryStatus.DLQ}
        )

        return {
            "total_deliveries": len(recent),
            "successful_deliveries": successful,
            "success_rate": successful / len(recent) if recent else 0.0,
            "endpoints_with_failures": len(failed_endpoints),
            "unique_endpoints": len(set(r.endpoint_id for r in recent)),
        }
