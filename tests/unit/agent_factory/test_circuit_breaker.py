# -*- coding: utf-8 -*-
"""
Unit tests for Circuit Breaker, Fallback Chain, and Bulkhead Isolation.

Tests the resilience primitives from the agent factory including state
transitions, failure rate detection, slow call detection, manual overrides,
fallback chains, and concurrency limiting.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.infrastructure.agent_factory.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitBreakerState,
    CircuitOpenError,
)
from greenlang.infrastructure.agent_factory.resilience.fallback import (
    FallbackChain,
    FallbackResult,
)
from greenlang.infrastructure.agent_factory.resilience.bulkhead import (
    BulkheadConfig,
    BulkheadFullError,
    BulkheadIsolation,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clear_registries() -> None:
    """Clear global registries before each test."""
    CircuitBreaker.clear_registry()
    BulkheadIsolation.clear_registry()


@pytest.fixture
def cb_config() -> CircuitBreakerConfig:
    return CircuitBreakerConfig(
        failure_rate_threshold=0.5,
        slow_call_threshold_s=1.0,
        slow_call_rate_threshold=0.8,
        wait_in_open_s=0.1,
        half_open_test_requests=2,
        sliding_window_size_s=60.0,
        minimum_calls=3,
    )


@pytest.fixture
def circuit_breaker(cb_config: CircuitBreakerConfig) -> CircuitBreaker:
    return CircuitBreaker("test-agent", cb_config)


@pytest.fixture
def fallback_chain() -> FallbackChain:
    return FallbackChain("test-agent")


@pytest.fixture
def bulkhead() -> BulkheadIsolation:
    config = BulkheadConfig(max_concurrent=3, queue_size=5, queue_timeout_s=0.5)
    return BulkheadIsolation("test-agent", config)


# ============================================================================
# Test CircuitBreaker
# ============================================================================


class TestCircuitBreaker:
    """Tests for the CircuitBreaker state machine and failure detection."""

    def test_circuit_breaker_initial_state_closed(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Circuit breaker starts in CLOSED state."""
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failure_threshold(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Circuit opens when failure rate exceeds the threshold."""
        # Record 3 failures (minimum_calls=3, threshold=0.5)
        for _ in range(3):
            await circuit_breaker.record_call(success=False, duration_s=0.1)

        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker.metrics.state_transitions >= 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_stays_closed_below_minimum_calls(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Circuit does not open until minimum_calls is reached."""
        await circuit_breaker.record_call(success=False, duration_s=0.1)
        await circuit_breaker.record_call(success=False, duration_s=0.1)
        # Only 2 calls, minimum is 3
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_after_wait(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Circuit transitions to HALF_OPEN after the wait period."""
        # Trip the circuit
        for _ in range(3):
            await circuit_breaker.record_call(success=False, duration_s=0.1)
        assert circuit_breaker.state == CircuitBreakerState.OPEN

        # Wait for the open duration
        await asyncio.sleep(0.15)

        # Next call should trigger HALF_OPEN transition
        try:
            async with circuit_breaker:
                pass
        except CircuitOpenError:
            pytest.fail("Should not raise CircuitOpenError after wait period")

        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_closes_after_successful_test(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Circuit closes when all half-open test requests succeed."""
        # Trip the circuit
        for _ in range(3):
            await circuit_breaker.record_call(success=False, duration_s=0.1)

        # Wait for HALF_OPEN
        await asyncio.sleep(0.15)

        # Trigger HALF_OPEN via _before_call
        await circuit_breaker._before_call()

        # Succeed enough test requests (half_open_test_requests=2)
        await circuit_breaker.record_call(success=True, duration_s=0.1)
        await circuit_breaker.record_call(success=True, duration_s=0.1)

        assert circuit_breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_reopens_on_half_open_failure(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Any failure in HALF_OPEN re-opens the circuit."""
        # Trip to OPEN
        for _ in range(3):
            await circuit_breaker.record_call(success=False, duration_s=0.1)

        # Wait for HALF_OPEN
        await asyncio.sleep(0.15)
        await circuit_breaker._before_call()
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

        # Fail in HALF_OPEN
        await circuit_breaker.record_call(success=False, duration_s=0.1)
        assert circuit_breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_slow_call_detection(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Slow calls are tracked in metrics."""
        await circuit_breaker.record_call(success=True, duration_s=2.0)
        assert circuit_breaker.metrics.slow_call_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_manual_force_open(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Manual force_open transitions to OPEN regardless of metrics."""
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        await circuit_breaker.force_open()
        assert circuit_breaker.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_manual_force_close(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Manual force_close transitions to CLOSED and resets counters."""
        await circuit_breaker.force_open()
        assert circuit_breaker.state == CircuitBreakerState.OPEN

        await circuit_breaker.force_close()
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_reset(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Reset clears all state and metrics."""
        for _ in range(3):
            await circuit_breaker.record_call(success=False, duration_s=0.1)
        await circuit_breaker.reset()
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.metrics.total_calls == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_rejects_when_open(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Calls are rejected with CircuitOpenError when circuit is OPEN."""
        await circuit_breaker.force_open()
        with pytest.raises(CircuitOpenError) as exc_info:
            await circuit_breaker._before_call()
        assert exc_info.value.agent_key == "test-agent"
        assert circuit_breaker.metrics.rejected_count >= 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_context_manager_success(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Context manager records success when no exception."""
        async with circuit_breaker:
            pass  # success
        assert circuit_breaker.metrics.success_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_context_manager_failure(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Context manager records failure when exception is raised."""
        with pytest.raises(ValueError):
            async with circuit_breaker:
                raise ValueError("test error")
        assert circuit_breaker.metrics.failure_count == 1

    def test_circuit_breaker_snapshot(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """Snapshot returns diagnostic information."""
        snap = circuit_breaker.snapshot()
        assert snap["agent_key"] == "test-agent"
        assert snap["state"] == "closed"
        assert "metrics" in snap

    def test_circuit_breaker_registry(self, cb_config: CircuitBreakerConfig) -> None:
        """Circuit breakers are stored in the class-level registry."""
        cb = CircuitBreaker("reg-test", cb_config)
        assert CircuitBreaker.get("reg-test") is cb

    def test_circuit_breaker_get_or_create(
        self, cb_config: CircuitBreakerConfig
    ) -> None:
        """get_or_create returns existing or creates new."""
        cb1 = CircuitBreaker.get_or_create("new-agent", cb_config)
        cb2 = CircuitBreaker.get_or_create("new-agent", cb_config)
        assert cb1 is cb2

    @pytest.mark.asyncio
    async def test_circuit_breaker_state_change_callback(
        self, circuit_breaker: CircuitBreaker
    ) -> None:
        """State change callback is invoked on transitions."""
        callback = AsyncMock()
        circuit_breaker.on_state_change(callback)
        await circuit_breaker.force_open()
        # Allow event loop to process the callback
        await asyncio.sleep(0.05)
        callback.assert_awaited()


# ============================================================================
# Test FallbackChain
# ============================================================================


class TestFallbackChain:
    """Tests for the FallbackChain execution pattern."""

    @pytest.mark.asyncio
    async def test_fallback_chain_primary_success(
        self, fallback_chain: FallbackChain
    ) -> None:
        """Primary handler success returns immediately."""
        async def primary(ctx: Dict[str, Any]) -> str:
            return "primary_result"

        fallback_chain.add_handler("primary", primary)
        result = await fallback_chain.execute({"key": "value"})
        assert result.result == "primary_result"
        assert result.handler_name == "primary"
        assert result.was_fallback is False

    @pytest.mark.asyncio
    async def test_fallback_chain_primary_fails_secondary_succeeds(
        self, fallback_chain: FallbackChain
    ) -> None:
        """When primary fails, secondary handler is tried."""
        async def primary(ctx: Dict[str, Any]) -> str:
            raise RuntimeError("primary failed")

        async def secondary(ctx: Dict[str, Any]) -> str:
            return "secondary_result"

        fallback_chain.add_handler("primary", primary)
        fallback_chain.add_handler("secondary", secondary)
        result = await fallback_chain.execute({})
        assert result.result == "secondary_result"
        assert result.handler_name == "secondary"
        assert result.was_fallback is True
        assert "primary" in result.errors

    @pytest.mark.asyncio
    async def test_fallback_chain_all_fail_default(
        self, fallback_chain: FallbackChain
    ) -> None:
        """When all handlers fail, default handler is used."""
        async def primary(ctx: Dict[str, Any]) -> str:
            raise RuntimeError("fail-1")

        async def secondary(ctx: Dict[str, Any]) -> str:
            raise RuntimeError("fail-2")

        async def default_handler(ctx: Dict[str, Any]) -> str:
            return "default_result"

        fallback_chain.add_handler("primary", primary)
        fallback_chain.add_handler("secondary", secondary)
        fallback_chain.set_default(default_handler)

        result = await fallback_chain.execute({})
        assert result.result == "default_result"
        assert result.handler_name == "default"
        assert result.was_fallback is True
        assert len(result.errors) == 2

    @pytest.mark.asyncio
    async def test_fallback_chain_all_fail_no_default_raises(
        self, fallback_chain: FallbackChain
    ) -> None:
        """When all handlers fail and no default, RuntimeError is raised."""
        async def failing(ctx: Dict[str, Any]) -> str:
            raise RuntimeError("failed")

        fallback_chain.add_handler("only", failing)
        with pytest.raises(RuntimeError, match="all .* handlers failed"):
            await fallback_chain.execute({})

    @pytest.mark.asyncio
    async def test_fallback_chain_no_handlers_raises(
        self, fallback_chain: FallbackChain
    ) -> None:
        """Executing with no handlers raises RuntimeError."""
        with pytest.raises(RuntimeError, match="no handlers configured"):
            await fallback_chain.execute({})

    @pytest.mark.asyncio
    async def test_fallback_chain_metrics(
        self, fallback_chain: FallbackChain
    ) -> None:
        """Fallback chain tracks per-handler metrics."""
        async def primary(ctx: Dict[str, Any]) -> str:
            return "ok"

        fallback_chain.add_handler("primary", primary)
        await fallback_chain.execute({})
        await fallback_chain.execute({})

        metrics = fallback_chain.get_metrics()
        assert metrics["primary"]["call_count"] == 2
        assert metrics["primary"]["success_count"] == 2
        assert metrics["primary"]["failure_count"] == 0

    @pytest.mark.asyncio
    async def test_fallback_chain_fluent_api(self) -> None:
        """Handlers can be added with fluent chaining."""
        chain = (
            FallbackChain("agent-x")
            .add_handler("h1", AsyncMock(return_value="r1"))
            .add_handler("h2", AsyncMock(return_value="r2"))
            .set_default(AsyncMock(return_value="default"))
        )
        result = await chain.execute({})
        assert result.result == "r1"


# ============================================================================
# Test BulkheadIsolation
# ============================================================================


class TestBulkheadIsolation:
    """Tests for the BulkheadIsolation concurrency limiter."""

    @pytest.mark.asyncio
    async def test_bulkhead_allows_within_limit(
        self, bulkhead: BulkheadIsolation
    ) -> None:
        """Requests within the concurrency limit pass through."""
        async with bulkhead.acquire():
            assert bulkhead.active_count == 1
        assert bulkhead.active_count == 0
        assert bulkhead.metrics.total_acquired == 1

    @pytest.mark.asyncio
    async def test_bulkhead_rejects_above_limit(self) -> None:
        """When queue is full, BulkheadFullError is raised."""
        config = BulkheadConfig(
            max_concurrent=1, queue_size=0, queue_timeout_s=0.1
        )
        bulkhead = BulkheadIsolation("limited-agent", config)

        acquired = asyncio.Event()
        hold = asyncio.Event()

        async def hold_slot():
            async with bulkhead.acquire():
                acquired.set()
                await hold.wait()

        task = asyncio.create_task(hold_slot())
        await acquired.wait()

        # Second request with queue_size=0 should be rejected
        with pytest.raises(BulkheadFullError) as exc_info:
            async with bulkhead.acquire():
                pass

        assert exc_info.value.agent_key == "limited-agent"
        hold.set()
        await task

    @pytest.mark.asyncio
    async def test_bulkhead_concurrent_within_limit(
        self, bulkhead: BulkheadIsolation
    ) -> None:
        """Multiple concurrent requests within limit all succeed."""
        results: List[bool] = []

        async def work():
            async with bulkhead.acquire():
                await asyncio.sleep(0.01)
                results.append(True)

        # max_concurrent=3, send 3
        tasks = [asyncio.create_task(work()) for _ in range(3)]
        await asyncio.gather(*tasks)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_bulkhead_tracks_peak_active(
        self, bulkhead: BulkheadIsolation
    ) -> None:
        """Peak active count is tracked in metrics."""
        barrier = asyncio.Barrier(3)

        async def work():
            async with bulkhead.acquire():
                await barrier.wait()

        tasks = [asyncio.create_task(work()) for _ in range(3)]
        await asyncio.gather(*tasks)
        assert bulkhead.metrics.peak_active == 3

    def test_bulkhead_snapshot(self, bulkhead: BulkheadIsolation) -> None:
        """Snapshot returns diagnostic information."""
        snap = bulkhead.snapshot()
        assert snap["agent_key"] == "test-agent"
        assert snap["active_count"] == 0
        assert snap["config"]["max_concurrent"] == 3

    def test_bulkhead_available_slots(
        self, bulkhead: BulkheadIsolation
    ) -> None:
        """Available slots reports correct count."""
        assert bulkhead.available == 3

    def test_bulkhead_registry(self) -> None:
        """Bulkheads are stored in the class-level registry."""
        bh = BulkheadIsolation("reg-test")
        assert BulkheadIsolation.get("reg-test") is bh

    def test_bulkhead_get_or_create(self) -> None:
        """get_or_create returns existing or creates new."""
        bh1 = BulkheadIsolation.get_or_create("new-agent")
        bh2 = BulkheadIsolation.get_or_create("new-agent")
        assert bh1 is bh2


# ============================================================================
# Test Retry and Timeout (inline implementations)
# ============================================================================


class TestRetryExponentialBackoff:
    """Tests for retry with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_exponential_backoff(self) -> None:
        """Retries with exponential backoff eventually succeed."""
        call_count = 0

        async def flaky_operation() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "success"

        result = None
        for attempt in range(5):
            try:
                result = await flaky_operation()
                break
            except ConnectionError:
                await asyncio.sleep(0.01 * (2 ** attempt))

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_max_attempts_exceeded(self) -> None:
        """After max attempts, the last exception propagates."""
        async def always_fail() -> str:
            raise RuntimeError("permanent failure")

        max_attempts = 3
        last_exc = None
        for attempt in range(max_attempts):
            try:
                await always_fail()
                break
            except RuntimeError as exc:
                last_exc = exc
                await asyncio.sleep(0.001)

        assert last_exc is not None
        assert "permanent failure" in str(last_exc)


class TestTimeoutGuard:
    """Tests for timeout enforcement."""

    @pytest.mark.asyncio
    async def test_timeout_guard_within_limit(self) -> None:
        """Operations completing within timeout succeed."""
        async def fast_op() -> str:
            return "fast"

        result = await asyncio.wait_for(fast_op(), timeout=1.0)
        assert result == "fast"

    @pytest.mark.asyncio
    async def test_timeout_guard_exceeds_limit(self) -> None:
        """Operations exceeding timeout raise TimeoutError."""
        async def slow_op() -> str:
            await asyncio.sleep(10.0)
            return "slow"

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_op(), timeout=0.01)
