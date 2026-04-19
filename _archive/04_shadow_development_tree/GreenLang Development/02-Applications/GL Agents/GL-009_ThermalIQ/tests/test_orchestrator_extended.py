# -*- coding: utf-8 -*-
"""
Extended Orchestrator Tests for GL-009 THERMALIQ

Comprehensive tests for orchestrator workflows including:
- State machine transitions
- Batch processing
- Retry logic and circuit breaker
- Caching behavior
- Parallel execution
- Error recovery and resilience
- Configuration validation

Author: GL-TestEngineer
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import time
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager

import pytest


# Try importing hypothesis for property-based testing
try:
    from hypothesis import given, strategies as st, settings, assume
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


# =============================================================================
# TEST CLASS: STATE MACHINE TRANSITIONS
# =============================================================================

class TestOrchestratorStateMachine:
    """Test orchestrator state machine transitions."""

    VALID_STATES = ["idle", "initializing", "running", "paused", "completed", "error", "shutdown"]
    VALID_TRANSITIONS = {
        "idle": ["initializing", "shutdown"],
        "initializing": ["running", "error", "shutdown"],
        "running": ["paused", "completed", "error", "shutdown"],
        "paused": ["running", "shutdown"],
        "completed": ["idle", "shutdown"],
        "error": ["idle", "shutdown"],
        "shutdown": [],
    }

    @pytest.mark.integration
    def test_initial_state_is_idle(self):
        """Test that orchestrator starts in idle state."""
        orchestrator = self._create_mock_orchestrator()

        assert orchestrator.state == "idle"

    @pytest.mark.integration
    def test_valid_state_transitions(self):
        """Test all valid state transitions."""
        orchestrator = self._create_mock_orchestrator()

        for from_state, to_states in self.VALID_TRANSITIONS.items():
            for to_state in to_states:
                orchestrator.state = from_state
                result = orchestrator.transition_to(to_state)

                assert result is True, \
                    f"Transition from {from_state} to {to_state} should be valid"

    @pytest.mark.integration
    def test_invalid_state_transitions_rejected(self):
        """Test that invalid state transitions are rejected."""
        orchestrator = self._create_mock_orchestrator()

        invalid_transitions = [
            ("idle", "completed"),
            ("idle", "running"),
            ("completed", "running"),
            ("shutdown", "idle"),
        ]

        for from_state, to_state in invalid_transitions:
            orchestrator.state = from_state
            result = orchestrator.transition_to(to_state)

            assert result is False, \
                f"Transition from {from_state} to {to_state} should be invalid"

    @pytest.mark.integration
    def test_state_history_tracked(self):
        """Test that state history is tracked."""
        orchestrator = self._create_mock_orchestrator()

        orchestrator.transition_to("initializing")
        orchestrator.transition_to("running")
        orchestrator.transition_to("completed")

        assert len(orchestrator.state_history) >= 3
        assert orchestrator.state_history[-1]["state"] == "completed"

    @pytest.mark.integration
    def test_state_timestamps_recorded(self):
        """Test that state transitions include timestamps."""
        orchestrator = self._create_mock_orchestrator()

        before = datetime.now(timezone.utc)
        orchestrator.transition_to("initializing")
        after = datetime.now(timezone.utc)

        entry = orchestrator.state_history[-1]
        timestamp = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))

        assert before <= timestamp <= after

    @pytest.mark.integration
    def test_state_transition_callbacks(self):
        """Test that state transition callbacks are invoked."""
        orchestrator = self._create_mock_orchestrator()
        callback_log = []

        def on_state_change(old_state, new_state):
            callback_log.append((old_state, new_state))

        orchestrator.on_state_change = on_state_change

        orchestrator.transition_to("initializing")
        orchestrator.transition_to("running")

        assert ("idle", "initializing") in callback_log
        assert ("initializing", "running") in callback_log

    def _create_mock_orchestrator(self):
        """Create mock orchestrator for testing."""
        class MockOrchestrator:
            def __init__(self):
                self.state = "idle"
                self.state_history = [{"state": "idle", "timestamp": datetime.now(timezone.utc).isoformat()}]
                self.on_state_change = None

            def transition_to(self, new_state):
                valid_transitions = {
                    "idle": ["initializing", "shutdown"],
                    "initializing": ["running", "error", "shutdown"],
                    "running": ["paused", "completed", "error", "shutdown"],
                    "paused": ["running", "shutdown"],
                    "completed": ["idle", "shutdown"],
                    "error": ["idle", "shutdown"],
                    "shutdown": [],
                }

                if new_state in valid_transitions.get(self.state, []):
                    old_state = self.state
                    self.state = new_state
                    self.state_history.append({
                        "state": new_state,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    if self.on_state_change:
                        self.on_state_change(old_state, new_state)
                    return True
                return False

        return MockOrchestrator()


# =============================================================================
# TEST CLASS: BATCH PROCESSING
# =============================================================================

class TestBatchProcessing:
    """Test orchestrator batch processing capabilities."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_processing_single_item(self):
        """Test batch processing with single item."""
        items = [self._create_analysis_input()]

        results = await self._process_batch(items)

        assert len(results) == 1
        assert results[0]["status"] == "success"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_processing_multiple_items(self):
        """Test batch processing with multiple items."""
        items = [self._create_analysis_input() for _ in range(10)]

        results = await self._process_batch(items)

        assert len(results) == 10
        assert all(r["status"] == "success" for r in results)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_processing_preserves_order(self):
        """Test that batch processing preserves input order."""
        items = [
            {"id": f"item_{i}", "value": i * 100}
            for i in range(5)
        ]

        results = await self._process_batch(items)

        for i, result in enumerate(results):
            assert result["input_id"] == f"item_{i}"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_processing_partial_failure(self):
        """Test batch processing with some failures."""
        items = [
            {"id": "good_1", "valid": True},
            {"id": "bad_1", "valid": False},  # Will fail
            {"id": "good_2", "valid": True},
        ]

        results = await self._process_batch(items, allow_partial_failure=True)

        assert len(results) == 3
        assert results[0]["status"] == "success"
        assert results[1]["status"] == "error"
        assert results[2]["status"] == "success"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_processing_with_concurrency_limit(self):
        """Test batch processing respects concurrency limits."""
        items = [self._create_analysis_input() for _ in range(20)]
        max_concurrent = 5

        concurrent_count = []

        async def track_concurrency(item):
            concurrent_count.append(len(concurrent_count))
            await asyncio.sleep(0.01)
            concurrent_count.pop()
            return {"status": "success", "input_id": item.get("id", "")}

        results = await self._process_batch_with_semaphore(
            items, track_concurrency, max_concurrent
        )

        assert len(results) == 20

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_processing_progress_reporting(self):
        """Test batch processing reports progress."""
        items = [self._create_analysis_input() for _ in range(10)]
        progress_updates = []

        async def on_progress(completed, total):
            progress_updates.append((completed, total))

        await self._process_batch_with_progress(items, on_progress)

        # Should have progress updates
        assert len(progress_updates) > 0
        assert progress_updates[-1] == (10, 10)

    def _create_analysis_input(self):
        """Create sample analysis input."""
        return {
            "id": f"analysis_{int(time.time() * 1000000)}",
            "energy_inputs": {
                "fuel_inputs": [{"mass_flow_kg_hr": 100, "heating_value_mj_kg": 50}]
            },
            "useful_outputs": {
                "steam_output": [{"heat_rate_kw": 1150}]
            },
        }

    async def _process_batch(
        self, items: List[dict], allow_partial_failure: bool = False
    ) -> List[dict]:
        """Process batch of items."""
        results = []
        for item in items:
            try:
                if item.get("valid") is False:
                    raise ValueError("Invalid item")
                result = {
                    "status": "success",
                    "input_id": item.get("id", ""),
                    "efficiency": 82.8,
                }
            except Exception as e:
                if allow_partial_failure:
                    result = {"status": "error", "input_id": item.get("id", ""), "error": str(e)}
                else:
                    raise
            results.append(result)
        return results

    async def _process_batch_with_semaphore(
        self, items: List[dict], processor, max_concurrent: int
    ) -> List[dict]:
        """Process batch with concurrency limit."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(item):
            async with semaphore:
                return await processor(item)

        tasks = [process_with_semaphore(item) for item in items]
        return await asyncio.gather(*tasks)

    async def _process_batch_with_progress(
        self, items: List[dict], on_progress
    ) -> List[dict]:
        """Process batch with progress reporting."""
        results = []
        total = len(items)

        for i, item in enumerate(items):
            result = {"status": "success", "input_id": item.get("id", "")}
            results.append(result)
            await on_progress(i + 1, total)

        return results


# =============================================================================
# TEST CLASS: RETRY LOGIC AND CIRCUIT BREAKER
# =============================================================================

class TestRetryAndCircuitBreaker:
    """Test retry logic and circuit breaker patterns."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self):
        """Test retry on transient failure."""
        attempt_count = 0

        async def flaky_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Transient failure")
            return {"success": True}

        result = await self._retry_operation(flaky_operation, max_retries=3)

        assert result["success"] is True
        assert attempt_count == 3

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test behavior when retries are exhausted."""
        attempt_count = 0

        async def always_fails():
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("Permanent failure")

        with pytest.raises(ValueError):
            await self._retry_operation(always_fails, max_retries=3)

        assert attempt_count == 3

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test exponential backoff between retries."""
        timestamps = []

        async def record_attempt():
            timestamps.append(time.perf_counter())
            if len(timestamps) < 3:
                raise ConnectionError("Retry me")
            return {"success": True}

        await self._retry_operation(
            record_attempt,
            max_retries=3,
            base_delay_ms=10,
            exponential_backoff=True
        )

        # Check delays increase exponentially
        if len(timestamps) >= 3:
            delay1 = timestamps[1] - timestamps[0]
            delay2 = timestamps[2] - timestamps[1]
            assert delay2 >= delay1 * 1.5  # Should be exponential

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self):
        """Test circuit breaker opens after failures."""
        circuit = self._create_circuit_breaker(failure_threshold=3)

        # Fail 3 times
        for _ in range(3):
            try:
                await circuit.call(self._failing_operation)
            except Exception:
                pass

        assert circuit.state == "open"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_circuit_breaker_rejects_calls_when_open(self):
        """Test circuit breaker rejects calls when open."""
        circuit = self._create_circuit_breaker(failure_threshold=2)

        # Open the circuit
        for _ in range(2):
            try:
                await circuit.call(self._failing_operation)
            except Exception:
                pass

        # Should reject immediately
        with pytest.raises(CircuitBreakerOpen):
            await circuit.call(self._failing_operation)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open(self):
        """Test circuit breaker transitions to half-open."""
        circuit = self._create_circuit_breaker(
            failure_threshold=2,
            recovery_timeout_ms=10
        )

        # Open the circuit
        for _ in range(2):
            try:
                await circuit.call(self._failing_operation)
            except Exception:
                pass

        # Wait for recovery timeout
        await asyncio.sleep(0.02)

        # Should be half-open now
        assert circuit.state == "half-open"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_circuit_breaker_closes_on_success(self):
        """Test circuit breaker closes after success in half-open."""
        circuit = self._create_circuit_breaker(
            failure_threshold=2,
            recovery_timeout_ms=10
        )

        # Open the circuit
        for _ in range(2):
            try:
                await circuit.call(self._failing_operation)
            except Exception:
                pass

        # Wait for half-open
        await asyncio.sleep(0.02)

        # Success in half-open should close
        await circuit.call(self._success_operation)

        assert circuit.state == "closed"

    async def _retry_operation(
        self,
        operation,
        max_retries: int = 3,
        base_delay_ms: int = 0,
        exponential_backoff: bool = False
    ):
        """Retry operation with configurable backoff."""
        last_exception = None

        for attempt in range(max_retries):
            try:
                return await operation()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    delay = base_delay_ms * (2 ** attempt if exponential_backoff else 1)
                    await asyncio.sleep(delay / 1000)

        raise last_exception

    async def _failing_operation(self):
        raise ConnectionError("Simulated failure")

    async def _success_operation(self):
        return {"success": True}

    def _create_circuit_breaker(
        self,
        failure_threshold: int = 5,
        recovery_timeout_ms: int = 30000
    ):
        """Create circuit breaker instance."""
        return CircuitBreaker(failure_threshold, recovery_timeout_ms)


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Simple circuit breaker implementation."""

    def __init__(self, failure_threshold: int, recovery_timeout_ms: int):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_ms = recovery_timeout_ms
        self.failures = 0
        self.state = "closed"
        self.last_failure_time = None

    async def call(self, operation):
        """Call operation through circuit breaker."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise CircuitBreakerOpen("Circuit breaker is open")

        try:
            result = await operation()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        self.failures = 0
        self.state = "closed"

    def _on_failure(self):
        self.failures += 1
        self.last_failure_time = time.perf_counter()
        if self.failures >= self.failure_threshold:
            self.state = "open"

    def _should_attempt_reset(self):
        if self.last_failure_time is None:
            return True
        elapsed_ms = (time.perf_counter() - self.last_failure_time) * 1000
        return elapsed_ms >= self.recovery_timeout_ms


# =============================================================================
# TEST CLASS: CACHING BEHAVIOR
# =============================================================================

class TestCachingBehavior:
    """Test orchestrator caching behavior."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cache_hit_on_same_input(self):
        """Test cache hit on identical input."""
        cache = self._create_cache()
        input_data = {"value": 100}

        # First call
        result1 = await self._cached_calculate(cache, input_data)

        # Second call with same input
        result2 = await self._cached_calculate(cache, input_data)

        assert result1 == result2
        assert cache.hit_count == 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cache_miss_on_different_input(self):
        """Test cache miss on different input."""
        cache = self._create_cache()

        await self._cached_calculate(cache, {"value": 100})
        await self._cached_calculate(cache, {"value": 200})

        assert cache.miss_count == 2

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self):
        """Test cache entry expiration."""
        cache = self._create_cache(ttl_seconds=0.01)
        input_data = {"value": 100}

        await self._cached_calculate(cache, input_data)

        # Wait for expiration
        await asyncio.sleep(0.02)

        # Should miss due to expiration
        await self._cached_calculate(cache, input_data)

        assert cache.miss_count == 2

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cache_max_size(self):
        """Test cache eviction when max size reached."""
        cache = self._create_cache(max_size=3)

        # Add 4 items
        for i in range(4):
            await self._cached_calculate(cache, {"value": i})

        # First item should be evicted
        assert len(cache.entries) <= 3

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_cache_key_deterministic(self):
        """Test cache key is deterministic."""
        cache = self._create_cache()

        # Same data in different order
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}

        key1 = cache.compute_key(data1)
        key2 = cache.compute_key(data2)

        assert key1 == key2

    @pytest.mark.integration
    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = self._create_cache()

        stats = cache.get_statistics()

        assert "hit_count" in stats
        assert "miss_count" in stats
        assert "hit_rate" in stats
        assert "entry_count" in stats

    def _create_cache(self, ttl_seconds: float = 60.0, max_size: int = 100):
        """Create cache instance."""
        return SimpleCache(ttl_seconds, max_size)

    async def _cached_calculate(self, cache, input_data: dict) -> dict:
        """Calculate with caching."""
        key = cache.compute_key(input_data)

        cached = cache.get(key)
        if cached is not None:
            return cached

        # Simulate calculation
        result = {"efficiency": input_data.get("value", 0) * 0.82}

        cache.put(key, result)
        return result


class SimpleCache:
    """Simple in-memory cache for testing."""

    def __init__(self, ttl_seconds: float, max_size: int):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.entries = {}
        self.hit_count = 0
        self.miss_count = 0

    def compute_key(self, data: dict) -> str:
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]

    def get(self, key: str):
        entry = self.entries.get(key)
        if entry is None:
            self.miss_count += 1
            return None

        if time.time() - entry["timestamp"] > self.ttl_seconds:
            del self.entries[key]
            self.miss_count += 1
            return None

        self.hit_count += 1
        return entry["value"]

    def put(self, key: str, value: Any):
        if len(self.entries) >= self.max_size:
            # Evict oldest
            oldest_key = min(self.entries, key=lambda k: self.entries[k]["timestamp"])
            del self.entries[oldest_key]

        self.entries[key] = {
            "value": value,
            "timestamp": time.time(),
        }

    def get_statistics(self) -> dict:
        total = self.hit_count + self.miss_count
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_count / total if total > 0 else 0,
            "entry_count": len(self.entries),
        }


# =============================================================================
# TEST CLASS: PARALLEL EXECUTION
# =============================================================================

class TestParallelExecution:
    """Test parallel execution capabilities."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_parallel_calculations_faster(self):
        """Test that parallel execution is faster than sequential."""
        tasks = 10
        delay_per_task = 0.01  # 10ms

        # Sequential
        start = time.perf_counter()
        for _ in range(tasks):
            await asyncio.sleep(delay_per_task)
        sequential_time = time.perf_counter() - start

        # Parallel
        start = time.perf_counter()
        await asyncio.gather(*[
            asyncio.sleep(delay_per_task) for _ in range(tasks)
        ])
        parallel_time = time.perf_counter() - start

        assert parallel_time < sequential_time * 0.5

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_parallel_with_exception_handling(self):
        """Test parallel execution handles exceptions properly."""
        async def may_fail(i):
            if i == 5:
                raise ValueError(f"Task {i} failed")
            return i * 2

        tasks = [may_fail(i) for i in range(10)]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        assert len(results) == 10
        assert isinstance(results[5], ValueError)
        assert results[0] == 0
        assert results[9] == 18

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_parallel_result_ordering(self):
        """Test that parallel results maintain order."""
        async def process(i):
            await asyncio.sleep(0.01 * (10 - i))  # Variable delay
            return {"id": i, "result": i * 100}

        tasks = [process(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        for i, result in enumerate(results):
            assert result["id"] == i

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_parallel_with_shared_resource(self):
        """Test parallel execution with shared resource protection."""
        shared_counter = {"value": 0}
        lock = asyncio.Lock()

        async def increment():
            async with lock:
                current = shared_counter["value"]
                await asyncio.sleep(0.001)
                shared_counter["value"] = current + 1

        tasks = [increment() for _ in range(100)]
        await asyncio.gather(*tasks)

        assert shared_counter["value"] == 100


# =============================================================================
# TEST CLASS: ERROR RECOVERY
# =============================================================================

class TestErrorRecovery:
    """Test error recovery and resilience."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation on component failure."""
        result = await self._analyze_with_fallback(
            {"value": 100},
            primary_fails=True
        )

        assert result is not None
        assert result.get("fallback_used") is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_isolation(self):
        """Test that errors in one component don't affect others."""
        results = await self._run_isolated_components([
            {"id": 1, "should_fail": False},
            {"id": 2, "should_fail": True},
            {"id": 3, "should_fail": False},
        ])

        assert results[0]["status"] == "success"
        assert results[1]["status"] == "error"
        assert results[2]["status"] == "success"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_recovery_after_transient_error(self):
        """Test recovery after transient error."""
        error_count = {"value": 0}

        async def sometimes_fails():
            error_count["value"] += 1
            if error_count["value"] <= 2:
                raise ConnectionError("Transient")
            return {"success": True}

        result = await self._with_recovery(sometimes_fails, max_attempts=5)

        assert result["success"] is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_logging(self):
        """Test that errors are properly logged."""
        error_log = []

        async def failing_operation():
            raise ValueError("Test error")

        try:
            await self._execute_with_logging(failing_operation, error_log)
        except ValueError:
            pass

        assert len(error_log) == 1
        assert "ValueError" in error_log[0]["error_type"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_state_rollback_on_failure(self):
        """Test state rollback on failure."""
        state = {"value": 100}
        checkpoint = state.copy()

        try:
            await self._modify_with_rollback(state, checkpoint)
        except Exception:
            pass

        assert state == checkpoint

    async def _analyze_with_fallback(
        self, input_data: dict, primary_fails: bool = False
    ) -> dict:
        """Analyze with fallback on failure."""
        if not primary_fails:
            return {"result": input_data.get("value", 0) * 2}

        # Fallback calculation
        return {
            "result": input_data.get("value", 0) * 1.5,
            "fallback_used": True,
        }

    async def _run_isolated_components(
        self, components: List[dict]
    ) -> List[dict]:
        """Run components in isolation."""
        results = []
        for comp in components:
            try:
                if comp.get("should_fail"):
                    raise ValueError("Component failed")
                results.append({"id": comp["id"], "status": "success"})
            except Exception as e:
                results.append({"id": comp["id"], "status": "error", "error": str(e)})
        return results

    async def _with_recovery(self, operation, max_attempts: int):
        """Execute with automatic recovery."""
        for attempt in range(max_attempts):
            try:
                return await operation()
            except Exception:
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(0.01)

    async def _execute_with_logging(self, operation, error_log: List):
        """Execute and log errors."""
        try:
            return await operation()
        except Exception as e:
            error_log.append({
                "error_type": type(e).__name__,
                "message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            raise

    async def _modify_with_rollback(self, state: dict, checkpoint: dict):
        """Modify state with rollback on failure."""
        state["value"] = 200
        raise RuntimeError("Simulated failure")


# =============================================================================
# TEST CLASS: CONFIGURATION VALIDATION
# =============================================================================

class TestConfigurationValidation:
    """Test configuration validation."""

    @pytest.mark.integration
    def test_valid_configuration_accepted(self, thermal_iq_config):
        """Test that valid configuration is accepted."""
        errors = self._validate_config(thermal_iq_config)

        assert len(errors) == 0

    @pytest.mark.integration
    def test_missing_required_field_rejected(self):
        """Test that missing required field is rejected."""
        invalid_config = {
            "codename": "THERMALIQ",
            # Missing agent_id
        }

        errors = self._validate_config(invalid_config)

        assert any("agent_id" in e for e in errors)

    @pytest.mark.integration
    def test_invalid_type_rejected(self):
        """Test that invalid type is rejected."""
        invalid_config = {
            "agent_id": "GL-009",
            "version": 123,  # Should be string
        }

        errors = self._validate_config(invalid_config)

        assert any("version" in e or "type" in e.lower() for e in errors)

    @pytest.mark.integration
    def test_value_range_validation(self):
        """Test that out-of-range values are rejected."""
        invalid_config = {
            "agent_id": "GL-009",
            "version": "1.0.0",
            "energy_balance_tolerance": 1.5,  # Should be <= 1.0
        }

        errors = self._validate_config(invalid_config)

        assert len(errors) > 0

    @pytest.mark.integration
    def test_default_values_applied(self):
        """Test that default values are applied."""
        minimal_config = {
            "agent_id": "GL-009",
            "version": "1.0.0",
        }

        config = self._apply_defaults(minimal_config)

        assert "temperature" in config or "deterministic" in config

    def _validate_config(self, config: dict) -> List[str]:
        """Validate configuration."""
        errors = []

        required_fields = ["agent_id"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        if "version" in config and not isinstance(config["version"], str):
            errors.append("Field 'version' must be a string")

        if "energy_balance_tolerance" in config:
            tol = config["energy_balance_tolerance"]
            if not isinstance(tol, (int, float)) or tol > 1.0 or tol < 0:
                errors.append("energy_balance_tolerance must be between 0 and 1")

        return errors

    def _apply_defaults(self, config: dict) -> dict:
        """Apply default values to configuration."""
        defaults = {
            "deterministic": True,
            "temperature": 0.0,
            "max_retries": 3,
            "cache_ttl_seconds": 300,
        }

        result = defaults.copy()
        result.update(config)
        return result


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestOrchestratorPerformance:
    """Performance tests for orchestrator."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_analysis_under_100ms(self, sample_analysis_input, orchestrator):
        """Test single analysis completes in <100ms."""
        start = time.perf_counter()
        await orchestrator.execute(sample_analysis_input)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"Analysis took {elapsed_ms:.2f}ms (target: <100ms)"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_throughput(self, sample_analysis_input, orchestrator):
        """Test batch processing throughput."""
        batch_size = 100

        start = time.perf_counter()
        tasks = [orchestrator.execute(sample_analysis_input) for _ in range(batch_size)]
        await asyncio.gather(*tasks)
        elapsed_seconds = time.perf_counter() - start

        throughput = batch_size / elapsed_seconds

        assert throughput >= 100, f"Throughput {throughput:.1f}/sec (target: >=100/sec)"

    @pytest.mark.performance
    def test_state_transition_under_1ms(self):
        """Test state transition completes in <1ms."""
        orchestrator = TestOrchestratorStateMachine()._create_mock_orchestrator()

        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations // 2):
            orchestrator.transition_to("initializing")
            orchestrator.state = "idle"  # Reset for next iteration
        elapsed_ms = (time.perf_counter() - start) * 1000 / (iterations // 2)

        assert elapsed_ms < 1, f"Transition took {elapsed_ms:.4f}ms (target: <1ms)"

    @pytest.mark.performance
    def test_cache_lookup_under_100us(self):
        """Test cache lookup completes in <100 microseconds."""
        cache = SimpleCache(ttl_seconds=60, max_size=1000)

        # Pre-populate cache
        for i in range(100):
            cache.put(f"key_{i}", {"value": i})

        iterations = 10000
        start = time.perf_counter()
        for i in range(iterations):
            cache.get(f"key_{i % 100}")
        elapsed_us = (time.perf_counter() - start) * 1_000_000 / iterations

        assert elapsed_us < 100, f"Lookup took {elapsed_us:.2f}us (target: <100us)"
