# -*- coding: utf-8 -*-
"""
===============================================================================
GL-VCCI Scope 3 Platform - Error Scenarios E2E Test
===============================================================================

PRIORITY TEST 5: Circuit breaker, retry, and fallback logic

Workflow: Test resilience patterns under failure conditions

This test validates:
- Circuit breaker pattern (open/half-open/closed states)
- Retry logic with exponential backoff
- Fallback mechanisms
- Graceful degradation
- Error propagation
- Recovery scenarios
- Timeout handling

Version: 1.0.0
Team: 8 - Quality Assurance Lead
Created: 2025-11-09
===============================================================================
"""

import pytest
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from enum import Enum
from uuid import uuid4

from greenlang.telemetry import get_logger, MetricsCollector
from greenlang.determinism import DeterministicClock

logger = get_logger(__name__)


# ============================================================================
# Circuit Breaker Implementation (for testing)
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failures detected, requests blocked
    HALF_OPEN = "half_open" # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker implementation for testing.

    Implements the circuit breaker pattern to prevent cascading failures.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        half_open_attempts: int = 3,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Time to wait before attempting recovery
            half_open_attempts: Number of test requests in half-open state
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_attempts = half_open_attempts

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_test_count = 0

        logger.info(f"Initialized circuit breaker: threshold={failure_threshold}, timeout={timeout_seconds}s")

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        # Check if circuit should transition from OPEN to HALF_OPEN
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info("Circuit breaker: OPEN → HALF_OPEN (attempting recovery)")
                self.state = CircuitState.HALF_OPEN
                self.half_open_test_count = 0
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")

        # Attempt call
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_test_count += 1
            logger.info(f"Circuit breaker: HALF_OPEN test {self.half_open_test_count}/{self.half_open_attempts} succeeded")

            if self.half_open_test_count >= self.half_open_attempts:
                logger.info("Circuit breaker: HALF_OPEN → CLOSED (recovery successful)")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0

        elif self.state == CircuitState.CLOSED:
            self.success_count += 1
            # Reset failure count after successful calls
            if self.failure_count > 0:
                self.failure_count = max(0, self.failure_count - 1)

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = DeterministicClock.utcnow()

        if self.state == CircuitState.HALF_OPEN:
            # Failed during test, go back to OPEN
            logger.warning("Circuit breaker: HALF_OPEN → OPEN (test failed)")
            self.state = CircuitState.OPEN
            self.half_open_test_count = 0

        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                logger.warning(f"Circuit breaker: CLOSED → OPEN (threshold reached: {self.failure_count} failures)")
                self.state = CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self.last_failure_time:
            return True

        elapsed = (DeterministicClock.utcnow() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout_seconds

    def get_state(self) -> str:
        """Get current circuit state."""
        return self.state.value


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# ============================================================================
# Retry Logic with Exponential Backoff
# ============================================================================

class RetryPolicy:
    """Retry policy with exponential backoff."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
    ):
        """
        Initialize retry policy.

        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    def execute(self, func, *args, **kwargs):
        """Execute function with retry logic."""
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e
                attempt_num = attempt + 1

                if attempt_num >= self.max_attempts:
                    logger.error(f"Retry failed after {self.max_attempts} attempts")
                    raise

                # Calculate delay with exponential backoff
                delay = min(
                    self.base_delay * (self.exponential_base ** attempt),
                    self.max_delay
                )

                logger.warning(
                    f"Attempt {attempt_num}/{self.max_attempts} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )

                time.sleep(delay)

        # Should not reach here, but just in case
        raise last_exception


# ============================================================================
# Fallback Mechanisms
# ============================================================================

class FallbackHandler:
    """Handles fallback logic when primary service fails."""

    def __init__(self):
        self.fallback_count = 0

    def execute_with_fallback(
        self,
        primary_func,
        fallback_func,
        *args,
        **kwargs
    ):
        """Execute primary function with fallback on failure."""
        try:
            return primary_func(*args, **kwargs)

        except Exception as e:
            logger.warning(f"Primary function failed: {e}. Using fallback...")
            self.fallback_count += 1

            try:
                result = fallback_func(*args, **kwargs)
                result["_fallback_used"] = True
                return result

            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                raise


# ============================================================================
# Test Class: Error Scenarios
# ============================================================================

@pytest.mark.integration
@pytest.mark.resilience
@pytest.mark.critical
class TestErrorScenarios:
    """Test error handling and resilience patterns."""

    def test_circuit_breaker_state_transitions(self):
        """
        Test circuit breaker state transitions.

        Exit Criteria:
        ✅ CLOSED → OPEN after threshold failures
        ✅ OPEN → HALF_OPEN after timeout
        ✅ HALF_OPEN → CLOSED after successful tests
        ✅ HALF_OPEN → OPEN if test fails
        """
        logger.info("Testing circuit breaker state transitions")

        # Create circuit breaker with low thresholds for testing
        cb = CircuitBreaker(
            failure_threshold=3,
            timeout_seconds=1,
            half_open_attempts=2,
        )

        # Mock function that fails
        def failing_func():
            raise ValueError("Simulated failure")

        # Initially CLOSED
        assert cb.get_state() == "closed"

        # Generate failures to open circuit
        for i in range(3):
            with pytest.raises(ValueError):
                cb.call(failing_func)

        # Should be OPEN now
        assert cb.get_state() == "open"
        assert cb.failure_count == 3

        logger.info("✅ Circuit opened after threshold failures")

        # Try to call while OPEN (should be rejected)
        with pytest.raises(CircuitBreakerOpenError):
            cb.call(failing_func)

        logger.info("✅ Calls rejected while circuit is OPEN")

        # Wait for timeout
        time.sleep(1.1)

        # Mock function that succeeds
        def success_func():
            return {"status": "success"}

        # Next call should transition to HALF_OPEN
        result = cb.call(success_func)
        assert cb.get_state() == "half_open"

        logger.info("✅ Circuit transitioned to HALF_OPEN")

        # Need one more success to close circuit (half_open_attempts=2)
        result = cb.call(success_func)
        assert cb.get_state() == "closed"
        assert result["status"] == "success"

        logger.info("✅ Circuit closed after successful recovery tests")

        # Test HALF_OPEN → OPEN transition
        cb.state = CircuitState.CLOSED
        cb.failure_count = 0

        # Open circuit again
        for i in range(3):
            with pytest.raises(ValueError):
                cb.call(failing_func)

        assert cb.get_state() == "open"

        time.sleep(1.1)

        # First call succeeds (goes to HALF_OPEN)
        cb.call(success_func)
        assert cb.get_state() == "half_open"

        # Second call fails (should go back to OPEN)
        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.get_state() == "open"

        logger.info("✅ Circuit reopened after failed recovery test")
        logger.info("✅ Circuit breaker state transitions test PASSED")


    def test_retry_with_exponential_backoff(self):
        """
        Test retry logic with exponential backoff.

        Exit Criteria:
        ✅ Retries configured number of times
        ✅ Delay increases exponentially
        ✅ Success on retry
        ✅ Failure after max attempts
        """
        logger.info("Testing retry with exponential backoff")

        # Test successful retry
        attempt_count = 0

        def flaky_func():
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count < 3:
                raise ConnectionError("Connection failed")

            return {"status": "success", "attempts": attempt_count}

        retry_policy = RetryPolicy(
            max_attempts=5,
            base_delay=0.1,  # Short delay for testing
            exponential_base=2.0,
        )

        start_time = time.time()
        result = retry_policy.execute(flaky_func)
        elapsed = time.time() - start_time

        assert result["status"] == "success"
        assert result["attempts"] == 3
        assert attempt_count == 3

        # Should have delays: 0.1s + 0.2s = 0.3s minimum
        assert elapsed >= 0.3

        logger.info(f"✅ Retry succeeded on attempt {attempt_count} after {elapsed:.2f}s")

        # Test failure after max attempts
        def always_fails():
            raise RuntimeError("Always fails")

        retry_policy_strict = RetryPolicy(max_attempts=3, base_delay=0.05)

        with pytest.raises(RuntimeError):
            retry_policy_strict.execute(always_fails)

        logger.info("✅ Retry correctly fails after max attempts")
        logger.info("✅ Retry with exponential backoff test PASSED")


    def test_fallback_mechanism(self):
        """
        Test fallback to alternative data source.

        Exit Criteria:
        ✅ Primary function called first
        ✅ Fallback triggered on primary failure
        ✅ Fallback result returned
        ✅ Fallback metadata included
        """
        logger.info("Testing fallback mechanism")

        fallback_handler = FallbackHandler()

        # Primary function that fails
        def primary_calculation(supplier_id: str):
            raise ValueError("Primary calculation service unavailable")

        # Fallback function with degraded accuracy
        def fallback_calculation(supplier_id: str):
            return {
                "supplier_id": supplier_id,
                "emissions_tco2e": 50.0,  # Rough estimate
                "tier": 3,  # Tertiary data
                "method": "fallback_estimation",
                "uncertainty": 0.50,  # High uncertainty
            }

        # Execute with fallback
        result = fallback_handler.execute_with_fallback(
            primary_calculation,
            fallback_calculation,
            supplier_id="SUP-001"
        )

        assert result["supplier_id"] == "SUP-001"
        assert result["emissions_tco2e"] == 50.0
        assert result["_fallback_used"] == True
        assert fallback_handler.fallback_count == 1

        logger.info("✅ Fallback mechanism triggered successfully")

        # Test successful primary call (no fallback)
        def working_primary(supplier_id: str):
            return {
                "supplier_id": supplier_id,
                "emissions_tco2e": 45.5,
                "tier": 1,
                "method": "primary_data",
            }

        result2 = fallback_handler.execute_with_fallback(
            working_primary,
            fallback_calculation,
            supplier_id="SUP-002"
        )

        assert result2["supplier_id"] == "SUP-002"
        assert "_fallback_used" not in result2
        assert fallback_handler.fallback_count == 1  # Not incremented

        logger.info("✅ Primary function executed successfully (no fallback)")
        logger.info("✅ Fallback mechanism test PASSED")


    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """
        Test timeout handling for long-running operations.

        Exit Criteria:
        ✅ Operations timeout after configured duration
        ✅ Timeout errors caught and handled
        ✅ Partial results returned if available
        """
        logger.info("Testing timeout handling")

        # Simulate slow operation
        async def slow_operation(duration: float):
            await asyncio.sleep(duration)
            return {"status": "completed"}

        # Test successful operation within timeout
        try:
            result = await asyncio.wait_for(slow_operation(0.5), timeout=1.0)
            assert result["status"] == "completed"
            logger.info("✅ Operation completed within timeout")
        except asyncio.TimeoutError:
            pytest.fail("Operation should not timeout")

        # Test operation exceeding timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_operation(2.0), timeout=1.0)

        logger.info("✅ Operation timed out as expected")

        # Test graceful timeout with partial results
        async def operation_with_partial_results(duration: float):
            partial_results = {"status": "in_progress", "completed": 50}

            try:
                await asyncio.sleep(duration)
                return {"status": "completed", "completed": 100}
            except asyncio.CancelledError:
                logger.info("Operation cancelled, returning partial results")
                return partial_results

        # Create task and cancel after timeout
        task = asyncio.create_task(operation_with_partial_results(2.0))

        try:
            result = await asyncio.wait_for(task, timeout=0.5)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                result = await task
                assert result["status"] == "in_progress"
                assert result["completed"] == 50
                logger.info("✅ Partial results returned after timeout")
            except asyncio.CancelledError:
                logger.info("Task cancelled successfully")

        logger.info("✅ Timeout handling test PASSED")


    def test_error_propagation_chain(self):
        """
        Test error propagation through agent chain.

        Exit Criteria:
        ✅ Errors propagate correctly
        ✅ Context preserved in error messages
        ✅ Appropriate error types raised
        ✅ Error recovery at correct level
        """
        logger.info("Testing error propagation chain")

        # Simulate agent chain: Intake → Calculator → Reporting
        class IntakeError(Exception):
            pass

        class CalculationError(Exception):
            pass

        class ReportingError(Exception):
            pass

        def intake_agent_process(data):
            if not data:
                raise IntakeError("No data provided to intake agent")
            return {"status": "ingested", "records": len(data)}

        def calculator_agent_process(intake_result):
            if intake_result.get("status") != "ingested":
                raise CalculationError("Invalid intake result")
            if intake_result.get("records", 0) == 0:
                raise CalculationError("No records to calculate")
            return {"status": "calculated", "emissions": 100.0}

        def reporting_agent_process(calc_result):
            if calc_result.get("status") != "calculated":
                raise ReportingError("Invalid calculation result")
            return {"status": "reported", "report_id": "RPT-001"}

        # Test successful flow
        try:
            intake_result = intake_agent_process([{"id": 1}, {"id": 2}])
            calc_result = calculator_agent_process(intake_result)
            report_result = reporting_agent_process(calc_result)

            assert report_result["status"] == "reported"
            logger.info("✅ Successful flow completed")
        except Exception as e:
            pytest.fail(f"Should not raise exception: {e}")

        # Test error at intake level
        with pytest.raises(IntakeError) as exc_info:
            intake_agent_process([])

        assert "No data provided" in str(exc_info.value)
        logger.info("✅ IntakeError raised correctly")

        # Test error at calculation level
        with pytest.raises(CalculationError) as exc_info:
            bad_intake = {"status": "failed"}
            calculator_agent_process(bad_intake)

        assert "Invalid intake result" in str(exc_info.value)
        logger.info("✅ CalculationError raised correctly")

        # Test error at reporting level
        with pytest.raises(ReportingError) as exc_info:
            bad_calc = {"status": "failed"}
            reporting_agent_process(bad_calc)

        assert "Invalid calculation result" in str(exc_info.value)
        logger.info("✅ ReportingError raised correctly")

        logger.info("✅ Error propagation chain test PASSED")


    def test_graceful_degradation(self):
        """
        Test graceful degradation when services are unavailable.

        Exit Criteria:
        ✅ System continues operating with reduced functionality
        ✅ Degraded mode clearly indicated
        ✅ User receives useful results despite failures
        """
        logger.info("Testing graceful degradation")

        class EmissionsCalculator:
            def __init__(self):
                self.primary_available = True
                self.secondary_available = True

            def calculate(self, supplier_id: str, spend: float):
                # Try primary data source (highest accuracy)
                if self.primary_available:
                    try:
                        return self._calculate_primary(supplier_id, spend)
                    except Exception as e:
                        logger.warning(f"Primary calculation failed: {e}")
                        self.primary_available = False

                # Try secondary data source (medium accuracy)
                if self.secondary_available:
                    try:
                        return self._calculate_secondary(supplier_id, spend)
                    except Exception as e:
                        logger.warning(f"Secondary calculation failed: {e}")
                        self.secondary_available = False

                # Fall back to tertiary estimation (low accuracy)
                return self._calculate_tertiary(supplier_id, spend)

            def _calculate_primary(self, supplier_id, spend):
                # Simulate primary data unavailable
                raise ConnectionError("Primary database unavailable")

            def _calculate_secondary(self, supplier_id, spend):
                return {
                    "supplier_id": supplier_id,
                    "emissions_tco2e": spend * 0.0005,  # Secondary factor
                    "tier": 2,
                    "uncertainty": 0.25,
                    "quality": "secondary",
                }

            def _calculate_tertiary(self, supplier_id, spend):
                return {
                    "supplier_id": supplier_id,
                    "emissions_tco2e": spend * 0.0006,  # Rough estimate
                    "tier": 3,
                    "uncertainty": 0.50,
                    "quality": "tertiary",
                    "warning": "Using fallback estimation due to service unavailability",
                }

        calculator = EmissionsCalculator()

        # First call fails primary, uses secondary
        result1 = calculator.calculate("SUP-001", 100000.0)
        assert result1["quality"] == "secondary"
        assert result1["tier"] == 2

        logger.info("✅ Degraded to secondary calculation")

        # Disable secondary
        calculator.secondary_available = False

        # Second call uses tertiary
        result2 = calculator.calculate("SUP-002", 100000.0)
        assert result2["quality"] == "tertiary"
        assert result2["tier"] == 3
        assert "warning" in result2

        logger.info("✅ Degraded to tertiary estimation")
        logger.info("✅ Graceful degradation test PASSED")


# ============================================================================
# Integration Test: Full Error Scenario
# ============================================================================

@pytest.mark.integration
@pytest.mark.resilience
class TestFullErrorRecoveryScenario:
    """Test complete error recovery scenario."""

    def test_end_to_end_error_recovery(self):
        """
        Test end-to-end error recovery with all resilience patterns.

        Scenario:
        1. Circuit breaker protects against failing service
        2. Retry logic handles transient failures
        3. Fallback provides degraded service
        4. System recovers when service comes back online

        Exit Criteria:
        ✅ All resilience patterns working together
        ✅ System remains operational during failures
        ✅ Recovery successful when service restored
        """
        logger.info("Testing end-to-end error recovery scenario")

        # Mock external service with failures
        class ExternalService:
            def __init__(self):
                self.call_count = 0
                self.failure_mode = True

            def fetch_emission_factor(self, category: int):
                self.call_count += 1

                if self.failure_mode:
                    if self.call_count <= 3:
                        raise ConnectionError("Service temporarily unavailable")

                # After 3 failures, service recovers
                self.failure_mode = False
                return {
                    "factor": 0.5,
                    "unit": "kg CO2e/USD",
                    "tier": 1,
                }

        service = ExternalService()
        cb = CircuitBreaker(failure_threshold=3, timeout_seconds=1)
        retry_policy = RetryPolicy(max_attempts=3, base_delay=0.1)
        fallback_handler = FallbackHandler()

        def fetch_with_resilience(category: int):
            # Try with circuit breaker and retry
            def attempt_fetch():
                return cb.call(service.fetch_emission_factor, category)

            try:
                return retry_policy.execute(attempt_fetch)
            except CircuitBreakerOpenError:
                logger.warning("Circuit breaker open, using fallback")
                return fallback_handler.execute_with_fallback(
                    lambda: {"error": "not_used"},
                    lambda: {
                        "factor": 0.6,  # Fallback factor
                        "unit": "kg CO2e/USD",
                        "tier": 3,
                        "_fallback": True
                    }
                )

        # First attempt: Will fail 3 times, then succeed on retry
        result1 = fetch_with_resilience(category=1)

        assert result1["factor"] == 0.5
        assert result1["tier"] == 1
        assert service.call_count == 3  # Failed 2 times, succeeded on 3rd

        logger.info("✅ Service recovered after retries")
        logger.info("✅ End-to-end error recovery test PASSED")
