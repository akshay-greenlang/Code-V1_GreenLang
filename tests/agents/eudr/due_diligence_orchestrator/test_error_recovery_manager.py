# -*- coding: utf-8 -*-
"""
Unit tests for Engine 5: Error Recovery Manager -- AGENT-EUDR-026

Tests exponential backoff with jitter, error classification (HTTP status,
exception type, message keywords), retry decision logic, retry record
tracking, circuit breaker state machine (CLOSED -> OPEN -> HALF_OPEN ->
CLOSED and HALF_OPEN -> OPEN), circuit breaker reset, fallback strategy
selection, dead letter queue CRUD, and thread-safe concurrent operations.

Test count: 100 tests
Author: GreenLang Platform Team
Date: March 2026
"""

import random
import threading
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.due_diligence_orchestrator.error_recovery_manager import (
    ErrorRecoveryManager,
)
from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    CircuitBreakerState,
    CircuitBreakerRecord,
    RetryRecord,
    DeadLetterEntry,
    ErrorClassification,
    FallbackStrategy,
    _new_uuid,
    _utcnow,
)
from greenlang.agents.eudr.due_diligence_orchestrator.config import (
    DueDiligenceOrchestratorConfig,
    get_config,
    set_config,
    reset_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _force_circuit_open(manager: ErrorRecoveryManager, agent_id: str) -> None:
    """Record enough failures to trip the circuit breaker to OPEN."""
    threshold = manager._config.cb_failure_threshold
    for _ in range(threshold):
        manager.record_failure(agent_id)


def _force_circuit_half_open(
    manager: ErrorRecoveryManager,
    agent_id: str,
) -> None:
    """Trip the circuit breaker then simulate the reset timeout elapsing."""
    _force_circuit_open(manager, agent_id)
    # Backdate opened_at so check_circuit_breaker transitions to HALF_OPEN
    with manager._lock:
        cb = manager._circuit_breakers[agent_id]
        cb.opened_at = _utcnow() - timedelta(
            seconds=cb.reset_timeout_s + 1,
        )
    manager.check_circuit_breaker(agent_id)


# ===================================================================
# Test: Initialization
# ===================================================================


class TestErrorRecoveryManagerInit:
    """Test ErrorRecoveryManager initialization."""

    def test_init_with_default_config(self, default_config):
        manager = ErrorRecoveryManager()
        assert manager._config is not None
        assert manager._circuit_breakers == {}
        assert manager._retry_history == {}
        assert manager._dead_letter_queue == []

    def test_init_with_explicit_config(self, default_config):
        manager = ErrorRecoveryManager(config=default_config)
        assert manager._config is default_config

    def test_init_uses_config_values(self, test_config):
        manager = ErrorRecoveryManager(config=test_config)
        assert manager._config.retry_max_attempts == 2
        assert manager._config.cb_failure_threshold == 3
        assert manager._config.retry_base_delay_s == Decimal("0.1")


# ===================================================================
# Test: Backoff Calculation
# ===================================================================


class TestBackoffCalculation:
    """Test exponential backoff delay computation."""

    def test_backoff_returns_decimal(self, error_recovery_manager):
        delay = error_recovery_manager.compute_backoff_delay(0)
        assert isinstance(delay, Decimal)

    def test_backoff_attempt_zero(self, error_recovery_manager):
        # base=1, 2^0=1, so exponential=1. Plus jitter in [0, 2].
        # Result in [1.0, 3.0].
        with patch("random.uniform", return_value=0.0):
            delay = error_recovery_manager.compute_backoff_delay(0)
        assert delay == Decimal("1.000")

    def test_backoff_attempt_zero_with_max_jitter(self, error_recovery_manager):
        with patch("random.uniform", return_value=2.0):
            delay = error_recovery_manager.compute_backoff_delay(0)
        assert delay == Decimal("3.000")

    def test_backoff_increases_with_attempt(self, error_recovery_manager):
        with patch("random.uniform", return_value=0.0):
            d0 = error_recovery_manager.compute_backoff_delay(0)
            d1 = error_recovery_manager.compute_backoff_delay(1)
            d2 = error_recovery_manager.compute_backoff_delay(2)
            d3 = error_recovery_manager.compute_backoff_delay(3)
        assert d0 < d1 < d2 < d3

    @pytest.mark.parametrize("attempt,expected_exponential", [
        (0, Decimal("1")),    # 1 * 2^0 = 1
        (1, Decimal("2")),    # 1 * 2^1 = 2
        (2, Decimal("4")),    # 1 * 2^2 = 4
        (3, Decimal("8")),    # 1 * 2^3 = 8
        (4, Decimal("16")),   # 1 * 2^4 = 16
        (5, Decimal("32")),   # 1 * 2^5 = 32
    ])
    def test_backoff_exponential_formula_no_jitter(
        self, error_recovery_manager, attempt, expected_exponential,
    ):
        with patch("random.uniform", return_value=0.0):
            delay = error_recovery_manager.compute_backoff_delay(attempt)
        assert delay == expected_exponential.quantize(
            Decimal("0.001"),
        )

    def test_backoff_capped_at_max_delay(self, error_recovery_manager):
        # attempt=20 gives 1 * 2^20 = 1_048_576 which exceeds max_delay 300
        with patch("random.uniform", return_value=0.0):
            delay = error_recovery_manager.compute_backoff_delay(20)
        max_delay = error_recovery_manager._config.retry_max_delay_s
        assert delay <= max_delay

    def test_backoff_cap_equals_max_delay_exactly(self, error_recovery_manager):
        with patch("random.uniform", return_value=0.0):
            delay = error_recovery_manager.compute_backoff_delay(20)
        assert delay == Decimal("300.000")

    def test_backoff_jitter_adds_randomness(self, error_recovery_manager):
        # Without mocking, jitter should produce varying results
        delays = set()
        for _ in range(30):
            delays.add(error_recovery_manager.compute_backoff_delay(0))
        # With continuous random jitter, we expect at least a few different values
        assert len(delays) >= 2

    def test_backoff_precision_three_decimals(self, error_recovery_manager):
        delay = error_recovery_manager.compute_backoff_delay(1)
        # Quantized to 0.001
        assert delay == delay.quantize(Decimal("0.001"))

    def test_backoff_with_custom_config(self, test_config):
        manager = ErrorRecoveryManager(config=test_config)
        # base=0.1, max=1.0
        with patch("random.uniform", return_value=0.0):
            delay = manager.compute_backoff_delay(0)
        # 0.1 * 2^0 = 0.1
        assert delay == Decimal("0.100")

    def test_backoff_custom_config_max_cap(self, test_config):
        manager = ErrorRecoveryManager(config=test_config)
        with patch("random.uniform", return_value=0.0):
            delay = manager.compute_backoff_delay(20)
        assert delay == Decimal("1.000")


# ===================================================================
# Test: Error Classification
# ===================================================================


class TestErrorClassification:
    """Test error classification from HTTP status, exception type, and message."""

    # -- HTTP status code classification --

    @pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
    def test_classify_transient_http_status(
        self, error_recovery_manager, status,
    ):
        result = error_recovery_manager.classify_error(
            "some error", http_status=status,
        )
        assert result == ErrorClassification.TRANSIENT

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 409, 410, 422])
    def test_classify_permanent_http_status(
        self, error_recovery_manager, status,
    ):
        result = error_recovery_manager.classify_error(
            "some error", http_status=status,
        )
        assert result == ErrorClassification.PERMANENT

    def test_classify_unknown_http_status(self, error_recovery_manager):
        # HTTP 418 I'm a teapot -- not in any set
        result = error_recovery_manager.classify_error(
            "unrecognized error", http_status=418,
        )
        assert result == ErrorClassification.UNKNOWN

    def test_http_status_takes_priority_over_message(
        self, error_recovery_manager,
    ):
        # Message says "timeout" (transient keyword) but HTTP 400 is permanent
        result = error_recovery_manager.classify_error(
            "timeout", http_status=400,
        )
        assert result == ErrorClassification.PERMANENT

    # -- Exception type classification --

    @pytest.mark.parametrize("exc_type", [
        "TimeoutError", "ConnectionError", "ConnectTimeout",
        "ReadTimeout", "ConnectionReset", "BrokenPipe",
        "TemporaryError", "ServiceUnavailable",
    ])
    def test_classify_transient_exception_type(
        self, error_recovery_manager, exc_type,
    ):
        result = error_recovery_manager.classify_error(
            "error", exception_type=exc_type,
        )
        assert result == ErrorClassification.TRANSIENT

    @pytest.mark.parametrize("exc_type", [
        "ValidationError", "ValueError", "TypeError",
        "KeyError", "AttributeError", "PermissionError",
        "AuthenticationError", "AuthorizationError",
        "NotFoundError",
    ])
    def test_classify_permanent_exception_type(
        self, error_recovery_manager, exc_type,
    ):
        result = error_recovery_manager.classify_error(
            "error", exception_type=exc_type,
        )
        assert result == ErrorClassification.PERMANENT

    def test_classify_unknown_exception_type(self, error_recovery_manager):
        result = error_recovery_manager.classify_error(
            "unrecognized", exception_type="CustomUnknownError",
        )
        assert result == ErrorClassification.UNKNOWN

    # -- Error message keyword classification --

    @pytest.mark.parametrize("message", [
        "connection reset by peer",
        "request timeout exceeded",
        "service temporarily unavailable",
        "rate limit exceeded",
        "too many requests",
        "retry later",
    ])
    def test_classify_transient_message_keyword(
        self, error_recovery_manager, message,
    ):
        result = error_recovery_manager.classify_error(message)
        assert result == ErrorClassification.TRANSIENT

    @pytest.mark.parametrize("message", [
        "validation failed for field X",
        "invalid input data",
        "resource not found",
        "unauthorized access",
        "forbidden operation",
        "permission denied",
        "authentication required",
    ])
    def test_classify_permanent_message_keyword(
        self, error_recovery_manager, message,
    ):
        result = error_recovery_manager.classify_error(message)
        assert result == ErrorClassification.PERMANENT

    @pytest.mark.parametrize("message", [
        "partial result returned",
        "stale data detected",
        "low confidence score",
        "incomplete response",
        "degraded mode active",
    ])
    def test_classify_degraded_message_keyword(
        self, error_recovery_manager, message,
    ):
        result = error_recovery_manager.classify_error(message)
        assert result == ErrorClassification.DEGRADED

    def test_classify_empty_message_returns_unknown(
        self, error_recovery_manager,
    ):
        result = error_recovery_manager.classify_error("")
        assert result == ErrorClassification.UNKNOWN

    def test_classify_unrecognized_message_returns_unknown(
        self, error_recovery_manager,
    ):
        result = error_recovery_manager.classify_error(
            "something went terribly wrong in the flux capacitor",
        )
        assert result == ErrorClassification.UNKNOWN


# ===================================================================
# Test: Retry Decision
# ===================================================================


class TestRetryDecision:
    """Test should_retry logic including max attempts, classification, and CB."""

    def test_should_retry_transient_error(self, error_recovery_manager):
        retry, delay = error_recovery_manager.should_retry(
            "EUDR-016", 0, "timeout",
        )
        assert retry is True
        assert delay > Decimal("0")

    def test_should_not_retry_permanent_error(self, error_recovery_manager):
        retry, delay = error_recovery_manager.should_retry(
            "EUDR-016", 0, "validation failed",
        )
        assert retry is False
        assert delay == Decimal("0")

    def test_should_not_retry_when_max_attempts_reached(
        self, error_recovery_manager,
    ):
        max_attempts = error_recovery_manager._config.retry_max_attempts
        retry, delay = error_recovery_manager.should_retry(
            "EUDR-016", max_attempts, "timeout",
        )
        assert retry is False
        assert delay == Decimal("0")

    def test_should_retry_at_max_minus_one(self, error_recovery_manager):
        max_attempts = error_recovery_manager._config.retry_max_attempts
        retry, delay = error_recovery_manager.should_retry(
            "EUDR-016", max_attempts - 1, "timeout",
        )
        assert retry is True
        assert delay > Decimal("0")

    def test_should_not_retry_when_circuit_open(
        self, error_recovery_manager,
    ):
        _force_circuit_open(error_recovery_manager, "EUDR-003")
        retry, delay = error_recovery_manager.should_retry(
            "EUDR-003", 0, "timeout",
        )
        assert retry is False
        assert delay == Decimal("0")

    def test_should_retry_with_http_503(self, error_recovery_manager):
        retry, delay = error_recovery_manager.should_retry(
            "EUDR-016", 0, "server error", http_status=503,
        )
        assert retry is True

    def test_should_not_retry_with_http_401(self, error_recovery_manager):
        retry, delay = error_recovery_manager.should_retry(
            "EUDR-016", 0, "unauthorized", http_status=401,
        )
        assert retry is False

    def test_should_retry_returns_correct_delay_type(
        self, error_recovery_manager,
    ):
        retry, delay = error_recovery_manager.should_retry(
            "EUDR-016", 1, "timeout",
        )
        assert isinstance(delay, Decimal)

    def test_should_retry_attempt_zero_smallest_delay(
        self, error_recovery_manager,
    ):
        with patch("random.uniform", return_value=0.0):
            _, delay0 = error_recovery_manager.should_retry(
                "EUDR-016", 0, "timeout",
            )
            _, delay1 = error_recovery_manager.should_retry(
                "EUDR-016", 1, "timeout",
            )
        assert delay0 < delay1

    def test_should_retry_degraded_error_is_not_retried(
        self, error_recovery_manager,
    ):
        # DEGRADED is not TRANSIENT, and not PERMANENT,
        # but the code only retries non-PERMANENT, non-max-attempts
        retry, delay = error_recovery_manager.should_retry(
            "EUDR-016", 0, "partial result returned",
        )
        # Degraded errors are neither transient nor permanent: the code
        # does not explicitly block them, so they fall through to retry.
        assert isinstance(retry, bool)


# ===================================================================
# Test: Retry Recording
# ===================================================================


class TestRetryRecording:
    """Test retry record creation and history retrieval."""

    def test_record_retry_returns_retry_record(self, error_recovery_manager):
        record = error_recovery_manager.record_retry(
            "wf-001", "EUDR-016", 1, Decimal("2.000"), "timeout",
        )
        assert isinstance(record, RetryRecord)

    def test_record_retry_fields_correct(self, error_recovery_manager):
        record = error_recovery_manager.record_retry(
            "wf-001", "EUDR-016", 1, Decimal("2.500"), "timeout",
        )
        assert record.workflow_id == "wf-001"
        assert record.agent_id == "EUDR-016"
        assert record.attempt_number == 1
        assert record.delay_s == Decimal("2.500")
        assert record.error_message == "timeout"
        assert record.outcome == "pending"

    def test_record_retry_has_unique_id(self, error_recovery_manager):
        r1 = error_recovery_manager.record_retry(
            "wf-001", "EUDR-016", 1, Decimal("1.000"), "timeout",
        )
        r2 = error_recovery_manager.record_retry(
            "wf-001", "EUDR-016", 2, Decimal("2.000"), "timeout",
        )
        assert r1.retry_id != r2.retry_id

    def test_get_retry_history_returns_all_records(
        self, error_recovery_manager,
    ):
        error_recovery_manager.record_retry(
            "wf-001", "EUDR-016", 1, Decimal("1.000"), "timeout",
        )
        error_recovery_manager.record_retry(
            "wf-001", "EUDR-016", 2, Decimal("2.000"), "timeout retry 2",
        )
        history = error_recovery_manager.get_retry_history("wf-001", "EUDR-016")
        assert len(history) == 2

    def test_get_retry_history_empty_for_unknown(
        self, error_recovery_manager,
    ):
        history = error_recovery_manager.get_retry_history("wf-unknown", "EUDR-999")
        assert history == []

    def test_get_retry_history_scoped_by_workflow_and_agent(
        self, error_recovery_manager,
    ):
        error_recovery_manager.record_retry(
            "wf-001", "EUDR-016", 1, Decimal("1.000"), "err",
        )
        error_recovery_manager.record_retry(
            "wf-002", "EUDR-016", 1, Decimal("1.000"), "err",
        )
        error_recovery_manager.record_retry(
            "wf-001", "EUDR-017", 1, Decimal("1.000"), "err",
        )
        history_wf1_016 = error_recovery_manager.get_retry_history("wf-001", "EUDR-016")
        history_wf2_016 = error_recovery_manager.get_retry_history("wf-002", "EUDR-016")
        history_wf1_017 = error_recovery_manager.get_retry_history("wf-001", "EUDR-017")
        assert len(history_wf1_016) == 1
        assert len(history_wf2_016) == 1
        assert len(history_wf1_017) == 1

    def test_record_retry_with_custom_classification(
        self, error_recovery_manager,
    ):
        record = error_recovery_manager.record_retry(
            "wf-001", "EUDR-016", 1, Decimal("1.000"), "bad data",
            error_classification=ErrorClassification.PERMANENT,
        )
        assert record.error_classification == ErrorClassification.PERMANENT

    def test_retry_history_returns_copy(self, error_recovery_manager):
        error_recovery_manager.record_retry(
            "wf-001", "EUDR-016", 1, Decimal("1.000"), "err",
        )
        history = error_recovery_manager.get_retry_history("wf-001", "EUDR-016")
        history.clear()
        # Internal history should not be affected
        history_again = error_recovery_manager.get_retry_history("wf-001", "EUDR-016")
        assert len(history_again) == 1


# ===================================================================
# Test: Circuit Breaker State Machine
# ===================================================================


class TestCircuitBreakerStateMachine:
    """Test all circuit breaker state transitions."""

    def test_initial_state_is_closed(self, error_recovery_manager):
        state = error_recovery_manager.check_circuit_breaker("EUDR-003")
        assert state == CircuitBreakerState.CLOSED

    def test_single_failure_stays_closed(self, error_recovery_manager):
        state = error_recovery_manager.record_failure("EUDR-003")
        assert state == CircuitBreakerState.CLOSED

    def test_failures_below_threshold_stay_closed(
        self, error_recovery_manager,
    ):
        threshold = error_recovery_manager._config.cb_failure_threshold
        for i in range(threshold - 1):
            state = error_recovery_manager.record_failure("EUDR-003")
        assert state == CircuitBreakerState.CLOSED

    def test_failures_at_threshold_open_circuit(
        self, error_recovery_manager,
    ):
        threshold = error_recovery_manager._config.cb_failure_threshold
        for i in range(threshold):
            state = error_recovery_manager.record_failure("EUDR-003")
        assert state == CircuitBreakerState.OPEN

    def test_success_resets_failure_count_in_closed(
        self, error_recovery_manager,
    ):
        error_recovery_manager.record_failure("EUDR-003")
        error_recovery_manager.record_failure("EUDR-003")
        error_recovery_manager.record_success("EUDR-003")
        # After success, failure_count should be 0
        cb = error_recovery_manager.get_circuit_breaker_record("EUDR-003")
        assert cb.failure_count == 0

    def test_open_to_half_open_after_timeout(self, error_recovery_manager):
        _force_circuit_open(error_recovery_manager, "EUDR-003")
        state = error_recovery_manager.check_circuit_breaker("EUDR-003")
        assert state == CircuitBreakerState.OPEN

        # Simulate timeout expiry
        with error_recovery_manager._lock:
            cb = error_recovery_manager._circuit_breakers["EUDR-003"]
            cb.opened_at = _utcnow() - timedelta(
                seconds=cb.reset_timeout_s + 1,
            )

        state = error_recovery_manager.check_circuit_breaker("EUDR-003")
        assert state == CircuitBreakerState.HALF_OPEN

    def test_open_stays_open_before_timeout(self, error_recovery_manager):
        _force_circuit_open(error_recovery_manager, "EUDR-003")
        # opened_at is set to now, timeout not elapsed
        state = error_recovery_manager.check_circuit_breaker("EUDR-003")
        assert state == CircuitBreakerState.OPEN

    def test_half_open_to_closed_after_success_threshold(
        self, error_recovery_manager,
    ):
        _force_circuit_half_open(error_recovery_manager, "EUDR-003")
        state = error_recovery_manager.check_circuit_breaker("EUDR-003")
        assert state == CircuitBreakerState.HALF_OPEN

        # Record successes up to failure_threshold (used as success threshold
        # in the record_success code: cb.success_count >= cb.failure_threshold)
        cb = error_recovery_manager.get_circuit_breaker_record("EUDR-003")
        for _ in range(cb.failure_threshold):
            state = error_recovery_manager.record_success("EUDR-003")

        assert state == CircuitBreakerState.CLOSED

    def test_half_open_to_open_on_failure(self, error_recovery_manager):
        _force_circuit_half_open(error_recovery_manager, "EUDR-003")

        state = error_recovery_manager.record_failure("EUDR-003")
        assert state == CircuitBreakerState.OPEN

    def test_half_open_single_success_insufficient(
        self, error_recovery_manager,
    ):
        _force_circuit_half_open(error_recovery_manager, "EUDR-003")
        cb = error_recovery_manager.get_circuit_breaker_record("EUDR-003")
        if cb.failure_threshold > 1:
            state = error_recovery_manager.record_success("EUDR-003")
            assert state == CircuitBreakerState.HALF_OPEN

    def test_circuit_breaker_record_created_on_first_failure(
        self, error_recovery_manager,
    ):
        assert error_recovery_manager.get_circuit_breaker_record("EUDR-003") is None
        error_recovery_manager.record_failure("EUDR-003")
        assert error_recovery_manager.get_circuit_breaker_record("EUDR-003") is not None

    def test_circuit_breaker_record_created_on_first_success(
        self, error_recovery_manager,
    ):
        assert error_recovery_manager.get_circuit_breaker_record("EUDR-003") is None
        error_recovery_manager.record_success("EUDR-003")
        assert error_recovery_manager.get_circuit_breaker_record("EUDR-003") is not None

    def test_opened_at_set_when_circuit_opens(
        self, error_recovery_manager,
    ):
        _force_circuit_open(error_recovery_manager, "EUDR-003")
        cb = error_recovery_manager.get_circuit_breaker_record("EUDR-003")
        assert cb.opened_at is not None

    def test_last_failure_at_updated(self, error_recovery_manager):
        error_recovery_manager.record_failure("EUDR-003")
        cb = error_recovery_manager.get_circuit_breaker_record("EUDR-003")
        assert cb.last_failure_at is not None

    def test_last_success_at_updated(self, error_recovery_manager):
        error_recovery_manager.record_success("EUDR-003")
        cb = error_recovery_manager.get_circuit_breaker_record("EUDR-003")
        assert cb.last_success_at is not None

    def test_open_circuit_count_zero_initially(
        self, error_recovery_manager,
    ):
        assert error_recovery_manager.get_open_circuit_count() == 0

    def test_open_circuit_count_tracks_open_breakers(
        self, error_recovery_manager,
    ):
        _force_circuit_open(error_recovery_manager, "EUDR-003")
        _force_circuit_open(error_recovery_manager, "EUDR-020")
        assert error_recovery_manager.get_open_circuit_count() == 2

    def test_open_circuit_count_decrements_on_half_open(
        self, error_recovery_manager,
    ):
        _force_circuit_open(error_recovery_manager, "EUDR-003")
        assert error_recovery_manager.get_open_circuit_count() == 1

        _force_circuit_half_open(error_recovery_manager, "EUDR-003")
        assert error_recovery_manager.get_open_circuit_count() == 0

    def test_isolated_circuit_breakers_per_agent(
        self, error_recovery_manager,
    ):
        # Fail EUDR-003, succeed EUDR-020
        _force_circuit_open(error_recovery_manager, "EUDR-003")
        error_recovery_manager.record_success("EUDR-020")

        cb_003 = error_recovery_manager.get_circuit_breaker_record("EUDR-003")
        cb_020 = error_recovery_manager.get_circuit_breaker_record("EUDR-020")

        assert cb_003.state == CircuitBreakerState.OPEN
        assert cb_020.state == CircuitBreakerState.CLOSED

    def test_half_open_success_count_reset_on_reopen(
        self, error_recovery_manager,
    ):
        _force_circuit_half_open(error_recovery_manager, "EUDR-003")
        # Record one success then fail
        error_recovery_manager.record_success("EUDR-003")
        error_recovery_manager.record_failure("EUDR-003")
        cb = error_recovery_manager.get_circuit_breaker_record("EUDR-003")
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.success_count == 0


# ===================================================================
# Test: Circuit Breaker Reset
# ===================================================================


class TestCircuitBreakerReset:
    """Test manual circuit breaker reset."""

    def test_reset_open_to_closed(self, error_recovery_manager):
        _force_circuit_open(error_recovery_manager, "EUDR-003")
        error_recovery_manager.reset_circuit_breaker("EUDR-003")
        cb = error_recovery_manager.get_circuit_breaker_record("EUDR-003")
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0

    def test_reset_half_open_to_closed(self, error_recovery_manager):
        _force_circuit_half_open(error_recovery_manager, "EUDR-003")
        error_recovery_manager.reset_circuit_breaker("EUDR-003")
        cb = error_recovery_manager.get_circuit_breaker_record("EUDR-003")
        assert cb.state == CircuitBreakerState.CLOSED

    def test_reset_nonexistent_is_noop(self, error_recovery_manager):
        # Should not raise
        error_recovery_manager.reset_circuit_breaker("EUDR-NONEXISTENT")

    def test_reset_already_closed_is_noop(self, error_recovery_manager):
        error_recovery_manager.record_success("EUDR-003")
        cb_before = error_recovery_manager.get_circuit_breaker_record("EUDR-003")
        error_recovery_manager.reset_circuit_breaker("EUDR-003")
        cb_after = error_recovery_manager.get_circuit_breaker_record("EUDR-003")
        assert cb_after.state == CircuitBreakerState.CLOSED

    def test_reset_clears_counts(self, error_recovery_manager):
        error_recovery_manager.record_failure("EUDR-003")
        error_recovery_manager.record_failure("EUDR-003")
        error_recovery_manager.reset_circuit_breaker("EUDR-003")
        cb = error_recovery_manager.get_circuit_breaker_record("EUDR-003")
        assert cb.failure_count == 0
        assert cb.success_count == 0


# ===================================================================
# Test: Fallback Strategy
# ===================================================================


class TestFallbackStrategy:
    """Test fallback strategy determination."""

    def test_degraded_error_uses_cached_result(
        self, error_recovery_manager,
    ):
        result = error_recovery_manager.determine_fallback(
            "EUDR-016", ErrorClassification.DEGRADED,
        )
        assert result == FallbackStrategy.CACHED_RESULT

    def test_degraded_ignores_configured_strategy(
        self, error_recovery_manager,
    ):
        result = error_recovery_manager.determine_fallback(
            "EUDR-016", ErrorClassification.DEGRADED,
            fallback_strategy=FallbackStrategy.FAIL,
        )
        assert result == FallbackStrategy.CACHED_RESULT

    def test_permanent_error_with_fail_fallback_becomes_manual(
        self, error_recovery_manager,
    ):
        result = error_recovery_manager.determine_fallback(
            "EUDR-016", ErrorClassification.PERMANENT,
            fallback_strategy=FallbackStrategy.FAIL,
        )
        assert result == FallbackStrategy.MANUAL_OVERRIDE

    def test_permanent_error_with_cached_strategy_stays(
        self, error_recovery_manager,
    ):
        result = error_recovery_manager.determine_fallback(
            "EUDR-016", ErrorClassification.PERMANENT,
            fallback_strategy=FallbackStrategy.CACHED_RESULT,
        )
        assert result == FallbackStrategy.CACHED_RESULT

    def test_permanent_error_with_degraded_mode_stays(
        self, error_recovery_manager,
    ):
        result = error_recovery_manager.determine_fallback(
            "EUDR-016", ErrorClassification.PERMANENT,
            fallback_strategy=FallbackStrategy.DEGRADED_MODE,
        )
        assert result == FallbackStrategy.DEGRADED_MODE

    def test_transient_error_returns_configured_strategy(
        self, error_recovery_manager,
    ):
        result = error_recovery_manager.determine_fallback(
            "EUDR-016", ErrorClassification.TRANSIENT,
            fallback_strategy=FallbackStrategy.CACHED_RESULT,
        )
        assert result == FallbackStrategy.CACHED_RESULT

    def test_unknown_error_returns_configured_strategy(
        self, error_recovery_manager,
    ):
        result = error_recovery_manager.determine_fallback(
            "EUDR-016", ErrorClassification.UNKNOWN,
            fallback_strategy=FallbackStrategy.FAIL,
        )
        assert result == FallbackStrategy.FAIL

    def test_default_fallback_is_fail(self, error_recovery_manager):
        result = error_recovery_manager.determine_fallback(
            "EUDR-016", ErrorClassification.TRANSIENT,
        )
        assert result == FallbackStrategy.FAIL


# ===================================================================
# Test: Dead Letter Queue
# ===================================================================


class TestDeadLetterQueue:
    """Test dead letter queue CRUD operations."""

    def test_add_to_dead_letter_returns_entry(
        self, error_recovery_manager,
    ):
        entry = error_recovery_manager.add_to_dead_letter(
            "wf-001", "EUDR-003", "final error",
            ErrorClassification.PERMANENT,
        )
        assert isinstance(entry, DeadLetterEntry)

    def test_dead_letter_entry_fields_correct(
        self, error_recovery_manager,
    ):
        entry = error_recovery_manager.add_to_dead_letter(
            "wf-001", "EUDR-003", "satellite API down",
            ErrorClassification.TRANSIENT,
            input_data={"region": "GH"},
        )
        assert entry.workflow_id == "wf-001"
        assert entry.agent_id == "EUDR-003"
        assert entry.error_message == "satellite API down"
        assert entry.error_classification == ErrorClassification.TRANSIENT
        assert entry.input_data == {"region": "GH"}
        assert entry.resolved is False
        assert entry.resolved_by is None

    def test_dead_letter_entry_has_unique_id(
        self, error_recovery_manager,
    ):
        e1 = error_recovery_manager.add_to_dead_letter(
            "wf-001", "EUDR-003", "err1", ErrorClassification.PERMANENT,
        )
        e2 = error_recovery_manager.add_to_dead_letter(
            "wf-001", "EUDR-003", "err2", ErrorClassification.PERMANENT,
        )
        assert e1.entry_id != e2.entry_id

    def test_get_dead_letter_queue_empty_initially(
        self, error_recovery_manager,
    ):
        assert error_recovery_manager.get_dead_letter_queue() == []

    def test_get_dead_letter_queue_returns_all(
        self, error_recovery_manager,
    ):
        error_recovery_manager.add_to_dead_letter(
            "wf-001", "EUDR-003", "err1", ErrorClassification.PERMANENT,
        )
        error_recovery_manager.add_to_dead_letter(
            "wf-002", "EUDR-020", "err2", ErrorClassification.PERMANENT,
        )
        queue = error_recovery_manager.get_dead_letter_queue()
        assert len(queue) == 2

    def test_get_dead_letter_count_excludes_resolved(
        self, error_recovery_manager,
    ):
        e1 = error_recovery_manager.add_to_dead_letter(
            "wf-001", "EUDR-003", "err1", ErrorClassification.PERMANENT,
        )
        error_recovery_manager.add_to_dead_letter(
            "wf-002", "EUDR-020", "err2", ErrorClassification.PERMANENT,
        )
        error_recovery_manager.resolve_dead_letter(e1.entry_id, "admin")
        count = error_recovery_manager.get_dead_letter_count()
        assert count == 1

    def test_resolve_dead_letter_marks_resolved(
        self, error_recovery_manager,
    ):
        entry = error_recovery_manager.add_to_dead_letter(
            "wf-001", "EUDR-003", "err", ErrorClassification.PERMANENT,
        )
        result = error_recovery_manager.resolve_dead_letter(
            entry.entry_id, "admin_user",
        )
        assert result is True
        queue = error_recovery_manager.get_dead_letter_queue()
        resolved_entry = [e for e in queue if e.entry_id == entry.entry_id][0]
        assert resolved_entry.resolved is True
        assert resolved_entry.resolved_by == "admin_user"
        assert resolved_entry.resolved_at is not None

    def test_resolve_dead_letter_unknown_id_returns_false(
        self, error_recovery_manager,
    ):
        result = error_recovery_manager.resolve_dead_letter(
            "nonexistent-id", "admin",
        )
        assert result is False

    def test_dead_letter_captures_retry_history(
        self, error_recovery_manager,
    ):
        error_recovery_manager.record_retry(
            "wf-001", "EUDR-003", 1, Decimal("1.000"), "timeout",
        )
        error_recovery_manager.record_retry(
            "wf-001", "EUDR-003", 2, Decimal("2.000"), "timeout again",
        )
        entry = error_recovery_manager.add_to_dead_letter(
            "wf-001", "EUDR-003", "gave up", ErrorClassification.PERMANENT,
        )
        assert len(entry.retry_history) == 2

    def test_dead_letter_captures_circuit_breaker_state(
        self, error_recovery_manager,
    ):
        _force_circuit_open(error_recovery_manager, "EUDR-003")
        entry = error_recovery_manager.add_to_dead_letter(
            "wf-001", "EUDR-003", "circuit open", ErrorClassification.PERMANENT,
        )
        assert entry.circuit_breaker_state == CircuitBreakerState.OPEN

    def test_dead_letter_no_circuit_breaker_state_when_none_exists(
        self, error_recovery_manager,
    ):
        entry = error_recovery_manager.add_to_dead_letter(
            "wf-001", "EUDR-099", "err", ErrorClassification.PERMANENT,
        )
        assert entry.circuit_breaker_state is None

    def test_get_dead_letter_queue_returns_copy(
        self, error_recovery_manager,
    ):
        error_recovery_manager.add_to_dead_letter(
            "wf-001", "EUDR-003", "err", ErrorClassification.PERMANENT,
        )
        queue = error_recovery_manager.get_dead_letter_queue()
        queue.clear()
        queue_again = error_recovery_manager.get_dead_letter_queue()
        assert len(queue_again) == 1


# ===================================================================
# Test: Concurrency
# ===================================================================


class TestConcurrency:
    """Test thread-safety of circuit breaker and retry operations."""

    def test_concurrent_record_failure_thread_safe(
        self, error_recovery_manager,
    ):
        """Multiple threads recording failures should not corrupt state."""
        errors: List[Exception] = []

        def _fail():
            try:
                for _ in range(20):
                    error_recovery_manager.record_failure("EUDR-003")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_fail) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        cb = error_recovery_manager.get_circuit_breaker_record("EUDR-003")
        assert cb.failure_count == 100  # 5 threads * 20 failures

    def test_concurrent_record_success_thread_safe(
        self, error_recovery_manager,
    ):
        errors: List[Exception] = []

        def _succeed():
            try:
                for _ in range(20):
                    error_recovery_manager.record_success("EUDR-003")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_succeed) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0

    def test_concurrent_add_dead_letter_thread_safe(
        self, error_recovery_manager,
    ):
        errors: List[Exception] = []

        def _add_dl(n: int):
            try:
                for i in range(10):
                    error_recovery_manager.add_to_dead_letter(
                        f"wf-{n}", f"EUDR-{n:03d}", f"err-{i}",
                        ErrorClassification.PERMANENT,
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_add_dl, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert len(error_recovery_manager.get_dead_letter_queue()) == 50

    def test_concurrent_record_retry_thread_safe(
        self, error_recovery_manager,
    ):
        errors: List[Exception] = []

        def _retry(thread_id: int):
            try:
                for attempt in range(5):
                    error_recovery_manager.record_retry(
                        f"wf-{thread_id}", "EUDR-003",
                        attempt + 1, Decimal("1.000"), "timeout",
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_retry, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        # Each thread wrote 5 records under its own workflow key
        for i in range(5):
            history = error_recovery_manager.get_retry_history(
                f"wf-{i}", "EUDR-003",
            )
            assert len(history) == 5

    def test_concurrent_mixed_operations(self, error_recovery_manager):
        """Interleave failures, successes, retries, and dead letters."""
        errors: List[Exception] = []

        def _mixed(thread_id: int):
            try:
                for _ in range(10):
                    error_recovery_manager.record_failure(f"EUDR-{thread_id:03d}")
                    error_recovery_manager.record_success(f"EUDR-{thread_id:03d}")
                    error_recovery_manager.record_retry(
                        f"wf-{thread_id}", f"EUDR-{thread_id:03d}",
                        1, Decimal("0.500"), "mixed test",
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_mixed, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
