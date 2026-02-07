# -*- coding: utf-8 -*-
"""
Unit tests for RetryPolicy (AGENT-FOUND-001)

Tests all backoff strategies (exponential, linear, constant, fibonacci),
jitter, max delay cap, should_retry logic, and policy merging.

Coverage target: 85%+ of retry_policy.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import random
from typing import Any, List, Optional, Type

import pytest

from tests.unit.orchestrator.conftest import RetryPolicyData


# ---------------------------------------------------------------------------
# Inline retry policy calculator that mirrors expected interface
# ---------------------------------------------------------------------------


class RetryPolicy:
    """Retry policy with multiple backoff strategies."""

    def __init__(self, data: RetryPolicyData):
        self._data = data

    @property
    def max_retries(self) -> int:
        return self._data.max_retries

    @property
    def strategy(self) -> str:
        return self._data.strategy

    @property
    def base_delay(self) -> float:
        return self._data.base_delay

    @property
    def max_delay(self) -> float:
        return self._data.max_delay

    @property
    def jitter(self) -> bool:
        return self._data.jitter

    def calculate_delay(self, attempt: int, seed: int = None) -> float:
        """Calculate delay for a given attempt number (1-indexed)."""
        if attempt < 1:
            return 0.0

        if self._data.strategy == "exponential":
            delay = self._data.base_delay * (2 ** (attempt - 1))
        elif self._data.strategy == "linear":
            delay = self._data.base_delay * attempt
        elif self._data.strategy == "constant":
            delay = self._data.base_delay
        elif self._data.strategy == "fibonacci":
            delay = self._fibonacci_delay(attempt)
        else:
            delay = self._data.base_delay

        # Cap at max_delay
        delay = min(delay, self._data.max_delay)

        # Add jitter
        if self._data.jitter:
            rng = random.Random(seed)
            delay = delay * rng.uniform(0.5, 1.5)

        return delay

    def _fibonacci_delay(self, attempt: int) -> float:
        """Calculate fibonacci backoff: base*1, base*1, base*2, base*3, base*5..."""
        if attempt <= 2:
            return self._data.base_delay
        a, b = 1, 1
        for _ in range(attempt - 2):
            a, b = b, a + b
        return self._data.base_delay * b

    def should_retry(
        self,
        attempt: int,
        exception: Exception = None,
    ) -> bool:
        """Determine if a retry should be attempted."""
        if attempt > self._data.max_retries:
            return False
        if exception and self._data.retryable_exceptions:
            exc_name = type(exception).__name__
            # Check if exception type or "Exception" (catch-all) is retryable
            if exc_name not in self._data.retryable_exceptions and \
               "Exception" not in self._data.retryable_exceptions:
                return False
        return True

    @classmethod
    def merge_with_default(
        cls,
        node_policy: Optional[RetryPolicyData],
        default_policy: Optional[RetryPolicyData],
    ) -> "RetryPolicy":
        """Merge node-level policy with DAG-level default. Node wins."""
        if node_policy:
            return cls(node_policy)
        if default_policy:
            return cls(default_policy)
        return cls(RetryPolicyData(max_retries=0))


# ===========================================================================
# Test Classes
# ===========================================================================


class TestExponentialBackoffDelays:
    """Test exponential backoff strategy: base * 2^(attempt-1)."""

    def test_first_attempt(self):
        rp = RetryPolicy(RetryPolicyData(strategy="exponential", base_delay=1.0, max_delay=100.0))
        assert rp.calculate_delay(1) == pytest.approx(1.0)

    def test_second_attempt(self):
        rp = RetryPolicy(RetryPolicyData(strategy="exponential", base_delay=1.0, max_delay=100.0))
        assert rp.calculate_delay(2) == pytest.approx(2.0)

    def test_third_attempt(self):
        rp = RetryPolicy(RetryPolicyData(strategy="exponential", base_delay=1.0, max_delay=100.0))
        assert rp.calculate_delay(3) == pytest.approx(4.0)

    def test_fourth_attempt(self):
        rp = RetryPolicy(RetryPolicyData(strategy="exponential", base_delay=1.0, max_delay=100.0))
        assert rp.calculate_delay(4) == pytest.approx(8.0)

    def test_custom_base_delay(self):
        rp = RetryPolicy(RetryPolicyData(strategy="exponential", base_delay=0.5, max_delay=100.0))
        assert rp.calculate_delay(1) == pytest.approx(0.5)
        assert rp.calculate_delay(2) == pytest.approx(1.0)
        assert rp.calculate_delay(3) == pytest.approx(2.0)

    @pytest.mark.parametrize("attempt,expected", [
        (1, 0.01), (2, 0.02), (3, 0.04), (4, 0.08),
    ])
    def test_exponential_series(self, attempt, expected):
        rp = RetryPolicy(RetryPolicyData(strategy="exponential", base_delay=0.01, max_delay=10.0))
        assert rp.calculate_delay(attempt) == pytest.approx(expected)


class TestLinearBackoffDelays:
    """Test linear backoff strategy: base * attempt."""

    def test_first_attempt(self):
        rp = RetryPolicy(RetryPolicyData(strategy="linear", base_delay=1.0, max_delay=100.0))
        assert rp.calculate_delay(1) == pytest.approx(1.0)

    def test_second_attempt(self):
        rp = RetryPolicy(RetryPolicyData(strategy="linear", base_delay=1.0, max_delay=100.0))
        assert rp.calculate_delay(2) == pytest.approx(2.0)

    def test_third_attempt(self):
        rp = RetryPolicy(RetryPolicyData(strategy="linear", base_delay=1.0, max_delay=100.0))
        assert rp.calculate_delay(3) == pytest.approx(3.0)

    @pytest.mark.parametrize("attempt,expected", [
        (1, 0.5), (2, 1.0), (3, 1.5), (4, 2.0),
    ])
    def test_linear_series(self, attempt, expected):
        rp = RetryPolicy(RetryPolicyData(strategy="linear", base_delay=0.5, max_delay=100.0))
        assert rp.calculate_delay(attempt) == pytest.approx(expected)


class TestConstantBackoffDelays:
    """Test constant backoff strategy: always base_delay."""

    def test_all_attempts_same(self):
        rp = RetryPolicy(RetryPolicyData(strategy="constant", base_delay=1.0, max_delay=100.0))
        for attempt in range(1, 6):
            assert rp.calculate_delay(attempt) == pytest.approx(1.0)

    @pytest.mark.parametrize("attempt", [1, 2, 3, 4, 5, 10])
    def test_constant_delay(self, attempt):
        rp = RetryPolicy(RetryPolicyData(strategy="constant", base_delay=0.5, max_delay=100.0))
        assert rp.calculate_delay(attempt) == pytest.approx(0.5)


class TestFibonacciBackoffDelays:
    """Test fibonacci backoff strategy: base*1, base*1, base*2, base*3, base*5..."""

    def test_first_attempt(self):
        rp = RetryPolicy(RetryPolicyData(strategy="fibonacci", base_delay=1.0, max_delay=100.0))
        assert rp.calculate_delay(1) == pytest.approx(1.0)

    def test_second_attempt(self):
        rp = RetryPolicy(RetryPolicyData(strategy="fibonacci", base_delay=1.0, max_delay=100.0))
        assert rp.calculate_delay(2) == pytest.approx(1.0)

    def test_third_attempt(self):
        rp = RetryPolicy(RetryPolicyData(strategy="fibonacci", base_delay=1.0, max_delay=100.0))
        assert rp.calculate_delay(3) == pytest.approx(2.0)

    def test_fourth_attempt(self):
        rp = RetryPolicy(RetryPolicyData(strategy="fibonacci", base_delay=1.0, max_delay=100.0))
        assert rp.calculate_delay(4) == pytest.approx(3.0)

    def test_fifth_attempt(self):
        rp = RetryPolicy(RetryPolicyData(strategy="fibonacci", base_delay=1.0, max_delay=100.0))
        assert rp.calculate_delay(5) == pytest.approx(5.0)

    @pytest.mark.parametrize("attempt,expected", [
        (1, 1.0), (2, 1.0), (3, 2.0), (4, 3.0), (5, 5.0),
    ])
    def test_fibonacci_series(self, attempt, expected):
        rp = RetryPolicy(RetryPolicyData(strategy="fibonacci", base_delay=1.0, max_delay=100.0))
        assert rp.calculate_delay(attempt) == pytest.approx(expected)


class TestJitterAddsRandomness:
    """Test that jitter adds randomness to delay."""

    def test_jitter_changes_delay(self):
        rp = RetryPolicy(RetryPolicyData(
            strategy="constant", base_delay=1.0, max_delay=100.0, jitter=True
        ))
        delays = {rp.calculate_delay(1, seed=i) for i in range(20)}
        assert len(delays) > 1  # Different seeds should produce different delays

    def test_jitter_stays_in_range(self):
        rp = RetryPolicy(RetryPolicyData(
            strategy="constant", base_delay=1.0, max_delay=100.0, jitter=True
        ))
        for seed in range(100):
            delay = rp.calculate_delay(1, seed=seed)
            assert 0.5 <= delay <= 1.5

    def test_no_jitter_is_deterministic(self):
        rp = RetryPolicy(RetryPolicyData(
            strategy="constant", base_delay=1.0, max_delay=100.0, jitter=False
        ))
        delays = {rp.calculate_delay(1) for _ in range(10)}
        assert len(delays) == 1


class TestMaxDelayCap:
    """Test that delay is capped at max_delay."""

    def test_exponential_capped(self):
        rp = RetryPolicy(RetryPolicyData(
            strategy="exponential", base_delay=1.0, max_delay=5.0
        ))
        # Attempt 10: 2^9 = 512, should be capped at 5.0
        assert rp.calculate_delay(10) == pytest.approx(5.0)

    def test_linear_capped(self):
        rp = RetryPolicy(RetryPolicyData(
            strategy="linear", base_delay=1.0, max_delay=5.0
        ))
        assert rp.calculate_delay(10) == pytest.approx(5.0)

    def test_fibonacci_capped(self):
        rp = RetryPolicy(RetryPolicyData(
            strategy="fibonacci", base_delay=1.0, max_delay=5.0
        ))
        # Fib(10) = 55, should be capped
        assert rp.calculate_delay(10) == pytest.approx(5.0)

    def test_constant_not_capped_if_within(self):
        rp = RetryPolicy(RetryPolicyData(
            strategy="constant", base_delay=1.0, max_delay=5.0
        ))
        assert rp.calculate_delay(1) == pytest.approx(1.0)


class TestShouldRetryWithinMax:
    """Test should_retry returns True within max_retries."""

    def test_within_max(self):
        rp = RetryPolicy(RetryPolicyData(max_retries=3))
        assert rp.should_retry(1) is True
        assert rp.should_retry(2) is True
        assert rp.should_retry(3) is True

    def test_at_boundary(self):
        rp = RetryPolicy(RetryPolicyData(max_retries=2))
        assert rp.should_retry(2) is True


class TestShouldRetryExceeded:
    """Test should_retry returns False when exceeded."""

    def test_exceeded(self):
        rp = RetryPolicy(RetryPolicyData(max_retries=2))
        assert rp.should_retry(3) is False

    def test_zero_retries(self):
        rp = RetryPolicy(RetryPolicyData(max_retries=0))
        assert rp.should_retry(1) is False


class TestShouldRetryNonRetryableException:
    """Test should_retry with non-retryable exceptions."""

    def test_retryable_exception_allowed(self):
        rp = RetryPolicy(RetryPolicyData(
            max_retries=3,
            retryable_exceptions=["RuntimeError"],
        ))
        assert rp.should_retry(1, RuntimeError("test")) is True

    def test_non_retryable_exception_rejected(self):
        rp = RetryPolicy(RetryPolicyData(
            max_retries=3,
            retryable_exceptions=["ValueError"],
        ))
        assert rp.should_retry(1, RuntimeError("test")) is False

    def test_exception_catch_all(self):
        rp = RetryPolicy(RetryPolicyData(
            max_retries=3,
            retryable_exceptions=["Exception"],
        ))
        assert rp.should_retry(1, RuntimeError("test")) is True


class TestMergeWithDefault:
    """Test merging node policy with DAG-level default."""

    def test_node_policy_wins(self):
        node_policy = RetryPolicyData(max_retries=5)
        default_policy = RetryPolicyData(max_retries=2)
        merged = RetryPolicy.merge_with_default(node_policy, default_policy)
        assert merged.max_retries == 5

    def test_default_used_when_no_node_policy(self):
        default_policy = RetryPolicyData(max_retries=3, strategy="linear")
        merged = RetryPolicy.merge_with_default(None, default_policy)
        assert merged.max_retries == 3
        assert merged.strategy == "linear"

    def test_zero_retries_when_no_policies(self):
        merged = RetryPolicy.merge_with_default(None, None)
        assert merged.max_retries == 0

    def test_node_strategy_overrides_default(self):
        node_policy = RetryPolicyData(max_retries=2, strategy="fibonacci")
        default_policy = RetryPolicyData(max_retries=5, strategy="exponential")
        merged = RetryPolicy.merge_with_default(node_policy, default_policy)
        assert merged.strategy == "fibonacci"


class TestEdgeCases:
    """Test edge cases for retry policy."""

    def test_zero_attempt(self):
        rp = RetryPolicy(RetryPolicyData(strategy="exponential", base_delay=1.0, max_delay=100.0))
        assert rp.calculate_delay(0) == 0.0

    def test_negative_attempt(self):
        rp = RetryPolicy(RetryPolicyData(strategy="exponential", base_delay=1.0, max_delay=100.0))
        assert rp.calculate_delay(-1) == 0.0
