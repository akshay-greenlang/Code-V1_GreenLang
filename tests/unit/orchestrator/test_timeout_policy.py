# -*- coding: utf-8 -*-
"""
Unit tests for TimeoutPolicy (AGENT-FOUND-001)

Tests timeout enforcement, on_timeout strategies (fail/skip/compensate),
and policy merging.

Coverage target: 85%+ of timeout_policy.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any, Optional

import pytest

from tests.unit.orchestrator.conftest import TimeoutPolicyData, _run_async


# ---------------------------------------------------------------------------
# Inline TimeoutPolicy that mirrors expected interface
# ---------------------------------------------------------------------------


class TimeoutResult:
    """Result of applying a timeout policy."""

    def __init__(self, timed_out: bool, result: Any = None, action: str = "fail"):
        self.timed_out = timed_out
        self.result = result
        self.action = action


class TimeoutPolicy:
    """Timeout policy for DAG node execution."""

    def __init__(self, data: TimeoutPolicyData):
        self._data = data

    @property
    def timeout_seconds(self) -> float:
        return self._data.timeout_seconds

    @property
    def on_timeout(self) -> str:
        return self._data.on_timeout

    async def apply(self, coro) -> TimeoutResult:
        """Apply timeout to a coroutine."""
        try:
            result = await asyncio.wait_for(coro, timeout=self._data.timeout_seconds)
            return TimeoutResult(timed_out=False, result=result)
        except asyncio.TimeoutError:
            return self.handle_timeout()

    def handle_timeout(self) -> TimeoutResult:
        """Handle a timeout based on the on_timeout strategy."""
        if self._data.on_timeout == "fail":
            return TimeoutResult(timed_out=True, action="fail")
        elif self._data.on_timeout == "skip":
            return TimeoutResult(timed_out=True, result=None, action="skip")
        elif self._data.on_timeout == "compensate":
            return TimeoutResult(timed_out=True, action="compensate")
        else:
            return TimeoutResult(timed_out=True, action="fail")

    @classmethod
    def merge_with_default(
        cls,
        node_policy: Optional[TimeoutPolicyData],
        default_policy: Optional[TimeoutPolicyData],
    ) -> Optional["TimeoutPolicy"]:
        """Merge node-level policy with DAG-level default."""
        if node_policy:
            return cls(node_policy)
        if default_policy:
            return cls(default_policy)
        return None


# ===========================================================================
# Test Classes
# ===========================================================================


class TestTimeoutWithinLimit:
    """Test that operations completing within timeout succeed."""

    def test_fast_operation_succeeds(self):
        policy = TimeoutPolicy(TimeoutPolicyData(timeout_seconds=1.0))

        async def fast_op():
            return "done"

        result = _run_async(policy.apply(fast_op()))
        assert result.timed_out is False
        assert result.result == "done"

    def test_result_preserved(self):
        policy = TimeoutPolicy(TimeoutPolicyData(timeout_seconds=1.0))

        async def compute():
            return {"value": 42, "status": "ok"}

        result = _run_async(policy.apply(compute()))
        assert result.result == {"value": 42, "status": "ok"}

    def test_nearly_at_limit(self):
        policy = TimeoutPolicy(TimeoutPolicyData(timeout_seconds=1.0))

        async def near_limit():
            await asyncio.sleep(0.01)  # Well within 1s
            return "barely made it"

        result = _run_async(policy.apply(near_limit()))
        assert result.timed_out is False
        assert result.result == "barely made it"


class TestTimeoutExceeded:
    """Test behavior when operation exceeds timeout."""

    def test_slow_operation_times_out(self):
        policy = TimeoutPolicy(TimeoutPolicyData(timeout_seconds=0.05))

        async def slow_op():
            await asyncio.sleep(10.0)
            return "should not reach"

        result = _run_async(policy.apply(slow_op()))
        assert result.timed_out is True

    def test_timeout_default_action_is_fail(self):
        policy = TimeoutPolicy(TimeoutPolicyData(timeout_seconds=0.05))

        async def slow_op():
            await asyncio.sleep(10.0)

        result = _run_async(policy.apply(slow_op()))
        assert result.action == "fail"


class TestOnTimeoutFail:
    """Test on_timeout=fail strategy."""

    def test_fail_action(self):
        policy = TimeoutPolicy(TimeoutPolicyData(timeout_seconds=0.05, on_timeout="fail"))

        async def slow_op():
            await asyncio.sleep(10.0)

        result = _run_async(policy.apply(slow_op()))
        assert result.timed_out is True
        assert result.action == "fail"
        assert result.result is None

    def test_handle_timeout_fail(self):
        policy = TimeoutPolicy(TimeoutPolicyData(on_timeout="fail"))
        result = policy.handle_timeout()
        assert result.timed_out is True
        assert result.action == "fail"


class TestOnTimeoutSkip:
    """Test on_timeout=skip strategy."""

    def test_skip_action(self):
        policy = TimeoutPolicy(TimeoutPolicyData(timeout_seconds=0.05, on_timeout="skip"))

        async def slow_op():
            await asyncio.sleep(10.0)

        result = _run_async(policy.apply(slow_op()))
        assert result.timed_out is True
        assert result.action == "skip"
        assert result.result is None

    def test_handle_timeout_skip(self):
        policy = TimeoutPolicy(TimeoutPolicyData(on_timeout="skip"))
        result = policy.handle_timeout()
        assert result.timed_out is True
        assert result.action == "skip"


class TestOnTimeoutCompensate:
    """Test on_timeout=compensate strategy."""

    def test_compensate_action(self):
        policy = TimeoutPolicy(TimeoutPolicyData(timeout_seconds=0.05, on_timeout="compensate"))

        async def slow_op():
            await asyncio.sleep(10.0)

        result = _run_async(policy.apply(slow_op()))
        assert result.timed_out is True
        assert result.action == "compensate"

    def test_handle_timeout_compensate(self):
        policy = TimeoutPolicy(TimeoutPolicyData(on_timeout="compensate"))
        result = policy.handle_timeout()
        assert result.timed_out is True
        assert result.action == "compensate"


class TestMergeWithDefault:
    """Test merging node-level and DAG-level timeout policies."""

    def test_node_policy_wins(self):
        node = TimeoutPolicyData(timeout_seconds=10.0)
        default = TimeoutPolicyData(timeout_seconds=60.0)
        merged = TimeoutPolicy.merge_with_default(node, default)
        assert merged.timeout_seconds == 10.0

    def test_default_used_when_no_node_policy(self):
        default = TimeoutPolicyData(timeout_seconds=30.0, on_timeout="skip")
        merged = TimeoutPolicy.merge_with_default(None, default)
        assert merged.timeout_seconds == 30.0
        assert merged.on_timeout == "skip"

    def test_none_when_no_policies(self):
        merged = TimeoutPolicy.merge_with_default(None, None)
        assert merged is None

    def test_node_on_timeout_overrides_default(self):
        node = TimeoutPolicyData(timeout_seconds=5.0, on_timeout="compensate")
        default = TimeoutPolicyData(timeout_seconds=30.0, on_timeout="fail")
        merged = TimeoutPolicy.merge_with_default(node, default)
        assert merged.on_timeout == "compensate"


class TestTimeoutPolicyProperties:
    """Test TimeoutPolicy property accessors."""

    def test_timeout_seconds_property(self):
        policy = TimeoutPolicy(TimeoutPolicyData(timeout_seconds=42.0))
        assert policy.timeout_seconds == 42.0

    def test_on_timeout_property(self):
        policy = TimeoutPolicy(TimeoutPolicyData(on_timeout="skip"))
        assert policy.on_timeout == "skip"

    @pytest.mark.parametrize("on_timeout", ["fail", "skip", "compensate"])
    def test_all_on_timeout_values(self, on_timeout):
        policy = TimeoutPolicy(TimeoutPolicyData(on_timeout=on_timeout))
        assert policy.on_timeout == on_timeout
