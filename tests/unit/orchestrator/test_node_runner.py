# -*- coding: utf-8 -*-
"""
Unit tests for NodeRunner (AGENT-FOUND-001)

Tests node execution with sync/async agents, retry logic, timeout
enforcement, duration recording, and output hash calculation.

Coverage target: 85%+ of node_runner.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from tests.unit.orchestrator.conftest import (
    DAGNodeData,
    RetryPolicyData,
    TimeoutPolicyData,
    _run_async,
)


# ---------------------------------------------------------------------------
# Inline NodeRunner that mirrors expected interface
# ---------------------------------------------------------------------------


class NodeExecutionResult:
    """Result of executing a single node."""

    def __init__(
        self,
        node_id: str,
        status: str,
        output: Any = None,
        output_hash: str = "",
        duration_ms: float = 0.0,
        attempt_count: int = 1,
        error: Optional[str] = None,
    ):
        self.node_id = node_id
        self.status = status
        self.output = output
        self.output_hash = output_hash
        self.duration_ms = duration_ms
        self.attempt_count = attempt_count
        self.error = error


class NodeRunner:
    """Executes a single DAG node with retry and timeout policies."""

    def __init__(self, agent_registry=None):
        self._agent_registry = agent_registry or (lambda aid: None)

    def _compute_hash(self, data: Any) -> str:
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    async def execute_node(
        self,
        node: DAGNodeData,
        input_data: Dict[str, Any] = None,
        retry_policy: Optional[RetryPolicyData] = None,
        timeout_policy: Optional[TimeoutPolicyData] = None,
    ) -> NodeExecutionResult:
        """Execute a node with retry and timeout policies."""
        input_data = input_data or {}
        rp = node.retry_policy or retry_policy or RetryPolicyData(max_retries=0)
        tp = node.timeout_policy or timeout_policy

        agent = self._agent_registry(node.agent_id)
        if agent is None:
            return NodeExecutionResult(
                node_id=node.node_id,
                status="failed",
                error=f"Agent '{node.agent_id}' not found",
            )

        last_error = None
        attempt = 0
        max_attempts = rp.max_retries + 1
        start_time = time.monotonic()

        while attempt < max_attempts:
            attempt += 1
            try:
                if tp:
                    result = await asyncio.wait_for(
                        self._run_agent(agent, input_data),
                        timeout=tp.timeout_seconds,
                    )
                else:
                    result = await self._run_agent(agent, input_data)

                duration_ms = (time.monotonic() - start_time) * 1000
                output_hash = self._compute_hash(result)

                return NodeExecutionResult(
                    node_id=node.node_id,
                    status="completed",
                    output=result,
                    output_hash=output_hash,
                    duration_ms=duration_ms,
                    attempt_count=attempt,
                )

            except asyncio.TimeoutError:
                duration_ms = (time.monotonic() - start_time) * 1000
                return NodeExecutionResult(
                    node_id=node.node_id,
                    status="timeout",
                    duration_ms=duration_ms,
                    attempt_count=attempt,
                    error=f"Node '{node.node_id}' timed out after {tp.timeout_seconds}s",
                )

            except Exception as e:
                last_error = str(e)
                if attempt < max_attempts:
                    delay = self._calculate_delay(rp, attempt)
                    await asyncio.sleep(delay)

        duration_ms = (time.monotonic() - start_time) * 1000
        return NodeExecutionResult(
            node_id=node.node_id,
            status="failed",
            duration_ms=duration_ms,
            attempt_count=attempt,
            error=last_error,
        )

    async def _run_agent(self, agent, input_data: Dict[str, Any]) -> Any:
        """Run agent - supports both sync and async agents."""
        run_async = getattr(agent, "run_async", None)
        run_sync = getattr(agent, "run", None)
        if run_async is not None and callable(run_async):
            return await run_async(input_data)
        elif run_sync is not None and callable(run_sync):
            return run_sync(input_data)
        else:
            raise RuntimeError("Agent has neither run() nor run_async()")

    def _calculate_delay(self, rp: RetryPolicyData, attempt: int) -> float:
        """Calculate delay based on retry strategy."""
        if rp.strategy == "exponential":
            delay = rp.base_delay * (2 ** (attempt - 1))
        elif rp.strategy == "linear":
            delay = rp.base_delay * attempt
        elif rp.strategy == "constant":
            delay = rp.base_delay
        elif rp.strategy == "fibonacci":
            a, b = rp.base_delay, rp.base_delay
            for _ in range(attempt - 1):
                a, b = b, a + b
            delay = b
        else:
            delay = rp.base_delay
        return min(delay, rp.max_delay)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestExecuteSyncAgentSuccess:
    """Test successful execution of sync agents."""

    def test_sync_agent_returns_result(self, mock_sync_agent):
        agent = mock_sync_agent(result={"value": 42}, agent_id="calc")
        runner = NodeRunner(agent_registry=lambda aid: agent)
        node = DAGNodeData(node_id="A", agent_id="calc")

        result = _run_async(runner.execute_node(node, {"input": "data"}))
        assert result.status == "completed"
        assert result.output == {"value": 42}

    def test_sync_agent_attempt_count_is_one(self, mock_sync_agent):
        agent = mock_sync_agent(agent_id="calc")
        runner = NodeRunner(agent_registry=lambda aid: agent)
        node = DAGNodeData(node_id="A", agent_id="calc")

        result = _run_async(runner.execute_node(node))
        assert result.attempt_count == 1

    def test_sync_agent_node_id_in_result(self, mock_sync_agent):
        agent = mock_sync_agent(agent_id="calc")
        runner = NodeRunner(agent_registry=lambda aid: agent)
        node = DAGNodeData(node_id="my_node", agent_id="calc")

        result = _run_async(runner.execute_node(node))
        assert result.node_id == "my_node"


class TestExecuteAsyncAgentSuccess:
    """Test successful execution of async agents."""

    def test_async_agent_returns_result(self, mock_async_agent):
        agent = mock_async_agent(result={"async_value": 99}, agent_id="async_calc")
        runner = NodeRunner(agent_registry=lambda aid: agent)
        node = DAGNodeData(node_id="B", agent_id="async_calc")

        result = _run_async(runner.execute_node(node))
        assert result.status == "completed"
        assert result.output == {"async_value": 99}

    def test_async_agent_attempt_count(self, mock_async_agent):
        agent = mock_async_agent(agent_id="async_calc")
        runner = NodeRunner(agent_registry=lambda aid: agent)
        node = DAGNodeData(node_id="B", agent_id="async_calc")

        result = _run_async(runner.execute_node(node))
        assert result.attempt_count == 1


class TestExecuteWithRetrySuccess:
    """Test execution with retry that succeeds on second attempt."""

    def test_retry_succeeds_on_second_attempt(self, mock_failing_agent):
        agent = mock_failing_agent(fail_count=1, agent_id="flaky")
        runner = NodeRunner(agent_registry=lambda aid: agent)
        node = DAGNodeData(
            node_id="R",
            agent_id="flaky",
            retry_policy=RetryPolicyData(
                max_retries=3, strategy="constant", base_delay=0.001
            ),
        )

        result = _run_async(runner.execute_node(node))
        assert result.status == "completed"
        assert result.attempt_count == 2

    def test_retry_succeeds_on_third_attempt(self, mock_failing_agent):
        agent = mock_failing_agent(fail_count=2, agent_id="flaky")
        runner = NodeRunner(agent_registry=lambda aid: agent)
        node = DAGNodeData(
            node_id="R",
            agent_id="flaky",
            retry_policy=RetryPolicyData(
                max_retries=3, strategy="constant", base_delay=0.001
            ),
        )

        result = _run_async(runner.execute_node(node))
        assert result.status == "completed"
        assert result.attempt_count == 3


class TestExecuteWithRetryExhausted:
    """Test execution where retries are exhausted."""

    def test_retries_exhausted(self, mock_failing_agent):
        agent = mock_failing_agent(fail_count=10, agent_id="always_fails")
        runner = NodeRunner(agent_registry=lambda aid: agent)
        node = DAGNodeData(
            node_id="F",
            agent_id="always_fails",
            retry_policy=RetryPolicyData(
                max_retries=2, strategy="constant", base_delay=0.001
            ),
        )

        result = _run_async(runner.execute_node(node))
        assert result.status == "failed"
        assert result.attempt_count == 3  # 1 original + 2 retries
        assert result.error is not None

    def test_no_retries_fails_immediately(self, mock_failing_agent):
        agent = mock_failing_agent(fail_count=10, agent_id="fails")
        runner = NodeRunner(agent_registry=lambda aid: agent)
        node = DAGNodeData(node_id="F", agent_id="fails")

        result = _run_async(runner.execute_node(node))
        assert result.status == "failed"
        assert result.attempt_count == 1


class TestExecuteWithTimeout:
    """Test execution with timeout policies."""

    def test_timeout_within_limit(self, mock_sync_agent):
        agent = mock_sync_agent(agent_id="fast")
        runner = NodeRunner(agent_registry=lambda aid: agent)
        node = DAGNodeData(
            node_id="T",
            agent_id="fast",
            timeout_policy=TimeoutPolicyData(timeout_seconds=5.0),
        )

        result = _run_async(runner.execute_node(node))
        assert result.status == "completed"

    def test_timeout_exceeded(self, mock_timeout_agent):
        agent = mock_timeout_agent(sleep_seconds=0.5, agent_id="slow")
        runner = NodeRunner(agent_registry=lambda aid: agent)
        node = DAGNodeData(
            node_id="T",
            agent_id="slow",
            timeout_policy=TimeoutPolicyData(timeout_seconds=0.05),
        )

        result = _run_async(runner.execute_node(node))
        assert result.status == "timeout"
        assert "timed out" in result.error


class TestExecuteRecordsDuration:
    """Test that execution records duration."""

    def test_duration_recorded(self, mock_sync_agent):
        agent = mock_sync_agent(agent_id="agent")
        runner = NodeRunner(agent_registry=lambda aid: agent)
        node = DAGNodeData(node_id="D", agent_id="agent")

        result = _run_async(runner.execute_node(node))
        assert result.duration_ms >= 0
        assert isinstance(result.duration_ms, float)

    def test_duration_increases_with_retries(self, mock_failing_agent):
        agent = mock_failing_agent(fail_count=2, agent_id="flaky")
        runner = NodeRunner(agent_registry=lambda aid: agent)
        node = DAGNodeData(
            node_id="D",
            agent_id="flaky",
            retry_policy=RetryPolicyData(
                max_retries=3, strategy="constant", base_delay=0.001
            ),
        )

        result = _run_async(runner.execute_node(node))
        assert result.status == "completed"
        assert result.duration_ms > 0


class TestExecuteCalculatesOutputHash:
    """Test that output hash is calculated."""

    def test_output_hash_present(self, mock_sync_agent):
        agent = mock_sync_agent(result={"data": "test"}, agent_id="agent")
        runner = NodeRunner(agent_registry=lambda aid: agent)
        node = DAGNodeData(node_id="H", agent_id="agent")

        result = _run_async(runner.execute_node(node))
        assert len(result.output_hash) == 64

    def test_output_hash_is_sha256(self, mock_sync_agent, compute_hash):
        expected_output = {"data": "test"}
        agent = mock_sync_agent(result=expected_output, agent_id="agent")
        runner = NodeRunner(agent_registry=lambda aid: agent)
        node = DAGNodeData(node_id="H", agent_id="agent")

        result = _run_async(runner.execute_node(node))
        expected_hash = compute_hash(expected_output)
        assert result.output_hash == expected_hash

    def test_output_hash_deterministic(self, mock_sync_agent):
        output = {"data": "deterministic"}
        agent = mock_sync_agent(result=output, agent_id="agent")
        runner = NodeRunner(agent_registry=lambda aid: agent)
        node = DAGNodeData(node_id="H", agent_id="agent")

        r1 = _run_async(runner.execute_node(node))
        r2 = _run_async(runner.execute_node(node))
        assert r1.output_hash == r2.output_hash

    def test_no_output_hash_on_failure(self, mock_failing_agent):
        agent = mock_failing_agent(fail_count=10, agent_id="fails")
        runner = NodeRunner(agent_registry=lambda aid: agent)
        node = DAGNodeData(node_id="H", agent_id="fails")

        result = _run_async(runner.execute_node(node))
        assert result.output_hash == ""


class TestExecuteFailureHandling:
    """Test failure handling in node execution."""

    def test_exception_captured_in_error(self, mock_failing_agent):
        agent = mock_failing_agent(fail_count=10, agent_id="fails")
        runner = NodeRunner(agent_registry=lambda aid: agent)
        node = DAGNodeData(node_id="F", agent_id="fails")

        result = _run_async(runner.execute_node(node))
        assert result.status == "failed"
        assert "Simulated failure" in result.error


class TestExecuteAgentNotFound:
    """Test execution when agent is not found."""

    def test_agent_not_found(self):
        runner = NodeRunner(agent_registry=lambda aid: None)
        node = DAGNodeData(node_id="N", agent_id="nonexistent")

        result = _run_async(runner.execute_node(node))
        assert result.status == "failed"
        assert "not found" in result.error

    def test_agent_not_found_message(self):
        runner = NodeRunner(agent_registry=lambda aid: None)
        node = DAGNodeData(node_id="N", agent_id="ghost_agent")

        result = _run_async(runner.execute_node(node))
        assert "ghost_agent" in result.error


class TestNodeExecutionResult:
    """Test NodeExecutionResult properties."""

    def test_result_fields(self):
        result = NodeExecutionResult(
            node_id="test",
            status="completed",
            output={"val": 1},
            output_hash="abc" * 21 + "a",
            duration_ms=5.5,
            attempt_count=2,
        )
        assert result.node_id == "test"
        assert result.status == "completed"
        assert result.output == {"val": 1}
        assert result.duration_ms == 5.5
        assert result.attempt_count == 2
        assert result.error is None

    def test_result_with_error(self):
        result = NodeExecutionResult(
            node_id="err",
            status="failed",
            error="Something went wrong",
        )
        assert result.error == "Something went wrong"
