# -*- coding: utf-8 -*-
"""
Node Runner - AGENT-FOUND-001: GreenLang DAG Orchestrator

Executes individual DAG nodes with retry, timeout, and provenance:
- Handles both sync (run_in_executor) and async agents
- Retry loop with configurable backoff and jitter
- Timeout via asyncio.wait_for
- Input/output hashing for provenance tracking
- Timing and attempt counting

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-001 GreenLang Orchestrator
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

from greenlang.orchestrator.config import OrchestratorConfig
from greenlang.orchestrator.models import (
    DAGNode,
    NodeExecutionResult,
    NodeStatus,
    RetryPolicy,
    TimeoutPolicy,
)
from greenlang.orchestrator.retry_policy import (
    DEFAULT_RETRY_POLICY,
    calculate_delay,
    should_retry,
)
from greenlang.orchestrator.timeout_policy import DEFAULT_TIMEOUT_POLICY
from greenlang.orchestrator import metrics as m

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional clock import
# ---------------------------------------------------------------------------

try:
    from greenlang.utilities.determinism.clock import DeterministicClock
    _CLOCK_AVAILABLE = True
except ImportError:
    DeterministicClock = None  # type: ignore[assignment, misc]
    _CLOCK_AVAILABLE = False


def _now():
    """Get current timestamp using DeterministicClock if available."""
    if _CLOCK_AVAILABLE and DeterministicClock is not None:
        return DeterministicClock.now()
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of data for provenance."""
    if isinstance(data, (dict, list)):
        content = json.dumps(data, sort_keys=True, default=str)
    elif isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    else:
        content = str(data)
    return hashlib.sha256(content.encode()).hexdigest()


# ===================================================================
# NodeRunner
# ===================================================================


class NodeRunner:
    """Executes individual DAG nodes with retry and timeout.

    Handles both synchronous and asynchronous agent execution,
    applying per-node retry and timeout policies.

    Attributes:
        config: Orchestrator configuration.
        _executor: Thread pool for running sync agents.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        """Initialize NodeRunner.

        Args:
            config: Orchestrator configuration (uses default if None).
        """
        self.config = config or OrchestratorConfig()
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_parallel_nodes,
        )
        logger.debug("NodeRunner initialized")

    async def execute_node(
        self,
        node: DAGNode,
        agent: Any,
        input_data: Dict[str, Any],
        retry_policy: Optional[RetryPolicy] = None,
        timeout_policy: Optional[TimeoutPolicy] = None,
        dag_id: str = "",
    ) -> NodeExecutionResult:
        """Execute a single DAG node with retry and timeout handling.

        Supports both sync and async agents:
        - If agent has ``run_async`` method, it is awaited directly.
        - If agent has ``run`` method, it is called in a thread executor.
        - If agent is a callable, it is called directly.

        Args:
            node: DAG node definition.
            agent: Agent instance or callable.
            input_data: Input data for the node.
            retry_policy: Retry policy (uses default if None).
            timeout_policy: Timeout policy (uses default if None).
            dag_id: Parent DAG ID for metrics labels.

        Returns:
            NodeExecutionResult with status, outputs, timing, and hashes.
        """
        effective_retry = retry_policy or DEFAULT_RETRY_POLICY
        effective_timeout = timeout_policy or DEFAULT_TIMEOUT_POLICY

        input_hash = _compute_hash(input_data)
        start_time = _now()
        wall_start = time.monotonic()

        attempt = 0
        last_error: Optional[Exception] = None

        while True:
            attempt += 1
            try:
                # Execute with timeout
                result = await self._execute_with_timeout(
                    node=node,
                    agent=agent,
                    input_data=input_data,
                    timeout_policy=effective_timeout,
                    dag_id=dag_id,
                )

                # Success
                wall_end = time.monotonic()
                duration_ms = (wall_end - wall_start) * 1000
                output_hash = _compute_hash(result)

                # Record metrics
                if self.config.enable_metrics:
                    m.record_node_execution(
                        dag_id=dag_id,
                        node_id=node.node_id,
                        status="completed",
                        duration_seconds=(wall_end - wall_start),
                    )

                return NodeExecutionResult(
                    node_id=node.node_id,
                    status=NodeStatus.COMPLETED,
                    outputs=result if isinstance(result, dict) else {"result": result},
                    output_hash=output_hash,
                    duration_ms=duration_ms,
                    attempt_count=attempt,
                    started_at=start_time,
                    completed_at=_now(),
                )

            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError(
                    f"Node '{node.node_id}' timed out after "
                    f"{effective_timeout.timeout_seconds}s"
                )
                logger.warning(
                    "Node '%s' timed out (attempt %d/%d)",
                    node.node_id, attempt, effective_retry.max_retries + 1,
                )
                if self.config.enable_metrics:
                    m.record_node_timeout(dag_id=dag_id, node_id=node.node_id)

                # Check if we should retry the timeout
                if not should_retry(effective_retry, last_error, attempt):
                    break

                # Record retry metric
                if self.config.enable_metrics:
                    m.record_node_retry(dag_id=dag_id, node_id=node.node_id)

                delay = calculate_delay(effective_retry, attempt - 1)
                logger.info(
                    "Retrying node '%s' in %.2fs (attempt %d/%d)",
                    node.node_id, delay, attempt + 1,
                    effective_retry.max_retries + 1,
                )
                await asyncio.sleep(delay)

            except Exception as e:
                last_error = e
                logger.warning(
                    "Node '%s' failed (attempt %d/%d): %s",
                    node.node_id, attempt,
                    effective_retry.max_retries + 1, str(e),
                )

                # Check if we should retry
                if not should_retry(effective_retry, e, attempt):
                    break

                # Record retry metric
                if self.config.enable_metrics:
                    m.record_node_retry(dag_id=dag_id, node_id=node.node_id)

                delay = calculate_delay(effective_retry, attempt - 1)
                logger.info(
                    "Retrying node '%s' in %.2fs (attempt %d/%d)",
                    node.node_id, delay, attempt + 1,
                    effective_retry.max_retries + 1,
                )
                await asyncio.sleep(delay)

        # All attempts exhausted - return failure
        wall_end = time.monotonic()
        duration_ms = (wall_end - wall_start) * 1000

        if self.config.enable_metrics:
            m.record_node_execution(
                dag_id=dag_id,
                node_id=node.node_id,
                status="failed",
                duration_seconds=(wall_end - wall_start),
            )

        return NodeExecutionResult(
            node_id=node.node_id,
            status=NodeStatus.FAILED,
            outputs={},
            output_hash="",
            duration_ms=duration_ms,
            attempt_count=attempt,
            error=str(last_error) if last_error else "Unknown error",
            started_at=start_time,
            completed_at=_now(),
        )

    async def _execute_with_timeout(
        self,
        node: DAGNode,
        agent: Any,
        input_data: Dict[str, Any],
        timeout_policy: TimeoutPolicy,
        dag_id: str = "",
    ) -> Any:
        """Execute an agent with timeout wrapping.

        Args:
            node: DAG node definition.
            agent: Agent instance or callable.
            input_data: Input data.
            timeout_policy: Timeout policy.
            dag_id: DAG identifier.

        Returns:
            Agent execution result.
        """
        coro = self._invoke_agent(node, agent, input_data)
        return await asyncio.wait_for(
            coro, timeout=timeout_policy.timeout_seconds,
        )

    async def _invoke_agent(
        self,
        node: DAGNode,
        agent: Any,
        input_data: Dict[str, Any],
    ) -> Any:
        """Invoke an agent, handling sync and async patterns.

        Args:
            node: DAG node definition.
            agent: Agent instance or callable.
            input_data: Input data.

        Returns:
            Agent result (dict or AgentResult).
        """
        # Async agent (has run_async method)
        if hasattr(agent, "run_async"):
            result = await agent.run_async(input_data)
            return self._normalize_result(result)

        # Sync agent (has run method) - run in executor
        if hasattr(agent, "run"):
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor, agent.run, input_data,
            )
            return self._normalize_result(result)

        # Async callable
        if asyncio.iscoroutinefunction(agent):
            result = await agent(input_data)
            return self._normalize_result(result)

        # Sync callable - run in executor
        if callable(agent):
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor, agent, input_data,
            )
            return self._normalize_result(result)

        raise TypeError(
            f"Agent for node '{node.node_id}' is not callable and has no "
            f"run/run_async method: {type(agent)}"
        )

    @staticmethod
    def _normalize_result(result: Any) -> Dict[str, Any]:
        """Normalize agent result to a dictionary.

        Args:
            result: Raw agent result.

        Returns:
            Dictionary representation of the result.
        """
        if isinstance(result, dict):
            return result

        # Handle AgentResult (has .data and .success attributes)
        if hasattr(result, "data") and hasattr(result, "success"):
            data = result.data if isinstance(result.data, dict) else {}
            data["_success"] = result.success
            if hasattr(result, "error") and result.error:
                data["_error"] = result.error
            return data

        # Handle BaseModel (has .model_dump or .dict)
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if hasattr(result, "dict"):
            return result.dict()

        # Fallback
        return {"result": result}

    def shutdown(self) -> None:
        """Shutdown the thread pool executor."""
        self._executor.shutdown(wait=False)
        logger.debug("NodeRunner thread pool shut down")


__all__ = [
    "NodeRunner",
]
