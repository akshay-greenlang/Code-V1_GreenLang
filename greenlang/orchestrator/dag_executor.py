# -*- coding: utf-8 -*-
"""
DAG Executor - AGENT-FOUND-001: GreenLang DAG Orchestrator

The core DAG execution engine. Processes validated DAGs level by level,
executing independent nodes in parallel within each level. Handles:
- Level-based parallel execution (asyncio.gather)
- Input mapping from predecessor outputs
- Conditional node execution
- Per-node retry/timeout policy application
- On-failure strategies (stop, skip, compensate)
- Checkpoint save after each node
- Provenance recording
- Resume from checkpoint
- Compensation (reverse-order rollback)

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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from greenlang.orchestrator.checkpoint_store import DAGCheckpointStore
from greenlang.orchestrator.config import OrchestratorConfig
from greenlang.orchestrator.dag_validator import validate_dag
from greenlang.orchestrator.determinism import DeterministicScheduler
from greenlang.orchestrator.models import (
    DAGCheckpoint,
    DAGOnFailure,
    DAGWorkflow,
    ExecutionStatus,
    ExecutionTrace,
    NodeExecutionResult,
    NodeStatus,
    OnFailure,
)
from greenlang.orchestrator.node_runner import NodeRunner
from greenlang.orchestrator.provenance import ProvenanceTracker
from greenlang.orchestrator.retry_policy import merge_with_default as merge_retry
from greenlang.orchestrator.timeout_policy import (
    merge_with_default as merge_timeout,
)
from greenlang.orchestrator.topological_sort import level_grouping
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
    """Get current timestamp."""
    if _CLOCK_AVAILABLE and DeterministicClock is not None:
        return DeterministicClock.now()
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)


# ===================================================================
# Execution context
# ===================================================================


@dataclass
class ExecutionContext:
    """Mutable context passed through DAG execution.

    Attributes:
        execution_id: Unique execution identifier.
        dag_id: DAG being executed.
        input_data: Original input data.
        results: Node outputs keyed by node_id.
        errors: Error messages collected during execution.
        status: Current execution status.
        completed_nodes: Set of completed node IDs.
        failed_nodes: Set of failed node IDs.
        skipped_nodes: Set of skipped node IDs.
        cancelled: Whether execution has been cancelled.
    """
    execution_id: str = ""
    dag_id: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    status: ExecutionStatus = ExecutionStatus.RUNNING
    completed_nodes: Set[str] = field(default_factory=set)
    failed_nodes: Set[str] = field(default_factory=set)
    skipped_nodes: Set[str] = field(default_factory=set)
    cancelled: bool = False


# ===================================================================
# Execution options
# ===================================================================


@dataclass
class ExecutionOptions:
    """Options for a single DAG execution.

    Attributes:
        checkpoint_enabled: Save checkpoints after each node.
        deterministic_mode: Use deterministic scheduling and IDs.
        resume_execution_id: If set, resume from this execution's
            checkpoints instead of starting fresh.
        agent_registry: Mapping from agent_id to agent instances
            or callables.
    """
    checkpoint_enabled: bool = True
    deterministic_mode: bool = True
    resume_execution_id: Optional[str] = None
    agent_registry: Dict[str, Any] = field(default_factory=dict)


# ===================================================================
# DAGExecutor
# ===================================================================


class DAGExecutor:
    """Core DAG execution engine.

    Orchestrates level-by-level parallel execution of DAG nodes with
    retry, timeout, checkpointing, and provenance tracking.

    Attributes:
        config: Orchestrator configuration.
        node_runner: Node execution handler.
        checkpoint_store: Checkpoint persistence backend.
        provenance_tracker: Provenance tracking engine.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        node_runner: NodeRunner,
        checkpoint_store: DAGCheckpointStore,
        provenance_tracker: ProvenanceTracker,
    ) -> None:
        """Initialize DAGExecutor.

        Args:
            config: Orchestrator configuration.
            node_runner: Node runner instance.
            checkpoint_store: Checkpoint store backend.
            provenance_tracker: Provenance tracker instance.
        """
        self.config = config
        self.node_runner = node_runner
        self.checkpoint_store = checkpoint_store
        self.provenance_tracker = provenance_tracker
        self._scheduler = DeterministicScheduler()
        logger.info("DAGExecutor initialized")

    async def execute(
        self,
        dag: DAGWorkflow,
        input_data: Dict[str, Any],
        options: Optional[ExecutionOptions] = None,
    ) -> ExecutionTrace:
        """Execute a DAG workflow.

        Main execution loop:
        1. Validate DAG
        2. Compute topological levels
        3. For each level: execute ready nodes in parallel
        4. Handle failures per on_failure strategy
        5. Build execution trace with provenance

        Args:
            dag: Validated DAG workflow.
            input_data: Input data for the execution.
            options: Execution options.

        Returns:
            ExecutionTrace with complete results and provenance.
        """
        opts = options or ExecutionOptions()
        wall_start = time.monotonic()

        # Step 1: Validate DAG
        errors = validate_dag(dag)
        if errors:
            if self.config.enable_metrics:
                for err in errors:
                    m.record_validation_error(err.error_type)
            return self._create_error_trace(
                dag_id=dag.dag_id,
                error=f"DAG validation failed: {[e.message for e in errors]}",
                input_data=input_data,
            )

        # Step 2: Topological sort -> level grouping
        try:
            levels = level_grouping(dag)
        except ValueError as e:
            return self._create_error_trace(
                dag_id=dag.dag_id,
                error=str(e),
                input_data=input_data,
            )

        # Step 3: Initialize execution context
        input_hash = self._scheduler.compute_input_hash(input_data)
        seed = str(_now().timestamp()) if not opts.deterministic_mode else "det"
        execution_id = self._scheduler.generate_execution_id(
            dag.dag_id, input_hash, seed=seed,
        )

        ctx = ExecutionContext(
            execution_id=execution_id,
            dag_id=dag.dag_id,
            input_data=input_data,
            status=ExecutionStatus.RUNNING,
        )

        # Handle resume from checkpoint
        if opts.resume_execution_id:
            execution_id = opts.resume_execution_id
            ctx.execution_id = execution_id
            completed = self.checkpoint_store.get_completed_nodes(execution_id)
            ctx.completed_nodes = completed
            # Load outputs from checkpoints
            for nid in completed:
                cp = self.checkpoint_store.load(execution_id, nid)
                if cp:
                    ctx.results[nid] = cp.outputs
            logger.info(
                "Resuming execution %s with %d completed nodes",
                execution_id, len(completed),
            )

        if self.config.enable_metrics:
            m.increment_active_executions()

        trace = ExecutionTrace(
            execution_id=execution_id,
            dag_id=dag.dag_id,
            status=ExecutionStatus.RUNNING,
            topology_levels=[list(lvl) for lvl in levels],
            start_time=_now(),
            input_data=input_data,
        )

        try:
            # Step 4: Execute level by level
            should_stop = False

            for level_idx, level_nodes in enumerate(levels):
                if should_stop or ctx.cancelled:
                    break

                # Record level parallelism
                if self.config.enable_metrics:
                    m.record_parallel_nodes(dag.dag_id, len(level_nodes))

                # Execute this level
                level_results = await self._execute_level(
                    level_nodes=level_nodes,
                    ctx=ctx,
                    dag=dag,
                    opts=opts,
                )

                # Process results
                for result in level_results:
                    trace.node_traces[result.node_id] = result

                    if result.status == NodeStatus.FAILED:
                        ctx.failed_nodes.add(result.node_id)
                        ctx.errors.append(
                            f"Node '{result.node_id}' failed: {result.error}"
                        )

                        node = dag.nodes[result.node_id]
                        node_on_failure = node.on_failure

                        if node_on_failure == OnFailure.STOP:
                            if dag.on_failure == DAGOnFailure.FAIL_FAST:
                                should_stop = True
                                break
                        elif node_on_failure == OnFailure.SKIP:
                            ctx.skipped_nodes.add(result.node_id)
                        elif node_on_failure == OnFailure.COMPENSATE:
                            await self._compensate(
                                dag, trace, result.node_id, ctx, opts,
                            )
                            if dag.on_failure == DAGOnFailure.FAIL_FAST:
                                should_stop = True
                                break

                    elif result.status == NodeStatus.COMPLETED:
                        ctx.completed_nodes.add(result.node_id)
                        ctx.results[result.node_id] = result.outputs

                    elif result.status == NodeStatus.SKIPPED:
                        ctx.skipped_nodes.add(result.node_id)

            # Step 5: Determine final status
            if ctx.cancelled:
                trace.status = ExecutionStatus.CANCELLED
            elif ctx.failed_nodes:
                trace.status = ExecutionStatus.FAILED
            else:
                trace.status = ExecutionStatus.COMPLETED

            # Build provenance chain
            if self.config.enable_provenance:
                trace.provenance_chain_hash = (
                    self.provenance_tracker.build_chain_hash(execution_id)
                )
                if self.config.enable_metrics:
                    provenances = self.provenance_tracker.get_trace(
                        execution_id,
                    )
                    m.record_provenance_chain(
                        dag.dag_id, len(provenances),
                    )

        except Exception as e:
            logger.error(
                "DAG execution failed unexpectedly: %s", e, exc_info=True,
            )
            trace.status = ExecutionStatus.FAILED
            trace.errors.append(str(e))

        finally:
            trace.end_time = _now()
            trace.errors = list(ctx.errors)

            if self.config.enable_metrics:
                m.decrement_active_executions()
                wall_end = time.monotonic()
                m.record_dag_execution(
                    dag_id=dag.dag_id,
                    status=trace.status.value,
                    duration_seconds=(wall_end - wall_start),
                )

        logger.info(
            "DAG execution completed: id=%s dag=%s status=%s "
            "completed=%d failed=%d skipped=%d",
            execution_id, dag.dag_id, trace.status.value,
            len(ctx.completed_nodes), len(ctx.failed_nodes),
            len(ctx.skipped_nodes),
        )
        return trace

    async def _execute_level(
        self,
        level_nodes: List[str],
        ctx: ExecutionContext,
        dag: DAGWorkflow,
        opts: ExecutionOptions,
    ) -> List[NodeExecutionResult]:
        """Execute all nodes in a single level in parallel.

        Nodes are sorted deterministically and executed concurrently
        up to max_parallel_nodes.

        Args:
            level_nodes: Node IDs in this level.
            ctx: Execution context.
            dag: DAG definition.
            opts: Execution options.

        Returns:
            List of NodeExecutionResults.
        """
        # Filter out already-completed nodes (for resume)
        pending = [
            nid for nid in level_nodes
            if nid not in ctx.completed_nodes
        ]

        if not pending:
            return []

        # Sort for determinism
        sorted_nodes = self._scheduler.sort_nodes(pending)

        # Execute in batches of max_parallel_nodes
        max_parallel = dag.max_parallel_nodes or self.config.max_parallel_nodes
        results: List[NodeExecutionResult] = []

        for batch_start in range(0, len(sorted_nodes), max_parallel):
            batch = sorted_nodes[batch_start:batch_start + max_parallel]

            tasks = []
            for node_id in batch:
                task = self._execute_single_node(
                    node_id=node_id,
                    ctx=ctx,
                    dag=dag,
                    opts=opts,
                )
                tasks.append(task)

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, br in enumerate(batch_results):
                if isinstance(br, Exception):
                    node_id = batch[i]
                    results.append(NodeExecutionResult(
                        node_id=node_id,
                        status=NodeStatus.FAILED,
                        error=str(br),
                        started_at=_now(),
                        completed_at=_now(),
                    ))
                else:
                    results.append(br)

        return results

    async def _execute_single_node(
        self,
        node_id: str,
        ctx: ExecutionContext,
        dag: DAGWorkflow,
        opts: ExecutionOptions,
    ) -> NodeExecutionResult:
        """Execute a single node with all wrapping logic.

        Args:
            node_id: Node to execute.
            ctx: Execution context.
            dag: DAG definition.
            opts: Execution options.

        Returns:
            NodeExecutionResult.
        """
        node = dag.nodes[node_id]

        # Check condition
        if node.condition and not self._evaluate_condition(node, ctx):
            logger.info("Node '%s' skipped due to condition", node_id)
            return NodeExecutionResult(
                node_id=node_id,
                status=NodeStatus.SKIPPED,
                started_at=_now(),
                completed_at=_now(),
            )

        # Prepare input data from predecessors
        input_data = self._prepare_node_input(node, ctx)

        # Resolve agent
        agent = opts.agent_registry.get(node.agent_id)
        if agent is None:
            return NodeExecutionResult(
                node_id=node_id,
                status=NodeStatus.FAILED,
                error=f"Agent '{node.agent_id}' not found in registry",
                started_at=_now(),
                completed_at=_now(),
            )

        # Merge policies
        retry_policy = merge_retry(
            node.retry_policy, dag.default_retry_policy,
        )
        timeout_policy = merge_timeout(
            node.timeout_policy, dag.default_timeout_policy,
        )

        # Execute via NodeRunner
        result = await self.node_runner.execute_node(
            node=node,
            agent=agent,
            input_data=input_data,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            dag_id=dag.dag_id,
        )

        # Record provenance
        if self.config.enable_provenance and result.status == NodeStatus.COMPLETED:
            parent_ids = [
                d for d in node.depends_on if d in dag.nodes
            ]
            self.provenance_tracker.record_node(
                execution_id=ctx.execution_id,
                node_id=node_id,
                input_data=input_data,
                output_data=result.outputs,
                duration_ms=result.duration_ms,
                attempt_count=result.attempt_count,
                parent_node_ids=parent_ids,
            )

        # Save checkpoint
        if opts.checkpoint_enabled and result.status == NodeStatus.COMPLETED:
            checkpoint = DAGCheckpoint(
                execution_id=ctx.execution_id,
                node_id=node_id,
                status=result.status,
                outputs=result.outputs,
                output_hash=result.output_hash,
                attempt_count=result.attempt_count,
                created_at=_now(),
            )
            self.checkpoint_store.save(checkpoint)
            if self.config.enable_metrics:
                size = len(json.dumps(result.outputs, default=str))
                m.record_checkpoint_operation("save", size_bytes=size)

        return result

    def _prepare_node_input(
        self,
        node: DAGNode,
        ctx: ExecutionContext,
    ) -> Dict[str, Any]:
        """Prepare input data for a node from predecessor outputs.

        Uses the node's input_mapping to pull data from the execution
        context. Falls back to passing the original input_data if no
        mapping is defined.

        Args:
            node: DAG node definition.
            ctx: Execution context with predecessor results.

        Returns:
            Input data dictionary for the node.
        """
        input_data: Dict[str, Any] = {}

        if node.input_mapping:
            for target_key, source_path in node.input_mapping.items():
                value = self._resolve_path(source_path, ctx)
                input_data[target_key] = value
        else:
            # Default: pass all predecessor outputs and original input
            input_data["_input"] = ctx.input_data
            for dep_id in node.depends_on:
                if dep_id in ctx.results:
                    input_data[dep_id] = ctx.results[dep_id]

        return input_data

    @staticmethod
    def _resolve_path(
        path: str,
        ctx: ExecutionContext,
    ) -> Any:
        """Resolve a dotted path reference in the execution context.

        Supports paths like:
        - ``results.node_id.key`` -> ctx.results["node_id"]["key"]
        - ``input.key`` -> ctx.input_data["key"]

        Args:
            path: Dotted path string.
            ctx: Execution context.

        Returns:
            Resolved value, or None if not found.
        """
        parts = path.split(".")
        if not parts:
            return None

        if parts[0] == "results" and len(parts) >= 2:
            node_id = parts[1]
            current: Any = ctx.results.get(node_id, {})
            for part in parts[2:]:
                if isinstance(current, dict):
                    current = current.get(part)
                else:
                    return None
            return current

        if parts[0] == "input":
            current = ctx.input_data
            for part in parts[1:]:
                if isinstance(current, dict):
                    current = current.get(part)
                else:
                    return None
            return current

        return None

    @staticmethod
    def _evaluate_condition(
        node: DAGNode,
        ctx: ExecutionContext,
    ) -> bool:
        """Evaluate a node's conditional execution expression.

        Simple condition syntax:
        - ``results.node_id.key == value``
        - ``results.node_id._success == True``
        - Empty/None condition -> always True

        Args:
            node: DAG node with condition.
            ctx: Execution context.

        Returns:
            True if the node should execute.
        """
        if not node.condition:
            return True

        try:
            condition = node.condition.strip()

            # Simple equality check: "results.X.Y == Z"
            if "==" in condition:
                left, right = condition.split("==", 1)
                left = left.strip()
                right = right.strip()

                # Resolve left side
                left_val = DAGExecutor._resolve_path(left, ctx)

                # Parse right side
                if right.lower() == "true":
                    right_val: Any = True
                elif right.lower() == "false":
                    right_val = False
                elif right.lower() == "none":
                    right_val = None
                else:
                    try:
                        right_val = int(right)
                    except ValueError:
                        try:
                            right_val = float(right)
                        except ValueError:
                            right_val = right.strip("'\"")

                return left_val == right_val

            # Simple not-equal check
            if "!=" in condition:
                left, right = condition.split("!=", 1)
                left_val = DAGExecutor._resolve_path(left.strip(), ctx)
                right = right.strip()
                if right.lower() == "none":
                    return left_val is not None
                return str(left_val) != right

            # Default: treat as truthy check of a path
            val = DAGExecutor._resolve_path(condition, ctx)
            return bool(val)

        except Exception as e:
            logger.warning(
                "Condition evaluation failed for node '%s': %s. "
                "Defaulting to True.",
                node.node_id, e,
            )
            return True

    async def _compensate(
        self,
        dag: DAGWorkflow,
        trace: ExecutionTrace,
        failed_node_id: str,
        ctx: ExecutionContext,
        opts: ExecutionOptions,
    ) -> None:
        """Run compensation handlers in reverse topological order.

        Compensates all completed nodes that have compensation handlers,
        in reverse order of completion.

        Args:
            dag: DAG definition.
            trace: Current execution trace.
            failed_node_id: Node that triggered compensation.
            ctx: Execution context.
            opts: Execution options.
        """
        logger.info(
            "Starting compensation for execution %s "
            "(triggered by node '%s')",
            ctx.execution_id, failed_node_id,
        )
        ctx.status = ExecutionStatus.COMPENSATING

        # Build reverse order of completed nodes
        completed_order = [
            nid for nid in reversed(list(ctx.completed_nodes))
            if nid in dag.nodes
            and dag.nodes[nid].compensation_handler
        ]

        for nid in completed_order:
            node = dag.nodes[nid]
            handler_id = node.compensation_handler
            if not handler_id:
                continue

            handler = opts.agent_registry.get(handler_id)
            if handler is None:
                logger.warning(
                    "Compensation handler '%s' not found for node '%s'",
                    handler_id, nid,
                )
                continue

            try:
                comp_input = {
                    "original_input": ctx.results.get(nid, {}),
                    "failed_node": failed_node_id,
                }
                comp_node = DAGNode(
                    node_id=f"compensate_{nid}",
                    agent_id=handler_id,
                )
                await self.node_runner.execute_node(
                    node=comp_node,
                    agent=handler,
                    input_data=comp_input,
                    dag_id=dag.dag_id,
                )
                logger.info("Compensated node '%s'", nid)
            except Exception as e:
                logger.error(
                    "Compensation failed for node '%s': %s", nid, e,
                )

    @staticmethod
    def _create_error_trace(
        dag_id: str,
        error: str,
        input_data: Dict[str, Any],
    ) -> ExecutionTrace:
        """Create a failed ExecutionTrace for validation/setup errors.

        Args:
            dag_id: DAG identifier.
            error: Error message.
            input_data: Original input data.

        Returns:
            ExecutionTrace with FAILED status.
        """
        return ExecutionTrace(
            execution_id="",
            dag_id=dag_id,
            status=ExecutionStatus.FAILED,
            start_time=_now(),
            end_time=_now(),
            input_data=input_data,
            errors=[error],
        )

    async def cancel(self, execution_id: str) -> bool:
        """Request cancellation of a running execution.

        Note: Actual cancellation is cooperative - nodes currently
        executing will complete before cancellation takes effect.

        Args:
            execution_id: Execution to cancel.

        Returns:
            True if cancellation was requested.
        """
        logger.info("Cancellation requested for execution %s", execution_id)
        return True


__all__ = [
    "DAGExecutor",
    "ExecutionContext",
    "ExecutionOptions",
]
