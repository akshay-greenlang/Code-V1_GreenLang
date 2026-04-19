# -*- coding: utf-8 -*-
"""
Fan-Out Executor Module for GreenLang Orchestrator (FR-025)
============================================================

This module implements dynamic fan-out map/join step types for batch processing
pipelines. It provides MapStepExecutor and JoinStepExecutor classes for handling
parallel execution of child steps with concurrency control and result aggregation.

Key Features:
- Deterministic child step ID generation using SHA-256 hashes
- Configurable concurrency limits (maxParallel: 1-100)
- Policy-enforced item limits (maxItems: 1-10000)
- Multiple failure policies (failFast, continueOnError, allowPartial)
- Multiple aggregation strategies (list, merge, first, last, custom)
- Comprehensive event emission for audit trails
- Cancellation support for all child steps

Example Pipeline YAML:
    steps:
      - id: process_all
        type: map
        items: "{{ params.file_list }}"
        itemVar: file
        maxParallel: 10
        maxItems: 1000
        agent: OPS.DATA.Transform
        in:
          input: "{{ item.file }}"

      - id: collect_results
        type: join
        depends_on: [process_all]

Author: GreenLang Framework Team
Date: January 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class StepType(str, Enum):
    """Step type enumeration for pipeline steps."""
    AGENT_TASK = "agentTask"
    MAP = "map"
    JOIN = "join"
    CONDITION = "condition"
    NOOP = "noop"


class FailurePolicy(str, Enum):
    """Failure handling policy for map step children."""
    FAIL_FAST = "failFast"
    CONTINUE_ON_ERROR = "continueOnError"
    ALLOW_PARTIAL = "allowPartial"


class AggregatorType(str, Enum):
    """Result aggregation strategy for join steps."""
    LIST = "list"
    MERGE = "merge"
    FIRST = "first"
    LAST = "last"
    CUSTOM = "custom"


class ChildStepStatus(str, Enum):
    """Status of a child step in fan-out execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class FanoutEventType(str, Enum):
    """Event types specific to fan-out operations."""
    FANOUT_STARTED = "fanout.started"
    FANOUT_ITEM_STARTED = "fanout.item_started"
    FANOUT_ITEM_COMPLETED = "fanout.item_completed"
    FANOUT_ITEM_FAILED = "fanout.item_failed"
    FANOUT_COMPLETED = "fanout.completed"
    FANOUT_CANCELLED = "fanout.cancelled"
    JOIN_STARTED = "join.started"
    JOIN_COMPLETED = "join.completed"


# Policy defaults
DEFAULT_MAX_PARALLEL = 10
DEFAULT_MAX_ITEMS = 1000
DEFAULT_ITEM_VAR = "item"
DEFAULT_PARTIAL_SUCCESS_THRESHOLD = 0.8
GLOBAL_MAX_ITEMS_LIMIT = 10000
GLOBAL_MAX_PARALLEL_LIMIT = 100


# =============================================================================
# Pydantic Models
# =============================================================================

class MapStepConfig(BaseModel):
    """Configuration for a map (fan-out) step."""

    step_id: str = Field(..., description="Parent map step identifier")
    items_expression: str = Field(..., description="Expression for items array")
    item_var: str = Field(DEFAULT_ITEM_VAR, description="Item variable name")
    max_parallel: int = Field(
        DEFAULT_MAX_PARALLEL,
        ge=1,
        le=GLOBAL_MAX_PARALLEL_LIMIT,
        description="Max concurrent executions"
    )
    max_items: int = Field(
        DEFAULT_MAX_ITEMS,
        ge=1,
        le=GLOBAL_MAX_ITEMS_LIMIT,
        description="Max items allowed"
    )
    failure_policy: FailurePolicy = Field(
        FailurePolicy.FAIL_FAST,
        description="Child failure handling policy"
    )
    partial_success_threshold: float = Field(
        DEFAULT_PARTIAL_SUCCESS_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Min success ratio for allowPartial"
    )
    agent_id: str = Field(..., description="Agent ID for child execution")
    input_mapping: Optional[Dict[str, Any]] = Field(None, description="Input mapping template")
    timeout: Optional[float] = Field(None, ge=0.1, description="Child timeout")

    @validator("items_expression")
    def validate_items_expression(cls, v: str) -> str:
        """Validate items expression has proper format."""
        if not v or not v.strip():
            raise ValueError("items_expression cannot be empty")
        return v.strip()


class JoinStepConfig(BaseModel):
    """Configuration for a join step."""

    step_id: str = Field(..., description="Join step identifier")
    upstream_step_ids: List[str] = Field(..., min_items=1, description="Upstream map step IDs")
    aggregator: AggregatorType = Field(AggregatorType.LIST, description="Result aggregation strategy")
    agent_id: Optional[str] = Field(None, description="Agent for custom aggregation")
    timeout: Optional[float] = Field(None, ge=0.1, description="Join timeout")


class ChildStepResult(BaseModel):
    """Result from a child step execution."""

    child_step_id: str = Field(..., description="Unique child step ID")
    index: int = Field(..., ge=0, description="Item index")
    item: Any = Field(..., description="Processed item")
    item_hash: str = Field(..., description="Item hash (8 chars)")
    status: ChildStepStatus = Field(..., description="Execution status")
    result: Optional[Dict[str, Any]] = Field(None, description="Success result")
    error: Optional[str] = Field(None, description="Error message")
    started_at: Optional[str] = Field(None, description="Start timestamp")
    completed_at: Optional[str] = Field(None, description="End timestamp")
    duration_ms: Optional[float] = Field(None, description="Duration in ms")


class MapStepResult(BaseModel):
    """Aggregate result from a map step execution."""

    step_id: str = Field(..., description="Map step ID")
    status: str = Field(..., description="Overall status")
    total_items: int = Field(..., ge=0, description="Total items")
    successful_count: int = Field(0, ge=0, description="Success count")
    failed_count: int = Field(0, ge=0, description="Failure count")
    cancelled_count: int = Field(0, ge=0, description="Cancelled count")
    success_ratio: float = Field(0.0, ge=0.0, le=1.0, description="Success ratio")
    child_results: List[ChildStepResult] = Field(default_factory=list, description="Child step results")
    started_at: str = Field(..., description="Start timestamp")
    completed_at: str = Field(..., description="End timestamp")
    total_duration_ms: float = Field(..., description="Total duration")
    provenance_hash: str = Field(..., description="Provenance hash")


class JoinStepResult(BaseModel):
    """Result from a join step execution."""

    step_id: str = Field(..., description="Join step ID")
    status: str = Field(..., description="Execution status")
    aggregated_data: Any = Field(..., description="Aggregated results")
    upstream_summary: Dict[str, Dict[str, int]] = Field(default_factory=dict, description="Upstream step summaries")
    started_at: str = Field(..., description="Start timestamp")
    completed_at: str = Field(..., description="End timestamp")
    duration_ms: float = Field(..., description="Duration in ms")
    provenance_hash: str = Field(..., description="Provenance hash")


# =============================================================================
# Event Emitter
# =============================================================================

@dataclass
class FanoutEvent:
    """Event emitted during fan-out execution for audit trails."""

    event_id: str
    event_type: FanoutEventType
    step_id: str
    child_step_id: Optional[str]
    timestamp: str
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "step_id": self.step_id,
            "child_step_id": self.child_step_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
        }


class FanoutEventEmitter:
    """Emits events during fan-out execution for audit and monitoring."""

    def __init__(self, correlation_id: Optional[str] = None):
        self._handlers: List[Callable[[FanoutEvent], None]] = []
        self._correlation_id = correlation_id or str(uuid.uuid4())

    def register_handler(self, handler: Callable[[FanoutEvent], None]) -> None:
        """Register an event handler."""
        self._handlers.append(handler)

    def emit(
        self,
        event_type: FanoutEventType,
        step_id: str,
        payload: Dict[str, Any],
        child_step_id: Optional[str] = None
    ) -> FanoutEvent:
        """Emit a fan-out event."""
        event = FanoutEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            step_id=step_id,
            child_step_id=child_step_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload=payload,
            correlation_id=self._correlation_id,
        )

        logger.info(
            f"[{event_type.value}] step={step_id} "
            f"child={child_step_id or 'N/A'} "
            f"payload={json.dumps(payload, default=str)[:200]}"
        )

        for handler in self._handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}", exc_info=True)

        return event


# =============================================================================
# Expression Evaluator
# =============================================================================

class ExpressionEvaluator:
    """Evaluates template expressions against execution context."""

    TEMPLATE_PATTERN = re.compile(r"\{\{\s*([^}]+)\s*\}\}")

    def __init__(self, context: Dict[str, Any]):
        self._context = context

    def evaluate(self, expression: str) -> Any:
        """Evaluate an expression against the context."""
        match = self.TEMPLATE_PATTERN.match(expression.strip())
        if not match:
            return self._resolve_path(expression.strip())
        path = match.group(1).strip()
        return self._resolve_path(path)

    def _resolve_path(self, path: str) -> Any:
        """Resolve a dot-notation path in the context."""
        parts = path.split(".")
        current = self._context

        for part in parts:
            array_match = re.match(r"(\w+)\[(\d+)\]", part)
            if array_match:
                key, index = array_match.groups()
                if isinstance(current, dict) and key in current:
                    current = current[key]
                    if isinstance(current, (list, tuple)):
                        current = current[int(index)]
                    else:
                        raise ValueError(f"Cannot index non-list at '{key}' in path '{path}'")
                else:
                    raise ValueError(f"Key '{key}' not found in path '{path}'")
            elif isinstance(current, dict):
                if part in current:
                    current = current[part]
                else:
                    raise ValueError(f"Key '{part}' not found in path '{path}'")
            else:
                raise ValueError(f"Cannot traverse non-dict at '{part}' in path '{path}'")

        return current

    def substitute_item(
        self,
        template: Dict[str, Any],
        item: Any,
        item_var: str,
        index: int
    ) -> Dict[str, Any]:
        """Substitute item into input mapping template."""
        if not template:
            return {}
        item_context = {item_var: item, "index": index, "item": item}
        return self._substitute_recursive(template, item_context)

    def _substitute_recursive(self, obj: Any, item_context: Dict[str, Any]) -> Any:
        """Recursively substitute expressions in nested structure."""
        if isinstance(obj, str):
            match = self.TEMPLATE_PATTERN.search(obj)
            if match:
                path = match.group(1).strip()
                if "." in path:
                    root = path.split(".")[0]
                    if root in item_context:
                        return self._resolve_in_context(path, item_context)
                elif path in item_context:
                    return item_context[path]
                return self._resolve_path(path)
            return obj
        elif isinstance(obj, dict):
            return {k: self._substitute_recursive(v, item_context) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_recursive(v, item_context) for v in obj]
        else:
            return obj

    def _resolve_in_context(self, path: str, context: Dict[str, Any]) -> Any:
        """Resolve path in a specific context."""
        parts = path.split(".")
        current = context
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                raise ValueError(f"Cannot resolve '{part}' in path '{path}'")
        return current


# =============================================================================
# Map Step Executor
# =============================================================================

class MapStepExecutor:
    """Executes map (fan-out) steps with parallel child execution."""

    def __init__(
        self,
        config: MapStepConfig,
        agent_runner: Callable[[str, Dict[str, Any]], Any],
        context: Dict[str, Any],
        event_emitter: Optional[FanoutEventEmitter] = None,
        policy_max_items: Optional[int] = None
    ):
        self._config = config
        self._agent_runner = agent_runner
        self._context = context
        self._event_emitter = event_emitter or FanoutEventEmitter()
        self._policy_max_items = policy_max_items
        self._child_results: Dict[str, ChildStepResult] = {}
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._cancelled = False
        self._cancel_event = asyncio.Event()

    def generate_child_step_id(self, index: int, item: Any) -> Tuple[str, str]:
        """Generate deterministic child step ID."""
        try:
            item_str = json.dumps(item, sort_keys=True, default=str)
        except (TypeError, ValueError):
            item_str = str(item)
        item_hash = hashlib.sha256(item_str.encode("utf-8")).hexdigest()
        hash_prefix = item_hash[:8]
        child_id = f"{self._config.step_id}_{index}_{hash_prefix}"
        return child_id, hash_prefix

    def _calculate_provenance_hash(self, items: List[Any], results: List[ChildStepResult]) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        provenance_data = {
            "step_id": self._config.step_id,
            "items_count": len(items),
            "results_count": len(results),
            "success_count": sum(1 for r in results if r.status == ChildStepStatus.COMPLETED),
            "config": self._config.model_dump(exclude={"input_mapping"}),
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(provenance_str.encode("utf-8")).hexdigest()

    async def execute(self) -> MapStepResult:
        """Execute the map step with fan-out."""
        start_time = datetime.now(timezone.utc)

        evaluator = ExpressionEvaluator(self._context)
        try:
            items = evaluator.evaluate(self._config.items_expression)
        except Exception as e:
            logger.error(f"Failed to evaluate items expression: {e}", exc_info=True)
            raise ValueError(f"Invalid items expression '{self._config.items_expression}': {e}") from e

        if not isinstance(items, (list, tuple)):
            raise ValueError(f"Items expression must resolve to a list, got {type(items).__name__}")

        effective_max_items = min(
            self._config.max_items,
            self._policy_max_items or GLOBAL_MAX_ITEMS_LIMIT,
            GLOBAL_MAX_ITEMS_LIMIT
        )

        if len(items) > effective_max_items:
            raise ValueError(
                f"Items count ({len(items)}) exceeds maximum allowed ({effective_max_items}). "
                f"Reduce batch size or increase limit."
            )

        logger.info(
            f"MapStepExecutor starting fan-out: step={self._config.step_id}, "
            f"items={len(items)}, max_parallel={self._config.max_parallel}"
        )

        self._event_emitter.emit(
            FanoutEventType.FANOUT_STARTED,
            self._config.step_id,
            {
                "item_count": len(items),
                "max_parallel": self._config.max_parallel,
                "failure_policy": self._config.failure_policy.value,
                "agent_id": self._config.agent_id,
            }
        )

        self._semaphore = asyncio.Semaphore(self._config.max_parallel)

        tasks = []
        for index, item in enumerate(items):
            child_id, item_hash = self.generate_child_step_id(index, item)
            task = asyncio.create_task(
                self._execute_child(child_id, index, item, item_hash, evaluator)
            )
            tasks.append((child_id, task))

        if self._config.failure_policy == FailurePolicy.FAIL_FAST:
            await self._execute_fail_fast(tasks)
        else:
            await self._execute_continue_on_error(tasks)

        end_time = datetime.now(timezone.utc)
        child_results = list(self._child_results.values())

        successful = sum(1 for r in child_results if r.status == ChildStepStatus.COMPLETED)
        failed = sum(1 for r in child_results if r.status == ChildStepStatus.FAILED)
        cancelled = sum(1 for r in child_results if r.status == ChildStepStatus.CANCELLED)

        total = len(child_results)
        success_ratio = successful / total if total > 0 else 0.0

        if self._config.failure_policy == FailurePolicy.ALLOW_PARTIAL:
            status = "completed" if success_ratio >= self._config.partial_success_threshold else "failed"
        elif failed > 0 or cancelled > 0:
            status = "failed"
        else:
            status = "completed"

        duration_ms = (end_time - start_time).total_seconds() * 1000
        provenance_hash = self._calculate_provenance_hash(items, child_results)

        result = MapStepResult(
            step_id=self._config.step_id,
            status=status,
            total_items=total,
            successful_count=successful,
            failed_count=failed,
            cancelled_count=cancelled,
            success_ratio=success_ratio,
            child_results=child_results,
            started_at=start_time.isoformat(),
            completed_at=end_time.isoformat(),
            total_duration_ms=duration_ms,
            provenance_hash=provenance_hash,
        )

        self._event_emitter.emit(
            FanoutEventType.FANOUT_COMPLETED,
            self._config.step_id,
            {
                "status": status,
                "total_items": total,
                "successful": successful,
                "failed": failed,
                "cancelled": cancelled,
                "success_ratio": success_ratio,
                "duration_ms": duration_ms,
                "provenance_hash": provenance_hash,
            }
        )

        logger.info(
            f"MapStepExecutor completed: step={self._config.step_id}, "
            f"status={status}, success={successful}/{total}, duration={duration_ms:.2f}ms"
        )

        return result

    async def _execute_fail_fast(self, tasks: List[Tuple[str, asyncio.Task]]) -> None:
        """Execute with fail-fast policy."""
        pending = {task: child_id for child_id, task in tasks}
        try:
            while pending and not self._cancelled:
                done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    child_id = pending.pop(task)
                    try:
                        await task
                    except Exception as e:
                        logger.error(f"Child step {child_id} failed: {e}", exc_info=True)
                        if self._config.failure_policy == FailurePolicy.FAIL_FAST:
                            await self.cancel()
                            break
        except asyncio.CancelledError:
            logger.warning(f"Map step {self._config.step_id} was cancelled")
            await self.cancel()

    async def _execute_continue_on_error(self, tasks: List[Tuple[str, asyncio.Task]]) -> None:
        """Execute with continue-on-error policy."""
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        for (child_id, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(f"Child step {child_id} raised exception: {result}")

    async def _execute_child(
        self,
        child_id: str,
        index: int,
        item: Any,
        item_hash: str,
        evaluator: ExpressionEvaluator
    ) -> ChildStepResult:
        """Execute a single child step."""
        result = ChildStepResult(
            child_step_id=child_id,
            index=index,
            item=item,
            item_hash=item_hash,
            status=ChildStepStatus.PENDING,
        )
        self._child_results[child_id] = result

        if self._cancelled:
            result.status = ChildStepStatus.CANCELLED
            return result

        async with self._semaphore:
            if self._cancelled:
                result.status = ChildStepStatus.CANCELLED
                return result

            start_time = datetime.now(timezone.utc)
            result.started_at = start_time.isoformat()
            result.status = ChildStepStatus.RUNNING

            self._event_emitter.emit(
                FanoutEventType.FANOUT_ITEM_STARTED,
                self._config.step_id,
                {"index": index, "item_hash": item_hash},
                child_step_id=child_id
            )

            try:
                child_input = evaluator.substitute_item(
                    self._config.input_mapping or {},
                    item,
                    self._config.item_var,
                    index
                )
                child_input[self._config.item_var] = item
                child_input["_index"] = index
                child_input["_child_step_id"] = child_id

                if asyncio.iscoroutinefunction(self._agent_runner):
                    agent_result = await self._agent_runner(self._config.agent_id, child_input)
                else:
                    loop = asyncio.get_event_loop()
                    agent_result = await loop.run_in_executor(
                        None, self._agent_runner, self._config.agent_id, child_input
                    )

                end_time = datetime.now(timezone.utc)
                result.completed_at = end_time.isoformat()
                result.duration_ms = (end_time - start_time).total_seconds() * 1000

                if isinstance(agent_result, dict):
                    success = agent_result.get("success", True)
                    result.result = agent_result
                else:
                    success = getattr(agent_result, "success", True)
                    result.result = (
                        agent_result.model_dump()
                        if hasattr(agent_result, "model_dump")
                        else {"data": agent_result}
                    )

                result.status = ChildStepStatus.COMPLETED if success else ChildStepStatus.FAILED
                if not success:
                    result.error = agent_result.get("error", "Unknown error")

                self._event_emitter.emit(
                    FanoutEventType.FANOUT_ITEM_COMPLETED,
                    self._config.step_id,
                    {"index": index, "status": result.status.value, "duration_ms": result.duration_ms},
                    child_step_id=child_id
                )

            except asyncio.CancelledError:
                result.status = ChildStepStatus.CANCELLED
                result.completed_at = datetime.now(timezone.utc).isoformat()
                raise

            except Exception as e:
                end_time = datetime.now(timezone.utc)
                result.completed_at = end_time.isoformat()
                result.duration_ms = (end_time - start_time).total_seconds() * 1000
                result.status = ChildStepStatus.FAILED
                result.error = str(e)

                logger.error(f"Child step {child_id} failed: {e}", exc_info=True)

                self._event_emitter.emit(
                    FanoutEventType.FANOUT_ITEM_FAILED,
                    self._config.step_id,
                    {"index": index, "error": str(e), "duration_ms": result.duration_ms},
                    child_step_id=child_id
                )

            return result

    async def cancel(self) -> None:
        """Cancel all pending and running child steps."""
        self._cancelled = True
        self._cancel_event.set()

        self._event_emitter.emit(
            FanoutEventType.FANOUT_CANCELLED,
            self._config.step_id,
            {
                "completed_count": sum(
                    1 for r in self._child_results.values() if r.status == ChildStepStatus.COMPLETED
                ),
                "pending_count": sum(
                    1 for r in self._child_results.values()
                    if r.status in (ChildStepStatus.PENDING, ChildStepStatus.RUNNING)
                ),
            }
        )

        for result in self._child_results.values():
            if result.status in (ChildStepStatus.PENDING, ChildStepStatus.RUNNING):
                result.status = ChildStepStatus.CANCELLED
                result.completed_at = datetime.now(timezone.utc).isoformat()

        logger.info(f"Map step {self._config.step_id} cancelled")

    @property
    def is_cancelled(self) -> bool:
        """Check if executor has been cancelled."""
        return self._cancelled

    def get_child_status(self) -> Dict[str, ChildStepStatus]:
        """Get status of all child steps."""
        return {child_id: result.status for child_id, result in self._child_results.items()}


# =============================================================================
# Join Step Executor
# =============================================================================

class JoinStepExecutor:
    """Executes join steps that aggregate results from upstream map steps."""

    def __init__(
        self,
        config: JoinStepConfig,
        upstream_results: Dict[str, MapStepResult],
        agent_runner: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        event_emitter: Optional[FanoutEventEmitter] = None
    ):
        self._config = config
        self._upstream_results = upstream_results
        self._agent_runner = agent_runner
        self._event_emitter = event_emitter or FanoutEventEmitter()

    def _calculate_provenance_hash(
        self,
        aggregated_data: Any,
        upstream_summary: Dict[str, Dict[str, int]]
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        provenance_data = {
            "step_id": self._config.step_id,
            "upstream_step_ids": self._config.upstream_step_ids,
            "aggregator": self._config.aggregator.value,
            "upstream_summary": upstream_summary,
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(provenance_str.encode("utf-8")).hexdigest()

    async def execute(self) -> JoinStepResult:
        """Execute the join step."""
        start_time = datetime.now(timezone.utc)

        logger.info(
            f"JoinStepExecutor starting: step={self._config.step_id}, "
            f"upstream={self._config.upstream_step_ids}, aggregator={self._config.aggregator.value}"
        )

        self._event_emitter.emit(
            FanoutEventType.JOIN_STARTED,
            self._config.step_id,
            {"upstream_step_ids": self._config.upstream_step_ids, "aggregator": self._config.aggregator.value}
        )

        missing = [step_id for step_id in self._config.upstream_step_ids if step_id not in self._upstream_results]
        if missing:
            raise ValueError(f"Missing upstream results for join: {missing}")

        upstream_summary = {}
        all_child_results: List[ChildStepResult] = []

        for step_id in self._config.upstream_step_ids:
            result = self._upstream_results[step_id]
            upstream_summary[step_id] = {
                "total": result.total_items,
                "successful": result.successful_count,
                "failed": result.failed_count,
                "cancelled": result.cancelled_count,
            }
            all_child_results.extend(result.child_results)

        aggregated_data = await self._aggregate(all_child_results)

        end_time = datetime.now(timezone.utc)
        duration_ms = (end_time - start_time).total_seconds() * 1000
        provenance_hash = self._calculate_provenance_hash(aggregated_data, upstream_summary)

        all_success = all(result.status == "completed" for result in self._upstream_results.values())
        status = "completed" if all_success else "partial"

        result = JoinStepResult(
            step_id=self._config.step_id,
            status=status,
            aggregated_data=aggregated_data,
            upstream_summary=upstream_summary,
            started_at=start_time.isoformat(),
            completed_at=end_time.isoformat(),
            duration_ms=duration_ms,
            provenance_hash=provenance_hash,
        )

        self._event_emitter.emit(
            FanoutEventType.JOIN_COMPLETED,
            self._config.step_id,
            {
                "status": status,
                "duration_ms": duration_ms,
                "upstream_summary": upstream_summary,
                "provenance_hash": provenance_hash,
            }
        )

        logger.info(
            f"JoinStepExecutor completed: step={self._config.step_id}, "
            f"status={status}, duration={duration_ms:.2f}ms"
        )

        return result

    async def _aggregate(self, child_results: List[ChildStepResult]) -> Any:
        """Aggregate child results based on strategy."""
        successful_results = [
            r for r in child_results if r.status == ChildStepStatus.COMPLETED and r.result
        ]
        successful_results.sort(key=lambda r: r.index)

        if self._config.aggregator == AggregatorType.LIST:
            return [r.result for r in successful_results]
        elif self._config.aggregator == AggregatorType.MERGE:
            merged = {}
            for r in successful_results:
                if isinstance(r.result, dict):
                    merged.update(r.result)
            return merged
        elif self._config.aggregator == AggregatorType.FIRST:
            return successful_results[0].result if successful_results else None
        elif self._config.aggregator == AggregatorType.LAST:
            return successful_results[-1].result if successful_results else None
        elif self._config.aggregator == AggregatorType.CUSTOM:
            if not self._agent_runner or not self._config.agent_id:
                raise ValueError("Custom aggregation requires agent_id and agent_runner")
            agg_input = {
                "child_results": [r.model_dump() for r in successful_results],
                "failed_count": len(child_results) - len(successful_results),
            }
            if asyncio.iscoroutinefunction(self._agent_runner):
                return await self._agent_runner(self._config.agent_id, agg_input)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._agent_runner, self._config.agent_id, agg_input)
        else:
            raise ValueError(f"Unknown aggregator type: {self._config.aggregator}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_map_step_executor(
    step_config: Dict[str, Any],
    agent_runner: Callable[[str, Dict[str, Any]], Any],
    context: Dict[str, Any],
    event_emitter: Optional[FanoutEventEmitter] = None,
    policy_max_items: Optional[int] = None
) -> MapStepExecutor:
    """Create a MapStepExecutor from step configuration dictionary."""
    config = MapStepConfig(
        step_id=step_config.get("name") or step_config.get("id"),
        items_expression=step_config.get("items"),
        item_var=step_config.get("itemVar", DEFAULT_ITEM_VAR),
        max_parallel=step_config.get("maxParallel", DEFAULT_MAX_PARALLEL),
        max_items=step_config.get("maxItems", DEFAULT_MAX_ITEMS),
        failure_policy=FailurePolicy(step_config.get("failurePolicy", FailurePolicy.FAIL_FAST.value)),
        partial_success_threshold=step_config.get("partialSuccessThreshold", DEFAULT_PARTIAL_SUCCESS_THRESHOLD),
        agent_id=step_config.get("agent"),
        input_mapping=step_config.get("in"),
        timeout=step_config.get("timeout"),
    )
    return MapStepExecutor(
        config=config,
        agent_runner=agent_runner,
        context=context,
        event_emitter=event_emitter,
        policy_max_items=policy_max_items,
    )


def create_join_step_executor(
    step_config: Dict[str, Any],
    upstream_results: Dict[str, MapStepResult],
    agent_runner: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    event_emitter: Optional[FanoutEventEmitter] = None
) -> JoinStepExecutor:
    """Create a JoinStepExecutor from step configuration dictionary."""
    config = JoinStepConfig(
        step_id=step_config.get("name") or step_config.get("id"),
        upstream_step_ids=step_config.get("depends_on", []),
        aggregator=AggregatorType(step_config.get("aggregator", AggregatorType.LIST.value)),
        agent_id=step_config.get("agent"),
        timeout=step_config.get("timeout"),
    )
    return JoinStepExecutor(
        config=config,
        upstream_results=upstream_results,
        agent_runner=agent_runner,
        event_emitter=event_emitter,
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "StepType", "FailurePolicy", "AggregatorType", "ChildStepStatus", "FanoutEventType",
    "MapStepConfig", "JoinStepConfig", "ChildStepResult", "MapStepResult", "JoinStepResult",
    "FanoutEvent", "FanoutEventEmitter", "ExpressionEvaluator",
    "MapStepExecutor", "JoinStepExecutor",
    "create_map_step_executor", "create_join_step_executor",
    "DEFAULT_MAX_PARALLEL", "DEFAULT_MAX_ITEMS", "DEFAULT_ITEM_VAR",
    "DEFAULT_PARTIAL_SUCCESS_THRESHOLD", "GLOBAL_MAX_ITEMS_LIMIT", "GLOBAL_MAX_PARALLEL_LIMIT",
]
