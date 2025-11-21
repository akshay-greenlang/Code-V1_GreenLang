# -*- coding: utf-8 -*-
"""
Pipeline - Agent pipeline orchestration for sequential, parallel, and conditional execution.

This module implements high-performance pipeline patterns for orchestrating
multi-agent workflows with support for sequential, parallel, and conditional
execution paths.

Example:
    >>> # Create sequential pipeline
    >>> pipeline = Pipeline([agent1, agent2, agent3])
    >>> result = await pipeline.execute(input_data)
    >>>
    >>> # Create parallel pipeline
    >>> parallel_stage = PipelineStage(
    ...     agents=[agent1, agent2],
    ...     mode=ExecutionMode.PARALLEL
    ... )
    >>> pipeline = Pipeline([parallel_stage])
    >>> results = await pipeline.execute(input_data)
"""

from typing import Dict, List, Optional, Any, Callable, Union, AsyncIterator
from pydantic import BaseModel, Field, validator
from enum import Enum
import asyncio
from simpleeval import simple_eval
import logging
from datetime import datetime, timezone
import uuid
from dataclasses import dataclass, field
import json
import hashlib

from prometheus_client import Counter, Histogram, Gauge
import networkx as nx

from .message_bus import MessageBus, Message, MessageType, Priority
from greenlang.determinism import deterministic_uuid, DeterministicClock

logger = logging.getLogger(__name__)

# Metrics
pipeline_execution_counter = Counter('pipeline_executions_total', 'Total pipeline executions', ['status'])
stage_execution_histogram = Histogram('pipeline_stage_duration_ms', 'Stage execution duration', ['stage_name'])
pipeline_throughput_gauge = Gauge('pipeline_throughput', 'Pipeline throughput (items/sec)')
pipeline_error_counter = Counter('pipeline_errors_total', 'Pipeline errors', ['stage', 'error_type'])


class ExecutionMode(str, Enum):
    """Pipeline execution modes."""
    SEQUENTIAL = "SEQUENTIAL"
    PARALLEL = "PARALLEL"
    CONDITIONAL = "CONDITIONAL"
    LOOP = "LOOP"
    MAP_REDUCE = "MAP_REDUCE"


class ConditionOperator(str, Enum):
    """Conditional operators for pipeline branching."""
    EQUALS = "EQUALS"
    NOT_EQUALS = "NOT_EQUALS"
    GREATER_THAN = "GREATER_THAN"
    LESS_THAN = "LESS_THAN"
    CONTAINS = "CONTAINS"
    MATCHES_REGEX = "MATCHES_REGEX"
    IN_LIST = "IN_LIST"
    CUSTOM = "CUSTOM"


class PipelineCondition(BaseModel):
    """Condition for conditional pipeline execution."""

    field: str = Field(..., description="Field to evaluate")
    operator: ConditionOperator = Field(..., description="Comparison operator")
    value: Any = Field(..., description="Value to compare against")
    custom_evaluator: Optional[str] = Field(None, description="Custom evaluator function name")

    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate condition against data."""
        field_value = data.get(self.field)

        if self.operator == ConditionOperator.EQUALS:
            return field_value == self.value
        elif self.operator == ConditionOperator.NOT_EQUALS:
            return field_value != self.value
        elif self.operator == ConditionOperator.GREATER_THAN:
            return field_value > self.value
        elif self.operator == ConditionOperator.LESS_THAN:
            return field_value < self.value
        elif self.operator == ConditionOperator.CONTAINS:
            return self.value in str(field_value)
        elif self.operator == ConditionOperator.IN_LIST:
            return field_value in self.value
        elif self.operator == ConditionOperator.MATCHES_REGEX:
            import re
            return bool(re.match(self.value, str(field_value)))
        elif self.operator == ConditionOperator.CUSTOM:
            # Custom evaluation logic would go here
            return True
        else:
            return False


class StageResult(BaseModel):
    """Result from pipeline stage execution."""

    stage_id: str = Field(..., description="Stage identifier")
    stage_name: str = Field(..., description="Stage name")
    status: str = Field(..., description="Execution status")
    output: Any = Field(..., description="Stage output data")
    errors: List[str] = Field(default_factory=list, description="Errors if any")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")


class PipelineStage(BaseModel):
    """Individual pipeline stage definition."""

    stage_id: str = Field(default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))))
    name: str = Field(..., description="Stage name")
    agents: List[str] = Field(..., description="Agent IDs for this stage")
    mode: ExecutionMode = Field(default=ExecutionMode.SEQUENTIAL)
    conditions: List[PipelineCondition] = Field(default_factory=list)
    retry_policy: Dict[str, Any] = Field(default_factory=lambda: {"max_attempts": 3, "backoff_ms": 1000})
    timeout_ms: int = Field(default=30000, ge=0)
    transform: Optional[str] = Field(None, description="Data transformation function")
    skip_on_error: bool = Field(default=False, description="Skip stage on error")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Pipeline configuration."""

    pipeline_id: str = Field(default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))))
    name: str = Field(..., description="Pipeline name")
    description: Optional[str] = Field(None, description="Pipeline description")
    stages: List[PipelineStage] = Field(..., description="Pipeline stages")
    global_timeout_ms: int = Field(default=300000, ge=0, description="Overall pipeline timeout")
    enable_provenance: bool = Field(default=True, description="Enable provenance tracking")
    enable_caching: bool = Field(default=False, description="Enable result caching")
    error_handling: str = Field(default="FAIL_FAST", description="FAIL_FAST or CONTINUE")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PipelineExecution(BaseModel):
    """Pipeline execution tracking."""

    execution_id: str = Field(default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))))
    pipeline_id: str = Field(...)
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = Field(None)
    status: str = Field(default="RUNNING")
    current_stage: Optional[str] = Field(None)
    stage_results: List[StageResult] = Field(default_factory=list)
    final_output: Optional[Any] = Field(None)
    errors: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    provenance_chain: List[str] = Field(default_factory=list)


class Pipeline:
    """
    Agent pipeline orchestration system.

    Supports sequential, parallel, and conditional execution patterns
    with complete provenance tracking and error recovery.
    """

    def __init__(
        self,
        stages: Union[List[PipelineStage], List[Any]],
        message_bus: Optional[MessageBus] = None,
        config: Optional[PipelineConfig] = None
    ):
        """
        Initialize pipeline.

        Args:
            stages: List of pipeline stages or agents
            message_bus: Optional message bus for communication
            config: Pipeline configuration
        """
        # Convert agent list to stages if needed
        if stages and not isinstance(stages[0], PipelineStage):
            stages = [
                PipelineStage(
                    name=f"stage_{i}",
                    agents=[str(agent)],
                    mode=ExecutionMode.SEQUENTIAL
                )
                for i, agent in enumerate(stages)
            ]

        self.config = config or PipelineConfig(
            name="default_pipeline",
            stages=stages
        )
        self.message_bus = message_bus

        # Execution tracking
        self.executions: Dict[str, PipelineExecution] = {}

        # Stage handlers and transforms
        self.stage_handlers: Dict[str, Callable] = {}
        self.transforms: Dict[str, Callable] = {}

        # Build execution graph
        self._build_execution_graph()

    def _build_execution_graph(self) -> None:
        """Build directed acyclic graph for pipeline execution."""
        self.graph = nx.DiGraph()

        for i, stage in enumerate(self.config.stages):
            self.graph.add_node(stage.stage_id, stage=stage)

            # Add edge from previous stage (sequential default)
            if i > 0 and stage.mode != ExecutionMode.PARALLEL:
                prev_stage = self.config.stages[i - 1]
                self.graph.add_edge(prev_stage.stage_id, stage.stage_id)

    async def execute(
        self,
        input_data: Any,
        execution_id: Optional[str] = None
    ) -> Any:
        """
        Execute pipeline with input data.

        Args:
            input_data: Initial input data
            execution_id: Optional execution ID for tracking

        Returns:
            Pipeline output

        Raises:
            TimeoutError: If pipeline timeout exceeded
            RuntimeError: If pipeline execution fails
        """
        execution = PipelineExecution(
            execution_id=execution_id or str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
            pipeline_id=self.config.pipeline_id
        )
        self.executions[execution.execution_id] = execution

        start_time = datetime.now(timezone.utc)

        try:
            logger.info(f"Starting pipeline {self.config.name} (execution: {execution.execution_id})")

            # Execute with global timeout
            result = await asyncio.wait_for(
                self._execute_pipeline(execution, input_data),
                timeout=self.config.global_timeout_ms / 1000
            )

            execution.status = "COMPLETED"
            execution.final_output = result
            pipeline_execution_counter.labels(status="success").inc()

            return result

        except asyncio.TimeoutError:
            execution.status = "TIMEOUT"
            execution.errors.append(f"Pipeline timeout after {self.config.global_timeout_ms}ms")
            pipeline_execution_counter.labels(status="timeout").inc()
            raise

        except Exception as e:
            execution.status = "FAILED"
            execution.errors.append(str(e))
            pipeline_execution_counter.labels(status="failure").inc()
            logger.error(f"Pipeline {self.config.name} failed: {e}")
            raise

        finally:
            execution.completed_at = datetime.now(timezone.utc).isoformat()
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            execution.metrics["total_duration_ms"] = duration_ms

            # Calculate throughput
            if duration_ms > 0:
                throughput = 1000 / duration_ms  # items/sec
                pipeline_throughput_gauge.set(throughput)

            logger.info(f"Pipeline {self.config.name} completed in {duration_ms:.2f}ms")

    async def _execute_pipeline(
        self,
        execution: PipelineExecution,
        input_data: Any
    ) -> Any:
        """Execute pipeline stages."""
        current_data = input_data
        provenance_parent = None

        # Execute stages in topological order
        for stage_id in nx.topological_sort(self.graph):
            stage = self.graph.nodes[stage_id]["stage"]
            execution.current_stage = stage.name

            # Check conditions
            if stage.conditions:
                should_execute = all(
                    condition.evaluate(current_data if isinstance(current_data, dict) else {})
                    for condition in stage.conditions
                )
                if not should_execute:
                    logger.info(f"Skipping stage {stage.name} due to conditions")
                    continue

            # Execute stage based on mode
            try:
                stage_result = await self._execute_stage(
                    stage,
                    current_data,
                    provenance_parent
                )

                execution.stage_results.append(stage_result)
                current_data = stage_result.output
                provenance_parent = stage_result.provenance_hash

                if self.config.enable_provenance:
                    execution.provenance_chain.append(stage_result.provenance_hash)

            except Exception as e:
                if stage.skip_on_error:
                    logger.warning(f"Stage {stage.name} failed but skipping: {e}")
                    continue
                elif self.config.error_handling == "CONTINUE":
                    logger.error(f"Stage {stage.name} failed but continuing: {e}")
                    execution.errors.append(f"Stage {stage.name}: {str(e)}")
                    continue
                else:
                    raise

        return current_data

    async def _execute_stage(
        self,
        stage: PipelineStage,
        input_data: Any,
        provenance_parent: Optional[str] = None
    ) -> StageResult:
        """Execute individual pipeline stage."""
        start_time = datetime.now(timezone.utc)
        stage_errors = []
        output = None

        try:
            logger.debug(f"Executing stage {stage.name} with mode {stage.mode}")

            # Apply input transformation if specified
            if stage.transform and stage.transform in self.transforms:
                input_data = await self.transforms[stage.transform](input_data)

            # Execute based on mode
            if stage.mode == ExecutionMode.SEQUENTIAL:
                output = await self._execute_sequential(stage, input_data)
            elif stage.mode == ExecutionMode.PARALLEL:
                output = await self._execute_parallel(stage, input_data)
            elif stage.mode == ExecutionMode.MAP_REDUCE:
                output = await self._execute_map_reduce(stage, input_data)
            elif stage.mode == ExecutionMode.LOOP:
                output = await self._execute_loop(stage, input_data)
            elif stage.mode == ExecutionMode.CONDITIONAL:
                output = await self._execute_conditional(stage, input_data)
            else:
                raise ValueError(f"Unsupported execution mode: {stage.mode}")

            status = "SUCCESS"

        except Exception as e:
            logger.error(f"Stage {stage.name} failed: {e}")
            stage_errors.append(str(e))
            pipeline_error_counter.labels(stage=stage.name, error_type=type(e).__name__).inc()
            status = "FAILED"
            output = input_data  # Pass through on error

            # Retry if configured
            if stage.retry_policy.get("max_attempts", 0) > 1:
                output = await self._retry_stage(stage, input_data, stage.retry_policy)
                if output is not None:
                    status = "SUCCESS"
                    stage_errors.clear()

        finally:
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            stage_execution_histogram.labels(stage_name=stage.name).observe(duration_ms)

            # Calculate provenance hash
            provenance_data = json.dumps({
                "stage": stage.name,
                "input": str(input_data)[:1000],  # Truncate for hashing
                "output": str(output)[:1000],
                "parent": provenance_parent
            }, sort_keys=True)
            provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

            return StageResult(
                stage_id=stage.stage_id,
                stage_name=stage.name,
                status=status,
                output=output,
                errors=stage_errors,
                metrics={"duration_ms": duration_ms},
                provenance_hash=provenance_hash
            )

    async def _execute_sequential(
        self,
        stage: PipelineStage,
        input_data: Any
    ) -> Any:
        """Execute agents sequentially."""
        current_data = input_data

        for agent_id in stage.agents:
            if self.message_bus:
                # Use message bus for communication
                message = Message(
                    sender_id=f"pipeline-{self.config.pipeline_id}",
                    recipient_id=agent_id,
                    message_type=MessageType.REQUEST,
                    priority=Priority.HIGH,
                    payload={"data": current_data, "stage": stage.name}
                )

                response = await self.message_bus.request_response(
                    message,
                    timeout_ms=stage.timeout_ms
                )

                if response:
                    current_data = response.payload.get("output", current_data)
                else:
                    raise TimeoutError(f"Agent {agent_id} timeout")
            else:
                # Direct execution if handler registered
                if agent_id in self.stage_handlers:
                    current_data = await self.stage_handlers[agent_id](current_data)
                else:
                    logger.warning(f"No handler for agent {agent_id}")

        return current_data

    async def _execute_parallel(
        self,
        stage: PipelineStage,
        input_data: Any
    ) -> List[Any]:
        """Execute agents in parallel."""
        tasks = []

        for agent_id in stage.agents:
            if self.message_bus:
                message = Message(
                    sender_id=f"pipeline-{self.config.pipeline_id}",
                    recipient_id=agent_id,
                    message_type=MessageType.REQUEST,
                    priority=Priority.HIGH,
                    payload={"data": input_data, "stage": stage.name}
                )
                tasks.append(
                    self.message_bus.request_response(message, timeout_ms=stage.timeout_ms)
                )
            else:
                if agent_id in self.stage_handlers:
                    tasks.append(self.stage_handlers[agent_id](input_data))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        outputs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Parallel execution error for agent {stage.agents[i]}: {result}")
                if not stage.skip_on_error:
                    raise result
            elif isinstance(result, Message):
                outputs.append(result.payload.get("output"))
            else:
                outputs.append(result)

        return outputs

    async def _execute_map_reduce(
        self,
        stage: PipelineStage,
        input_data: List[Any]
    ) -> Any:
        """Execute map-reduce pattern."""
        if not isinstance(input_data, list):
            input_data = [input_data]

        # Map phase - process each item in parallel
        map_results = await self._execute_parallel(stage, input_data)

        # Reduce phase - aggregate results
        if "reducer" in stage.metadata:
            reducer = stage.metadata["reducer"]
            if callable(reducer):
                return reducer(map_results)
            elif reducer == "sum":
                return sum(map_results)
            elif reducer == "concat":
                return map_results
            elif reducer == "merge":
                merged = {}
                for result in map_results:
                    if isinstance(result, dict):
                        merged.update(result)
                return merged

        return map_results

    async def _execute_loop(
        self,
        stage: PipelineStage,
        input_data: Any
    ) -> Any:
        """Execute stage in a loop."""
        max_iterations = stage.metadata.get("max_iterations", 10)
        condition_func = stage.metadata.get("condition")

        current_data = input_data
        iteration = 0

        while iteration < max_iterations:
            # Check loop condition
            if condition_func and not condition_func(current_data, iteration):
                break

            # Execute stage
            current_data = await self._execute_sequential(stage, current_data)
            iteration += 1

        return current_data

    async def _execute_conditional(
        self,
        stage: PipelineStage,
        input_data: Any
    ) -> Any:
        """Execute conditional branching."""
        branches = stage.metadata.get("branches", {})

        for condition, branch_agents in branches.items():
            if self._evaluate_condition(condition, input_data):
                # Execute branch
                branch_stage = PipelineStage(
                    name=f"{stage.name}_branch",
                    agents=branch_agents,
                    mode=ExecutionMode.SEQUENTIAL
                )
                return await self._execute_sequential(branch_stage, input_data)

        # Default branch
        if "default" in branches:
            branch_stage = PipelineStage(
                name=f"{stage.name}_default",
                agents=branches["default"],
                mode=ExecutionMode.SEQUENTIAL
            )
            return await self._execute_sequential(branch_stage, input_data)

        return input_data

    async def _retry_stage(
        self,
        stage: PipelineStage,
        input_data: Any,
        retry_policy: Dict[str, Any]
    ) -> Optional[Any]:
        """Retry failed stage execution."""
        max_attempts = retry_policy.get("max_attempts", 3)
        backoff_ms = retry_policy.get("backoff_ms", 1000)
        backoff_multiplier = retry_policy.get("backoff_multiplier", 2)

        for attempt in range(1, max_attempts):
            await asyncio.sleep(backoff_ms / 1000)

            try:
                logger.info(f"Retrying stage {stage.name} (attempt {attempt + 1}/{max_attempts})")
                return await self._execute_sequential(stage, input_data)
            except Exception as e:
                logger.warning(f"Retry {attempt} failed: {e}")
                backoff_ms *= backoff_multiplier

        return None

    def _evaluate_condition(self, condition: str, data: Any) -> bool:
        """Evaluate condition string."""
        try:
            # Safe evaluation context
            context = {
                "data": data,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool
            }
            return simple_eval(condition, names=context)  # SECURITY FIX
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False

    def register_handler(self, agent_id: str, handler: Callable) -> None:
        """Register agent handler for direct execution."""
        self.stage_handlers[agent_id] = handler

    def register_transform(self, name: str, transform: Callable) -> None:
        """Register data transformation function."""
        self.transforms[name] = transform

    async def get_execution_status(self, execution_id: str) -> Optional[PipelineExecution]:
        """Get pipeline execution status."""
        return self.executions.get(execution_id)

    async def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        completed = [e for e in self.executions.values() if e.status == "COMPLETED"]
        failed = [e for e in self.executions.values() if e.status == "FAILED"]

        avg_duration = 0
        if completed:
            durations = [e.metrics.get("total_duration_ms", 0) for e in completed]
            avg_duration = sum(durations) / len(durations)

        return {
            "total_executions": len(self.executions),
            "completed": len(completed),
            "failed": len(failed),
            "success_rate": len(completed) / len(self.executions) if self.executions else 0,
            "average_duration_ms": avg_duration,
            "stages": len(self.config.stages)
        }

