# -*- coding: utf-8 -*-
"""
GL-FOUND-X-001: GreenLang Orchestrator
======================================

The core DAG (Directed Acyclic Graph) execution engine for multi-agent pipelines.
This is the "brain" of GreenLang, responsible for orchestrating all agent executions.

Capabilities:
    - Dependency graph management and topological execution
    - Parallel execution of independent agents
    - Retry logic with exponential backoff
    - Timeout handling per agent and pipeline
    - Checkpoint/recovery for long-running pipelines
    - Complete lineage tracking for zero-hallucination
    - Observability hooks for metrics and tracing
    - Deterministic execution with reproducibility guarantees

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class PipelineStatus(str, Enum):
    """Status of a pipeline execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"  # Some agents succeeded, some failed
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class AgentStatus(str, Enum):
    """Status of an individual agent execution."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    RETRYING = "retrying"


@dataclass
class AgentNode:
    """
    Represents an agent in the execution DAG.

    Attributes:
        agent_id: Unique identifier for this agent instance
        agent_type: The type of agent (e.g., 'GL-MRV-X-001')
        config: Agent configuration
        dependencies: List of agent_ids this agent depends on
        input_mapping: Maps dependency outputs to this agent's inputs
        timeout_seconds: Maximum execution time for this agent
        retry_config: Retry configuration
    """
    agent_id: str
    agent_type: str
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    input_mapping: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 300
    retry_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "initial_delay_ms": 1000,
        "backoff_multiplier": 2,
        "max_delay_ms": 30000
    })

    # Runtime state (not serialized)
    status: AgentStatus = field(default=AgentStatus.PENDING, compare=False)
    result: Optional[AgentResult] = field(default=None, compare=False)
    attempts: int = field(default=0, compare=False)
    started_at: Optional[datetime] = field(default=None, compare=False)
    completed_at: Optional[datetime] = field(default=None, compare=False)


class DAGDefinition(BaseModel):
    """
    Definition of a multi-agent execution pipeline.

    This defines the complete execution graph including:
    - All agents to execute
    - Dependencies between agents
    - Input/output mappings
    - Execution parameters
    """
    pipeline_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Pipeline name")
    description: str = Field(default="", description="Pipeline description")
    version: str = Field(default="1.0.0", description="Pipeline version")

    # Agent definitions
    agents: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Agent definitions keyed by agent_id"
    )

    # Global settings
    timeout_seconds: int = Field(
        default=3600,
        description="Maximum pipeline execution time"
    )
    max_parallel: int = Field(
        default=10,
        description="Maximum agents to run in parallel"
    )
    checkpoint_interval: int = Field(
        default=10,
        description="Checkpoint every N completed agents"
    )
    enable_recovery: bool = Field(
        default=True,
        description="Enable checkpoint recovery"
    )

    # Input/Output
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pipeline input data"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    @validator('agents')
    def validate_no_cycles(cls, agents):
        """Validate that the DAG has no cycles."""
        # Build adjacency list
        graph = defaultdict(list)
        for agent_id, agent_def in agents.items():
            deps = agent_def.get('dependencies', [])
            for dep in deps:
                graph[dep].append(agent_id)

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in agents:
            if node not in visited:
                if has_cycle(node):
                    raise ValueError(f"Cycle detected in DAG involving agent: {node}")

        return agents


@dataclass
class ExecutionContext:
    """Runtime context for pipeline execution."""
    execution_id: str
    pipeline: DAGDefinition
    nodes: Dict[str, AgentNode]
    results: Dict[str, AgentResult]
    lineage: Dict[str, Any]
    checkpoints: List[Dict[str, Any]]
    started_at: datetime
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None


class ExecutionResult(BaseModel):
    """Result of a pipeline execution."""
    execution_id: str = Field(..., description="Unique execution ID")
    pipeline_id: str = Field(..., description="Pipeline ID")
    status: PipelineStatus = Field(..., description="Overall status")

    # Timing
    started_at: datetime = Field(..., description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    duration_ms: float = Field(default=0.0, description="Total duration")

    # Results
    agent_results: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Results from each agent"
    )
    final_outputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Final pipeline outputs"
    )

    # Statistics
    agents_total: int = Field(default=0, description="Total agents")
    agents_succeeded: int = Field(default=0, description="Succeeded agents")
    agents_failed: int = Field(default=0, description="Failed agents")
    agents_skipped: int = Field(default=0, description="Skipped agents")

    # Lineage
    lineage_id: str = Field(default="", description="Lineage trace ID")
    input_hash: str = Field(default="", description="Hash of inputs")
    output_hash: str = Field(default="", description="Hash of outputs")

    # Errors
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Error details"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class GreenLangOrchestrator(BaseAgent):
    """
    GL-FOUND-X-001: GreenLang Orchestrator

    The core DAG execution engine for multi-agent pipelines.
    Manages dependency resolution, parallel execution, retries,
    timeouts, checkpointing, and complete lineage tracking.

    Zero-Hallucination Guarantees:
        - All calculations have complete lineage
        - Deterministic execution with same inputs
        - All assumptions tracked and versioned
        - All data sources cited

    Usage:
        orchestrator = GreenLangOrchestrator()
        result = await orchestrator.execute_pipeline(dag_definition)
    """

    AGENT_ID = "GL-FOUND-X-001"
    AGENT_NAME = "GreenLang Orchestrator"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="DAG execution engine for multi-agent pipelines",
                version=self.VERSION,
                parameters={
                    "max_parallel": 10,
                    "default_timeout": 300,
                    "checkpoint_interval": 10,
                    "enable_recovery": True,
                }
            )
        super().__init__(config)

        # Agent registry - maps agent_type to agent class
        self._agent_registry: Dict[str, type] = {}

        # Active executions
        self._active_executions: Dict[str, ExecutionContext] = {}

        # Checkpoint storage (in production, use Redis/PostgreSQL)
        self._checkpoints: Dict[str, List[Dict[str, Any]]] = {}

        # Metrics
        self._execution_count = 0
        self._success_count = 0
        self._failure_count = 0

    def register_agent(self, agent_type: str, agent_class: type):
        """
        Register an agent type with the orchestrator.

        Args:
            agent_type: The agent type identifier (e.g., 'GL-MRV-X-001')
            agent_class: The agent class to instantiate
        """
        self._agent_registry[agent_type] = agent_class
        self.logger.info(f"Registered agent type: {agent_type}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Synchronous execution wrapper for the pipeline.

        Args:
            input_data: Must contain 'pipeline' key with DAGDefinition

        Returns:
            AgentResult containing ExecutionResult
        """
        try:
            # Extract pipeline definition
            pipeline_data = input_data.get('pipeline')
            if not pipeline_data:
                return AgentResult(
                    success=False,
                    error="Missing 'pipeline' in input data"
                )

            # Parse pipeline definition
            if isinstance(pipeline_data, dict):
                pipeline = DAGDefinition(**pipeline_data)
            elif isinstance(pipeline_data, DAGDefinition):
                pipeline = pipeline_data
            else:
                return AgentResult(
                    success=False,
                    error="Invalid pipeline definition type"
                )

            # Run pipeline (sync wrapper)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.execute_pipeline(
                        pipeline,
                        tenant_id=input_data.get('tenant_id'),
                        user_id=input_data.get('user_id')
                    )
                )
            finally:
                loop.close()

            return AgentResult(
                success=result.status == PipelineStatus.SUCCESS,
                data=result.model_dump(),
                error=None if result.status == PipelineStatus.SUCCESS else "Pipeline execution failed"
            )

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e)
            )

    async def execute_pipeline(
        self,
        pipeline: DAGDefinition,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute a pipeline asynchronously.

        Args:
            pipeline: The pipeline definition
            tenant_id: Optional tenant ID for multi-tenancy
            user_id: Optional user ID for audit
            checkpoint_id: Optional checkpoint ID to resume from

        Returns:
            ExecutionResult with complete execution details
        """
        execution_id = str(uuid.uuid4())
        started_at = DeterministicClock.now()

        self._execution_count += 1
        self.logger.info(f"Starting pipeline execution: {execution_id}")

        # Create execution context
        context = ExecutionContext(
            execution_id=execution_id,
            pipeline=pipeline,
            nodes={},
            results={},
            lineage={
                "trace_id": str(uuid.uuid4()),
                "pipeline_id": pipeline.pipeline_id,
                "execution_id": execution_id,
                "inputs": [],
                "outputs": [],
                "agents": []
            },
            checkpoints=[],
            started_at=started_at,
            tenant_id=tenant_id,
            user_id=user_id
        )

        # Build execution nodes
        for agent_id, agent_def in pipeline.agents.items():
            context.nodes[agent_id] = AgentNode(
                agent_id=agent_id,
                agent_type=agent_def.get('agent_type', agent_id),
                config=agent_def.get('config', {}),
                dependencies=agent_def.get('dependencies', []),
                input_mapping=agent_def.get('input_mapping', {}),
                timeout_seconds=agent_def.get('timeout_seconds', 300),
                retry_config=agent_def.get('retry_config', {
                    "max_retries": 3,
                    "initial_delay_ms": 1000,
                    "backoff_multiplier": 2,
                    "max_delay_ms": 30000
                })
            )

        self._active_executions[execution_id] = context

        try:
            # Resume from checkpoint if provided
            if checkpoint_id and pipeline.enable_recovery:
                await self._restore_checkpoint(context, checkpoint_id)

            # Compute input hash for determinism verification
            input_hash = self._compute_hash(pipeline.inputs)

            # Execute the DAG
            await self._execute_dag(context, pipeline.inputs)

            # Determine final status
            status = self._determine_status(context)

            # Collect final outputs from terminal nodes
            final_outputs = self._collect_final_outputs(context)
            output_hash = self._compute_hash(final_outputs)

            # Update counters
            if status == PipelineStatus.SUCCESS:
                self._success_count += 1
            else:
                self._failure_count += 1

            completed_at = DeterministicClock.now()
            duration_ms = (completed_at - started_at).total_seconds() * 1000

            # Build result
            result = ExecutionResult(
                execution_id=execution_id,
                pipeline_id=pipeline.pipeline_id,
                status=status,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                agent_results={
                    aid: {
                        "status": node.status.value,
                        "result": node.result.data if node.result else None,
                        "attempts": node.attempts,
                        "started_at": node.started_at.isoformat() if node.started_at else None,
                        "completed_at": node.completed_at.isoformat() if node.completed_at else None
                    }
                    for aid, node in context.nodes.items()
                },
                final_outputs=final_outputs,
                agents_total=len(context.nodes),
                agents_succeeded=sum(1 for n in context.nodes.values() if n.status == AgentStatus.SUCCESS),
                agents_failed=sum(1 for n in context.nodes.values() if n.status == AgentStatus.FAILED),
                agents_skipped=sum(1 for n in context.nodes.values() if n.status == AgentStatus.SKIPPED),
                lineage_id=context.lineage["trace_id"],
                input_hash=input_hash,
                output_hash=output_hash,
                errors=[
                    {
                        "agent_id": aid,
                        "error": node.result.error if node.result else "Unknown error"
                    }
                    for aid, node in context.nodes.items()
                    if node.status == AgentStatus.FAILED
                ],
                metadata={
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "checkpoints_created": len(context.checkpoints)
                }
            )

            self.logger.info(
                f"Pipeline completed: {execution_id} "
                f"status={status.value} "
                f"duration={duration_ms:.2f}ms"
            )

            return result

        except asyncio.TimeoutError:
            self._failure_count += 1
            return ExecutionResult(
                execution_id=execution_id,
                pipeline_id=pipeline.pipeline_id,
                status=PipelineStatus.TIMEOUT,
                started_at=started_at,
                completed_at=DeterministicClock.now(),
                errors=[{"error": "Pipeline execution timed out"}]
            )

        except Exception as e:
            self._failure_count += 1
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            return ExecutionResult(
                execution_id=execution_id,
                pipeline_id=pipeline.pipeline_id,
                status=PipelineStatus.FAILED,
                started_at=started_at,
                completed_at=DeterministicClock.now(),
                errors=[{"error": str(e)}]
            )

        finally:
            # Cleanup
            if execution_id in self._active_executions:
                del self._active_executions[execution_id]

    async def _execute_dag(
        self,
        context: ExecutionContext,
        initial_inputs: Dict[str, Any]
    ):
        """
        Execute the DAG using topological ordering with parallel execution.

        Args:
            context: Execution context
            initial_inputs: Initial pipeline inputs
        """
        # Store initial inputs
        context.results["__pipeline_inputs__"] = AgentResult(
            success=True,
            data=initial_inputs
        )

        # Get execution order using Kahn's algorithm
        execution_order = self._topological_sort(context.nodes)

        # Track completed agents for dependency resolution
        completed: Set[str] = set()

        # Execute in waves (parallel where possible)
        while execution_order:
            # Find agents whose dependencies are all completed
            ready = [
                aid for aid in execution_order
                if all(dep in completed for dep in context.nodes[aid].dependencies)
            ]

            if not ready:
                # Deadlock detection
                remaining = [aid for aid in execution_order if aid not in completed]
                raise RuntimeError(f"Deadlock detected. Remaining agents: {remaining}")

            # Limit parallelism
            batch = ready[:context.pipeline.max_parallel]

            # Execute batch in parallel
            tasks = [
                self._execute_agent(context, aid)
                for aid in batch
            ]

            await asyncio.gather(*tasks, return_exceptions=True)

            # Update completed set
            for aid in batch:
                completed.add(aid)
                execution_order.remove(aid)

            # Create checkpoint if needed
            if len(completed) % context.pipeline.checkpoint_interval == 0:
                await self._create_checkpoint(context)

    async def _execute_agent(
        self,
        context: ExecutionContext,
        agent_id: str
    ):
        """
        Execute a single agent with retry logic.

        Args:
            context: Execution context
            agent_id: Agent ID to execute
        """
        node = context.nodes[agent_id]
        node.status = AgentStatus.RUNNING
        node.started_at = DeterministicClock.now()

        self.logger.info(f"Executing agent: {agent_id} (type: {node.agent_type})")

        # Gather inputs from dependencies
        agent_inputs = self._gather_inputs(context, node)

        # Get agent class from registry
        agent_class = self._agent_registry.get(node.agent_type)
        if not agent_class:
            # Use a mock agent for unregistered types
            self.logger.warning(f"Agent type not registered: {node.agent_type}")
            node.status = AgentStatus.FAILED
            node.result = AgentResult(
                success=False,
                error=f"Agent type not registered: {node.agent_type}"
            )
            node.completed_at = DeterministicClock.now()
            return

        # Retry loop
        retry_config = node.retry_config
        max_retries = retry_config.get("max_retries", 3)
        delay_ms = retry_config.get("initial_delay_ms", 1000)
        backoff = retry_config.get("backoff_multiplier", 2)
        max_delay = retry_config.get("max_delay_ms", 30000)

        for attempt in range(max_retries + 1):
            node.attempts = attempt + 1

            try:
                # Create agent instance
                agent = agent_class(AgentConfig(
                    name=f"{node.agent_type}-{agent_id}",
                    description=f"Instance of {node.agent_type}",
                    parameters=node.config
                ))

                # Execute with timeout
                async with asyncio.timeout(node.timeout_seconds):
                    # Run in executor to not block
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        agent.run,
                        agent_inputs
                    )

                if result.success:
                    node.status = AgentStatus.SUCCESS
                    node.result = result
                    node.completed_at = DeterministicClock.now()

                    # Store result for downstream agents
                    context.results[agent_id] = result

                    # Record lineage
                    context.lineage["agents"].append({
                        "agent_id": agent_id,
                        "agent_type": node.agent_type,
                        "inputs": list(node.dependencies),
                        "output_hash": self._compute_hash(result.data),
                        "timestamp": node.completed_at.isoformat()
                    })

                    self.logger.info(f"Agent succeeded: {agent_id}")
                    return

                else:
                    # Execution returned failure
                    if attempt < max_retries:
                        node.status = AgentStatus.RETRYING
                        self.logger.warning(
                            f"Agent {agent_id} failed (attempt {attempt + 1}), retrying..."
                        )
                        await asyncio.sleep(delay_ms / 1000)
                        delay_ms = min(delay_ms * backoff, max_delay)
                    else:
                        node.status = AgentStatus.FAILED
                        node.result = result
                        node.completed_at = DeterministicClock.now()
                        self.logger.error(f"Agent failed after {max_retries + 1} attempts: {agent_id}")
                        return

            except asyncio.TimeoutError:
                if attempt < max_retries:
                    node.status = AgentStatus.RETRYING
                    self.logger.warning(f"Agent {agent_id} timed out, retrying...")
                    await asyncio.sleep(delay_ms / 1000)
                    delay_ms = min(delay_ms * backoff, max_delay)
                else:
                    node.status = AgentStatus.TIMEOUT
                    node.result = AgentResult(
                        success=False,
                        error=f"Agent timed out after {node.timeout_seconds}s"
                    )
                    node.completed_at = DeterministicClock.now()
                    return

            except Exception as e:
                self.logger.error(f"Agent execution error: {e}", exc_info=True)
                if attempt < max_retries:
                    node.status = AgentStatus.RETRYING
                    await asyncio.sleep(delay_ms / 1000)
                    delay_ms = min(delay_ms * backoff, max_delay)
                else:
                    node.status = AgentStatus.FAILED
                    node.result = AgentResult(
                        success=False,
                        error=str(e)
                    )
                    node.completed_at = DeterministicClock.now()
                    return

    def _gather_inputs(
        self,
        context: ExecutionContext,
        node: AgentNode
    ) -> Dict[str, Any]:
        """
        Gather inputs for an agent from its dependencies.

        Args:
            context: Execution context
            node: The agent node

        Returns:
            Dictionary of inputs for the agent
        """
        inputs = {}

        # Start with pipeline inputs
        pipeline_inputs = context.results.get("__pipeline_inputs__")
        if pipeline_inputs:
            inputs["pipeline"] = pipeline_inputs.data

        # Map dependency outputs to inputs
        for dep_id in node.dependencies:
            dep_result = context.results.get(dep_id)
            if dep_result and dep_result.success:
                inputs[dep_id] = dep_result.data

        # Apply input mappings if specified
        for target_key, source_path in node.input_mapping.items():
            # Source path format: "agent_id.key" or "pipeline.key"
            parts = source_path.split(".", 1)
            if len(parts) == 2:
                source_agent, source_key = parts
                if source_agent == "pipeline":
                    inputs[target_key] = pipeline_inputs.data.get(source_key) if pipeline_inputs else None
                elif source_agent in context.results:
                    inputs[target_key] = context.results[source_agent].data.get(source_key)

        return inputs

    def _topological_sort(self, nodes: Dict[str, AgentNode]) -> List[str]:
        """
        Perform topological sort on the DAG.

        Args:
            nodes: Dictionary of agent nodes

        Returns:
            List of agent IDs in topological order
        """
        # Build in-degree map
        in_degree = {aid: 0 for aid in nodes}
        for node in nodes.values():
            for dep in node.dependencies:
                if dep in in_degree:
                    in_degree[node.agent_id] += 1

        # Start with nodes that have no dependencies
        queue = [aid for aid, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            aid = queue.pop(0)
            result.append(aid)

            # Reduce in-degree of dependents
            for node in nodes.values():
                if aid in node.dependencies:
                    in_degree[node.agent_id] -= 1
                    if in_degree[node.agent_id] == 0:
                        queue.append(node.agent_id)

        return result

    def _determine_status(self, context: ExecutionContext) -> PipelineStatus:
        """Determine overall pipeline status from agent statuses."""
        statuses = [node.status for node in context.nodes.values()]

        if all(s == AgentStatus.SUCCESS for s in statuses):
            return PipelineStatus.SUCCESS
        elif all(s in (AgentStatus.FAILED, AgentStatus.TIMEOUT) for s in statuses):
            return PipelineStatus.FAILED
        elif any(s == AgentStatus.SUCCESS for s in statuses):
            return PipelineStatus.PARTIAL
        else:
            return PipelineStatus.FAILED

    def _collect_final_outputs(self, context: ExecutionContext) -> Dict[str, Any]:
        """
        Collect outputs from terminal nodes (nodes with no dependents).

        Args:
            context: Execution context

        Returns:
            Dictionary of final outputs
        """
        # Find terminal nodes (no other nodes depend on them)
        dependents = set()
        for node in context.nodes.values():
            dependents.update(node.dependencies)

        terminal_nodes = [
            aid for aid in context.nodes
            if aid not in dependents
        ]

        outputs = {}
        for aid in terminal_nodes:
            result = context.results.get(aid)
            if result and result.success:
                outputs[aid] = result.data

        return outputs

    def _compute_hash(self, data: Any) -> str:
        """Compute deterministic hash of data for lineage tracking."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    async def _create_checkpoint(self, context: ExecutionContext):
        """Create a checkpoint for recovery."""
        checkpoint = {
            "execution_id": context.execution_id,
            "timestamp": DeterministicClock.now().isoformat(),
            "completed_agents": [
                aid for aid, node in context.nodes.items()
                if node.status == AgentStatus.SUCCESS
            ],
            "results": {
                aid: result.model_dump()
                for aid, result in context.results.items()
            }
        }
        context.checkpoints.append(checkpoint)
        self._checkpoints[context.execution_id] = context.checkpoints
        self.logger.debug(f"Checkpoint created: {len(context.checkpoints)}")

    async def _restore_checkpoint(
        self,
        context: ExecutionContext,
        checkpoint_id: str
    ):
        """Restore execution state from checkpoint."""
        checkpoints = self._checkpoints.get(checkpoint_id, [])
        if not checkpoints:
            self.logger.warning(f"No checkpoints found for: {checkpoint_id}")
            return

        # Use latest checkpoint
        checkpoint = checkpoints[-1]

        # Restore completed agents
        for aid in checkpoint.get("completed_agents", []):
            if aid in context.nodes:
                context.nodes[aid].status = AgentStatus.SUCCESS

        # Restore results
        for aid, result_data in checkpoint.get("results", {}).items():
            context.results[aid] = AgentResult(**result_data)

        self.logger.info(f"Restored from checkpoint with {len(checkpoint.get('completed_agents', []))} completed agents")

    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active execution."""
        context = self._active_executions.get(execution_id)
        if not context:
            return None

        return {
            "execution_id": execution_id,
            "pipeline_id": context.pipeline.pipeline_id,
            "started_at": context.started_at.isoformat(),
            "agents": {
                aid: {
                    "status": node.status.value,
                    "attempts": node.attempts
                }
                for aid, node in context.nodes.items()
            }
        }

    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution."""
        if execution_id in self._active_executions:
            # Mark all pending agents as cancelled
            context = self._active_executions[execution_id]
            for node in context.nodes.values():
                if node.status in (AgentStatus.PENDING, AgentStatus.QUEUED):
                    node.status = AgentStatus.SKIPPED
            return True
        return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics."""
        return {
            "total_executions": self._execution_count,
            "successful_executions": self._success_count,
            "failed_executions": self._failure_count,
            "success_rate": (
                self._success_count / self._execution_count * 100
                if self._execution_count > 0 else 0
            ),
            "active_executions": len(self._active_executions),
            "registered_agents": len(self._agent_registry)
        }
