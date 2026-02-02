# -*- coding: utf-8 -*-
"""
GLIP v1 Orchestrator Extension
==============================

Extends the GreenLang Orchestrator with GLIP v1 protocol support.

This module provides:
    - K8s Job execution backend integration
    - S3 artifact store integration
    - Policy engine enforcement
    - Hash-chained audit events
    - Agent registry with GLIP v1 metadata
    - Support for both GLIP v1 and Legacy HTTP execution modes

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult
from greenlang.agents.foundation.orchestrator import (
    AgentNode,
    AgentStatus,
    DAGDefinition,
    ExecutionContext,
    ExecutionResult,
    GreenLangOrchestrator,
    PipelineStatus,
)
from greenlang.agents.foundation.agent_registry import (
    AgentMetadataEntry,
    ContainerSpec,
    ExecutionMode,
    IdempotencySupport,
    ResourceProfile,
    VersionedAgentRegistry,
)
# Core executor interfaces (always available)
from greenlang.orchestrator.executors.base import (
    ExecutorBackend,
    RunContext,
    StepResult,
    ExecutionStatus as StepStatus,
)

# Optional: K8s executor (requires kubernetes_asyncio)
try:
    from greenlang.orchestrator.executors.k8s_executor import K8sExecutor, K8sExecutorConfig
except ImportError:
    K8sExecutor = None
    K8sExecutorConfig = None

# Core artifact interfaces (always available)
from greenlang.orchestrator.artifacts.base import ArtifactStore

# Optional: S3 store (requires aioboto3)
try:
    from greenlang.orchestrator.artifacts.s3_store import S3ArtifactStore, S3StoreConfig
except ImportError:
    S3ArtifactStore = None
    S3StoreConfig = None

# Policy engine (always available - no external deps)
from greenlang.orchestrator.governance.policy_engine import (
    PolicyEngine,
    PolicyEngineConfig,
    PolicyDecision,
    EvaluationPoint,
)

# Audit event store (always available - no external deps)
try:
    from greenlang.orchestrator.audit.event_store import (
        EventStore as AuditEventStore,
        EventType,
    )
    AuditEventStoreConfig = None  # Not yet implemented
except ImportError:
    AuditEventStore = None
    AuditEventStoreConfig = None
    EventType = None

# Legacy HTTP adapter (always available - uses aiohttp)
try:
    from greenlang.orchestrator.adapters.http_legacy_adapter import (
        HttpLegacyAdapter,
        AdapterConfig,
    )
except ImportError:
    HttpLegacyAdapter = None
    AdapterConfig = None

# Deterministic clock utility
try:
    from greenlang.utilities.determinism import DeterministicClock
except ImportError:
    from datetime import datetime as dt
    class DeterministicClock:
        @classmethod
        def now(cls):
            return dt.utcnow()

logger = logging.getLogger(__name__)


class GLIPExecutionMode(str, Enum):
    """Execution mode for a step."""
    GLIP_V1 = "glip_v1"         # K8s Job with artifact protocol
    LEGACY_HTTP = "legacy_http"  # HTTP endpoint via adapter
    IN_PROCESS = "in_process"    # Direct Python execution


class ApprovalStatus(str, Enum):
    """Status of a human approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


class ApprovalRequest(BaseModel):
    """Human approval gate request."""
    request_id: str = Field(..., description="Unique approval request ID")
    run_id: str = Field(..., description="Associated run ID")
    step_id: str = Field(..., description="Step requiring approval")
    pipeline_id: str = Field(..., description="Pipeline ID")
    tenant_id: str = Field(..., description="Tenant ID")
    user_id: Optional[str] = Field(None, description="Requesting user")
    reason: str = Field(..., description="Reason for approval request")
    requested_at: datetime = Field(default_factory=DeterministicClock.now)
    expires_at: Optional[datetime] = Field(None, description="Expiration time")
    status: ApprovalStatus = Field(default=ApprovalStatus.PENDING)
    approved_by: Optional[str] = Field(None, description="Approver identity")
    approved_at: Optional[datetime] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FanOutSpec(BaseModel):
    """Dynamic fan-out specification."""
    source_field: str = Field(..., description="Field containing items to fan out")
    item_param_name: str = Field(default="item", description="Parameter name for each item")
    max_parallel: int = Field(default=10, description="Maximum parallel executions")
    continue_on_failure: bool = Field(default=False, description="Continue if some items fail")
    aggregate_results: bool = Field(default=True, description="Aggregate all results")


class ConcurrencyConfig(BaseModel):
    """Concurrency control configuration."""
    # Global limits
    max_concurrent_runs: int = Field(default=100, description="Max concurrent pipeline runs")
    max_concurrent_steps_per_run: int = Field(default=10, description="Max parallel steps per run")

    # Rate limiting
    rate_limit_requests_per_minute: int = Field(default=1000, description="API rate limit")
    rate_limit_k8s_jobs_per_minute: int = Field(default=60, description="K8s job creation rate")

    # Step-level limits
    agent_concurrency_limits: Dict[str, int] = Field(
        default_factory=dict,
        description="Per-agent concurrency limits (agent_id -> max_concurrent)"
    )

    # Queue configuration
    queue_timeout_seconds: int = Field(default=300, description="Max time waiting in queue")
    enable_fair_scheduling: bool = Field(default=True, description="Fair scheduling across tenants")


class GLIPRunConfig(BaseModel):
    """Configuration for a GLIP v1 pipeline run."""
    pipeline_id: str = Field(..., description="Pipeline identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Distributed trace ID")
    prefer_glip: bool = Field(default=True, description="Prefer GLIP v1 execution for hybrid agents")
    dry_run: bool = Field(default=False, description="Validate without executing")
    skip_policy_check: bool = Field(default=False, description="Skip policy enforcement (dev only)")
    approval_callback_url: Optional[str] = Field(None, description="URL for approval notifications")

    # Resource overrides
    default_cpu_limit: str = Field(default="1", description="Default CPU limit")
    default_memory_limit: str = Field(default="1Gi", description="Default memory limit")
    default_timeout_seconds: int = Field(default=3600, description="Default timeout")

    # Approval gates
    enable_approval_gates: bool = Field(default=True, description="Enable human approval gates")
    approval_timeout_seconds: int = Field(default=86400, description="Approval timeout (24h default)")
    auto_approve_in_dev: bool = Field(default=False, description="Auto-approve in dev environment")

    # Fan-out configuration
    max_fan_out_items: int = Field(default=100, description="Maximum items in a fan-out")
    fan_out_batch_size: int = Field(default=10, description="Batch size for fan-out processing")

    # Concurrency overrides
    concurrency: Optional[ConcurrencyConfig] = Field(None, description="Concurrency configuration")


class GLIPStepContext(BaseModel):
    """Extended context for GLIP v1 step execution."""
    run_id: str
    step_id: str
    pipeline_id: str
    tenant_id: str
    agent_id: str
    agent_version: str
    execution_mode: GLIPExecutionMode
    idempotency_key: str
    trace_id: str
    input_uri: Optional[str] = None
    output_uri: Optional[str] = None
    container_image: Optional[str] = None
    resource_profile: Optional[Dict[str, Any]] = None


class GLIPOrchestratorConfig(BaseModel):
    """Configuration for the GLIP v1 Orchestrator."""
    # K8s Executor
    k8s_namespace: str = Field(default="greenlang", description="K8s namespace for jobs")
    k8s_service_account: str = Field(default="greenlang-runner", description="K8s service account")
    k8s_image_pull_secrets: List[str] = Field(default_factory=list, description="Image pull secrets")

    # S3 Artifact Store
    s3_bucket: str = Field(default="greenlang-artifacts", description="S3 bucket for artifacts")
    s3_endpoint: Optional[str] = Field(None, description="S3 endpoint (for MinIO)")
    s3_region: str = Field(default="us-east-1", description="S3 region")

    # Policy Engine
    opa_url: Optional[str] = Field(None, description="OPA server URL")
    policy_bundle_path: Optional[str] = Field(None, description="Path to policy bundle")

    # Audit Store
    audit_db_url: Optional[str] = Field(None, description="PostgreSQL URL for audit")

    # Execution
    max_parallel_steps: int = Field(default=10, description="Max parallel step execution")
    default_timeout_seconds: int = Field(default=3600, description="Default step timeout")
    enable_legacy_fallback: bool = Field(default=True, description="Fallback to legacy HTTP")


class GLIPOrchestrator(GreenLangOrchestrator):
    """
    GLIP v1 Extended Orchestrator

    Extends the base GreenLang Orchestrator with:
    - K8s Job execution backend
    - S3 artifact store for inputs/outputs
    - OPA policy engine integration
    - Hash-chained audit events
    - Support for GLIP v1 and Legacy HTTP modes

    Usage:
        config = GLIPOrchestratorConfig(
            k8s_namespace="production",
            s3_bucket="my-artifacts",
        )
        orchestrator = GLIPOrchestrator(config)

        run_config = GLIPRunConfig(
            pipeline_id="my-pipeline",
            tenant_id="tenant-123",
        )
        result = await orchestrator.execute_glip_pipeline(pipeline, run_config)
    """

    VERSION = "2.0.0"  # GLIP v1 version

    def __init__(
        self,
        config: Optional[GLIPOrchestratorConfig] = None,
        agent_registry: Optional[VersionedAgentRegistry] = None,
    ):
        """Initialize the GLIP v1 Orchestrator."""
        super().__init__(AgentConfig(
            name="GLIP v1 Orchestrator",
            description="Extended orchestrator with GLIP v1 protocol support",
            version=self.VERSION,
        ))

        self.glip_config = config or GLIPOrchestratorConfig()

        # Agent registry with GLIP v1 extensions
        self.agent_registry = agent_registry or VersionedAgentRegistry()

        # Execution backends
        self._k8s_executor: Optional[K8sExecutor] = None
        self._artifact_store: Optional[ArtifactStore] = None
        self._policy_engine: Optional[PolicyEngine] = None
        self._audit_store: Optional[AuditEventStore] = None
        self._legacy_adapters: Dict[str, HttpLegacyAdapter] = {}

        # Run tracking
        self._active_runs: Dict[str, Dict[str, Any]] = {}

        # Approval gates tracking
        self._pending_approvals: Dict[str, Dict[str, ApprovalRequest]] = {}

        # Concurrency control tracking
        self._concurrency_slots: Dict[str, Dict[str, Any]] = {}

        logger.info(f"Initialized GLIPOrchestrator v{self.VERSION}")

    async def initialize(self):
        """Initialize all backends and stores."""
        logger.info("Initializing GLIP v1 backends...")

        # Initialize K8s executor
        try:
            k8s_config = K8sExecutorConfig(
                namespace=self.glip_config.k8s_namespace,
                service_account=self.glip_config.k8s_service_account,
                image_pull_secrets=self.glip_config.k8s_image_pull_secrets,
                default_timeout_seconds=self.glip_config.default_timeout_seconds,
            )
            self._k8s_executor = K8sExecutor(k8s_config)
            await self._k8s_executor.initialize()
            logger.info("K8s executor initialized")
        except Exception as e:
            logger.warning(f"K8s executor initialization failed: {e}")

        # Initialize S3 artifact store
        try:
            s3_config = S3StoreConfig(
                bucket=self.glip_config.s3_bucket,
                endpoint_url=self.glip_config.s3_endpoint,
                region=self.glip_config.s3_region,
            )
            self._artifact_store = S3ArtifactStore(s3_config)
            await self._artifact_store.initialize()
            logger.info("S3 artifact store initialized")
        except Exception as e:
            logger.warning(f"S3 artifact store initialization failed: {e}")

        # Initialize policy engine
        if self.glip_config.opa_url or self.glip_config.policy_bundle_path:
            try:
                policy_config = PolicyEngineConfig(
                    opa_url=self.glip_config.opa_url,
                    yaml_rules_path=self.glip_config.policy_bundle_path,
                )
                self._policy_engine = PolicyEngine(policy_config)
                await self._policy_engine.initialize()
                logger.info("Policy engine initialized")
            except Exception as e:
                logger.warning(f"Policy engine initialization failed: {e}")

        # Initialize audit store
        if self.glip_config.audit_db_url:
            try:
                audit_config = AuditEventStoreConfig(
                    database_url=self.glip_config.audit_db_url,
                )
                self._audit_store = AuditEventStore(audit_config)
                await self._audit_store.initialize()
                logger.info("Audit event store initialized")
            except Exception as e:
                logger.warning(f"Audit store initialization failed: {e}")

        logger.info("GLIP v1 backends initialized")

    async def shutdown(self):
        """Shutdown all backends gracefully."""
        logger.info("Shutting down GLIP v1 backends...")

        if self._k8s_executor:
            await self._k8s_executor.shutdown()

        if self._artifact_store:
            await self._artifact_store.shutdown()

        if self._policy_engine:
            await self._policy_engine.shutdown()

        if self._audit_store:
            await self._audit_store.shutdown()

        for adapter in self._legacy_adapters.values():
            await adapter.shutdown()

        logger.info("GLIP v1 backends shut down")

    async def execute_glip_pipeline(
        self,
        pipeline: DAGDefinition,
        run_config: GLIPRunConfig,
    ) -> ExecutionResult:
        """
        Execute a pipeline using GLIP v1 protocol.

        This method:
        1. Validates the pipeline and run configuration
        2. Checks pre-run policies
        3. Compiles the execution plan
        4. Executes steps using appropriate backends (K8s/HTTP)
        5. Emits audit events throughout
        6. Returns the execution result

        Args:
            pipeline: The pipeline definition
            run_config: GLIP v1 run configuration

        Returns:
            ExecutionResult with complete execution details
        """
        run_id = str(uuid.uuid4())
        started_at = DeterministicClock.now()

        logger.info(f"Starting GLIP v1 pipeline run: {run_id}")

        # Emit run submitted event
        await self._emit_audit_event(
            run_id=run_id,
            event_type=EventType.RUN_SUBMITTED,
            payload={
                "pipeline_id": pipeline.pipeline_id,
                "tenant_id": run_config.tenant_id,
                "user_id": run_config.user_id,
            }
        )

        try:
            # 1. Pre-run policy check
            if not run_config.skip_policy_check and self._policy_engine:
                decision = await self._policy_engine.evaluate_pre_run(
                    pipeline=pipeline.model_dump(),
                    run_config=run_config.model_dump(),
                )

                if not decision.allowed:
                    await self._emit_audit_event(
                        run_id=run_id,
                        event_type=EventType.POLICY_VIOLATION,
                        payload={
                            "evaluation_point": "pre_run",
                            "reasons": [r.model_dump() for r in decision.reasons],
                        }
                    )

                    return ExecutionResult(
                        execution_id=run_id,
                        pipeline_id=pipeline.pipeline_id,
                        status=PipelineStatus.FAILED,
                        started_at=started_at,
                        completed_at=DeterministicClock.now(),
                        errors=[{
                            "type": "policy_violation",
                            "message": decision.reasons[0].message if decision.reasons else "Policy check failed",
                        }]
                    )

            # 2. Compile execution plan
            plan = await self._compile_plan(pipeline, run_config, run_id)

            await self._emit_audit_event(
                run_id=run_id,
                event_type=EventType.PLAN_COMPILED,
                payload={
                    "plan_id": plan["plan_id"],
                    "step_count": len(plan["steps"]),
                }
            )

            # 3. Dry run check
            if run_config.dry_run:
                return ExecutionResult(
                    execution_id=run_id,
                    pipeline_id=pipeline.pipeline_id,
                    status=PipelineStatus.SUCCESS,
                    started_at=started_at,
                    completed_at=DeterministicClock.now(),
                    metadata={
                        "dry_run": True,
                        "plan": plan,
                    }
                )

            # 4. Execute the plan
            await self._emit_audit_event(
                run_id=run_id,
                event_type=EventType.RUN_STARTED,
                payload={"plan_id": plan["plan_id"]}
            )

            # Track this run
            self._active_runs[run_id] = {
                "pipeline": pipeline,
                "config": run_config,
                "plan": plan,
                "started_at": started_at,
            }

            # Execute using parent's DAG execution with GLIP extensions
            result = await self._execute_glip_dag(pipeline, run_config, run_id, plan)

            # 5. Emit completion event
            await self._emit_audit_event(
                run_id=run_id,
                event_type=EventType.RUN_COMPLETED,
                payload={
                    "status": result.status.value,
                    "duration_ms": result.duration_ms,
                    "agents_succeeded": result.agents_succeeded,
                    "agents_failed": result.agents_failed,
                }
            )

            return result

        except Exception as e:
            logger.error(f"GLIP pipeline execution failed: {e}", exc_info=True)

            await self._emit_audit_event(
                run_id=run_id,
                event_type=EventType.RUN_FAILED,
                payload={"error": str(e)}
            )

            return ExecutionResult(
                execution_id=run_id,
                pipeline_id=pipeline.pipeline_id,
                status=PipelineStatus.FAILED,
                started_at=started_at,
                completed_at=DeterministicClock.now(),
                errors=[{"error": str(e)}]
            )

        finally:
            # Cleanup
            if run_id in self._active_runs:
                del self._active_runs[run_id]

    async def _compile_plan(
        self,
        pipeline: DAGDefinition,
        run_config: GLIPRunConfig,
        run_id: str,
    ) -> Dict[str, Any]:
        """
        Compile the execution plan with GLIP v1 metadata.

        The plan includes:
        - Deterministic plan_id based on content hash
        - Execution mode for each step (GLIP v1 / Legacy HTTP)
        - Resource profiles and container specs
        - Idempotency keys for retry safety
        """
        steps = []

        for agent_id, agent_def in pipeline.agents.items():
            agent_type = agent_def.get("agent_type", agent_id)

            # Get agent metadata from registry
            metadata = self.agent_registry.get_agent(agent_type)

            # Determine execution mode
            if metadata:
                if run_config.prefer_glip and metadata.is_glip_compatible:
                    exec_mode = GLIPExecutionMode.GLIP_V1
                elif metadata.is_legacy_compatible:
                    exec_mode = GLIPExecutionMode.LEGACY_HTTP
                else:
                    exec_mode = GLIPExecutionMode.IN_PROCESS
            else:
                exec_mode = GLIPExecutionMode.IN_PROCESS

            # Generate idempotency key
            idempotency_content = f"{run_id}:{agent_id}:{json.dumps(agent_def, sort_keys=True)}"
            idempotency_key = hashlib.sha256(idempotency_content.encode()).hexdigest()[:32]

            # Generate step_id
            step_id = f"{run_id}-{agent_id}"

            step = {
                "step_id": step_id,
                "agent_id": agent_id,
                "agent_type": agent_type,
                "execution_mode": exec_mode.value,
                "idempotency_key": idempotency_key,
                "dependencies": agent_def.get("dependencies", []),
                "config": agent_def.get("config", {}),
            }

            # Add GLIP v1 specific fields
            if exec_mode == GLIPExecutionMode.GLIP_V1 and metadata:
                step["container_image"] = metadata.container_image
                step["resource_profile"] = metadata.resource_profile.model_dump() if metadata.resource_profile else None
                step["timeout_seconds"] = metadata.get_execution_timeout()
            elif exec_mode == GLIPExecutionMode.LEGACY_HTTP and metadata:
                step["http_config"] = metadata.legacy_http_config.model_dump() if metadata.legacy_http_config else None
                step["timeout_seconds"] = metadata.get_execution_timeout()

            steps.append(step)

        # Generate content-addressable plan_id
        plan_content = json.dumps(steps, sort_keys=True)
        plan_id = hashlib.sha256(plan_content.encode()).hexdigest()[:16]

        return {
            "plan_id": plan_id,
            "run_id": run_id,
            "pipeline_id": pipeline.pipeline_id,
            "steps": steps,
            "compiled_at": DeterministicClock.now().isoformat(),
        }

    async def _execute_glip_dag(
        self,
        pipeline: DAGDefinition,
        run_config: GLIPRunConfig,
        run_id: str,
        plan: Dict[str, Any],
    ) -> ExecutionResult:
        """Execute the DAG using GLIP v1 protocol."""
        started_at = DeterministicClock.now()

        # Build step lookup
        step_lookup = {s["agent_id"]: s for s in plan["steps"]}

        # Create execution context
        context = ExecutionContext(
            execution_id=run_id,
            pipeline=pipeline,
            nodes={},
            results={},
            lineage={
                "trace_id": run_config.trace_id,
                "pipeline_id": pipeline.pipeline_id,
                "run_id": run_id,
                "plan_id": plan["plan_id"],
                "inputs": [],
                "outputs": [],
                "steps": [],
            },
            checkpoints=[],
            started_at=started_at,
            tenant_id=run_config.tenant_id,
            user_id=run_config.user_id,
        )

        # Build execution nodes
        for agent_id, agent_def in pipeline.agents.items():
            context.nodes[agent_id] = AgentNode(
                agent_id=agent_id,
                agent_type=agent_def.get("agent_type", agent_id),
                config=agent_def.get("config", {}),
                dependencies=agent_def.get("dependencies", []),
                input_mapping=agent_def.get("input_mapping", {}),
                timeout_seconds=agent_def.get("timeout_seconds", run_config.default_timeout_seconds),
            )

        # Store initial inputs
        context.results["__pipeline_inputs__"] = AgentResult(
            success=True,
            data=pipeline.inputs,
        )

        # Get execution order
        execution_order = self._topological_sort(context.nodes)
        completed: set = set()

        # Execute in waves
        while execution_order:
            ready = [
                aid for aid in execution_order
                if all(dep in completed for dep in context.nodes[aid].dependencies)
            ]

            if not ready:
                remaining = [aid for aid in execution_order if aid not in completed]
                raise RuntimeError(f"Deadlock detected. Remaining: {remaining}")

            batch = ready[:self.glip_config.max_parallel_steps]

            # Execute batch in parallel
            tasks = [
                self._execute_glip_step(context, aid, step_lookup[aid], run_config)
                for aid in batch
            ]

            await asyncio.gather(*tasks, return_exceptions=True)

            for aid in batch:
                completed.add(aid)
                execution_order.remove(aid)

            # Checkpoint
            if len(completed) % pipeline.checkpoint_interval == 0:
                await self._create_checkpoint(context)

        # Determine final status
        status = self._determine_status(context)

        # Collect outputs
        final_outputs = self._collect_final_outputs(context)

        completed_at = DeterministicClock.now()
        duration_ms = (completed_at - started_at).total_seconds() * 1000

        return ExecutionResult(
            execution_id=run_id,
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
                }
                for aid, node in context.nodes.items()
            },
            final_outputs=final_outputs,
            agents_total=len(context.nodes),
            agents_succeeded=sum(1 for n in context.nodes.values() if n.status == AgentStatus.SUCCESS),
            agents_failed=sum(1 for n in context.nodes.values() if n.status == AgentStatus.FAILED),
            agents_skipped=sum(1 for n in context.nodes.values() if n.status == AgentStatus.SKIPPED),
            lineage_id=context.lineage["trace_id"],
            input_hash=self._compute_hash(pipeline.inputs),
            output_hash=self._compute_hash(final_outputs),
            metadata={
                "plan_id": plan["plan_id"],
                "tenant_id": run_config.tenant_id,
                "glip_version": "1.0.0",
            }
        )

    async def _execute_glip_step(
        self,
        context: ExecutionContext,
        agent_id: str,
        step: Dict[str, Any],
        run_config: GLIPRunConfig,
    ):
        """Execute a single step using GLIP v1 protocol."""
        node = context.nodes[agent_id]
        node.status = AgentStatus.RUNNING
        node.started_at = DeterministicClock.now()

        step_id = step["step_id"]
        exec_mode = GLIPExecutionMode(step["execution_mode"])

        logger.info(f"Executing step {step_id} via {exec_mode.value}")

        # Emit step started event
        await self._emit_audit_event(
            run_id=context.execution_id,
            event_type=EventType.STEP_STARTED,
            payload={
                "step_id": step_id,
                "agent_id": agent_id,
                "execution_mode": exec_mode.value,
            }
        )

        # Pre-step policy check
        if not run_config.skip_policy_check and self._policy_engine:
            decision = await self._policy_engine.evaluate_pre_step(
                step=step,
                context={
                    "run_id": context.execution_id,
                    "tenant_id": run_config.tenant_id,
                }
            )

            if not decision.allowed:
                node.status = AgentStatus.FAILED
                node.result = AgentResult(
                    success=False,
                    error=f"Policy violation: {decision.reasons[0].message if decision.reasons else 'Denied'}"
                )
                node.completed_at = DeterministicClock.now()
                return

        # Gather inputs
        inputs = self._gather_inputs(context, node)

        try:
            if exec_mode == GLIPExecutionMode.GLIP_V1:
                result = await self._execute_via_k8s(context, node, step, inputs, run_config)
            elif exec_mode == GLIPExecutionMode.LEGACY_HTTP:
                result = await self._execute_via_http(context, node, step, inputs, run_config)
            else:
                result = await self._execute_in_process(context, node, step, inputs)

            if result.success:
                node.status = AgentStatus.SUCCESS
                node.result = result
                context.results[agent_id] = result

                # Record lineage
                context.lineage["steps"].append({
                    "step_id": step_id,
                    "agent_id": agent_id,
                    "output_hash": self._compute_hash(result.data),
                    "timestamp": DeterministicClock.now().isoformat(),
                })

                await self._emit_audit_event(
                    run_id=context.execution_id,
                    event_type=EventType.STEP_COMPLETED,
                    payload={
                        "step_id": step_id,
                        "status": "success",
                    }
                )
            else:
                node.status = AgentStatus.FAILED
                node.result = result

                await self._emit_audit_event(
                    run_id=context.execution_id,
                    event_type=EventType.STEP_FAILED,
                    payload={
                        "step_id": step_id,
                        "error": result.error,
                    }
                )

        except Exception as e:
            logger.error(f"Step execution failed: {e}", exc_info=True)
            node.status = AgentStatus.FAILED
            node.result = AgentResult(success=False, error=str(e))

            await self._emit_audit_event(
                run_id=context.execution_id,
                event_type=EventType.STEP_FAILED,
                payload={
                    "step_id": step_id,
                    "error": str(e),
                }
            )

        finally:
            node.completed_at = DeterministicClock.now()

    async def _execute_via_k8s(
        self,
        context: ExecutionContext,
        node: AgentNode,
        step: Dict[str, Any],
        inputs: Dict[str, Any],
        run_config: GLIPRunConfig,
    ) -> AgentResult:
        """Execute step via K8s Job with GLIP v1 protocol."""
        if not self._k8s_executor:
            raise RuntimeError("K8s executor not initialized")

        if not self._artifact_store:
            raise RuntimeError("Artifact store not initialized")

        step_id = step["step_id"]

        # Write input context to S3
        input_uri = await self._artifact_store.write_input_context(
            run_id=context.execution_id,
            step_id=step_id,
            tenant_id=run_config.tenant_id,
            inputs=inputs,
            params=step.get("config", {}),
        )

        # Create run context for K8s executor
        run_context = RunContext(
            run_id=context.execution_id,
            step_id=step_id,
            pipeline_id=context.pipeline.pipeline_id,
            tenant_id=run_config.tenant_id,
            agent_id=step["agent_type"],
            agent_version=step.get("agent_version", "latest"),
            params=step.get("config", {}),
            inputs={},  # Inputs are in S3
            idempotency_key=step["idempotency_key"],
            trace_id=run_config.trace_id,
        )

        # Execute via K8s
        step_result = await self._k8s_executor.execute(
            context=run_context,
            container_image=step["container_image"],
            resource_profile=step.get("resource_profile", {}),
        )

        if step_result.status == StepStatus.COMPLETED:
            # Read output from S3
            output_data = await self._artifact_store.read_result(
                run_id=context.execution_id,
                step_id=step_id,
                tenant_id=run_config.tenant_id,
            )

            return AgentResult(
                success=True,
                data=output_data.get("outputs", {}),
                metadata={
                    "step_metadata": step_result.metadata.model_dump() if step_result.metadata else None,
                }
            )
        else:
            return AgentResult(
                success=False,
                error=step_result.error or "K8s job failed",
            )

    async def _execute_via_http(
        self,
        context: ExecutionContext,
        node: AgentNode,
        step: Dict[str, Any],
        inputs: Dict[str, Any],
        run_config: GLIPRunConfig,
    ) -> AgentResult:
        """Execute step via Legacy HTTP adapter."""
        agent_type = step["agent_type"]

        # Get or create adapter
        if agent_type not in self._legacy_adapters:
            http_config = step.get("http_config", {})
            if not http_config:
                raise RuntimeError(f"No HTTP config for agent: {agent_type}")

            adapter_config = AdapterConfig(
                endpoint=http_config["endpoint"],
                method=http_config.get("method", "POST"),
                timeout_seconds=http_config.get("timeout_seconds", 300),
            )
            self._legacy_adapters[agent_type] = HttpLegacyAdapter(adapter_config)

        adapter = self._legacy_adapters[agent_type]

        # Execute via adapter
        return await adapter.execute(
            agent_id=agent_type,
            inputs=inputs,
            config=step.get("config", {}),
            idempotency_key=step["idempotency_key"],
        )

    async def _execute_in_process(
        self,
        context: ExecutionContext,
        node: AgentNode,
        step: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> AgentResult:
        """Execute step in-process (fallback for unregistered agents)."""
        agent_class = self._agent_registry.get(node.agent_type)

        if not agent_class:
            return AgentResult(
                success=False,
                error=f"Agent type not registered: {node.agent_type}"
            )

        # Use base orchestrator execution
        agent = agent_class(AgentConfig(
            name=f"{node.agent_type}-{node.agent_id}",
            parameters=node.config,
        ))

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, agent.run, inputs)

    async def _emit_audit_event(
        self,
        run_id: str,
        event_type: EventType,
        payload: Dict[str, Any],
    ):
        """Emit a hash-chained audit event."""
        if self._audit_store:
            try:
                await self._audit_store.emit(
                    run_id=run_id,
                    event_type=event_type,
                    payload=payload,
                )
            except Exception as e:
                logger.warning(f"Failed to emit audit event: {e}")
        else:
            logger.debug(f"Audit event (no store): {event_type.value} - {payload}")

    def get_run_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active GLIP run."""
        run_info = self._active_runs.get(run_id)
        if not run_info:
            return None

        return {
            "run_id": run_id,
            "pipeline_id": run_info["pipeline"].pipeline_id,
            "started_at": run_info["started_at"].isoformat(),
            "plan_id": run_info["plan"]["plan_id"],
        }

    # =========================================================================
    # APPROVAL GATES (P1 Feature)
    # =========================================================================

    async def request_approval(
        self,
        run_id: str,
        step_id: str,
        reason: str,
        run_config: GLIPRunConfig,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ApprovalRequest:
        """
        Request human approval for a step.

        This pauses execution and creates an approval request that must be
        approved/rejected before the step can proceed.

        Args:
            run_id: Run identifier
            step_id: Step requiring approval
            reason: Human-readable reason for approval
            run_config: Run configuration
            metadata: Additional context for approvers

        Returns:
            ApprovalRequest object
        """
        request_id = f"approval-{run_id}-{step_id}-{uuid.uuid4().hex[:8]}"

        expires_at = None
        if run_config.approval_timeout_seconds:
            expires_at = DeterministicClock.now() + timedelta(
                seconds=run_config.approval_timeout_seconds
            )

        request = ApprovalRequest(
            request_id=request_id,
            run_id=run_id,
            step_id=step_id,
            pipeline_id=run_config.pipeline_id,
            tenant_id=run_config.tenant_id,
            user_id=run_config.user_id,
            reason=reason,
            expires_at=expires_at,
            metadata=metadata or {},
        )

        # Store pending approval
        if run_id not in self._pending_approvals:
            self._pending_approvals[run_id] = {}
        self._pending_approvals[run_id][step_id] = request

        # Emit audit event
        await self._emit_audit_event(
            run_id=run_id,
            event_type=EventType.APPROVAL_REQUESTED,
            payload={
                "request_id": request_id,
                "step_id": step_id,
                "reason": reason,
                "expires_at": expires_at.isoformat() if expires_at else None,
            }
        )

        # Send callback notification if configured
        if run_config.approval_callback_url:
            await self._send_approval_callback(request, run_config.approval_callback_url)

        logger.info(f"Approval requested: {request_id} for step {step_id}")

        return request

    async def process_approval(
        self,
        request_id: str,
        approved: bool,
        approved_by: str,
        comment: Optional[str] = None,
    ) -> ApprovalRequest:
        """
        Process an approval decision.

        Args:
            request_id: Approval request ID
            approved: Whether approved or rejected
            approved_by: Identity of the approver
            comment: Optional comment

        Returns:
            Updated ApprovalRequest
        """
        # Find the request
        request = None
        for run_id, approvals in self._pending_approvals.items():
            for step_id, req in approvals.items():
                if req.request_id == request_id:
                    request = req
                    break

        if not request:
            raise ValueError(f"Approval request not found: {request_id}")

        # Check expiration
        if request.expires_at and DeterministicClock.now() > request.expires_at:
            request.status = ApprovalStatus.TIMEOUT
            raise ValueError(f"Approval request expired: {request_id}")

        # Update status
        request.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        request.approved_by = approved_by
        request.approved_at = DeterministicClock.now()
        if comment:
            request.metadata["approval_comment"] = comment

        # Emit audit event
        await self._emit_audit_event(
            run_id=request.run_id,
            event_type=EventType.APPROVAL_DECISION,
            payload={
                "request_id": request_id,
                "approved": approved,
                "approved_by": approved_by,
                "comment": comment,
            }
        )

        logger.info(f"Approval processed: {request_id} = {request.status.value}")

        return request

    async def wait_for_approval(
        self,
        request: ApprovalRequest,
        poll_interval_seconds: float = 5.0,
    ) -> bool:
        """
        Wait for an approval decision.

        Args:
            request: ApprovalRequest to wait for
            poll_interval_seconds: How often to check status

        Returns:
            True if approved, False if rejected/timeout
        """
        while request.status == ApprovalStatus.PENDING:
            # Check expiration
            if request.expires_at and DeterministicClock.now() > request.expires_at:
                request.status = ApprovalStatus.TIMEOUT
                return False

            await asyncio.sleep(poll_interval_seconds)

        return request.status == ApprovalStatus.APPROVED

    async def _send_approval_callback(
        self,
        request: ApprovalRequest,
        callback_url: str,
    ):
        """Send approval notification to callback URL."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(
                    callback_url,
                    json={
                        "type": "approval_required",
                        "request_id": request.request_id,
                        "run_id": request.run_id,
                        "step_id": request.step_id,
                        "reason": request.reason,
                        "expires_at": request.expires_at.isoformat() if request.expires_at else None,
                    },
                    timeout=10.0,
                )
        except Exception as e:
            logger.warning(f"Failed to send approval callback: {e}")

    # =========================================================================
    # DYNAMIC FAN-OUT (P1 Feature)
    # =========================================================================

    async def execute_fan_out(
        self,
        context: ExecutionContext,
        node: AgentNode,
        step: Dict[str, Any],
        inputs: Dict[str, Any],
        run_config: GLIPRunConfig,
        fan_out_spec: FanOutSpec,
    ) -> AgentResult:
        """
        Execute a step with dynamic fan-out.

        Creates parallel executions for each item in the source field,
        then aggregates results.

        Args:
            context: Execution context
            node: Agent node to execute
            step: Step definition
            inputs: Input data containing items to fan out
            run_config: Run configuration
            fan_out_spec: Fan-out specification

        Returns:
            Aggregated AgentResult
        """
        # Extract items to fan out
        source_data = inputs.get(fan_out_spec.source_field, [])
        if not isinstance(source_data, list):
            source_data = [source_data]

        # Validate limits
        if len(source_data) > run_config.max_fan_out_items:
            return AgentResult(
                success=False,
                error=f"Fan-out exceeds limit: {len(source_data)} > {run_config.max_fan_out_items}"
            )

        logger.info(f"Fan-out: {len(source_data)} items for step {step['step_id']}")

        # Create tasks for each item
        results = []
        errors = []

        # Process in batches
        for batch_start in range(0, len(source_data), fan_out_spec.max_parallel):
            batch_end = min(batch_start + fan_out_spec.max_parallel, len(source_data))
            batch = source_data[batch_start:batch_end]

            # Create tasks for this batch
            tasks = []
            for idx, item in enumerate(batch, start=batch_start):
                item_inputs = {**inputs, fan_out_spec.item_param_name: item}
                item_step = {
                    **step,
                    "step_id": f"{step['step_id']}-fanout-{idx}",
                    "idempotency_key": f"{step['idempotency_key']}-{idx}",
                }

                exec_mode = GLIPExecutionMode(step["execution_mode"])

                if exec_mode == GLIPExecutionMode.GLIP_V1:
                    task = self._execute_via_k8s(context, node, item_step, item_inputs, run_config)
                elif exec_mode == GLIPExecutionMode.LEGACY_HTTP:
                    task = self._execute_via_http(context, node, item_step, item_inputs, run_config)
                else:
                    task = self._execute_in_process(context, node, item_step, item_inputs)

                tasks.append(task)

            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for idx, result in enumerate(batch_results, start=batch_start):
                if isinstance(result, Exception):
                    errors.append({
                        "item_index": idx,
                        "error": str(result),
                    })
                    if not fan_out_spec.continue_on_failure:
                        return AgentResult(
                            success=False,
                            error=f"Fan-out item {idx} failed: {result}",
                            data={"partial_results": results, "errors": errors},
                        )
                elif result.success:
                    results.append({
                        "item_index": idx,
                        "data": result.data,
                    })
                else:
                    errors.append({
                        "item_index": idx,
                        "error": result.error,
                    })
                    if not fan_out_spec.continue_on_failure:
                        return AgentResult(
                            success=False,
                            error=result.error,
                            data={"partial_results": results, "errors": errors},
                        )

        # Aggregate results
        if fan_out_spec.aggregate_results:
            aggregated_data = {
                "items": results,
                "total_items": len(source_data),
                "successful_items": len(results),
                "failed_items": len(errors),
            }
            if errors:
                aggregated_data["errors"] = errors
        else:
            aggregated_data = results

        return AgentResult(
            success=len(errors) == 0 or fan_out_spec.continue_on_failure,
            data=aggregated_data,
        )

    # =========================================================================
    # CONCURRENCY CONTROLS (P1 Feature)
    # =========================================================================

    async def acquire_concurrency_slot(
        self,
        run_id: str,
        agent_id: str,
        tenant_id: str,
        concurrency_config: ConcurrencyConfig,
    ) -> bool:
        """
        Acquire a concurrency slot for execution.

        Implements fair scheduling and respects per-agent limits.

        Args:
            run_id: Run identifier
            agent_id: Agent being executed
            tenant_id: Tenant identifier
            concurrency_config: Concurrency configuration

        Returns:
            True if slot acquired, False if at capacity
        """
        # Check global concurrent runs
        if len(self._active_runs) >= concurrency_config.max_concurrent_runs:
            logger.warning(f"At max concurrent runs: {concurrency_config.max_concurrent_runs}")
            return False

        # Check per-agent limit
        agent_limit = concurrency_config.agent_concurrency_limits.get(agent_id)
        if agent_limit:
            current_agent_runs = sum(
                1 for run_info in self._active_runs.values()
                if any(
                    step.get("agent_type") == agent_id
                    for step in run_info.get("plan", {}).get("steps", [])
                )
            )
            if current_agent_runs >= agent_limit:
                logger.warning(f"Agent {agent_id} at concurrency limit: {agent_limit}")
                return False

        # Fair scheduling: check tenant quota
        if concurrency_config.enable_fair_scheduling:
            tenant_runs = sum(
                1 for run_info in self._active_runs.values()
                if run_info.get("config", GLIPRunConfig(pipeline_id="", tenant_id="")).tenant_id == tenant_id
            )
            # Fair share = max_runs / active_tenants (minimum 1)
            active_tenants = len(set(
                run_info.get("config", GLIPRunConfig(pipeline_id="", tenant_id="")).tenant_id
                for run_info in self._active_runs.values()
            ))
            fair_share = max(1, concurrency_config.max_concurrent_runs // max(1, active_tenants + 1))
            if tenant_runs >= fair_share * 2:  # Allow some burst
                logger.warning(f"Tenant {tenant_id} exceeds fair share: {tenant_runs} >= {fair_share * 2}")
                return False

        # Track slot
        if run_id not in self._concurrency_slots:
            self._concurrency_slots[run_id] = {
                "agent_id": agent_id,
                "tenant_id": tenant_id,
                "acquired_at": DeterministicClock.now(),
            }

        return True

    def release_concurrency_slot(self, run_id: str):
        """Release a concurrency slot."""
        if run_id in self._concurrency_slots:
            del self._concurrency_slots[run_id]
            logger.debug(f"Released concurrency slot for run {run_id}")

    async def wait_for_concurrency_slot(
        self,
        run_id: str,
        agent_id: str,
        tenant_id: str,
        concurrency_config: ConcurrencyConfig,
        poll_interval_seconds: float = 1.0,
    ) -> bool:
        """
        Wait for a concurrency slot to become available.

        Args:
            run_id: Run identifier
            agent_id: Agent being executed
            tenant_id: Tenant identifier
            concurrency_config: Concurrency configuration
            poll_interval_seconds: How often to check

        Returns:
            True if slot acquired, False if timeout
        """
        start_time = DeterministicClock.now()
        timeout = timedelta(seconds=concurrency_config.queue_timeout_seconds)

        while DeterministicClock.now() - start_time < timeout:
            if await self.acquire_concurrency_slot(run_id, agent_id, tenant_id, concurrency_config):
                return True
            await asyncio.sleep(poll_interval_seconds)

        logger.warning(f"Concurrency slot timeout for run {run_id}")
        return False

    def get_concurrency_stats(self) -> Dict[str, Any]:
        """Get current concurrency statistics."""
        return {
            "active_runs": len(self._active_runs),
            "active_slots": len(self._concurrency_slots),
            "runs_by_tenant": self._count_runs_by_tenant(),
            "runs_by_agent": self._count_runs_by_agent(),
        }

    def _count_runs_by_tenant(self) -> Dict[str, int]:
        """Count active runs by tenant."""
        counts: Dict[str, int] = {}
        for run_info in self._active_runs.values():
            tenant_id = run_info.get("config", GLIPRunConfig(pipeline_id="", tenant_id="")).tenant_id
            counts[tenant_id] = counts.get(tenant_id, 0) + 1
        return counts

    def _count_runs_by_agent(self) -> Dict[str, int]:
        """Count active runs by agent type."""
        counts: Dict[str, int] = {}
        for run_info in self._active_runs.values():
            for step in run_info.get("plan", {}).get("steps", []):
                agent_type = step.get("agent_type", "unknown")
                counts[agent_type] = counts.get(agent_type, 0) + 1
        return counts


# Factory function
def create_glip_orchestrator(
    config: Optional[GLIPOrchestratorConfig] = None,
) -> GLIPOrchestrator:
    """Create a GLIP v1 Orchestrator instance."""
    return GLIPOrchestrator(config)
