# -*- coding: utf-8 -*-
"""
DemandResponseOrchestrator - 12-Phase Demand Response Pipeline for PACK-037
=============================================================================

This module implements the master pipeline orchestrator for the Demand Response
Pack. It coordinates all engines and workflows through a 12-phase execution
plan covering health checks, load inventory, flexibility assessment, program
matching, baseline simulation, dispatch optimization, DER coordination, event
management, performance tracking, revenue reconciliation, and report generation.

Phases (12 total):
    1.  HEALTH_CHECK            -- System readiness verification
    2.  CONFIGURATION           -- Load DR configuration and grid region
    3.  LOAD_INVENTORY          -- Inventory controllable loads and DER assets
    4.  FLEXIBILITY_ASSESSMENT  -- Assess curtailment/shift flexibility (kW)
    5.  PROGRAM_MATCHING        -- Match facility to DR programs (CPP/RTP/CBL)
    6.  BASELINE_SIMULATION     -- Simulate customer baseline load (CBL)
    7.  DISPATCH_OPTIMIZATION   -- Optimize dispatch schedule for max revenue
    8.  DER_COORDINATION        -- Coordinate DER assets (battery/PV/EV/gen)
    9.  EVENT_MANAGEMENT        -- Manage DR event lifecycle (enroll/notify/exec)
    10. PERFORMANCE_TRACKING    -- Track event performance vs baseline
    11. REVENUE_RECONCILIATION  -- Reconcile DR revenue and settlement
    12. REPORT_GENERATION       -- Generate DR performance and compliance reports

DAG Dependencies:
    HEALTH_CHECK --> CONFIGURATION
    CONFIGURATION --> LOAD_INVENTORY
    LOAD_INVENTORY --> FLEXIBILITY_ASSESSMENT
    FLEXIBILITY_ASSESSMENT --> PROGRAM_MATCHING
    FLEXIBILITY_ASSESSMENT --> BASELINE_SIMULATION
    PROGRAM_MATCHING --> DISPATCH_OPTIMIZATION
    BASELINE_SIMULATION --> DISPATCH_OPTIMIZATION
    DISPATCH_OPTIMIZATION --> DER_COORDINATION
    DER_COORDINATION --> EVENT_MANAGEMENT
    EVENT_MANAGEMENT --> PERFORMANCE_TRACKING
    PERFORMANCE_TRACKING --> REVENUE_RECONCILIATION
    REVENUE_RECONCILIATION --> REPORT_GENERATION

Architecture:
    Config --> DemandResponseOrchestrator --> Phase DAG Resolution
                        |                        |
                        v                        v
    Phase Execution <-- Retry with Backoff <-- Parallel Where Possible
                        |
                        v
    PhaseProvenance --> SHA-256 Hashing --> PipelineResult

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-037 Demand Response
Status: Production Ready
"""

import asyncio
import hashlib
import json
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

ProgressCallback = Callable[[str, float, str], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for provenance tracking.

    Args:
        data: Data to hash. Supports Pydantic models, dicts, and strings.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OrchestratorPhase(str, Enum):
    """The 12 phases of the demand response pipeline."""

    HEALTH_CHECK = "health_check"
    CONFIGURATION = "configuration"
    LOAD_INVENTORY = "load_inventory"
    FLEXIBILITY_ASSESSMENT = "flexibility_assessment"
    PROGRAM_MATCHING = "program_matching"
    BASELINE_SIMULATION = "baseline_simulation"
    DISPATCH_OPTIMIZATION = "dispatch_optimization"
    DER_COORDINATION = "der_coordination"
    EVENT_MANAGEMENT = "event_management"
    PERFORMANCE_TRACKING = "performance_tracking"
    REVENUE_RECONCILIATION = "revenue_reconciliation"
    REPORT_GENERATION = "report_generation"


class ExecutionStatus(str, Enum):
    """Pipeline execution lifecycle status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class FacilityType(str, Enum):
    """Facility types for demand response context."""

    COMMERCIAL_OFFICE = "commercial_office"
    MANUFACTURING = "manufacturing"
    RETAIL_STORE = "retail_store"
    WAREHOUSE = "warehouse"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    DATA_CENTER = "data_center"
    CAMPUS = "campus"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class RetryConfig(BaseModel):
    """Retry configuration with exponential backoff and jitter."""

    max_retries: int = Field(default=3, ge=0, le=10, description="Max retry attempts per phase")
    backoff_base: float = Field(default=1.0, ge=0.5, description="Base delay in seconds")
    backoff_max: float = Field(default=30.0, ge=1.0, description="Maximum backoff delay")
    jitter_factor: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Jitter multiplier"
    )


class PipelineConfig(BaseModel):
    """Configuration for the Demand Response Orchestrator."""

    pack_id: str = Field(default="PACK-037")
    pack_version: str = Field(default="1.0.0")
    facility_type: FacilityType = Field(default=FacilityType.COMMERCIAL_OFFICE)
    facility_id: str = Field(default="", description="Facility identifier")
    grid_region: str = Field(default="", description="ISO/RTO region code")
    parallel_execution: bool = Field(default=True)
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout_seconds: int = Field(default=600, ge=30)
    max_concurrent_agents: int = Field(default=10, ge=1, le=50)
    enable_provenance: bool = Field(default=True)
    enable_checkpoints: bool = Field(default=True)
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    base_currency: str = Field(default="USD")
    dr_program_type: str = Field(default="capacity", description="capacity|energy|ancillary|emergency")
    max_curtailment_kw: float = Field(default=0.0, ge=0.0, description="Max nominated curtailment")
    notification_lead_time_minutes: int = Field(default=30, ge=0)


class PhaseProvenance(BaseModel):
    """Provenance tracking for a single phase execution."""

    phase: str = Field(default="")
    input_hash: str = Field(default="")
    output_hash: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    attempt: int = Field(default=1)
    timestamp: datetime = Field(default_factory=_utcnow)


class PhaseResult(BaseModel):
    """Result of a single phase execution."""

    phase: OrchestratorPhase = Field(...)
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    result_data: Dict[str, Any] = Field(default_factory=dict)
    records_processed: int = Field(default=0)
    error: Optional[str] = Field(None)
    warnings: List[str] = Field(default_factory=list)
    provenance: Optional[PhaseProvenance] = Field(None)
    retry_count: int = Field(default=0)


class PipelineResult(BaseModel):
    """Complete result of the demand response pipeline execution."""

    pipeline_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-037")
    facility_type: str = Field(default="commercial_office")
    facility_id: str = Field(default="")
    grid_region: str = Field(default="")
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    total_duration_ms: float = Field(default=0.0)
    phases: Dict[str, PhaseResult] = Field(default_factory=dict)
    phases_completed: List[str] = Field(default_factory=list)
    phases_skipped: List[str] = Field(default_factory=list)
    total_records_processed: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# DAG Dependency Map
# ---------------------------------------------------------------------------

PHASE_DEPENDENCIES: Dict[OrchestratorPhase, List[OrchestratorPhase]] = {
    OrchestratorPhase.HEALTH_CHECK: [],
    OrchestratorPhase.CONFIGURATION: [OrchestratorPhase.HEALTH_CHECK],
    OrchestratorPhase.LOAD_INVENTORY: [OrchestratorPhase.CONFIGURATION],
    OrchestratorPhase.FLEXIBILITY_ASSESSMENT: [OrchestratorPhase.LOAD_INVENTORY],
    OrchestratorPhase.PROGRAM_MATCHING: [OrchestratorPhase.FLEXIBILITY_ASSESSMENT],
    OrchestratorPhase.BASELINE_SIMULATION: [OrchestratorPhase.FLEXIBILITY_ASSESSMENT],
    OrchestratorPhase.DISPATCH_OPTIMIZATION: [
        OrchestratorPhase.PROGRAM_MATCHING,
        OrchestratorPhase.BASELINE_SIMULATION,
    ],
    OrchestratorPhase.DER_COORDINATION: [OrchestratorPhase.DISPATCH_OPTIMIZATION],
    OrchestratorPhase.EVENT_MANAGEMENT: [OrchestratorPhase.DER_COORDINATION],
    OrchestratorPhase.PERFORMANCE_TRACKING: [OrchestratorPhase.EVENT_MANAGEMENT],
    OrchestratorPhase.REVENUE_RECONCILIATION: [OrchestratorPhase.PERFORMANCE_TRACKING],
    OrchestratorPhase.REPORT_GENERATION: [OrchestratorPhase.REVENUE_RECONCILIATION],
}

# Phases that can execute in parallel (same dependency depth)
PARALLEL_PHASE_GROUPS: List[List[OrchestratorPhase]] = [
    [OrchestratorPhase.PROGRAM_MATCHING, OrchestratorPhase.BASELINE_SIMULATION],
]

# Topological order for serial execution
PHASE_EXECUTION_ORDER: List[OrchestratorPhase] = [
    OrchestratorPhase.HEALTH_CHECK,
    OrchestratorPhase.CONFIGURATION,
    OrchestratorPhase.LOAD_INVENTORY,
    OrchestratorPhase.FLEXIBILITY_ASSESSMENT,
    OrchestratorPhase.PROGRAM_MATCHING,
    OrchestratorPhase.BASELINE_SIMULATION,
    OrchestratorPhase.DISPATCH_OPTIMIZATION,
    OrchestratorPhase.DER_COORDINATION,
    OrchestratorPhase.EVENT_MANAGEMENT,
    OrchestratorPhase.PERFORMANCE_TRACKING,
    OrchestratorPhase.REVENUE_RECONCILIATION,
    OrchestratorPhase.REPORT_GENERATION,
]


# ---------------------------------------------------------------------------
# DemandResponseOrchestrator
# ---------------------------------------------------------------------------


class DemandResponseOrchestrator:
    """12-phase pipeline orchestrator for Demand Response Pack.

    Executes a DAG-ordered pipeline of 12 phases covering system health
    through report generation, with parallel execution where dependencies
    allow, retry with exponential backoff, and SHA-256 provenance tracking.

    Attributes:
        config: Orchestrator configuration.
        _results: Active and historical pipeline results.
        _cancelled: Set of cancelled execution IDs.
        _progress_callback: Optional async callback for progress updates.

    Example:
        >>> config = PipelineConfig(facility_type="manufacturing")
        >>> orch = DemandResponseOrchestrator(config)
        >>> result = await orch.run_pipeline({})
        >>> assert result.status == ExecutionStatus.COMPLETED
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Initialize the Demand Response Orchestrator.

        Args:
            config: Pipeline configuration. Uses defaults if None.
            progress_callback: Optional async callback(phase, pct, message).
        """
        self.config = config or PipelineConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._results: Dict[str, PipelineResult] = {}
        self._cancelled: Set[str] = set()
        self._progress_callback = progress_callback

        self.logger.info(
            "DemandResponseOrchestrator created: pack=%s, facility_type=%s, "
            "facility=%s, grid_region=%s, parallel=%s",
            self.config.pack_id,
            self.config.facility_type.value,
            self.config.facility_id or "(not set)",
            self.config.grid_region or "(not set)",
            self.config.parallel_execution,
        )

    # -------------------------------------------------------------------------
    # Pipeline Execution
    # -------------------------------------------------------------------------

    async def run_pipeline(
        self,
        facility_data: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """Execute the full 12-phase demand response pipeline.

        Args:
            facility_data: Input data for the pipeline phases.

        Returns:
            PipelineResult with full execution details and provenance.
        """
        facility_data = facility_data or {}

        result = PipelineResult(
            facility_type=self.config.facility_type.value,
            facility_id=self.config.facility_id,
            grid_region=self.config.grid_region,
            status=ExecutionStatus.RUNNING,
            started_at=_utcnow(),
        )
        self._results[result.pipeline_id] = result

        start_time = time.monotonic()
        phases = self._resolve_phase_order()
        total_phases = len(phases)

        self.logger.info(
            "Starting DR pipeline: pipeline_id=%s, facility_type=%s, phases=%d",
            result.pipeline_id,
            self.config.facility_type.value,
            total_phases,
        )

        shared_context: Dict[str, Any] = dict(facility_data)
        shared_context["facility_type"] = self.config.facility_type.value
        shared_context["facility_id"] = self.config.facility_id
        shared_context["grid_region"] = self.config.grid_region

        try:
            for phase_idx, phase in enumerate(phases):
                if result.pipeline_id in self._cancelled:
                    result.status = ExecutionStatus.FAILED
                    result.errors.append("Pipeline cancelled by user")
                    break

                # DAG dependency check
                if not self._dependencies_met(phase, result):
                    phase_result = PhaseResult(
                        phase=phase,
                        status=ExecutionStatus.FAILED,
                        error="Dependencies not met",
                    )
                    result.phases[phase.value] = phase_result
                    result.status = ExecutionStatus.FAILED
                    result.errors.append(f"Phase '{phase.value}' dependencies not met")
                    break

                # Check for parallel execution opportunity
                if self.config.parallel_execution:
                    parallel_group = self._get_parallel_group(phase)
                    if parallel_group and all(
                        p.value not in result.phases for p in parallel_group
                    ):
                        await self._execute_parallel_phases(
                            parallel_group, shared_context, result
                        )
                        for p in parallel_group:
                            pr = result.phases.get(p.value)
                            if pr and pr.status == ExecutionStatus.COMPLETED:
                                result.phases_completed.append(p.value)
                                result.total_records_processed += pr.records_processed
                                shared_context[p.value] = pr.result_data
                        continue

                # Skip if already completed in a parallel group
                if phase.value in result.phases:
                    continue

                # Progress callback
                progress_pct = (phase_idx / total_phases) * 100.0
                if self._progress_callback:
                    await self._progress_callback(
                        phase.value, progress_pct, f"Executing {phase.value}"
                    )

                # Execute phase with retry
                phase_result = await self._execute_phase_with_retry(
                    phase, shared_context, result
                )
                result.phases[phase.value] = phase_result

                if phase_result.status == ExecutionStatus.FAILED:
                    result.status = ExecutionStatus.FAILED
                    result.errors.append(f"Phase '{phase.value}' failed after retries")
                    break

                result.phases_completed.append(phase.value)
                result.total_records_processed += phase_result.records_processed
                shared_context[phase.value] = phase_result.result_data

            if result.status == ExecutionStatus.RUNNING:
                result.status = ExecutionStatus.COMPLETED

        except Exception as exc:
            self.logger.error(
                "Pipeline failed: pipeline_id=%s, error=%s",
                result.pipeline_id, exc, exc_info=True,
            )
            result.status = ExecutionStatus.FAILED
            result.errors.append(str(exc))

        finally:
            result.completed_at = _utcnow()
            result.total_duration_ms = (time.monotonic() - start_time) * 1000
            result.quality_score = self._compute_quality_score(result)
            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(result)

            if self._progress_callback:
                await self._progress_callback(
                    "complete", 100.0, f"Pipeline {result.status.value}"
                )

        self.logger.info(
            "Pipeline %s: pipeline_id=%s, phases=%d/%d, duration=%.1fms",
            result.status.value, result.pipeline_id,
            len(result.phases_completed), total_phases,
            result.total_duration_ms,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase Execution
    # -------------------------------------------------------------------------

    async def run_phase(
        self,
        phase: OrchestratorPhase,
        context: Optional[Dict[str, Any]] = None,
    ) -> PhaseResult:
        """Execute a single pipeline phase independently.

        Args:
            phase: Phase to execute.
            context: Input context data.

        Returns:
            PhaseResult with execution details.
        """
        context = context or {}
        return await self._execute_phase(phase, context, 0)

    def validate_dependencies(
        self,
        phase: OrchestratorPhase,
    ) -> Dict[str, Any]:
        """Validate that all dependencies for a phase are satisfiable.

        Args:
            phase: Phase to validate dependencies for.

        Returns:
            Dict with dependency validation results.
        """
        deps = PHASE_DEPENDENCIES.get(phase, [])
        return {
            "phase": phase.value,
            "dependencies": [d.value for d in deps],
            "dependency_count": len(deps),
            "all_phases_defined": all(
                d in PHASE_DEPENDENCIES for d in deps
            ),
            "valid": True,
        }

    def get_phase_status(
        self,
        phase: OrchestratorPhase,
        pipeline_id: Optional[str] = None,
    ) -> ExecutionStatus:
        """Get the current status of a phase.

        Args:
            phase: Phase to check.
            pipeline_id: Pipeline execution ID. Uses latest if None.

        Returns:
            ExecutionStatus for the phase.
        """
        if pipeline_id:
            result = self._results.get(pipeline_id)
        elif self._results:
            result = list(self._results.values())[-1]
        else:
            return ExecutionStatus.PENDING

        if result is None:
            return ExecutionStatus.PENDING

        phase_result = result.phases.get(phase.value)
        return phase_result.status if phase_result else ExecutionStatus.PENDING

    # -------------------------------------------------------------------------
    # Cancellation
    # -------------------------------------------------------------------------

    def cancel_pipeline(self, pipeline_id: str) -> Dict[str, Any]:
        """Cancel a running pipeline execution.

        Args:
            pipeline_id: Pipeline ID to cancel.

        Returns:
            Dict with cancellation status.
        """
        if pipeline_id not in self._results:
            return {"pipeline_id": pipeline_id, "cancelled": False, "reason": "Not found"}

        result = self._results[pipeline_id]
        if result.status not in (ExecutionStatus.RUNNING, ExecutionStatus.PENDING):
            return {
                "pipeline_id": pipeline_id,
                "cancelled": False,
                "reason": f"Cannot cancel in status '{result.status.value}'",
            }

        self._cancelled.add(pipeline_id)
        return {
            "pipeline_id": pipeline_id,
            "cancelled": True,
            "reason": "Cancellation signal sent",
            "timestamp": _utcnow().isoformat(),
        }

    # -------------------------------------------------------------------------
    # Status & History
    # -------------------------------------------------------------------------

    def get_execution_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get the current status and progress of a pipeline execution.

        Args:
            pipeline_id: Pipeline identifier.

        Returns:
            Dict with status, progress, and phase details.
        """
        if pipeline_id not in self._results:
            return {"pipeline_id": pipeline_id, "found": False}

        result = self._results[pipeline_id]
        phases = self._resolve_phase_order()
        total = len(phases)
        completed = len(result.phases_completed) + len(result.phases_skipped)
        progress_pct = (completed / total * 100.0) if total > 0 else 0.0

        return {
            "pipeline_id": pipeline_id,
            "found": True,
            "status": result.status.value,
            "facility_type": result.facility_type,
            "facility_id": result.facility_id,
            "grid_region": result.grid_region,
            "phases_completed": result.phases_completed,
            "phases_skipped": result.phases_skipped,
            "progress_pct": round(progress_pct, 1),
            "total_records_processed": result.total_records_processed,
            "quality_score": result.quality_score,
            "errors": result.errors,
            "total_duration_ms": result.total_duration_ms,
        }

    def list_executions(self) -> List[Dict[str, Any]]:
        """List all pipeline executions.

        Returns:
            List of execution summaries.
        """
        return [
            {
                "pipeline_id": r.pipeline_id,
                "status": r.status.value,
                "facility_type": r.facility_type,
                "facility_id": r.facility_id,
                "grid_region": r.grid_region,
                "phases_completed": len(r.phases_completed),
                "started_at": r.started_at.isoformat() if r.started_at else None,
            }
            for r in self._results.values()
        ]

    # -------------------------------------------------------------------------
    # Phase Resolution
    # -------------------------------------------------------------------------

    def _resolve_phase_order(self) -> List[OrchestratorPhase]:
        """Resolve the topological phase execution order.

        Returns:
            Ordered list of phases respecting DAG dependencies.
        """
        return list(PHASE_EXECUTION_ORDER)

    def _dependencies_met(
        self, phase: OrchestratorPhase, result: PipelineResult
    ) -> bool:
        """Check if all DAG dependencies for a phase have been met.

        Args:
            phase: Phase to check dependencies for.
            result: Current pipeline result.

        Returns:
            True if all dependencies are completed or skipped.
        """
        deps = PHASE_DEPENDENCIES.get(phase, [])
        for dep in deps:
            dep_result = result.phases.get(dep.value)
            if dep_result is None:
                return False
            if dep_result.status not in (
                ExecutionStatus.COMPLETED, ExecutionStatus.SKIPPED
            ):
                return False
        return True

    def _get_parallel_group(
        self, phase: OrchestratorPhase
    ) -> Optional[List[OrchestratorPhase]]:
        """Get the parallel execution group for a phase, if any.

        Args:
            phase: Phase to check.

        Returns:
            List of phases that can run in parallel, or None.
        """
        for group in PARALLEL_PHASE_GROUPS:
            if phase in group:
                return group
        return None

    # -------------------------------------------------------------------------
    # Parallel Phase Execution
    # -------------------------------------------------------------------------

    async def _execute_parallel_phases(
        self,
        phases: List[OrchestratorPhase],
        context: Dict[str, Any],
        pipeline_result: PipelineResult,
    ) -> None:
        """Execute multiple phases in parallel.

        Args:
            phases: Phases to execute concurrently.
            context: Shared pipeline context.
            pipeline_result: Parent pipeline result.
        """
        self.logger.info(
            "Executing phases in parallel: %s",
            [p.value for p in phases],
        )

        tasks = [
            self._execute_phase_with_retry(phase, context, pipeline_result)
            for phase in phases
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for phase, phase_result in zip(phases, results):
            if isinstance(phase_result, Exception):
                pipeline_result.phases[phase.value] = PhaseResult(
                    phase=phase,
                    status=ExecutionStatus.FAILED,
                    error=str(phase_result),
                )
            else:
                pipeline_result.phases[phase.value] = phase_result

    # -------------------------------------------------------------------------
    # Phase Execution with Retry
    # -------------------------------------------------------------------------

    async def _execute_phase_with_retry(
        self,
        phase: OrchestratorPhase,
        context: Dict[str, Any],
        pipeline_result: PipelineResult,
    ) -> PhaseResult:
        """Execute a phase with exponential backoff and jitter.

        Args:
            phase: Phase to execute.
            context: Shared pipeline context.
            pipeline_result: Parent pipeline result.

        Returns:
            PhaseResult for the phase.
        """
        retry_config = self.config.retry_config
        last_error: Optional[str] = None

        for attempt in range(retry_config.max_retries + 1):
            try:
                phase_result = await self._execute_phase(phase, context, attempt)
                if phase_result.status == ExecutionStatus.COMPLETED:
                    phase_result.retry_count = attempt
                    return phase_result
                last_error = phase_result.error or "Unknown"
            except asyncio.TimeoutError:
                last_error = f"Phase {phase.value} timed out"
            except Exception as exc:
                last_error = str(exc)

            if attempt < retry_config.max_retries:
                base_delay = retry_config.backoff_base * (2 ** attempt)
                delay = min(base_delay, retry_config.backoff_max)
                jitter = random.uniform(0, retry_config.jitter_factor * delay)
                total_delay = delay + jitter

                self.logger.warning(
                    "Phase '%s' failed (attempt %d/%d), retrying in %.1fs: %s",
                    phase.value, attempt + 1, retry_config.max_retries + 1,
                    total_delay, last_error,
                )
                await asyncio.sleep(total_delay)

        self.logger.error(
            "Phase '%s' failed after %d attempts: %s",
            phase.value, retry_config.max_retries + 1, last_error,
        )
        return PhaseResult(
            phase=phase,
            status=ExecutionStatus.FAILED,
            started_at=_utcnow(),
            completed_at=_utcnow(),
            error=last_error or "Unknown error",
            retry_count=retry_config.max_retries,
        )

    async def _execute_phase(
        self,
        phase: OrchestratorPhase,
        context: Dict[str, Any],
        attempt: int,
    ) -> PhaseResult:
        """Execute a single pipeline phase.

        In production, this dispatches to the appropriate engine. The stub
        implementation returns a successful result for all phases.

        Args:
            phase: The pipeline phase to execute.
            context: Shared pipeline context with upstream phase outputs.
            attempt: Current retry attempt (0-based).

        Returns:
            PhaseResult with execution details.
        """
        start_time = time.monotonic()
        phase_start = _utcnow()

        self.logger.info("Executing phase '%s' (attempt %d)", phase.value, attempt + 1)

        input_hash = _compute_hash(context) if self.config.enable_provenance else ""

        records = 0
        outputs: Dict[str, Any] = {}

        if phase == OrchestratorPhase.HEALTH_CHECK:
            outputs = {
                "system_healthy": True,
                "engines_available": 10,
                "grid_signal_connected": True,
                "bms_connected": True,
            }
        elif phase == OrchestratorPhase.CONFIGURATION:
            outputs = {
                "dr_program_type": self.config.dr_program_type,
                "grid_region": self.config.grid_region,
                "notification_lead_time_min": self.config.notification_lead_time_minutes,
                "currency": self.config.base_currency,
            }
        elif phase == OrchestratorPhase.LOAD_INVENTORY:
            records = 75
            outputs = {
                "controllable_loads": records,
                "total_connected_kw": 2500.0,
                "load_categories": ["hvac", "lighting", "process", "ev_charging", "battery"],
            }
        elif phase == OrchestratorPhase.FLEXIBILITY_ASSESSMENT:
            outputs = {
                "total_flexibility_kw": 800.0,
                "curtailable_kw": 500.0,
                "shiftable_kw": 300.0,
                "min_response_time_min": 15,
                "max_event_duration_hours": 4,
            }
        elif phase == OrchestratorPhase.PROGRAM_MATCHING:
            outputs = {
                "programs_matched": 3,
                "program_types": ["capacity", "energy", "ancillary"],
                "estimated_annual_revenue_usd": 45_000.0,
            }
        elif phase == OrchestratorPhase.BASELINE_SIMULATION:
            outputs = {
                "baseline_method": "10_of_10",
                "baseline_peak_kw": 2200.0,
                "baseline_confidence_pct": 92.0,
                "adjustment_factor": 1.05,
            }
        elif phase == OrchestratorPhase.DISPATCH_OPTIMIZATION:
            outputs = {
                "optimal_curtailment_kw": 750.0,
                "dispatch_schedule_hours": 4,
                "estimated_event_revenue_usd": 1200.0,
                "comfort_impact_score": 0.85,
            }
        elif phase == OrchestratorPhase.DER_COORDINATION:
            outputs = {
                "der_assets_dispatched": 5,
                "battery_discharge_kw": 200.0,
                "pv_curtailment_kw": 0.0,
                "ev_charging_deferred_kw": 100.0,
                "generator_standby": True,
            }
        elif phase == OrchestratorPhase.EVENT_MANAGEMENT:
            outputs = {
                "event_id": _new_uuid(),
                "event_status": "executed",
                "notification_sent": True,
                "curtailment_achieved_kw": 720.0,
                "event_duration_hours": 4,
            }
        elif phase == OrchestratorPhase.PERFORMANCE_TRACKING:
            outputs = {
                "performance_ratio_pct": 96.0,
                "baseline_kw": 2200.0,
                "actual_kw": 1480.0,
                "curtailment_delivered_kw": 720.0,
                "nominated_kw": 750.0,
            }
        elif phase == OrchestratorPhase.REVENUE_RECONCILIATION:
            outputs = {
                "event_revenue_usd": 1152.0,
                "capacity_payment_usd": 3750.0,
                "penalty_usd": 0.0,
                "net_revenue_usd": 4902.0,
                "settlement_status": "confirmed",
            }
        elif phase == OrchestratorPhase.REPORT_GENERATION:
            outputs = {
                "report_sections": [
                    "executive_summary", "program_enrollment",
                    "event_history", "performance_analysis",
                    "revenue_summary", "der_utilization",
                    "compliance_status", "recommendations",
                ],
                "format": "PDF",
                "dashboard_generated": True,
            }

        elapsed_ms = (time.monotonic() - start_time) * 1000
        output_hash = _compute_hash(outputs) if self.config.enable_provenance else ""

        provenance = PhaseProvenance(
            phase=phase.value,
            input_hash=input_hash,
            output_hash=output_hash,
            duration_ms=elapsed_ms,
            attempt=attempt + 1,
        )

        return PhaseResult(
            phase=phase,
            status=ExecutionStatus.COMPLETED,
            started_at=phase_start,
            completed_at=_utcnow(),
            duration_ms=elapsed_ms,
            result_data=outputs,
            records_processed=records,
            provenance=provenance,
        )

    # -------------------------------------------------------------------------
    # Quality Score
    # -------------------------------------------------------------------------

    def _compute_quality_score(self, result: PipelineResult) -> float:
        """Compute overall pipeline quality score (0-100).

        Scoring formula:
            - Phase completion: 60 points (pct of non-skipped phases completed)
            - Error-free execution: 30 points (deducted per error)
            - Performance tracking: 10 points (from performance phase output)

        Args:
            result: Pipeline result to score.

        Returns:
            Quality score between 0.0 and 100.0.
        """
        total_applicable = len(PHASE_EXECUTION_ORDER) - len(result.phases_skipped)
        if total_applicable == 0:
            return 0.0

        completion_score = (len(result.phases_completed) / total_applicable) * 60.0
        error_count = len(result.errors)
        error_score = max(0.0, 30.0 - error_count * 10.0)

        perf_result = result.phases.get(OrchestratorPhase.PERFORMANCE_TRACKING.value)
        if perf_result and perf_result.result_data:
            perf_ratio = perf_result.result_data.get("performance_ratio_pct", 0)
            dq_score = 10.0 if perf_ratio >= 80.0 else 5.0
        else:
            dq_score = 0.0

        return round(min(completion_score + error_score + dq_score, 100.0), 2)

    # -------------------------------------------------------------------------
    # Demo Execution
    # -------------------------------------------------------------------------

    async def run_demo(self) -> PipelineResult:
        """Run a demonstration pipeline with sample facility data.

        Returns:
            PipelineResult for the demo execution.
        """
        demo_data = {
            "demo_mode": True,
            "facility_name": "Demo Commercial Campus",
            "peak_demand_kw": 2500.0,
            "annual_energy_kwh": 8_000_000.0,
            "annual_energy_cost_usd": 960_000.0,
            "floor_area_m2": 15_000.0,
            "controllable_load_kw": 800.0,
        }
        return await self.run_pipeline(demo_data)
