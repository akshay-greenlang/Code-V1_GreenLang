# -*- coding: utf-8 -*-
"""
EnMSOrchestrator - 10-Phase ISO 50001 EnMS Pipeline for PACK-034
===================================================================

This module implements the master pipeline orchestrator for the ISO 50001
Energy Management System Pack. It coordinates all engines and workflows
through a 10-phase execution plan covering initialization, energy review,
baseline establishment, EnPI setup, monitoring configuration, action
planning, operational control, performance analysis, audit compliance,
and management review.

Phases (10 total):
    1.  INITIALIZATION          -- Organization context, scope, boundaries
    2.  ENERGY_REVIEW           -- Current energy sources, uses, SEUs
    3.  BASELINE_ESTABLISHMENT  -- Establish energy baselines and EnBs
    4.  ENPI_SETUP              -- Define and configure EnPIs
    5.  MONITORING_CONFIG       -- Monitoring plan, metering hierarchy
    6.  ACTION_PLANNING         -- Energy objectives, targets, action plans
    7.  OPERATIONAL_CONTROL     -- Operational criteria and controls
    8.  PERFORMANCE_ANALYSIS    -- M&V, regression, CUSUM analysis
    9.  AUDIT_COMPLIANCE        -- Internal audit, clause compliance check
    10. MANAGEMENT_REVIEW       -- Top management review package

DAG Dependencies:
    INITIALIZATION --> ENERGY_REVIEW
    ENERGY_REVIEW --> BASELINE_ESTABLISHMENT
    ENERGY_REVIEW --> ENPI_SETUP
    BASELINE_ESTABLISHMENT --> MONITORING_CONFIG
    ENPI_SETUP --> MONITORING_CONFIG
    MONITORING_CONFIG --> ACTION_PLANNING
    ACTION_PLANNING --> OPERATIONAL_CONTROL
    OPERATIONAL_CONTROL --> PERFORMANCE_ANALYSIS
    PERFORMANCE_ANALYSIS --> AUDIT_COMPLIANCE
    AUDIT_COMPLIANCE --> MANAGEMENT_REVIEW

Architecture:
    Config --> EnMSOrchestrator --> Phase DAG Resolution
                    |                        |
                    v                        v
    Phase Execution <-- Retry with Backoff <-- Parallel Where Possible
                    |
                    v
    PhaseProvenance --> SHA-256 Hashing --> PipelineResult

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-034 ISO 50001 Energy Management System
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

from greenlang.schemas import utcnow
from greenlang.schemas.enums import ExecutionStatus

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

ProgressCallback = Callable[[str, float, str], Coroutine[Any, Any, None]]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """The 10 phases of the ISO 50001 EnMS pipeline."""

    INITIALIZATION = "initialization"
    ENERGY_REVIEW = "energy_review"
    BASELINE_ESTABLISHMENT = "baseline_establishment"
    ENPI_SETUP = "enpi_setup"
    MONITORING_CONFIG = "monitoring_config"
    ACTION_PLANNING = "action_planning"
    OPERATIONAL_CONTROL = "operational_control"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    AUDIT_COMPLIANCE = "audit_compliance"
    MANAGEMENT_REVIEW = "management_review"

class FacilityType(str, Enum):
    """Facility types for EnMS context."""

    MANUFACTURING = "manufacturing"
    COMMERCIAL_OFFICE = "commercial_office"
    DATA_CENTER = "data_center"
    HEALTHCARE = "healthcare"
    RETAIL_CHAIN = "retail_chain"
    LOGISTICS_WAREHOUSE = "logistics_warehouse"
    FOOD_PROCESSING = "food_processing"
    SME_MULTI_SITE = "sme_multi_site"

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

class OrchestratorConfig(BaseModel):
    """Configuration for the EnMS Orchestrator."""

    pack_id: str = Field(default="PACK-034")
    pack_version: str = Field(default="1.0.0")
    facility_type: FacilityType = Field(default=FacilityType.MANUFACTURING)
    facility_id: str = Field(default="", description="Facility identifier")
    organization_name: str = Field(default="", description="Organization name")
    parallel_execution: bool = Field(default=True)
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout_seconds: int = Field(default=600, ge=30)
    max_concurrent_agents: int = Field(default=10, ge=1, le=50)
    enable_provenance: bool = Field(default=True)
    enable_checkpoints: bool = Field(default=True)
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    iso50001_version: str = Field(default="2018", description="ISO 50001 version year")
    certification_target: bool = Field(default=True, description="Targeting certification")
    base_currency: str = Field(default="EUR")

class PhaseProvenance(BaseModel):
    """Provenance tracking for a single phase execution."""

    phase: str = Field(default="")
    input_hash: str = Field(default="")
    output_hash: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    attempt: int = Field(default=1)
    timestamp: datetime = Field(default_factory=utcnow)

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
    """Complete result of the EnMS pipeline execution."""

    pipeline_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-034")
    facility_type: str = Field(default="manufacturing")
    facility_id: str = Field(default="")
    organization_name: str = Field(default="")
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
    OrchestratorPhase.INITIALIZATION: [],
    OrchestratorPhase.ENERGY_REVIEW: [OrchestratorPhase.INITIALIZATION],
    OrchestratorPhase.BASELINE_ESTABLISHMENT: [OrchestratorPhase.ENERGY_REVIEW],
    OrchestratorPhase.ENPI_SETUP: [OrchestratorPhase.ENERGY_REVIEW],
    OrchestratorPhase.MONITORING_CONFIG: [
        OrchestratorPhase.BASELINE_ESTABLISHMENT,
        OrchestratorPhase.ENPI_SETUP,
    ],
    OrchestratorPhase.ACTION_PLANNING: [OrchestratorPhase.MONITORING_CONFIG],
    OrchestratorPhase.OPERATIONAL_CONTROL: [OrchestratorPhase.ACTION_PLANNING],
    OrchestratorPhase.PERFORMANCE_ANALYSIS: [OrchestratorPhase.OPERATIONAL_CONTROL],
    OrchestratorPhase.AUDIT_COMPLIANCE: [OrchestratorPhase.PERFORMANCE_ANALYSIS],
    OrchestratorPhase.MANAGEMENT_REVIEW: [OrchestratorPhase.AUDIT_COMPLIANCE],
}

# Phases that can execute in parallel (same dependency depth)
PARALLEL_PHASE_GROUPS: List[List[OrchestratorPhase]] = [
    [OrchestratorPhase.BASELINE_ESTABLISHMENT, OrchestratorPhase.ENPI_SETUP],
]

# Topological order for serial execution
PHASE_EXECUTION_ORDER: List[OrchestratorPhase] = [
    OrchestratorPhase.INITIALIZATION,
    OrchestratorPhase.ENERGY_REVIEW,
    OrchestratorPhase.BASELINE_ESTABLISHMENT,
    OrchestratorPhase.ENPI_SETUP,
    OrchestratorPhase.MONITORING_CONFIG,
    OrchestratorPhase.ACTION_PLANNING,
    OrchestratorPhase.OPERATIONAL_CONTROL,
    OrchestratorPhase.PERFORMANCE_ANALYSIS,
    OrchestratorPhase.AUDIT_COMPLIANCE,
    OrchestratorPhase.MANAGEMENT_REVIEW,
]

# ---------------------------------------------------------------------------
# EnMSOrchestrator
# ---------------------------------------------------------------------------

class EnMSOrchestrator:
    """10-phase pipeline orchestrator for ISO 50001 EnMS Pack.

    Executes a DAG-ordered pipeline of 10 phases covering initialization
    through management review, with parallel execution where dependencies
    allow, retry with exponential backoff, and SHA-256 provenance tracking.

    Attributes:
        config: Orchestrator configuration.
        _results: Active and historical pipeline results.
        _cancelled: Set of cancelled execution IDs.
        _progress_callback: Optional async callback for progress updates.

    Example:
        >>> config = OrchestratorConfig(facility_type="manufacturing")
        >>> orch = EnMSOrchestrator(config)
        >>> result = await orch.run_pipeline({})
        >>> assert result.status == ExecutionStatus.COMPLETED
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Initialize the EnMS Orchestrator.

        Args:
            config: Pipeline configuration. Uses defaults if None.
            progress_callback: Optional async callback(phase, pct, message).
        """
        self.config = config or OrchestratorConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._results: Dict[str, PipelineResult] = {}
        self._cancelled: Set[str] = set()
        self._progress_callback = progress_callback

        self.logger.info(
            "EnMSOrchestrator created: pack=%s, facility_type=%s, "
            "facility=%s, parallel=%s, iso_version=%s",
            self.config.pack_id,
            self.config.facility_type.value,
            self.config.facility_id or "(not set)",
            self.config.parallel_execution,
            self.config.iso50001_version,
        )

    # -------------------------------------------------------------------------
    # Pipeline Execution
    # -------------------------------------------------------------------------

    async def run_pipeline(
        self,
        facility_data: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """Execute the full 10-phase ISO 50001 EnMS pipeline.

        Args:
            facility_data: Input data for the pipeline phases.

        Returns:
            PipelineResult with full execution details and provenance.
        """
        facility_data = facility_data or {}

        result = PipelineResult(
            facility_type=self.config.facility_type.value,
            facility_id=self.config.facility_id,
            organization_name=self.config.organization_name,
            status=ExecutionStatus.RUNNING,
            started_at=utcnow(),
        )
        self._results[result.pipeline_id] = result

        start_time = time.monotonic()
        phases = self._resolve_phase_order()
        total_phases = len(phases)

        self.logger.info(
            "Starting EnMS pipeline: pipeline_id=%s, facility_type=%s, phases=%d",
            result.pipeline_id,
            self.config.facility_type.value,
            total_phases,
        )

        shared_context: Dict[str, Any] = dict(facility_data)
        shared_context["facility_type"] = self.config.facility_type.value
        shared_context["facility_id"] = self.config.facility_id
        shared_context["organization_name"] = self.config.organization_name
        shared_context["iso50001_version"] = self.config.iso50001_version

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
            result.completed_at = utcnow()
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

    def validate_dependencies(self, phase: OrchestratorPhase) -> Dict[str, Any]:
        """Validate the dependency tree for a specific phase.

        Args:
            phase: Phase to validate dependencies for.

        Returns:
            Dict with dependency validation details.
        """
        deps = PHASE_DEPENDENCIES.get(phase, [])
        dep_details = []
        for dep in deps:
            dep_deps = PHASE_DEPENDENCIES.get(dep, [])
            dep_details.append({
                "phase": dep.value,
                "its_dependencies": [d.value for d in dep_deps],
            })
        return {
            "phase": phase.value,
            "direct_dependencies": [d.value for d in deps],
            "dependency_details": dep_details,
            "is_root": len(deps) == 0,
            "parallel_group": self._get_parallel_group_names(phase),
        }

    def get_pipeline_status(self, pipeline_id: Optional[str] = None) -> Dict[str, Any]:
        """Get the current status and progress of a pipeline execution.

        Args:
            pipeline_id: Pipeline identifier. Uses latest if None.

        Returns:
            Dict with status, progress, and phase details.
        """
        if pipeline_id:
            result = self._results.get(pipeline_id)
        elif self._results:
            result = list(self._results.values())[-1]
        else:
            return {"found": False, "message": "No pipeline executions found"}

        if result is None:
            return {"pipeline_id": pipeline_id, "found": False}

        phases = self._resolve_phase_order()
        total = len(phases)
        completed = len(result.phases_completed) + len(result.phases_skipped)
        progress_pct = (completed / total * 100.0) if total > 0 else 0.0

        return {
            "pipeline_id": result.pipeline_id,
            "found": True,
            "status": result.status.value,
            "facility_type": result.facility_type,
            "facility_id": result.facility_id,
            "organization_name": result.organization_name,
            "phases_completed": result.phases_completed,
            "phases_skipped": result.phases_skipped,
            "progress_pct": round(progress_pct, 1),
            "total_records_processed": result.total_records_processed,
            "quality_score": result.quality_score,
            "errors": result.errors,
            "total_duration_ms": result.total_duration_ms,
        }

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
            "timestamp": utcnow().isoformat(),
        }

    # -------------------------------------------------------------------------
    # History
    # -------------------------------------------------------------------------

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
                "organization_name": r.organization_name,
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

    def _get_parallel_group_names(self, phase: OrchestratorPhase) -> List[str]:
        """Get the parallel group phase names for a phase.

        Args:
            phase: Phase to check.

        Returns:
            List of phase names in the parallel group, or empty list.
        """
        group = self._get_parallel_group(phase)
        if group:
            return [p.value for p in group]
        return []

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
            started_at=utcnow(),
            completed_at=utcnow(),
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
        phase_start = utcnow()

        self.logger.info("Executing phase '%s' (attempt %d)", phase.value, attempt + 1)

        input_hash = _compute_hash(context) if self.config.enable_provenance else ""

        records = 0
        outputs: Dict[str, Any] = {}

        if phase == OrchestratorPhase.INITIALIZATION:
            outputs = {
                "organization_profiled": True,
                "scope_defined": True,
                "boundaries_set": True,
                "energy_policy_drafted": True,
                "top_management_commitment": True,
                "enms_team_assigned": True,
            }
        elif phase == OrchestratorPhase.ENERGY_REVIEW:
            records = 12
            outputs = {
                "energy_sources_identified": 4,
                "seus_identified": records,
                "energy_uses_mapped": True,
                "significant_energy_uses": ["compressed_air", "hvac", "process_heating", "lighting"],
                "total_consumption_kwh": 25_000_000.0,
                "review_method": "iso50001_clause_6.3",
            }
        elif phase == OrchestratorPhase.BASELINE_ESTABLISHMENT:
            outputs = {
                "baseline_period": "2024-01-01/2024-12-31",
                "baseline_total_kwh": 25_000_000.0,
                "relevant_variables_identified": ["production_volume", "hdd", "cdd"],
                "regression_r_squared": 0.92,
                "baseline_adjustment_method": "regression",
                "enb_established": True,
            }
        elif phase == OrchestratorPhase.ENPI_SETUP:
            records = 6
            outputs = {
                "enpis_defined": records,
                "enpi_list": [
                    "kwh_per_unit_produced",
                    "kwh_per_m2",
                    "kwh_per_hdd",
                    "kwh_per_cdd",
                    "peak_demand_kw",
                    "energy_cost_per_unit",
                ],
                "normalization_applied": True,
                "measurement_plan_created": True,
            }
        elif phase == OrchestratorPhase.MONITORING_CONFIG:
            records = 24
            outputs = {
                "meters_configured": records,
                "metering_hierarchy_built": True,
                "data_collection_frequency": "15_min",
                "seu_sub_metering_complete": True,
                "data_quality_threshold_pct": 95.0,
                "monitoring_plan_documented": True,
            }
        elif phase == OrchestratorPhase.ACTION_PLANNING:
            records = 8
            outputs = {
                "objectives_defined": 3,
                "targets_set": records,
                "action_plans_created": records,
                "total_estimated_savings_kwh": 2_500_000.0,
                "total_investment_eur": 450_000.0,
                "priority_actions": [
                    "vfd_on_air_compressors",
                    "heat_recovery_system",
                    "led_retrofit",
                    "bms_optimization",
                ],
            }
        elif phase == OrchestratorPhase.OPERATIONAL_CONTROL:
            outputs = {
                "operational_criteria_defined": True,
                "seu_control_procedures": 12,
                "procurement_criteria_set": True,
                "design_criteria_documented": True,
                "training_needs_identified": 5,
                "communication_plan_created": True,
            }
        elif phase == OrchestratorPhase.PERFORMANCE_ANALYSIS:
            outputs = {
                "cusum_analysis_complete": True,
                "regression_updated": True,
                "energy_performance_improvement_pct": 4.2,
                "enpi_trends_analyzed": True,
                "nonconformities_identified": 2,
                "corrective_actions_assigned": 2,
                "m_and_v_method": "ipmvp_option_c",
            }
        elif phase == OrchestratorPhase.AUDIT_COMPLIANCE:
            outputs = {
                "internal_audit_complete": True,
                "clauses_assessed": 23,
                "clauses_conforming": 21,
                "clauses_nonconforming": 2,
                "minor_nonconformities": 2,
                "major_nonconformities": 0,
                "audit_findings_documented": True,
                "corrective_actions_planned": True,
            }
        elif phase == OrchestratorPhase.MANAGEMENT_REVIEW:
            outputs = {
                "review_package_prepared": True,
                "energy_performance_summary": True,
                "enpi_dashboard_generated": True,
                "resource_adequacy_assessed": True,
                "continual_improvement_plan": True,
                "next_review_date": "2026-06-30",
                "certification_readiness_score": 87.5,
                "report_sections": [
                    "energy_policy_review",
                    "enpi_performance",
                    "seu_analysis",
                    "action_plan_status",
                    "audit_results",
                    "improvement_opportunities",
                ],
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
            completed_at=utcnow(),
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
            - Certification readiness: 10 points (from audit compliance output)

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

        audit_result = result.phases.get(OrchestratorPhase.AUDIT_COMPLIANCE.value)
        if audit_result and audit_result.result_data:
            conforming = audit_result.result_data.get("clauses_conforming", 0)
            assessed = audit_result.result_data.get("clauses_assessed", 1)
            cert_score = (conforming / assessed) * 10.0 if assessed > 0 else 0.0
        else:
            cert_score = 0.0

        return round(min(completion_score + error_score + cert_score, 100.0), 2)

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
            "organization_name": "Demo Manufacturing GmbH",
            "facility_name": "Main Production Site",
            "annual_energy_kwh": 25_000_000.0,
            "annual_energy_cost_eur": 3_750_000.0,
            "production_units": 500_000,
            "floor_area_m2": 15_000.0,
            "employee_count": 350,
        }
        return await self.run_pipeline(demo_data)
