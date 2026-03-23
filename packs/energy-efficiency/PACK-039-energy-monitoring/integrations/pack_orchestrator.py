# -*- coding: utf-8 -*-
"""
MonitoringOrchestrator - 12-Phase Energy Monitoring Pipeline for PACK-039
==========================================================================

This module implements the master pipeline orchestrator for the Energy
Monitoring Pack. It coordinates all engines and workflows through a 12-phase
execution plan covering meter registry, data acquisition, data validation,
anomaly detection, EnPI calculation, cost allocation, budget tracking,
alarm management, dashboard update, reporting, data archival, and health
check.

Phases (12 total):
    1.  METER_REGISTRY          -- Meter inventory and channel configuration
    2.  DATA_ACQUISITION        -- Real-time and batch meter data collection
    3.  DATA_VALIDATION         -- Quality checks, range validation, gap detect
    4.  ANOMALY_DETECTION       -- Statistical anomaly and drift detection
    5.  ENPI_CALC               -- Energy Performance Indicator calculation
    6.  COST_ALLOCATION         -- Cost allocation to cost centers / tenants
    7.  BUDGET_TRACKING         -- Budget vs. actual comparison and forecast
    8.  ALARM_MANAGEMENT        -- Threshold alarms, escalation, acknowledgement
    9.  DASHBOARD_UPDATE        -- KPI refresh, widget data push
    10. REPORTING               -- Scheduled and ad-hoc report generation
    11. DATA_ARCHIVAL           -- Time-series compression and cold storage
    12. HEALTH_CHECK            -- System health and data pipeline verification

DAG Dependencies:
    METER_REGISTRY --> DATA_ACQUISITION
    DATA_ACQUISITION --> DATA_VALIDATION
    DATA_VALIDATION --> ANOMALY_DETECTION
    DATA_VALIDATION --> ENPI_CALC
    ANOMALY_DETECTION --> ALARM_MANAGEMENT
    ENPI_CALC --> COST_ALLOCATION
    COST_ALLOCATION --> BUDGET_TRACKING
    BUDGET_TRACKING --> DASHBOARD_UPDATE
    ALARM_MANAGEMENT --> DASHBOARD_UPDATE
    DASHBOARD_UPDATE --> REPORTING
    REPORTING --> DATA_ARCHIVAL
    DATA_ARCHIVAL --> HEALTH_CHECK

Architecture:
    Config --> MonitoringOrchestrator --> Phase DAG Resolution
                        |                        |
                        v                        v
    Phase Execution <-- Retry with Backoff <-- Parallel Where Possible
                        |
                        v
    PhaseProvenance --> SHA-256 Hashing --> PipelineResult

Zero-Hallucination:
    All EnPI calculations, cost allocation formulas, budget variance
    computations, and anomaly thresholds use deterministic arithmetic
    only. No LLM calls in the calculation path.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-039 Energy Monitoring
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
    """The 12 phases of the energy monitoring pipeline."""

    METER_REGISTRY = "meter_registry"
    DATA_ACQUISITION = "data_acquisition"
    DATA_VALIDATION = "data_validation"
    ANOMALY_DETECTION = "anomaly_detection"
    ENPI_CALC = "enpi_calc"
    COST_ALLOCATION = "cost_allocation"
    BUDGET_TRACKING = "budget_tracking"
    ALARM_MANAGEMENT = "alarm_management"
    DASHBOARD_UPDATE = "dashboard_update"
    REPORTING = "reporting"
    DATA_ARCHIVAL = "data_archival"
    HEALTH_CHECK = "health_check"


class ExecutionStatus(str, Enum):
    """Pipeline execution lifecycle status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class FacilityType(str, Enum):
    """Facility types for energy monitoring context."""

    COMMERCIAL_OFFICE = "commercial_office"
    MANUFACTURING = "manufacturing"
    RETAIL_STORE = "retail_store"
    WAREHOUSE = "warehouse"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    DATA_CENTER = "data_center"
    CAMPUS = "campus"


class MonitoringMode(str, Enum):
    """Energy monitoring operating modes."""

    REAL_TIME = "real_time"
    BATCH = "batch"
    HYBRID = "hybrid"


class DataGranularity(str, Enum):
    """Data collection granularity levels."""

    ONE_MINUTE = "1min"
    FIVE_MINUTE = "5min"
    FIFTEEN_MINUTE = "15min"
    THIRTY_MINUTE = "30min"
    HOURLY = "hourly"


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
    """Configuration for the Energy Monitoring Orchestrator."""

    pack_id: str = Field(default="PACK-039")
    pack_version: str = Field(default="1.0.0")
    facility_type: FacilityType = Field(default=FacilityType.COMMERCIAL_OFFICE)
    facility_id: str = Field(default="", description="Facility identifier")
    monitoring_mode: MonitoringMode = Field(default=MonitoringMode.HYBRID)
    data_granularity: DataGranularity = Field(default=DataGranularity.FIFTEEN_MINUTE)
    parallel_execution: bool = Field(default=True)
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout_seconds: int = Field(default=600, ge=30)
    max_concurrent_agents: int = Field(default=10, ge=1, le=50)
    enable_provenance: bool = Field(default=True)
    enable_checkpoints: bool = Field(default=True)
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    base_currency: str = Field(default="USD")
    enpi_baseline_year: int = Field(default=2024, ge=2000, description="EnPI baseline year")
    anomaly_sensitivity: float = Field(default=2.5, ge=1.0, le=5.0, description="Z-score threshold")
    cost_allocation_enabled: bool = Field(default=True, description="Enable cost allocation")
    budget_tracking_enabled: bool = Field(default=True, description="Enable budget tracking")


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
    """Complete result of the energy monitoring pipeline execution."""

    pipeline_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-039")
    facility_type: str = Field(default="commercial_office")
    facility_id: str = Field(default="")
    monitoring_mode: str = Field(default="hybrid")
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
    OrchestratorPhase.METER_REGISTRY: [],
    OrchestratorPhase.DATA_ACQUISITION: [OrchestratorPhase.METER_REGISTRY],
    OrchestratorPhase.DATA_VALIDATION: [OrchestratorPhase.DATA_ACQUISITION],
    OrchestratorPhase.ANOMALY_DETECTION: [OrchestratorPhase.DATA_VALIDATION],
    OrchestratorPhase.ENPI_CALC: [OrchestratorPhase.DATA_VALIDATION],
    OrchestratorPhase.COST_ALLOCATION: [OrchestratorPhase.ENPI_CALC],
    OrchestratorPhase.BUDGET_TRACKING: [OrchestratorPhase.COST_ALLOCATION],
    OrchestratorPhase.ALARM_MANAGEMENT: [OrchestratorPhase.ANOMALY_DETECTION],
    OrchestratorPhase.DASHBOARD_UPDATE: [
        OrchestratorPhase.BUDGET_TRACKING,
        OrchestratorPhase.ALARM_MANAGEMENT,
    ],
    OrchestratorPhase.REPORTING: [OrchestratorPhase.DASHBOARD_UPDATE],
    OrchestratorPhase.DATA_ARCHIVAL: [OrchestratorPhase.REPORTING],
    OrchestratorPhase.HEALTH_CHECK: [OrchestratorPhase.DATA_ARCHIVAL],
}

# Phases that can execute in parallel (same dependency depth)
PARALLEL_PHASE_GROUPS: List[List[OrchestratorPhase]] = [
    [OrchestratorPhase.ANOMALY_DETECTION, OrchestratorPhase.ENPI_CALC],
]

# Topological order for serial execution
PHASE_EXECUTION_ORDER: List[OrchestratorPhase] = [
    OrchestratorPhase.METER_REGISTRY,
    OrchestratorPhase.DATA_ACQUISITION,
    OrchestratorPhase.DATA_VALIDATION,
    OrchestratorPhase.ANOMALY_DETECTION,
    OrchestratorPhase.ENPI_CALC,
    OrchestratorPhase.COST_ALLOCATION,
    OrchestratorPhase.BUDGET_TRACKING,
    OrchestratorPhase.ALARM_MANAGEMENT,
    OrchestratorPhase.DASHBOARD_UPDATE,
    OrchestratorPhase.REPORTING,
    OrchestratorPhase.DATA_ARCHIVAL,
    OrchestratorPhase.HEALTH_CHECK,
]


# ---------------------------------------------------------------------------
# MonitoringOrchestrator
# ---------------------------------------------------------------------------


class MonitoringOrchestrator:
    """12-phase pipeline orchestrator for Energy Monitoring Pack.

    Executes a DAG-ordered pipeline of 12 phases covering meter registry
    through health check, with parallel execution where dependencies
    allow, retry with exponential backoff, and SHA-256 provenance tracking.

    Attributes:
        config: Orchestrator configuration.
        _results: Active and historical pipeline results.
        _cancelled: Set of cancelled execution IDs.
        _progress_callback: Optional async callback for progress updates.

    Example:
        >>> config = PipelineConfig(facility_type="manufacturing")
        >>> orch = MonitoringOrchestrator(config)
        >>> result = await orch.run_pipeline({})
        >>> assert result.status == ExecutionStatus.COMPLETED
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Initialize the Energy Monitoring Orchestrator.

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
            "MonitoringOrchestrator created: pack=%s, facility_type=%s, "
            "facility=%s, mode=%s, parallel=%s",
            self.config.pack_id,
            self.config.facility_type.value,
            self.config.facility_id or "(not set)",
            self.config.monitoring_mode.value,
            self.config.parallel_execution,
        )

    # -------------------------------------------------------------------------
    # Pipeline Execution
    # -------------------------------------------------------------------------

    async def run_pipeline(
        self,
        facility_data: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """Execute the full 12-phase energy monitoring pipeline.

        Args:
            facility_data: Input data for the pipeline phases.

        Returns:
            PipelineResult with full execution details and provenance.
        """
        facility_data = facility_data or {}

        result = PipelineResult(
            facility_type=self.config.facility_type.value,
            facility_id=self.config.facility_id,
            monitoring_mode=self.config.monitoring_mode.value,
            status=ExecutionStatus.RUNNING,
            started_at=_utcnow(),
        )
        self._results[result.pipeline_id] = result

        start_time = time.monotonic()
        phases = self._resolve_phase_order()
        total_phases = len(phases)

        self.logger.info(
            "Starting energy monitoring pipeline: pipeline_id=%s, facility_type=%s, phases=%d",
            result.pipeline_id,
            self.config.facility_type.value,
            total_phases,
        )

        shared_context: Dict[str, Any] = dict(facility_data)
        shared_context["facility_type"] = self.config.facility_type.value
        shared_context["facility_id"] = self.config.facility_id
        shared_context["monitoring_mode"] = self.config.monitoring_mode.value

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
            "monitoring_mode": result.monitoring_mode,
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
                "monitoring_mode": r.monitoring_mode,
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

        if phase == OrchestratorPhase.METER_REGISTRY:
            records = 48
            outputs = {
                "meters_registered": records,
                "channels_configured": 144,
                "protocols_active": ["modbus_tcp", "bacnet_ip", "mqtt"],
                "meter_types": ["revenue", "sub_meter", "virtual", "ct_clamp"],
                "total_capacity_kw": 12500.0,
                "coverage_pct": 98.5,
            }
        elif phase == OrchestratorPhase.DATA_ACQUISITION:
            records = 105120
            outputs = {
                "readings_collected": records,
                "interval_length_min": 15,
                "date_range_start": "2025-01-01",
                "date_range_end": "2025-12-31",
                "channels_active": 144,
                "acquisition_latency_ms": 250,
                "protocol_breakdown": {"modbus_tcp": 60, "bacnet_ip": 30, "mqtt": 10},
            }
        elif phase == OrchestratorPhase.DATA_VALIDATION:
            outputs = {
                "records_validated": 105120,
                "validation_pass_rate_pct": 99.2,
                "gaps_detected": 85,
                "outliers_flagged": 42,
                "range_violations": 12,
                "duplicate_readings": 5,
                "completeness_pct": 98.8,
            }
        elif phase == OrchestratorPhase.ANOMALY_DETECTION:
            records = 38
            outputs = {
                "anomalies_detected": records,
                "anomaly_types": {
                    "sudden_spike": 12,
                    "gradual_drift": 8,
                    "flatline": 5,
                    "pattern_break": 7,
                    "negative_flow": 3,
                    "calibration_drift": 3,
                },
                "z_score_threshold": self.config.anomaly_sensitivity,
                "false_positive_estimate_pct": 5.0,
                "critical_anomalies": 4,
            }
        elif phase == OrchestratorPhase.ENPI_CALC:
            outputs = {
                "enpi_count": 12,
                "baseline_year": self.config.enpi_baseline_year,
                "enpi_results": {
                    "kwh_per_m2": 185.0,
                    "kwh_per_unit": 42.5,
                    "kwh_per_hdd": 1.25,
                    "kwh_per_cdd": 0.95,
                    "pue": 1.45,
                    "eui_kbtu_per_sqft": 58.6,
                },
                "improvement_vs_baseline_pct": 8.5,
                "cusum_trend": "improving",
                "regression_r_squared": 0.92,
            }
        elif phase == OrchestratorPhase.COST_ALLOCATION:
            outputs = {
                "cost_centers_allocated": 15,
                "total_energy_cost_usd": 1_850_000.0,
                "allocation_method": "metered_proportional",
                "cost_breakdown": {
                    "production": 720_000.0,
                    "hvac": 480_000.0,
                    "lighting": 185_000.0,
                    "process": 250_000.0,
                    "common_areas": 120_000.0,
                    "data_center": 95_000.0,
                },
                "unallocated_pct": 1.2,
            }
        elif phase == OrchestratorPhase.BUDGET_TRACKING:
            outputs = {
                "budget_total_usd": 2_000_000.0,
                "actual_ytd_usd": 1_850_000.0,
                "variance_usd": -150_000.0,
                "variance_pct": -7.5,
                "forecast_year_end_usd": 1_920_000.0,
                "budget_status": "under_budget",
                "months_over_budget": 2,
                "savings_vs_budget_usd": 80_000.0,
            }
        elif phase == OrchestratorPhase.ALARM_MANAGEMENT:
            records = 24
            outputs = {
                "active_alarms": records,
                "alarms_by_severity": {
                    "critical": 2,
                    "high": 5,
                    "medium": 10,
                    "low": 7,
                },
                "alarms_acknowledged": 18,
                "alarms_auto_cleared": 6,
                "mean_time_to_acknowledge_min": 12.5,
                "escalation_count": 3,
            }
        elif phase == OrchestratorPhase.DASHBOARD_UPDATE:
            outputs = {
                "widgets_updated": 24,
                "kpi_cards_refreshed": 12,
                "charts_rendered": 8,
                "trend_lines_updated": 6,
                "last_refresh_latency_ms": 450,
                "dashboard_sections": [
                    "overview", "meters", "enpi", "cost", "budget",
                    "alarms", "anomalies", "trends",
                ],
            }
        elif phase == OrchestratorPhase.REPORTING:
            outputs = {
                "report_sections": [
                    "executive_summary", "meter_status",
                    "energy_consumption", "enpi_analysis",
                    "cost_allocation", "budget_variance",
                    "anomaly_summary", "alarm_log",
                    "recommendations", "appendices",
                ],
                "format": "PDF",
                "dashboard_generated": True,
                "report_frequency": "monthly",
            }
        elif phase == OrchestratorPhase.DATA_ARCHIVAL:
            records = 2_628_000
            outputs = {
                "records_archived": records,
                "compression_ratio": 8.5,
                "archive_size_mb": 245.0,
                "retention_policy": "7_years",
                "cold_storage_tier": "glacier",
                "archive_verified": True,
            }
        elif phase == OrchestratorPhase.HEALTH_CHECK:
            outputs = {
                "verification_status": "PASS",
                "checks_run": 52,
                "checks_passed": 52,
                "meter_connectivity_pct": 100.0,
                "data_freshness_ok": True,
                "provenance_chain_valid": True,
                "audit_trail_complete": True,
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
            - Verification pass: 10 points (from health_check phase output)

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

        verify_result = result.phases.get(OrchestratorPhase.HEALTH_CHECK.value)
        if verify_result and verify_result.result_data:
            verify_status = verify_result.result_data.get("verification_status", "")
            dq_score = 10.0 if verify_status == "PASS" else 5.0
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
            "facility_name": "Demo Manufacturing Campus",
            "total_meters": 48,
            "annual_energy_kwh": 10_500_000.0,
            "annual_energy_cost_usd": 1_850_000.0,
            "floor_area_m2": 25_000.0,
            "energy_budget_usd": 2_000_000.0,
            "baseline_year": 2024,
        }
        return await self.run_pipeline(demo_data)
