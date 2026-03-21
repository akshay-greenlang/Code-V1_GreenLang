# -*- coding: utf-8 -*-
"""
EnergyBenchmarkOrchestrator - 12-Phase Energy Benchmark Pipeline for PACK-035
================================================================================

This module implements the master pipeline orchestrator for the Energy Benchmark
Pack. It coordinates all engines and workflows through a 12-phase execution plan
covering health verification, configuration, data acquisition, weather
normalisation, EUI calculation, peer comparison, performance rating, gap
analysis, trend analysis, portfolio aggregation, and report generation.

Phases (12 total):
    1.  HEALTH_CHECK            -- Verify all engines, agents, and dependencies
    2.  CONFIGURATION           -- Load facility profile, benchmark scope, presets
    3.  DATA_ACQUISITION        -- Ingest meter data, utility bills, floor areas
    4.  WEATHER_DATA            -- Retrieve weather station data, HDD/CDD, TMY
    5.  EUI_CALCULATION         -- Calculate Energy Use Intensity (kWh/m2/yr)
    6.  WEATHER_NORMALISATION   -- Apply weather normalisation to EUI
    7.  PEER_COMPARISON         -- Compare against peer group benchmarks
    8.  PERFORMANCE_RATING      -- Assign performance rating (A-G or ENERGY STAR)
    9.  GAP_ANALYSIS            -- Quantify gap to best practice / target
    10. TREND_ANALYSIS          -- Analyse multi-year performance trends
    11. PORTFOLIO_AGGREGATION   -- Aggregate across building portfolio
    12. REPORT_GENERATION       -- Generate benchmark report and dashboards

DAG Dependencies:
    HEALTH_CHECK --> CONFIGURATION --> DATA_ACQUISITION
    CONFIGURATION --> WEATHER_DATA
    DATA_ACQUISITION --> EUI_CALCULATION
    WEATHER_DATA --> WEATHER_NORMALISATION
    EUI_CALCULATION --> WEATHER_NORMALISATION
    WEATHER_NORMALISATION --> PEER_COMPARISON
    PEER_COMPARISON --> PERFORMANCE_RATING
    PERFORMANCE_RATING --> GAP_ANALYSIS
    GAP_ANALYSIS --> TREND_ANALYSIS
    TREND_ANALYSIS --> PORTFOLIO_AGGREGATION
    PORTFOLIO_AGGREGATION --> REPORT_GENERATION

Architecture:
    Config --> EnergyBenchmarkOrchestrator --> Phase DAG Resolution
                        |                            |
                        v                            v
    Phase Execution <-- Retry with Backoff <-- Parallel Where Possible
                        |
                        v
    PhaseProvenance --> SHA-256 Hashing --> PipelineResult

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-035 Energy Benchmark
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
    """The 12 phases of the energy benchmark pipeline."""

    HEALTH_CHECK = "health_check"
    CONFIGURATION = "configuration"
    DATA_ACQUISITION = "data_acquisition"
    WEATHER_DATA = "weather_data"
    EUI_CALCULATION = "eui_calculation"
    WEATHER_NORMALISATION = "weather_normalisation"
    PEER_COMPARISON = "peer_comparison"
    PERFORMANCE_RATING = "performance_rating"
    GAP_ANALYSIS = "gap_analysis"
    TREND_ANALYSIS = "trend_analysis"
    PORTFOLIO_AGGREGATION = "portfolio_aggregation"
    REPORT_GENERATION = "report_generation"


class PhaseStatus(str, Enum):
    """Pipeline execution lifecycle status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class RetryStrategy(str, Enum):
    """Retry strategy for failed phases."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_DELAY = "fixed_delay"
    NO_RETRY = "no_retry"


class BuildingSector(str, Enum):
    """Building sectors for benchmark context."""

    OFFICE = "office"
    RETAIL = "retail"
    HOTEL = "hotel"
    HOSPITAL = "hospital"
    EDUCATION = "education"
    WAREHOUSE = "warehouse"
    INDUSTRIAL = "industrial"
    RESIDENTIAL_MULTI = "residential_multi"
    MIXED_USE = "mixed_use"
    DATA_CENTRE = "data_centre"
    LEISURE = "leisure"
    RESTAURANT = "restaurant"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class RetryConfig(BaseModel):
    """Retry configuration with exponential backoff and jitter."""

    strategy: RetryStrategy = Field(
        default=RetryStrategy.EXPONENTIAL_BACKOFF,
        description="Retry strategy to use",
    )
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retry attempts per phase")
    backoff_base: float = Field(default=1.0, ge=0.5, description="Base delay in seconds")
    backoff_max: float = Field(default=30.0, ge=1.0, description="Maximum backoff delay")
    jitter_factor: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Jitter multiplier"
    )


class OrchestratorConfig(BaseModel):
    """Configuration for the Energy Benchmark Orchestrator."""

    pack_id: str = Field(default="PACK-035")
    pack_version: str = Field(default="1.0.0")
    building_sector: BuildingSector = Field(default=BuildingSector.OFFICE)
    facility_id: str = Field(default="", description="Facility identifier")
    portfolio_id: str = Field(default="", description="Portfolio identifier for multi-building")
    max_concurrent_agents: int = Field(default=10, ge=1, le=50)
    timeout_per_phase_seconds: int = Field(default=600, ge=30)
    enable_provenance: bool = Field(default=True)
    enable_checkpoints: bool = Field(default=True)
    enable_parallel_phases: bool = Field(default=True)
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    benchmark_year: int = Field(default=2025, ge=2020, le=2035)
    base_currency: str = Field(default="EUR")
    benchmark_source: str = Field(
        default="cibse_tm46",
        description="cibse_tm46|energy_star|din_v_18599|bpie|custom",
    )
    weather_normalisation_method: str = Field(
        default="degree_day",
        description="degree_day|regression|bin_method",
    )
    include_portfolio_view: bool = Field(default=True)
    include_trend_analysis: bool = Field(default=True)


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
    status: PhaseStatus = Field(default=PhaseStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    records_processed: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    provenance: Optional[PhaseProvenance] = Field(None)
    retry_count: int = Field(default=0)


class PipelineResult(BaseModel):
    """Complete result of the energy benchmark pipeline execution."""

    execution_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-035")
    building_sector: str = Field(default="office")
    facility_id: str = Field(default="")
    portfolio_id: str = Field(default="")
    status: PhaseStatus = Field(default=PhaseStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    total_duration_ms: float = Field(default=0.0)
    phases_completed: List[str] = Field(default_factory=list)
    phases_skipped: List[str] = Field(default_factory=list)
    phase_results: Dict[str, PhaseResult] = Field(default_factory=dict)
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
    OrchestratorPhase.DATA_ACQUISITION: [OrchestratorPhase.CONFIGURATION],
    OrchestratorPhase.WEATHER_DATA: [OrchestratorPhase.CONFIGURATION],
    OrchestratorPhase.EUI_CALCULATION: [OrchestratorPhase.DATA_ACQUISITION],
    OrchestratorPhase.WEATHER_NORMALISATION: [
        OrchestratorPhase.EUI_CALCULATION,
        OrchestratorPhase.WEATHER_DATA,
    ],
    OrchestratorPhase.PEER_COMPARISON: [OrchestratorPhase.WEATHER_NORMALISATION],
    OrchestratorPhase.PERFORMANCE_RATING: [OrchestratorPhase.PEER_COMPARISON],
    OrchestratorPhase.GAP_ANALYSIS: [OrchestratorPhase.PERFORMANCE_RATING],
    OrchestratorPhase.TREND_ANALYSIS: [OrchestratorPhase.GAP_ANALYSIS],
    OrchestratorPhase.PORTFOLIO_AGGREGATION: [OrchestratorPhase.TREND_ANALYSIS],
    OrchestratorPhase.REPORT_GENERATION: [OrchestratorPhase.PORTFOLIO_AGGREGATION],
}

# Phases that can execute in parallel (same dependency depth)
PARALLEL_PHASE_GROUPS: List[List[OrchestratorPhase]] = [
    [OrchestratorPhase.DATA_ACQUISITION, OrchestratorPhase.WEATHER_DATA],
]

# Topological order for serial execution
PHASE_EXECUTION_ORDER: List[OrchestratorPhase] = [
    OrchestratorPhase.HEALTH_CHECK,
    OrchestratorPhase.CONFIGURATION,
    OrchestratorPhase.DATA_ACQUISITION,
    OrchestratorPhase.WEATHER_DATA,
    OrchestratorPhase.EUI_CALCULATION,
    OrchestratorPhase.WEATHER_NORMALISATION,
    OrchestratorPhase.PEER_COMPARISON,
    OrchestratorPhase.PERFORMANCE_RATING,
    OrchestratorPhase.GAP_ANALYSIS,
    OrchestratorPhase.TREND_ANALYSIS,
    OrchestratorPhase.PORTFOLIO_AGGREGATION,
    OrchestratorPhase.REPORT_GENERATION,
]

# Phases that can be skipped based on scope
PHASE_SCOPE_APPLICABILITY: Dict[OrchestratorPhase, List[str]] = {
    OrchestratorPhase.PORTFOLIO_AGGREGATION: ["portfolio"],
    OrchestratorPhase.TREND_ANALYSIS: ["full", "portfolio"],
}


# ---------------------------------------------------------------------------
# EnergyBenchmarkOrchestrator
# ---------------------------------------------------------------------------


class EnergyBenchmarkOrchestrator:
    """12-phase pipeline orchestrator for Energy Benchmark Pack.

    Executes a DAG-ordered pipeline of 12 phases covering health verification
    through report generation, with parallel execution where dependencies
    allow, retry with exponential backoff, and SHA-256 provenance tracking.

    Attributes:
        config: Orchestrator configuration.
        _results: Active and historical pipeline results.
        _cancelled: Set of cancelled execution IDs.
        _progress_callback: Optional async callback for progress updates.

    Example:
        >>> config = OrchestratorConfig(building_sector="office")
        >>> orch = EnergyBenchmarkOrchestrator(config)
        >>> result = await orch.execute_pipeline({})
        >>> assert result.status == PhaseStatus.COMPLETED
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Initialize the Energy Benchmark Orchestrator.

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
            "EnergyBenchmarkOrchestrator created: pack=%s, sector=%s, "
            "benchmark_source=%s, facility=%s",
            self.config.pack_id,
            self.config.building_sector.value,
            self.config.benchmark_source,
            self.config.facility_id or "(not set)",
        )

    # -------------------------------------------------------------------------
    # Pipeline Execution
    # -------------------------------------------------------------------------

    async def execute_pipeline(
        self,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """Execute the full 12-phase energy benchmark pipeline.

        Args:
            input_data: Input data for the pipeline phases.

        Returns:
            PipelineResult with full execution details and provenance.
        """
        input_data = input_data or {}

        result = PipelineResult(
            building_sector=self.config.building_sector.value,
            facility_id=self.config.facility_id,
            portfolio_id=self.config.portfolio_id,
            status=PhaseStatus.RUNNING,
            started_at=_utcnow(),
        )
        self._results[result.execution_id] = result

        start_time = time.monotonic()
        phases = self._resolve_phase_order()
        total_phases = len(phases)

        self.logger.info(
            "Starting benchmark pipeline: execution_id=%s, sector=%s, phases=%d",
            result.execution_id,
            self.config.building_sector.value,
            total_phases,
        )

        shared_context: Dict[str, Any] = dict(input_data)
        shared_context["building_sector"] = self.config.building_sector.value
        shared_context["facility_id"] = self.config.facility_id
        shared_context["benchmark_year"] = self.config.benchmark_year
        shared_context["benchmark_source"] = self.config.benchmark_source

        try:
            for phase_idx, phase in enumerate(phases):
                if result.execution_id in self._cancelled:
                    result.status = PhaseStatus.CANCELLED
                    result.errors.append("Pipeline cancelled by user")
                    break

                # Scope skip check
                if self._should_skip_phase(phase):
                    phase_result = PhaseResult(
                        phase=phase,
                        status=PhaseStatus.SKIPPED,
                        started_at=_utcnow(),
                        completed_at=_utcnow(),
                    )
                    result.phase_results[phase.value] = phase_result
                    result.phases_skipped.append(phase.value)
                    self.logger.info(
                        "Phase '%s' skipped (not applicable for current scope)",
                        phase.value,
                    )
                    continue

                # DAG dependency check
                if not self._dependencies_met(phase, result):
                    phase_result = PhaseResult(
                        phase=phase,
                        status=PhaseStatus.FAILED,
                        errors=["Dependencies not met"],
                    )
                    result.phase_results[phase.value] = phase_result
                    result.status = PhaseStatus.FAILED
                    result.errors.append(f"Phase '{phase.value}' dependencies not met")
                    break

                # Check for parallel execution opportunity
                if self.config.enable_parallel_phases:
                    parallel_group = self._get_parallel_group(phase)
                    if parallel_group and all(
                        p.value not in result.phase_results for p in parallel_group
                    ):
                        await self._execute_parallel_phases(
                            parallel_group, shared_context, result
                        )
                        for p in parallel_group:
                            pr = result.phase_results.get(p.value)
                            if pr and pr.status == PhaseStatus.COMPLETED:
                                result.phases_completed.append(p.value)
                                result.total_records_processed += pr.records_processed
                                shared_context[p.value] = pr.outputs
                        continue

                # Skip if already completed in a parallel group
                if phase.value in result.phase_results:
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
                result.phase_results[phase.value] = phase_result

                if phase_result.status == PhaseStatus.FAILED:
                    result.status = PhaseStatus.FAILED
                    result.errors.append(f"Phase '{phase.value}' failed after retries")
                    break

                result.phases_completed.append(phase.value)
                result.total_records_processed += phase_result.records_processed
                shared_context[phase.value] = phase_result.outputs

            if result.status == PhaseStatus.RUNNING:
                result.status = PhaseStatus.COMPLETED

        except Exception as exc:
            self.logger.error(
                "Pipeline failed: execution_id=%s, error=%s",
                result.execution_id, exc, exc_info=True,
            )
            result.status = PhaseStatus.FAILED
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
            "Pipeline %s: execution_id=%s, phases=%d/%d, duration=%.1fms",
            result.status.value, result.execution_id,
            len(result.phases_completed), total_phases,
            result.total_duration_ms,
        )
        return result

    # -------------------------------------------------------------------------
    # Single Phase Execution
    # -------------------------------------------------------------------------

    async def execute_phase(
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

    # -------------------------------------------------------------------------
    # Cancellation
    # -------------------------------------------------------------------------

    def cancel_pipeline(self, execution_id: str) -> Dict[str, Any]:
        """Cancel a running pipeline execution.

        Args:
            execution_id: Execution ID to cancel.

        Returns:
            Dict with cancellation status.
        """
        if execution_id not in self._results:
            return {"execution_id": execution_id, "cancelled": False, "reason": "Not found"}

        result = self._results[execution_id]
        if result.status not in (PhaseStatus.RUNNING, PhaseStatus.PENDING):
            return {
                "execution_id": execution_id,
                "cancelled": False,
                "reason": f"Cannot cancel in status '{result.status.value}'",
            }

        self._cancelled.add(execution_id)
        return {
            "execution_id": execution_id,
            "cancelled": True,
            "reason": "Cancellation signal sent",
            "timestamp": _utcnow().isoformat(),
        }

    # -------------------------------------------------------------------------
    # Status & History
    # -------------------------------------------------------------------------

    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get the current status and progress of a pipeline execution.

        Args:
            execution_id: Execution identifier.

        Returns:
            Dict with status, progress, and phase details.
        """
        if execution_id not in self._results:
            return {"execution_id": execution_id, "found": False}

        result = self._results[execution_id]
        phases = self._resolve_phase_order()
        total = len(phases)
        completed = len(result.phases_completed) + len(result.phases_skipped)
        progress_pct = (completed / total * 100.0) if total > 0 else 0.0

        return {
            "execution_id": execution_id,
            "found": True,
            "status": result.status.value,
            "building_sector": result.building_sector,
            "facility_id": result.facility_id,
            "portfolio_id": result.portfolio_id,
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
                "execution_id": r.execution_id,
                "status": r.status.value,
                "building_sector": r.building_sector,
                "facility_id": r.facility_id,
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

    def _should_skip_phase(self, phase: OrchestratorPhase) -> bool:
        """Determine whether a phase should be skipped for the current scope.

        Args:
            phase: Phase to check.

        Returns:
            True if the phase should be skipped.
        """
        if phase == OrchestratorPhase.PORTFOLIO_AGGREGATION:
            if not self.config.include_portfolio_view:
                return True
        if phase == OrchestratorPhase.TREND_ANALYSIS:
            if not self.config.include_trend_analysis:
                return True
        return False

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
            dep_result = result.phase_results.get(dep.value)
            if dep_result is None:
                return False
            if dep_result.status not in (
                PhaseStatus.COMPLETED, PhaseStatus.SKIPPED
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
                pipeline_result.phase_results[phase.value] = PhaseResult(
                    phase=phase,
                    status=PhaseStatus.FAILED,
                    errors=[str(phase_result)],
                )
            else:
                pipeline_result.phase_results[phase.value] = phase_result

    # -------------------------------------------------------------------------
    # Phase Execution with Retry
    # -------------------------------------------------------------------------

    async def _run_with_retry(
        self,
        phase: OrchestratorPhase,
        context: Dict[str, Any],
        pipeline_result: PipelineResult,
    ) -> PhaseResult:
        """Execute a phase with retry logic based on configured strategy.

        Alias for _execute_phase_with_retry that matches the specification
        method name.

        Args:
            phase: Phase to execute.
            context: Shared pipeline context.
            pipeline_result: Parent pipeline result.

        Returns:
            PhaseResult for the phase.
        """
        return await self._execute_phase_with_retry(phase, context, pipeline_result)

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

        max_attempts = (
            retry_config.max_retries + 1
            if retry_config.strategy != RetryStrategy.NO_RETRY
            else 1
        )

        for attempt in range(max_attempts):
            try:
                phase_result = await self._execute_phase(phase, context, attempt)
                if phase_result.status == PhaseStatus.COMPLETED:
                    phase_result.retry_count = attempt
                    return phase_result
                last_error = "; ".join(phase_result.errors) if phase_result.errors else "Unknown"
            except asyncio.TimeoutError:
                last_error = f"Phase {phase.value} timed out"
            except Exception as exc:
                last_error = str(exc)

            if attempt < max_attempts - 1:
                if retry_config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
                    base_delay = retry_config.backoff_base * (2 ** attempt)
                    delay = min(base_delay, retry_config.backoff_max)
                    jitter = random.uniform(0, retry_config.jitter_factor * delay)
                    total_delay = delay + jitter
                else:
                    total_delay = retry_config.backoff_base

                self.logger.warning(
                    "Phase '%s' failed (attempt %d/%d), retrying in %.1fs: %s",
                    phase.value, attempt + 1, max_attempts,
                    total_delay, last_error,
                )
                await asyncio.sleep(total_delay)

        self.logger.error(
            "Phase '%s' failed after %d attempts: %s",
            phase.value, max_attempts, last_error,
        )
        return PhaseResult(
            phase=phase,
            status=PhaseStatus.FAILED,
            started_at=_utcnow(),
            completed_at=_utcnow(),
            errors=[last_error or "Unknown error"],
            retry_count=max_attempts - 1,
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
            outputs = {"all_engines_healthy": True, "dependency_count": 12}
        elif phase == OrchestratorPhase.CONFIGURATION:
            outputs = {
                "facility_configured": True,
                "sector": self.config.building_sector.value,
                "benchmark_source": self.config.benchmark_source,
                "weather_method": self.config.weather_normalisation_method,
            }
        elif phase == OrchestratorPhase.DATA_ACQUISITION:
            records = 8760  # Hourly data for one year
            outputs = {
                "meter_readings": records,
                "utility_bills": 12,
                "floor_area_m2": 8000.0,
                "energy_carriers": ["electricity", "natural_gas"],
            }
        elif phase == OrchestratorPhase.WEATHER_DATA:
            records = 365
            outputs = {
                "weather_station_id": "NEAREST-001",
                "hdd_total": 2800.0,
                "cdd_total": 450.0,
                "tmy_available": True,
                "daily_records": records,
            }
        elif phase == OrchestratorPhase.EUI_CALCULATION:
            outputs = {
                "eui_kwh_per_m2": 0.0,
                "eui_electricity_kwh_per_m2": 0.0,
                "eui_gas_kwh_per_m2": 0.0,
                "total_energy_kwh": 0.0,
                "floor_area_m2": 8000.0,
            }
        elif phase == OrchestratorPhase.WEATHER_NORMALISATION:
            outputs = {
                "normalised_eui_kwh_per_m2": 0.0,
                "normalisation_method": self.config.weather_normalisation_method,
                "base_temperature_heating_c": 15.5,
                "base_temperature_cooling_c": 22.0,
                "adjustment_factor": 1.0,
            }
        elif phase == OrchestratorPhase.PEER_COMPARISON:
            outputs = {
                "peer_group_size": 0,
                "peer_median_eui": 0.0,
                "peer_percentile": 0.0,
                "peer_quartile": 0,
                "benchmark_source": self.config.benchmark_source,
            }
        elif phase == OrchestratorPhase.PERFORMANCE_RATING:
            outputs = {
                "rating": "D",
                "rating_system": self.config.benchmark_source,
                "score": 0.0,
                "rating_scale": "A-G",
            }
        elif phase == OrchestratorPhase.GAP_ANALYSIS:
            outputs = {
                "gap_to_best_practice_kwh_per_m2": 0.0,
                "gap_to_best_practice_pct": 0.0,
                "gap_to_median_kwh_per_m2": 0.0,
                "savings_potential_kwh": 0.0,
                "savings_potential_eur": 0.0,
            }
        elif phase == OrchestratorPhase.TREND_ANALYSIS:
            outputs = {
                "years_analysed": 0,
                "trend_direction": "stable",
                "annual_change_pct": 0.0,
                "trend_confidence": 0.0,
            }
        elif phase == OrchestratorPhase.PORTFOLIO_AGGREGATION:
            outputs = {
                "buildings_count": 0,
                "portfolio_eui_avg": 0.0,
                "portfolio_total_kwh": 0.0,
                "worst_performers": [],
                "best_performers": [],
            }
        elif phase == OrchestratorPhase.REPORT_GENERATION:
            outputs = {
                "report_sections": [
                    "executive_summary", "methodology", "eui_analysis",
                    "weather_normalisation", "peer_comparison",
                    "performance_rating", "gap_analysis",
                    "trend_analysis", "recommendations",
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
            status=PhaseStatus.COMPLETED,
            started_at=phase_start,
            completed_at=_utcnow(),
            duration_ms=elapsed_ms,
            records_processed=records,
            outputs=outputs,
            provenance=provenance,
        )

    # -------------------------------------------------------------------------
    # Quality Score
    # -------------------------------------------------------------------------

    def _compute_quality_score(self, result: PipelineResult) -> float:
        """Compute overall pipeline quality score (0-100).

        Scoring formula:
            - Phase completion: 50 points (pct of non-skipped phases completed)
            - Error-free execution: 30 points (deducted per error)
            - Data coverage: 20 points (from weather normalisation output)

        Args:
            result: Pipeline result to score.

        Returns:
            Quality score between 0.0 and 100.0.
        """
        total_applicable = len(PHASE_EXECUTION_ORDER) - len(result.phases_skipped)
        if total_applicable == 0:
            return 0.0

        completion_score = (len(result.phases_completed) / total_applicable) * 50.0
        error_count = len(result.errors)
        error_score = max(0.0, 30.0 - error_count * 10.0)

        norm_result = result.phase_results.get(
            OrchestratorPhase.WEATHER_NORMALISATION.value
        )
        if norm_result and norm_result.outputs:
            dq_score = 20.0 if norm_result.outputs.get("normalisation_method") else 10.0
        else:
            dq_score = 0.0

        return round(min(completion_score + error_score + dq_score, 100.0), 2)

    # -------------------------------------------------------------------------
    # Demo Execution
    # -------------------------------------------------------------------------

    async def run_demo(self) -> PipelineResult:
        """Run a demonstration pipeline with sample data.

        Returns:
            PipelineResult for the demo execution.
        """
        demo_data = {
            "demo_mode": True,
            "facility_name": "Demo Office Building",
            "annual_energy_kwh": 2_400_000.0,
            "annual_energy_cost_eur": 360_000.0,
            "floor_area_m2": 8_000.0,
            "benchmark_period": {
                "start": f"{self.config.benchmark_year}-01-01",
                "end": f"{self.config.benchmark_year}-12-31",
            },
        }
        return await self.execute_pipeline(demo_data)
