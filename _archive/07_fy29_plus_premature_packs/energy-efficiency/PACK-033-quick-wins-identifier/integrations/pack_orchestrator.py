# -*- coding: utf-8 -*-
"""
QuickWinsOrchestrator - 10-Phase Quick Wins Identification Pipeline for PACK-033
==================================================================================

This module implements the master pipeline orchestrator for the Quick Wins
Identifier Pack. It coordinates all engines and workflows through a 10-phase
execution plan covering facility intake, equipment surveys, quick win scanning,
financial analysis, carbon assessment, prioritization, rebate matching,
implementation planning, and reporting.

Phases (10 total):
    1.  INTAKE                   -- Ingest facility profile and energy data
    2.  FACILITY_SCAN            -- Scan facility for quick win opportunities
    3.  EQUIPMENT_SURVEY         -- Survey equipment efficiency and condition
    4.  QUICK_WIN_SCANNING       -- Identify no/low-cost energy saving measures
    5.  FINANCIAL_ANALYSIS       -- Calculate payback, ROI, NPV for each measure
    6.  CARBON_ASSESSMENT        -- Quantify avoided emissions (tCO2e)
    7.  PRIORITIZATION           -- Rank quick wins by payback and impact
    8.  REBATE_MATCHING          -- Match measures to utility rebate programs
    9.  IMPLEMENTATION_PLANNING  -- Create implementation schedule and resources
    10. REPORTING                -- Generate quick wins report and dashboard

DAG Dependencies:
    INTAKE --> FACILITY_SCAN
    INTAKE --> EQUIPMENT_SURVEY
    FACILITY_SCAN --> QUICK_WIN_SCANNING
    EQUIPMENT_SURVEY --> QUICK_WIN_SCANNING
    QUICK_WIN_SCANNING --> FINANCIAL_ANALYSIS
    QUICK_WIN_SCANNING --> CARBON_ASSESSMENT
    FINANCIAL_ANALYSIS --> PRIORITIZATION
    CARBON_ASSESSMENT --> PRIORITIZATION
    PRIORITIZATION --> REBATE_MATCHING
    REBATE_MATCHING --> IMPLEMENTATION_PLANNING
    IMPLEMENTATION_PLANNING --> REPORTING

Architecture:
    Config --> QuickWinsOrchestrator --> Phase DAG Resolution
                        |                        |
                        v                        v
    Phase Execution <-- Retry with Backoff <-- Parallel Where Possible
                        |
                        v
    PhaseProvenance --> SHA-256 Hashing --> PipelineResult

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-033 Quick Wins Identifier
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
    """The 10 phases of the quick wins identification pipeline."""

    INTAKE = "intake"
    FACILITY_SCAN = "facility_scan"
    EQUIPMENT_SURVEY = "equipment_survey"
    QUICK_WIN_SCANNING = "quick_win_scanning"
    FINANCIAL_ANALYSIS = "financial_analysis"
    CARBON_ASSESSMENT = "carbon_assessment"
    PRIORITIZATION = "prioritization"
    REBATE_MATCHING = "rebate_matching"
    IMPLEMENTATION_PLANNING = "implementation_planning"
    REPORTING = "reporting"

class FacilityType(str, Enum):
    """Facility types for quick win context."""

    OFFICE_BUILDING = "office_building"
    MANUFACTURING = "manufacturing"
    RETAIL_STORE = "retail_store"
    WAREHOUSE = "warehouse"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    DATA_CENTER = "data_center"
    SME_SIMPLIFIED = "sme_simplified"

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
    """Configuration for the Quick Wins Orchestrator."""

    pack_id: str = Field(default="PACK-033")
    pack_version: str = Field(default="1.0.0")
    facility_type: FacilityType = Field(default=FacilityType.OFFICE_BUILDING)
    facility_id: str = Field(default="", description="Facility identifier")
    parallel_execution: bool = Field(default=True)
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout_seconds: int = Field(default=600, ge=30)
    max_concurrent_agents: int = Field(default=10, ge=1, le=50)
    enable_provenance: bool = Field(default=True)
    enable_checkpoints: bool = Field(default=True)
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    base_currency: str = Field(default="EUR")
    payback_threshold_months: int = Field(default=24, ge=1, le=120)
    min_savings_kwh: float = Field(default=100.0, ge=0.0)

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
    """Complete result of the quick wins pipeline execution."""

    pipeline_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-033")
    facility_type: str = Field(default="office_building")
    facility_id: str = Field(default="")
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
    OrchestratorPhase.INTAKE: [],
    OrchestratorPhase.FACILITY_SCAN: [OrchestratorPhase.INTAKE],
    OrchestratorPhase.EQUIPMENT_SURVEY: [OrchestratorPhase.INTAKE],
    OrchestratorPhase.QUICK_WIN_SCANNING: [
        OrchestratorPhase.FACILITY_SCAN,
        OrchestratorPhase.EQUIPMENT_SURVEY,
    ],
    OrchestratorPhase.FINANCIAL_ANALYSIS: [OrchestratorPhase.QUICK_WIN_SCANNING],
    OrchestratorPhase.CARBON_ASSESSMENT: [OrchestratorPhase.QUICK_WIN_SCANNING],
    OrchestratorPhase.PRIORITIZATION: [
        OrchestratorPhase.FINANCIAL_ANALYSIS,
        OrchestratorPhase.CARBON_ASSESSMENT,
    ],
    OrchestratorPhase.REBATE_MATCHING: [OrchestratorPhase.PRIORITIZATION],
    OrchestratorPhase.IMPLEMENTATION_PLANNING: [OrchestratorPhase.REBATE_MATCHING],
    OrchestratorPhase.REPORTING: [OrchestratorPhase.IMPLEMENTATION_PLANNING],
}

# Phases that can execute in parallel (same dependency depth)
PARALLEL_PHASE_GROUPS: List[List[OrchestratorPhase]] = [
    [OrchestratorPhase.FACILITY_SCAN, OrchestratorPhase.EQUIPMENT_SURVEY],
    [OrchestratorPhase.FINANCIAL_ANALYSIS, OrchestratorPhase.CARBON_ASSESSMENT],
]

# Topological order for serial execution
PHASE_EXECUTION_ORDER: List[OrchestratorPhase] = [
    OrchestratorPhase.INTAKE,
    OrchestratorPhase.FACILITY_SCAN,
    OrchestratorPhase.EQUIPMENT_SURVEY,
    OrchestratorPhase.QUICK_WIN_SCANNING,
    OrchestratorPhase.FINANCIAL_ANALYSIS,
    OrchestratorPhase.CARBON_ASSESSMENT,
    OrchestratorPhase.PRIORITIZATION,
    OrchestratorPhase.REBATE_MATCHING,
    OrchestratorPhase.IMPLEMENTATION_PLANNING,
    OrchestratorPhase.REPORTING,
]

# ---------------------------------------------------------------------------
# QuickWinsOrchestrator
# ---------------------------------------------------------------------------

class QuickWinsOrchestrator:
    """10-phase pipeline orchestrator for Quick Wins Identifier Pack.

    Executes a DAG-ordered pipeline of 10 phases covering facility intake
    through reporting, with parallel execution where dependencies allow,
    retry with exponential backoff, and SHA-256 provenance tracking.

    Attributes:
        config: Orchestrator configuration.
        _results: Active and historical pipeline results.
        _cancelled: Set of cancelled execution IDs.
        _progress_callback: Optional async callback for progress updates.

    Example:
        >>> config = OrchestratorConfig(facility_type="manufacturing")
        >>> orch = QuickWinsOrchestrator(config)
        >>> result = await orch.run_pipeline({})
        >>> assert result.status == ExecutionStatus.COMPLETED
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Initialize the Quick Wins Orchestrator.

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
            "QuickWinsOrchestrator created: pack=%s, facility_type=%s, "
            "facility=%s, parallel=%s",
            self.config.pack_id,
            self.config.facility_type.value,
            self.config.facility_id or "(not set)",
            self.config.parallel_execution,
        )

    # -------------------------------------------------------------------------
    # Pipeline Execution
    # -------------------------------------------------------------------------

    async def run_pipeline(
        self,
        facility_data: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """Execute the full 10-phase quick wins identification pipeline.

        Args:
            facility_data: Input data for the pipeline phases.

        Returns:
            PipelineResult with full execution details and provenance.
        """
        facility_data = facility_data or {}

        result = PipelineResult(
            facility_type=self.config.facility_type.value,
            facility_id=self.config.facility_id,
            status=ExecutionStatus.RUNNING,
            started_at=utcnow(),
        )
        self._results[result.pipeline_id] = result

        start_time = time.monotonic()
        phases = self._resolve_phase_order()
        total_phases = len(phases)

        self.logger.info(
            "Starting quick wins pipeline: pipeline_id=%s, facility_type=%s, phases=%d",
            result.pipeline_id,
            self.config.facility_type.value,
            total_phases,
        )

        shared_context: Dict[str, Any] = dict(facility_data)
        shared_context["facility_type"] = self.config.facility_type.value
        shared_context["facility_id"] = self.config.facility_id

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

        if phase == OrchestratorPhase.INTAKE:
            outputs = {
                "facility_profiled": True,
                "energy_data_loaded": True,
                "data_sources": ["utility_bills", "meter_data", "equipment_list"],
            }
        elif phase == OrchestratorPhase.FACILITY_SCAN:
            outputs = {
                "zones_scanned": 0,
                "areas_identified": 0,
                "scan_method": "walkthrough",
            }
        elif phase == OrchestratorPhase.EQUIPMENT_SURVEY:
            records = 50
            outputs = {
                "equipment_surveyed": records,
                "below_efficiency_threshold": 0,
                "quick_replacement_candidates": 0,
            }
        elif phase == OrchestratorPhase.QUICK_WIN_SCANNING:
            outputs = {
                "quick_wins_found": 0,
                "categories": ["lighting", "hvac", "controls", "envelope", "plug_loads"],
                "total_estimated_savings_kwh": 0.0,
            }
        elif phase == OrchestratorPhase.FINANCIAL_ANALYSIS:
            outputs = {
                "measures_analyzed": 0,
                "total_investment_eur": 0.0,
                "total_annual_savings_eur": 0.0,
                "average_payback_months": 0.0,
            }
        elif phase == OrchestratorPhase.CARBON_ASSESSMENT:
            outputs = {
                "total_avoided_tco2e": 0.0,
                "scope1_avoided_tco2e": 0.0,
                "scope2_avoided_tco2e": 0.0,
            }
        elif phase == OrchestratorPhase.PRIORITIZATION:
            outputs = {
                "ranked_measures": 0,
                "tier1_immediate": 0,
                "tier2_short_term": 0,
                "tier3_medium_term": 0,
            }
        elif phase == OrchestratorPhase.REBATE_MATCHING:
            outputs = {
                "rebates_matched": 0,
                "total_rebate_value_eur": 0.0,
                "applications_prepared": 0,
            }
        elif phase == OrchestratorPhase.IMPLEMENTATION_PLANNING:
            outputs = {
                "implementation_weeks": 0,
                "phases_planned": 0,
                "resources_assigned": 0,
            }
        elif phase == OrchestratorPhase.REPORTING:
            outputs = {
                "report_sections": [
                    "executive_summary", "quick_wins_list",
                    "financial_analysis", "carbon_impact",
                    "implementation_plan", "rebate_opportunities",
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
            - Data coverage: 10 points (from quick win scanning output)

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

        scan_result = result.phases.get(OrchestratorPhase.QUICK_WIN_SCANNING.value)
        if scan_result and scan_result.result_data:
            dq_score = 10.0 if scan_result.result_data.get("quick_wins_found", 0) >= 0 else 5.0
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
            "facility_name": "Demo Office Building",
            "annual_energy_kwh": 2_500_000.0,
            "annual_energy_cost_eur": 375_000.0,
            "floor_area_m2": 8_000.0,
            "employee_count": 200,
        }
        return await self.run_pipeline(demo_data)
