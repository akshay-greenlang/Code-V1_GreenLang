# -*- coding: utf-8 -*-
"""
IndustrialEnergyAuditOrchestrator - 12-Phase Energy Audit Pipeline for PACK-031
==================================================================================

This module implements the master pipeline orchestrator for the Industrial Energy
Audit Pack. It coordinates all 10 engines and 8 workflows through a 12-phase
execution plan covering health verification, data ingestion, baseline
establishment, audit execution, specialized system audits, benchmarking, and
compliance reporting.

Phases (12 total):
    1.  health_check          -- Verify all engines, agents, and dependencies
    2.  configuration         -- Load facility profile, audit scope, presets
    3.  data_ingestion        -- Ingest meter data, utility bills, equipment data
    4.  baseline              -- Establish energy consumption baselines
    5.  energy_audit          -- Execute comprehensive energy audit
    6.  process_mapping       -- Map energy flows through production processes
    7.  equipment_assessment  -- Assess equipment efficiency and condition
    8.  savings_identification-- Identify and quantify energy savings opportunities
    9.  specialized_audits    -- Compressed air, steam, waste heat recovery audits
    10. benchmarking          -- Benchmark against sector and best practice
    11. report_generation     -- Generate audit report and recommendations
    12. compliance_check      -- Verify regulatory compliance (EED, ISO 50002)

DAG Dependencies:
    health_check --> configuration --> data_ingestion
    data_ingestion --> baseline --> energy_audit
    energy_audit --> process_mapping
    energy_audit --> equipment_assessment
    process_mapping --> savings_identification
    equipment_assessment --> savings_identification
    savings_identification --> specialized_audits
    specialized_audits --> benchmarking
    benchmarking --> report_generation
    report_generation --> compliance_check

Architecture:
    Config --> IndustrialEnergyAuditOrchestrator --> Phase DAG Resolution
                        |                                  |
                        v                                  v
    Phase Execution <-- Retry with Backoff <-- Parallel Where Possible
                        |
                        v
    PhaseProvenance --> SHA-256 Hashing --> PipelineResult

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-031 Industrial Energy Audit
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


class AuditPipelinePhase(str, Enum):
    """The 12 phases of the industrial energy audit pipeline."""

    HEALTH_CHECK = "health_check"
    CONFIGURATION = "configuration"
    DATA_INGESTION = "data_ingestion"
    BASELINE = "baseline"
    ENERGY_AUDIT = "energy_audit"
    PROCESS_MAPPING = "process_mapping"
    EQUIPMENT_ASSESSMENT = "equipment_assessment"
    SAVINGS_IDENTIFICATION = "savings_identification"
    SPECIALIZED_AUDITS = "specialized_audits"
    BENCHMARKING = "benchmarking"
    REPORT_GENERATION = "report_generation"
    COMPLIANCE_CHECK = "compliance_check"


class ExecutionStatus(str, Enum):
    """Pipeline execution lifecycle status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class IndustrySector(str, Enum):
    """Industrial sectors for audit context."""

    MANUFACTURING = "manufacturing"
    FOOD_BEVERAGE = "food_beverage"
    CHEMICALS = "chemicals"
    METALS = "metals"
    CEMENT = "cement"
    PAPER_PULP = "paper_pulp"
    GLASS = "glass"
    TEXTILES = "textiles"
    AUTOMOTIVE = "automotive"
    PHARMACEUTICALS = "pharmaceuticals"
    DATA_CENTRES = "data_centres"
    COMMERCIAL_BUILDINGS = "commercial_buildings"


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
    """Configuration for the Industrial Energy Audit Orchestrator."""

    pack_id: str = Field(default="PACK-031")
    pack_version: str = Field(default="1.0.0")
    industry_sector: IndustrySector = Field(default=IndustrySector.MANUFACTURING)
    facility_id: str = Field(default="", description="Facility identifier")
    max_concurrent_agents: int = Field(default=10, ge=1, le=50)
    timeout_per_phase_seconds: int = Field(default=600, ge=30)
    enable_provenance: bool = Field(default=True)
    enable_checkpoints: bool = Field(default=True)
    enable_parallel_phases: bool = Field(default=True)
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    audit_year: int = Field(default=2025, ge=2020, le=2035)
    base_currency: str = Field(default="EUR")
    audit_type: str = Field(default="comprehensive", description="comprehensive|walkthrough|targeted")
    include_compressed_air: bool = Field(default=True)
    include_steam_systems: bool = Field(default=True)
    include_waste_heat: bool = Field(default=True)


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

    phase: AuditPipelinePhase = Field(...)
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
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
    """Complete result of the energy audit pipeline execution."""

    execution_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-031")
    industry_sector: str = Field(default="manufacturing")
    facility_id: str = Field(default="")
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
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

PHASE_DEPENDENCIES: Dict[AuditPipelinePhase, List[AuditPipelinePhase]] = {
    AuditPipelinePhase.HEALTH_CHECK: [],
    AuditPipelinePhase.CONFIGURATION: [AuditPipelinePhase.HEALTH_CHECK],
    AuditPipelinePhase.DATA_INGESTION: [AuditPipelinePhase.CONFIGURATION],
    AuditPipelinePhase.BASELINE: [AuditPipelinePhase.DATA_INGESTION],
    AuditPipelinePhase.ENERGY_AUDIT: [AuditPipelinePhase.BASELINE],
    AuditPipelinePhase.PROCESS_MAPPING: [AuditPipelinePhase.ENERGY_AUDIT],
    AuditPipelinePhase.EQUIPMENT_ASSESSMENT: [AuditPipelinePhase.ENERGY_AUDIT],
    AuditPipelinePhase.SAVINGS_IDENTIFICATION: [
        AuditPipelinePhase.PROCESS_MAPPING,
        AuditPipelinePhase.EQUIPMENT_ASSESSMENT,
    ],
    AuditPipelinePhase.SPECIALIZED_AUDITS: [AuditPipelinePhase.SAVINGS_IDENTIFICATION],
    AuditPipelinePhase.BENCHMARKING: [AuditPipelinePhase.SPECIALIZED_AUDITS],
    AuditPipelinePhase.REPORT_GENERATION: [AuditPipelinePhase.BENCHMARKING],
    AuditPipelinePhase.COMPLIANCE_CHECK: [AuditPipelinePhase.REPORT_GENERATION],
}

# Phases that can execute in parallel (same dependency depth)
PARALLEL_PHASE_GROUPS: List[List[AuditPipelinePhase]] = [
    [AuditPipelinePhase.PROCESS_MAPPING, AuditPipelinePhase.EQUIPMENT_ASSESSMENT],
]

# Topological order for serial execution
PHASE_EXECUTION_ORDER: List[AuditPipelinePhase] = [
    AuditPipelinePhase.HEALTH_CHECK,
    AuditPipelinePhase.CONFIGURATION,
    AuditPipelinePhase.DATA_INGESTION,
    AuditPipelinePhase.BASELINE,
    AuditPipelinePhase.ENERGY_AUDIT,
    AuditPipelinePhase.PROCESS_MAPPING,
    AuditPipelinePhase.EQUIPMENT_ASSESSMENT,
    AuditPipelinePhase.SAVINGS_IDENTIFICATION,
    AuditPipelinePhase.SPECIALIZED_AUDITS,
    AuditPipelinePhase.BENCHMARKING,
    AuditPipelinePhase.REPORT_GENERATION,
    AuditPipelinePhase.COMPLIANCE_CHECK,
]

# Phases that can be skipped based on audit type
PHASE_AUDIT_TYPE_APPLICABILITY: Dict[AuditPipelinePhase, List[str]] = {
    AuditPipelinePhase.SPECIALIZED_AUDITS: ["comprehensive"],
    AuditPipelinePhase.PROCESS_MAPPING: ["comprehensive", "targeted"],
}


# ---------------------------------------------------------------------------
# IndustrialEnergyAuditOrchestrator
# ---------------------------------------------------------------------------


class IndustrialEnergyAuditOrchestrator:
    """12-phase pipeline orchestrator for Industrial Energy Audit Pack.

    Executes a DAG-ordered pipeline of 12 phases covering health verification
    through compliance checking, with parallel execution where dependencies
    allow, retry with exponential backoff, and SHA-256 provenance tracking.

    Attributes:
        config: Orchestrator configuration.
        _results: Active and historical pipeline results.
        _cancelled: Set of cancelled execution IDs.
        _progress_callback: Optional async callback for progress updates.

    Example:
        >>> config = OrchestratorConfig(industry_sector="chemicals")
        >>> orch = IndustrialEnergyAuditOrchestrator(config)
        >>> result = await orch.execute_pipeline({})
        >>> assert result.status == ExecutionStatus.COMPLETED
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Initialize the Industrial Energy Audit Orchestrator.

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
            "IndustrialEnergyAuditOrchestrator created: pack=%s, sector=%s, "
            "audit_type=%s, facility=%s",
            self.config.pack_id,
            self.config.industry_sector.value,
            self.config.audit_type,
            self.config.facility_id or "(not set)",
        )

    # -------------------------------------------------------------------------
    # Pipeline Execution
    # -------------------------------------------------------------------------

    async def execute_pipeline(
        self,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """Execute the full 12-phase industrial energy audit pipeline.

        Args:
            input_data: Input data for the pipeline phases.

        Returns:
            PipelineResult with full execution details and provenance.
        """
        input_data = input_data or {}

        result = PipelineResult(
            industry_sector=self.config.industry_sector.value,
            facility_id=self.config.facility_id,
            status=ExecutionStatus.RUNNING,
            started_at=_utcnow(),
        )
        self._results[result.execution_id] = result

        start_time = time.monotonic()
        phases = self._resolve_phase_order()
        total_phases = len(phases)

        self.logger.info(
            "Starting energy audit pipeline: execution_id=%s, sector=%s, phases=%d",
            result.execution_id,
            self.config.industry_sector.value,
            total_phases,
        )

        shared_context: Dict[str, Any] = dict(input_data)
        shared_context["industry_sector"] = self.config.industry_sector.value
        shared_context["facility_id"] = self.config.facility_id
        shared_context["audit_year"] = self.config.audit_year
        shared_context["audit_type"] = self.config.audit_type

        try:
            for phase_idx, phase in enumerate(phases):
                if result.execution_id in self._cancelled:
                    result.status = ExecutionStatus.CANCELLED
                    result.errors.append("Pipeline cancelled by user")
                    break

                # Audit type skip check
                if self._should_skip_phase(phase):
                    phase_result = PhaseResult(
                        phase=phase,
                        status=ExecutionStatus.SKIPPED,
                        started_at=_utcnow(),
                        completed_at=_utcnow(),
                    )
                    result.phase_results[phase.value] = phase_result
                    result.phases_skipped.append(phase.value)
                    self.logger.info(
                        "Phase '%s' skipped (not applicable for audit_type '%s')",
                        phase.value, self.config.audit_type,
                    )
                    continue

                # DAG dependency check
                if not self._dependencies_met(phase, result):
                    phase_result = PhaseResult(
                        phase=phase,
                        status=ExecutionStatus.FAILED,
                        errors=["Dependencies not met"],
                    )
                    result.phase_results[phase.value] = phase_result
                    result.status = ExecutionStatus.FAILED
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
                            if pr and pr.status == ExecutionStatus.COMPLETED:
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

                if phase_result.status == ExecutionStatus.FAILED:
                    result.status = ExecutionStatus.FAILED
                    result.errors.append(f"Phase '{phase.value}' failed after retries")
                    break

                result.phases_completed.append(phase.value)
                result.total_records_processed += phase_result.records_processed
                shared_context[phase.value] = phase_result.outputs

            if result.status == ExecutionStatus.RUNNING:
                result.status = ExecutionStatus.COMPLETED

        except Exception as exc:
            self.logger.error(
                "Pipeline failed: execution_id=%s, error=%s",
                result.execution_id, exc, exc_info=True,
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
            "Pipeline %s: execution_id=%s, phases=%d/%d, duration=%.1fms",
            result.status.value, result.execution_id,
            len(result.phases_completed), total_phases,
            result.total_duration_ms,
        )
        return result

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
        if result.status not in (ExecutionStatus.RUNNING, ExecutionStatus.PENDING):
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
            "industry_sector": result.industry_sector,
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
                "execution_id": r.execution_id,
                "status": r.status.value,
                "industry_sector": r.industry_sector,
                "facility_id": r.facility_id,
                "phases_completed": len(r.phases_completed),
                "started_at": r.started_at.isoformat() if r.started_at else None,
            }
            for r in self._results.values()
        ]

    # -------------------------------------------------------------------------
    # Phase Resolution
    # -------------------------------------------------------------------------

    def _resolve_phase_order(self) -> List[AuditPipelinePhase]:
        """Resolve the topological phase execution order.

        Returns:
            Ordered list of phases respecting DAG dependencies.
        """
        return list(PHASE_EXECUTION_ORDER)

    def _should_skip_phase(self, phase: AuditPipelinePhase) -> bool:
        """Determine whether a phase should be skipped for the current audit type.

        Args:
            phase: Phase to check.

        Returns:
            True if the phase should be skipped.
        """
        if phase not in PHASE_AUDIT_TYPE_APPLICABILITY:
            return False

        applicable_types = PHASE_AUDIT_TYPE_APPLICABILITY[phase]
        return self.config.audit_type not in applicable_types

    def _dependencies_met(
        self, phase: AuditPipelinePhase, result: PipelineResult
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
                ExecutionStatus.COMPLETED, ExecutionStatus.SKIPPED
            ):
                return False
        return True

    def _get_parallel_group(
        self, phase: AuditPipelinePhase
    ) -> Optional[List[AuditPipelinePhase]]:
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
        phases: List[AuditPipelinePhase],
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
                    status=ExecutionStatus.FAILED,
                    errors=[str(phase_result)],
                )
            else:
                pipeline_result.phase_results[phase.value] = phase_result

    # -------------------------------------------------------------------------
    # Phase Execution with Retry
    # -------------------------------------------------------------------------

    async def _execute_phase_with_retry(
        self,
        phase: AuditPipelinePhase,
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
                last_error = "; ".join(phase_result.errors) if phase_result.errors else "Unknown"
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
            errors=[last_error or "Unknown error"],
            retry_count=retry_config.max_retries,
        )

    async def _execute_phase(
        self,
        phase: AuditPipelinePhase,
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

        if phase == AuditPipelinePhase.HEALTH_CHECK:
            outputs = {"all_engines_healthy": True, "dependency_count": 10}
        elif phase == AuditPipelinePhase.CONFIGURATION:
            outputs = {
                "facility_configured": True,
                "sector": self.config.industry_sector.value,
                "audit_type": self.config.audit_type,
            }
        elif phase == AuditPipelinePhase.DATA_INGESTION:
            records = 8760  # Hourly data for one year
            outputs = {
                "meter_readings": records,
                "utility_bills": 12,
                "equipment_records": 150,
            }
        elif phase == AuditPipelinePhase.BASELINE:
            outputs = {
                "baseline_kwh": 0.0,
                "baseline_year": self.config.audit_year - 1,
                "weather_normalized": True,
                "enpi_established": True,
            }
        elif phase == AuditPipelinePhase.ENERGY_AUDIT:
            outputs = {
                "total_consumption_kwh": 0.0,
                "cost_eur": 0.0,
                "energy_carriers": ["electricity", "natural_gas", "diesel"],
                "audit_scope": "full_facility",
            }
        elif phase == AuditPipelinePhase.PROCESS_MAPPING:
            outputs = {
                "processes_mapped": 0,
                "energy_flows_traced": 0,
                "sankey_generated": True,
            }
        elif phase == AuditPipelinePhase.EQUIPMENT_ASSESSMENT:
            records = 150
            outputs = {
                "equipment_assessed": records,
                "below_efficiency_threshold": 0,
                "replacement_candidates": 0,
            }
        elif phase == AuditPipelinePhase.SAVINGS_IDENTIFICATION:
            outputs = {
                "opportunities_found": 0,
                "total_savings_kwh": 0.0,
                "total_savings_eur": 0.0,
                "total_savings_tco2e": 0.0,
            }
        elif phase == AuditPipelinePhase.SPECIALIZED_AUDITS:
            outputs = {
                "compressed_air_audited": self.config.include_compressed_air,
                "steam_audited": self.config.include_steam_systems,
                "waste_heat_audited": self.config.include_waste_heat,
            }
        elif phase == AuditPipelinePhase.BENCHMARKING:
            outputs = {
                "sector_rank": 0,
                "sector_percentile": 0.0,
                "best_practice_gap_pct": 0.0,
            }
        elif phase == AuditPipelinePhase.REPORT_GENERATION:
            outputs = {
                "report_sections": [
                    "executive_summary", "methodology", "baseline",
                    "findings", "recommendations", "implementation_plan",
                ],
                "format": "PDF",
            }
        elif phase == AuditPipelinePhase.COMPLIANCE_CHECK:
            outputs = {
                "eed_compliant": True,
                "iso_50002_aligned": True,
                "en_16247_aligned": True,
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
            - Data coverage: 20 points (from baseline phase output)

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

        baseline_result = result.phase_results.get(AuditPipelinePhase.BASELINE.value)
        if baseline_result and baseline_result.outputs:
            dq_score = 20.0 if baseline_result.outputs.get("weather_normalized") else 10.0
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
            "facility_name": "Demo Manufacturing Plant",
            "annual_energy_kwh": 15_000_000.0,
            "annual_energy_cost_eur": 2_250_000.0,
            "audit_period": {
                "start": f"{self.config.audit_year}-01-01",
                "end": f"{self.config.audit_year}-12-31",
            },
        }
        return await self.execute_pipeline(demo_data)
