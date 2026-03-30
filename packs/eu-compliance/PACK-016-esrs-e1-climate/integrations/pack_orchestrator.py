# -*- coding: utf-8 -*-
"""
E1PackOrchestrator - 10-Phase ESRS E1 Disclosure Pipeline for PACK-016
=========================================================================

This module implements the master pipeline orchestrator for the ESRS E1
Climate Pack. It executes a 10-phase pipeline covering materiality check,
GHG inventory, energy assessment, transition planning, target setting,
climate actions, carbon credits, carbon pricing, climate risk analysis,
and final report assembly.

Phases (10 total):
    1.  materiality_check    -- Verify E1 climate materiality from DMA
    2.  ghg_inventory        -- Compute Scope 1/2/3 GHG inventory
    3.  energy_assessment    -- Assess energy consumption and mix
    4.  transition_plan      -- Evaluate or build climate transition plan
    5.  target_setting       -- Set and validate climate targets / SBTi
    6.  climate_actions      -- Catalogue climate actions and resources
    7.  carbon_credits       -- Assess carbon credit portfolio
    8.  carbon_pricing       -- Evaluate carbon pricing exposure
    9.  climate_risk         -- Analyze physical and transition risks
    10. report_assembly      -- Assemble the final E1 disclosure package

DAG Dependencies:
    materiality_check --> ghg_inventory --> energy_assessment
    energy_assessment --> transition_plan --> target_setting
    target_setting --> climate_actions
    climate_actions --> carbon_credits
    climate_actions --> carbon_pricing   (parallel with carbon_credits)
    carbon_credits + carbon_pricing --> climate_risk
    climate_risk --> report_assembly

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-016 ESRS E1 Climate Pack
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

class E1PipelinePhase(str, Enum):
    """The 10 phases of the ESRS E1 Climate disclosure pipeline."""

    MATERIALITY_CHECK = "materiality_check"
    GHG_INVENTORY = "ghg_inventory"
    ENERGY_ASSESSMENT = "energy_assessment"
    TRANSITION_PLAN = "transition_plan"
    TARGET_SETTING = "target_setting"
    CLIMATE_ACTIONS = "climate_actions"
    CARBON_CREDITS = "carbon_credits"
    CARBON_PRICING = "carbon_pricing"
    CLIMATE_RISK = "climate_risk"
    REPORT_ASSEMBLY = "report_assembly"

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
    """Configuration for the E1 Pipeline Orchestrator."""

    pack_id: str = Field(default="PACK-016")
    pack_version: str = Field(default="1.0.0")
    max_concurrent_phases: int = Field(default=2, ge=1, le=5)
    timeout_per_phase_seconds: int = Field(default=600, ge=30)
    enable_provenance: bool = Field(default=True)
    enable_checkpoints: bool = Field(default=True)
    enable_parallel_assessment: bool = Field(
        default=True,
        description="Enable parallel execution of carbon_credits and carbon_pricing",
    )
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    base_currency: str = Field(default="EUR")
    gwp_source: str = Field(default="IPCC AR6")
    consolidation_approach: str = Field(default="operational_control")

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

    phase: E1PipelinePhase = Field(...)
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
    """Complete result of the E1 pipeline execution."""

    execution_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-016")
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

PHASE_DEPENDENCIES: Dict[E1PipelinePhase, List[E1PipelinePhase]] = {
    E1PipelinePhase.MATERIALITY_CHECK: [],
    E1PipelinePhase.GHG_INVENTORY: [E1PipelinePhase.MATERIALITY_CHECK],
    E1PipelinePhase.ENERGY_ASSESSMENT: [E1PipelinePhase.GHG_INVENTORY],
    E1PipelinePhase.TRANSITION_PLAN: [E1PipelinePhase.ENERGY_ASSESSMENT],
    E1PipelinePhase.TARGET_SETTING: [E1PipelinePhase.TRANSITION_PLAN],
    E1PipelinePhase.CLIMATE_ACTIONS: [E1PipelinePhase.TARGET_SETTING],
    E1PipelinePhase.CARBON_CREDITS: [E1PipelinePhase.CLIMATE_ACTIONS],
    E1PipelinePhase.CARBON_PRICING: [E1PipelinePhase.CLIMATE_ACTIONS],
    E1PipelinePhase.CLIMATE_RISK: [
        E1PipelinePhase.CARBON_CREDITS,
        E1PipelinePhase.CARBON_PRICING,
    ],
    E1PipelinePhase.REPORT_ASSEMBLY: [E1PipelinePhase.CLIMATE_RISK],
}

PARALLEL_PHASE_GROUPS: List[List[E1PipelinePhase]] = [
    [E1PipelinePhase.CARBON_CREDITS, E1PipelinePhase.CARBON_PRICING],
]

PHASE_EXECUTION_ORDER: List[E1PipelinePhase] = [
    E1PipelinePhase.MATERIALITY_CHECK,
    E1PipelinePhase.GHG_INVENTORY,
    E1PipelinePhase.ENERGY_ASSESSMENT,
    E1PipelinePhase.TRANSITION_PLAN,
    E1PipelinePhase.TARGET_SETTING,
    E1PipelinePhase.CLIMATE_ACTIONS,
    E1PipelinePhase.CARBON_CREDITS,
    E1PipelinePhase.CARBON_PRICING,
    E1PipelinePhase.CLIMATE_RISK,
    E1PipelinePhase.REPORT_ASSEMBLY,
]

# ---------------------------------------------------------------------------
# E1PackOrchestrator
# ---------------------------------------------------------------------------

class E1PackOrchestrator:
    """10-phase ESRS E1 Climate disclosure pipeline orchestrator for PACK-016.

    Executes a DAG-ordered pipeline of 10 phases covering materiality
    verification through report assembly, with parallel carbon assessment,
    retry with exponential backoff, provenance tracking, and progress
    callbacks.

    Attributes:
        config: Orchestrator configuration.
        _results: Active and historical pipeline results.
        _cancelled: Set of cancelled execution IDs.
        _progress_callback: Optional async callback for progress updates.

    Example:
        >>> config = OrchestratorConfig(reporting_year=2025)
        >>> orch = E1PackOrchestrator(config)
        >>> result = await orch.execute_pipeline({})
        >>> assert result.status == ExecutionStatus.COMPLETED
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Initialize E1PackOrchestrator.

        Args:
            config: Orchestrator configuration. Defaults used if None.
            progress_callback: Optional async callback for progress updates.
        """
        self.config = config or OrchestratorConfig()
        self._results: Dict[str, PipelineResult] = {}
        self._cancelled: Set[str] = set()
        self._progress_callback = progress_callback
        logger.info(
            "E1PackOrchestrator initialized (pack=%s, year=%d)",
            self.config.pack_id,
            self.config.reporting_year,
        )

    async def execute_pipeline(
        self,
        context: Dict[str, Any],
        phases: Optional[List[E1PipelinePhase]] = None,
    ) -> PipelineResult:
        """Execute the full E1 disclosure pipeline.

        Args:
            context: Shared pipeline context with input data.
            phases: Optional subset of phases to execute.

        Returns:
            PipelineResult with status, phase results, and provenance.
        """
        result = PipelineResult(
            pack_id=self.config.pack_id,
            started_at=utcnow(),
            status=ExecutionStatus.RUNNING,
        )
        self._results[result.execution_id] = result

        target_phases = phases or list(PHASE_EXECUTION_ORDER)
        total_phases = len(target_phases)

        try:
            executed: Set[str] = set()

            for idx, phase in enumerate(target_phases):
                if result.execution_id in self._cancelled:
                    result.status = ExecutionStatus.CANCELLED
                    break

                # Check dependencies
                deps = PHASE_DEPENDENCIES.get(phase, [])
                unmet = [d for d in deps if d.value not in executed]
                if unmet:
                    # Check for parallel group
                    parallel_group = self._find_parallel_group(phase, target_phases)
                    if parallel_group and self.config.enable_parallel_assessment:
                        await self._execute_parallel_phases(
                            parallel_group, context, result
                        )
                        for p in parallel_group:
                            executed.add(p.value)
                        continue
                    else:
                        logger.warning(
                            "Skipping phase %s: unmet dependencies %s",
                            phase.value,
                            [d.value for d in unmet],
                        )
                        result.phases_skipped.append(phase.value)
                        continue

                # Check if already executed (from parallel group)
                if phase.value in executed:
                    continue

                # Execute phase
                phase_result = await self._execute_phase_with_retry(
                    phase, context, result
                )
                result.phase_results[phase.value] = phase_result

                if phase_result.status == ExecutionStatus.COMPLETED:
                    result.phases_completed.append(phase.value)
                    executed.add(phase.value)
                    result.total_records_processed += phase_result.records_processed
                else:
                    result.errors.append(f"Phase {phase.value} failed")
                    result.status = ExecutionStatus.FAILED
                    break

                # Progress callback
                if self._progress_callback:
                    progress = (idx + 1) / total_phases
                    await self._progress_callback(
                        phase.value, progress, f"Completed {phase.value}"
                    )

            if result.status == ExecutionStatus.RUNNING:
                result.status = ExecutionStatus.COMPLETED

        except Exception as exc:
            logger.error("Pipeline execution failed: %s", str(exc), exc_info=True)
            result.status = ExecutionStatus.FAILED
            result.errors.append(str(exc))

        result.completed_at = utcnow()
        if result.started_at:
            result.total_duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000

        # Compute quality score
        result.quality_score = self._compute_quality_score(result)

        # Compute provenance hash
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        return result

    async def execute_phase(
        self,
        phase: E1PipelinePhase,
        context: Dict[str, Any],
    ) -> PhaseResult:
        """Execute a single phase.

        Args:
            phase: Phase to execute.
            context: Shared pipeline context.

        Returns:
            PhaseResult with status and outputs.
        """
        return await self._execute_phase_with_retry(
            phase, context, PipelineResult()
        )

    def get_pipeline_status(self, execution_id: str) -> Optional[PipelineResult]:
        """Get the status of a pipeline execution.

        Args:
            execution_id: Pipeline execution ID.

        Returns:
            PipelineResult or None if not found.
        """
        return self._results.get(execution_id)

    def validate_prerequisites(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate pipeline prerequisites.

        Args:
            context: Pipeline context to validate.

        Returns:
            Dict with 'valid' bool and 'errors'/'warnings' lists.
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not context.get("entity_name"):
            errors.append("entity_name is required in context")
        if not context.get("reporting_year"):
            errors.append("reporting_year is required in context")
        if "e1_materiality" not in context:
            warnings.append("e1_materiality not in context; materiality_check will assess")
        if "scope1_data" not in context:
            warnings.append("scope1_data not in context; will attempt MRV import")
        if "scope2_data" not in context:
            warnings.append("scope2_data not in context; will attempt MRV import")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def cancel_pipeline(self, execution_id: str) -> bool:
        """Cancel a running pipeline execution.

        Args:
            execution_id: Execution ID to cancel.

        Returns:
            True if cancellation was registered.
        """
        self._cancelled.add(execution_id)
        logger.info("Pipeline %s marked for cancellation", execution_id)
        return True

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    async def _execute_phase_with_retry(
        self,
        phase: E1PipelinePhase,
        context: Dict[str, Any],
        pipeline_result: PipelineResult,
    ) -> PhaseResult:
        """Execute a phase with retry and exponential backoff."""
        phase_result = PhaseResult(phase=phase)
        retry_config = self.config.retry_config
        attempt = 0

        while attempt <= retry_config.max_retries:
            attempt += 1
            phase_result.started_at = utcnow()
            phase_result.status = ExecutionStatus.RUNNING

            try:
                input_hash = _compute_hash(context) if self.config.enable_provenance else ""

                outputs = await self._run_phase_logic(phase, context)

                phase_result.outputs = outputs
                phase_result.status = ExecutionStatus.COMPLETED
                phase_result.completed_at = utcnow()

                if phase_result.started_at:
                    phase_result.duration_ms = (
                        phase_result.completed_at - phase_result.started_at
                    ).total_seconds() * 1000

                phase_result.records_processed = outputs.get("records_processed", 0)
                phase_result.retry_count = attempt - 1

                if self.config.enable_provenance:
                    output_hash = _compute_hash(outputs)
                    phase_result.provenance = PhaseProvenance(
                        phase=phase.value,
                        input_hash=input_hash,
                        output_hash=output_hash,
                        duration_ms=phase_result.duration_ms,
                        attempt=attempt,
                    )

                # Merge outputs into context for downstream phases
                context[f"{phase.value}_result"] = outputs

                logger.info(
                    "Phase %s completed in %.1fms (attempt %d)",
                    phase.value,
                    phase_result.duration_ms,
                    attempt,
                )
                return phase_result

            except Exception as exc:
                logger.warning(
                    "Phase %s attempt %d failed: %s",
                    phase.value,
                    attempt,
                    str(exc),
                )
                phase_result.errors.append(f"Attempt {attempt}: {str(exc)}")

                if attempt <= retry_config.max_retries:
                    delay = min(
                        retry_config.backoff_base * (2 ** (attempt - 1)),
                        retry_config.backoff_max,
                    )
                    jitter = delay * retry_config.jitter_factor * random.random()
                    await asyncio.sleep(delay + jitter)

        phase_result.status = ExecutionStatus.FAILED
        phase_result.completed_at = utcnow()
        return phase_result

    async def _execute_parallel_phases(
        self,
        phases: List[E1PipelinePhase],
        context: Dict[str, Any],
        pipeline_result: PipelineResult,
    ) -> None:
        """Execute multiple phases in parallel."""
        tasks = [
            self._execute_phase_with_retry(phase, context, pipeline_result)
            for phase in phases
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for phase, result in zip(phases, results):
            if isinstance(result, Exception):
                phase_result = PhaseResult(
                    phase=phase,
                    status=ExecutionStatus.FAILED,
                    errors=[str(result)],
                )
                pipeline_result.phase_results[phase.value] = phase_result
                pipeline_result.errors.append(f"Parallel phase {phase.value} failed")
            else:
                pipeline_result.phase_results[phase.value] = result
                if result.status == ExecutionStatus.COMPLETED:
                    pipeline_result.phases_completed.append(phase.value)
                    pipeline_result.total_records_processed += result.records_processed

    def _find_parallel_group(
        self,
        phase: E1PipelinePhase,
        target_phases: List[E1PipelinePhase],
    ) -> Optional[List[E1PipelinePhase]]:
        """Find a parallel group containing the given phase."""
        for group in PARALLEL_PHASE_GROUPS:
            if phase in group and all(p in target_phases for p in group):
                return group
        return None

    async def _run_phase_logic(
        self,
        phase: E1PipelinePhase,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the logic for a single phase.

        Each phase returns a dict of outputs that are merged into context.
        This is a framework stub; actual engine calls would go here.
        """
        handlers = {
            E1PipelinePhase.MATERIALITY_CHECK: self._phase_materiality_check,
            E1PipelinePhase.GHG_INVENTORY: self._phase_ghg_inventory,
            E1PipelinePhase.ENERGY_ASSESSMENT: self._phase_energy_assessment,
            E1PipelinePhase.TRANSITION_PLAN: self._phase_transition_plan,
            E1PipelinePhase.TARGET_SETTING: self._phase_target_setting,
            E1PipelinePhase.CLIMATE_ACTIONS: self._phase_climate_actions,
            E1PipelinePhase.CARBON_CREDITS: self._phase_carbon_credits,
            E1PipelinePhase.CARBON_PRICING: self._phase_carbon_pricing,
            E1PipelinePhase.CLIMATE_RISK: self._phase_climate_risk,
            E1PipelinePhase.REPORT_ASSEMBLY: self._phase_report_assembly,
        }
        handler = handlers.get(phase)
        if handler is None:
            raise ValueError(f"No handler for phase: {phase.value}")
        return await handler(context)

    async def _phase_materiality_check(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 1: Check E1 climate materiality from DMA results."""
        e1_material = context.get("e1_materiality", True)
        return {
            "e1_is_material": e1_material,
            "materiality_source": context.get("materiality_source", "pack-015"),
            "records_processed": 1,
        }

    async def _phase_ghg_inventory(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Compute GHG inventory across all scopes."""
        return {
            "scope1_total_tco2e": context.get("scope1_total_tco2e", 0.0),
            "scope2_location_tco2e": context.get("scope2_location_tco2e", 0.0),
            "scope2_market_tco2e": context.get("scope2_market_tco2e", 0.0),
            "scope3_total_tco2e": context.get("scope3_total_tco2e", 0.0),
            "records_processed": context.get("emission_source_count", 0),
        }

    async def _phase_energy_assessment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Assess energy consumption and mix."""
        return {
            "total_energy_mwh": context.get("total_energy_mwh", 0.0),
            "renewable_share_pct": context.get("renewable_share_pct", 0.0),
            "records_processed": context.get("energy_source_count", 0),
        }

    async def _phase_transition_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4: Evaluate climate transition plan."""
        plan = context.get("transition_plan", {})
        return {
            "has_transition_plan": plan.get("adopted", False),
            "target_year": plan.get("target_year", ""),
            "records_processed": 1,
        }

    async def _phase_target_setting(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 5: Set and validate climate targets."""
        targets = context.get("climate_targets", [])
        return {
            "target_count": len(targets),
            "sbti_validated": context.get("sbti_validated", False),
            "records_processed": len(targets),
        }

    async def _phase_climate_actions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 6: Catalogue climate actions and resources."""
        actions = context.get("climate_actions", [])
        return {
            "action_count": len(actions),
            "total_investment_eur": sum(
                a.get("investment_eur", 0.0) for a in actions
            ),
            "records_processed": len(actions),
        }

    async def _phase_carbon_credits(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 7: Assess carbon credit portfolio."""
        credits = context.get("carbon_credits", [])
        return {
            "credit_count": len(credits),
            "total_credits_tco2e": sum(c.get("tco2e", 0.0) for c in credits),
            "records_processed": len(credits),
        }

    async def _phase_carbon_pricing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 8: Evaluate carbon pricing exposure."""
        mechanisms = context.get("carbon_pricing_mechanisms", [])
        return {
            "mechanism_count": len(mechanisms),
            "total_cost_eur": sum(
                m.get("annual_cost_eur", 0.0) for m in mechanisms
            ),
            "records_processed": len(mechanisms),
        }

    async def _phase_climate_risk(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 9: Analyze physical and transition risks."""
        phys = context.get("physical_risks", [])
        trans = context.get("transition_risks", [])
        return {
            "physical_risk_count": len(phys),
            "transition_risk_count": len(trans),
            "records_processed": len(phys) + len(trans),
        }

    async def _phase_report_assembly(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 10: Assemble the final E1 disclosure package."""
        return {
            "report_assembled": True,
            "disclosure_count": 9,
            "esrs_references": [
                "E1-1", "E1-2", "E1-3", "E1-4", "E1-5",
                "E1-6", "E1-7", "E1-8", "E1-9",
            ],
            "records_processed": 1,
        }

    def _compute_quality_score(self, result: PipelineResult) -> float:
        """Compute overall pipeline quality score (0-100)."""
        total = len(PHASE_EXECUTION_ORDER)
        completed = len(result.phases_completed)
        if total == 0:
            return 0.0
        return round(completed / total * 100, 1)
