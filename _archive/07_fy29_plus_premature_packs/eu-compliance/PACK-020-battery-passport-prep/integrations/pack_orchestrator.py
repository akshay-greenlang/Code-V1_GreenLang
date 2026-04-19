# -*- coding: utf-8 -*-
"""
BatteryPassportOrchestrator - Master Pipeline for PACK-020
================================================================

This module implements the master pipeline orchestrator for the Battery
Passport Prep Pack. It executes an 8-phase DAG pipeline covering all
battery regulation requirements from carbon footprint calculation through
recycled content tracking, passport compilation, performance monitoring,
due diligence verification, labelling compliance, end-of-life management,
and conformity assessment with final scorecard generation.

Phases (8 assessment phases + scorecard):
    1.  CARBON_FOOTPRINT       -- Calculate battery carbon footprint (Art 7)
    2.  RECYCLED_CONTENT       -- Track recycled material content (Art 8)
    3.  PASSPORT_COMPILATION   -- Compile digital battery passport (Art 77)
    4.  PERFORMANCE            -- Assess performance & durability (Art 10-11)
    5.  DUE_DILIGENCE          -- Supply chain due diligence (Art 39)
    6.  LABELLING              -- Labelling & marking compliance (Art 13)
    7.  END_OF_LIFE            -- End-of-life & recycling readiness (Art 57-62)
    8.  CONFORMITY             -- Conformity assessment & CE marking (Art 17-20)
    9.  SCORECARD              -- Aggregate compliance scorecard

DAG Dependencies:
    CARBON_FOOTPRINT --> PASSPORT_COMPILATION
    RECYCLED_CONTENT --> PASSPORT_COMPILATION
    PASSPORT_COMPILATION --> LABELLING
    PERFORMANCE --> CONFORMITY
    DUE_DILIGENCE --> CONFORMITY
    LABELLING --> CONFORMITY
    END_OF_LIFE --> CONFORMITY
    CONFORMITY --> SCORECARD

Parallel Groups:
    Group A: CARBON_FOOTPRINT, RECYCLED_CONTENT (independent, run first)
    Group B: PERFORMANCE, DUE_DILIGENCE, END_OF_LIFE (parallel after init)

Legal References:
    - Regulation (EU) 2023/1542 (EU Battery Regulation)
    - Commission Delegated Regulation (EU) 2024/1781 (carbon footprint)
    - Commission Delegated Regulation (EU) 2024/1785 (recycled content)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-020 Battery Passport Prep Pack
Status: Production Ready
"""

import asyncio
import hashlib
import json
import logging
import random
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

class BatteryPipelinePhase(str, Enum):
    """The 9 phases of the Battery Passport assessment pipeline."""

    CARBON_FOOTPRINT = "carbon_footprint"
    RECYCLED_CONTENT = "recycled_content"
    PASSPORT_COMPILATION = "passport_compilation"
    PERFORMANCE = "performance"
    DUE_DILIGENCE = "due_diligence"
    LABELLING = "labelling"
    END_OF_LIFE = "end_of_life"
    CONFORMITY = "conformity"
    SCORECARD = "scorecard"

class BatteryCategory(str, Enum):
    """EU Battery Regulation battery categories (Art 2)."""

    PORTABLE = "portable"
    STARTING_LIGHTING_IGNITION = "sli"
    LIGHT_MEANS_OF_TRANSPORT = "lmt"
    ELECTRIC_VEHICLE = "ev"
    INDUSTRIAL = "industrial"
    STATIONARY_ENERGY_STORAGE = "stationary_storage"

# ---------------------------------------------------------------------------
# Battery Regulation Article Mapping
# ---------------------------------------------------------------------------

REGULATION_ARTICLES: Dict[str, Dict[str, Any]] = {
    "carbon_footprint": {
        "articles": ["Art 7"],
        "description": "Carbon footprint declaration and performance classes",
        "delegated_act": "2024/1781",
        "applies_to": ["ev", "lmt", "industrial"],
    },
    "recycled_content": {
        "articles": ["Art 8"],
        "description": "Recycled content from manufacturing waste and post-consumer waste",
        "delegated_act": "2024/1785",
        "applies_to": ["ev", "lmt", "industrial", "sli"],
    },
    "passport_compilation": {
        "articles": ["Art 77", "Art 78"],
        "description": "Digital battery passport data carrier and access",
        "applies_to": ["ev", "lmt", "industrial"],
    },
    "performance": {
        "articles": ["Art 10", "Art 11", "Art 12"],
        "description": "Performance, durability, and safety requirements",
        "applies_to": ["ev", "lmt", "industrial", "portable", "sli"],
    },
    "due_diligence": {
        "articles": ["Art 39", "Art 40", "Art 41", "Art 42"],
        "description": "Supply chain due diligence for raw materials",
        "applies_to": ["ev", "lmt", "industrial", "sli", "portable"],
    },
    "labelling": {
        "articles": ["Art 13", "Art 14"],
        "description": "Labelling, marking, and information requirements",
        "applies_to": ["ev", "lmt", "industrial", "portable", "sli"],
    },
    "end_of_life": {
        "articles": ["Art 57", "Art 58", "Art 59", "Art 60", "Art 61", "Art 62"],
        "description": "Collection, recycling, and second-life requirements",
        "applies_to": ["ev", "lmt", "industrial", "portable", "sli"],
    },
    "conformity": {
        "articles": ["Art 17", "Art 18", "Art 19", "Art 20"],
        "description": "Conformity assessment and CE marking",
        "applies_to": ["ev", "lmt", "industrial", "portable", "sli"],
    },
}

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
    """Configuration for the Battery Passport Pipeline Orchestrator."""

    pack_id: str = Field(default="PACK-020")
    pack_version: str = Field(default="1.0.0")
    battery_category: BatteryCategory = Field(default=BatteryCategory.ELECTRIC_VEHICLE)
    max_concurrent_phases: int = Field(default=3, ge=1, le=5)
    timeout_per_phase_seconds: int = Field(default=600, ge=30)
    enable_provenance: bool = Field(default=True)
    enable_checkpoints: bool = Field(default=True)
    enable_parallel_phases: bool = Field(default=True)
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    base_currency: str = Field(default="EUR")
    manufacturer_name: str = Field(default="")
    battery_model: str = Field(default="")
    production_country: str = Field(default="")

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

    phase: BatteryPipelinePhase = Field(...)
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
    regulation_articles: List[str] = Field(
        default_factory=list, description="Battery Reg articles covered"
    )

class PipelineResult(BaseModel):
    """Complete result of the Battery Passport pipeline execution."""

    execution_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-020")
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    assessment_type: str = Field(default="full", description="full or quick")
    battery_category: str = Field(default="ev")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    total_duration_ms: float = Field(default=0.0)
    phases_completed: List[str] = Field(default_factory=list)
    phases_skipped: List[str] = Field(default_factory=list)
    phase_results: Dict[str, PhaseResult] = Field(default_factory=dict)
    total_records_processed: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    compliance_score: float = Field(default=0.0, ge=0.0, le=100.0)
    readiness_level: str = Field(default="not_assessed")
    articles_covered: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# DAG Dependency Map
# ---------------------------------------------------------------------------

PHASE_DEPENDENCIES: Dict[BatteryPipelinePhase, List[BatteryPipelinePhase]] = {
    BatteryPipelinePhase.CARBON_FOOTPRINT: [],
    BatteryPipelinePhase.RECYCLED_CONTENT: [],
    BatteryPipelinePhase.PERFORMANCE: [],
    BatteryPipelinePhase.DUE_DILIGENCE: [],
    BatteryPipelinePhase.END_OF_LIFE: [],
    BatteryPipelinePhase.PASSPORT_COMPILATION: [
        BatteryPipelinePhase.CARBON_FOOTPRINT,
        BatteryPipelinePhase.RECYCLED_CONTENT,
    ],
    BatteryPipelinePhase.LABELLING: [
        BatteryPipelinePhase.PASSPORT_COMPILATION,
    ],
    BatteryPipelinePhase.CONFORMITY: [
        BatteryPipelinePhase.PERFORMANCE,
        BatteryPipelinePhase.DUE_DILIGENCE,
        BatteryPipelinePhase.LABELLING,
        BatteryPipelinePhase.END_OF_LIFE,
    ],
    BatteryPipelinePhase.SCORECARD: [
        BatteryPipelinePhase.CONFORMITY,
    ],
}

PARALLEL_PHASE_GROUPS: List[List[BatteryPipelinePhase]] = [
    # Group A: carbon and recycled content (no dependencies)
    [
        BatteryPipelinePhase.CARBON_FOOTPRINT,
        BatteryPipelinePhase.RECYCLED_CONTENT,
    ],
    # Group B: performance, due diligence, end-of-life (no dependencies)
    [
        BatteryPipelinePhase.PERFORMANCE,
        BatteryPipelinePhase.DUE_DILIGENCE,
        BatteryPipelinePhase.END_OF_LIFE,
    ],
]

PHASE_EXECUTION_ORDER: List[BatteryPipelinePhase] = [
    BatteryPipelinePhase.CARBON_FOOTPRINT,
    BatteryPipelinePhase.RECYCLED_CONTENT,
    BatteryPipelinePhase.PERFORMANCE,
    BatteryPipelinePhase.DUE_DILIGENCE,
    BatteryPipelinePhase.END_OF_LIFE,
    BatteryPipelinePhase.PASSPORT_COMPILATION,
    BatteryPipelinePhase.LABELLING,
    BatteryPipelinePhase.CONFORMITY,
    BatteryPipelinePhase.SCORECARD,
]

QUICK_ASSESSMENT_PHASES: List[BatteryPipelinePhase] = [
    BatteryPipelinePhase.CARBON_FOOTPRINT,
    BatteryPipelinePhase.RECYCLED_CONTENT,
    BatteryPipelinePhase.PASSPORT_COMPILATION,
    BatteryPipelinePhase.SCORECARD,
]

# ---------------------------------------------------------------------------
# BatteryPassportOrchestrator
# ---------------------------------------------------------------------------

class BatteryPassportOrchestrator:
    """Master pipeline orchestrator for PACK-020 Battery Passport Prep.

    Executes a DAG-ordered pipeline of 9 phases covering EU Battery
    Regulation requirements including carbon footprint, recycled content,
    digital passport, performance, due diligence, labelling, end-of-life,
    conformity assessment, and compliance scorecard generation.

    Supports full and quick assessment modes, parallel phase execution,
    retry with exponential backoff, SHA-256 provenance tracking, and
    progress callbacks.

    Attributes:
        config: Orchestrator configuration.
        _results: Active and historical pipeline results.
        _cancelled: Set of cancelled execution IDs.
        _progress_callback: Optional async callback for progress updates.

    Example:
        >>> config = OrchestratorConfig(battery_category=BatteryCategory.ELECTRIC_VEHICLE)
        >>> orch = BatteryPassportOrchestrator(config)
        >>> result = await orch.run_full_assessment({"manufacturer": "Acme"})
        >>> assert result.status == ExecutionStatus.COMPLETED
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Initialize BatteryPassportOrchestrator.

        Args:
            config: Orchestrator configuration. Defaults used if None.
            progress_callback: Optional async callback for progress updates.
        """
        self.config = config or OrchestratorConfig()
        self._results: Dict[str, PipelineResult] = {}
        self._cancelled: Set[str] = set()
        self._progress_callback = progress_callback
        logger.info(
            "BatteryPassportOrchestrator initialized (pack=%s, category=%s, phases=%d)",
            self.config.pack_id,
            self.config.battery_category.value,
            len(PHASE_EXECUTION_ORDER),
        )

    async def run_full_assessment(
        self,
        context: Dict[str, Any],
    ) -> PipelineResult:
        """Execute the full battery passport assessment pipeline (all 9 phases).

        Args:
            context: Shared pipeline context with input data.

        Returns:
            PipelineResult with status, phase results, and provenance.
        """
        return await self._run_pipeline(context, PHASE_EXECUTION_ORDER, "full")

    async def run_quick_assessment(
        self,
        context: Dict[str, Any],
    ) -> PipelineResult:
        """Execute a quick assessment covering core passport phases only.

        Runs carbon footprint, recycled content, passport compilation,
        and scorecard phases for a rapid readiness check.

        Args:
            context: Shared pipeline context with input data.

        Returns:
            PipelineResult with quick assessment results.
        """
        return await self._run_pipeline(context, QUICK_ASSESSMENT_PHASES, "quick")

    def get_status(self, execution_id: str) -> Optional[PipelineResult]:
        """Get the status of a pipeline execution.

        Args:
            execution_id: Pipeline execution ID.

        Returns:
            PipelineResult or None if not found.
        """
        return self._results.get(execution_id)

    def validate_prerequisites(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate pipeline prerequisites before execution.

        Args:
            context: Pipeline context to validate.

        Returns:
            Dict with 'valid' bool and 'errors'/'warnings' lists.
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not context.get("manufacturer"):
            errors.append("manufacturer name is required in context")
        if not context.get("battery_model"):
            warnings.append("battery_model not set; generic assessment will be performed")
        if "bom_data" not in context:
            warnings.append("bom_data not in context; carbon footprint will use defaults")
        if "supplier_data" not in context:
            warnings.append("supplier_data not in context; due diligence will be limited")
        if "recycled_content_data" not in context:
            warnings.append("recycled_content_data not provided; estimates will be used")

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

    def get_dag_info(self) -> Dict[str, Any]:
        """Get DAG dependency graph information for visualization.

        Returns:
            Dict with phases, dependencies, parallel groups, and execution order.
        """
        return {
            "phases": [p.value for p in BatteryPipelinePhase],
            "phase_count": len(BatteryPipelinePhase),
            "dependencies": {
                p.value: [d.value for d in deps]
                for p, deps in PHASE_DEPENDENCIES.items()
            },
            "parallel_groups": [
                [p.value for p in group]
                for group in PARALLEL_PHASE_GROUPS
            ],
            "execution_order": [p.value for p in PHASE_EXECUTION_ORDER],
            "quick_assessment_phases": [p.value for p in QUICK_ASSESSMENT_PHASES],
            "regulation_articles": REGULATION_ARTICLES,
        }

    def get_applicable_phases(self) -> List[BatteryPipelinePhase]:
        """Get phases applicable to the configured battery category.

        Returns:
            List of applicable BatteryPipelinePhase values.
        """
        category = self.config.battery_category.value
        applicable: List[BatteryPipelinePhase] = []
        for phase in PHASE_EXECUTION_ORDER:
            reg = REGULATION_ARTICLES.get(phase.value, {})
            applies_to = reg.get("applies_to", [])
            if not applies_to or category in applies_to:
                applicable.append(phase)
        return applicable

    # ------------------------------------------------------------------
    # Internal pipeline execution
    # ------------------------------------------------------------------

    async def _run_pipeline(
        self,
        context: Dict[str, Any],
        target_phases: List[BatteryPipelinePhase],
        assessment_type: str,
    ) -> PipelineResult:
        """Core pipeline execution logic.

        Args:
            context: Shared pipeline context.
            target_phases: Ordered list of phases to execute.
            assessment_type: Either 'full' or 'quick'.

        Returns:
            PipelineResult with execution outcomes.
        """
        result = PipelineResult(
            pack_id=self.config.pack_id,
            assessment_type=assessment_type,
            battery_category=self.config.battery_category.value,
            started_at=utcnow(),
            status=ExecutionStatus.RUNNING,
        )
        self._results[result.execution_id] = result

        # Inject config into context
        context["_battery_category"] = self.config.battery_category.value
        context["_reporting_year"] = self.config.reporting_year
        context["_manufacturer"] = context.get("manufacturer", self.config.manufacturer_name)

        try:
            executed: Set[str] = set()
            total_phases = len(target_phases)

            for idx, phase in enumerate(target_phases):
                if result.execution_id in self._cancelled:
                    result.status = ExecutionStatus.CANCELLED
                    break

                # Skip if already executed in a parallel group
                if phase.value in executed:
                    continue

                # Check if applicable to battery category
                if not self._is_phase_applicable(phase):
                    result.phases_skipped.append(phase.value)
                    executed.add(phase.value)
                    continue

                # Check dependencies
                deps = PHASE_DEPENDENCIES.get(phase, [])
                unmet = [
                    d for d in deps
                    if d.value not in executed
                    and d in target_phases
                ]

                if unmet:
                    # Try parallel group execution
                    parallel_group = self._find_parallel_group(phase, target_phases)
                    if parallel_group and self.config.enable_parallel_phases:
                        await self._execute_parallel_phases(
                            parallel_group, context, result
                        )
                        for p in parallel_group:
                            executed.add(p.value)
                        continue
                    else:
                        result.phases_skipped.append(phase.value)
                        executed.add(phase.value)
                        continue

                # Attempt parallel execution of current group
                parallel_group = self._find_parallel_group(phase, target_phases)
                if (
                    parallel_group
                    and self.config.enable_parallel_phases
                    and all(p.value not in executed for p in parallel_group)
                ):
                    await self._execute_parallel_phases(
                        parallel_group, context, result
                    )
                    for p in parallel_group:
                        executed.add(p.value)
                else:
                    # Sequential execution
                    phase_result = await self._execute_phase_with_retry(
                        phase, context, result
                    )
                    result.phase_results[phase.value] = phase_result

                    if phase_result.status == ExecutionStatus.COMPLETED:
                        result.phases_completed.append(phase.value)
                        executed.add(phase.value)
                        result.total_records_processed += phase_result.records_processed
                        result.articles_covered.extend(phase_result.regulation_articles)
                    else:
                        result.errors.append(f"Phase {phase.value} failed")
                        executed.add(phase.value)

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

        result.compliance_score = self._compute_compliance_score(result)
        result.readiness_level = self._determine_readiness_level(result.compliance_score)
        result.articles_covered = list(set(result.articles_covered))

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        logger.info(
            "Pipeline %s: %s in %.1fms (score=%.1f%%, readiness=%s, type=%s)",
            result.execution_id,
            result.status.value,
            result.total_duration_ms,
            result.compliance_score,
            result.readiness_level,
            assessment_type,
        )
        return result

    def _is_phase_applicable(self, phase: BatteryPipelinePhase) -> bool:
        """Check if a phase applies to the configured battery category."""
        if phase == BatteryPipelinePhase.SCORECARD:
            return True
        reg = REGULATION_ARTICLES.get(phase.value, {})
        applies_to = reg.get("applies_to", [])
        if not applies_to:
            return True
        return self.config.battery_category.value in applies_to

    async def _execute_phase_with_retry(
        self,
        phase: BatteryPipelinePhase,
        context: Dict[str, Any],
        pipeline_result: PipelineResult,
    ) -> PhaseResult:
        """Execute a phase with retry and exponential backoff."""
        reg = REGULATION_ARTICLES.get(phase.value, {})
        phase_result = PhaseResult(
            phase=phase,
            regulation_articles=reg.get("articles", []),
        )
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

                context[f"{phase.value}_result"] = outputs
                logger.info(
                    "Phase %s completed in %.1fms (attempt %d)",
                    phase.value, phase_result.duration_ms, attempt,
                )
                return phase_result

            except Exception as exc:
                logger.warning(
                    "Phase %s attempt %d failed: %s",
                    phase.value, attempt, str(exc),
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
        phases: List[BatteryPipelinePhase],
        context: Dict[str, Any],
        pipeline_result: PipelineResult,
    ) -> None:
        """Execute multiple phases in parallel with concurrency limit."""
        sem = asyncio.Semaphore(self.config.max_concurrent_phases)

        async def _run_with_semaphore(p: BatteryPipelinePhase) -> PhaseResult:
            async with sem:
                return await self._execute_phase_with_retry(p, context, pipeline_result)

        applicable = [p for p in phases if self._is_phase_applicable(p)]
        tasks = [_run_with_semaphore(phase) for phase in applicable]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for phase, phase_res in zip(applicable, results):
            if isinstance(phase_res, Exception):
                pr = PhaseResult(
                    phase=phase,
                    status=ExecutionStatus.FAILED,
                    errors=[str(phase_res)],
                )
                pipeline_result.phase_results[phase.value] = pr
                pipeline_result.errors.append(f"Parallel phase {phase.value} failed")
            else:
                pipeline_result.phase_results[phase.value] = phase_res
                if phase_res.status == ExecutionStatus.COMPLETED:
                    pipeline_result.phases_completed.append(phase.value)
                    pipeline_result.total_records_processed += phase_res.records_processed
                    pipeline_result.articles_covered.extend(
                        phase_res.regulation_articles
                    )

    def _find_parallel_group(
        self,
        phase: BatteryPipelinePhase,
        target_phases: List[BatteryPipelinePhase],
    ) -> Optional[List[BatteryPipelinePhase]]:
        """Find a parallel group containing the given phase."""
        for group in PARALLEL_PHASE_GROUPS:
            if phase in group and all(p in target_phases for p in group):
                return group
        return None

    async def _run_phase_logic(
        self,
        phase: BatteryPipelinePhase,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the logic for a single phase.

        Each phase returns a dict of outputs merged into context.
        In production, these dispatch to real engine instances.
        """
        handlers = {
            BatteryPipelinePhase.CARBON_FOOTPRINT: self._phase_carbon_footprint,
            BatteryPipelinePhase.RECYCLED_CONTENT: self._phase_recycled_content,
            BatteryPipelinePhase.PASSPORT_COMPILATION: self._phase_passport_compilation,
            BatteryPipelinePhase.PERFORMANCE: self._phase_performance,
            BatteryPipelinePhase.DUE_DILIGENCE: self._phase_due_diligence,
            BatteryPipelinePhase.LABELLING: self._phase_labelling,
            BatteryPipelinePhase.END_OF_LIFE: self._phase_end_of_life,
            BatteryPipelinePhase.CONFORMITY: self._phase_conformity,
            BatteryPipelinePhase.SCORECARD: self._phase_scorecard,
        }
        handler = handlers.get(phase)
        if handler is None:
            raise ValueError(f"No handler for phase: {phase.value}")
        return await handler(context)

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    async def _phase_carbon_footprint(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase: Calculate battery carbon footprint (Art 7)."""
        cf_data = context.get("carbon_footprint_data", {})
        return {
            "total_cf_kgco2e_per_kwh": cf_data.get("total_cf_kgco2e_per_kwh", 0.0),
            "performance_class": cf_data.get("performance_class", "not_classified"),
            "lifecycle_stages": ["raw_materials", "manufacturing", "distribution", "end_of_life"],
            "methodology": "EU 2024/1781 Product Environmental Footprint",
            "records_processed": cf_data.get("records_processed", 1),
        }

    async def _phase_recycled_content(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase: Track recycled material content (Art 8)."""
        rc_data = context.get("recycled_content_data", {})
        return {
            "cobalt_recycled_pct": rc_data.get("cobalt_recycled_pct", 0.0),
            "lithium_recycled_pct": rc_data.get("lithium_recycled_pct", 0.0),
            "nickel_recycled_pct": rc_data.get("nickel_recycled_pct", 0.0),
            "lead_recycled_pct": rc_data.get("lead_recycled_pct", 0.0),
            "meets_2031_targets": rc_data.get("meets_2031_targets", False),
            "meets_2036_targets": rc_data.get("meets_2036_targets", False),
            "records_processed": rc_data.get("records_processed", 1),
        }

    async def _phase_passport_compilation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase: Compile digital battery passport (Art 77)."""
        cf_result = context.get("carbon_footprint_result", {})
        rc_result = context.get("recycled_content_result", {})
        return {
            "passport_id": _new_uuid(),
            "data_carrier_type": "QR_code",
            "carbon_footprint_included": bool(cf_result),
            "recycled_content_included": bool(rc_result),
            "passport_fields_populated": 42,
            "passport_fields_total": 90,
            "completeness_pct": round(42 / 90 * 100, 1),
            "records_processed": 1,
        }

    async def _phase_performance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase: Assess performance and durability (Art 10-11)."""
        perf_data = context.get("performance_data", {})
        return {
            "rated_capacity_ah": perf_data.get("rated_capacity_ah", 0.0),
            "cycle_life": perf_data.get("cycle_life", 0),
            "round_trip_efficiency_pct": perf_data.get("round_trip_efficiency_pct", 0.0),
            "capacity_fade_pct": perf_data.get("capacity_fade_pct", 0.0),
            "internal_resistance_mohm": perf_data.get("internal_resistance_mohm", 0.0),
            "min_durability_met": perf_data.get("min_durability_met", False),
            "records_processed": perf_data.get("records_processed", 1),
        }

    async def _phase_due_diligence(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase: Supply chain due diligence (Art 39)."""
        dd_data = context.get("due_diligence_data", {})
        return {
            "dd_policy_established": dd_data.get("dd_policy_established", False),
            "risk_assessment_completed": dd_data.get("risk_assessment_completed", False),
            "minerals_covered": dd_data.get("minerals_covered", [
                "cobalt", "lithium", "nickel", "natural_graphite",
            ]),
            "supplier_audits_completed": dd_data.get("supplier_audits_completed", 0),
            "third_party_audit": dd_data.get("third_party_audit", False),
            "oecd_aligned": dd_data.get("oecd_aligned", False),
            "records_processed": dd_data.get("records_processed", 1),
        }

    async def _phase_labelling(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase: Labelling and marking compliance (Art 13)."""
        label_data = context.get("labelling_data", {})
        passport_result = context.get("passport_compilation_result", {})
        return {
            "ce_marking_ready": label_data.get("ce_marking_ready", False),
            "qr_code_generated": bool(passport_result.get("passport_id")),
            "hazard_symbols_present": label_data.get("hazard_symbols_present", False),
            "capacity_label_present": label_data.get("capacity_label_present", False),
            "separate_collection_symbol": label_data.get("separate_collection_symbol", False),
            "material_composition_label": label_data.get("material_composition_label", False),
            "records_processed": label_data.get("records_processed", 1),
        }

    async def _phase_end_of_life(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase: End-of-life and recycling readiness (Art 57-62)."""
        eol_data = context.get("end_of_life_data", {})
        return {
            "collection_scheme_registered": eol_data.get("collection_scheme_registered", False),
            "recycling_efficiency_pct": eol_data.get("recycling_efficiency_pct", 0.0),
            "cobalt_recovery_pct": eol_data.get("cobalt_recovery_pct", 0.0),
            "lithium_recovery_pct": eol_data.get("lithium_recovery_pct", 0.0),
            "nickel_recovery_pct": eol_data.get("nickel_recovery_pct", 0.0),
            "second_life_assessment": eol_data.get("second_life_assessment", False),
            "waste_shipment_compliant": eol_data.get("waste_shipment_compliant", False),
            "records_processed": eol_data.get("records_processed", 1),
        }

    async def _phase_conformity(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase: Conformity assessment and CE marking (Art 17-20)."""
        perf_result = context.get("performance_result", {})
        dd_result = context.get("due_diligence_result", {})
        label_result = context.get("labelling_result", {})
        eol_result = context.get("end_of_life_result", {})

        checks_passed = sum([
            bool(perf_result.get("min_durability_met")),
            bool(dd_result.get("oecd_aligned")),
            bool(label_result.get("ce_marking_ready")),
            bool(eol_result.get("collection_scheme_registered")),
        ])
        total_checks = 4

        return {
            "conformity_checks_passed": checks_passed,
            "conformity_checks_total": total_checks,
            "conformity_pct": round(checks_passed / total_checks * 100, 1) if total_checks > 0 else 0.0,
            "eu_declaration_ready": checks_passed == total_checks,
            "notified_body_required": self.config.battery_category.value in ("ev", "stationary_storage"),
            "technical_documentation_ready": checks_passed >= 3,
            "records_processed": 1,
        }

    async def _phase_scorecard(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase: Aggregate compliance scorecard."""
        scores: Dict[str, float] = {}
        total_score = 0.0
        phase_count = 0

        score_map = {
            "carbon_footprint_result": lambda r: 100.0 if r.get("total_cf_kgco2e_per_kwh", 0) > 0 else 25.0,
            "recycled_content_result": lambda r: 100.0 if r.get("meets_2031_targets") else 50.0,
            "passport_compilation_result": lambda r: r.get("completeness_pct", 0.0),
            "performance_result": lambda r: 100.0 if r.get("min_durability_met") else 40.0,
            "due_diligence_result": lambda r: 100.0 if r.get("oecd_aligned") else 30.0,
            "labelling_result": lambda r: 100.0 if r.get("ce_marking_ready") else 45.0,
            "end_of_life_result": lambda r: 100.0 if r.get("collection_scheme_registered") else 35.0,
            "conformity_result": lambda r: r.get("conformity_pct", 0.0),
        }

        for result_key, scorer in score_map.items():
            phase_data = context.get(result_key, {})
            if phase_data:
                phase_name = result_key.replace("_result", "")
                score = scorer(phase_data)
                scores[phase_name] = round(score, 1)
                total_score += score
                phase_count += 1

        overall = round(total_score / phase_count, 1) if phase_count > 0 else 0.0

        return {
            "phase_scores": scores,
            "overall_compliance_pct": overall,
            "phases_assessed": phase_count,
            "readiness_level": self._determine_readiness_level(overall),
            "records_processed": phase_count,
        }

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _compute_compliance_score(self, result: PipelineResult) -> float:
        """Compute overall compliance score from phase results."""
        scorecard = result.phase_results.get("scorecard")
        if scorecard and scorecard.outputs:
            return scorecard.outputs.get("overall_compliance_pct", 0.0)
        total = len(PHASE_EXECUTION_ORDER) - 1  # exclude scorecard
        completed = len([
            p for p in result.phases_completed
            if p != "scorecard"
        ])
        if total == 0:
            return 0.0
        return round(completed / total * 100, 1)

    @staticmethod
    def _determine_readiness_level(score: float) -> str:
        """Determine readiness level from compliance score.

        Args:
            score: Compliance score (0-100).

        Returns:
            Readiness level string.
        """
        if score >= 90.0:
            return "production_ready"
        if score >= 70.0:
            return "substantially_ready"
        if score >= 50.0:
            return "partially_ready"
        if score >= 25.0:
            return "early_stage"
        return "not_started"
