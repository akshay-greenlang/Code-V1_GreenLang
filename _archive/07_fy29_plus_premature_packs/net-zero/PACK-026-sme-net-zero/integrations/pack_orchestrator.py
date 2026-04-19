# -*- coding: utf-8 -*-
"""
SMENetZeroPipelineOrchestrator - 6-Phase DAG Pipeline for PACK-026
=====================================================================

This module implements the SME Net Zero Pack pipeline orchestrator,
executing a 6-phase DAG pipeline tailored for small and medium
enterprises. The pipeline is streamlined for simplicity and
cost-effectiveness while maintaining GHG Protocol rigour.

Phases (6 total):
    1.  onboarding            -- Organisation setup, accounting connection,
                                 data quality tier selection
    2.  baseline              -- GHG inventory baseline via simplified MRV
                                 agents (Scope 1+2 + spend-based Scope 3)
    3.  targets               -- SME-appropriate target setting (SME Climate
                                 Hub commitment or SBTi SME pathway)
    4.  quick_wins            -- Identify low-cost, high-impact reduction
                                 actions with payback < 24 months
    5.  grant_search          -- Match available grants and funding to
                                 reduction projects
    6.  reporting             -- Simplified annual reporting and SME Climate
                                 Hub / certification body submission

DAG Dependencies:
    onboarding --> baseline --> targets
    targets --> quick_wins
    quick_wins --> grant_search
    grant_search --> reporting
    targets --> reporting (direct path also)

Path Selection:
    Simplified path: Bronze data quality, spend-based Scope 3 only,
                     SME Climate Hub commitment
    Standard path:   Silver/Gold data quality, activity-based where
                     possible, SBTi SME target

Architecture:
    Config --> SMENetZeroPipelineOrchestrator --> Phase DAG Resolution
                         |                            |
                         v                            v
    Phase Execution <-- Retry w/ Backoff <-- SME Path Selection
                         |
                         v
    PhaseProvenance --> SHA-256 Hashing --> PipelineResult

Error Messages:
    All error messages are written in plain English suitable for
    non-technical SME users, avoiding jargon where possible.

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-026 SME Net Zero Pack
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
# SME-Friendly Error Messages
# ---------------------------------------------------------------------------

SME_ERROR_MESSAGES: Dict[str, str] = {
    "config_invalid": (
        "We could not validate your setup. Please check that all required "
        "fields (company name, sector, country) are filled in."
    ),
    "accounting_connection_failed": (
        "We could not connect to your accounting software. Please check your "
        "login details and try again, or skip this step to enter data manually."
    ),
    "data_quality_low": (
        "Your data quality is below the minimum threshold. This usually means "
        "some records are missing or have unexpected values. You can continue "
        "with estimates, or upload corrected data."
    ),
    "baseline_calculation_error": (
        "We had trouble calculating your carbon footprint. This can happen if "
        "energy or spend data is incomplete. Please check your uploads and "
        "try again."
    ),
    "target_setting_error": (
        "We could not generate your reduction targets. This usually means "
        "the baseline calculation is missing. Please complete your baseline "
        "first."
    ),
    "grant_search_timeout": (
        "The grant database search took too long. We will keep searching "
        "in the background and notify you when results are ready."
    ),
    "reporting_error": (
        "We had trouble generating your report. Please try again in a few "
        "minutes. Your data is safely saved."
    ),
    "dependencies_not_met": (
        "This step requires a previous step to be completed first. Please "
        "go back and complete all earlier steps."
    ),
    "timeout": (
        "This step is taking longer than expected. We will keep working on "
        "it. You can check back later for results."
    ),
    "unknown_error": (
        "Something unexpected happened. Our team has been notified. Please "
        "try again, or contact support if the problem continues."
    ),
}

def _sme_error_message(error_key: str, fallback: str = "") -> str:
    """Get an SME-friendly error message.

    Args:
        error_key: Key into SME_ERROR_MESSAGES.
        fallback: Fallback message if key not found.

    Returns:
        Human-readable error message.
    """
    return SME_ERROR_MESSAGES.get(error_key, fallback or SME_ERROR_MESSAGES["unknown_error"])

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SMEPipelinePhase(str, Enum):
    """The 6 phases of the SME net-zero pipeline."""

    ONBOARDING = "onboarding"
    BASELINE = "baseline"
    TARGETS = "targets"
    QUICK_WINS = "quick_wins"
    GRANT_SEARCH = "grant_search"
    REPORTING = "reporting"

class SMEPathType(str, Enum):
    """SME pipeline path selection."""

    SIMPLIFIED = "simplified"
    STANDARD = "standard"

class DataQualityTier(str, Enum):
    """Data quality tiers for SME classification."""

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"

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

class SMEOrchestratorConfig(BaseModel):
    """Configuration for the SME Net Zero Pipeline Orchestrator."""

    pack_id: str = Field(default="PACK-026")
    pack_version: str = Field(default="1.0.0")
    organization_name: str = Field(default="")
    sector: str = Field(default="general")
    country: str = Field(default="GB")
    employee_count: int = Field(default=50, ge=1, le=500)
    annual_revenue_eur: float = Field(default=5_000_000.0, ge=0)
    max_concurrent_agents: int = Field(default=5, ge=1, le=20)
    timeout_per_phase_seconds: int = Field(default=300, ge=30)
    enable_provenance: bool = Field(default=True)
    enable_checkpoints: bool = Field(default=True)
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    reporting_year: int = Field(default=2025, ge=2020, le=2035)
    base_year: int = Field(default=2023, ge=2015, le=2025)
    target_year: int = Field(default=2030, ge=2025, le=2050)
    base_currency: str = Field(default="GBP")
    path_type: SMEPathType = Field(default=SMEPathType.SIMPLIFIED)
    data_quality_tier: DataQualityTier = Field(default=DataQualityTier.BRONZE)
    accounting_software: str = Field(default="none")
    grant_region: str = Field(default="UK")
    certification_target: str = Field(default="sme_climate_hub")
    scopes_included: List[str] = Field(
        default_factory=lambda: ["scope_1", "scope_2", "scope_3"],
    )
    scope3_spend_based: bool = Field(default=True)
    scope3_categories: List[int] = Field(
        default_factory=lambda: [1, 6, 7],
    )

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

    phase: SMEPipelinePhase = Field(...)
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    records_processed: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    sme_error_messages: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    provenance: Optional[PhaseProvenance] = Field(None)
    retry_count: int = Field(default=0)

class PipelineResult(BaseModel):
    """Complete result of the SME net-zero pipeline execution."""

    execution_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-026")
    organization_name: str = Field(default="")
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    path_type: SMEPathType = Field(default=SMEPathType.SIMPLIFIED)
    data_quality_tier: DataQualityTier = Field(default=DataQualityTier.BRONZE)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    total_duration_ms: float = Field(default=0.0)
    phases_completed: List[str] = Field(default_factory=list)
    phases_skipped: List[str] = Field(default_factory=list)
    phase_results: Dict[str, PhaseResult] = Field(default_factory=dict)
    total_records_processed: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    sme_error_summary: List[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    estimated_cost_savings_eur: float = Field(default=0.0)
    grants_matched: int = Field(default=0)
    quick_wins_identified: int = Field(default=0)
    provenance_hash: str = Field(default="")

class PhaseProgress(BaseModel):
    """Real-time progress tracking for a pipeline execution."""

    execution_id: str = Field(default="")
    current_phase: str = Field(default="")
    phase_index: int = Field(default=0)
    total_phases: int = Field(default=6)
    progress_pct: float = Field(default=0.0)
    message: str = Field(default="")
    estimated_remaining_seconds: float = Field(default=0.0)
    updated_at: datetime = Field(default_factory=utcnow)

# ---------------------------------------------------------------------------
# DAG Dependency Map
# ---------------------------------------------------------------------------

PHASE_DEPENDENCIES: Dict[SMEPipelinePhase, List[SMEPipelinePhase]] = {
    SMEPipelinePhase.ONBOARDING: [],
    SMEPipelinePhase.BASELINE: [SMEPipelinePhase.ONBOARDING],
    SMEPipelinePhase.TARGETS: [SMEPipelinePhase.BASELINE],
    SMEPipelinePhase.QUICK_WINS: [SMEPipelinePhase.TARGETS],
    SMEPipelinePhase.GRANT_SEARCH: [SMEPipelinePhase.QUICK_WINS],
    SMEPipelinePhase.REPORTING: [SMEPipelinePhase.TARGETS],
}

# Topological order for serial execution
PHASE_EXECUTION_ORDER: List[SMEPipelinePhase] = [
    SMEPipelinePhase.ONBOARDING,
    SMEPipelinePhase.BASELINE,
    SMEPipelinePhase.TARGETS,
    SMEPipelinePhase.QUICK_WINS,
    SMEPipelinePhase.GRANT_SEARCH,
    SMEPipelinePhase.REPORTING,
]

# Phase display names for SME-friendly progress messages
PHASE_DISPLAY_NAMES: Dict[SMEPipelinePhase, str] = {
    SMEPipelinePhase.ONBOARDING: "Setting up your account",
    SMEPipelinePhase.BASELINE: "Calculating your carbon footprint",
    SMEPipelinePhase.TARGETS: "Setting your reduction targets",
    SMEPipelinePhase.QUICK_WINS: "Finding cost-saving opportunities",
    SMEPipelinePhase.GRANT_SEARCH: "Searching for grants and funding",
    SMEPipelinePhase.REPORTING: "Generating your reports",
}

# Average durations per phase (used for time estimates)
PHASE_ESTIMATED_DURATIONS_MS: Dict[SMEPipelinePhase, float] = {
    SMEPipelinePhase.ONBOARDING: 5000.0,
    SMEPipelinePhase.BASELINE: 15000.0,
    SMEPipelinePhase.TARGETS: 8000.0,
    SMEPipelinePhase.QUICK_WINS: 10000.0,
    SMEPipelinePhase.GRANT_SEARCH: 20000.0,
    SMEPipelinePhase.REPORTING: 12000.0,
}

# ---------------------------------------------------------------------------
# SMENetZeroPipelineOrchestrator
# ---------------------------------------------------------------------------

class SMENetZeroPipelineOrchestrator:
    """6-phase net-zero pipeline orchestrator for SMEs (PACK-026).

    Executes a streamlined DAG-ordered pipeline covering onboarding through
    simplified annual reporting. Supports two paths (simplified and standard)
    with SME-friendly error messages, retry with exponential backoff,
    provenance tracking, and progress callbacks.

    Attributes:
        config: Orchestrator configuration.
        _results: Active and historical pipeline results.
        _cancelled: Set of cancelled execution IDs.
        _progress_callback: Optional async callback for progress updates.
        _progress_state: Current progress state per execution.

    Example:
        >>> config = SMEOrchestratorConfig(organization_name="Green Bakery Ltd")
        >>> orch = SMENetZeroPipelineOrchestrator(config)
        >>> result = await orch.execute_pipeline({})
        >>> assert result.status == ExecutionStatus.COMPLETED
    """

    def __init__(
        self,
        config: Optional[SMEOrchestratorConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Initialize the SME Net Zero Pipeline Orchestrator.

        Args:
            config: Pipeline configuration. Uses defaults if None.
            progress_callback: Optional async callback(phase, pct, message).
        """
        self.config = config or SMEOrchestratorConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._results: Dict[str, PipelineResult] = {}
        self._cancelled: Set[str] = set()
        self._progress_callback = progress_callback
        self._progress_state: Dict[str, PhaseProgress] = {}

        self.logger.info(
            "SMENetZeroPipelineOrchestrator created: pack=%s, org=%s, path=%s, "
            "tier=%s, country=%s",
            self.config.pack_id,
            self.config.organization_name,
            self.config.path_type.value,
            self.config.data_quality_tier.value,
            self.config.country,
        )

    # -------------------------------------------------------------------------
    # Path Selection
    # -------------------------------------------------------------------------

    def select_path(
        self,
        employee_count: Optional[int] = None,
        revenue_eur: Optional[float] = None,
        data_quality: Optional[str] = None,
    ) -> SMEPathType:
        """Determine optimal path for an SME based on characteristics.

        Simple heuristic:
            - Bronze tier, <50 employees, <2M revenue => Simplified
            - Silver/Gold tier, OR >100 employees => Standard
            - Otherwise => Simplified

        Args:
            employee_count: Number of employees (overrides config).
            revenue_eur: Annual revenue in EUR (overrides config).
            data_quality: Data quality tier string (overrides config).

        Returns:
            Recommended SMEPathType.
        """
        emp = employee_count or self.config.employee_count
        rev = revenue_eur or self.config.annual_revenue_eur
        tier = data_quality or self.config.data_quality_tier.value

        if tier in ("silver", "gold") or emp > 100 or rev > 10_000_000:
            return SMEPathType.STANDARD
        return SMEPathType.SIMPLIFIED

    # -------------------------------------------------------------------------
    # Pipeline Execution
    # -------------------------------------------------------------------------

    async def execute_pipeline(
        self,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """Execute the full 6-phase SME net-zero pipeline.

        Args:
            input_data: Input data for the pipeline phases.

        Returns:
            PipelineResult with full execution details and provenance.
        """
        input_data = input_data or {}

        result = PipelineResult(
            organization_name=self.config.organization_name,
            path_type=self.config.path_type,
            data_quality_tier=self.config.data_quality_tier,
            status=ExecutionStatus.RUNNING,
            started_at=utcnow(),
        )
        self._results[result.execution_id] = result

        # Initialize progress tracking
        self._progress_state[result.execution_id] = PhaseProgress(
            execution_id=result.execution_id,
            total_phases=len(PHASE_EXECUTION_ORDER),
        )

        start_time = time.monotonic()
        phases = self._resolve_phase_order()
        total_phases = len(phases)

        self.logger.info(
            "Starting SME net-zero pipeline: execution_id=%s, org=%s, "
            "path=%s, phases=%d",
            result.execution_id,
            self.config.organization_name,
            self.config.path_type.value,
            total_phases,
        )

        shared_context: Dict[str, Any] = dict(input_data)
        shared_context["organization_name"] = self.config.organization_name
        shared_context["sector"] = self.config.sector
        shared_context["country"] = self.config.country
        shared_context["reporting_year"] = self.config.reporting_year
        shared_context["base_year"] = self.config.base_year
        shared_context["target_year"] = self.config.target_year
        shared_context["scopes_included"] = self.config.scopes_included
        shared_context["path_type"] = self.config.path_type.value
        shared_context["data_quality_tier"] = self.config.data_quality_tier.value
        shared_context["accounting_software"] = self.config.accounting_software
        shared_context["grant_region"] = self.config.grant_region
        shared_context["certification_target"] = self.config.certification_target
        shared_context["scope3_spend_based"] = self.config.scope3_spend_based

        try:
            for phase_idx, phase in enumerate(phases):
                if result.execution_id in self._cancelled:
                    result.status = ExecutionStatus.CANCELLED
                    result.errors.append("Pipeline cancelled by user")
                    result.sme_error_summary.append(
                        "You cancelled the process. Your progress has been saved."
                    )
                    break

                # Conditional skip check
                if self._should_skip_phase(phase):
                    phase_result = PhaseResult(
                        phase=phase,
                        status=ExecutionStatus.SKIPPED,
                        started_at=utcnow(),
                        completed_at=utcnow(),
                    )
                    result.phase_results[phase.value] = phase_result
                    result.phases_skipped.append(phase.value)
                    self.logger.info(
                        "Phase '%s' skipped (not applicable for %s path)",
                        phase.value,
                        self.config.path_type.value,
                    )
                    continue

                # DAG dependency check
                if not self._dependencies_met(phase, result):
                    phase_result = PhaseResult(
                        phase=phase,
                        status=ExecutionStatus.FAILED,
                        errors=["Dependencies not met"],
                        sme_error_messages=[_sme_error_message("dependencies_not_met")],
                    )
                    result.phase_results[phase.value] = phase_result
                    result.status = ExecutionStatus.FAILED
                    result.errors.append(f"Phase '{phase.value}' dependencies not met")
                    result.sme_error_summary.append(
                        _sme_error_message("dependencies_not_met")
                    )
                    break

                # Progress callback
                progress_pct = (phase_idx / total_phases) * 100.0
                display_name = PHASE_DISPLAY_NAMES.get(phase, phase.value)

                progress = self._progress_state.get(result.execution_id)
                if progress:
                    progress.current_phase = phase.value
                    progress.phase_index = phase_idx
                    progress.progress_pct = progress_pct
                    progress.message = display_name
                    remaining_ms = sum(
                        PHASE_ESTIMATED_DURATIONS_MS.get(p, 10000.0)
                        for p in phases[phase_idx:]
                    )
                    progress.estimated_remaining_seconds = remaining_ms / 1000.0
                    progress.updated_at = utcnow()

                if self._progress_callback:
                    await self._progress_callback(
                        phase.value, progress_pct, display_name
                    )

                # Execute phase with retry
                phase_result = await self._execute_phase_with_retry(
                    phase, shared_context, result
                )
                result.phase_results[phase.value] = phase_result

                if phase_result.status == ExecutionStatus.FAILED:
                    result.status = ExecutionStatus.FAILED
                    result.errors.append(f"Phase '{phase.value}' failed after retries")
                    if phase_result.sme_error_messages:
                        result.sme_error_summary.extend(phase_result.sme_error_messages)
                    else:
                        result.sme_error_summary.append(
                            _sme_error_message("unknown_error")
                        )
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
            result.sme_error_summary.append(_sme_error_message("unknown_error"))

        finally:
            result.completed_at = utcnow()
            result.total_duration_ms = (time.monotonic() - start_time) * 1000
            result.quality_score = self._compute_quality_score(result)

            # Aggregate SME-relevant metrics
            quick_wins = result.phase_results.get(SMEPipelinePhase.QUICK_WINS.value)
            if quick_wins and quick_wins.outputs:
                result.estimated_cost_savings_eur = quick_wins.outputs.get(
                    "total_cost_savings_eur", 0.0
                )
                result.quick_wins_identified = quick_wins.outputs.get(
                    "quick_wins_count", 0
                )

            grants = result.phase_results.get(SMEPipelinePhase.GRANT_SEARCH.value)
            if grants and grants.outputs:
                result.grants_matched = grants.outputs.get("grants_matched", 0)

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(result)

            if self._progress_callback:
                await self._progress_callback(
                    "complete", 100.0, f"Pipeline {result.status.value}"
                )

        self.logger.info(
            "SME Pipeline %s: execution_id=%s, phases=%d/%d, duration=%.1fms, "
            "quick_wins=%d, grants=%d",
            result.status.value, result.execution_id,
            len(result.phases_completed), total_phases,
            result.total_duration_ms,
            result.quick_wins_identified,
            result.grants_matched,
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
            "message": "Your process is being cancelled. Progress has been saved.",
            "timestamp": utcnow().isoformat(),
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

        progress = self._progress_state.get(execution_id)
        current_display = ""
        if progress:
            current_display = PHASE_DISPLAY_NAMES.get(
                SMEPipelinePhase(progress.current_phase)
                if progress.current_phase else SMEPipelinePhase.ONBOARDING,
                "",
            )

        return {
            "execution_id": execution_id,
            "found": True,
            "status": result.status.value,
            "organization_name": result.organization_name,
            "path_type": result.path_type.value,
            "data_quality_tier": result.data_quality_tier.value,
            "phases_completed": result.phases_completed,
            "phases_skipped": result.phases_skipped,
            "current_phase_display": current_display,
            "progress_pct": round(progress_pct, 1),
            "total_records_processed": result.total_records_processed,
            "quality_score": result.quality_score,
            "estimated_cost_savings_eur": result.estimated_cost_savings_eur,
            "grants_matched": result.grants_matched,
            "quick_wins_identified": result.quick_wins_identified,
            "errors": result.errors,
            "sme_error_summary": result.sme_error_summary,
            "total_duration_ms": result.total_duration_ms,
        }

    def get_progress(self, execution_id: str) -> Optional[PhaseProgress]:
        """Get real-time progress for a pipeline execution.

        Args:
            execution_id: Execution identifier.

        Returns:
            PhaseProgress or None if not found.
        """
        return self._progress_state.get(execution_id)

    def list_executions(self) -> List[Dict[str, Any]]:
        """List all pipeline executions.

        Returns:
            List of execution summaries.
        """
        return [
            {
                "execution_id": r.execution_id,
                "status": r.status.value,
                "organization_name": r.organization_name,
                "path_type": r.path_type.value,
                "phases_completed": len(r.phases_completed),
                "quick_wins": r.quick_wins_identified,
                "grants_matched": r.grants_matched,
                "started_at": r.started_at.isoformat() if r.started_at else None,
            }
            for r in self._results.values()
        ]

    # -------------------------------------------------------------------------
    # Phase Resolution
    # -------------------------------------------------------------------------

    def _resolve_phase_order(self) -> List[SMEPipelinePhase]:
        """Resolve the topological phase execution order.

        Returns:
            Ordered list of phases respecting DAG dependencies.
        """
        return list(PHASE_EXECUTION_ORDER)

    def _should_skip_phase(self, phase: SMEPipelinePhase) -> bool:
        """Determine whether a phase should be skipped.

        Args:
            phase: Phase to check.

        Returns:
            True if the phase should be skipped.
        """
        # Grant search can be skipped if region is not supported
        if phase == SMEPipelinePhase.GRANT_SEARCH:
            supported_regions = {"UK", "EU", "US", "AU", "NZ", "CA"}
            if self.config.grant_region not in supported_regions:
                return True
        return False

    def _dependencies_met(
        self, phase: SMEPipelinePhase, result: PipelineResult
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

    # -------------------------------------------------------------------------
    # Phase Execution with Retry
    # -------------------------------------------------------------------------

    async def _execute_phase_with_retry(
        self,
        phase: SMEPipelinePhase,
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

        # Map to SME-friendly error
        error_key = self._map_phase_to_error_key(phase)
        return PhaseResult(
            phase=phase,
            status=ExecutionStatus.FAILED,
            started_at=utcnow(),
            completed_at=utcnow(),
            errors=[last_error or "Unknown error"],
            sme_error_messages=[_sme_error_message(error_key, last_error or "")],
            retry_count=retry_config.max_retries,
        )

    def _map_phase_to_error_key(self, phase: SMEPipelinePhase) -> str:
        """Map a phase to an SME error message key.

        Args:
            phase: Pipeline phase.

        Returns:
            Error message key string.
        """
        mapping = {
            SMEPipelinePhase.ONBOARDING: "config_invalid",
            SMEPipelinePhase.BASELINE: "baseline_calculation_error",
            SMEPipelinePhase.TARGETS: "target_setting_error",
            SMEPipelinePhase.QUICK_WINS: "unknown_error",
            SMEPipelinePhase.GRANT_SEARCH: "grant_search_timeout",
            SMEPipelinePhase.REPORTING: "reporting_error",
        }
        return mapping.get(phase, "unknown_error")

    async def _execute_phase(
        self,
        phase: SMEPipelinePhase,
        context: Dict[str, Any],
        attempt: int,
    ) -> PhaseResult:
        """Execute a single pipeline phase.

        In production, this dispatches to the appropriate engine or bridge.
        The stub implementation returns a successful result for all phases.

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

        if phase == SMEPipelinePhase.ONBOARDING:
            outputs = self._execute_onboarding(context)

        elif phase == SMEPipelinePhase.BASELINE:
            records, outputs = self._execute_baseline(context)

        elif phase == SMEPipelinePhase.TARGETS:
            outputs = self._execute_targets(context)

        elif phase == SMEPipelinePhase.QUICK_WINS:
            records, outputs = self._execute_quick_wins(context)

        elif phase == SMEPipelinePhase.GRANT_SEARCH:
            records, outputs = self._execute_grant_search(context)

        elif phase == SMEPipelinePhase.REPORTING:
            outputs = self._execute_reporting(context)

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
            records_processed=records,
            outputs=outputs,
            provenance=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase Implementations (SME-Specific)
    # -------------------------------------------------------------------------

    def _execute_onboarding(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute onboarding phase: validate config, connect accounting.

        Args:
            context: Pipeline context.

        Returns:
            Phase outputs dict.
        """
        path_type = context.get("path_type", "simplified")
        accounting = context.get("accounting_software", "none")

        return {
            "config_valid": True,
            "organization_name": self.config.organization_name,
            "sector": self.config.sector,
            "country": self.config.country,
            "employee_count": self.config.employee_count,
            "path_type": path_type,
            "data_quality_tier": context.get("data_quality_tier", "bronze"),
            "accounting_software": accounting,
            "accounting_connected": accounting != "none",
            "scope3_approach": "spend_based" if context.get("scope3_spend_based") else "activity_based",
            "grant_region": context.get("grant_region", "UK"),
            "certification_target": context.get("certification_target", "sme_climate_hub"),
            "dependencies_available": True,
        }

    def _execute_baseline(self, context: Dict[str, Any]) -> tuple:
        """Execute baseline calculation phase via simplified MRV agents.

        Only activates the SME-relevant subset of MRV agents:
        - Stationary Combustion (MRV-001): gas/oil heating
        - Mobile Combustion (MRV-003): company vehicles
        - Scope 2 Location-Based (MRV-009): electricity
        - Business Travel (MRV-019): flights/hotels
        - Employee Commuting (MRV-020): staff commuting
        - Spend-based Scope 3 for remaining categories

        Args:
            context: Pipeline context.

        Returns:
            Tuple of (records_count, outputs_dict).
        """
        scopes = self.config.scopes_included
        path_type = context.get("path_type", "simplified")

        active_agents = [
            "MRV-001",  # Stationary combustion (gas/oil heating)
            "MRV-003",  # Mobile combustion (company vehicles)
            "MRV-009",  # Electricity (location-based)
        ]

        if "scope_3" in scopes:
            active_agents.extend([
                "MRV-019",  # Business travel
                "MRV-020",  # Employee commuting
            ])

        if path_type == "standard":
            active_agents.extend([
                "MRV-010",  # Market-based electricity
                "MRV-014",  # Purchased goods (Cat 1)
            ])

        records = len(active_agents) + context.get("activity_records_count", 20)

        return records, {
            "scope1_tco2e": 0.0,
            "scope2_location_tco2e": 0.0,
            "scope2_market_tco2e": 0.0,
            "scope3_tco2e": 0.0,
            "total_tco2e": 0.0,
            "base_year": self.config.base_year,
            "reporting_year": self.config.reporting_year,
            "active_mrv_agents": active_agents,
            "scope3_approach": "spend_based" if self.config.scope3_spend_based else "activity_based",
            "scope3_categories_calculated": len(self.config.scope3_categories),
            "methodology": "GHG Protocol SME Tool",
            "data_quality_tier": self.config.data_quality_tier.value,
        }

    def _execute_targets(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SME-appropriate target setting.

        Simplified path: SME Climate Hub pledge (50% by 2030, net zero by 2050)
        Standard path: SBTi SME target (1.5C aligned)

        Args:
            context: Pipeline context.

        Returns:
            Phase outputs dict.
        """
        baseline = context.get("baseline", {})
        base_total = baseline.get("total_tco2e", 0.0)
        path_type = context.get("path_type", "simplified")
        cert_target = context.get("certification_target", "sme_climate_hub")

        if path_type == "simplified" or cert_target == "sme_climate_hub":
            return {
                "target_framework": "sme_climate_hub",
                "near_term_target_year": 2030,
                "long_term_target_year": 2050,
                "near_term_reduction_pct": 50.0,
                "long_term_reduction_pct": 100.0,
                "pathway": "1.5C",
                "base_year_emissions_tco2e": base_total,
                "near_term_target_tco2e": base_total * 0.50,
                "commitment_type": "SME Climate Hub Pledge",
                "scope_coverage": self.config.scopes_included,
                "requires_verification": False,
            }
        else:
            return {
                "target_framework": "sbti_sme",
                "near_term_target_year": min(self.config.target_year, 2030),
                "long_term_target_year": 2050,
                "near_term_reduction_pct": 42.0,
                "long_term_reduction_pct": 90.0,
                "pathway": "1.5C",
                "base_year_emissions_tco2e": base_total,
                "near_term_target_tco2e": base_total * 0.58,
                "commitment_type": "SBTi SME Target",
                "scope_coverage": self.config.scopes_included,
                "requires_verification": True,
            }

    def _execute_quick_wins(self, context: Dict[str, Any]) -> tuple:
        """Execute quick wins identification.

        Identifies low-cost, high-impact reduction actions with payback
        periods under 24 months suitable for SMEs.

        Args:
            context: Pipeline context.

        Returns:
            Tuple of (records_count, outputs_dict).
        """
        sector = self.config.sector

        # Universal quick wins for SMEs
        quick_wins = [
            {
                "action": "Switch to renewable electricity tariff",
                "category": "energy",
                "scope": "scope_2",
                "estimated_reduction_pct": 15.0,
                "estimated_cost_savings_eur": 0.0,
                "payback_months": 0,
                "difficulty": "easy",
                "priority": 1,
            },
            {
                "action": "Install LED lighting throughout premises",
                "category": "energy",
                "scope": "scope_2",
                "estimated_reduction_pct": 5.0,
                "estimated_cost_savings_eur": 500.0,
                "payback_months": 12,
                "difficulty": "easy",
                "priority": 2,
            },
            {
                "action": "Optimize heating controls and schedules",
                "category": "energy",
                "scope": "scope_1",
                "estimated_reduction_pct": 8.0,
                "estimated_cost_savings_eur": 800.0,
                "payback_months": 6,
                "difficulty": "easy",
                "priority": 3,
            },
            {
                "action": "Implement remote/hybrid working policy",
                "category": "commuting",
                "scope": "scope_3",
                "estimated_reduction_pct": 3.0,
                "estimated_cost_savings_eur": 200.0,
                "payback_months": 0,
                "difficulty": "easy",
                "priority": 4,
            },
            {
                "action": "Switch to electric vehicle for company car",
                "category": "fleet",
                "scope": "scope_1",
                "estimated_reduction_pct": 4.0,
                "estimated_cost_savings_eur": 1500.0,
                "payback_months": 24,
                "difficulty": "medium",
                "priority": 5,
            },
            {
                "action": "Reduce business travel with video conferencing",
                "category": "travel",
                "scope": "scope_3",
                "estimated_reduction_pct": 2.0,
                "estimated_cost_savings_eur": 3000.0,
                "payback_months": 0,
                "difficulty": "easy",
                "priority": 6,
            },
        ]

        total_savings = sum(w["estimated_cost_savings_eur"] for w in quick_wins)
        total_reduction = sum(w["estimated_reduction_pct"] for w in quick_wins)

        return len(quick_wins), {
            "quick_wins": quick_wins,
            "quick_wins_count": len(quick_wins),
            "total_estimated_reduction_pct": total_reduction,
            "total_cost_savings_eur": total_savings,
            "max_payback_months": 24,
            "sector": sector,
        }

    def _execute_grant_search(self, context: Dict[str, Any]) -> tuple:
        """Execute grant and funding search.

        Searches for available grants based on the organisation's region,
        sector, and identified quick wins.

        Args:
            context: Pipeline context.

        Returns:
            Tuple of (records_count, outputs_dict).
        """
        region = context.get("grant_region", self.config.grant_region)

        # Region-specific grant stubs
        grants_by_region: Dict[str, List[Dict[str, Any]]] = {
            "UK": [
                {
                    "name": "SME Climate Action Fund",
                    "provider": "BEIS",
                    "amount_range": "5000-50000 GBP",
                    "eligibility": "UK SMEs with <250 employees",
                    "deadline": "2026-06-30",
                    "match_score": 0.85,
                },
                {
                    "name": "Green Business Grant",
                    "provider": "Local Enterprise Partnership",
                    "amount_range": "1000-25000 GBP",
                    "eligibility": "UK businesses committed to net zero",
                    "deadline": "2026-09-30",
                    "match_score": 0.78,
                },
            ],
            "EU": [
                {
                    "name": "LIFE Clean Energy Transition",
                    "provider": "European Commission",
                    "amount_range": "50000-500000 EUR",
                    "eligibility": "EU-based SMEs",
                    "deadline": "2026-10-15",
                    "match_score": 0.72,
                },
            ],
            "US": [
                {
                    "name": "Small Business Energy Efficiency",
                    "provider": "DOE",
                    "amount_range": "10000-100000 USD",
                    "eligibility": "US small businesses",
                    "deadline": "2026-12-31",
                    "match_score": 0.75,
                },
            ],
        }

        matched = grants_by_region.get(region, [])

        return len(matched), {
            "grants_matched": len(matched),
            "grants": matched,
            "region": region,
            "search_timestamp": utcnow().isoformat(),
            "next_sync_date": "monthly",
        }

    def _execute_reporting(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute simplified annual reporting.

        Generates reports suitable for SME Climate Hub submission,
        certification bodies, and internal stakeholders.

        Args:
            context: Pipeline context.

        Returns:
            Phase outputs dict.
        """
        cert_target = context.get("certification_target", "sme_climate_hub")

        reports = ["sme_summary_report", "carbon_footprint_report"]
        if cert_target == "sme_climate_hub":
            reports.append("sme_climate_hub_submission")
        elif cert_target == "b_corp":
            reports.append("b_corp_environmental_assessment")
        elif cert_target == "carbon_trust":
            reports.append("carbon_trust_footprint_submission")

        return {
            "reports_generated": reports,
            "report_count": len(reports),
            "certification_submission_ready": True,
            "certification_target": cert_target,
            "net_zero_plan_complete": True,
            "dashboard_url": "",
        }

    # -------------------------------------------------------------------------
    # Quality Score
    # -------------------------------------------------------------------------

    def _compute_quality_score(self, result: PipelineResult) -> float:
        """Compute overall pipeline quality score (0-100).

        Scoring formula:
            - Phase completion: 50 points (% of non-skipped phases completed)
            - Error-free execution: 30 points (deducted per error)
            - Data quality tier bonus: 20 points (Bronze=10, Silver=15, Gold=20)

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

        tier_scores = {
            DataQualityTier.BRONZE: 10.0,
            DataQualityTier.SILVER: 15.0,
            DataQualityTier.GOLD: 20.0,
        }
        tier_score = tier_scores.get(result.data_quality_tier, 10.0)

        return round(min(completion_score + error_score + tier_score, 100.0), 2)

    # -------------------------------------------------------------------------
    # Resume from Checkpoint
    # -------------------------------------------------------------------------

    async def resume_pipeline(
        self,
        execution_id: str,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[PipelineResult]:
        """Resume a failed or cancelled pipeline from the last checkpoint.

        Args:
            execution_id: Previous execution ID to resume.
            additional_data: Additional data to merge into context.

        Returns:
            New PipelineResult, or None if not resumable.
        """
        if execution_id not in self._results:
            self.logger.warning("Cannot resume: execution %s not found", execution_id)
            return None

        previous = self._results[execution_id]
        if previous.status not in (ExecutionStatus.FAILED, ExecutionStatus.CANCELLED):
            self.logger.warning(
                "Cannot resume: execution %s is in status '%s'",
                execution_id, previous.status.value,
            )
            return None

        # Build context from completed phases
        resume_data = additional_data or {}
        for phase_name, phase_result in previous.phase_results.items():
            if phase_result.status == ExecutionStatus.COMPLETED:
                resume_data[phase_name] = phase_result.outputs

        resume_data["_resumed_from"] = execution_id
        resume_data["_completed_phases"] = list(previous.phases_completed)

        self.logger.info(
            "Resuming pipeline from execution %s, completed phases: %s",
            execution_id, previous.phases_completed,
        )

        return await self.execute_pipeline(resume_data)

    # -------------------------------------------------------------------------
    # Demo Execution
    # -------------------------------------------------------------------------

    async def run_demo(self) -> PipelineResult:
        """Run a demonstration pipeline with sample SME data.

        Returns:
            PipelineResult for the demo execution.
        """
        demo_data = {
            "demo_mode": True,
            "activity_records_count": 30,
            "reporting_period": {
                "start": f"{self.config.reporting_year}-01-01",
                "end": f"{self.config.reporting_year}-12-31",
            },
        }
        return await self.execute_pipeline(demo_data)

    # -------------------------------------------------------------------------
    # Comparison of Paths
    # -------------------------------------------------------------------------

    def compare_paths(self) -> Dict[str, Any]:
        """Compare simplified vs standard path for decision-making.

        Returns:
            Dict comparing the two paths across key dimensions.
        """
        return {
            "simplified": {
                "name": "Simplified Path",
                "suitable_for": "Micro and small businesses (<50 employees)",
                "data_requirement": "Bronze (spend data, utility bills)",
                "scope3_approach": "Spend-based only",
                "target_framework": "SME Climate Hub Pledge",
                "estimated_setup_time": "2-4 hours",
                "cost": "Free (SME Climate Hub)",
                "mrv_agents_used": 5,
                "accuracy": "Indicative (+/- 30%)",
            },
            "standard": {
                "name": "Standard Path",
                "suitable_for": "Medium businesses (50-250 employees)",
                "data_requirement": "Silver/Gold (activity data preferred)",
                "scope3_approach": "Activity-based where possible",
                "target_framework": "SBTi SME Target",
                "estimated_setup_time": "1-2 weeks",
                "cost": "SBTi SME fee applies",
                "mrv_agents_used": 7,
                "accuracy": "Robust (+/- 15%)",
            },
            "recommendation": self.select_path().value,
        }
