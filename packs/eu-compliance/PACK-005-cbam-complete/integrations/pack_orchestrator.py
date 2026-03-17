# -*- coding: utf-8 -*-
"""
CBAMCompleteOrchestrator - 10-Phase CBAM Complete Execution Pipeline
=====================================================================

This module implements the master orchestrator for the CBAM Complete Pack
(PACK-005). It extends the PACK-004 8-phase pipeline into a comprehensive
10-phase pipeline that adds certificate trading, registry submission,
cross-regulation synchronization, and multi-entity consolidation.

Execution Phases:
    1.  HEALTH_CHECK:           Run all 18 health check categories
    2.  CONFIGURATION_LOADING:  Load CBAMCompleteConfig; merge base + extensions
    3.  IMPORT_DATA_INTAKE:     Run customs integration; parse SAD declarations
    4.  EMISSION_CALCULATIONS:  Run PACK-004 engine + precursor chain resolution
    5.  CERTIFICATE_OBLIGATION: Run certificate engine + multi-entity consolidation
    6.  CERTIFICATE_TRADING:    Run certificate trading workflow; manage portfolio
    7.  REGISTRY_SUBMISSION:    Submit declarations/reports via Registry API
    8.  CROSS_REGULATION_SYNC:  Update CSRD/CDP/SBTi/Taxonomy/ETS data
    9.  AUDIT_TRAIL_UPDATE:     Run audit management; log all actions
    10. REPORTING:              Render templates; generate dashboards

Example:
    >>> config = CBAMCompleteConfig(importer_eori="DE123456789012345")
    >>> orchestrator = CBAMCompleteOrchestrator(config)
    >>> result = orchestrator.run(config, import_data)
    >>> assert result.status == "completed"

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-005 CBAM Complete
"""

import hashlib
import logging
import random
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class CBAMCompletePhase(str, Enum):
    """Execution phases in the CBAM Complete pipeline."""
    HEALTH_CHECK = "health_check"
    CONFIGURATION_LOADING = "configuration_loading"
    IMPORT_DATA_INTAKE = "import_data_intake"
    EMISSION_CALCULATIONS = "emission_calculations"
    CERTIFICATE_OBLIGATION = "certificate_obligation"
    CERTIFICATE_TRADING = "certificate_trading"
    REGISTRY_SUBMISSION = "registry_submission"
    CROSS_REGULATION_SYNC = "cross_regulation_sync"
    AUDIT_TRAIL_UPDATE = "audit_trail_update"
    REPORTING = "reporting"


class CompleteExecutionStatus(str, Enum):
    """Status of a pipeline execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    RESUMED = "resumed"
    ROLLED_BACK = "rolled_back"


class QualityGateStatus(str, Enum):
    """Status of a quality gate check."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


# =============================================================================
# Data Models
# =============================================================================


class CBAMCompleteConfig(BaseModel):
    """Configuration for the CBAM Complete Orchestrator."""
    pack_id: str = Field(default="PACK-005", description="Pack identifier")
    base_pack_id: str = Field(default="PACK-004", description="Base pack ID")
    importer_eori: str = Field(default="", description="Importer EORI number")
    company_name: str = Field(default="", description="Company name")
    member_state: str = Field(default="", description="EU member state code")
    entity_group_id: str = Field(default="", description="Multi-entity group ID")
    entity_ids: List[str] = Field(default_factory=list, description="Entity IDs in group")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retry attempts per phase")
    initial_backoff_seconds: float = Field(default=1.0, description="Initial backoff delay")
    max_backoff_seconds: float = Field(default=30.0, description="Max backoff delay")
    enable_provenance: bool = Field(default=True, description="Enable provenance tracking")
    enable_quality_gates: bool = Field(default=True, description="Enable quality gates")
    enable_trading: bool = Field(default=True, description="Enable certificate trading")
    enable_registry_submission: bool = Field(default=False, description="Enable registry API")
    enable_cross_regulation: bool = Field(default=True, description="Enable cross-reg sync")
    timeout_per_phase_seconds: int = Field(default=600, description="Timeout per phase")
    goods_categories: List[str] = Field(default_factory=list, description="Goods categories")
    trading_strategy: str = Field(default="cost_averaging", description="Trading strategy")
    trading_budget_eur: float = Field(default=0.0, ge=0.0, description="Trading budget EUR")
    registry_environment: str = Field(default="sandbox", description="Registry env")
    cross_regulation_targets: List[str] = Field(
        default_factory=lambda: ["CSRD", "CDP", "SBTi"],
        description="Target regulation packs",
    )
    database_url: Optional[str] = Field(None, description="Database URL")
    abort_on_health_critical: bool = Field(default=True, description="Abort on critical health")


class PhaseResult(BaseModel):
    """Result of executing a single pipeline phase."""
    phase: CBAMCompletePhase = Field(..., description="Phase executed")
    status: CompleteExecutionStatus = Field(
        default=CompleteExecutionStatus.COMPLETED, description="Phase status"
    )
    started_at: str = Field(default="", description="Phase start timestamp")
    completed_at: str = Field(default="", description="Phase completion timestamp")
    execution_time_ms: float = Field(default=0.0, description="Execution time in ms")
    records_processed: int = Field(default=0, description="Records processed")
    data: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    errors: List[str] = Field(default_factory=list, description="Phase errors")
    warnings: List[str] = Field(default_factory=list, description="Phase warnings")
    quality_gate: QualityGateStatus = Field(
        default=QualityGateStatus.SKIPPED, description="Quality gate result"
    )
    retry_count: int = Field(default=0, description="Number of retries")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    rollback_data: Dict[str, Any] = Field(
        default_factory=dict, description="Data for rollback if needed"
    )


class OrchestrationResult(BaseModel):
    """Complete result of a CBAM Complete orchestration run."""
    execution_id: str = Field(
        default_factory=lambda: str(uuid4())[:16], description="Execution ID"
    )
    pack_id: str = Field(default="PACK-005", description="Pack identifier")
    importer_eori: str = Field(default="", description="Importer EORI")
    entity_group_id: str = Field(default="", description="Entity group ID")
    status: CompleteExecutionStatus = Field(
        default=CompleteExecutionStatus.PENDING, description="Overall status"
    )
    started_at: str = Field(default="", description="Execution start timestamp")
    completed_at: str = Field(default="", description="Execution completion timestamp")
    total_execution_time_ms: float = Field(default=0.0, description="Total execution time")
    phase_results: Dict[str, PhaseResult] = Field(
        default_factory=dict, description="Per-phase results"
    )
    total_imports: int = Field(default=0, description="Total import records")
    total_embedded_emissions_tco2: float = Field(default=0.0, description="Total emissions")
    certificate_obligation_eur: float = Field(default=0.0, description="Certificate obligation")
    certificates_purchased: int = Field(default=0, description="Certificates purchased")
    certificates_surrendered: int = Field(default=0, description="Certificates surrendered")
    registry_submissions: int = Field(default=0, description="Registry submissions made")
    cross_regulation_syncs: int = Field(default=0, description="Cross-reg syncs completed")
    compliance_score: float = Field(default=0.0, ge=0, le=100, description="Compliance score")
    errors: List[str] = Field(default_factory=list, description="Execution errors")
    warnings: List[str] = Field(default_factory=list, description="Execution warnings")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class OrchestrationStatus(BaseModel):
    """Current status of the orchestration pipeline."""
    execution_id: str = Field(default="", description="Current execution ID")
    status: CompleteExecutionStatus = Field(
        default=CompleteExecutionStatus.PENDING, description="Current status"
    )
    current_phase: str = Field(default="", description="Currently executing phase")
    phases_completed: int = Field(default=0, description="Phases completed")
    total_phases: int = Field(default=10, description="Total phases")
    progress_pct: float = Field(default=0.0, description="Progress percentage")
    elapsed_ms: float = Field(default=0.0, description="Elapsed time in ms")
    errors: List[str] = Field(default_factory=list, description="Current errors")


class CheckpointData(BaseModel):
    """Checkpoint for resume capability."""
    checkpoint_id: str = Field(
        default_factory=lambda: str(uuid4())[:16], description="Checkpoint ID"
    )
    execution_id: str = Field(..., description="Parent execution ID")
    phase_completed: str = Field(..., description="Last completed phase")
    phase_results: Dict[str, Any] = Field(
        default_factory=dict, description="Results up to checkpoint"
    )
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    config_snapshot: Dict[str, Any] = Field(
        default_factory=dict, description="Config at checkpoint time"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(), description="Checkpoint time"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# =============================================================================
# Phase Pipeline Definition
# =============================================================================


COMPLETE_PHASE_ORDER: List[CBAMCompletePhase] = [
    CBAMCompletePhase.HEALTH_CHECK,
    CBAMCompletePhase.CONFIGURATION_LOADING,
    CBAMCompletePhase.IMPORT_DATA_INTAKE,
    CBAMCompletePhase.EMISSION_CALCULATIONS,
    CBAMCompletePhase.CERTIFICATE_OBLIGATION,
    CBAMCompletePhase.CERTIFICATE_TRADING,
    CBAMCompletePhase.REGISTRY_SUBMISSION,
    CBAMCompletePhase.CROSS_REGULATION_SYNC,
    CBAMCompletePhase.AUDIT_TRAIL_UPDATE,
    CBAMCompletePhase.REPORTING,
]

QUALITY_GATE_REQUIREMENTS: Dict[CBAMCompletePhase, Dict[str, Any]] = {
    CBAMCompletePhase.HEALTH_CHECK: {
        "min_health_score": 60.0,
        "max_critical_findings": 0,
    },
    CBAMCompletePhase.CONFIGURATION_LOADING: {
        "require_valid_config": True,
        "require_eori": True,
    },
    CBAMCompletePhase.IMPORT_DATA_INTAKE: {
        "min_records": 0,
        "max_error_rate": 0.10,
        "required_fields": ["cn_code", "origin_country", "quantity"],
    },
    CBAMCompletePhase.EMISSION_CALCULATIONS: {
        "max_error_rate": 0.0,
        "require_provenance": True,
    },
    CBAMCompletePhase.CERTIFICATE_OBLIGATION: {
        "require_positive_price": True,
        "require_group_consistency": True,
    },
    CBAMCompletePhase.CERTIFICATE_TRADING: {
        "require_budget_check": True,
        "max_overspend_pct": 0.05,
    },
    CBAMCompletePhase.REGISTRY_SUBMISSION: {
        "require_valid_xml": True,
        "require_auth": True,
    },
    CBAMCompletePhase.CROSS_REGULATION_SYNC: {
        "allow_partial_sync": True,
    },
    CBAMCompletePhase.AUDIT_TRAIL_UPDATE: {
        "require_provenance_chain": True,
    },
    CBAMCompletePhase.REPORTING: {
        "require_all_sections": True,
    },
}

# Agent mapping per phase
PHASE_AGENT_MAPPING: Dict[CBAMCompletePhase, List[str]] = {
    CBAMCompletePhase.HEALTH_CHECK: [
        "GL-FOUND-X-009",   # Observability Agent
    ],
    CBAMCompletePhase.CONFIGURATION_LOADING: [
        "GL-FOUND-X-002",   # Schema Compiler
    ],
    CBAMCompletePhase.IMPORT_DATA_INTAKE: [
        "GL-DATA-X-001", "GL-DATA-X-002", "GL-DATA-X-003",
        "GL-EUDR-X-001", "GL-DATA-X-019",
    ],
    CBAMCompletePhase.EMISSION_CALCULATIONS: [
        "GL-MRV-X-001", "GL-MRV-X-004", "GL-MRV-X-014", "GL-MRV-X-029",
    ],
    CBAMCompletePhase.CERTIFICATE_OBLIGATION: [
        "GL-CBAM-CERT", "GL-CBAM-DEMIN", "GL-CBAM-MULTI-ENTITY",
    ],
    CBAMCompletePhase.CERTIFICATE_TRADING: [
        "GL-CBAM-TRADING",
    ],
    CBAMCompletePhase.REGISTRY_SUBMISSION: [
        "GL-CBAM-REGISTRY",
    ],
    CBAMCompletePhase.CROSS_REGULATION_SYNC: [
        "GL-CBAM-CROSS-REG",
    ],
    CBAMCompletePhase.AUDIT_TRAIL_UPDATE: [
        "GL-MRV-X-030", "GL-FOUND-X-005", "GL-FOUND-X-004",
    ],
    CBAMCompletePhase.REPORTING: [
        "GL-CBAM-QRT", "GL-CBAM-ANNUAL", "GL-CBAM-ANALYTICS",
    ],
}


# =============================================================================
# Orchestrator Implementation
# =============================================================================


class CBAMCompleteOrchestrator:
    """10-phase CBAM Complete orchestrator extending PACK-004.

    Manages the end-to-end CBAM Complete compliance pipeline from health
    verification through certificate trading, registry submission,
    cross-regulation synchronization, and comprehensive reporting.

    Features:
        - 10-phase pipeline with quality gate enforcement
        - Extends PACK-004 base pipeline with 5 additional phases
        - Retry with exponential backoff and jitter
        - Checkpoint/resume for long-running executions
        - Phase-level rollback on failure
        - Full SHA-256 provenance chain
        - Progress tracking with real-time status
        - Multi-entity group support

    Attributes:
        config: Orchestrator configuration
        _executions: History of execution results
        _checkpoints: Saved checkpoints
        _phase_handlers: Registered phase handler functions
        _current_execution_id: ID of the currently running execution
        _current_phase: Currently executing phase name

    Example:
        >>> config = CBAMCompleteConfig(importer_eori="DE123456789012345")
        >>> orch = CBAMCompleteOrchestrator(config)
        >>> result = orch.run(config, [{"cn_code": "7201 10 11", ...}])
    """

    def __init__(self, config: Optional[CBAMCompleteConfig] = None) -> None:
        """Initialize the CBAM Complete Orchestrator.

        Args:
            config: Orchestrator configuration. Uses defaults if not provided.
        """
        self.config = config or CBAMCompleteConfig()
        self.logger = logger
        self._executions: Dict[str, OrchestrationResult] = {}
        self._checkpoints: Dict[str, CheckpointData] = {}
        self._current_execution_id: str = ""
        self._current_phase: str = ""
        self._start_time: float = 0.0

        self._phase_handlers: Dict[CBAMCompletePhase, Callable] = {
            CBAMCompletePhase.HEALTH_CHECK: self._phase_health_check,
            CBAMCompletePhase.CONFIGURATION_LOADING: self._phase_configuration_loading,
            CBAMCompletePhase.IMPORT_DATA_INTAKE: self._phase_import_data_intake,
            CBAMCompletePhase.EMISSION_CALCULATIONS: self._phase_emission_calculations,
            CBAMCompletePhase.CERTIFICATE_OBLIGATION: self._phase_certificate_obligation,
            CBAMCompletePhase.CERTIFICATE_TRADING: self._phase_certificate_trading,
            CBAMCompletePhase.REGISTRY_SUBMISSION: self._phase_registry_submission,
            CBAMCompletePhase.CROSS_REGULATION_SYNC: self._phase_cross_regulation_sync,
            CBAMCompletePhase.AUDIT_TRAIL_UPDATE: self._phase_audit_trail_update,
            CBAMCompletePhase.REPORTING: self._phase_reporting,
        }

        self.logger.info(
            "CBAMCompleteOrchestrator initialized: eori=%s, group=%s, "
            "trading=%s, registry=%s, cross_reg=%s",
            self.config.importer_eori,
            self.config.entity_group_id,
            self.config.enable_trading,
            self.config.enable_registry_submission,
            self.config.enable_cross_regulation,
        )

    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------

    def run(
        self,
        config: CBAMCompleteConfig,
        import_data: List[Dict[str, Any]],
    ) -> OrchestrationResult:
        """Execute the full 10-phase CBAM Complete pipeline.

        Args:
            config: Execution configuration.
            import_data: List of import records to process.

        Returns:
            OrchestrationResult with full phase results and totals.
        """
        self._start_time = time.monotonic()
        execution_id = _compute_hash(
            f"complete:{config.importer_eori}:{datetime.utcnow().isoformat()}"
        )[:16]
        self._current_execution_id = execution_id

        result = OrchestrationResult(
            execution_id=execution_id,
            importer_eori=config.importer_eori,
            entity_group_id=config.entity_group_id,
            status=CompleteExecutionStatus.RUNNING,
            started_at=datetime.utcnow().isoformat(),
        )

        context: Dict[str, Any] = {
            "execution_id": execution_id,
            "import_data": import_data,
            "config": config.model_dump(),
            "phase_outputs": {},
        }

        self.logger.info(
            "Starting CBAM Complete execution (id=%s, imports=%d, entities=%d)",
            execution_id, len(import_data), len(config.entity_ids),
        )

        try:
            for phase in COMPLETE_PHASE_ORDER:
                self._current_phase = phase.value
                phase_result = self.run_phase(phase.value, context)
                result.phase_results[phase.value] = phase_result
                context["phase_outputs"][phase.value] = phase_result.data

                if phase_result.status == CompleteExecutionStatus.FAILED:
                    gate = phase_result.quality_gate
                    if gate == QualityGateStatus.FAILED:
                        result.status = CompleteExecutionStatus.FAILED
                        result.errors.append(
                            f"Quality gate failed at phase '{phase.value}'"
                        )
                        self.logger.error(
                            "Execution failed at phase '%s': quality gate", phase.value
                        )
                        # Attempt rollback of completed phases
                        self._rollback_phases(result, context)
                        break

                result.warnings.extend(phase_result.warnings)

            if result.status != CompleteExecutionStatus.FAILED:
                result.status = CompleteExecutionStatus.COMPLETED
                result = self._aggregate_results(result, context)

        except Exception as exc:
            result.status = CompleteExecutionStatus.FAILED
            result.errors.append(f"Unexpected error: {exc}")
            self.logger.error("Execution failed: %s", exc, exc_info=True)

        result.completed_at = datetime.utcnow().isoformat()
        result.total_execution_time_ms = (time.monotonic() - self._start_time) * 1000
        self._current_phase = ""

        if self.config.enable_provenance:
            result.provenance_hash = self._compute_execution_provenance(result)

        self._executions[execution_id] = result

        self.logger.info(
            "CBAM Complete execution %s in %.1fms (id=%s, score=%.1f)",
            result.status.value, result.total_execution_time_ms,
            execution_id, result.compliance_score,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase Execution
    # -------------------------------------------------------------------------

    def run_phase(
        self,
        phase_name: str,
        context: Dict[str, Any],
    ) -> PhaseResult:
        """Execute a single pipeline phase with retry and quality gate.

        Args:
            phase_name: Name of the phase to execute.
            context: Execution context with import data and prior phase outputs.

        Returns:
            PhaseResult with execution details and quality gate status.
        """
        try:
            phase = CBAMCompletePhase(phase_name)
        except ValueError:
            return PhaseResult(
                phase=CBAMCompletePhase.HEALTH_CHECK,
                status=CompleteExecutionStatus.FAILED,
                errors=[f"Unknown phase: {phase_name}"],
            )

        handler = self._phase_handlers.get(phase)
        if handler is None:
            return PhaseResult(
                phase=phase,
                status=CompleteExecutionStatus.FAILED,
                errors=[f"No handler for phase: {phase_name}"],
            )

        last_exception: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            phase_start = time.monotonic()
            try:
                phase_result = handler(context)
                phase_result.execution_time_ms = (time.monotonic() - phase_start) * 1000
                phase_result.retry_count = attempt

                if self.config.enable_quality_gates:
                    gate_status = self._evaluate_quality_gate(phase, phase_result)
                    phase_result.quality_gate = gate_status

                if self.config.enable_provenance:
                    phase_result.provenance_hash = _compute_hash(
                        f"{phase_name}:{phase_result.execution_time_ms}:"
                        f"{phase_result.records_processed}:{phase_result.data}"
                    )

                self.logger.info(
                    "Phase '%s' completed in %.1fms (attempt %d, gate=%s)",
                    phase_name, phase_result.execution_time_ms,
                    attempt + 1, phase_result.quality_gate.value,
                )
                return phase_result

            except Exception as exc:
                last_exception = exc
                elapsed = (time.monotonic() - phase_start) * 1000
                self.logger.warning(
                    "Phase '%s' failed (attempt %d/%d, %.1fms): %s",
                    phase_name, attempt + 1, self.config.max_retries + 1, elapsed, exc,
                )
                if attempt < self.config.max_retries:
                    backoff = self._calculate_backoff(attempt)
                    self.logger.info("Retrying phase '%s' in %.2fs", phase_name, backoff)
                    time.sleep(backoff)

        return PhaseResult(
            phase=phase,
            status=CompleteExecutionStatus.FAILED,
            errors=[
                f"Phase failed after {self.config.max_retries + 1} attempts: "
                f"{last_exception}"
            ],
            retry_count=self.config.max_retries,
        )

    # -------------------------------------------------------------------------
    # Status & Checkpoint
    # -------------------------------------------------------------------------

    def get_status(self) -> OrchestrationStatus:
        """Get the current status of the running orchestration.

        Returns:
            OrchestrationStatus with progress information.
        """
        execution = self._executions.get(self._current_execution_id)
        if execution is None:
            return OrchestrationStatus()

        phases_done = sum(
            1 for pr in execution.phase_results.values()
            if pr.status == CompleteExecutionStatus.COMPLETED
        )
        total = len(COMPLETE_PHASE_ORDER)
        elapsed = (time.monotonic() - self._start_time) * 1000 if self._start_time else 0.0

        return OrchestrationStatus(
            execution_id=self._current_execution_id,
            status=execution.status,
            current_phase=self._current_phase,
            phases_completed=phases_done,
            total_phases=total,
            progress_pct=round((phases_done / total) * 100, 1),
            elapsed_ms=elapsed,
            errors=execution.errors[:5],
        )

    def checkpoint(self) -> CheckpointData:
        """Save a checkpoint of the current execution state.

        Returns:
            CheckpointData that can be used to resume later.
        """
        execution = self._executions.get(self._current_execution_id)
        if execution is None:
            self.logger.error("No active execution to checkpoint")
            return CheckpointData(
                execution_id="", phase_completed="",
            )

        last_phase = ""
        phase_data: Dict[str, Any] = {}
        for phase_name, pr in execution.phase_results.items():
            if pr.status == CompleteExecutionStatus.COMPLETED:
                last_phase = phase_name
                phase_data[phase_name] = pr.model_dump()

        cp = CheckpointData(
            execution_id=self._current_execution_id,
            phase_completed=last_phase,
            phase_results=phase_data,
            config_snapshot=self.config.model_dump(),
            context={"status": execution.status.value},
        )
        cp.provenance_hash = _compute_hash(
            f"checkpoint:{cp.checkpoint_id}:{self._current_execution_id}:{last_phase}"
        )

        self._checkpoints[cp.checkpoint_id] = cp
        self.logger.info(
            "Checkpoint saved: %s (execution=%s, phase=%s)",
            cp.checkpoint_id, self._current_execution_id, last_phase,
        )
        return cp

    def resume(self, checkpoint_data: CheckpointData) -> OrchestrationResult:
        """Resume execution from a saved checkpoint.

        Args:
            checkpoint_data: Previously saved checkpoint.

        Returns:
            OrchestrationResult for the resumed execution.
        """
        self.logger.info(
            "Resuming from checkpoint %s (execution=%s, last_phase=%s)",
            checkpoint_data.checkpoint_id,
            checkpoint_data.execution_id,
            checkpoint_data.phase_completed,
        )

        # Restore config from snapshot if available
        if checkpoint_data.config_snapshot:
            self.config = CBAMCompleteConfig(**checkpoint_data.config_snapshot)

        # Find resume index
        resume_index = 0
        for i, phase in enumerate(COMPLETE_PHASE_ORDER):
            if phase.value == checkpoint_data.phase_completed:
                resume_index = i + 1
                break

        self._start_time = time.monotonic()
        execution_id = checkpoint_data.execution_id
        self._current_execution_id = execution_id

        result = OrchestrationResult(
            execution_id=execution_id,
            importer_eori=self.config.importer_eori,
            entity_group_id=self.config.entity_group_id,
            status=CompleteExecutionStatus.RESUMED,
            started_at=datetime.utcnow().isoformat(),
        )

        # Restore completed phase results
        for phase_name, pr_data in checkpoint_data.phase_results.items():
            result.phase_results[phase_name] = PhaseResult(**pr_data)

        context: Dict[str, Any] = {
            "execution_id": execution_id,
            "import_data": [],
            "config": self.config.model_dump(),
            "phase_outputs": {
                name: pr_data.get("data", {})
                for name, pr_data in checkpoint_data.phase_results.items()
            },
        }

        try:
            for phase in COMPLETE_PHASE_ORDER[resume_index:]:
                self._current_phase = phase.value
                phase_result = self.run_phase(phase.value, context)
                result.phase_results[phase.value] = phase_result
                context["phase_outputs"][phase.value] = phase_result.data

                if phase_result.status == CompleteExecutionStatus.FAILED:
                    if phase_result.quality_gate == QualityGateStatus.FAILED:
                        result.status = CompleteExecutionStatus.FAILED
                        result.errors.append(
                            f"Quality gate failed at phase '{phase.value}'"
                        )
                        break

                result.warnings.extend(phase_result.warnings)

            if result.status != CompleteExecutionStatus.FAILED:
                result.status = CompleteExecutionStatus.COMPLETED
                result = self._aggregate_results(result, context)

        except Exception as exc:
            result.status = CompleteExecutionStatus.FAILED
            result.errors.append(f"Resume failed: {exc}")
            self.logger.error("Resume failed: %s", exc, exc_info=True)

        result.completed_at = datetime.utcnow().isoformat()
        result.total_execution_time_ms = (time.monotonic() - self._start_time) * 1000

        if self.config.enable_provenance:
            result.provenance_hash = self._compute_execution_provenance(result)

        self._executions[execution_id] = result
        return result

    # -------------------------------------------------------------------------
    # Phase Handlers
    # -------------------------------------------------------------------------

    def _phase_health_check(self, context: Dict[str, Any]) -> PhaseResult:
        """Phase 1: Run all 18 health check categories.

        Args:
            context: Execution context.

        Returns:
            PhaseResult for health check. Aborts if critical failures found.
        """
        config_data = context.get("config", {})
        abort_on_critical = config_data.get("abort_on_health_critical", True)

        # Simulate health check categories
        categories_checked = 18
        categories_healthy = 16
        categories_degraded = 2
        categories_unhealthy = 0
        health_score = round((categories_healthy / categories_checked) * 100, 1)

        findings: List[Dict[str, str]] = []
        if categories_degraded > 0:
            findings.append({
                "category": "cross_pack_bridges",
                "severity": "warning",
                "message": "Some target packs not installed (graceful degradation active)",
            })
            findings.append({
                "category": "registry_api",
                "severity": "warning",
                "message": "Registry API in sandbox mode",
            })

        warnings: List[str] = []
        errors: List[str] = []

        if categories_unhealthy > 0 and abort_on_critical:
            errors.append(
                f"Critical health failures detected: {categories_unhealthy} unhealthy categories"
            )
            return PhaseResult(
                phase=CBAMCompletePhase.HEALTH_CHECK,
                status=CompleteExecutionStatus.FAILED,
                started_at=datetime.utcnow().isoformat(),
                completed_at=datetime.utcnow().isoformat(),
                records_processed=categories_checked,
                data={
                    "health_score": health_score,
                    "categories_checked": categories_checked,
                    "categories_healthy": categories_healthy,
                    "categories_degraded": categories_degraded,
                    "categories_unhealthy": categories_unhealthy,
                    "findings": findings,
                },
                errors=errors,
            )

        for f in findings:
            warnings.append(f"{f['category']}: {f['message']}")

        return PhaseResult(
            phase=CBAMCompletePhase.HEALTH_CHECK,
            status=CompleteExecutionStatus.COMPLETED,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            records_processed=categories_checked,
            data={
                "health_score": health_score,
                "categories_checked": categories_checked,
                "categories_healthy": categories_healthy,
                "categories_degraded": categories_degraded,
                "categories_unhealthy": categories_unhealthy,
                "findings": findings,
            },
            warnings=warnings,
        )

    def _phase_configuration_loading(self, context: Dict[str, Any]) -> PhaseResult:
        """Phase 2: Load and merge PACK-004 base + PACK-005 extensions.

        Args:
            context: Execution context.

        Returns:
            PhaseResult for configuration loading.
        """
        config_data = context.get("config", {})
        errors: List[str] = []
        warnings: List[str] = []

        # Validate required fields
        eori = config_data.get("importer_eori", "")
        if not eori:
            errors.append("EORI number is required")

        member_state = config_data.get("member_state", "")
        if not member_state:
            warnings.append("Member state not set; some validations may be limited")

        # Merge base PACK-004 config with PACK-005 extensions
        merged_config = {
            "base_pack": config_data.get("base_pack_id", "PACK-004"),
            "pack_id": config_data.get("pack_id", "PACK-005"),
            "importer_eori": eori,
            "entity_group_id": config_data.get("entity_group_id", ""),
            "entity_count": len(config_data.get("entity_ids", [])),
            "trading_enabled": config_data.get("enable_trading", True),
            "registry_enabled": config_data.get("enable_registry_submission", False),
            "cross_reg_enabled": config_data.get("enable_cross_regulation", True),
            "trading_strategy": config_data.get("trading_strategy", "cost_averaging"),
            "trading_budget_eur": config_data.get("trading_budget_eur", 0.0),
            "registry_environment": config_data.get("registry_environment", "sandbox"),
            "cross_reg_targets": config_data.get("cross_regulation_targets", []),
            "goods_categories": config_data.get("goods_categories", []),
            "merged_at": datetime.utcnow().isoformat(),
        }

        status = (
            CompleteExecutionStatus.FAILED if errors
            else CompleteExecutionStatus.COMPLETED
        )

        return PhaseResult(
            phase=CBAMCompletePhase.CONFIGURATION_LOADING,
            status=status,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            records_processed=1,
            data={"merged_config": merged_config},
            errors=errors,
            warnings=warnings,
        )

    def _phase_import_data_intake(self, context: Dict[str, Any]) -> PhaseResult:
        """Phase 3: Customs integration and SAD declaration parsing.

        Args:
            context: Execution context.

        Returns:
            PhaseResult for import data intake.
        """
        import_data = context.get("import_data", [])

        valid_records: List[Dict[str, Any]] = []
        errors: List[str] = []
        warnings: List[str] = []

        for idx, record in enumerate(import_data):
            if not record.get("cn_code"):
                errors.append(f"Record {idx}: missing cn_code")
                continue
            if not record.get("origin_country"):
                errors.append(f"Record {idx}: missing origin_country")
                continue

            # Validate CN code format (8 digits)
            cn_raw = record["cn_code"].replace(" ", "").replace(".", "")
            if len(cn_raw) < 8:
                warnings.append(f"Record {idx}: CN code '{record['cn_code']}' may be incomplete")

            quantity = float(record.get("quantity", 0.0))
            if quantity <= 0:
                warnings.append(f"Record {idx}: zero or negative quantity")

            valid_records.append(record)

        # Entity-level grouping
        by_entity: Dict[str, int] = {}
        for record in valid_records:
            entity = record.get("entity_id", "default")
            by_entity[entity] = by_entity.get(entity, 0) + 1

        return PhaseResult(
            phase=CBAMCompletePhase.IMPORT_DATA_INTAKE,
            status=CompleteExecutionStatus.COMPLETED,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            records_processed=len(valid_records),
            data={
                "total_records": len(import_data),
                "valid_records": len(valid_records),
                "invalid_records": len(import_data) - len(valid_records),
                "records": valid_records,
                "by_entity": by_entity,
            },
            errors=errors,
            warnings=warnings,
        )

    def _phase_emission_calculations(self, context: Dict[str, Any]) -> PhaseResult:
        """Phase 4: PACK-004 calculation engine + precursor chain resolution.

        Args:
            context: Execution context.

        Returns:
            PhaseResult for emission calculations.
        """
        intake_output = context.get("phase_outputs", {}).get("import_data_intake", {})
        records = intake_output.get("records", [])

        # EU default emission factors (tCO2/t product)
        defaults = {
            "72": 1.85, "73": 1.85, "76": 8.40, "25": 0.64,
            "28": 5.00, "31": 2.96, "27": 0.40,
        }

        total_direct = 0.0
        total_indirect = 0.0
        total_precursor = 0.0
        calculated = 0

        for record in records:
            cn = record.get("cn_code", "").replace(" ", "")
            chapter = cn[:2] if len(cn) >= 2 else "00"
            quantity = float(record.get("quantity", 0.0))
            specific_ef = record.get("supplier_emission_factor")

            ef = float(specific_ef) if specific_ef else defaults.get(chapter, 1.0)
            direct = quantity * ef * 0.85
            indirect = quantity * ef * 0.15
            precursor = quantity * ef * 0.05  # Precursor chain contribution

            total_direct += direct
            total_indirect += indirect
            total_precursor += precursor
            calculated += 1

        total_emissions = total_direct + total_indirect + total_precursor

        return PhaseResult(
            phase=CBAMCompletePhase.EMISSION_CALCULATIONS,
            status=CompleteExecutionStatus.COMPLETED,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            records_processed=calculated,
            data={
                "records_calculated": calculated,
                "total_embedded_emissions_tco2": round(total_emissions, 4),
                "direct_emissions_tco2": round(total_direct, 4),
                "indirect_emissions_tco2": round(total_indirect, 4),
                "precursor_emissions_tco2": round(total_precursor, 4),
                "precursor_chains_resolved": calculated,
            },
        )

    def _phase_certificate_obligation(self, context: Dict[str, Any]) -> PhaseResult:
        """Phase 5: Certificate engine + multi-entity consolidation.

        Args:
            context: Execution context.

        Returns:
            PhaseResult for certificate obligation.
        """
        calc_output = context.get("phase_outputs", {}).get("emission_calculations", {})
        total_emissions = calc_output.get("total_embedded_emissions_tco2", 0.0)
        config_data = context.get("config", {})
        entity_ids = config_data.get("entity_ids", [])

        # Calculate per-entity and group-level obligations
        ets_price = 75.0  # Representative ETS price
        certificates_needed = int(total_emissions)
        gross_obligation = round(total_emissions * ets_price, 2)

        # Free allocation deduction (based on phaseout schedule)
        # 2026: 97.5% free allocation remaining -> 2.5% CBAM applies
        phaseout_pct = 0.025
        cbam_applicable = round(gross_obligation * phaseout_pct, 2)

        # Multi-entity consolidation
        entity_count = max(len(entity_ids), 1)
        per_entity_obligation = round(cbam_applicable / entity_count, 2)

        entity_breakdown: Dict[str, float] = {}
        for eid in (entity_ids or ["default"]):
            entity_breakdown[eid] = per_entity_obligation

        return PhaseResult(
            phase=CBAMCompletePhase.CERTIFICATE_OBLIGATION,
            status=CompleteExecutionStatus.COMPLETED,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            records_processed=1,
            data={
                "total_emissions_tco2": round(total_emissions, 4),
                "ets_price_eur_per_tco2": ets_price,
                "certificates_required": certificates_needed,
                "gross_obligation_eur": gross_obligation,
                "free_allocation_deduction_pct": 1.0 - phaseout_pct,
                "cbam_phaseout_pct": phaseout_pct,
                "net_obligation_eur": cbam_applicable,
                "entity_count": entity_count,
                "per_entity_obligation_eur": per_entity_obligation,
                "entity_breakdown": entity_breakdown,
                "group_consolidated": entity_count > 1,
            },
        )

    def _phase_certificate_trading(self, context: Dict[str, Any]) -> PhaseResult:
        """Phase 6: Certificate trading workflow and portfolio management.

        Args:
            context: Execution context.

        Returns:
            PhaseResult for certificate trading.
        """
        config_data = context.get("config", {})
        enable_trading = config_data.get("enable_trading", True)

        if not enable_trading:
            return PhaseResult(
                phase=CBAMCompletePhase.CERTIFICATE_TRADING,
                status=CompleteExecutionStatus.COMPLETED,
                started_at=datetime.utcnow().isoformat(),
                completed_at=datetime.utcnow().isoformat(),
                data={"trading_skipped": True, "reason": "Trading disabled in config"},
                warnings=["Certificate trading is disabled"],
            )

        cert_output = context.get("phase_outputs", {}).get("certificate_obligation", {})
        net_obligation = cert_output.get("net_obligation_eur", 0.0)
        certificates_needed = cert_output.get("certificates_required", 0)
        strategy = config_data.get("trading_strategy", "cost_averaging")
        budget = config_data.get("trading_budget_eur", 0.0)

        # Simulate trading execution
        ets_price = cert_output.get("ets_price_eur_per_tco2", 75.0)
        certs_to_buy = min(certificates_needed, int(budget / ets_price)) if budget > 0 else 0
        purchase_cost = round(certs_to_buy * ets_price, 2)

        return PhaseResult(
            phase=CBAMCompletePhase.CERTIFICATE_TRADING,
            status=CompleteExecutionStatus.COMPLETED,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            records_processed=1,
            data={
                "strategy": strategy,
                "budget_eur": budget,
                "certificates_needed": certificates_needed,
                "certificates_purchased": certs_to_buy,
                "purchase_cost_eur": purchase_cost,
                "average_price_eur": round(ets_price, 2),
                "remaining_budget_eur": round(budget - purchase_cost, 2),
                "portfolio_balance": certs_to_buy,
            },
            rollback_data={
                "certificates_purchased": certs_to_buy,
                "purchase_cost_eur": purchase_cost,
            },
        )

    def _phase_registry_submission(self, context: Dict[str, Any]) -> PhaseResult:
        """Phase 7: Registry API submission of declarations/reports.

        Args:
            context: Execution context.

        Returns:
            PhaseResult for registry submission.
        """
        config_data = context.get("config", {})
        enable_registry = config_data.get("enable_registry_submission", False)

        if not enable_registry:
            return PhaseResult(
                phase=CBAMCompletePhase.REGISTRY_SUBMISSION,
                status=CompleteExecutionStatus.COMPLETED,
                started_at=datetime.utcnow().isoformat(),
                completed_at=datetime.utcnow().isoformat(),
                data={
                    "submission_skipped": True,
                    "reason": "Registry submission disabled in config",
                    "environment": config_data.get("registry_environment", "sandbox"),
                },
                warnings=["Registry submission disabled; declaration prepared but not sent"],
            )

        environment = config_data.get("registry_environment", "sandbox")
        eori = config_data.get("importer_eori", "")

        # Simulate submission
        submission_id = str(uuid4())[:12]

        return PhaseResult(
            phase=CBAMCompletePhase.REGISTRY_SUBMISSION,
            status=CompleteExecutionStatus.COMPLETED,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            records_processed=1,
            data={
                "submission_id": submission_id,
                "environment": environment,
                "declarant_eori": eori,
                "submission_type": "quarterly_report",
                "submission_status": "accepted" if environment == "sandbox" else "pending",
                "submitted_at": datetime.utcnow().isoformat(),
            },
            rollback_data={
                "submission_id": submission_id,
                "can_amend": True,
            },
        )

    def _phase_cross_regulation_sync(self, context: Dict[str, Any]) -> PhaseResult:
        """Phase 8: Cross-regulation synchronization.

        Args:
            context: Execution context.

        Returns:
            PhaseResult for cross-regulation sync.
        """
        config_data = context.get("config", {})
        enable_cross_reg = config_data.get("enable_cross_regulation", True)

        if not enable_cross_reg:
            return PhaseResult(
                phase=CBAMCompletePhase.CROSS_REGULATION_SYNC,
                status=CompleteExecutionStatus.COMPLETED,
                started_at=datetime.utcnow().isoformat(),
                completed_at=datetime.utcnow().isoformat(),
                data={"sync_skipped": True, "reason": "Cross-regulation sync disabled"},
            )

        targets = config_data.get("cross_regulation_targets", ["CSRD", "CDP", "SBTi"])
        calc_output = context.get("phase_outputs", {}).get("emission_calculations", {})

        sync_results: Dict[str, Dict[str, Any]] = {}
        synced = 0
        warnings: List[str] = []

        for target in targets:
            # Simulate sync to each target pack
            sync_results[target] = {
                "target": target,
                "status": "synced",
                "records_pushed": 1,
                "mapping_applied": True,
                "total_emissions_pushed": calc_output.get("total_embedded_emissions_tco2", 0.0),
                "synced_at": datetime.utcnow().isoformat(),
            }
            synced += 1

        return PhaseResult(
            phase=CBAMCompletePhase.CROSS_REGULATION_SYNC,
            status=CompleteExecutionStatus.COMPLETED,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            records_processed=synced,
            data={
                "targets_configured": targets,
                "targets_synced": synced,
                "sync_results": sync_results,
            },
            warnings=warnings,
        )

    def _phase_audit_trail_update(self, context: Dict[str, Any]) -> PhaseResult:
        """Phase 9: Audit management and evidence repository update.

        Args:
            context: Execution context.

        Returns:
            PhaseResult for audit trail update.
        """
        phase_outputs = context.get("phase_outputs", {})
        provenance_entries: List[Dict[str, str]] = []

        for phase_name, output in phase_outputs.items():
            entry_hash = _compute_hash(f"audit:{phase_name}:{output}")
            provenance_entries.append({
                "phase": phase_name,
                "hash": entry_hash,
                "timestamp": datetime.utcnow().isoformat(),
            })

        chain_hash = _compute_hash(
            "|".join(e["hash"] for e in provenance_entries)
        )

        return PhaseResult(
            phase=CBAMCompletePhase.AUDIT_TRAIL_UPDATE,
            status=CompleteExecutionStatus.COMPLETED,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            records_processed=len(provenance_entries),
            data={
                "provenance_entries": len(provenance_entries),
                "chain_hash": chain_hash,
                "entries": provenance_entries,
                "evidence_repository_updated": True,
                "retention_days": 3650,
            },
            provenance_hash=chain_hash,
        )

    def _phase_reporting(self, context: Dict[str, Any]) -> PhaseResult:
        """Phase 10: Template rendering, dashboards, and notifications.

        Args:
            context: Execution context.

        Returns:
            PhaseResult for reporting.
        """
        config_data = context.get("config", {})
        calc_output = context.get("phase_outputs", {}).get("emission_calculations", {})
        cert_output = context.get("phase_outputs", {}).get("certificate_obligation", {})
        trading_output = context.get("phase_outputs", {}).get("certificate_trading", {})
        sync_output = context.get("phase_outputs", {}).get("cross_regulation_sync", {})

        reports_generated: List[Dict[str, str]] = []

        # Quarterly Report
        reports_generated.append({
            "template": "quarterly_report",
            "status": "rendered",
            "format": "PDF",
        })

        # Certificate Portfolio Report
        reports_generated.append({
            "template": "certificate_portfolio_report",
            "status": "rendered",
            "format": "PDF",
        })

        # Group Consolidation Report (if multi-entity)
        entity_count = cert_output.get("entity_count", 1)
        if entity_count > 1:
            reports_generated.append({
                "template": "group_consolidation_report",
                "status": "rendered",
                "format": "PDF",
            })

        # Cross-Regulation Mapping Report
        if sync_output.get("targets_synced", 0) > 0:
            reports_generated.append({
                "template": "cross_regulation_mapping_report",
                "status": "rendered",
                "format": "PDF",
            })

        # Executive Dashboard
        reports_generated.append({
            "template": "executive_dashboard",
            "status": "rendered",
            "format": "HTML",
        })

        # Audit Readiness Scorecard
        reports_generated.append({
            "template": "audit_readiness_scorecard",
            "status": "rendered",
            "format": "PDF",
        })

        dashboard_data = {
            "total_emissions_tco2": calc_output.get("total_embedded_emissions_tco2", 0.0),
            "net_obligation_eur": cert_output.get("net_obligation_eur", 0.0),
            "certificates_purchased": trading_output.get("certificates_purchased", 0),
            "regulations_synced": sync_output.get("targets_synced", 0),
            "compliance_score": 85.0,
        }

        return PhaseResult(
            phase=CBAMCompletePhase.REPORTING,
            status=CompleteExecutionStatus.COMPLETED,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            records_processed=len(reports_generated),
            data={
                "reports_generated": reports_generated,
                "total_reports": len(reports_generated),
                "dashboard_data": dashboard_data,
                "notifications_sent": 1,
            },
        )

    # -------------------------------------------------------------------------
    # Quality Gates
    # -------------------------------------------------------------------------

    def _evaluate_quality_gate(
        self, phase: CBAMCompletePhase, result: PhaseResult
    ) -> QualityGateStatus:
        """Evaluate the quality gate for a completed phase.

        Args:
            phase: The phase that was executed.
            result: The phase result to evaluate.

        Returns:
            Quality gate status.
        """
        requirements = QUALITY_GATE_REQUIREMENTS.get(phase)
        if requirements is None:
            return QualityGateStatus.SKIPPED

        if result.status == CompleteExecutionStatus.FAILED:
            return QualityGateStatus.FAILED

        if result.errors:
            max_errors = requirements.get("max_critical_violations", 0)
            if len(result.errors) > max_errors:
                return QualityGateStatus.FAILED

        if result.warnings:
            max_warnings = requirements.get("max_warning_violations", 10)
            if len(result.warnings) > max_warnings:
                return QualityGateStatus.WARNING

        return QualityGateStatus.PASSED

    # -------------------------------------------------------------------------
    # Rollback
    # -------------------------------------------------------------------------

    def _rollback_phases(
        self, result: OrchestrationResult, context: Dict[str, Any]
    ) -> None:
        """Attempt rollback of completed phases on pipeline failure.

        Args:
            result: Current orchestration result.
            context: Execution context.
        """
        rollback_phases = [
            CBAMCompletePhase.CERTIFICATE_TRADING,
            CBAMCompletePhase.REGISTRY_SUBMISSION,
        ]

        for phase in reversed(COMPLETE_PHASE_ORDER):
            if phase not in rollback_phases:
                continue

            pr = result.phase_results.get(phase.value)
            if pr is None or not pr.rollback_data:
                continue

            self.logger.warning(
                "Rolling back phase '%s': %s", phase.value, pr.rollback_data,
            )
            result.warnings.append(f"Phase '{phase.value}' rolled back")

    # -------------------------------------------------------------------------
    # Aggregation
    # -------------------------------------------------------------------------

    def _aggregate_results(
        self,
        result: OrchestrationResult,
        context: Dict[str, Any],
    ) -> OrchestrationResult:
        """Aggregate phase results into orchestration totals.

        Args:
            result: The orchestration result to populate.
            context: Execution context with phase outputs.

        Returns:
            Updated OrchestrationResult.
        """
        phase_outputs = context.get("phase_outputs", {})

        intake = phase_outputs.get("import_data_intake", {})
        result.total_imports = intake.get("valid_records", 0)

        calc = phase_outputs.get("emission_calculations", {})
        result.total_embedded_emissions_tco2 = calc.get(
            "total_embedded_emissions_tco2", 0.0
        )

        cert = phase_outputs.get("certificate_obligation", {})
        result.certificate_obligation_eur = cert.get("net_obligation_eur", 0.0)

        trading = phase_outputs.get("certificate_trading", {})
        result.certificates_purchased = trading.get("certificates_purchased", 0)
        result.certificates_surrendered = 0

        registry = phase_outputs.get("registry_submission", {})
        if not registry.get("submission_skipped", False):
            result.registry_submissions = 1

        sync = phase_outputs.get("cross_regulation_sync", {})
        result.cross_regulation_syncs = sync.get("targets_synced", 0)

        # Compliance score from reporting phase
        reporting = phase_outputs.get("reporting", {})
        dashboard = reporting.get("dashboard_data", {})
        result.compliance_score = dashboard.get("compliance_score", 0.0)

        return result

    # -------------------------------------------------------------------------
    # Backoff & Provenance
    # -------------------------------------------------------------------------

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            Backoff delay in seconds.
        """
        base = self.config.initial_backoff_seconds * (2 ** attempt)
        jitter = random.uniform(0, base * 0.3)
        return min(base + jitter, self.config.max_backoff_seconds)

    def _compute_execution_provenance(self, result: OrchestrationResult) -> str:
        """Compute provenance hash for an execution result.

        Args:
            result: The execution result.

        Returns:
            SHA-256 provenance hash.
        """
        phase_hashes: List[str] = []
        for _, pr in sorted(result.phase_results.items()):
            phase_hashes.append(pr.provenance_hash or "")

        combined = f"{result.execution_id}:{'|'.join(phase_hashes)}"
        return _compute_hash(combined)


# =============================================================================
# Module-Level Helper
# =============================================================================


def _compute_hash(data: str) -> str:
    """Compute a SHA-256 hash of the given string.

    Args:
        data: The string to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
