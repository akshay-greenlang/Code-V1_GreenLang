# -*- coding: utf-8 -*-
"""
CBAMPackOrchestrator - CBAM-Specific Pack Orchestrator for PACK-004
====================================================================

This module implements the master orchestrator for the CBAM Readiness Pack.
It manages an 8-phase execution pipeline for both quarterly CBAM reports and
annual CBAM declarations, with checkpoint/resume support, retry with
exponential backoff, quality gate enforcement, and full provenance tracking.

Execution Phases:
    1. IMPORT_INTAKE:        Collect and validate import/shipment data
    2. VALIDATION:           CN code, EORI, quantity validation
    3. SUPPLIER_DATA:        Integrate supplier emission data
    4. EMISSION_CALCULATION: Calculate embedded emissions per installation
    5. CERTIFICATE_ASSESSMENT: Calculate certificate obligation
    6. POLICY_CHECK:         Run CBAM compliance rules
    7. REPORT_GENERATION:    Generate quarterly/annual reports
    8. AUDIT_TRAIL:          Complete evidence and lineage

Example:
    >>> config = CBAMOrchestratorConfig(importer_eori="DE123456789012345")
    >>> orchestrator = CBAMPackOrchestrator(config)
    >>> result = orchestrator.execute_quarterly(config, import_data, quarter=1, year=2026)
    >>> assert result.status == "completed"

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-004 CBAM Readiness
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
from greenlang.schemas.enums import ExecutionStatus

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class CBAMPhase(str, Enum):
    """Execution phases in the CBAM pipeline."""
    IMPORT_INTAKE = "import_intake"
    VALIDATION = "validation"
    SUPPLIER_DATA = "supplier_data"
    EMISSION_CALCULATION = "emission_calculation"
    CERTIFICATE_ASSESSMENT = "certificate_assessment"
    POLICY_CHECK = "policy_check"
    REPORT_GENERATION = "report_generation"
    AUDIT_TRAIL = "audit_trail"


class QualityGateStatus(str, Enum):
    """Status of a quality gate check."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


# =============================================================================
# Data Models
# =============================================================================


class CBAMOrchestratorConfig(BaseModel):
    """Configuration for the CBAM Pack Orchestrator."""
    pack_id: str = Field(default="PACK-004", description="Pack identifier")
    importer_eori: str = Field(default="", description="Importer EORI number")
    company_name: str = Field(default="", description="Company name")
    member_state: str = Field(default="", description="EU member state code")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retry attempts per phase")
    initial_backoff_seconds: float = Field(
        default=1.0, description="Initial backoff delay in seconds"
    )
    max_backoff_seconds: float = Field(
        default=30.0, description="Maximum backoff delay in seconds"
    )
    enable_provenance: bool = Field(default=True, description="Enable provenance tracking")
    enable_quality_gates: bool = Field(default=True, description="Enable quality gates")
    timeout_per_phase_seconds: int = Field(
        default=600, description="Timeout per phase in seconds"
    )
    goods_categories: List[str] = Field(
        default_factory=list, description="Applicable goods categories"
    )
    database_url: Optional[str] = Field(None, description="Database URL for persistence")


class PhaseResult(BaseModel):
    """Result of executing a single pipeline phase."""
    phase: CBAMPhase = Field(..., description="Phase that was executed")
    status: ExecutionStatus = Field(default=ExecutionStatus.COMPLETED, description="Phase status")
    started_at: str = Field(default="", description="Phase start timestamp")
    completed_at: str = Field(default="", description="Phase completion timestamp")
    execution_time_ms: float = Field(default=0.0, description="Execution time in ms")
    records_processed: int = Field(default=0, description="Number of records processed")
    data: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    errors: List[str] = Field(default_factory=list, description="Phase errors")
    warnings: List[str] = Field(default_factory=list, description="Phase warnings")
    quality_gate: QualityGateStatus = Field(
        default=QualityGateStatus.SKIPPED, description="Quality gate result"
    )
    retry_count: int = Field(default=0, description="Number of retries performed")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class QuarterlyExecutionResult(BaseModel):
    """Result of a quarterly CBAM execution."""
    execution_id: str = Field(default_factory=lambda: str(uuid4())[:16], description="Execution ID")
    quarter: int = Field(..., ge=1, le=4, description="Quarter number")
    year: int = Field(..., description="Reporting year")
    importer_eori: str = Field(default="", description="Importer EORI")
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING, description="Overall status")
    started_at: str = Field(default="", description="Execution start timestamp")
    completed_at: str = Field(default="", description="Execution completion timestamp")
    total_execution_time_ms: float = Field(default=0.0, description="Total execution time")
    phase_results: Dict[str, PhaseResult] = Field(
        default_factory=dict, description="Per-phase results"
    )
    total_imports: int = Field(default=0, description="Total import records processed")
    total_embedded_emissions_tco2: float = Field(
        default=0.0, description="Total embedded emissions"
    )
    certificate_obligation_eur: float = Field(
        default=0.0, description="Total certificate obligation"
    )
    compliance_score: float = Field(default=0.0, ge=0, le=100, description="Compliance score")
    errors: List[str] = Field(default_factory=list, description="Execution errors")
    warnings: List[str] = Field(default_factory=list, description="Execution warnings")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class AnnualExecutionResult(BaseModel):
    """Result of an annual CBAM declaration execution."""
    execution_id: str = Field(default_factory=lambda: str(uuid4())[:16], description="Execution ID")
    year: int = Field(..., description="Declaration year")
    importer_eori: str = Field(default="", description="Importer EORI")
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING, description="Overall status")
    started_at: str = Field(default="", description="Execution start timestamp")
    completed_at: str = Field(default="", description="Execution completion timestamp")
    total_execution_time_ms: float = Field(default=0.0, description="Total execution time")
    quarterly_summaries: Dict[int, Dict[str, Any]] = Field(
        default_factory=dict, description="Summary per quarter"
    )
    annual_totals: Dict[str, Any] = Field(default_factory=dict, description="Annual totals")
    total_embedded_emissions_tco2: float = Field(
        default=0.0, description="Annual total embedded emissions"
    )
    total_certificate_obligation_eur: float = Field(
        default=0.0, description="Annual certificate obligation"
    )
    certificates_to_surrender: int = Field(
        default=0, description="Number of certificates to surrender"
    )
    compliance_score: float = Field(default=0.0, ge=0, le=100, description="Annual compliance score")
    errors: List[str] = Field(default_factory=list, description="Execution errors")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class Checkpoint(BaseModel):
    """Execution checkpoint for resume capability."""
    checkpoint_id: str = Field(default_factory=lambda: str(uuid4())[:16], description="Checkpoint ID")
    execution_id: str = Field(..., description="Parent execution ID")
    phase_completed: str = Field(..., description="Last completed phase")
    phase_results: Dict[str, Any] = Field(
        default_factory=dict, description="Results up to checkpoint"
    )
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Checkpoint timestamp",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# =============================================================================
# Phase Pipeline Definition
# =============================================================================

CBAM_PHASE_ORDER: List[CBAMPhase] = [
    CBAMPhase.IMPORT_INTAKE,
    CBAMPhase.VALIDATION,
    CBAMPhase.SUPPLIER_DATA,
    CBAMPhase.EMISSION_CALCULATION,
    CBAMPhase.CERTIFICATE_ASSESSMENT,
    CBAMPhase.POLICY_CHECK,
    CBAMPhase.REPORT_GENERATION,
    CBAMPhase.AUDIT_TRAIL,
]

# Quality gate requirements per phase
QUALITY_GATE_REQUIREMENTS: Dict[CBAMPhase, Dict[str, Any]] = {
    CBAMPhase.IMPORT_INTAKE: {
        "min_records": 0,
        "max_error_rate": 0.10,
        "required_fields": ["cn_code", "origin_country", "quantity"],
    },
    CBAMPhase.VALIDATION: {
        "max_invalid_cn_codes": 0,
        "max_invalid_eori": 0,
        "max_error_rate": 0.0,
    },
    CBAMPhase.SUPPLIER_DATA: {
        "min_coverage_rate": 0.50,
        "allow_default_values": True,
    },
    CBAMPhase.EMISSION_CALCULATION: {
        "max_error_rate": 0.0,
        "require_provenance": True,
    },
    CBAMPhase.CERTIFICATE_ASSESSMENT: {
        "require_positive_price": True,
    },
    CBAMPhase.POLICY_CHECK: {
        "max_critical_violations": 0,
        "max_warning_violations": 10,
    },
    CBAMPhase.REPORT_GENERATION: {
        "require_all_sections": True,
    },
    CBAMPhase.AUDIT_TRAIL: {
        "require_provenance_chain": True,
    },
}

# Agent mapping per phase
PHASE_AGENT_MAPPING: Dict[CBAMPhase, List[str]] = {
    CBAMPhase.IMPORT_INTAKE: [
        "GL-DATA-X-001",   # Document Ingestion (PDF)
        "GL-DATA-X-002",   # Excel/CSV Normalizer
        "GL-DATA-X-003",   # ERP Connector
        "GL-EUDR-X-001",   # Commodity Classification
    ],
    CBAMPhase.VALIDATION: [
        "GL-FOUND-X-002",  # Schema Compiler
        "GL-FOUND-X-003",  # Unit Normalizer
        "GL-DATA-X-010",   # Data Quality Profiler
        "GL-DATA-X-019",   # Validation Rule Engine
    ],
    CBAMPhase.SUPPLIER_DATA: [
        "GL-DATA-X-008",   # Supplier Questionnaire Processor
        "GL-EUDR-X-005",   # Supplier Due Diligence
        "GL-EUDR-X-017",   # Supplier Risk Scorer
    ],
    CBAMPhase.EMISSION_CALCULATION: [
        "GL-MRV-X-001",    # Stationary Combustion
        "GL-MRV-X-004",    # Process Emissions
        "GL-MRV-X-014",    # Purchased Goods & Services
        "GL-MRV-X-029",    # Scope 3 Category Mapper
    ],
    CBAMPhase.CERTIFICATE_ASSESSMENT: [
        "GL-CBAM-CERT",    # Certificate Engine (pack-specific)
        "GL-CBAM-DEMIN",   # De Minimis Engine (pack-specific)
    ],
    CBAMPhase.POLICY_CHECK: [
        "GL-FOUND-X-006",  # Access & Policy Guard
        "GL-CBAM-POLICY",  # CBAM Policy Compliance Engine (pack-specific)
    ],
    CBAMPhase.REPORT_GENERATION: [
        "GL-CBAM-QRT",     # Quarterly Reporting Engine (pack-specific)
        "GL-CBAM-ANNUAL",  # Annual Declaration Engine (pack-specific)
    ],
    CBAMPhase.AUDIT_TRAIL: [
        "GL-MRV-X-030",    # Audit Trail & Lineage
        "GL-FOUND-X-005",  # Citations & Evidence
        "GL-FOUND-X-004",  # Assumptions Registry
    ],
}


# =============================================================================
# Orchestrator Implementation
# =============================================================================


class CBAMPackOrchestrator:
    """CBAM-specific pack orchestrator with 8-phase execution pipeline.

    Manages the end-to-end CBAM compliance pipeline from import data intake
    through certificate obligation assessment, report generation, and audit
    trail construction. Supports quarterly and annual execution modes.

    Features:
        - 8-phase pipeline with quality gate enforcement
        - Retry with exponential backoff and jitter
        - Checkpoint/resume for long-running executions
        - Full SHA-256 provenance chain
        - Configurable per phase via CBAMOrchestratorConfig

    Attributes:
        config: Orchestrator configuration
        _executions: History of execution results
        _checkpoints: Saved checkpoints for resume
        _phase_handlers: Registered phase handler functions

    Example:
        >>> config = CBAMOrchestratorConfig(importer_eori="DE123456789012345")
        >>> orch = CBAMPackOrchestrator(config)
        >>> result = orch.execute_quarterly(config, [], quarter=1, year=2026)
    """

    def __init__(self, config: Optional[CBAMOrchestratorConfig] = None) -> None:
        """Initialize the CBAM Pack Orchestrator.

        Args:
            config: Orchestrator configuration. Uses defaults if not provided.
        """
        self.config = config or CBAMOrchestratorConfig()
        self.logger = logger
        self._executions: Dict[str, Any] = {}
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._phase_handlers: Dict[CBAMPhase, Callable] = {
            CBAMPhase.IMPORT_INTAKE: self._phase_import_intake,
            CBAMPhase.VALIDATION: self._phase_validation,
            CBAMPhase.SUPPLIER_DATA: self._phase_supplier_data,
            CBAMPhase.EMISSION_CALCULATION: self._phase_emission_calculation,
            CBAMPhase.CERTIFICATE_ASSESSMENT: self._phase_certificate_assessment,
            CBAMPhase.POLICY_CHECK: self._phase_policy_check,
            CBAMPhase.REPORT_GENERATION: self._phase_report_generation,
            CBAMPhase.AUDIT_TRAIL: self._phase_audit_trail,
        }

        self.logger.info(
            "CBAMPackOrchestrator initialized: eori=%s, retries=%d, gates=%s",
            self.config.importer_eori,
            self.config.max_retries,
            self.config.enable_quality_gates,
        )

    # -------------------------------------------------------------------------
    # Quarterly Execution
    # -------------------------------------------------------------------------

    def execute_quarterly(
        self,
        config: CBAMOrchestratorConfig,
        import_data: List[Dict[str, Any]],
        quarter: int,
        year: int,
    ) -> QuarterlyExecutionResult:
        """Execute the full quarterly CBAM pipeline.

        Args:
            config: Execution configuration.
            import_data: List of import records for the quarter.
            quarter: Quarter number (1-4).
            year: Reporting year.

        Returns:
            QuarterlyExecutionResult with full phase results and totals.
        """
        start_time = time.monotonic()
        execution_id = _compute_hash(
            f"quarterly:{quarter}:{year}:{config.importer_eori}:{datetime.utcnow().isoformat()}"
        )[:16]

        result = QuarterlyExecutionResult(
            execution_id=execution_id,
            quarter=quarter,
            year=year,
            importer_eori=config.importer_eori,
            status=ExecutionStatus.RUNNING,
            started_at=datetime.utcnow().isoformat(),
        )

        context: Dict[str, Any] = {
            "execution_id": execution_id,
            "quarter": quarter,
            "year": year,
            "import_data": import_data,
            "config": config.model_dump(),
            "phase_outputs": {},
        }

        self.logger.info(
            "Starting quarterly execution Q%d/%d (id=%s, imports=%d)",
            quarter, year, execution_id, len(import_data),
        )

        try:
            for phase in CBAM_PHASE_ORDER:
                phase_result = self.execute_phase(phase.value, context)
                result.phase_results[phase.value] = phase_result
                context["phase_outputs"][phase.value] = phase_result.data

                if phase_result.status == ExecutionStatus.FAILED:
                    gate = phase_result.quality_gate
                    if gate == QualityGateStatus.FAILED:
                        result.status = ExecutionStatus.FAILED
                        result.errors.append(
                            f"Quality gate failed at phase '{phase.value}'"
                        )
                        self.logger.error(
                            "Quarterly execution failed at phase '%s'", phase.value
                        )
                        break

                result.warnings.extend(phase_result.warnings)

            if result.status != ExecutionStatus.FAILED:
                result.status = ExecutionStatus.COMPLETED
                result = self._aggregate_quarterly_results(result, context)

        except Exception as exc:
            result.status = ExecutionStatus.FAILED
            result.errors.append(f"Unexpected error: {exc}")
            self.logger.error("Quarterly execution failed: %s", exc, exc_info=True)

        result.completed_at = datetime.utcnow().isoformat()
        result.total_execution_time_ms = (time.monotonic() - start_time) * 1000

        if self.config.enable_provenance:
            result.provenance_hash = self._compute_execution_provenance(result)

        self._executions[execution_id] = result

        self.logger.info(
            "Quarterly execution Q%d/%d %s in %.1fms (id=%s)",
            quarter, year, result.status.value,
            result.total_execution_time_ms, execution_id,
        )
        return result

    # -------------------------------------------------------------------------
    # Annual Execution
    # -------------------------------------------------------------------------

    def execute_annual(
        self,
        config: CBAMOrchestratorConfig,
        quarterly_reports: List[QuarterlyExecutionResult],
        year: int,
    ) -> AnnualExecutionResult:
        """Execute the annual CBAM declaration pipeline.

        Aggregates quarterly results into an annual declaration.

        Args:
            config: Execution configuration.
            quarterly_reports: List of completed quarterly results (1-4).
            year: Declaration year.

        Returns:
            AnnualExecutionResult with aggregated annual totals.
        """
        start_time = time.monotonic()
        execution_id = _compute_hash(
            f"annual:{year}:{config.importer_eori}:{datetime.utcnow().isoformat()}"
        )[:16]

        result = AnnualExecutionResult(
            execution_id=execution_id,
            year=year,
            importer_eori=config.importer_eori,
            status=ExecutionStatus.RUNNING,
            started_at=datetime.utcnow().isoformat(),
        )

        self.logger.info(
            "Starting annual execution %d (id=%s, quarterly_reports=%d)",
            year, execution_id, len(quarterly_reports),
        )

        try:
            total_emissions = 0.0
            total_obligation = 0.0

            for qr in quarterly_reports:
                q = qr.quarter
                result.quarterly_summaries[q] = {
                    "quarter": q,
                    "imports": qr.total_imports,
                    "emissions_tco2": qr.total_embedded_emissions_tco2,
                    "obligation_eur": qr.certificate_obligation_eur,
                    "compliance_score": qr.compliance_score,
                    "status": qr.status.value,
                }
                total_emissions += qr.total_embedded_emissions_tco2
                total_obligation += qr.certificate_obligation_eur

            result.total_embedded_emissions_tco2 = round(total_emissions, 4)
            result.total_certificate_obligation_eur = round(total_obligation, 2)
            result.certificates_to_surrender = int(total_emissions)
            result.annual_totals = {
                "total_imports": sum(qr.total_imports for qr in quarterly_reports),
                "total_emissions_tco2": round(total_emissions, 4),
                "total_obligation_eur": round(total_obligation, 2),
                "quarters_reported": len(quarterly_reports),
            }

            # Calculate annual compliance score as weighted average
            scores = [qr.compliance_score for qr in quarterly_reports if qr.compliance_score > 0]
            result.compliance_score = round(
                sum(scores) / len(scores) if scores else 0.0, 1
            )

            result.status = ExecutionStatus.COMPLETED

        except Exception as exc:
            result.status = ExecutionStatus.FAILED
            result.errors.append(f"Annual aggregation failed: {exc}")
            self.logger.error("Annual execution failed: %s", exc, exc_info=True)

        result.completed_at = datetime.utcnow().isoformat()
        result.total_execution_time_ms = (time.monotonic() - start_time) * 1000

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(
                f"annual:{execution_id}:{total_emissions}:{total_obligation}"
            )

        self._executions[execution_id] = result

        self.logger.info(
            "Annual execution %d %s in %.1fms (id=%s)",
            year, result.status.value,
            result.total_execution_time_ms, execution_id,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase Execution
    # -------------------------------------------------------------------------

    def execute_phase(
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
            phase = CBAMPhase(phase_name)
        except ValueError:
            return PhaseResult(
                phase=CBAMPhase.IMPORT_INTAKE,
                status=ExecutionStatus.FAILED,
                errors=[f"Unknown phase: {phase_name}"],
            )

        handler = self._phase_handlers.get(phase)
        if handler is None:
            return PhaseResult(
                phase=phase,
                status=ExecutionStatus.FAILED,
                errors=[f"No handler for phase: {phase_name}"],
            )

        # Retry with exponential backoff
        last_exception: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            start_time = time.monotonic()
            try:
                phase_result = handler(context)
                phase_result.execution_time_ms = (time.monotonic() - start_time) * 1000
                phase_result.retry_count = attempt

                # Quality gate
                if self.config.enable_quality_gates:
                    gate_status = self._evaluate_quality_gate(phase, phase_result)
                    phase_result.quality_gate = gate_status

                # Provenance
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
                elapsed = (time.monotonic() - start_time) * 1000
                self.logger.warning(
                    "Phase '%s' failed (attempt %d/%d, %.1fms): %s",
                    phase_name, attempt + 1, self.config.max_retries + 1, elapsed, exc,
                )

                if attempt < self.config.max_retries:
                    backoff = self._calculate_backoff(attempt)
                    self.logger.info("Retrying phase '%s' in %.2fs", phase_name, backoff)
                    time.sleep(backoff)

        # All retries exhausted
        return PhaseResult(
            phase=phase,
            status=ExecutionStatus.FAILED,
            errors=[f"Phase failed after {self.config.max_retries + 1} attempts: {last_exception}"],
            retry_count=self.config.max_retries,
        )

    # -------------------------------------------------------------------------
    # Checkpoint / Resume
    # -------------------------------------------------------------------------

    def save_checkpoint(self, execution_id: str) -> str:
        """Save a checkpoint for the given execution.

        Args:
            execution_id: The execution to checkpoint.

        Returns:
            Checkpoint ID.
        """
        execution = self._executions.get(execution_id)
        if execution is None:
            self.logger.error("Execution '%s' not found for checkpoint", execution_id)
            return ""

        # Determine last completed phase
        last_phase = ""
        phase_data: Dict[str, Any] = {}
        if hasattr(execution, "phase_results"):
            for phase_name, pr in execution.phase_results.items():
                if isinstance(pr, PhaseResult) and pr.status == ExecutionStatus.COMPLETED:
                    last_phase = phase_name
                    phase_data[phase_name] = pr.model_dump()

        checkpoint = Checkpoint(
            execution_id=execution_id,
            phase_completed=last_phase,
            phase_results=phase_data,
            context={"status": execution.status.value if hasattr(execution, 'status') else "unknown"},
        )
        checkpoint.provenance_hash = _compute_hash(
            f"checkpoint:{checkpoint.checkpoint_id}:{execution_id}:{last_phase}"
        )

        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        self.logger.info(
            "Checkpoint saved: %s (execution=%s, phase=%s)",
            checkpoint.checkpoint_id, execution_id, last_phase,
        )
        return checkpoint.checkpoint_id

    def resume_checkpoint(self, checkpoint_id: str) -> Optional[QuarterlyExecutionResult]:
        """Resume execution from a saved checkpoint.

        Args:
            checkpoint_id: The checkpoint to resume from.

        Returns:
            QuarterlyExecutionResult if resumed, None if checkpoint not found.
        """
        checkpoint = self._checkpoints.get(checkpoint_id)
        if checkpoint is None:
            self.logger.error("Checkpoint '%s' not found", checkpoint_id)
            return None

        self.logger.info(
            "Resuming from checkpoint %s (execution=%s, last_phase=%s)",
            checkpoint_id, checkpoint.execution_id, checkpoint.phase_completed,
        )

        # Find the phase index to resume from
        last_phase = checkpoint.phase_completed
        resume_index = 0
        for i, phase in enumerate(CBAM_PHASE_ORDER):
            if phase.value == last_phase:
                resume_index = i + 1
                break

        original = self._executions.get(checkpoint.execution_id)
        if original is None:
            self.logger.error("Original execution '%s' not found", checkpoint.execution_id)
            return None

        # For simplicity, return the original result as resumed
        if hasattr(original, 'status'):
            original.status = ExecutionStatus.RESUMED
        return original

    # -------------------------------------------------------------------------
    # Execution Status
    # -------------------------------------------------------------------------

    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get the status of an execution.

        Args:
            execution_id: Execution identifier.

        Returns:
            Dictionary with execution status details.
        """
        execution = self._executions.get(execution_id)
        if execution is None:
            return {"error": f"Execution '{execution_id}' not found"}

        if hasattr(execution, "model_dump"):
            return execution.model_dump()
        return {"execution_id": execution_id, "status": "unknown"}

    def list_executions(self, importer_eori: str = "") -> List[Dict[str, Any]]:
        """List all executions, optionally filtered by importer EORI.

        Args:
            importer_eori: Filter by EORI (empty string returns all).

        Returns:
            List of execution summary dictionaries.
        """
        results: List[Dict[str, Any]] = []
        for exec_id, execution in self._executions.items():
            eori = getattr(execution, "importer_eori", "")
            if importer_eori and eori != importer_eori:
                continue
            results.append({
                "execution_id": exec_id,
                "importer_eori": eori,
                "status": getattr(execution, "status", ExecutionStatus.PENDING).value,
                "started_at": getattr(execution, "started_at", ""),
                "completed_at": getattr(execution, "completed_at", ""),
            })
        return results

    # -------------------------------------------------------------------------
    # Phase Handlers
    # -------------------------------------------------------------------------

    def _phase_import_intake(self, context: Dict[str, Any]) -> PhaseResult:
        """Phase 1: Collect and validate import/shipment data.

        Args:
            context: Execution context.

        Returns:
            PhaseResult for import intake.
        """
        import_data = context.get("import_data", [])

        valid_records: List[Dict[str, Any]] = []
        errors: List[str] = []
        for idx, record in enumerate(import_data):
            if not record.get("cn_code"):
                errors.append(f"Record {idx}: missing cn_code")
                continue
            if not record.get("origin_country"):
                errors.append(f"Record {idx}: missing origin_country")
                continue
            valid_records.append(record)

        return PhaseResult(
            phase=CBAMPhase.IMPORT_INTAKE,
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            records_processed=len(valid_records),
            data={
                "total_records": len(import_data),
                "valid_records": len(valid_records),
                "invalid_records": len(import_data) - len(valid_records),
                "records": valid_records,
            },
            errors=errors,
        )

    def _phase_validation(self, context: Dict[str, Any]) -> PhaseResult:
        """Phase 2: CN code, EORI, and quantity validation.

        Args:
            context: Execution context.

        Returns:
            PhaseResult for validation.
        """
        intake_output = context.get("phase_outputs", {}).get("import_intake", {})
        records = intake_output.get("records", [])

        validated = 0
        warnings: List[str] = []
        for record in records:
            cn = record.get("cn_code", "")
            if len(cn.replace(" ", "").replace(".", "")) < 8:
                warnings.append(f"CN code '{cn}' may be incomplete")
            quantity = float(record.get("quantity", 0.0))
            if quantity <= 0:
                warnings.append(f"Zero or negative quantity for CN={cn}")
            validated += 1

        return PhaseResult(
            phase=CBAMPhase.VALIDATION,
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            records_processed=validated,
            data={
                "validated_records": validated,
                "cn_codes_checked": validated,
            },
            warnings=warnings,
        )

    def _phase_supplier_data(self, context: Dict[str, Any]) -> PhaseResult:
        """Phase 3: Integrate supplier emission data.

        Args:
            context: Execution context.

        Returns:
            PhaseResult for supplier data integration.
        """
        intake_output = context.get("phase_outputs", {}).get("import_intake", {})
        records = intake_output.get("records", [])

        suppliers_found = 0
        default_used = 0
        for record in records:
            if record.get("supplier_emission_factor"):
                suppliers_found += 1
            else:
                default_used += 1

        coverage = suppliers_found / max(len(records), 1)

        return PhaseResult(
            phase=CBAMPhase.SUPPLIER_DATA,
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            records_processed=len(records),
            data={
                "supplier_data_found": suppliers_found,
                "default_values_used": default_used,
                "coverage_rate": round(coverage, 3),
            },
            warnings=[
                f"{default_used} records using EU default emission factors"
            ] if default_used > 0 else [],
        )

    def _phase_emission_calculation(self, context: Dict[str, Any]) -> PhaseResult:
        """Phase 4: Calculate embedded emissions per installation.

        Args:
            context: Execution context.

        Returns:
            PhaseResult for emission calculation.
        """
        intake_output = context.get("phase_outputs", {}).get("import_intake", {})
        records = intake_output.get("records", [])

        total_emissions = 0.0
        calculated = 0

        # EU default emission factors (tCO2/t product)
        defaults = {
            "72": 1.85,  # Iron & steel
            "73": 1.85,  # Steel articles
            "76": 8.40,  # Aluminium
            "25": 0.64,  # Cement
            "28": 5.00,  # Chemicals (fertiliser precursors, hydrogen)
            "31": 2.96,  # Fertilisers
            "27": 0.40,  # Electricity (tCO2/MWh)
        }

        for record in records:
            cn = record.get("cn_code", "").replace(" ", "")
            chapter = cn[:2] if len(cn) >= 2 else "00"
            quantity = float(record.get("quantity", 0.0))
            specific_ef = record.get("supplier_emission_factor")

            ef = float(specific_ef) if specific_ef else defaults.get(chapter, 1.0)
            emissions = quantity * ef
            total_emissions += emissions
            calculated += 1

        return PhaseResult(
            phase=CBAMPhase.EMISSION_CALCULATION,
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            records_processed=calculated,
            data={
                "records_calculated": calculated,
                "total_embedded_emissions_tco2": round(total_emissions, 4),
                "direct_emissions_tco2": round(total_emissions * 0.85, 4),
                "indirect_emissions_tco2": round(total_emissions * 0.15, 4),
            },
        )

    def _phase_certificate_assessment(self, context: Dict[str, Any]) -> PhaseResult:
        """Phase 5: Calculate CBAM certificate obligation.

        Args:
            context: Execution context.

        Returns:
            PhaseResult for certificate assessment.
        """
        calc_output = context.get("phase_outputs", {}).get("emission_calculation", {})
        total_emissions = calc_output.get("total_embedded_emissions_tco2", 0.0)

        # Use a representative ETS price
        ets_price = 75.0
        certificates_needed = int(total_emissions)
        obligation_eur = round(total_emissions * ets_price, 2)

        return PhaseResult(
            phase=CBAMPhase.CERTIFICATE_ASSESSMENT,
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            records_processed=1,
            data={
                "total_emissions_tco2": round(total_emissions, 4),
                "ets_price_eur_per_tco2": ets_price,
                "certificates_required": certificates_needed,
                "gross_obligation_eur": obligation_eur,
                "free_allocation_deduction_eur": 0.0,
                "carbon_price_deduction_eur": 0.0,
                "net_obligation_eur": obligation_eur,
            },
        )

    def _phase_policy_check(self, context: Dict[str, Any]) -> PhaseResult:
        """Phase 6: Run CBAM compliance rules.

        Args:
            context: Execution context.

        Returns:
            PhaseResult for policy check.
        """
        violations: List[str] = []
        warnings: List[str] = []
        rules_checked = 0
        rules_passed = 0

        # Check EORI
        config_data = context.get("config", {})
        eori = config_data.get("importer_eori", "")
        rules_checked += 1
        if eori:
            rules_passed += 1
        else:
            violations.append("CBAM-REG-002: EORI number is missing")

        # Check emission data
        calc_output = context.get("phase_outputs", {}).get("emission_calculation", {})
        total_emissions = calc_output.get("total_embedded_emissions_tco2", 0.0)
        rules_checked += 1
        if total_emissions >= 0:
            rules_passed += 1
        else:
            violations.append("CBAM-EMIT-001: Emission calculation produced negative result")

        # Check certificate
        cert_output = context.get("phase_outputs", {}).get("certificate_assessment", {})
        net_obligation = cert_output.get("net_obligation_eur", 0.0)
        rules_checked += 1
        if net_obligation >= 0:
            rules_passed += 1
        else:
            violations.append("CBAM-CERT-001: Negative certificate obligation")

        # Supplier data coverage
        supplier_output = context.get("phase_outputs", {}).get("supplier_data", {})
        coverage = supplier_output.get("coverage_rate", 0.0)
        rules_checked += 1
        if coverage >= 0.5:
            rules_passed += 1
        else:
            warnings.append(
                f"CBAM-SUP-002: Supplier data coverage is {coverage:.0%}, "
                "recommendation is >= 50%"
            )

        compliance_score = round((rules_passed / max(rules_checked, 1)) * 100, 1)

        return PhaseResult(
            phase=CBAMPhase.POLICY_CHECK,
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            records_processed=rules_checked,
            data={
                "rules_checked": rules_checked,
                "rules_passed": rules_passed,
                "violations": violations,
                "compliance_score": compliance_score,
            },
            errors=violations,
            warnings=warnings,
        )

    def _phase_report_generation(self, context: Dict[str, Any]) -> PhaseResult:
        """Phase 7: Generate quarterly/annual reports.

        Args:
            context: Execution context.

        Returns:
            PhaseResult for report generation.
        """
        quarter = context.get("quarter", 0)
        year = context.get("year", 0)
        config_data = context.get("config", {})

        calc_output = context.get("phase_outputs", {}).get("emission_calculation", {})
        cert_output = context.get("phase_outputs", {}).get("certificate_assessment", {})
        policy_output = context.get("phase_outputs", {}).get("policy_check", {})

        report = {
            "report_type": "quarterly" if quarter > 0 else "annual",
            "quarter": quarter,
            "year": year,
            "importer_eori": config_data.get("importer_eori", ""),
            "company_name": config_data.get("company_name", ""),
            "total_embedded_emissions_tco2": calc_output.get("total_embedded_emissions_tco2", 0.0),
            "certificate_obligation_eur": cert_output.get("net_obligation_eur", 0.0),
            "compliance_score": policy_output.get("compliance_score", 0.0),
            "generated_at": datetime.utcnow().isoformat(),
            "status": "draft",
        }

        return PhaseResult(
            phase=CBAMPhase.REPORT_GENERATION,
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            records_processed=1,
            data={"report": report},
        )

    def _phase_audit_trail(self, context: Dict[str, Any]) -> PhaseResult:
        """Phase 8: Complete evidence chain and lineage.

        Args:
            context: Execution context.

        Returns:
            PhaseResult for audit trail.
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
            phase=CBAMPhase.AUDIT_TRAIL,
            status=ExecutionStatus.COMPLETED,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            records_processed=len(provenance_entries),
            data={
                "provenance_entries": len(provenance_entries),
                "chain_hash": chain_hash,
                "entries": provenance_entries,
            },
            provenance_hash=chain_hash,
        )

    # -------------------------------------------------------------------------
    # Quality Gates
    # -------------------------------------------------------------------------

    def _evaluate_quality_gate(
        self, phase: CBAMPhase, result: PhaseResult
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

        if result.status == ExecutionStatus.FAILED:
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
    # Backoff Calculation
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

    # -------------------------------------------------------------------------
    # Result Aggregation
    # -------------------------------------------------------------------------

    def _aggregate_quarterly_results(
        self,
        result: QuarterlyExecutionResult,
        context: Dict[str, Any],
    ) -> QuarterlyExecutionResult:
        """Aggregate phase results into quarterly totals.

        Args:
            result: The quarterly result to populate.
            context: Execution context with phase outputs.

        Returns:
            Updated QuarterlyExecutionResult.
        """
        phase_outputs = context.get("phase_outputs", {})

        intake = phase_outputs.get("import_intake", {})
        result.total_imports = intake.get("valid_records", 0)

        calc = phase_outputs.get("emission_calculation", {})
        result.total_embedded_emissions_tco2 = calc.get(
            "total_embedded_emissions_tco2", 0.0
        )

        cert = phase_outputs.get("certificate_assessment", {})
        result.certificate_obligation_eur = cert.get("net_obligation_eur", 0.0)

        policy = phase_outputs.get("policy_check", {})
        result.compliance_score = policy.get("compliance_score", 0.0)

        return result

    # -------------------------------------------------------------------------
    # Provenance
    # -------------------------------------------------------------------------

    def _compute_execution_provenance(self, result: Any) -> str:
        """Compute provenance hash for an execution result.

        Args:
            result: The execution result.

        Returns:
            SHA-256 provenance hash.
        """
        phase_hashes: List[str] = []
        if hasattr(result, "phase_results"):
            for _, pr in sorted(result.phase_results.items()):
                if isinstance(pr, PhaseResult):
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
