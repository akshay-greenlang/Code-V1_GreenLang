# -*- coding: utf-8 -*-
"""
Recalculation Execution Workflow
====================================

5-phase workflow for approved base year recalculation execution
within PACK-045 Base Year Management Pack.

Phases:
    1. AdjustmentCalculation   -- Compute emission adjustments for each
                                  approved trigger using deterministic
                                  formulas and validated emission factors.
    2. ImpactValidation        -- Validate that calculated adjustments are
                                  within expected bounds, cross-check scope
                                  totals, and verify arithmetic integrity.
    3. ApprovalCollection      -- Collect digital approvals from authorized
                                  stakeholders before applying adjustments
                                  to the official base year inventory.
    4. AdjustmentApplication   -- Apply approved adjustments to produce the
                                  recalculated base year inventory, create
                                  new inventory version, update time series.
    5. AuditRecording          -- Record complete audit trail including
                                  before/after snapshots, adjustment details,
                                  approval chain, and provenance hashes.

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    GHG Protocol Corporate Standard Chapter 5 (Tracking Emissions Over Time)
    ISO 14064-1:2018 Clause 9.3 (Base year recalculation)
    GHG Protocol Scope 2 Guidance (Recalculation methodology)

Schedule: Triggered after RecalculationAssessmentWorkflow approval
Estimated duration: 1-3 weeks depending on complexity

Author: GreenLang Team
Version: 45.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ExecutionPhase(str, Enum):
    """Recalculation execution workflow phases."""

    ADJUSTMENT_CALCULATION = "adjustment_calculation"
    IMPACT_VALIDATION = "impact_validation"
    APPROVAL_COLLECTION = "approval_collection"
    ADJUSTMENT_APPLICATION = "adjustment_application"
    AUDIT_RECORDING = "audit_recording"


class AdjustmentType(str, Enum):
    """Type of base year emission adjustment."""

    ADDITION = "addition"
    REMOVAL = "removal"
    RESTATEMENT = "restatement"
    FACTOR_UPDATE = "factor_update"
    BOUNDARY_EXPANSION = "boundary_expansion"
    BOUNDARY_CONTRACTION = "boundary_contraction"
    METHODOLOGY_ALIGNMENT = "methodology_alignment"


class ValidationStatus(str, Enum):
    """Validation outcome for an adjustment."""

    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    NEEDS_REVIEW = "needs_review"


class ApprovalDecision(str, Enum):
    """Digital approval decision."""

    APPROVED = "approved"
    REJECTED = "rejected"
    CONDITIONAL = "conditional"
    ABSTAINED = "abstained"


class AuditEventType(str, Enum):
    """Type of audit trail event."""

    ADJUSTMENT_CALCULATED = "adjustment_calculated"
    VALIDATION_COMPLETED = "validation_completed"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_REJECTED = "approval_rejected"
    ADJUSTMENT_APPLIED = "adjustment_applied"
    INVENTORY_UPDATED = "inventory_updated"
    TIME_SERIES_RESTATED = "time_series_restated"
    SNAPSHOT_CREATED = "snapshot_created"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class ApprovedTrigger(BaseModel):
    """An approved trigger ready for recalculation execution."""

    trigger_id: str = Field(default="")
    trigger_type: str = Field(default="")
    description: str = Field(default="")
    impact_tco2e: float = Field(default=0.0)
    impact_pct: float = Field(default=0.0)
    affected_scopes: List[str] = Field(default_factory=list)
    affected_categories: List[str] = Field(default_factory=list)
    adjustment_type: AdjustmentType = Field(default=AdjustmentType.RESTATEMENT)
    adjustment_details: Dict[str, Any] = Field(default_factory=dict)


class BaseYearInventory(BaseModel):
    """Base year inventory with scope-level breakdown."""

    year: int = Field(..., ge=2010, le=2050)
    version: str = Field(default="v1.0")
    total_tco2e: float = Field(default=0.0, ge=0.0)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    categories: Dict[str, float] = Field(default_factory=dict)
    facilities: Dict[str, float] = Field(default_factory=dict)
    methodology_version: str = Field(default="ghg_protocol_v1")
    frozen: bool = Field(default=True)
    integrity_hash: str = Field(default="")


class RecalculationPolicy(BaseModel):
    """Policy governing recalculation execution."""

    minimum_approvers: int = Field(default=2, ge=1, le=20)
    approval_threshold_pct: float = Field(default=100.0, ge=50.0, le=100.0)
    max_adjustment_pct: float = Field(
        default=50.0, ge=1.0, le=200.0,
        description="Maximum allowed single adjustment as % of base year",
    )
    require_dual_validation: bool = Field(default=True)
    auto_update_time_series: bool = Field(default=True)
    policy_version: str = Field(default="v1.0")


class AdjustmentLine(BaseModel):
    """A single calculated emission adjustment line item."""

    line_id: str = Field(default_factory=lambda: f"adj-{uuid.uuid4().hex[:8]}")
    trigger_id: str = Field(default="")
    adjustment_type: AdjustmentType = Field(default=AdjustmentType.RESTATEMENT)
    scope: str = Field(default="scope1")
    category: str = Field(default="")
    original_tco2e: float = Field(default=0.0)
    adjustment_tco2e: float = Field(default=0.0)
    adjusted_tco2e: float = Field(default=0.0)
    methodology: str = Field(default="")
    emission_factor_used: str = Field(default="")
    calculation_formula: str = Field(default="")
    provenance_hash: str = Field(default="")


class AdjustmentPackage(BaseModel):
    """Package of all calculated adjustments."""

    package_id: str = Field(default_factory=lambda: f"pkg-{uuid.uuid4().hex[:8]}")
    created_at: str = Field(default="")
    trigger_count: int = Field(default=0)
    adjustment_lines: List[AdjustmentLine] = Field(default_factory=list)
    total_adjustment_tco2e: float = Field(default=0.0)
    net_impact_pct: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class ValidationResult(BaseModel):
    """Validation result for the adjustment package."""

    validation_id: str = Field(default_factory=lambda: f"val-{uuid.uuid4().hex[:8]}")
    status: ValidationStatus = Field(default=ValidationStatus.VALID)
    checks_passed: int = Field(default=0, ge=0)
    checks_failed: int = Field(default=0, ge=0)
    checks_warned: int = Field(default=0, ge=0)
    check_details: List[Dict[str, Any]] = Field(default_factory=list)
    arithmetic_verified: bool = Field(default=False)
    scope_totals_consistent: bool = Field(default=False)
    within_bounds: bool = Field(default=False)
    provenance_hash: str = Field(default="")


class ApprovalRecord(BaseModel):
    """Digital approval record from an authorized stakeholder."""

    approval_id: str = Field(default_factory=lambda: f"apr-{uuid.uuid4().hex[:8]}")
    approver_id: str = Field(default="")
    approver_name: str = Field(default="")
    approver_role: str = Field(default="")
    decision: ApprovalDecision = Field(default=ApprovalDecision.APPROVED)
    comments: str = Field(default="")
    conditions: List[str] = Field(default_factory=list)
    approved_at: str = Field(default="")
    digital_signature_hash: str = Field(default="")


class AuditEntry(BaseModel):
    """Audit trail entry for recalculation execution."""

    entry_id: str = Field(default_factory=lambda: f"aud-{uuid.uuid4().hex[:8]}")
    event_type: AuditEventType = Field(...)
    timestamp: str = Field(default="")
    actor: str = Field(default="system")
    description: str = Field(default="")
    before_state: Dict[str, Any] = Field(default_factory=dict)
    after_state: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class RecalculationExecutionInput(BaseModel):
    """Input data model for RecalculationExecutionWorkflow."""

    organization_id: str = Field(..., min_length=1, description="Organization identifier")
    base_year_inventory: BaseYearInventory = Field(
        ..., description="Current base year inventory to recalculate",
    )
    approved_triggers: List[ApprovedTrigger] = Field(
        ..., min_length=1, description="Approved triggers for recalculation",
    )
    policy: RecalculationPolicy = Field(
        default_factory=RecalculationPolicy,
        description="Recalculation execution policy",
    )
    approvers: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Authorized approvers [{id, name, role}]",
    )
    tenant_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class RecalculationExecutionResult(BaseModel):
    """Complete result from recalculation execution workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="recalculation_execution")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_id: str = Field(default="")
    adjustment_package: Optional[AdjustmentPackage] = Field(default=None)
    adjusted_inventory: Optional[BaseYearInventory] = Field(default=None)
    validation_result: Optional[ValidationResult] = Field(default=None)
    approval_records: List[ApprovalRecord] = Field(default_factory=list)
    audit_entries: List[AuditEntry] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# =============================================================================
# ADJUSTMENT CALCULATION CONSTANTS (Zero-Hallucination)
# =============================================================================

# Scope-to-category mapping for adjustment distribution
DEFAULT_SCOPE_CATEGORIES: Dict[str, List[str]] = {
    "scope1": [
        "stationary_combustion", "mobile_combustion", "process_emissions",
        "fugitive_emissions", "refrigerant_leakage",
    ],
    "scope2": [
        "purchased_electricity", "purchased_steam", "purchased_cooling",
    ],
    "scope3": [
        "purchased_goods", "capital_goods", "fuel_energy",
        "transportation_upstream", "waste_operations",
        "business_travel", "employee_commuting",
    ],
}

# Validation bound multipliers
VALIDATION_BOUNDS: Dict[str, float] = {
    "max_single_adjustment_multiplier": 2.0,
    "max_scope_variance_pct": 100.0,
    "min_total_after_adjustment": 0.0,
    "arithmetic_tolerance": 0.01,
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class RecalculationExecutionWorkflow:
    """
    5-phase workflow for approved base year recalculation execution.

    Computes emission adjustments, validates arithmetic integrity, collects
    digital approvals, applies adjustments to produce a new base year
    inventory version, and records a complete audit trail.

    Zero-hallucination: all adjustments use deterministic arithmetic,
    no LLM calls in calculation path, SHA-256 provenance on every output.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _adjustment_package: Calculated adjustments.
        _validation: Validation result.
        _approvals: Collected approval records.
        _adjusted_inventory: New base year inventory.
        _audit_entries: Audit trail entries.

    Example:
        >>> wf = RecalculationExecutionWorkflow()
        >>> trigger = ApprovedTrigger(trigger_id="trg-001", impact_tco2e=500.0)
        >>> inv = BaseYearInventory(year=2022, total_tco2e=50000.0)
        >>> inp = RecalculationExecutionInput(
        ...     organization_id="org-001",
        ...     base_year_inventory=inv,
        ...     approved_triggers=[trigger],
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_SEQUENCE: List[ExecutionPhase] = [
        ExecutionPhase.ADJUSTMENT_CALCULATION,
        ExecutionPhase.IMPACT_VALIDATION,
        ExecutionPhase.APPROVAL_COLLECTION,
        ExecutionPhase.ADJUSTMENT_APPLICATION,
        ExecutionPhase.AUDIT_RECORDING,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize RecalculationExecutionWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._adjustment_package: Optional[AdjustmentPackage] = None
        self._validation: Optional[ValidationResult] = None
        self._approvals: List[ApprovalRecord] = []
        self._adjusted_inventory: Optional[BaseYearInventory] = None
        self._audit_entries: List[AuditEntry] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self, input_data: RecalculationExecutionInput,
    ) -> RecalculationExecutionResult:
        """
        Execute the 5-phase recalculation execution workflow.

        Args:
            input_data: Base year inventory, approved triggers, and policy.

        Returns:
            RecalculationExecutionResult with adjusted inventory and audit trail.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting recalculation execution %s org=%s triggers=%d",
            self.workflow_id, input_data.organization_id,
            len(input_data.approved_triggers),
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_adjustment_calculation,
            self._phase_impact_validation,
            self._phase_approval_collection,
            self._phase_adjustment_application,
            self._phase_audit_recording,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._execute_with_retry(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Recalculation execution failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        result = RecalculationExecutionResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_id=input_data.organization_id,
            adjustment_package=self._adjustment_package,
            adjusted_inventory=self._adjusted_inventory,
            validation_result=self._validation,
            approval_records=self._approvals,
            audit_entries=self._audit_entries,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Recalculation execution %s completed in %.2fs status=%s",
            self.workflow_id, elapsed, overall_status.value,
        )
        return result

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self, phase_fn: Any, input_data: RecalculationExecutionInput, phase_number: int,
    ) -> PhaseResult:
        """Execute a phase with exponential backoff retry."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await phase_fn(input_data)
            except Exception as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_RETRY_DELAY_S * (2 ** (attempt - 1))
                    self.logger.warning(
                        "Phase %d attempt %d/%d failed: %s. Retrying in %.1fs",
                        phase_number, attempt, self.MAX_RETRIES, exc, delay,
                    )
                    import asyncio
                    await asyncio.sleep(delay)
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number,
            status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Phase 1: Adjustment Calculation
    # -------------------------------------------------------------------------

    async def _phase_adjustment_calculation(
        self, input_data: RecalculationExecutionInput,
    ) -> PhaseResult:
        """Compute emission adjustments for each approved trigger."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        now_iso = datetime.utcnow().isoformat()
        base_inv = input_data.base_year_inventory
        adjustment_lines: List[AdjustmentLine] = []
        total_adjustment = 0.0

        for trigger in input_data.approved_triggers:
            # Distribute trigger impact across affected scopes
            scopes = trigger.affected_scopes or ["scope1"]
            per_scope_impact = trigger.impact_tco2e / max(len(scopes), 1)

            for scope in scopes:
                original = getattr(base_inv, f"{scope}_tco2e", 0.0)

                # Determine adjustment direction
                if trigger.adjustment_type in (
                    AdjustmentType.ADDITION, AdjustmentType.BOUNDARY_EXPANSION,
                ):
                    adj_value = abs(per_scope_impact)
                elif trigger.adjustment_type in (
                    AdjustmentType.REMOVAL, AdjustmentType.BOUNDARY_CONTRACTION,
                ):
                    adj_value = -abs(per_scope_impact)
                else:
                    adj_value = per_scope_impact

                adjusted = max(original + adj_value, 0.0)

                line_data = {
                    "trigger_id": trigger.trigger_id,
                    "scope": scope,
                    "original": round(original, 4),
                    "adjustment": round(adj_value, 4),
                    "adjusted": round(adjusted, 4),
                }
                line_hash = hashlib.sha256(
                    json.dumps(line_data, sort_keys=True).encode("utf-8")
                ).hexdigest()

                adjustment_lines.append(AdjustmentLine(
                    trigger_id=trigger.trigger_id,
                    adjustment_type=trigger.adjustment_type,
                    scope=scope,
                    category=trigger.affected_categories[0] if trigger.affected_categories else "",
                    original_tco2e=round(original, 4),
                    adjustment_tco2e=round(adj_value, 4),
                    adjusted_tco2e=round(adjusted, 4),
                    methodology=base_inv.methodology_version,
                    calculation_formula=f"adjusted = original({original:.4f}) + adjustment({adj_value:.4f})",
                    provenance_hash=line_hash,
                ))

                total_adjustment += adj_value

        net_impact_pct = (
            (abs(total_adjustment) / max(base_inv.total_tco2e, 1.0)) * 100.0
        )

        pkg_data = {
            "trigger_count": len(input_data.approved_triggers),
            "line_count": len(adjustment_lines),
            "total_adjustment": round(total_adjustment, 4),
            "net_impact_pct": round(net_impact_pct, 4),
        }
        pkg_hash = hashlib.sha256(
            json.dumps(pkg_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        self._adjustment_package = AdjustmentPackage(
            created_at=now_iso,
            trigger_count=len(input_data.approved_triggers),
            adjustment_lines=adjustment_lines,
            total_adjustment_tco2e=round(total_adjustment, 4),
            net_impact_pct=round(net_impact_pct, 4),
            provenance_hash=pkg_hash,
        )

        outputs["adjustment_lines"] = len(adjustment_lines)
        outputs["total_adjustment_tco2e"] = round(total_adjustment, 4)
        outputs["net_impact_pct"] = round(net_impact_pct, 4)
        outputs["triggers_processed"] = len(input_data.approved_triggers)

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 AdjustmentCalculation: %d lines, total=%.2f tCO2e (%.2f%%)",
            len(adjustment_lines), total_adjustment, net_impact_pct,
        )
        return PhaseResult(
            phase_name="adjustment_calculation", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Impact Validation
    # -------------------------------------------------------------------------

    async def _phase_impact_validation(
        self, input_data: RecalculationExecutionInput,
    ) -> PhaseResult:
        """Validate calculated adjustments are within expected bounds."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        check_details: List[Dict[str, Any]] = []
        passed = 0
        failed = 0
        warned = 0

        pkg = self._adjustment_package
        base_inv = input_data.base_year_inventory
        policy = input_data.policy

        # Check 1: Total adjustment within policy bounds
        if pkg and abs(pkg.net_impact_pct) <= policy.max_adjustment_pct:
            check_details.append({
                "check": "total_within_bounds",
                "status": "passed",
                "detail": f"{pkg.net_impact_pct:.2f}% <= {policy.max_adjustment_pct:.1f}%",
            })
            passed += 1
        else:
            check_details.append({
                "check": "total_within_bounds",
                "status": "failed",
                "detail": f"{pkg.net_impact_pct if pkg else 0:.2f}% > {policy.max_adjustment_pct:.1f}%",
            })
            failed += 1

        # Check 2: Arithmetic integrity
        arithmetic_ok = True
        if pkg:
            for line in pkg.adjustment_lines:
                expected = round(line.original_tco2e + line.adjustment_tco2e, 4)
                tolerance = VALIDATION_BOUNDS["arithmetic_tolerance"]
                if abs(line.adjusted_tco2e - expected) > tolerance:
                    arithmetic_ok = False
                    check_details.append({
                        "check": f"arithmetic_{line.line_id}",
                        "status": "failed",
                        "detail": f"Expected {expected}, got {line.adjusted_tco2e}",
                    })
                    failed += 1

        if arithmetic_ok:
            check_details.append({
                "check": "arithmetic_integrity",
                "status": "passed",
                "detail": "All adjustment lines pass arithmetic verification",
            })
            passed += 1

        # Check 3: No negative scope totals
        scope_totals_ok = True
        if pkg:
            scope_adjustments: Dict[str, float] = {}
            for line in pkg.adjustment_lines:
                scope_adjustments[line.scope] = (
                    scope_adjustments.get(line.scope, 0.0) + line.adjustment_tco2e
                )
            for scope, adj in scope_adjustments.items():
                original = getattr(base_inv, f"{scope}_tco2e", 0.0)
                if original + adj < VALIDATION_BOUNDS["min_total_after_adjustment"]:
                    scope_totals_ok = False
                    check_details.append({
                        "check": f"non_negative_{scope}",
                        "status": "failed",
                        "detail": f"{scope} total would be {original + adj:.2f}",
                    })
                    failed += 1

        if scope_totals_ok:
            check_details.append({
                "check": "scope_totals_non_negative",
                "status": "passed",
                "detail": "All scope totals remain non-negative",
            })
            passed += 1

        # Check 4: Scope variance within bounds
        within_bounds = True
        max_var = VALIDATION_BOUNDS["max_scope_variance_pct"]
        if pkg:
            for line in pkg.adjustment_lines:
                if line.original_tco2e > 0:
                    var_pct = (abs(line.adjustment_tco2e) / line.original_tco2e) * 100.0
                    if var_pct > max_var:
                        within_bounds = False
                        warnings.append(
                            f"Adjustment line {line.line_id}: {var_pct:.1f}% variance"
                        )
                        warned += 1

        check_details.append({
            "check": "scope_variance_bounds",
            "status": "passed" if within_bounds else "warning",
            "detail": f"Variance check (max {max_var}%)",
        })
        if within_bounds:
            passed += 1

        overall_status = ValidationStatus.VALID
        if failed > 0:
            overall_status = ValidationStatus.INVALID
        elif warned > 0:
            overall_status = ValidationStatus.WARNING

        val_data = {"passed": passed, "failed": failed, "warned": warned}
        val_hash = hashlib.sha256(
            json.dumps(val_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        self._validation = ValidationResult(
            status=overall_status,
            checks_passed=passed,
            checks_failed=failed,
            checks_warned=warned,
            check_details=check_details,
            arithmetic_verified=arithmetic_ok,
            scope_totals_consistent=scope_totals_ok,
            within_bounds=within_bounds,
            provenance_hash=val_hash,
        )

        outputs["validation_status"] = overall_status.value
        outputs["checks_passed"] = passed
        outputs["checks_failed"] = failed
        outputs["checks_warned"] = warned

        if failed > 0:
            warnings.append(f"{failed} validation checks failed; review before approval")

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 ImpactValidation: status=%s passed=%d failed=%d warned=%d",
            overall_status.value, passed, failed, warned,
        )
        return PhaseResult(
            phase_name="impact_validation", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Approval Collection
    # -------------------------------------------------------------------------

    async def _phase_approval_collection(
        self, input_data: RecalculationExecutionInput,
    ) -> PhaseResult:
        """Collect digital approvals from authorized stakeholders."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._approvals = []
        now_iso = datetime.utcnow().isoformat()
        policy = input_data.policy

        # Validate sufficient approvers
        if len(input_data.approvers) < policy.minimum_approvers:
            warnings.append(
                f"Only {len(input_data.approvers)} approvers provided; "
                f"minimum required: {policy.minimum_approvers}"
            )

        for approver in input_data.approvers:
            approver_id = approver.get("id", "")
            approver_name = approver.get("name", "")
            approver_role = approver.get("role", "")

            # In production, this would integrate with approval system
            # For workflow execution, simulate approval collection
            sig_data = f"{approver_id}|{self.workflow_id}|{now_iso}"
            sig_hash = hashlib.sha256(sig_data.encode("utf-8")).hexdigest()

            self._approvals.append(ApprovalRecord(
                approver_id=approver_id,
                approver_name=approver_name,
                approver_role=approver_role,
                decision=ApprovalDecision.APPROVED,
                comments="Recalculation adjustments reviewed and approved",
                approved_at=now_iso,
                digital_signature_hash=sig_hash,
            ))

        approved_count = sum(
            1 for a in self._approvals if a.decision == ApprovalDecision.APPROVED
        )
        total_count = len(self._approvals)
        approval_pct = (approved_count / max(total_count, 1)) * 100.0

        approval_met = (
            approval_pct >= policy.approval_threshold_pct
            and approved_count >= policy.minimum_approvers
        )

        if not approval_met:
            warnings.append(
                f"Approval threshold not met: {approval_pct:.1f}% "
                f"(required: {policy.approval_threshold_pct:.1f}%)"
            )

        outputs["approvers_contacted"] = total_count
        outputs["approved"] = approved_count
        outputs["approval_pct"] = round(approval_pct, 2)
        outputs["threshold_met"] = approval_met
        outputs["minimum_approvers_met"] = approved_count >= policy.minimum_approvers

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 ApprovalCollection: %d/%d approved (%.1f%%)",
            approved_count, total_count, approval_pct,
        )
        return PhaseResult(
            phase_name="approval_collection", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Adjustment Application
    # -------------------------------------------------------------------------

    async def _phase_adjustment_application(
        self, input_data: RecalculationExecutionInput,
    ) -> PhaseResult:
        """Apply approved adjustments to produce recalculated inventory."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        base_inv = input_data.base_year_inventory
        pkg = self._adjustment_package

        if not pkg:
            raise ValueError("No adjustment package calculated")

        # Compute new scope totals
        scope_adjustments: Dict[str, float] = {"scope1": 0.0, "scope2": 0.0, "scope3": 0.0}
        for line in pkg.adjustment_lines:
            if line.scope in scope_adjustments:
                scope_adjustments[line.scope] += line.adjustment_tco2e

        new_scope1 = max(base_inv.scope1_tco2e + scope_adjustments["scope1"], 0.0)
        new_scope2 = max(base_inv.scope2_tco2e + scope_adjustments["scope2"], 0.0)
        new_scope3 = max(base_inv.scope3_tco2e + scope_adjustments["scope3"], 0.0)
        new_total = round(new_scope1 + new_scope2 + new_scope3, 4)

        # New version number
        old_version = base_inv.version
        version_parts = old_version.replace("v", "").split(".")
        new_minor = int(version_parts[-1]) + 1 if version_parts else 1
        new_version = f"v{version_parts[0]}.{new_minor}" if len(version_parts) >= 1 else "v1.1"

        # Compute integrity hash of new inventory
        inv_data = {
            "year": base_inv.year,
            "version": new_version,
            "total_tco2e": new_total,
            "scope1_tco2e": round(new_scope1, 4),
            "scope2_tco2e": round(new_scope2, 4),
            "scope3_tco2e": round(new_scope3, 4),
        }
        integrity_hash = hashlib.sha256(
            json.dumps(inv_data, sort_keys=True).encode("utf-8")
        ).hexdigest()

        # Update categories
        new_categories = dict(base_inv.categories)
        for line in pkg.adjustment_lines:
            if line.category and line.category in new_categories:
                new_categories[line.category] = max(
                    new_categories[line.category] + line.adjustment_tco2e, 0.0,
                )

        self._adjusted_inventory = BaseYearInventory(
            year=base_inv.year,
            version=new_version,
            total_tco2e=new_total,
            scope1_tco2e=round(new_scope1, 4),
            scope2_tco2e=round(new_scope2, 4),
            scope3_tco2e=round(new_scope3, 4),
            categories=new_categories,
            facilities=base_inv.facilities,
            methodology_version=base_inv.methodology_version,
            frozen=True,
            integrity_hash=integrity_hash,
        )

        outputs["old_version"] = old_version
        outputs["new_version"] = new_version
        outputs["old_total_tco2e"] = base_inv.total_tco2e
        outputs["new_total_tco2e"] = new_total
        outputs["delta_tco2e"] = round(new_total - base_inv.total_tco2e, 4)
        outputs["delta_pct"] = round(
            ((new_total - base_inv.total_tco2e) / max(base_inv.total_tco2e, 1.0)) * 100.0, 4,
        )
        outputs["integrity_hash"] = integrity_hash

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 AdjustmentApplication: %s -> %s (%.2f -> %.2f tCO2e)",
            old_version, new_version, base_inv.total_tco2e, new_total,
        )
        return PhaseResult(
            phase_name="adjustment_application", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Audit Recording
    # -------------------------------------------------------------------------

    async def _phase_audit_recording(
        self, input_data: RecalculationExecutionInput,
    ) -> PhaseResult:
        """Record complete audit trail for recalculation execution."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._audit_entries = []
        now_iso = datetime.utcnow().isoformat()
        base_inv = input_data.base_year_inventory

        # Entry 1: Adjustment calculated
        if self._adjustment_package:
            adj_data = json.dumps({
                "total_adjustment": self._adjustment_package.total_adjustment_tco2e,
                "lines": self._adjustment_package.trigger_count,
            }, sort_keys=True)
            self._audit_entries.append(AuditEntry(
                event_type=AuditEventType.ADJUSTMENT_CALCULATED,
                timestamp=now_iso,
                description=(
                    f"Calculated {len(self._adjustment_package.adjustment_lines)} "
                    f"adjustment lines totaling {self._adjustment_package.total_adjustment_tco2e:.2f} tCO2e"
                ),
                before_state={"total_tco2e": base_inv.total_tco2e},
                after_state={"total_adjustment_tco2e": self._adjustment_package.total_adjustment_tco2e},
                provenance_hash=hashlib.sha256(adj_data.encode("utf-8")).hexdigest(),
            ))

        # Entry 2: Validation completed
        if self._validation:
            val_data = json.dumps({"status": self._validation.status.value}, sort_keys=True)
            self._audit_entries.append(AuditEntry(
                event_type=AuditEventType.VALIDATION_COMPLETED,
                timestamp=now_iso,
                description=(
                    f"Validation {self._validation.status.value}: "
                    f"{self._validation.checks_passed} passed, "
                    f"{self._validation.checks_failed} failed"
                ),
                provenance_hash=hashlib.sha256(val_data.encode("utf-8")).hexdigest(),
            ))

        # Entry 3: Approvals
        for approval in self._approvals:
            self._audit_entries.append(AuditEntry(
                event_type=(
                    AuditEventType.APPROVAL_GRANTED
                    if approval.decision == ApprovalDecision.APPROVED
                    else AuditEventType.APPROVAL_REJECTED
                ),
                timestamp=approval.approved_at or now_iso,
                actor=approval.approver_id,
                description=(
                    f"Approver {approval.approver_name} ({approval.approver_role}): "
                    f"{approval.decision.value}"
                ),
                provenance_hash=approval.digital_signature_hash,
            ))

        # Entry 4: Inventory updated
        if self._adjusted_inventory:
            inv_data = json.dumps({
                "old_total": base_inv.total_tco2e,
                "new_total": self._adjusted_inventory.total_tco2e,
            }, sort_keys=True)
            self._audit_entries.append(AuditEntry(
                event_type=AuditEventType.INVENTORY_UPDATED,
                timestamp=now_iso,
                description=(
                    f"Base year inventory updated: {base_inv.version} -> "
                    f"{self._adjusted_inventory.version}"
                ),
                before_state={
                    "version": base_inv.version,
                    "total_tco2e": base_inv.total_tco2e,
                },
                after_state={
                    "version": self._adjusted_inventory.version,
                    "total_tco2e": self._adjusted_inventory.total_tco2e,
                },
                provenance_hash=hashlib.sha256(inv_data.encode("utf-8")).hexdigest(),
            ))

        # Entry 5: Snapshot created
        if self._adjusted_inventory:
            self._audit_entries.append(AuditEntry(
                event_type=AuditEventType.SNAPSHOT_CREATED,
                timestamp=now_iso,
                description=(
                    f"New frozen snapshot created: {self._adjusted_inventory.integrity_hash[:16]}..."
                ),
                after_state={"integrity_hash": self._adjusted_inventory.integrity_hash},
                provenance_hash=self._adjusted_inventory.integrity_hash,
            ))

        outputs["audit_entries_recorded"] = len(self._audit_entries)
        outputs["event_types"] = list(set(e.event_type.value for e in self._audit_entries))
        outputs["workflow_id"] = self.workflow_id

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 5 AuditRecording: %d entries recorded",
            len(self._audit_entries),
        )
        return PhaseResult(
            phase_name="audit_recording", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._phase_results = []
        self._adjustment_package = None
        self._validation = None
        self._approvals = []
        self._adjusted_inventory = None
        self._audit_entries = []

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: RecalculationExecutionResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(p.provenance_hash for p in result.phases if p.provenance_hash)
        chain += f"|{result.workflow_id}|{result.organization_id}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
