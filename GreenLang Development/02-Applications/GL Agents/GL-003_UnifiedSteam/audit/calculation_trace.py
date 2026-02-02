"""
Calculation Tracer for GL-003 UnifiedSteam SteamSystemOptimizer

This module implements step-by-step calculation audit trails with
formula version tracking for complete transparency and reproducibility.

Key Features:
    - Start/end trace context management
    - Step-by-step calculation recording
    - Constraint check recording
    - Formula version tracking
    - Export for auditor review

Example:
    >>> tracer = CalculationTracer(provenance_tracker)
    >>> ctx = tracer.start_trace("steam_balance")
    >>> tracer.record_step(ctx, "input_validation", inputs, outputs, formula)
    >>> tracer.record_constraint_check(ctx, constraint, result)
    >>> trace = tracer.end_trace(ctx)
    >>> auditable = tracer.export_trace_for_audit(trace.trace_id)

Author: GreenLang Steam Systems Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class TraceStatus(str, Enum):
    """Status of a calculation trace."""

    STARTED = "STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ConstraintType(str, Enum):
    """Types of constraints in calculations."""

    EQUALITY = "EQUALITY"
    INEQUALITY = "INEQUALITY"
    BOUND = "BOUND"
    PHYSICAL = "PHYSICAL"
    SAFETY = "SAFETY"
    OPERATIONAL = "OPERATIONAL"


class FormulaVersion(BaseModel):
    """
    Version information for a calculation formula.

    Tracks the formula definition, version, and hash for
    complete audit trail.
    """

    formula_id: str = Field(..., description="Unique formula identifier")
    formula_name: str = Field(..., description="Human-readable formula name")
    version: str = Field(..., description="Semantic version")
    description: Optional[str] = Field(None, description="Formula description")

    # Formula definition
    formula_expression: str = Field(..., description="Formula expression/equation")
    input_variables: List[str] = Field(
        default_factory=list, description="Input variable names"
    )
    output_variables: List[str] = Field(
        default_factory=list, description="Output variable names"
    )
    constants: Dict[str, float] = Field(
        default_factory=dict, description="Formula constants"
    )

    # Verification
    formula_hash: str = Field(..., description="SHA-256 hash of formula definition")
    validation_status: str = Field(
        default="VALIDATED", description="Validation status"
    )
    validated_date: Optional[datetime] = Field(
        None, description="Date formula was validated"
    )
    validated_by: Optional[str] = Field(None, description="Who validated the formula")

    # References
    reference_standard: Optional[str] = Field(
        None, description="Reference standard (ASME, etc.)"
    )
    source_document: Optional[str] = Field(None, description="Source document")

    class Config:
        frozen = True
        json_encoders = {datetime: lambda v: v.isoformat()}

    @classmethod
    def create(
        cls,
        formula_id: str,
        formula_name: str,
        version: str,
        formula_expression: str,
        input_variables: List[str],
        output_variables: List[str],
        constants: Optional[Dict[str, float]] = None,
        description: Optional[str] = None,
        reference_standard: Optional[str] = None,
    ) -> "FormulaVersion":
        """Factory method to create FormulaVersion with hash."""
        formula_def = {
            "formula_id": formula_id,
            "formula_expression": formula_expression,
            "input_variables": input_variables,
            "output_variables": output_variables,
            "constants": constants or {},
        }
        formula_hash = hashlib.sha256(
            json.dumps(formula_def, sort_keys=True).encode()
        ).hexdigest()

        return cls(
            formula_id=formula_id,
            formula_name=formula_name,
            version=version,
            description=description,
            formula_expression=formula_expression,
            input_variables=input_variables,
            output_variables=output_variables,
            constants=constants or {},
            formula_hash=formula_hash,
            reference_standard=reference_standard,
        )


class TraceStep(BaseModel):
    """
    A single step in a calculation trace.

    Records inputs, outputs, formula used, and timing for
    each step of a multi-step calculation.
    """

    step_id: UUID = Field(default_factory=uuid4, description="Unique step identifier")
    step_number: int = Field(..., ge=1, description="Step sequence number")
    step_name: str = Field(..., description="Step name")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Step timestamp"
    )

    # Inputs and outputs
    inputs: Dict[str, Any] = Field(..., description="Step input values")
    outputs: Dict[str, Any] = Field(..., description="Step output values")

    # Hashes
    inputs_hash: str = Field(..., description="SHA-256 hash of inputs")
    outputs_hash: str = Field(..., description="SHA-256 hash of outputs")

    # Formula
    formula: FormulaVersion = Field(..., description="Formula used in this step")

    # Timing
    duration_ms: float = Field(0.0, ge=0, description="Step duration in ms")

    # Status
    status: str = Field(default="COMPLETED", description="Step status")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        frozen = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class ConstraintCheckResult(BaseModel):
    """
    Result of a constraint check during calculation.

    Records whether constraints were satisfied and the
    margin/slack values.
    """

    check_id: UUID = Field(default_factory=uuid4, description="Unique check identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Check timestamp"
    )

    # Constraint definition
    constraint_id: str = Field(..., description="Constraint identifier")
    constraint_name: str = Field(..., description="Human-readable constraint name")
    constraint_type: ConstraintType = Field(..., description="Type of constraint")
    constraint_expression: str = Field(..., description="Constraint expression")

    # Check result
    is_satisfied: bool = Field(..., description="Whether constraint is satisfied")
    actual_value: float = Field(..., description="Actual calculated value")
    limit_value: float = Field(..., description="Constraint limit value")
    margin: float = Field(..., description="Margin to limit (positive = within limit)")
    margin_pct: float = Field(..., description="Margin as percentage of limit")

    # Context
    step_id: Optional[str] = Field(None, description="Associated step ID if applicable")
    variables: Dict[str, float] = Field(
        default_factory=dict, description="Variable values used"
    )

    # Violation details
    violation_severity: Optional[str] = Field(
        None, description="Severity if violated (WARNING, ERROR, CRITICAL)"
    )
    remediation_action: Optional[str] = Field(
        None, description="Suggested remediation if violated"
    )

    class Config:
        frozen = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class TraceContext(BaseModel):
    """
    Active context for an in-progress calculation trace.

    Mutable during tracing, becomes immutable when converted
    to CalculationTrace.
    """

    trace_id: UUID = Field(default_factory=uuid4, description="Unique trace identifier")
    calculation_type: str = Field(..., description="Type of calculation being traced")
    status: TraceStatus = Field(default=TraceStatus.STARTED, description="Trace status")
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Trace start time"
    )

    # Tracking
    current_step: int = Field(0, ge=0, description="Current step number")
    steps: List[TraceStep] = Field(default_factory=list, description="Completed steps")
    constraint_checks: List[ConstraintCheckResult] = Field(
        default_factory=list, description="Constraint check results"
    )

    # Context
    correlation_id: Optional[str] = Field(
        None, description="Correlation ID for distributed tracing"
    )
    user_id: Optional[str] = Field(None, description="User who initiated calculation")
    asset_id: Optional[str] = Field(None, description="Asset being calculated")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class CalculationTrace(BaseModel):
    """
    Complete, immutable trace of a calculation.

    Created when a trace context is finalized, containing
    all steps, constraint checks, and summary information.
    """

    trace_id: UUID = Field(..., description="Unique trace identifier")
    calculation_type: str = Field(..., description="Type of calculation")
    status: TraceStatus = Field(..., description="Final trace status")

    # Timing
    start_time: datetime = Field(..., description="Calculation start time")
    end_time: datetime = Field(..., description="Calculation end time")
    total_duration_ms: float = Field(..., ge=0, description="Total duration in ms")

    # Steps
    steps: List[TraceStep] = Field(..., description="All calculation steps")
    total_steps: int = Field(..., ge=0, description="Total number of steps")

    # Constraint checks
    constraint_checks: List[ConstraintCheckResult] = Field(
        ..., description="All constraint checks"
    )
    total_constraints: int = Field(..., ge=0, description="Total constraints checked")
    constraints_satisfied: int = Field(..., ge=0, description="Constraints satisfied")
    constraints_violated: int = Field(..., ge=0, description="Constraints violated")

    # Input/Output summary
    initial_inputs_hash: str = Field(..., description="Hash of initial inputs")
    final_outputs_hash: str = Field(..., description="Hash of final outputs")

    # Formulas used
    formulas_used: List[str] = Field(
        ..., description="List of formula IDs used"
    )
    formula_versions: Dict[str, str] = Field(
        ..., description="Formula ID to version mapping"
    )

    # Context
    correlation_id: Optional[str] = Field(None, description="Correlation ID")
    user_id: Optional[str] = Field(None, description="User ID")
    asset_id: Optional[str] = Field(None, description="Asset ID")

    # Hash for integrity
    trace_hash: str = Field(..., description="SHA-256 hash of complete trace")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        frozen = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class AuditableTrace(BaseModel):
    """
    Trace formatted for external auditor review.

    Contains human-readable summaries and all verification
    information needed for audit.
    """

    trace_id: str = Field(..., description="Trace identifier")
    export_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Export timestamp"
    )
    export_version: str = Field(default="1.0.0", description="Export format version")

    # Calculation summary
    calculation_type: str = Field(..., description="Type of calculation")
    calculation_status: str = Field(..., description="Final status")
    calculation_time_range: Dict[str, str] = Field(
        ..., description="Start and end timestamps"
    )
    total_duration_ms: float = Field(..., description="Total duration")

    # Step-by-step breakdown
    steps_summary: List[Dict[str, Any]] = Field(
        ..., description="Simplified step summaries"
    )

    # Constraint summary
    constraints_summary: Dict[str, Any] = Field(
        ..., description="Constraint check summary"
    )

    # Formula documentation
    formulas_documentation: List[Dict[str, Any]] = Field(
        ..., description="Documentation for all formulas used"
    )

    # Verification information
    verification: Dict[str, str] = Field(
        ..., description="Hash verification information"
    )

    # Provenance chain
    provenance_chain: List[Dict[str, str]] = Field(
        ..., description="Provenance record chain"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class CalculationTracer:
    """
    Tracer for step-by-step calculation audit trails.

    Provides comprehensive tracing of multi-step calculations
    with formula version tracking and constraint checking.

    Attributes:
        provenance_tracker: Optional provenance tracker for integration

    Example:
        >>> tracer = CalculationTracer()
        >>> ctx = tracer.start_trace("enthalpy_calculation")
        >>> tracer.record_step(ctx, "input_validation", inputs, outputs, formula)
        >>> tracer.record_constraint_check(ctx, constraint, result)
        >>> trace = tracer.end_trace(ctx)
    """

    def __init__(
        self,
        provenance_tracker: Optional[Any] = None,
    ):
        """
        Initialize calculation tracer.

        Args:
            provenance_tracker: Optional ProvenanceTracker for integration
        """
        self.provenance_tracker = provenance_tracker

        # In-memory storage of active traces
        self._active_traces: Dict[str, TraceContext] = {}
        self._completed_traces: Dict[str, CalculationTrace] = {}

        # Formula registry
        self._formula_registry: Dict[str, FormulaVersion] = {}

        logger.info("CalculationTracer initialized")

    def register_formula(self, formula: FormulaVersion) -> None:
        """
        Register a formula version for use in traces.

        Args:
            formula: FormulaVersion to register
        """
        self._formula_registry[formula.formula_id] = formula
        logger.debug(f"Formula registered: {formula.formula_id} v{formula.version}")

    def get_formula(self, formula_id: str) -> Optional[FormulaVersion]:
        """Get registered formula by ID."""
        return self._formula_registry.get(formula_id)

    def start_trace(
        self,
        calculation_type: str,
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceContext:
        """
        Start a new calculation trace.

        Args:
            calculation_type: Type of calculation (e.g., "steam_balance")
            correlation_id: Optional correlation ID for tracing
            user_id: Optional user who initiated calculation
            asset_id: Optional asset being calculated
            metadata: Optional additional metadata

        Returns:
            TraceContext for recording steps

        Example:
            >>> ctx = tracer.start_trace("enthalpy_calculation")
        """
        ctx = TraceContext(
            calculation_type=calculation_type,
            correlation_id=correlation_id,
            user_id=user_id,
            asset_id=asset_id,
            metadata=metadata or {},
        )

        # Store active trace
        trace_id = str(ctx.trace_id)
        self._active_traces[trace_id] = ctx

        logger.info(
            f"Trace started: {calculation_type}",
            extra={
                "trace_id": trace_id,
                "correlation_id": correlation_id,
            }
        )

        return ctx

    def record_step(
        self,
        context: TraceContext,
        step_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        formula: FormulaVersion,
        duration_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceStep:
        """
        Record a calculation step in the trace.

        Args:
            context: Active trace context
            step_name: Name of this step
            inputs: Input values for this step
            outputs: Output values from this step
            formula: Formula used in this step
            duration_ms: Step duration in milliseconds
            metadata: Optional additional metadata

        Returns:
            Created TraceStep

        Example:
            >>> step = tracer.record_step(
            ...     ctx,
            ...     "compute_enthalpy",
            ...     {"temp": 450, "pressure": 150},
            ...     {"enthalpy": 1200.5},
            ...     formula
            ... )
        """
        # Compute hashes
        inputs_hash = hashlib.sha256(
            json.dumps(inputs, sort_keys=True, default=str).encode()
        ).hexdigest()
        outputs_hash = hashlib.sha256(
            json.dumps(outputs, sort_keys=True, default=str).encode()
        ).hexdigest()

        # Increment step number
        context.current_step += 1

        step = TraceStep(
            step_number=context.current_step,
            step_name=step_name,
            inputs=inputs,
            outputs=outputs,
            inputs_hash=inputs_hash,
            outputs_hash=outputs_hash,
            formula=formula,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        # Add to context
        context.steps.append(step)
        context.status = TraceStatus.IN_PROGRESS

        # Record in provenance tracker if available
        if self.provenance_tracker:
            self.provenance_tracker.create_provenance_record(
                calculation_id=f"{context.trace_id}_{step.step_id}",
                input_hash=inputs_hash,
                output_hash=outputs_hash,
                formula_version=formula.formula_id,
                correlation_id=context.correlation_id,
            )

        logger.debug(
            f"Step recorded: {step_name}",
            extra={
                "trace_id": str(context.trace_id),
                "step_number": step.step_number,
            }
        )

        return step

    def record_constraint_check(
        self,
        context: TraceContext,
        constraint_id: str,
        constraint_name: str,
        constraint_type: ConstraintType,
        constraint_expression: str,
        actual_value: float,
        limit_value: float,
        variables: Optional[Dict[str, float]] = None,
        step_id: Optional[str] = None,
    ) -> ConstraintCheckResult:
        """
        Record a constraint check in the trace.

        Args:
            context: Active trace context
            constraint_id: Constraint identifier
            constraint_name: Human-readable name
            constraint_type: Type of constraint
            constraint_expression: Constraint expression
            actual_value: Actual calculated value
            limit_value: Constraint limit
            variables: Variable values used
            step_id: Associated step ID if applicable

        Returns:
            ConstraintCheckResult

        Example:
            >>> result = tracer.record_constraint_check(
            ...     ctx,
            ...     "MAX_TEMP",
            ...     "Maximum temperature limit",
            ...     ConstraintType.BOUND,
            ...     "temperature <= 500",
            ...     450.0,
            ...     500.0
            ... )
        """
        # Calculate margin
        if constraint_type in [ConstraintType.BOUND, ConstraintType.INEQUALITY]:
            margin = limit_value - actual_value
        else:
            margin = abs(limit_value - actual_value)

        margin_pct = (margin / limit_value * 100) if limit_value != 0 else 0
        is_satisfied = margin >= 0

        # Determine severity if violated
        violation_severity = None
        if not is_satisfied:
            violation_pct = abs(margin_pct)
            if violation_pct > 20:
                violation_severity = "CRITICAL"
            elif violation_pct > 10:
                violation_severity = "ERROR"
            else:
                violation_severity = "WARNING"

        result = ConstraintCheckResult(
            constraint_id=constraint_id,
            constraint_name=constraint_name,
            constraint_type=constraint_type,
            constraint_expression=constraint_expression,
            is_satisfied=is_satisfied,
            actual_value=actual_value,
            limit_value=limit_value,
            margin=margin,
            margin_pct=margin_pct,
            step_id=step_id,
            variables=variables or {},
            violation_severity=violation_severity,
        )

        context.constraint_checks.append(result)

        log_level = logging.DEBUG if is_satisfied else logging.WARNING
        logger.log(
            log_level,
            f"Constraint check: {constraint_name} - {'PASS' if is_satisfied else 'FAIL'}",
            extra={
                "trace_id": str(context.trace_id),
                "margin_pct": margin_pct,
            }
        )

        return result

    def end_trace(
        self,
        context: TraceContext,
        status: TraceStatus = TraceStatus.COMPLETED,
        error_message: Optional[str] = None,
    ) -> CalculationTrace:
        """
        End an active trace and create immutable CalculationTrace.

        Args:
            context: Active trace context to finalize
            status: Final status (COMPLETED or FAILED)
            error_message: Error message if failed

        Returns:
            Immutable CalculationTrace

        Example:
            >>> trace = tracer.end_trace(ctx)
            >>> print(trace.total_duration_ms)
        """
        end_time = datetime.now(timezone.utc)
        total_duration = (end_time - context.start_time).total_seconds() * 1000

        # Count constraints
        satisfied = sum(1 for c in context.constraint_checks if c.is_satisfied)
        violated = len(context.constraint_checks) - satisfied

        # Get initial inputs and final outputs hashes
        initial_hash = context.steps[0].inputs_hash if context.steps else ""
        final_hash = context.steps[-1].outputs_hash if context.steps else ""

        # Collect formulas used
        formulas_used = list(set(s.formula.formula_id for s in context.steps))
        formula_versions = {s.formula.formula_id: s.formula.version for s in context.steps}

        # Calculate trace hash
        trace_data = {
            "trace_id": str(context.trace_id),
            "calculation_type": context.calculation_type,
            "steps": [s.dict() for s in context.steps],
            "constraint_checks": [c.dict() for c in context.constraint_checks],
        }
        trace_hash = hashlib.sha256(
            json.dumps(trace_data, sort_keys=True, default=str).encode()
        ).hexdigest()

        trace = CalculationTrace(
            trace_id=context.trace_id,
            calculation_type=context.calculation_type,
            status=status,
            start_time=context.start_time,
            end_time=end_time,
            total_duration_ms=total_duration,
            steps=context.steps,
            total_steps=len(context.steps),
            constraint_checks=context.constraint_checks,
            total_constraints=len(context.constraint_checks),
            constraints_satisfied=satisfied,
            constraints_violated=violated,
            initial_inputs_hash=initial_hash,
            final_outputs_hash=final_hash,
            formulas_used=formulas_used,
            formula_versions=formula_versions,
            correlation_id=context.correlation_id,
            user_id=context.user_id,
            asset_id=context.asset_id,
            trace_hash=trace_hash,
            metadata=context.metadata,
        )

        # Move from active to completed
        trace_id = str(context.trace_id)
        if trace_id in self._active_traces:
            del self._active_traces[trace_id]
        self._completed_traces[trace_id] = trace

        logger.info(
            f"Trace completed: {context.calculation_type}",
            extra={
                "trace_id": trace_id,
                "total_steps": len(context.steps),
                "duration_ms": total_duration,
                "constraints_violated": violated,
            }
        )

        return trace

    def export_trace_for_audit(
        self,
        trace_id: Union[str, UUID],
    ) -> AuditableTrace:
        """
        Export a trace in auditor-friendly format.

        Args:
            trace_id: Trace identifier

        Returns:
            AuditableTrace for auditor review

        Raises:
            ValueError: If trace not found

        Example:
            >>> auditable = tracer.export_trace_for_audit(trace.trace_id)
            >>> print(auditable.steps_summary)
        """
        trace_id_str = str(trace_id)
        trace = self._completed_traces.get(trace_id_str)

        if not trace:
            raise ValueError(f"Trace not found: {trace_id_str}")

        # Build step summaries
        steps_summary = []
        for step in trace.steps:
            steps_summary.append({
                "step_number": step.step_number,
                "step_name": step.step_name,
                "timestamp": step.timestamp.isoformat(),
                "formula_id": step.formula.formula_id,
                "formula_name": step.formula.formula_name,
                "inputs_summary": {k: str(v)[:50] for k, v in list(step.inputs.items())[:5]},
                "outputs_summary": {k: str(v)[:50] for k, v in list(step.outputs.items())[:5]},
                "duration_ms": step.duration_ms,
                "status": step.status,
            })

        # Build constraint summary
        constraints_summary = {
            "total_checked": trace.total_constraints,
            "satisfied": trace.constraints_satisfied,
            "violated": trace.constraints_violated,
            "pass_rate_pct": (
                trace.constraints_satisfied / trace.total_constraints * 100
                if trace.total_constraints > 0 else 100
            ),
            "violations": [
                {
                    "constraint_id": c.constraint_id,
                    "constraint_name": c.constraint_name,
                    "actual": c.actual_value,
                    "limit": c.limit_value,
                    "margin_pct": c.margin_pct,
                    "severity": c.violation_severity,
                }
                for c in trace.constraint_checks if not c.is_satisfied
            ],
        }

        # Build formula documentation
        formulas_documentation = []
        for step in trace.steps:
            if step.formula.formula_id not in [f["formula_id"] for f in formulas_documentation]:
                formulas_documentation.append({
                    "formula_id": step.formula.formula_id,
                    "formula_name": step.formula.formula_name,
                    "version": step.formula.version,
                    "expression": step.formula.formula_expression,
                    "inputs": step.formula.input_variables,
                    "outputs": step.formula.output_variables,
                    "reference_standard": step.formula.reference_standard,
                    "formula_hash": step.formula.formula_hash,
                })

        # Build verification info
        verification = {
            "trace_hash": trace.trace_hash,
            "initial_inputs_hash": trace.initial_inputs_hash,
            "final_outputs_hash": trace.final_outputs_hash,
            "hash_algorithm": "SHA-256",
        }

        # Build provenance chain (simplified)
        provenance_chain = []
        for step in trace.steps:
            provenance_chain.append({
                "step": str(step.step_number),
                "inputs_hash": step.inputs_hash[:16] + "...",
                "outputs_hash": step.outputs_hash[:16] + "...",
                "formula_hash": step.formula.formula_hash[:16] + "...",
            })

        auditable = AuditableTrace(
            trace_id=trace_id_str,
            calculation_type=trace.calculation_type,
            calculation_status=trace.status.value,
            calculation_time_range={
                "start": trace.start_time.isoformat(),
                "end": trace.end_time.isoformat(),
            },
            total_duration_ms=trace.total_duration_ms,
            steps_summary=steps_summary,
            constraints_summary=constraints_summary,
            formulas_documentation=formulas_documentation,
            verification=verification,
            provenance_chain=provenance_chain,
        )

        logger.info(
            f"Trace exported for audit: {trace_id_str}",
            extra={"steps": len(steps_summary)}
        )

        return auditable

    def get_trace(self, trace_id: Union[str, UUID]) -> Optional[CalculationTrace]:
        """Get a completed trace by ID."""
        return self._completed_traces.get(str(trace_id))

    def get_active_trace(self, trace_id: Union[str, UUID]) -> Optional[TraceContext]:
        """Get an active trace context by ID."""
        return self._active_traces.get(str(trace_id))

    def list_traces(
        self,
        calculation_type: Optional[str] = None,
        status: Optional[TraceStatus] = None,
        limit: int = 100,
    ) -> List[CalculationTrace]:
        """
        List completed traces with optional filters.

        Args:
            calculation_type: Filter by calculation type
            status: Filter by status
            limit: Maximum traces to return

        Returns:
            List of matching CalculationTraces
        """
        traces = list(self._completed_traces.values())

        if calculation_type:
            traces = [t for t in traces if t.calculation_type == calculation_type]

        if status:
            traces = [t for t in traces if t.status == status]

        # Sort by start time descending
        traces.sort(key=lambda t: t.start_time, reverse=True)

        return traces[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get tracer statistics.

        Returns:
            Dictionary of statistics
        """
        completed = list(self._completed_traces.values())

        # Count by calculation type
        by_type: Dict[str, int] = {}
        for trace in completed:
            by_type[trace.calculation_type] = by_type.get(trace.calculation_type, 0) + 1

        # Count by status
        by_status: Dict[str, int] = {}
        for trace in completed:
            by_status[trace.status.value] = by_status.get(trace.status.value, 0) + 1

        # Average duration
        durations = [t.total_duration_ms for t in completed]
        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "active_traces": len(self._active_traces),
            "completed_traces": len(self._completed_traces),
            "registered_formulas": len(self._formula_registry),
            "traces_by_type": by_type,
            "traces_by_status": by_status,
            "average_duration_ms": avg_duration,
        }

    def clear_completed(self) -> int:
        """
        Clear all completed traces from memory.

        Returns:
            Number of traces cleared
        """
        count = len(self._completed_traces)
        self._completed_traces.clear()
        logger.info(f"Cleared {count} completed traces")
        return count
