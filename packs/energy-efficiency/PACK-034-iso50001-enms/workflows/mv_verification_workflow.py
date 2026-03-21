# -*- coding: utf-8 -*-
"""
M&V Verification Workflow - IPMVP Verification
===================================

3-phase workflow for measurement and verification of energy savings
per IPMVP within PACK-034 ISO 50001 Energy Management System Pack.

Phases:
    1. BaselineComparison       -- Adjust baseline for current conditions, compare
    2. SavingsQuantification    -- Calculate gross/net savings, adjust for non-routine
    3. UncertaintyAnalysis      -- Calculate uncertainty, confidence, significance

The workflow follows GreenLang zero-hallucination principles: baseline
adjustments use regression model predictions, savings quantification
uses deterministic arithmetic (adjusted baseline minus reporting period),
and uncertainty calculations use standard statistical propagation formulas.
SHA-256 provenance hashes guarantee auditability.

Schedule: quarterly / annually
Estimated duration: 35 minutes

Author: GreenLang Team
Version: 34.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime
from decimal import Decimal
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


class VerificationPhase(str, Enum):
    """Phases of the M&V verification workflow."""

    BASELINE_COMPARISON = "baseline_comparison"
    SAVINGS_QUANTIFICATION = "savings_quantification"
    UNCERTAINTY_ANALYSIS = "uncertainty_analysis"


class IPMVPOption(str, Enum):
    """IPMVP verification options."""

    OPTION_A = "option_a"  # Retrofit Isolation: Key Parameter Measurement
    OPTION_B = "option_b"  # Retrofit Isolation: All Parameter Measurement
    OPTION_C = "option_c"  # Whole Facility
    OPTION_D = "option_d"  # Calibrated Simulation


class SignificanceLevel(str, Enum):
    """Statistical significance classification."""

    HIGHLY_SIGNIFICANT = "highly_significant"  # p < 0.01
    SIGNIFICANT = "significant"                 # p < 0.05
    MARGINALLY_SIGNIFICANT = "marginally_significant"  # p < 0.10
    NOT_SIGNIFICANT = "not_significant"         # p >= 0.10


# =============================================================================
# IPMVP REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# IPMVP Option characteristics
IPMVP_OPTIONS: Dict[str, Dict[str, Any]] = {
    "option_a": {
        "name": "Retrofit Isolation: Key Parameter Measurement",
        "description": "Savings determined by measuring key parameters",
        "typical_uncertainty_pct": 15.0,
        "min_data_points": 12,
        "cost_complexity": "low",
    },
    "option_b": {
        "name": "Retrofit Isolation: All Parameter Measurement",
        "description": "Savings determined by measuring all parameters",
        "typical_uncertainty_pct": 10.0,
        "min_data_points": 12,
        "cost_complexity": "medium",
    },
    "option_c": {
        "name": "Whole Facility",
        "description": "Savings determined by whole-facility metering",
        "typical_uncertainty_pct": 20.0,
        "min_data_points": 12,
        "cost_complexity": "low",
    },
    "option_d": {
        "name": "Calibrated Simulation",
        "description": "Savings determined by calibrated energy model",
        "typical_uncertainty_pct": 25.0,
        "min_data_points": 12,
        "cost_complexity": "high",
    },
}

# t-values for common confidence levels (approximation for large samples)
T_VALUES: Dict[float, float] = {
    0.80: 1.282,
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}

# Default confidence level
DEFAULT_CONFIDENCE: float = 0.90

# ASHRAE Guideline 14 thresholds for model validity
ASHRAE_CV_RMSE_MONTHLY: float = 15.0  # percent
ASHRAE_CV_RMSE_HOURLY: float = 30.0   # percent
ASHRAE_NMBE: float = 5.0              # percent


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_ms: float = Field(default=0.0, description="Phase duration in milliseconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class BaselineAdjustment(BaseModel):
    """Baseline adjustment record for current conditions."""

    adjustment_id: str = Field(default_factory=lambda: f"adj-{uuid.uuid4().hex[:8]}")
    period: str = Field(default="", description="Reporting period")
    original_baseline_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    adjusted_baseline_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    adjustment_kwh: Decimal = Field(default=Decimal("0"), description="Adjustment amount")
    adjustment_reason: str = Field(default="", description="Reason for adjustment")
    variable_values: Dict[str, float] = Field(default_factory=dict)


class NonRoutineAdjustment(BaseModel):
    """Non-routine adjustment for events outside normal operations."""

    event_id: str = Field(default_factory=lambda: f"nra-{uuid.uuid4().hex[:8]}")
    description: str = Field(default="", description="Event description")
    impact_kwh: Decimal = Field(default=Decimal("0"), description="Energy impact of event")
    period: str = Field(default="", description="Period of event")
    adjustment_type: str = Field(default="addition", description="addition|subtraction")


class SavingsBreakdown(BaseModel):
    """Detailed breakdown of savings quantification."""

    total_adjusted_baseline_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    total_reporting_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    gross_savings_kwh: Decimal = Field(default=Decimal("0"))
    non_routine_adjustments_kwh: Decimal = Field(default=Decimal("0"))
    net_savings_kwh: Decimal = Field(default=Decimal("0"))
    net_savings_pct: Decimal = Field(default=Decimal("0"))


class UncertaintyBreakdown(BaseModel):
    """Detailed uncertainty analysis results."""

    model_uncertainty_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    measurement_uncertainty_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    sampling_uncertainty_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    total_uncertainty_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    total_uncertainty_pct: Decimal = Field(default=Decimal("0"), ge=0)
    fractional_savings_uncertainty: Decimal = Field(default=Decimal("0"), ge=0)


class MVVerificationInput(BaseModel):
    """Input data model for MVVerificationWorkflow."""

    enms_id: str = Field(default="", description="EnMS program identifier")
    baseline_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Baseline period data: [{period, kwh, variables: {}}]",
    )
    reporting_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Reporting period data: [{period, kwh, variables: {}}]",
    )
    model_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Baseline model parameters: {intercept, coefficients, r_squared, rmse}",
    )
    non_routine_events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Non-routine events: [{description, impact_kwh, period, type}]",
    )
    verification_standard: str = Field(
        default="option_c",
        description="IPMVP option: option_a|option_b|option_c|option_d",
    )
    confidence_level: float = Field(
        default=0.90, ge=0.80, le=0.99,
        description="Confidence level for uncertainty analysis",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("verification_standard")
    @classmethod
    def validate_standard(cls, v: str) -> str:
        """Ensure verification standard is valid."""
        valid = list(IPMVP_OPTIONS.keys())
        if v not in valid:
            raise ValueError(f"Invalid verification_standard '{v}'. Valid: {valid}")
        return v


class MVVerificationResult(BaseModel):
    """Complete result from M&V verification workflow."""

    verification_id: str = Field(..., description="Unique verification ID")
    enms_id: str = Field(default="", description="EnMS program identifier")
    adjusted_baseline_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    reporting_period_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    gross_savings_kwh: Decimal = Field(default=Decimal("0"))
    net_savings_kwh: Decimal = Field(default=Decimal("0"))
    savings_pct: Decimal = Field(default=Decimal("0"))
    uncertainty_pct: Decimal = Field(default=Decimal("0"), ge=0)
    confidence_level: Decimal = Field(default=Decimal("0.90"))
    is_significant: bool = Field(default=False, description="True if savings > uncertainty")
    significance_level: str = Field(default="not_significant")
    ipmvp_option: str = Field(default="option_c", description="IPMVP option used")
    baseline_adjustments: List[BaselineAdjustment] = Field(default_factory=list)
    non_routine_adjustments: List[NonRoutineAdjustment] = Field(default_factory=list)
    savings_breakdown: SavingsBreakdown = Field(default_factory=SavingsBreakdown)
    uncertainty_breakdown: UncertaintyBreakdown = Field(default_factory=UncertaintyBreakdown)
    phases_completed: List[str] = Field(default_factory=list)
    execution_time_ms: float = Field(default=0.0)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class MVVerificationWorkflow:
    """
    3-phase M&V verification workflow per IPMVP.

    Adjusts baseline for current conditions using regression models,
    quantifies gross and net savings with non-routine adjustments,
    and performs uncertainty analysis to determine statistical significance.

    Zero-hallucination: baseline adjustments use deterministic regression
    predictions, savings are calculated by subtraction of measured values,
    and uncertainty uses standard error propagation formulas. No LLM calls
    in the numeric computation path.

    Attributes:
        verification_id: Unique verification execution identifier.
        _adjustments: Baseline adjustment records.
        _non_routine: Non-routine adjustment records.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = MVVerificationWorkflow()
        >>> inp = MVVerificationInput(
        ...     enms_id="enms-001",
        ...     baseline_data=[{"period": "2025-01", "kwh": 50000}],
        ...     reporting_data=[{"period": "2026-01", "kwh": 45000}],
        ...     model_config={"intercept": 10000, "coefficients": {"hdd": 25.0}, "rmse": 2000},
        ... )
        >>> result = wf.execute(inp)
        >>> assert result.gross_savings_kwh >= 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MVVerificationWorkflow."""
        self.verification_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._adjustments: List[BaselineAdjustment] = []
        self._non_routine: List[NonRoutineAdjustment] = []
        self._adjusted_baseline_total: Decimal = Decimal("0")
        self._reporting_total: Decimal = Decimal("0")
        self._gross_savings: Decimal = Decimal("0")
        self._net_savings: Decimal = Decimal("0")
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def execute(self, input_data: MVVerificationInput) -> MVVerificationResult:
        """
        Execute the 3-phase M&V verification workflow.

        Args:
            input_data: Validated M&V verification input.

        Returns:
            MVVerificationResult with savings, uncertainty, and significance.

        Raises:
            ValueError: If input validation fails.
        """
        t_start = time.perf_counter()
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting M&V verification workflow %s enms=%s option=%s",
            self.verification_id, input_data.enms_id, input_data.verification_standard,
        )

        self._phase_results = []
        self._adjustments = []
        self._non_routine = []
        self._adjusted_baseline_total = Decimal("0")
        self._reporting_total = Decimal("0")
        self._gross_savings = Decimal("0")
        self._net_savings = Decimal("0")

        try:
            phase1 = self._phase_baseline_comparison(input_data)
            self._phase_results.append(phase1)

            phase2 = self._phase_savings_quantification(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_uncertainty_analysis(input_data)
            self._phase_results.append(phase3)

        except Exception as exc:
            self.logger.error(
                "M&V verification workflow failed: %s", exc, exc_info=True,
            )
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Extract uncertainty from phase 3
        uncertainty_pct = Decimal("0")
        uncertainty_breakdown = UncertaintyBreakdown()
        is_significant = False
        significance_level = "not_significant"

        for pr in self._phase_results:
            if pr.phase_name == VerificationPhase.UNCERTAINTY_ANALYSIS.value:
                uncertainty_pct = Decimal(str(pr.outputs.get("uncertainty_pct", 0)))
                is_significant = pr.outputs.get("is_significant", False)
                significance_level = pr.outputs.get("significance_level", "not_significant")
                uncertainty_breakdown = UncertaintyBreakdown(
                    model_uncertainty_kwh=Decimal(str(pr.outputs.get("model_uncertainty_kwh", 0))),
                    measurement_uncertainty_kwh=Decimal(str(pr.outputs.get("measurement_uncertainty_kwh", 0))),
                    sampling_uncertainty_kwh=Decimal(str(pr.outputs.get("sampling_uncertainty_kwh", 0))),
                    total_uncertainty_kwh=Decimal(str(pr.outputs.get("total_uncertainty_kwh", 0))),
                    total_uncertainty_pct=uncertainty_pct,
                    fractional_savings_uncertainty=Decimal(str(pr.outputs.get("fractional_savings_uncertainty", 0))),
                )

        savings_pct = Decimal("0")
        if self._adjusted_baseline_total > 0:
            savings_pct = Decimal(str(round(
                float(self._net_savings) / float(self._adjusted_baseline_total) * 100.0, 2
            )))

        savings_breakdown = SavingsBreakdown(
            total_adjusted_baseline_kwh=self._adjusted_baseline_total,
            total_reporting_kwh=self._reporting_total,
            gross_savings_kwh=self._gross_savings,
            non_routine_adjustments_kwh=sum(
                nra.impact_kwh for nra in self._non_routine
            ),
            net_savings_kwh=self._net_savings,
            net_savings_pct=savings_pct,
        )

        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        result = MVVerificationResult(
            verification_id=self.verification_id,
            enms_id=input_data.enms_id,
            adjusted_baseline_kwh=self._adjusted_baseline_total,
            reporting_period_kwh=self._reporting_total,
            gross_savings_kwh=self._gross_savings,
            net_savings_kwh=self._net_savings,
            savings_pct=savings_pct,
            uncertainty_pct=uncertainty_pct,
            confidence_level=Decimal(str(input_data.confidence_level)),
            is_significant=is_significant,
            significance_level=significance_level,
            ipmvp_option=input_data.verification_standard,
            baseline_adjustments=self._adjustments,
            non_routine_adjustments=self._non_routine,
            savings_breakdown=savings_breakdown,
            uncertainty_breakdown=uncertainty_breakdown,
            phases_completed=completed_phases,
            execution_time_ms=round(elapsed_ms, 2),
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "M&V verification workflow %s completed in %.0fms "
            "gross=%.0f net=%.0f savings=%.1f%% uncertainty=%.1f%% significant=%s",
            self.verification_id, elapsed_ms, float(self._gross_savings),
            float(self._net_savings), float(savings_pct),
            float(uncertainty_pct), is_significant,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Baseline Comparison
    # -------------------------------------------------------------------------

    def _phase_baseline_comparison(
        self, input_data: MVVerificationInput
    ) -> PhaseResult:
        """Adjust baseline for current conditions, compare against reporting period."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        model = input_data.model_config
        intercept = float(model.get("intercept", 0))
        coefficients = model.get("coefficients", {})

        # Adjust baseline for each reporting period using regression model
        for rp_data in input_data.reporting_data:
            period = rp_data.get("period", "")
            actual_kwh = float(rp_data.get("kwh", 0))
            variables = rp_data.get("variables", {})

            # Calculate adjusted baseline using regression model
            adjusted = intercept
            for var_name, coeff in coefficients.items():
                var_value = float(variables.get(var_name, 0))
                adjusted += float(coeff) * var_value

            # Find original baseline for this period
            original = 0.0
            for bl_data in input_data.baseline_data:
                if bl_data.get("period") == period:
                    original = float(bl_data.get("kwh", 0))
                    break

            if original == 0.0:
                # Use mean of all baseline data as fallback
                original = sum(
                    float(bl.get("kwh", 0)) for bl in input_data.baseline_data
                ) / max(len(input_data.baseline_data), 1)

            adjustment = BaselineAdjustment(
                period=period,
                original_baseline_kwh=Decimal(str(round(original, 2))),
                adjusted_baseline_kwh=Decimal(str(round(adjusted, 2))),
                adjustment_kwh=Decimal(str(round(adjusted - original, 2))),
                adjustment_reason="Regression model adjustment for current conditions",
                variable_values={k: float(v) for k, v in variables.items()},
            )
            self._adjustments.append(adjustment)

        # Calculate totals
        self._adjusted_baseline_total = sum(
            a.adjusted_baseline_kwh for a in self._adjustments
        )
        self._reporting_total = Decimal(str(round(
            sum(float(rp.get("kwh", 0)) for rp in input_data.reporting_data), 2
        )))

        if not coefficients:
            warnings.append("No regression coefficients provided; using raw baseline values")

        outputs["periods_adjusted"] = len(self._adjustments)
        outputs["adjusted_baseline_total_kwh"] = str(self._adjusted_baseline_total)
        outputs["reporting_total_kwh"] = str(self._reporting_total)
        outputs["model_variables"] = list(coefficients.keys())
        outputs["has_regression_model"] = bool(coefficients)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 BaselineComparison: %d periods, adjusted=%.0f, reporting=%.0f",
            len(self._adjustments), float(self._adjusted_baseline_total),
            float(self._reporting_total),
        )
        return PhaseResult(
            phase_name=VerificationPhase.BASELINE_COMPARISON.value, phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Savings Quantification
    # -------------------------------------------------------------------------

    def _phase_savings_quantification(
        self, input_data: MVVerificationInput
    ) -> PhaseResult:
        """Calculate gross and net savings, adjust for non-routine events."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Gross savings = adjusted baseline - reporting period
        self._gross_savings = self._adjusted_baseline_total - self._reporting_total

        # Process non-routine adjustments
        total_nra = Decimal("0")
        for event in input_data.non_routine_events:
            nra = NonRoutineAdjustment(
                description=event.get("description", ""),
                impact_kwh=Decimal(str(event.get("impact_kwh", 0))),
                period=event.get("period", ""),
                adjustment_type=event.get("type", "addition"),
            )
            self._non_routine.append(nra)

            if nra.adjustment_type == "addition":
                total_nra += nra.impact_kwh
            else:
                total_nra -= nra.impact_kwh

        # Net savings = gross savings - non-routine adjustments
        self._net_savings = self._gross_savings - total_nra

        if self._gross_savings < 0:
            warnings.append(
                "Gross savings are negative; energy consumption increased vs baseline"
            )

        outputs["gross_savings_kwh"] = str(self._gross_savings)
        outputs["non_routine_adjustments_kwh"] = str(total_nra)
        outputs["non_routine_events_count"] = len(self._non_routine)
        outputs["net_savings_kwh"] = str(self._net_savings)
        outputs["net_savings_pct"] = str(
            round(
                float(self._net_savings) / float(self._adjusted_baseline_total) * 100.0, 2
            ) if self._adjusted_baseline_total > 0 else 0
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 SavingsQuantification: gross=%.0f, NRA=%.0f, net=%.0f",
            float(self._gross_savings), float(total_nra), float(self._net_savings),
        )
        return PhaseResult(
            phase_name=VerificationPhase.SAVINGS_QUANTIFICATION.value, phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Uncertainty Analysis
    # -------------------------------------------------------------------------

    def _phase_uncertainty_analysis(
        self, input_data: MVVerificationInput
    ) -> PhaseResult:
        """Calculate uncertainty, determine confidence level, validate significance."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        model = input_data.model_config
        rmse = float(model.get("rmse", 0))
        r_squared = float(model.get("r_squared", 0))
        n_baseline = len(input_data.baseline_data)
        n_reporting = len(input_data.reporting_data)
        n_total = n_baseline + n_reporting

        # Get t-value for confidence level
        t_value = T_VALUES.get(input_data.confidence_level, 1.645)

        # 1. Model uncertainty (from regression RMSE)
        model_uncertainty = rmse * math.sqrt(n_reporting) if rmse > 0 else 0.0

        # 2. Measurement uncertainty (assumed 2% of total metered consumption)
        measurement_uncertainty_pct = 2.0
        measurement_uncertainty = (
            float(self._reporting_total) * measurement_uncertainty_pct / 100.0
        )

        # 3. Sampling uncertainty (if sub-metering used)
        ipmvp_option = input_data.verification_standard
        option_data = IPMVP_OPTIONS.get(ipmvp_option, IPMVP_OPTIONS["option_c"])
        typical_uncertainty = option_data["typical_uncertainty_pct"]

        sampling_uncertainty = 0.0
        if ipmvp_option in ("option_a", "option_b"):
            # Sampling uncertainty for retrofit isolation
            sampling_uncertainty = (
                float(self._adjusted_baseline_total) * typical_uncertainty / 100.0
                / math.sqrt(max(n_reporting, 1))
            )

        # Total uncertainty (root sum of squares)
        total_uncertainty = math.sqrt(
            model_uncertainty ** 2
            + measurement_uncertainty ** 2
            + sampling_uncertainty ** 2
        )

        # Uncertainty as percentage of adjusted baseline
        uncertainty_pct = (
            total_uncertainty / float(self._adjusted_baseline_total) * 100.0
            if float(self._adjusted_baseline_total) > 0 else 0.0
        )

        # Fractional savings uncertainty
        fsu = (
            total_uncertainty / abs(float(self._net_savings)) * 100.0
            if abs(float(self._net_savings)) > 0 else 999.0
        )

        # Significance test: savings significant if |savings| > t * total_uncertainty
        threshold = t_value * total_uncertainty
        is_significant = abs(float(self._net_savings)) > threshold

        # Classify significance level
        if abs(float(self._net_savings)) > 2.576 * total_uncertainty:
            significance = SignificanceLevel.HIGHLY_SIGNIFICANT
        elif abs(float(self._net_savings)) > 1.960 * total_uncertainty:
            significance = SignificanceLevel.SIGNIFICANT
        elif abs(float(self._net_savings)) > 1.645 * total_uncertainty:
            significance = SignificanceLevel.MARGINALLY_SIGNIFICANT
        else:
            significance = SignificanceLevel.NOT_SIGNIFICANT

        outputs["model_uncertainty_kwh"] = round(model_uncertainty, 2)
        outputs["measurement_uncertainty_kwh"] = round(measurement_uncertainty, 2)
        outputs["sampling_uncertainty_kwh"] = round(sampling_uncertainty, 2)
        outputs["total_uncertainty_kwh"] = round(total_uncertainty, 2)
        outputs["uncertainty_pct"] = round(uncertainty_pct, 2)
        outputs["fractional_savings_uncertainty"] = round(fsu, 2)
        outputs["confidence_level"] = input_data.confidence_level
        outputs["t_value"] = t_value
        outputs["is_significant"] = is_significant
        outputs["significance_level"] = significance.value
        outputs["ipmvp_option"] = ipmvp_option
        outputs["savings_range_lower"] = str(round(float(self._net_savings) - total_uncertainty, 2))
        outputs["savings_range_upper"] = str(round(float(self._net_savings) + total_uncertainty, 2))

        if not is_significant:
            warnings.append(
                f"Savings ({float(self._net_savings):.0f} kWh) are not statistically "
                f"significant at {input_data.confidence_level * 100:.0f}% confidence level "
                f"(threshold: {threshold:.0f} kWh)"
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 UncertaintyAnalysis: uncertainty=%.1f%%, significant=%s (%s), "
            "savings range=[%.0f, %.0f] kWh",
            uncertainty_pct, is_significant, significance.value,
            float(self._net_savings) - total_uncertainty,
            float(self._net_savings) + total_uncertainty,
        )
        return PhaseResult(
            phase_name=VerificationPhase.UNCERTAINTY_ANALYSIS.value, phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: MVVerificationResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
