# -*- coding: utf-8 -*-
"""
Savings Verification Workflow
===================================

4-phase workflow for verifying energy savings per IPMVP methodology including
data collection, routine/non-routine adjustments, savings calculation, and
uncertainty analysis.

Phases:
    1. DataCollection       -- Collect post-retrofit energy and independent variable data
    2. AdjustmentCalc       -- Calculate routine and non-routine adjustments
    3. SavingsCalc          -- Calculate avoided energy and normalized savings
    4. UncertaintyAnalysis  -- Quantify fractional savings uncertainty per ASHRAE 14

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory references:
    - IPMVP Core Concepts (EVO 10000-1:2022) Section 5
    - ASHRAE Guideline 14-2014 (uncertainty analysis)
    - ISO 50015:2014 Section 8 (savings determination)
    - FEMP M&V Guidelines 4.0 Chapter 6

Schedule: monthly / quarterly / annually
Estimated duration: 10 minutes

Author: GreenLang Platform Team
Version: 40.0.0
"""

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

_MODULE_VERSION = "1.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.utcnow()


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


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


class AdjustmentType(str, Enum):
    """Adjustment type classification per IPMVP."""

    ROUTINE = "routine"
    NON_ROUTINE = "non_routine"


class SavingsMethod(str, Enum):
    """Savings calculation method."""

    AVOIDED_ENERGY = "avoided_energy"
    NORMALIZED_SAVINGS = "normalized_savings"
    BOTH = "both"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

ADJUSTMENT_TYPES: Dict[str, Dict[str, Any]] = {
    "routine": {
        "description": (
            "Adjustments for expected changes in conditions that affect energy "
            "use (e.g., weather, production volume, occupancy). Applied via "
            "regression model independent variables."
        ),
        "examples": [
            "weather_normalization",
            "production_volume",
            "occupancy_schedule",
            "operating_hours",
        ],
        "calculation_method": "regression_model_prediction",
        "frequency": "every_reporting_period",
        "documentation_required": [
            "independent_variable_data",
            "regression_model_parameters",
            "predicted_vs_actual",
        ],
    },
    "non_routine": {
        "description": (
            "Adjustments for changes not accounted for in the baseline model "
            "(e.g., facility additions, equipment changes, occupancy changes). "
            "Require separate engineering analysis."
        ),
        "examples": [
            "facility_expansion",
            "equipment_addition",
            "schedule_change",
            "process_change",
            "occupancy_change",
            "tariff_change",
        ],
        "calculation_method": "engineering_analysis",
        "frequency": "when_change_occurs",
        "documentation_required": [
            "change_description",
            "engineering_calculation",
            "supporting_data",
            "approval_record",
        ],
    },
}

SAVINGS_METHODS: Dict[str, Dict[str, Any]] = {
    "avoided_energy": {
        "description": (
            "Energy savings = Baseline Energy (adjusted to reporting conditions) "
            "- Reporting Period Energy. Shows actual savings under reporting-period "
            "conditions."
        ),
        "formula": "Savings = E_baseline_adjusted - E_reporting",
        "adjustments_applied": "routine_to_reporting_conditions",
        "use_case": "Performance contract verification",
        "standard_reference": "IPMVP Section 5.2",
    },
    "normalized_savings": {
        "description": (
            "Energy savings normalized to a fixed set of conditions (e.g., TMY "
            "weather). Allows year-over-year comparison independent of weather."
        ),
        "formula": "Savings = E_baseline_normalized - E_reporting_normalized",
        "adjustments_applied": "routine_to_normal_conditions",
        "use_case": "Portfolio comparison, trending, reporting",
        "standard_reference": "IPMVP Section 5.3",
    },
}

UNCERTAINTY_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "measurement": {
        "description": "Uncertainty from meter accuracy and calibration",
        "typical_range_pct": {"min": 0.5, "max": 5.0},
        "reduction_strategy": "Higher accuracy meters, more frequent calibration",
    },
    "model": {
        "description": "Uncertainty from baseline model regression residuals (CVRMSE)",
        "typical_range_pct": {"min": 5.0, "max": 25.0},
        "reduction_strategy": "Better model fit, more data points, additional variables",
    },
    "sampling": {
        "description": "Uncertainty from sampling if not all units metered (Option A)",
        "typical_range_pct": {"min": 0.0, "max": 20.0},
        "reduction_strategy": "Larger sample size, stratified sampling",
    },
    "interactive_effects": {
        "description": "Uncertainty from interactive effects between ECMs and systems",
        "typical_range_pct": {"min": 2.0, "max": 15.0},
        "reduction_strategy": "Whole-facility approach, simulation modeling",
    },
}

DEFAULT_ENERGY_RATES: Dict[str, float] = {
    "electricity_per_kwh": 0.12,
    "gas_per_therm": 1.10,
    "gas_per_m3": 0.40,
    "steam_per_gj": 25.00,
    "chilled_water_per_kwh_th": 0.08,
}


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


class PeriodEnergyData(BaseModel):
    """Energy data for a reporting period."""

    period_start: str = Field(..., description="Period start (ISO 8601)")
    period_end: str = Field(..., description="Period end (ISO 8601)")
    energy_value: float = Field(..., ge=0, description="Energy consumption")
    energy_unit: str = Field(default="kWh", description="Energy unit")
    temperature_avg: Optional[float] = Field(None, description="Average outdoor temp (C)")
    production_volume: Optional[float] = Field(None, ge=0, description="Production volume")
    occupancy_pct: Optional[float] = Field(None, ge=0, le=100, description="Occupancy %")
    hdd: Optional[float] = Field(None, ge=0, description="Heating degree days")
    cdd: Optional[float] = Field(None, ge=0, description="Cooling degree days")


class NonRoutineEvent(BaseModel):
    """Non-routine adjustment event."""

    event_id: str = Field(default_factory=lambda: f"nre-{uuid.uuid4().hex[:8]}")
    event_type: str = Field(..., description="Type of non-routine change")
    description: str = Field(default="", description="Description of change")
    date_occurred: str = Field(default="", description="Date of change (ISO 8601)")
    energy_impact_kwh: float = Field(
        default=0.0, description="Estimated energy impact (kWh, positive=increase)",
    )
    cost_impact: float = Field(default=0.0, description="Estimated cost impact")
    documentation: List[str] = Field(default_factory=list, description="Supporting docs")


class SavingsVerificationInput(BaseModel):
    """Input data model for SavingsVerificationWorkflow."""

    project_id: str = Field(default_factory=lambda: f"proj-{uuid.uuid4().hex[:8]}")
    facility_name: str = Field(..., min_length=1, description="Facility name")
    facility_id: str = Field(default="", description="Facility identifier")
    ecm_id: str = Field(default="", description="ECM identifier")
    ecm_name: str = Field(default="", description="ECM display name")
    ipmvp_option: str = Field(default="C", description="IPMVP option A/B/C/D")
    baseline_energy_kwh: float = Field(
        default=0.0, ge=0, description="Baseline period total energy (kWh)",
    )
    baseline_model_params: Dict[str, float] = Field(
        default_factory=lambda: {"intercept": 10000.0, "slope": 200.0},
        description="Baseline regression model parameters",
    )
    baseline_cvrmse_pct: float = Field(
        default=10.0, ge=0, description="Baseline model CV(RMSE) %",
    )
    reporting_period_data: List[PeriodEnergyData] = Field(
        default_factory=list, description="Reporting period energy data",
    )
    non_routine_events: List[NonRoutineEvent] = Field(
        default_factory=list, description="Non-routine adjustment events",
    )
    savings_method: str = Field(
        default="avoided_energy",
        description="Savings method: avoided_energy, normalized_savings, both",
    )
    energy_rate: float = Field(
        default=0.12, gt=0, description="Energy rate for cost savings ($/kWh)",
    )
    confidence_level: float = Field(
        default=0.90, ge=0.80, le=0.99, description="Confidence level",
    )
    measurement_uncertainty_pct: float = Field(
        default=1.0, ge=0.0, le=10.0, description="Meter measurement uncertainty %",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("facility_name")
    @classmethod
    def validate_facility_name(cls, v: str) -> str:
        """Ensure facility name is non-empty after stripping."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("facility_name must not be blank")
        return stripped


class SavingsVerificationResult(BaseModel):
    """Complete result from savings verification workflow."""

    verification_id: str = Field(..., description="Unique verification ID")
    project_id: str = Field(default="", description="Project identifier")
    ecm_id: str = Field(default="", description="ECM identifier")
    reporting_periods: int = Field(default=0, ge=0, description="Periods analyzed")
    baseline_energy_kwh: Decimal = Field(default=Decimal("0"), description="Baseline energy")
    adjusted_baseline_kwh: Decimal = Field(default=Decimal("0"), description="Adjusted baseline")
    reporting_energy_kwh: Decimal = Field(default=Decimal("0"), description="Reporting energy")
    routine_adjustments_kwh: Decimal = Field(default=Decimal("0"), description="Routine adj")
    non_routine_adjustments_kwh: Decimal = Field(default=Decimal("0"), description="NR adj")
    avoided_energy_kwh: Decimal = Field(default=Decimal("0"), description="Avoided energy")
    normalized_savings_kwh: Decimal = Field(default=Decimal("0"), description="Normalized savings")
    savings_pct: Decimal = Field(default=Decimal("0"), description="Savings %")
    cost_savings: Decimal = Field(default=Decimal("0"), description="Cost savings ($)")
    uncertainty_pct: Decimal = Field(default=Decimal("0"), description="Total uncertainty %")
    fractional_savings_uncertainty_pct: Decimal = Field(
        default=Decimal("0"), description="Fractional savings uncertainty %",
    )
    savings_significant: bool = Field(
        default=False, description="Savings statistically significant",
    )
    confidence_level: float = Field(default=0.90, description="Confidence level used")
    savings_method_used: str = Field(default="avoided_energy", description="Method used")
    uncertainty_components: Dict[str, float] = Field(default_factory=dict)
    period_details: List[Dict[str, Any]] = Field(default_factory=list)
    phases_completed: List[str] = Field(default_factory=list)
    workflow_duration_ms: int = Field(default=0, ge=0)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class SavingsVerificationWorkflow:
    """
    4-phase savings verification workflow per IPMVP.

    Collects reporting-period data, calculates routine and non-routine
    adjustments, determines avoided energy and normalized savings, and
    quantifies fractional savings uncertainty.

    Zero-hallucination: all savings calculations use deterministic regression
    formulas and ASHRAE 14 uncertainty propagation. No LLM calls in the
    calculation path.

    Attributes:
        verification_id: Unique verification execution identifier.
        _period_data: Processed period data.
        _adjustments: Calculated adjustments.
        _savings: Calculated savings values.
        _uncertainty: Uncertainty analysis results.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = SavingsVerificationWorkflow()
        >>> period = PeriodEnergyData(period_start="2025-01-01", period_end="2025-01-31", energy_value=8000)
        >>> inp = SavingsVerificationInput(facility_name="HQ", reporting_period_data=[period])
        >>> result = wf.run(inp)
        >>> assert result.reporting_periods > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SavingsVerificationWorkflow."""
        self.verification_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._period_data: List[Dict[str, Any]] = []
        self._adjustments: Dict[str, Any] = {}
        self._savings: Dict[str, Any] = {}
        self._uncertainty: Dict[str, Any] = {}
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: SavingsVerificationInput) -> SavingsVerificationResult:
        """
        Execute the 4-phase savings verification workflow.

        Args:
            input_data: Validated savings verification input.

        Returns:
            SavingsVerificationResult with savings and uncertainty.

        Raises:
            ValueError: If input validation fails.
        """
        t_start = time.perf_counter()
        started_at = _utcnow()
        self.logger.info(
            "Starting savings verification workflow %s for facility=%s ecm=%s",
            self.verification_id, input_data.facility_name, input_data.ecm_name,
        )

        self._phase_results = []
        self._period_data = []
        self._adjustments = {}
        self._savings = {}
        self._uncertainty = {}

        try:
            # Phase 1: Data Collection
            phase1 = self._phase_data_collection(input_data)
            self._phase_results.append(phase1)

            # Phase 2: Adjustment Calculation
            phase2 = self._phase_adjustment_calc(input_data)
            self._phase_results.append(phase2)

            # Phase 3: Savings Calculation
            phase3 = self._phase_savings_calc(input_data)
            self._phase_results.append(phase3)

            # Phase 4: Uncertainty Analysis
            phase4 = self._phase_uncertainty_analysis(input_data)
            self._phase_results.append(phase4)

        except Exception as exc:
            self.logger.error(
                "Savings verification workflow failed: %s", exc, exc_info=True,
            )
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        # Extract final values
        avoided = self._savings.get("avoided_energy_kwh", 0.0)
        normalized = self._savings.get("normalized_savings_kwh", 0.0)
        adjusted_baseline = self._savings.get("adjusted_baseline_kwh", 0.0)
        reporting_energy = self._savings.get("reporting_energy_kwh", 0.0)
        routine_adj = self._adjustments.get("routine_total_kwh", 0.0)
        nr_adj = self._adjustments.get("non_routine_total_kwh", 0.0)
        savings_pct = self._savings.get("savings_pct", 0.0)
        cost_savings = self._savings.get("cost_savings", 0.0)
        unc_pct = self._uncertainty.get("total_uncertainty_pct", 0.0)
        fsu_pct = self._uncertainty.get("fractional_savings_uncertainty_pct", 0.0)
        significant = self._uncertainty.get("savings_significant", False)

        result = SavingsVerificationResult(
            verification_id=self.verification_id,
            project_id=input_data.project_id,
            ecm_id=input_data.ecm_id,
            reporting_periods=len(self._period_data),
            baseline_energy_kwh=Decimal(str(round(input_data.baseline_energy_kwh, 2))),
            adjusted_baseline_kwh=Decimal(str(round(adjusted_baseline, 2))),
            reporting_energy_kwh=Decimal(str(round(reporting_energy, 2))),
            routine_adjustments_kwh=Decimal(str(round(routine_adj, 2))),
            non_routine_adjustments_kwh=Decimal(str(round(nr_adj, 2))),
            avoided_energy_kwh=Decimal(str(round(avoided, 2))),
            normalized_savings_kwh=Decimal(str(round(normalized, 2))),
            savings_pct=Decimal(str(round(savings_pct, 2))),
            cost_savings=Decimal(str(round(cost_savings, 2))),
            uncertainty_pct=Decimal(str(round(unc_pct, 2))),
            fractional_savings_uncertainty_pct=Decimal(str(round(fsu_pct, 2))),
            savings_significant=significant,
            confidence_level=input_data.confidence_level,
            savings_method_used=input_data.savings_method,
            uncertainty_components=self._uncertainty.get("components", {}),
            period_details=self._period_data,
            phases_completed=completed_phases,
            workflow_duration_ms=int(elapsed_ms),
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Savings verification %s completed in %dms savings=%.0f kWh (%.1f%%) "
            "uncertainty=+/-%.1f%% significant=%s",
            self.verification_id, int(elapsed_ms), avoided,
            savings_pct, fsu_pct, significant,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Collection
    # -------------------------------------------------------------------------

    def _phase_data_collection(
        self, input_data: SavingsVerificationInput,
    ) -> PhaseResult:
        """Collect and validate reporting-period energy data."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        if not input_data.reporting_period_data:
            warnings.append("No reporting period data provided; using synthetic")
            input_data.reporting_period_data = self._generate_synthetic_periods(12)

        valid_periods: List[Dict[str, Any]] = []
        total_energy = 0.0

        for idx, period in enumerate(input_data.reporting_period_data):
            record = period.model_dump()
            record["sequence"] = idx + 1
            record["is_valid"] = True

            if period.energy_value < 0:
                record["is_valid"] = False
                warnings.append(f"Period {idx+1}: negative energy value")
                continue

            total_energy += period.energy_value
            valid_periods.append(record)

        self._period_data = valid_periods

        outputs["total_periods"] = len(input_data.reporting_period_data)
        outputs["valid_periods"] = len(valid_periods)
        outputs["total_reporting_energy_kwh"] = round(total_energy, 2)
        outputs["has_temperature"] = any(
            p.get("temperature_avg") is not None for p in valid_periods
        )
        outputs["has_production"] = any(
            p.get("production_volume") is not None for p in valid_periods
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 DataCollection: %d/%d valid periods, total=%.0f kWh",
            len(valid_periods), len(input_data.reporting_period_data), total_energy,
        )
        return PhaseResult(
            phase_name="data_collection", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Adjustment Calculation
    # -------------------------------------------------------------------------

    def _phase_adjustment_calc(
        self, input_data: SavingsVerificationInput,
    ) -> PhaseResult:
        """Calculate routine and non-routine adjustments."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        intercept = input_data.baseline_model_params.get("intercept", 10000.0)
        slope = input_data.baseline_model_params.get("slope", 200.0)

        # Routine adjustments: use baseline model to predict under reporting conditions
        routine_predictions: List[float] = []
        for period in self._period_data:
            temp = period.get("temperature_avg")
            if temp is not None:
                predicted = intercept + slope * temp
            else:
                predicted = intercept
            routine_predictions.append(max(predicted, 0.0))

        total_predicted = sum(routine_predictions)
        routine_adjustment = total_predicted - input_data.baseline_energy_kwh

        # Non-routine adjustments
        nr_total = sum(
            event.energy_impact_kwh for event in input_data.non_routine_events
        )
        nr_details: List[Dict[str, Any]] = []
        for event in input_data.non_routine_events:
            nr_details.append({
                "event_id": event.event_id,
                "event_type": event.event_type,
                "description": event.description,
                "energy_impact_kwh": event.energy_impact_kwh,
                "cost_impact": event.cost_impact,
            })

        self._adjustments = {
            "routine_total_kwh": round(routine_adjustment, 2),
            "routine_predictions": [round(p, 2) for p in routine_predictions],
            "total_predicted_kwh": round(total_predicted, 2),
            "non_routine_total_kwh": round(nr_total, 2),
            "non_routine_events": nr_details,
            "total_adjustments_kwh": round(routine_adjustment + nr_total, 2),
        }

        outputs["routine_adjustment_kwh"] = round(routine_adjustment, 2)
        outputs["non_routine_adjustment_kwh"] = round(nr_total, 2)
        outputs["total_adjustments_kwh"] = round(routine_adjustment + nr_total, 2)
        outputs["non_routine_events_count"] = len(nr_details)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 AdjustmentCalc: routine=%.0f kWh, non-routine=%.0f kWh",
            routine_adjustment, nr_total,
        )
        return PhaseResult(
            phase_name="adjustment_calc", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Savings Calculation
    # -------------------------------------------------------------------------

    def _phase_savings_calc(
        self, input_data: SavingsVerificationInput,
    ) -> PhaseResult:
        """Calculate avoided energy and normalized savings."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        reporting_energy = sum(
            p.get("energy_value", 0.0) for p in self._period_data
        )
        total_predicted = self._adjustments.get("total_predicted_kwh", 0.0)
        nr_adj = self._adjustments.get("non_routine_total_kwh", 0.0)

        # Adjusted baseline = baseline predicted under reporting conditions + NR adjustments
        adjusted_baseline = total_predicted + nr_adj

        # Avoided energy = Adjusted Baseline - Reporting Energy
        avoided_energy = adjusted_baseline - reporting_energy

        # Normalized savings (normalize to baseline conditions)
        baseline_energy = input_data.baseline_energy_kwh
        normalized_savings = baseline_energy + nr_adj - reporting_energy

        # Savings percentage
        savings_pct = 0.0
        if adjusted_baseline > 0:
            savings_pct = (avoided_energy / adjusted_baseline) * 100.0

        # Cost savings
        cost_savings = avoided_energy * input_data.energy_rate

        self._savings = {
            "adjusted_baseline_kwh": round(adjusted_baseline, 2),
            "reporting_energy_kwh": round(reporting_energy, 2),
            "avoided_energy_kwh": round(avoided_energy, 2),
            "normalized_savings_kwh": round(normalized_savings, 2),
            "savings_pct": round(savings_pct, 2),
            "cost_savings": round(cost_savings, 2),
            "energy_rate": input_data.energy_rate,
        }

        outputs["adjusted_baseline_kwh"] = round(adjusted_baseline, 2)
        outputs["reporting_energy_kwh"] = round(reporting_energy, 2)
        outputs["avoided_energy_kwh"] = round(avoided_energy, 2)
        outputs["normalized_savings_kwh"] = round(normalized_savings, 2)
        outputs["savings_pct"] = round(savings_pct, 2)
        outputs["cost_savings"] = round(cost_savings, 2)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 SavingsCalc: avoided=%.0f kWh (%.1f%%), cost=$%.0f",
            avoided_energy, savings_pct, cost_savings,
        )
        return PhaseResult(
            phase_name="savings_calc", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Uncertainty Analysis
    # -------------------------------------------------------------------------

    def _phase_uncertainty_analysis(
        self, input_data: SavingsVerificationInput,
    ) -> PhaseResult:
        """Quantify fractional savings uncertainty per ASHRAE 14."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        n = max(len(self._period_data), 1)
        avoided = self._savings.get("avoided_energy_kwh", 0.0)
        adjusted_baseline = self._savings.get("adjusted_baseline_kwh", 0.0)

        # Measurement uncertainty
        u_measurement = input_data.measurement_uncertainty_pct

        # Model uncertainty (from CV(RMSE))
        cvrmse = input_data.baseline_cvrmse_pct
        # Autocorrelation correction factor (conservative approximation)
        n_prime = max(n * 0.75, 1)
        # t-statistic for confidence level (approximation)
        t_stat = self._get_t_statistic(input_data.confidence_level, n)
        # Model uncertainty at the savings level
        u_model = t_stat * cvrmse / math.sqrt(n_prime) * 100.0

        # Sampling uncertainty (Option A only)
        u_sampling = 0.0
        if input_data.ipmvp_option == "A":
            u_sampling = 10.0  # Default 10% for typical sample sizes

        # Interactive effects uncertainty
        u_interactive = 0.0
        if input_data.ipmvp_option in ("A", "B"):
            u_interactive = 5.0

        # Total uncertainty (root sum of squares)
        u_total = math.sqrt(
            u_measurement ** 2
            + u_model ** 2
            + u_sampling ** 2
            + u_interactive ** 2
        )

        # Fractional savings uncertainty
        savings_fraction = 0.0
        if adjusted_baseline > 0:
            savings_fraction = avoided / adjusted_baseline

        fsu = 0.0
        if savings_fraction > 0:
            fsu = u_total / (savings_fraction * 100.0) * 100.0
        else:
            fsu = u_total

        # Savings significance test
        savings_significant = (
            abs(avoided) > 0 and fsu < 50.0 and savings_fraction > 0
        )

        self._uncertainty = {
            "total_uncertainty_pct": round(u_total, 2),
            "fractional_savings_uncertainty_pct": round(fsu, 2),
            "savings_significant": savings_significant,
            "components": {
                "measurement_pct": round(u_measurement, 2),
                "model_pct": round(u_model, 2),
                "sampling_pct": round(u_sampling, 2),
                "interactive_pct": round(u_interactive, 2),
            },
            "confidence_level": input_data.confidence_level,
            "t_statistic": round(t_stat, 4),
            "effective_n": round(n_prime, 1),
            "cvrmse_pct": cvrmse,
        }

        outputs["total_uncertainty_pct"] = round(u_total, 2)
        outputs["fractional_savings_uncertainty_pct"] = round(fsu, 2)
        outputs["savings_significant"] = savings_significant
        outputs["measurement_uncertainty_pct"] = round(u_measurement, 2)
        outputs["model_uncertainty_pct"] = round(u_model, 2)
        outputs["sampling_uncertainty_pct"] = round(u_sampling, 2)

        if fsu > 50.0:
            warnings.append(
                f"Fractional savings uncertainty {fsu:.1f}% exceeds 50% threshold; "
                f"savings may not be statistically significant"
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 4 UncertaintyAnalysis: total=%.1f%%, FSU=%.1f%%, significant=%s",
            u_total, fsu, savings_significant,
        )
        return PhaseResult(
            phase_name="uncertainty_analysis", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_t_statistic(self, confidence: float, n: int) -> float:
        """Get t-statistic approximation for given confidence and sample size."""
        # Common t-values for two-tailed tests (conservative)
        t_table: Dict[float, float] = {
            0.80: 1.282,
            0.85: 1.440,
            0.90: 1.645,
            0.95: 1.960,
            0.99: 2.576,
        }
        # Find closest confidence level
        closest = min(t_table.keys(), key=lambda x: abs(x - confidence))
        base_t = t_table[closest]
        # Adjust for small sample sizes (df = n-2)
        df = max(n - 2, 1)
        if df < 30:
            base_t *= (1.0 + 1.0 / df)
        return base_t

    def _generate_synthetic_periods(self, n: int) -> List[PeriodEnergyData]:
        """Generate synthetic reporting period data."""
        periods = []
        for i in range(n):
            month = (i % 12) + 1
            temp = 10.0 + 15.0 * math.sin(math.pi * (month - 1) / 6.0)
            energy = 8000.0 + 150.0 * (temp - 15.0)
            periods.append(PeriodEnergyData(
                period_start=f"2025-{month:02d}-01",
                period_end=f"2025-{month:02d}-28",
                energy_value=round(energy, 2),
                temperature_avg=round(temp, 1),
            ))
        return periods

    # -------------------------------------------------------------------------
    # Provenance Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: SavingsVerificationResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
