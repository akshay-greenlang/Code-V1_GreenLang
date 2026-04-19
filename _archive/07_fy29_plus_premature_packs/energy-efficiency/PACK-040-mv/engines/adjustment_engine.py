# -*- coding: utf-8 -*-
"""
AdjustmentEngine - PACK-040 M&V Engine 2
==========================================

Routine and non-routine adjustment calculation engine for Measurement &
Verification per IPMVP Core Concepts 2022.  Computes adjustments that
account for changes in conditions between the baseline and reporting
periods so that observed savings reflect the actual impact of the
Energy Conservation Measure (ECM) rather than exogenous changes.

Adjustment Methodology:

    Routine Adjustments (conditions expected to change):
        Weather:
            E_adj = E_baseline_model(HDD_rp, CDD_rp)
            where HDD_rp/CDD_rp are reporting-period degree-days.

        Production:
            E_adj = E_baseline * (Production_rp / Production_bl)

        Occupancy:
            E_adj = E_baseline * (Occupancy_rp / Occupancy_bl)

        Operating Hours:
            E_adj = E_baseline * (Hours_rp / Hours_bl)

    Non-Routine Adjustments (conditions not expected to change):
        Floor Area Change:
            E_adj = delta_area * EUI_baseline

        Equipment Addition:
            E_adj = rated_power * operating_hours * load_factor

        Equipment Removal:
            E_adj = -(rated_power * operating_hours * load_factor)

        Schedule Change:
            E_adj = delta_hours * average_load

        Static Factor:
            E_adj = fixed_adjustment_value

    Adjusted Baseline:
        E_adjusted = E_baseline_model(routine_conditions_rp)
                     + sum(non_routine_adjustments)

Regulatory References:
    - IPMVP Core Concepts 2022, Chapter 5 (Adjustments)
    - ASHRAE Guideline 14-2014, Section 5.2 (Routine Adjustments)
    - ISO 50015:2014, Clause 8.3 (Adjustments)
    - ISO 50006:2014, Clause 7.4 (Baseline Adjustment)
    - FEMP M&V Guidelines 4.0, Chapter 5

Zero-Hallucination:
    - All adjustments computed via deterministic formulas
    - Engineering estimates use rated specifications only
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-040 M&V
Engine:  2 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AdjustmentCategory(str, Enum):
    """Top-level adjustment category per IPMVP.

    ROUTINE:      Conditions expected to change routinely between baseline
                  and reporting periods (weather, production, occupancy).
    NON_ROUTINE:  Conditions not expected to change (floor area, equipment
                  additions/removals, schedule changes).
    """
    ROUTINE = "routine"
    NON_ROUTINE = "non_routine"

class RoutineAdjustmentType(str, Enum):
    """Type of routine adjustment.

    WEATHER:          HDD/CDD-based weather normalisation.
    PRODUCTION:       Production-volume normalisation.
    OCCUPANCY:        Occupancy-level adjustment.
    OPERATING_HOURS:  Operating-hours proportional adjustment.
    CUSTOM_ROUTINE:   Custom routine variable adjustment.
    """
    WEATHER = "weather"
    PRODUCTION = "production"
    OCCUPANCY = "occupancy"
    OPERATING_HOURS = "operating_hours"
    CUSTOM_ROUTINE = "custom_routine"

class NonRoutineAdjustmentType(str, Enum):
    """Type of non-routine adjustment.

    FLOOR_AREA_CHANGE:    Building expansion or contraction.
    EQUIPMENT_ADDITION:   New equipment added post-baseline.
    EQUIPMENT_REMOVAL:    Equipment removed/decommissioned post-baseline.
    SCHEDULE_CHANGE:      Operating schedule change.
    STATIC_FACTOR:        Fixed one-time adjustment.
    FUEL_SWITCH:          Change in fuel source.
    PROCESS_CHANGE:       Change in manufacturing process.
    ENVELOPE_CHANGE:      Building envelope modification.
    CUSTOM_NON_ROUTINE:   Custom non-routine adjustment.
    """
    FLOOR_AREA_CHANGE = "floor_area_change"
    EQUIPMENT_ADDITION = "equipment_addition"
    EQUIPMENT_REMOVAL = "equipment_removal"
    SCHEDULE_CHANGE = "schedule_change"
    STATIC_FACTOR = "static_factor"
    FUEL_SWITCH = "fuel_switch"
    PROCESS_CHANGE = "process_change"
    ENVELOPE_CHANGE = "envelope_change"
    CUSTOM_NON_ROUTINE = "custom_non_routine"

class AdjustmentStatus(str, Enum):
    """Status of an adjustment in the M&V process.

    DRAFT:      Initial calculation, not yet reviewed.
    REVIEWED:   Reviewed by M&V practitioner.
    APPROVED:   Approved by project stakeholder.
    DISPUTED:   Under dispute or re-evaluation.
    ARCHIVED:   Historical, superseded by newer adjustment.
    """
    DRAFT = "draft"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    DISPUTED = "disputed"
    ARCHIVED = "archived"

class UncertaintyLevel(str, Enum):
    """Qualitative uncertainty level for adjustment estimate.

    LOW:      Well-documented, metered, uncertainty <10%.
    MEDIUM:   Engineering estimate, uncertainty 10-25%.
    HIGH:     Rough estimate, significant assumptions, uncertainty >25%.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class WeatherNormMethod(str, Enum):
    """Weather normalisation method.

    DEGREE_DAY:     HDD/CDD degree-day regression normalisation.
    BIN_METHOD:     Temperature bin analysis.
    REGRESSION:     Multivariate regression with weather variables.
    TMY_NORMAL:     Typical Meteorological Year normalisation.
    """
    DEGREE_DAY = "degree_day"
    BIN_METHOD = "bin_method"
    REGRESSION = "regression"
    TMY_NORMAL = "tmy_normal"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default balance points (Fahrenheit).
DEFAULT_HEATING_BP_F: Decimal = Decimal("65")
DEFAULT_COOLING_BP_F: Decimal = Decimal("65")

# Temperature conversion.
NINE_FIFTHS: Decimal = Decimal("1.8")
THIRTY_TWO: Decimal = Decimal("32")

# Typical equipment load factors by category.
EQUIPMENT_LOAD_FACTORS: Dict[str, Decimal] = {
    "lighting": Decimal("1.0"),
    "hvac_chiller": Decimal("0.75"),
    "hvac_boiler": Decimal("0.70"),
    "hvac_ahu": Decimal("0.80"),
    "motor": Decimal("0.75"),
    "pump": Decimal("0.70"),
    "fan": Decimal("0.80"),
    "compressor": Decimal("0.85"),
    "transformer": Decimal("0.60"),
    "server": Decimal("0.50"),
    "kitchen": Decimal("0.40"),
    "process": Decimal("0.65"),
    "generic": Decimal("0.70"),
}

# Maximum reasonable adjustment as percentage of baseline energy.
MAX_ADJUSTMENT_PCT: Decimal = Decimal("50")

# Default hours per year for annualisation.
HOURS_PER_YEAR: Decimal = Decimal("8760")

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class WeatherAdjustmentInput(BaseModel):
    """Input for weather-based routine adjustment.

    Attributes:
        baseline_hdd: Total HDD in baseline period.
        baseline_cdd: Total CDD in baseline period.
        reporting_hdd: Total HDD in reporting period.
        reporting_cdd: Total CDD in reporting period.
        heating_coefficient: Energy per HDD from regression model.
        cooling_coefficient: Energy per CDD from regression model.
        baseline_intercept: Model intercept (base load).
        n_periods_baseline: Number of periods in baseline.
        n_periods_reporting: Number of periods in reporting.
        heating_balance_point_f: Heating balance point (F).
        cooling_balance_point_f: Cooling balance point (F).
        method: Weather normalisation method.
    """
    baseline_hdd: Decimal = Field(
        default=Decimal("0"), ge=0, description="Baseline period HDD"
    )
    baseline_cdd: Decimal = Field(
        default=Decimal("0"), ge=0, description="Baseline period CDD"
    )
    reporting_hdd: Decimal = Field(
        default=Decimal("0"), ge=0, description="Reporting period HDD"
    )
    reporting_cdd: Decimal = Field(
        default=Decimal("0"), ge=0, description="Reporting period CDD"
    )
    heating_coefficient: Decimal = Field(
        default=Decimal("0"), description="Energy per HDD"
    )
    cooling_coefficient: Decimal = Field(
        default=Decimal("0"), description="Energy per CDD"
    )
    baseline_intercept: Decimal = Field(
        default=Decimal("0"), description="Model intercept"
    )
    n_periods_baseline: int = Field(
        default=12, ge=1, description="Baseline periods"
    )
    n_periods_reporting: int = Field(
        default=12, ge=1, description="Reporting periods"
    )
    heating_balance_point_f: Decimal = Field(
        default=DEFAULT_HEATING_BP_F, description="Heating balance point (F)"
    )
    cooling_balance_point_f: Decimal = Field(
        default=DEFAULT_COOLING_BP_F, description="Cooling balance point (F)"
    )
    method: WeatherNormMethod = Field(
        default=WeatherNormMethod.DEGREE_DAY, description="Normalisation method"
    )

class ProductionAdjustmentInput(BaseModel):
    """Input for production-based routine adjustment.

    Attributes:
        baseline_production: Total production in baseline period.
        reporting_production: Total production in reporting period.
        baseline_energy: Total energy in baseline period.
        energy_per_unit: Energy intensity per production unit.
        fixed_component_pct: Percentage of energy that is fixed (non-production).
        production_unit: Unit of production measure.
    """
    baseline_production: Decimal = Field(
        default=Decimal("0"), ge=0, description="Baseline production"
    )
    reporting_production: Decimal = Field(
        default=Decimal("0"), ge=0, description="Reporting production"
    )
    baseline_energy: Decimal = Field(
        default=Decimal("0"), ge=0, description="Baseline energy"
    )
    energy_per_unit: Decimal = Field(
        default=Decimal("0"), ge=0, description="Energy per production unit"
    )
    fixed_component_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
        description="Fixed energy percentage"
    )
    production_unit: str = Field(
        default="units", description="Production unit name"
    )

class OccupancyAdjustmentInput(BaseModel):
    """Input for occupancy-based routine adjustment.

    Attributes:
        baseline_occupancy_pct: Average occupancy in baseline (%).
        reporting_occupancy_pct: Average occupancy in reporting (%).
        baseline_energy: Total energy in baseline period.
        occupancy_sensitivity: Energy change per 1% occupancy change.
        fixed_component_pct: Percentage of energy independent of occupancy.
    """
    baseline_occupancy_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=100, description="Baseline occupancy %"
    )
    reporting_occupancy_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=100, description="Reporting occupancy %"
    )
    baseline_energy: Decimal = Field(
        default=Decimal("0"), ge=0, description="Baseline energy"
    )
    occupancy_sensitivity: Decimal = Field(
        default=Decimal("0"), ge=0, description="Energy per 1% occupancy"
    )
    fixed_component_pct: Decimal = Field(
        default=Decimal("30"), ge=0, le=100,
        description="Fixed energy percentage"
    )

class OperatingHoursAdjustmentInput(BaseModel):
    """Input for operating hours-based routine adjustment.

    Attributes:
        baseline_hours: Total operating hours in baseline.
        reporting_hours: Total operating hours in reporting.
        baseline_energy: Total energy in baseline period.
        fixed_component_pct: Percentage of energy independent of hours.
    """
    baseline_hours: Decimal = Field(
        default=Decimal("0"), ge=0, description="Baseline operating hours"
    )
    reporting_hours: Decimal = Field(
        default=Decimal("0"), ge=0, description="Reporting operating hours"
    )
    baseline_energy: Decimal = Field(
        default=Decimal("0"), ge=0, description="Baseline energy"
    )
    fixed_component_pct: Decimal = Field(
        default=Decimal("20"), ge=0, le=100,
        description="Fixed energy percentage"
    )

class NonRoutineAdjustmentInput(BaseModel):
    """Input for a non-routine adjustment.

    Attributes:
        adjustment_type: Type of non-routine adjustment.
        description: Human-readable description of the change.
        effective_date: Date the change took effect.
        delta_floor_area_sqft: Change in floor area (sq ft).
        baseline_eui: Baseline energy use intensity (kWh/sqft/yr).
        rated_power_kw: Rated power of added/removed equipment (kW).
        operating_hours: Annual operating hours of equipment.
        load_factor: Equipment load factor (0-1).
        equipment_category: Category for default load factor.
        delta_hours_per_day: Change in daily operating hours.
        average_load_kw: Average facility load during added hours (kW).
        days_affected: Number of days affected in the period.
        static_adjustment_value: Fixed adjustment value (kWh).
        energy_impact_kwh: Pre-calculated energy impact if known.
        uncertainty_pct: Estimated uncertainty percentage.
        uncertainty_level: Qualitative uncertainty level.
        justification: Engineering justification.
        source_document: Reference document for adjustment.
    """
    adjustment_type: NonRoutineAdjustmentType = Field(
        default=NonRoutineAdjustmentType.STATIC_FACTOR,
        description="Type of non-routine adjustment",
    )
    description: str = Field(
        default="", max_length=1000, description="Description"
    )
    effective_date: datetime = Field(
        default_factory=utcnow, description="Effective date"
    )
    delta_floor_area_sqft: Decimal = Field(
        default=Decimal("0"), description="Floor area change (sq ft)"
    )
    baseline_eui: Decimal = Field(
        default=Decimal("0"), ge=0, description="Baseline EUI (kWh/sqft/yr)"
    )
    rated_power_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Rated power (kW)"
    )
    operating_hours: Decimal = Field(
        default=Decimal("0"), ge=0, description="Annual operating hours"
    )
    load_factor: Optional[Decimal] = Field(
        default=None, ge=0, le=1, description="Load factor (0-1)"
    )
    equipment_category: str = Field(
        default="generic", description="Equipment category"
    )
    delta_hours_per_day: Decimal = Field(
        default=Decimal("0"), description="Change in daily hours"
    )
    average_load_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Average load (kW)"
    )
    days_affected: int = Field(
        default=365, ge=0, description="Days affected"
    )
    static_adjustment_value: Decimal = Field(
        default=Decimal("0"), description="Static adjustment (kWh)"
    )
    energy_impact_kwh: Optional[Decimal] = Field(
        default=None, description="Pre-calculated energy impact"
    )
    uncertainty_pct: Decimal = Field(
        default=Decimal("15"), ge=0, le=100,
        description="Uncertainty percentage"
    )
    uncertainty_level: UncertaintyLevel = Field(
        default=UncertaintyLevel.MEDIUM, description="Uncertainty level"
    )
    justification: str = Field(
        default="", max_length=2000, description="Justification"
    )
    source_document: str = Field(
        default="", max_length=500, description="Source document"
    )

class AdjustmentConfig(BaseModel):
    """Configuration for adjustment calculations.

    Attributes:
        project_id: M&V project identifier.
        ecm_id: ECM identifier.
        facility_id: Facility identifier.
        reporting_period_start: Start of reporting period.
        reporting_period_end: End of reporting period.
        baseline_energy: Total baseline-period energy.
        energy_unit: Unit of energy measurement.
        max_adjustment_pct: Maximum single adjustment as % of baseline.
        require_justification: Whether justification is required.
    """
    project_id: str = Field(default="", description="M&V project ID")
    ecm_id: str = Field(default="", description="ECM identifier")
    facility_id: str = Field(default="", description="Facility ID")
    reporting_period_start: datetime = Field(
        default_factory=utcnow, description="Reporting period start"
    )
    reporting_period_end: datetime = Field(
        default_factory=utcnow, description="Reporting period end"
    )
    baseline_energy: Decimal = Field(
        default=Decimal("0"), ge=0, description="Baseline energy"
    )
    energy_unit: str = Field(default="kWh", description="Energy unit")
    max_adjustment_pct: Decimal = Field(
        default=MAX_ADJUSTMENT_PCT, ge=0,
        description="Max adjustment as % of baseline"
    )
    require_justification: bool = Field(
        default=True, description="Require justification for non-routine"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class AdjustmentRecord(BaseModel):
    """Record of a single calculated adjustment.

    Attributes:
        adjustment_id: Unique identifier for this adjustment.
        category: Routine or non-routine.
        adjustment_type: Specific type of adjustment.
        description: Human-readable description.
        energy_adjustment: Energy adjustment value (kWh).
        adjustment_pct: Adjustment as percentage of baseline.
        direction: "increase" or "decrease" to baseline.
        uncertainty_pct: Uncertainty of this adjustment.
        uncertainty_level: Qualitative uncertainty level.
        uncertainty_absolute: Absolute uncertainty (kWh).
        justification: Engineering justification.
        source_document: Reference document.
        effective_date: Date adjustment takes effect.
        calculation_method: Method used for calculation.
        input_parameters: Key input parameters.
        status: Approval status.
        is_capped: Whether adjustment was capped at max.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    adjustment_id: str = Field(default_factory=_new_uuid)
    category: AdjustmentCategory = Field(default=AdjustmentCategory.ROUTINE)
    adjustment_type: str = Field(default="")
    description: str = Field(default="", max_length=1000)
    energy_adjustment: Decimal = Field(default=Decimal("0"))
    adjustment_pct: Decimal = Field(default=Decimal("0"))
    direction: str = Field(default="increase")
    uncertainty_pct: Decimal = Field(default=Decimal("0"))
    uncertainty_level: UncertaintyLevel = Field(default=UncertaintyLevel.MEDIUM)
    uncertainty_absolute: Decimal = Field(default=Decimal("0"))
    justification: str = Field(default="", max_length=2000)
    source_document: str = Field(default="", max_length=500)
    effective_date: datetime = Field(default_factory=utcnow)
    calculation_method: str = Field(default="")
    input_parameters: Dict[str, str] = Field(default_factory=dict)
    status: AdjustmentStatus = Field(default=AdjustmentStatus.DRAFT)
    is_capped: bool = Field(default=False)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class RoutineAdjustmentResult(BaseModel):
    """Result of all routine adjustments for a reporting period.

    Attributes:
        result_id: Unique result identifier.
        adjusted_baseline_energy: Baseline adjusted for routine conditions.
        unadjusted_baseline_energy: Original baseline energy.
        total_routine_adjustment: Sum of all routine adjustments.
        adjustment_records: Individual routine adjustment records.
        weather_adjustment: Weather component (if applicable).
        production_adjustment: Production component (if applicable).
        occupancy_adjustment: Occupancy component (if applicable).
        hours_adjustment: Operating hours component (if applicable).
        processing_time_ms: Processing duration.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    adjusted_baseline_energy: Decimal = Field(default=Decimal("0"))
    unadjusted_baseline_energy: Decimal = Field(default=Decimal("0"))
    total_routine_adjustment: Decimal = Field(default=Decimal("0"))
    adjustment_records: List[AdjustmentRecord] = Field(default_factory=list)
    weather_adjustment: Decimal = Field(default=Decimal("0"))
    production_adjustment: Decimal = Field(default=Decimal("0"))
    occupancy_adjustment: Decimal = Field(default=Decimal("0"))
    hours_adjustment: Decimal = Field(default=Decimal("0"))
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class NonRoutineAdjustmentResult(BaseModel):
    """Result of all non-routine adjustments for a reporting period.

    Attributes:
        result_id: Unique result identifier.
        total_non_routine_adjustment: Sum of all non-routine adjustments.
        adjustment_records: Individual non-routine adjustment records.
        n_adjustments: Number of non-routine adjustments.
        total_positive: Sum of positive (increase) adjustments.
        total_negative: Sum of negative (decrease) adjustments.
        combined_uncertainty_pct: Combined uncertainty of all adjustments.
        combined_uncertainty_absolute: Absolute combined uncertainty.
        processing_time_ms: Processing duration.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    total_non_routine_adjustment: Decimal = Field(default=Decimal("0"))
    adjustment_records: List[AdjustmentRecord] = Field(default_factory=list)
    n_adjustments: int = Field(default=0)
    total_positive: Decimal = Field(default=Decimal("0"))
    total_negative: Decimal = Field(default=Decimal("0"))
    combined_uncertainty_pct: Decimal = Field(default=Decimal("0"))
    combined_uncertainty_absolute: Decimal = Field(default=Decimal("0"))
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class FullAdjustmentResult(BaseModel):
    """Complete adjustment result combining routine and non-routine.

    Attributes:
        result_id: Unique result identifier.
        project_id: M&V project identifier.
        ecm_id: ECM identifier.
        facility_id: Facility identifier.
        reporting_period_start: Start of reporting period.
        reporting_period_end: End of reporting period.
        unadjusted_baseline_energy: Original baseline energy.
        routine_result: Routine adjustment results.
        non_routine_result: Non-routine adjustment results.
        total_adjustment: Sum of all adjustments.
        adjusted_baseline_energy: Fully adjusted baseline energy.
        adjustment_pct_of_baseline: Total adjustment as % of baseline.
        total_adjustment_records: All adjustment records combined.
        energy_unit: Unit of energy measurement.
        warnings: Any warnings generated.
        recommendations: Analysis recommendations.
        processing_time_ms: Processing duration.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    project_id: str = Field(default="")
    ecm_id: str = Field(default="")
    facility_id: str = Field(default="")
    reporting_period_start: datetime = Field(default_factory=utcnow)
    reporting_period_end: datetime = Field(default_factory=utcnow)
    unadjusted_baseline_energy: Decimal = Field(default=Decimal("0"))
    routine_result: Optional[RoutineAdjustmentResult] = Field(default=None)
    non_routine_result: Optional[NonRoutineAdjustmentResult] = Field(
        default=None
    )
    total_adjustment: Decimal = Field(default=Decimal("0"))
    adjusted_baseline_energy: Decimal = Field(default=Decimal("0"))
    adjustment_pct_of_baseline: Decimal = Field(default=Decimal("0"))
    total_adjustment_records: List[AdjustmentRecord] = Field(
        default_factory=list
    )
    energy_unit: str = Field(default="kWh")
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AdjustmentEngine:
    """Routine and non-routine adjustment engine for M&V per IPMVP.

    Calculates adjustments to account for changes in conditions between
    the baseline and reporting periods.  Supports weather normalisation,
    production scaling, occupancy adjustment, operating hours correction,
    and multiple non-routine adjustment types including floor area changes,
    equipment additions/removals, schedule changes, and static factors.

    Usage::

        engine = AdjustmentEngine()
        config = AdjustmentConfig(
            project_id="PRJ-001",
            baseline_energy=Decimal("120000"),
        )
        weather_input = WeatherAdjustmentInput(
            baseline_hdd=Decimal("4500"), reporting_hdd=Decimal("5000"),
            heating_coefficient=Decimal("8.5"),
        )
        result = engine.calculate_weather_adjustment(weather_input, config)
        print(f"Weather adjustment: {result.energy_adjustment} kWh")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise AdjustmentEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - max_adjustment_pct (float): max single adjustment %
                - default_load_factor (float): default equipment load factor
                - require_justification (bool): require NRA justification
        """
        self.config = config or {}
        self._max_adj_pct = _decimal(
            self.config.get("max_adjustment_pct", MAX_ADJUSTMENT_PCT)
        )
        self._default_load_factor = _decimal(
            self.config.get("default_load_factor", Decimal("0.70"))
        )
        self._require_justification = self.config.get(
            "require_justification", True
        )
        logger.info(
            "AdjustmentEngine v%s initialised (max_adj=%.0f%%)",
            self.engine_version, float(self._max_adj_pct),
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def calculate_full_adjustment(
        self,
        adj_config: AdjustmentConfig,
        weather_input: Optional[WeatherAdjustmentInput] = None,
        production_input: Optional[ProductionAdjustmentInput] = None,
        occupancy_input: Optional[OccupancyAdjustmentInput] = None,
        hours_input: Optional[OperatingHoursAdjustmentInput] = None,
        non_routine_inputs: Optional[List[NonRoutineAdjustmentInput]] = None,
    ) -> FullAdjustmentResult:
        """Calculate complete routine and non-routine adjustments.

        Orchestrates all adjustment calculations for a reporting period,
        combines routine and non-routine adjustments, and validates the
        total adjustment against the baseline.

        Args:
            adj_config: Adjustment configuration.
            weather_input: Optional weather adjustment input.
            production_input: Optional production adjustment input.
            occupancy_input: Optional occupancy adjustment input.
            hours_input: Optional operating hours adjustment input.
            non_routine_inputs: Optional list of non-routine adjustments.

        Returns:
            FullAdjustmentResult with all adjustments combined.
        """
        t0 = time.perf_counter()
        logger.info(
            "Calculating full adjustment: project=%s, baseline=%.0f",
            adj_config.project_id, float(adj_config.baseline_energy),
        )

        # Calculate routine adjustments
        routine_result = self.calculate_routine_adjustments(
            adj_config, weather_input, production_input,
            occupancy_input, hours_input,
        )

        # Calculate non-routine adjustments
        non_routine_result: Optional[NonRoutineAdjustmentResult] = None
        if non_routine_inputs:
            non_routine_result = self.calculate_non_routine_adjustments(
                adj_config, non_routine_inputs,
            )

        # Combine
        total_routine = routine_result.total_routine_adjustment
        total_non_routine = (
            non_routine_result.total_non_routine_adjustment
            if non_routine_result else Decimal("0")
        )
        total_adjustment = total_routine + total_non_routine

        adjusted_baseline = adj_config.baseline_energy + total_adjustment
        adj_pct = _safe_pct(total_adjustment, adj_config.baseline_energy)

        # Collect all records
        all_records = list(routine_result.adjustment_records)
        if non_routine_result:
            all_records.extend(non_routine_result.adjustment_records)

        # Generate warnings
        warnings = self._generate_warnings(
            total_adjustment, adj_config.baseline_energy,
            adjusted_baseline, all_records,
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            routine_result, non_routine_result, adj_config,
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000.0)

        result = FullAdjustmentResult(
            project_id=adj_config.project_id,
            ecm_id=adj_config.ecm_id,
            facility_id=adj_config.facility_id,
            reporting_period_start=adj_config.reporting_period_start,
            reporting_period_end=adj_config.reporting_period_end,
            unadjusted_baseline_energy=adj_config.baseline_energy,
            routine_result=routine_result,
            non_routine_result=non_routine_result,
            total_adjustment=_round_val(total_adjustment, 2),
            adjusted_baseline_energy=_round_val(adjusted_baseline, 2),
            adjustment_pct_of_baseline=_round_val(adj_pct, 4),
            total_adjustment_records=all_records,
            energy_unit=adj_config.energy_unit,
            warnings=warnings,
            recommendations=recommendations,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Full adjustment: total=%.1f %s (%.2f%%), "
            "adjusted_baseline=%.1f, routine=%.1f, non_routine=%.1f, "
            "hash=%s (%.1f ms)",
            float(total_adjustment), adj_config.energy_unit,
            float(adj_pct), float(adjusted_baseline),
            float(total_routine), float(total_non_routine),
            result.provenance_hash[:16], float(elapsed_ms),
        )
        return result

    def calculate_routine_adjustments(
        self,
        adj_config: AdjustmentConfig,
        weather_input: Optional[WeatherAdjustmentInput] = None,
        production_input: Optional[ProductionAdjustmentInput] = None,
        occupancy_input: Optional[OccupancyAdjustmentInput] = None,
        hours_input: Optional[OperatingHoursAdjustmentInput] = None,
    ) -> RoutineAdjustmentResult:
        """Calculate all routine adjustments for a reporting period.

        Args:
            adj_config: Adjustment configuration.
            weather_input: Optional weather adjustment input.
            production_input: Optional production adjustment input.
            occupancy_input: Optional occupancy adjustment input.
            hours_input: Optional operating hours adjustment input.

        Returns:
            RoutineAdjustmentResult with all routine adjustments.
        """
        t0 = time.perf_counter()
        logger.info("Calculating routine adjustments")

        records: List[AdjustmentRecord] = []
        weather_adj = Decimal("0")
        prod_adj = Decimal("0")
        occ_adj = Decimal("0")
        hours_adj = Decimal("0")

        if weather_input:
            rec = self.calculate_weather_adjustment(weather_input, adj_config)
            records.append(rec)
            weather_adj = rec.energy_adjustment

        if production_input:
            rec = self.calculate_production_adjustment(
                production_input, adj_config
            )
            records.append(rec)
            prod_adj = rec.energy_adjustment

        if occupancy_input:
            rec = self.calculate_occupancy_adjustment(
                occupancy_input, adj_config
            )
            records.append(rec)
            occ_adj = rec.energy_adjustment

        if hours_input:
            rec = self.calculate_hours_adjustment(hours_input, adj_config)
            records.append(rec)
            hours_adj = rec.energy_adjustment

        total_routine = weather_adj + prod_adj + occ_adj + hours_adj
        adjusted_baseline = adj_config.baseline_energy + total_routine

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000.0)

        result = RoutineAdjustmentResult(
            adjusted_baseline_energy=_round_val(adjusted_baseline, 2),
            unadjusted_baseline_energy=adj_config.baseline_energy,
            total_routine_adjustment=_round_val(total_routine, 2),
            adjustment_records=records,
            weather_adjustment=_round_val(weather_adj, 2),
            production_adjustment=_round_val(prod_adj, 2),
            occupancy_adjustment=_round_val(occ_adj, 2),
            hours_adjustment=_round_val(hours_adj, 2),
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Routine adjustments: total=%.1f (weather=%.1f, prod=%.1f, "
            "occ=%.1f, hours=%.1f), hash=%s (%.1f ms)",
            float(total_routine), float(weather_adj), float(prod_adj),
            float(occ_adj), float(hours_adj),
            result.provenance_hash[:16], float(elapsed_ms),
        )
        return result

    def calculate_weather_adjustment(
        self,
        weather_input: WeatherAdjustmentInput,
        adj_config: AdjustmentConfig,
    ) -> AdjustmentRecord:
        """Calculate weather-based routine adjustment.

        Uses the difference in HDD/CDD between baseline and reporting
        periods, multiplied by the regression coefficients, to determine
        the energy adjustment for weather normalisation.

        Args:
            weather_input: Weather adjustment parameters.
            adj_config: Adjustment configuration.

        Returns:
            AdjustmentRecord for the weather adjustment.
        """
        t0 = time.perf_counter()
        logger.info("Calculating weather adjustment")

        # Adjusted baseline at reporting-period weather conditions
        # E_adj = intercept*n + heat_coeff*HDD_rp + cool_coeff*CDD_rp
        adjusted_energy = (
            weather_input.baseline_intercept
            * _decimal(weather_input.n_periods_reporting)
            + weather_input.heating_coefficient * weather_input.reporting_hdd
            + weather_input.cooling_coefficient * weather_input.reporting_cdd
        )

        # Baseline energy at baseline-period weather conditions
        baseline_energy = (
            weather_input.baseline_intercept
            * _decimal(weather_input.n_periods_baseline)
            + weather_input.heating_coefficient * weather_input.baseline_hdd
            + weather_input.cooling_coefficient * weather_input.baseline_cdd
        )

        # Weather adjustment = difference
        weather_adj = adjusted_energy - baseline_energy

        # Cap adjustment
        weather_adj, is_capped = self._cap_adjustment(
            weather_adj, adj_config.baseline_energy
        )

        direction = "increase" if weather_adj >= Decimal("0") else "decrease"
        adj_pct = _safe_pct(abs(weather_adj), adj_config.baseline_energy)

        record = AdjustmentRecord(
            category=AdjustmentCategory.ROUTINE,
            adjustment_type=RoutineAdjustmentType.WEATHER.value,
            description=(
                f"Weather normalisation: HDD {float(weather_input.baseline_hdd):.0f}"
                f" -> {float(weather_input.reporting_hdd):.0f}, "
                f"CDD {float(weather_input.baseline_cdd):.0f}"
                f" -> {float(weather_input.reporting_cdd):.0f}"
            ),
            energy_adjustment=_round_val(weather_adj, 2),
            adjustment_pct=_round_val(adj_pct, 4),
            direction=direction,
            uncertainty_pct=Decimal("5"),
            uncertainty_level=UncertaintyLevel.LOW,
            uncertainty_absolute=_round_val(
                abs(weather_adj) * Decimal("0.05"), 2
            ),
            calculation_method=weather_input.method.value,
            input_parameters={
                "baseline_hdd": str(weather_input.baseline_hdd),
                "reporting_hdd": str(weather_input.reporting_hdd),
                "baseline_cdd": str(weather_input.baseline_cdd),
                "reporting_cdd": str(weather_input.reporting_cdd),
                "heating_coefficient": str(weather_input.heating_coefficient),
                "cooling_coefficient": str(weather_input.cooling_coefficient),
            },
            is_capped=is_capped,
        )
        record.provenance_hash = _compute_hash(record)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Weather adjustment: %.1f kWh (%s), hash=%s (%.1f ms)",
            float(weather_adj), direction,
            record.provenance_hash[:16], elapsed,
        )
        return record

    def calculate_production_adjustment(
        self,
        prod_input: ProductionAdjustmentInput,
        adj_config: AdjustmentConfig,
    ) -> AdjustmentRecord:
        """Calculate production-based routine adjustment.

        Adjusts energy for production volume differences using either
        the ratio method or the energy-per-unit method with a fixed
        component split.

        Args:
            prod_input: Production adjustment parameters.
            adj_config: Adjustment configuration.

        Returns:
            AdjustmentRecord for the production adjustment.
        """
        t0 = time.perf_counter()
        logger.info("Calculating production adjustment")

        bl_prod = prod_input.baseline_production
        rp_prod = prod_input.reporting_production
        bl_energy = prod_input.baseline_energy
        fixed_pct = prod_input.fixed_component_pct

        if bl_prod == Decimal("0"):
            prod_adj = Decimal("0")
        else:
            # Split into fixed and variable components
            fixed_energy = bl_energy * fixed_pct / Decimal("100")
            variable_energy = bl_energy - fixed_energy

            # Scale variable component by production ratio
            prod_ratio = _safe_divide(rp_prod, bl_prod)
            adjusted_variable = variable_energy * prod_ratio
            adjusted_total = fixed_energy + adjusted_variable

            prod_adj = adjusted_total - bl_energy

        prod_adj, is_capped = self._cap_adjustment(
            prod_adj, adj_config.baseline_energy
        )
        direction = "increase" if prod_adj >= Decimal("0") else "decrease"
        adj_pct = _safe_pct(abs(prod_adj), adj_config.baseline_energy)

        record = AdjustmentRecord(
            category=AdjustmentCategory.ROUTINE,
            adjustment_type=RoutineAdjustmentType.PRODUCTION.value,
            description=(
                f"Production normalisation: {float(bl_prod):.0f}"
                f" -> {float(rp_prod):.0f} {prod_input.production_unit}"
            ),
            energy_adjustment=_round_val(prod_adj, 2),
            adjustment_pct=_round_val(adj_pct, 4),
            direction=direction,
            uncertainty_pct=Decimal("10"),
            uncertainty_level=UncertaintyLevel.LOW,
            uncertainty_absolute=_round_val(
                abs(prod_adj) * Decimal("0.10"), 2
            ),
            calculation_method="ratio_with_fixed_split",
            input_parameters={
                "baseline_production": str(bl_prod),
                "reporting_production": str(rp_prod),
                "baseline_energy": str(bl_energy),
                "fixed_component_pct": str(fixed_pct),
            },
            is_capped=is_capped,
        )
        record.provenance_hash = _compute_hash(record)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Production adjustment: %.1f kWh (%s), hash=%s (%.1f ms)",
            float(prod_adj), direction,
            record.provenance_hash[:16], elapsed,
        )
        return record

    def calculate_occupancy_adjustment(
        self,
        occ_input: OccupancyAdjustmentInput,
        adj_config: AdjustmentConfig,
    ) -> AdjustmentRecord:
        """Calculate occupancy-based routine adjustment.

        Adjusts energy for occupancy differences between baseline and
        reporting periods, accounting for a fixed energy component.

        Args:
            occ_input: Occupancy adjustment parameters.
            adj_config: Adjustment configuration.

        Returns:
            AdjustmentRecord for the occupancy adjustment.
        """
        t0 = time.perf_counter()
        logger.info("Calculating occupancy adjustment")

        bl_occ = occ_input.baseline_occupancy_pct
        rp_occ = occ_input.reporting_occupancy_pct
        bl_energy = occ_input.baseline_energy
        fixed_pct = occ_input.fixed_component_pct

        if bl_occ == Decimal("0"):
            occ_adj = Decimal("0")
        elif occ_input.occupancy_sensitivity > Decimal("0"):
            # Use sensitivity coefficient
            delta_occ = rp_occ - bl_occ
            occ_adj = delta_occ * occ_input.occupancy_sensitivity
        else:
            # Ratio method with fixed split
            fixed_energy = bl_energy * fixed_pct / Decimal("100")
            variable_energy = bl_energy - fixed_energy
            occ_ratio = _safe_divide(rp_occ, bl_occ)
            adjusted_variable = variable_energy * occ_ratio
            occ_adj = (fixed_energy + adjusted_variable) - bl_energy

        occ_adj, is_capped = self._cap_adjustment(
            occ_adj, adj_config.baseline_energy
        )
        direction = "increase" if occ_adj >= Decimal("0") else "decrease"
        adj_pct = _safe_pct(abs(occ_adj), adj_config.baseline_energy)

        record = AdjustmentRecord(
            category=AdjustmentCategory.ROUTINE,
            adjustment_type=RoutineAdjustmentType.OCCUPANCY.value,
            description=(
                f"Occupancy adjustment: {float(bl_occ):.1f}%"
                f" -> {float(rp_occ):.1f}%"
            ),
            energy_adjustment=_round_val(occ_adj, 2),
            adjustment_pct=_round_val(adj_pct, 4),
            direction=direction,
            uncertainty_pct=Decimal("15"),
            uncertainty_level=UncertaintyLevel.MEDIUM,
            uncertainty_absolute=_round_val(
                abs(occ_adj) * Decimal("0.15"), 2
            ),
            calculation_method="ratio_with_fixed_split",
            input_parameters={
                "baseline_occupancy_pct": str(bl_occ),
                "reporting_occupancy_pct": str(rp_occ),
                "fixed_component_pct": str(fixed_pct),
            },
            is_capped=is_capped,
        )
        record.provenance_hash = _compute_hash(record)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Occupancy adjustment: %.1f kWh (%s), hash=%s (%.1f ms)",
            float(occ_adj), direction,
            record.provenance_hash[:16], elapsed,
        )
        return record

    def calculate_hours_adjustment(
        self,
        hours_input: OperatingHoursAdjustmentInput,
        adj_config: AdjustmentConfig,
    ) -> AdjustmentRecord:
        """Calculate operating-hours routine adjustment.

        Adjusts energy for differences in operating hours between
        baseline and reporting periods, accounting for a fixed component.

        Args:
            hours_input: Operating hours adjustment parameters.
            adj_config: Adjustment configuration.

        Returns:
            AdjustmentRecord for the hours adjustment.
        """
        t0 = time.perf_counter()
        logger.info("Calculating operating hours adjustment")

        bl_hours = hours_input.baseline_hours
        rp_hours = hours_input.reporting_hours
        bl_energy = hours_input.baseline_energy
        fixed_pct = hours_input.fixed_component_pct

        if bl_hours == Decimal("0"):
            hours_adj = Decimal("0")
        else:
            fixed_energy = bl_energy * fixed_pct / Decimal("100")
            variable_energy = bl_energy - fixed_energy
            hours_ratio = _safe_divide(rp_hours, bl_hours)
            adjusted_variable = variable_energy * hours_ratio
            hours_adj = (fixed_energy + adjusted_variable) - bl_energy

        hours_adj, is_capped = self._cap_adjustment(
            hours_adj, adj_config.baseline_energy
        )
        direction = "increase" if hours_adj >= Decimal("0") else "decrease"
        adj_pct = _safe_pct(abs(hours_adj), adj_config.baseline_energy)

        record = AdjustmentRecord(
            category=AdjustmentCategory.ROUTINE,
            adjustment_type=RoutineAdjustmentType.OPERATING_HOURS.value,
            description=(
                f"Operating hours adjustment: {float(bl_hours):.0f}"
                f" -> {float(rp_hours):.0f} hours"
            ),
            energy_adjustment=_round_val(hours_adj, 2),
            adjustment_pct=_round_val(adj_pct, 4),
            direction=direction,
            uncertainty_pct=Decimal("5"),
            uncertainty_level=UncertaintyLevel.LOW,
            uncertainty_absolute=_round_val(
                abs(hours_adj) * Decimal("0.05"), 2
            ),
            calculation_method="ratio_with_fixed_split",
            input_parameters={
                "baseline_hours": str(bl_hours),
                "reporting_hours": str(rp_hours),
                "fixed_component_pct": str(fixed_pct),
            },
            is_capped=is_capped,
        )
        record.provenance_hash = _compute_hash(record)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Hours adjustment: %.1f kWh (%s), hash=%s (%.1f ms)",
            float(hours_adj), direction,
            record.provenance_hash[:16], elapsed,
        )
        return record

    def calculate_non_routine_adjustments(
        self,
        adj_config: AdjustmentConfig,
        non_routine_inputs: List[NonRoutineAdjustmentInput],
    ) -> NonRoutineAdjustmentResult:
        """Calculate all non-routine adjustments for a reporting period.

        Iterates through each non-routine adjustment input, calculates
        the energy impact, and combines with root-sum-square uncertainty.

        Args:
            adj_config: Adjustment configuration.
            non_routine_inputs: List of non-routine adjustment inputs.

        Returns:
            NonRoutineAdjustmentResult with all adjustments combined.
        """
        t0 = time.perf_counter()
        logger.info(
            "Calculating %d non-routine adjustments",
            len(non_routine_inputs),
        )

        records: List[AdjustmentRecord] = []
        total_positive = Decimal("0")
        total_negative = Decimal("0")
        uncertainty_squares = Decimal("0")

        for nr_input in non_routine_inputs:
            record = self._calculate_single_non_routine(nr_input, adj_config)
            records.append(record)

            if record.energy_adjustment >= Decimal("0"):
                total_positive += record.energy_adjustment
            else:
                total_negative += record.energy_adjustment

            # Root-sum-square for uncertainty
            uncertainty_squares += record.uncertainty_absolute ** 2

        total_nra = total_positive + total_negative
        combined_uncertainty = _decimal(
            math.sqrt(max(0, float(uncertainty_squares)))
        )
        combined_unc_pct = _safe_pct(combined_uncertainty, abs(total_nra)) if total_nra != Decimal("0") else Decimal("0")

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000.0)

        result = NonRoutineAdjustmentResult(
            total_non_routine_adjustment=_round_val(total_nra, 2),
            adjustment_records=records,
            n_adjustments=len(records),
            total_positive=_round_val(total_positive, 2),
            total_negative=_round_val(total_negative, 2),
            combined_uncertainty_pct=_round_val(combined_unc_pct, 4),
            combined_uncertainty_absolute=_round_val(combined_uncertainty, 2),
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Non-routine adjustments: total=%.1f, pos=%.1f, neg=%.1f, "
            "unc=%.1f, hash=%s (%.1f ms)",
            float(total_nra), float(total_positive), float(total_negative),
            float(combined_uncertainty),
            result.provenance_hash[:16], float(elapsed_ms),
        )
        return result

    def calculate_non_routine_single(
        self,
        nr_input: NonRoutineAdjustmentInput,
        adj_config: AdjustmentConfig,
    ) -> AdjustmentRecord:
        """Calculate a single non-routine adjustment.

        Public wrapper for _calculate_single_non_routine.

        Args:
            nr_input: Non-routine adjustment input.
            adj_config: Adjustment configuration.

        Returns:
            AdjustmentRecord for this non-routine adjustment.
        """
        return self._calculate_single_non_routine(nr_input, adj_config)

    # ------------------------------------------------------------------ #
    # Private: Non-Routine Calculation                                     #
    # ------------------------------------------------------------------ #

    def _calculate_single_non_routine(
        self,
        nr_input: NonRoutineAdjustmentInput,
        adj_config: AdjustmentConfig,
    ) -> AdjustmentRecord:
        """Calculate a single non-routine adjustment by type."""
        t0 = time.perf_counter()

        # If pre-calculated energy impact is provided, use it directly
        if nr_input.energy_impact_kwh is not None:
            energy_impact = nr_input.energy_impact_kwh
            method = "pre_calculated"
        else:
            energy_impact, method = self._compute_nra_energy(nr_input)

        energy_impact, is_capped = self._cap_adjustment(
            energy_impact, adj_config.baseline_energy
        )

        direction = "increase" if energy_impact >= Decimal("0") else "decrease"
        adj_pct = _safe_pct(abs(energy_impact), adj_config.baseline_energy)
        unc_absolute = abs(energy_impact) * nr_input.uncertainty_pct / Decimal("100")

        record = AdjustmentRecord(
            category=AdjustmentCategory.NON_ROUTINE,
            adjustment_type=nr_input.adjustment_type.value,
            description=nr_input.description or f"Non-routine: {nr_input.adjustment_type.value}",
            energy_adjustment=_round_val(energy_impact, 2),
            adjustment_pct=_round_val(adj_pct, 4),
            direction=direction,
            uncertainty_pct=nr_input.uncertainty_pct,
            uncertainty_level=nr_input.uncertainty_level,
            uncertainty_absolute=_round_val(unc_absolute, 2),
            justification=nr_input.justification,
            source_document=nr_input.source_document,
            effective_date=nr_input.effective_date,
            calculation_method=method,
            input_parameters=self._nra_input_params(nr_input),
            is_capped=is_capped,
        )
        record.provenance_hash = _compute_hash(record)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "NRA %s: %.1f kWh (%s), unc=%.1f%%, hash=%s (%.1f ms)",
            nr_input.adjustment_type.value, float(energy_impact),
            direction, float(nr_input.uncertainty_pct),
            record.provenance_hash[:16], elapsed,
        )
        return record

    def _compute_nra_energy(
        self,
        nr_input: NonRoutineAdjustmentInput,
    ) -> Tuple[Decimal, str]:
        """Compute energy impact for a non-routine adjustment by type."""
        adj_type = nr_input.adjustment_type

        if adj_type == NonRoutineAdjustmentType.FLOOR_AREA_CHANGE:
            # E_adj = delta_area * EUI
            energy = nr_input.delta_floor_area_sqft * nr_input.baseline_eui
            return energy, "floor_area_eui"

        elif adj_type == NonRoutineAdjustmentType.EQUIPMENT_ADDITION:
            # E_adj = rated_power * hours * load_factor
            lf = self._get_load_factor(nr_input)
            energy = nr_input.rated_power_kw * nr_input.operating_hours * lf
            return energy, "equipment_engineering_estimate"

        elif adj_type == NonRoutineAdjustmentType.EQUIPMENT_REMOVAL:
            # E_adj = -(rated_power * hours * load_factor)
            lf = self._get_load_factor(nr_input)
            energy = -(nr_input.rated_power_kw * nr_input.operating_hours * lf)
            return energy, "equipment_engineering_estimate"

        elif adj_type == NonRoutineAdjustmentType.SCHEDULE_CHANGE:
            # E_adj = delta_hours * avg_load * days
            energy = (
                nr_input.delta_hours_per_day
                * nr_input.average_load_kw
                * _decimal(nr_input.days_affected)
            )
            return energy, "schedule_delta"

        elif adj_type == NonRoutineAdjustmentType.STATIC_FACTOR:
            return nr_input.static_adjustment_value, "static_value"

        elif adj_type == NonRoutineAdjustmentType.FUEL_SWITCH:
            return nr_input.static_adjustment_value, "fuel_switch_estimate"

        elif adj_type == NonRoutineAdjustmentType.PROCESS_CHANGE:
            return nr_input.static_adjustment_value, "process_change_estimate"

        elif adj_type == NonRoutineAdjustmentType.ENVELOPE_CHANGE:
            return nr_input.static_adjustment_value, "envelope_change_estimate"

        elif adj_type == NonRoutineAdjustmentType.CUSTOM_NON_ROUTINE:
            return nr_input.static_adjustment_value, "custom"

        return Decimal("0"), "unknown"

    def _get_load_factor(
        self,
        nr_input: NonRoutineAdjustmentInput,
    ) -> Decimal:
        """Get equipment load factor from input or lookup table."""
        if nr_input.load_factor is not None:
            return nr_input.load_factor
        return EQUIPMENT_LOAD_FACTORS.get(
            nr_input.equipment_category, self._default_load_factor
        )

    def _nra_input_params(
        self,
        nr_input: NonRoutineAdjustmentInput,
    ) -> Dict[str, str]:
        """Extract key input parameters for documentation."""
        params: Dict[str, str] = {
            "adjustment_type": nr_input.adjustment_type.value,
        }
        if nr_input.delta_floor_area_sqft != Decimal("0"):
            params["delta_floor_area_sqft"] = str(nr_input.delta_floor_area_sqft)
        if nr_input.rated_power_kw > Decimal("0"):
            params["rated_power_kw"] = str(nr_input.rated_power_kw)
        if nr_input.operating_hours > Decimal("0"):
            params["operating_hours"] = str(nr_input.operating_hours)
        if nr_input.load_factor is not None:
            params["load_factor"] = str(nr_input.load_factor)
        if nr_input.delta_hours_per_day != Decimal("0"):
            params["delta_hours_per_day"] = str(nr_input.delta_hours_per_day)
        if nr_input.average_load_kw > Decimal("0"):
            params["average_load_kw"] = str(nr_input.average_load_kw)
        if nr_input.static_adjustment_value != Decimal("0"):
            params["static_adjustment_value"] = str(nr_input.static_adjustment_value)
        return params

    # ------------------------------------------------------------------ #
    # Private: Capping & Validation                                        #
    # ------------------------------------------------------------------ #

    def _cap_adjustment(
        self,
        adjustment: Decimal,
        baseline_energy: Decimal,
    ) -> Tuple[Decimal, bool]:
        """Cap an adjustment at the maximum allowed percentage of baseline."""
        if baseline_energy == Decimal("0"):
            return adjustment, False

        max_allowed = baseline_energy * self._max_adj_pct / Decimal("100")
        if abs(adjustment) > max_allowed:
            logger.warning(
                "Adjustment %.1f exceeds max (%.1f = %.0f%% of baseline), capping",
                float(adjustment), float(max_allowed), float(self._max_adj_pct),
            )
            capped = max_allowed if adjustment > Decimal("0") else -max_allowed
            return capped, True
        return adjustment, False

    # ------------------------------------------------------------------ #
    # Private: Warnings & Recommendations                                  #
    # ------------------------------------------------------------------ #

    def _generate_warnings(
        self,
        total_adjustment: Decimal,
        baseline_energy: Decimal,
        adjusted_baseline: Decimal,
        records: List[AdjustmentRecord],
    ) -> List[str]:
        """Generate warnings for the adjustment calculation."""
        warnings: List[str] = []

        if baseline_energy > Decimal("0"):
            adj_pct = float(_safe_pct(abs(total_adjustment), baseline_energy))
            if adj_pct > 30:
                warnings.append(
                    f"Total adjustment is {adj_pct:.1f}% of baseline energy, "
                    "which is unusually large. Review adjustment inputs."
                )

        if adjusted_baseline < Decimal("0"):
            warnings.append(
                "Adjusted baseline energy is negative, indicating "
                "adjustment inputs may be incorrect."
            )

        capped = [r for r in records if r.is_capped]
        if capped:
            warnings.append(
                f"{len(capped)} adjustment(s) were capped at the maximum "
                f"allowed percentage ({float(self._max_adj_pct):.0f}%)."
            )

        high_unc = [
            r for r in records
            if r.uncertainty_level == UncertaintyLevel.HIGH
        ]
        if high_unc:
            warnings.append(
                f"{len(high_unc)} adjustment(s) have HIGH uncertainty. "
                "Consider metering or better documentation."
            )

        return warnings

    def _generate_recommendations(
        self,
        routine_result: RoutineAdjustmentResult,
        non_routine_result: Optional[NonRoutineAdjustmentResult],
        adj_config: AdjustmentConfig,
    ) -> List[str]:
        """Generate recommendations for the adjustment calculation."""
        recs: List[str] = []

        if routine_result.weather_adjustment == Decimal("0"):
            recs.append(
                "No weather adjustment applied. Consider adding weather "
                "normalisation if the facility is weather-dependent."
            )

        if non_routine_result and non_routine_result.n_adjustments == 0:
            recs.append(
                "No non-routine adjustments applied. Verify that no "
                "non-routine changes occurred during the reporting period."
            )

        if non_routine_result:
            for rec in non_routine_result.adjustment_records:
                if (not rec.justification
                        and rec.category == AdjustmentCategory.NON_ROUTINE):
                    recs.append(
                        f"Non-routine adjustment '{rec.adjustment_type}' "
                        "lacks justification. IPMVP requires documented "
                        "justification for all non-routine adjustments."
                    )
                    break  # One warning is sufficient

        return recs
