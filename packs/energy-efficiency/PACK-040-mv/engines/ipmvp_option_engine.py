# -*- coding: utf-8 -*-
"""
IPMVPOptionEngine - PACK-040 M&V Engine 5
============================================

IPMVP Options A/B/C/D implementation engine for Measurement &
Verification.  Implements all four IPMVP options with automated option
selection based on ECM type, measurement boundary, cost-effectiveness,
and accuracy requirements.  Generates option comparison matrices and
suitability assessments.

IPMVP Options:
    Option A - Retrofit Isolation: Key Parameter Measurement
        - Isolates the retrofit from the rest of the facility
        - Measures key parameter(s) that define energy use of the ECM
        - Stipulates values for non-measured parameters
        - Best for single systems with one dominant variable
        - Example: Lighting retrofit (measure hours, stipulate wattage)

    Option B - Retrofit Isolation: All Parameter Measurement
        - Isolates the retrofit from the rest of the facility
        - Measures all parameters needed to determine energy use
        - Short-term or continuous metering at the retrofit boundary
        - Best for complex retrofits where stipulation is unreliable
        - Example: Chiller replacement (measure kW and tons continuously)

    Option C - Whole Facility
        - Uses utility meter data for the whole facility
        - Compares baseline model prediction to reporting-period actual
        - Requires regression model (typically weather-based)
        - Best when ECM affects >10% of total facility energy
        - Example: Multiple ECMs implemented simultaneously

    Option D - Calibrated Simulation
        - Uses calibrated energy simulation (DOE-2, EnergyPlus)
        - Baseline and post-retrofit simulated energy compared
        - Calibrated to actual utility data per ASHRAE 14 criteria
        - Best for new construction or complex multi-system retrofits
        - Example: Deep energy retrofit of entire building envelope + HVAC

    Automated Option Selection:
        Score = w1*suitability + w2*accuracy + w3*(1-cost) + w4*complexity
        where weights are configurable per facility type.

Regulatory References:
    - IPMVP Core Concepts 2022, Chapters 3-4 (Options A-D)
    - ASHRAE Guideline 14-2014, Section 5 (M&V Approaches)
    - ISO 50015:2014, Annex A (M&V Methods)
    - FEMP M&V Guidelines 4.0, Chapters 3-4
    - DOE-2 / EnergyPlus calibration criteria (ASHRAE 14, Section 5.3.2.4)

Zero-Hallucination:
    - Option selection via deterministic scoring matrix
    - Suitability rules are explicit, documented decision trees
    - No LLM involvement in any calculation or selection path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-040 M&V
Engine:  5 of 10
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

class IPMVPOption(str, Enum):
    """IPMVP Option designator.

    OPTION_A:  Retrofit Isolation - Key Parameter Measurement.
    OPTION_B:  Retrofit Isolation - All Parameter Measurement.
    OPTION_C:  Whole Facility.
    OPTION_D:  Calibrated Simulation.
    """
    OPTION_A = "option_a"
    OPTION_B = "option_b"
    OPTION_C = "option_c"
    OPTION_D = "option_d"

class ECMType(str, Enum):
    """Energy Conservation Measure type.

    LIGHTING:           Lighting retrofit / LED upgrade.
    HVAC_EQUIPMENT:     HVAC equipment replacement (chiller, boiler, RTU).
    HVAC_CONTROLS:      HVAC controls upgrade / BAS.
    ENVELOPE:           Building envelope improvement.
    MOTORS_DRIVES:      Motor and VFD upgrades.
    COMPRESSED_AIR:     Compressed air system improvements.
    PROCESS:            Process efficiency improvements.
    COMBINED:           Multiple ECM types combined.
    RETRO_COMMISSIONING: Retro-commissioning / operational improvements.
    RENEWABLE:          On-site renewable energy (solar, wind).
    COGENERATION:       Combined heat and power (CHP).
    BEHAVIOURAL:        Behavioural / occupant engagement.
    NEW_CONSTRUCTION:   New construction / major renovation.
    """
    LIGHTING = "lighting"
    HVAC_EQUIPMENT = "hvac_equipment"
    HVAC_CONTROLS = "hvac_controls"
    ENVELOPE = "envelope"
    MOTORS_DRIVES = "motors_drives"
    COMPRESSED_AIR = "compressed_air"
    PROCESS = "process"
    COMBINED = "combined"
    RETRO_COMMISSIONING = "retro_commissioning"
    RENEWABLE = "renewable"
    COGENERATION = "cogeneration"
    BEHAVIOURAL = "behavioural"
    NEW_CONSTRUCTION = "new_construction"

class MeasurementBoundary(str, Enum):
    """Measurement boundary for M&V.

    RETROFIT_ISOLATION:  Around the specific retrofit.
    WHOLE_FACILITY:      Around the entire facility.
    SUB_SYSTEM:          Around a specific sub-system.
    EQUIPMENT:           Around individual equipment.
    """
    RETROFIT_ISOLATION = "retrofit_isolation"
    WHOLE_FACILITY = "whole_facility"
    SUB_SYSTEM = "sub_system"
    EQUIPMENT = "equipment"

class MeteringApproach(str, Enum):
    """Metering approach for the selected option.

    KEY_PARAMETER:  Measure only the key parameter(s).
    ALL_PARAMETERS: Measure all parameters affecting energy use.
    UTILITY_METER:  Use existing utility meter(s).
    SIMULATION:     Calibrated energy simulation model.
    SPOT_MEASURE:   Short-term spot measurements.
    CONTINUOUS:     Continuous monitoring/metering.
    """
    KEY_PARAMETER = "key_parameter"
    ALL_PARAMETERS = "all_parameters"
    UTILITY_METER = "utility_meter"
    SIMULATION = "simulation"
    SPOT_MEASURE = "spot_measure"
    CONTINUOUS = "continuous"

class CostLevel(str, Enum):
    """Relative cost level for M&V implementation.

    VERY_LOW:   <$1,000 M&V cost.
    LOW:        $1,000-$5,000 M&V cost.
    MEDIUM:     $5,000-$20,000 M&V cost.
    HIGH:       $20,000-$50,000 M&V cost.
    VERY_HIGH:  >$50,000 M&V cost.
    """
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class AccuracyLevel(str, Enum):
    """Relative accuracy level of the M&V option.

    LOW:     Uncertainty >50%.
    MEDIUM:  Uncertainty 25-50%.
    HIGH:    Uncertainty 10-25%.
    VERY_HIGH: Uncertainty <10%.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class SelectionConfidence(str, Enum):
    """Confidence level in the automated option recommendation.

    HIGH:    Strong match, clear best option.
    MEDIUM:  Good match, but alternatives viable.
    LOW:     Marginal, multiple options similarly scored.
    MANUAL:  Cannot auto-select, manual review required.
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MANUAL = "manual"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default scoring weights for option selection.
DEFAULT_WEIGHTS: Dict[str, Decimal] = {
    "suitability": Decimal("0.35"),
    "accuracy": Decimal("0.25"),
    "cost_effectiveness": Decimal("0.25"),
    "complexity": Decimal("0.15"),
}

# Option characteristics: (cost_level, accuracy_level, complexity_factor).
OPTION_CHARACTERISTICS: Dict[str, Dict[str, Any]] = {
    IPMVPOption.OPTION_A.value: {
        "cost": CostLevel.LOW,
        "accuracy": AccuracyLevel.MEDIUM,
        "complexity": Decimal("0.3"),
        "cost_score": Decimal("0.80"),
        "accuracy_score": Decimal("0.50"),
        "complexity_score": Decimal("0.80"),
        "boundary": MeasurementBoundary.RETROFIT_ISOLATION,
        "metering": MeteringApproach.KEY_PARAMETER,
        "stipulation_allowed": True,
        "min_savings_fraction": Decimal("0"),
    },
    IPMVPOption.OPTION_B.value: {
        "cost": CostLevel.MEDIUM,
        "accuracy": AccuracyLevel.HIGH,
        "complexity": Decimal("0.5"),
        "cost_score": Decimal("0.50"),
        "accuracy_score": Decimal("0.80"),
        "complexity_score": Decimal("0.50"),
        "boundary": MeasurementBoundary.RETROFIT_ISOLATION,
        "metering": MeteringApproach.ALL_PARAMETERS,
        "stipulation_allowed": False,
        "min_savings_fraction": Decimal("0"),
    },
    IPMVPOption.OPTION_C.value: {
        "cost": CostLevel.LOW,
        "accuracy": AccuracyLevel.MEDIUM,
        "complexity": Decimal("0.4"),
        "cost_score": Decimal("0.70"),
        "accuracy_score": Decimal("0.60"),
        "complexity_score": Decimal("0.60"),
        "boundary": MeasurementBoundary.WHOLE_FACILITY,
        "metering": MeteringApproach.UTILITY_METER,
        "stipulation_allowed": False,
        "min_savings_fraction": Decimal("0.10"),
    },
    IPMVPOption.OPTION_D.value: {
        "cost": CostLevel.HIGH,
        "accuracy": AccuracyLevel.VERY_HIGH,
        "complexity": Decimal("0.9"),
        "cost_score": Decimal("0.20"),
        "accuracy_score": Decimal("0.90"),
        "complexity_score": Decimal("0.20"),
        "boundary": MeasurementBoundary.WHOLE_FACILITY,
        "metering": MeteringApproach.SIMULATION,
        "stipulation_allowed": False,
        "min_savings_fraction": Decimal("0"),
    },
}

# ECM type to recommended options mapping.
ECM_OPTION_SUITABILITY: Dict[str, Dict[str, Decimal]] = {
    ECMType.LIGHTING.value: {
        IPMVPOption.OPTION_A.value: Decimal("0.90"),
        IPMVPOption.OPTION_B.value: Decimal("0.60"),
        IPMVPOption.OPTION_C.value: Decimal("0.50"),
        IPMVPOption.OPTION_D.value: Decimal("0.20"),
    },
    ECMType.HVAC_EQUIPMENT.value: {
        IPMVPOption.OPTION_A.value: Decimal("0.40"),
        IPMVPOption.OPTION_B.value: Decimal("0.85"),
        IPMVPOption.OPTION_C.value: Decimal("0.70"),
        IPMVPOption.OPTION_D.value: Decimal("0.50"),
    },
    ECMType.HVAC_CONTROLS.value: {
        IPMVPOption.OPTION_A.value: Decimal("0.30"),
        IPMVPOption.OPTION_B.value: Decimal("0.60"),
        IPMVPOption.OPTION_C.value: Decimal("0.80"),
        IPMVPOption.OPTION_D.value: Decimal("0.50"),
    },
    ECMType.ENVELOPE.value: {
        IPMVPOption.OPTION_A.value: Decimal("0.20"),
        IPMVPOption.OPTION_B.value: Decimal("0.40"),
        IPMVPOption.OPTION_C.value: Decimal("0.75"),
        IPMVPOption.OPTION_D.value: Decimal("0.85"),
    },
    ECMType.MOTORS_DRIVES.value: {
        IPMVPOption.OPTION_A.value: Decimal("0.70"),
        IPMVPOption.OPTION_B.value: Decimal("0.85"),
        IPMVPOption.OPTION_C.value: Decimal("0.40"),
        IPMVPOption.OPTION_D.value: Decimal("0.30"),
    },
    ECMType.COMPRESSED_AIR.value: {
        IPMVPOption.OPTION_A.value: Decimal("0.50"),
        IPMVPOption.OPTION_B.value: Decimal("0.80"),
        IPMVPOption.OPTION_C.value: Decimal("0.40"),
        IPMVPOption.OPTION_D.value: Decimal("0.30"),
    },
    ECMType.PROCESS.value: {
        IPMVPOption.OPTION_A.value: Decimal("0.50"),
        IPMVPOption.OPTION_B.value: Decimal("0.80"),
        IPMVPOption.OPTION_C.value: Decimal("0.30"),
        IPMVPOption.OPTION_D.value: Decimal("0.40"),
    },
    ECMType.COMBINED.value: {
        IPMVPOption.OPTION_A.value: Decimal("0.20"),
        IPMVPOption.OPTION_B.value: Decimal("0.30"),
        IPMVPOption.OPTION_C.value: Decimal("0.85"),
        IPMVPOption.OPTION_D.value: Decimal("0.60"),
    },
    ECMType.RETRO_COMMISSIONING.value: {
        IPMVPOption.OPTION_A.value: Decimal("0.20"),
        IPMVPOption.OPTION_B.value: Decimal("0.30"),
        IPMVPOption.OPTION_C.value: Decimal("0.85"),
        IPMVPOption.OPTION_D.value: Decimal("0.50"),
    },
    ECMType.RENEWABLE.value: {
        IPMVPOption.OPTION_A.value: Decimal("0.60"),
        IPMVPOption.OPTION_B.value: Decimal("0.80"),
        IPMVPOption.OPTION_C.value: Decimal("0.40"),
        IPMVPOption.OPTION_D.value: Decimal("0.30"),
    },
    ECMType.COGENERATION.value: {
        IPMVPOption.OPTION_A.value: Decimal("0.30"),
        IPMVPOption.OPTION_B.value: Decimal("0.85"),
        IPMVPOption.OPTION_C.value: Decimal("0.40"),
        IPMVPOption.OPTION_D.value: Decimal("0.50"),
    },
    ECMType.BEHAVIOURAL.value: {
        IPMVPOption.OPTION_A.value: Decimal("0.10"),
        IPMVPOption.OPTION_B.value: Decimal("0.20"),
        IPMVPOption.OPTION_C.value: Decimal("0.90"),
        IPMVPOption.OPTION_D.value: Decimal("0.30"),
    },
    ECMType.NEW_CONSTRUCTION.value: {
        IPMVPOption.OPTION_A.value: Decimal("0.10"),
        IPMVPOption.OPTION_B.value: Decimal("0.20"),
        IPMVPOption.OPTION_C.value: Decimal("0.30"),
        IPMVPOption.OPTION_D.value: Decimal("0.95"),
    },
}

# DOE-2/EnergyPlus calibration criteria (ASHRAE 14, Section 5.3.2.4).
SIMULATION_CALIBRATION_CRITERIA: Dict[str, Dict[str, Decimal]] = {
    "monthly": {
        "cvrmse_max": Decimal("15"),
        "nmbe_max": Decimal("5"),
    },
    "hourly": {
        "cvrmse_max": Decimal("30"),
        "nmbe_max": Decimal("10"),
    },
}

# M&V cost as percentage of project cost (typical ranges).
MV_COST_RANGES: Dict[str, Tuple[Decimal, Decimal]] = {
    IPMVPOption.OPTION_A.value: (Decimal("1"), Decimal("3")),
    IPMVPOption.OPTION_B.value: (Decimal("3"), Decimal("10")),
    IPMVPOption.OPTION_C.value: (Decimal("1"), Decimal("5")),
    IPMVPOption.OPTION_D.value: (Decimal("5"), Decimal("15")),
}

# Confidence threshold for selection confidence rating.
CONFIDENCE_HIGH_THRESHOLD: Decimal = Decimal("0.15")
CONFIDENCE_MEDIUM_THRESHOLD: Decimal = Decimal("0.08")

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class ECMDescription(BaseModel):
    """Description of an Energy Conservation Measure for option selection.

    Attributes:
        ecm_id: ECM identifier.
        ecm_name: Human-readable name.
        ecm_type: Type of ECM.
        description: Detailed description.
        estimated_savings_kwh: Estimated annual savings.
        estimated_savings_pct: Estimated savings as % of facility.
        total_facility_energy_kwh: Total facility annual energy.
        project_cost: Total project cost (for M&V cost ratio).
        affected_systems: List of affected building systems.
        interactive_effects: Whether ECM has interactive effects.
        is_weather_dependent: Whether savings are weather-dependent.
        is_production_dependent: Whether savings are production-dependent.
        can_isolate_retrofit: Whether retrofit can be isolated.
        number_of_ecms: Number of ECMs being implemented.
        measurement_boundary_preference: Preferred boundary.
        existing_meters: List of existing meter types.
        simulation_model_available: Whether an energy model exists.
        budget_constraint: M&V budget constraint.
        accuracy_requirement: Minimum accuracy requirement.
    """
    ecm_id: str = Field(default="", description="ECM ID")
    ecm_name: str = Field(
        default="", max_length=500, description="ECM name"
    )
    ecm_type: ECMType = Field(
        default=ECMType.COMBINED, description="ECM type"
    )
    description: str = Field(
        default="", max_length=2000, description="Description"
    )
    estimated_savings_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Estimated savings (kWh)"
    )
    estimated_savings_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
        description="Savings as % of facility"
    )
    total_facility_energy_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total facility energy"
    )
    project_cost: Decimal = Field(
        default=Decimal("0"), ge=0, description="Project cost ($)"
    )
    affected_systems: List[str] = Field(
        default_factory=list, description="Affected systems"
    )
    interactive_effects: bool = Field(
        default=False, description="Has interactive effects"
    )
    is_weather_dependent: bool = Field(
        default=False, description="Weather-dependent savings"
    )
    is_production_dependent: bool = Field(
        default=False, description="Production-dependent savings"
    )
    can_isolate_retrofit: bool = Field(
        default=True, description="Can isolate retrofit"
    )
    number_of_ecms: int = Field(
        default=1, ge=1, description="Number of ECMs"
    )
    measurement_boundary_preference: Optional[MeasurementBoundary] = Field(
        default=None, description="Preferred boundary"
    )
    existing_meters: List[str] = Field(
        default_factory=list, description="Existing meter types"
    )
    simulation_model_available: bool = Field(
        default=False, description="Energy model available"
    )
    budget_constraint: Optional[Decimal] = Field(
        default=None, ge=0, description="M&V budget ($)"
    )
    accuracy_requirement: Optional[AccuracyLevel] = Field(
        default=None, description="Min accuracy level"
    )

class StipulatedValue(BaseModel):
    """A stipulated (non-measured) value for Option A.

    Attributes:
        parameter_name: Name of the stipulated parameter.
        stipulated_value: Stipulated value.
        unit: Unit of measurement.
        source: Source of stipulation (manufacturer, design, etc.).
        confidence_pct: Confidence in stipulated value.
        justification: Engineering justification.
        review_frequency: How often to review/verify.
    """
    parameter_name: str = Field(
        default="", description="Parameter name"
    )
    stipulated_value: Decimal = Field(
        default=Decimal("0"), description="Stipulated value"
    )
    unit: str = Field(default="", description="Unit")
    source: str = Field(default="", description="Source")
    confidence_pct: Decimal = Field(
        default=Decimal("80"), ge=0, le=100,
        description="Confidence %"
    )
    justification: str = Field(
        default="", max_length=1000, description="Justification"
    )
    review_frequency: str = Field(
        default="annual", description="Review frequency"
    )

class SimulationCalibrationData(BaseModel):
    """Calibration data for Option D simulation verification.

    Attributes:
        software: Simulation software (DOE-2, EnergyPlus, etc.).
        version: Software version.
        calibration_level: Monthly or hourly calibration.
        cvrmse_pct: Achieved CVRMSE.
        nmbe_pct: Achieved NMBE.
        months_of_data: Months of calibration data.
        weather_file: Weather file used.
        is_calibrated: Whether model meets calibration criteria.
    """
    software: str = Field(default="", description="Simulation software")
    version: str = Field(default="", description="Software version")
    calibration_level: str = Field(
        default="monthly", description="Calibration level"
    )
    cvrmse_pct: Decimal = Field(
        default=Decimal("0"), ge=0, description="CVRMSE %"
    )
    nmbe_pct: Decimal = Field(
        default=Decimal("0"), description="NMBE %"
    )
    months_of_data: int = Field(
        default=12, ge=1, description="Months of data"
    )
    weather_file: str = Field(
        default="", description="Weather file"
    )
    is_calibrated: bool = Field(
        default=False, description="Meets calibration criteria"
    )

class OptionSelectionConfig(BaseModel):
    """Configuration for automated option selection.

    Attributes:
        project_id: M&V project identifier.
        facility_id: Facility identifier.
        facility_name: Facility name.
        scoring_weights: Custom scoring weights.
        exclude_options: Options to exclude from consideration.
        prefer_isolation: Prefer retrofit isolation if possible.
        max_mv_cost_pct: Maximum M&V cost as % of project.
    """
    project_id: str = Field(default="", description="Project ID")
    facility_id: str = Field(default="", description="Facility ID")
    facility_name: str = Field(
        default="", max_length=500, description="Facility name"
    )
    scoring_weights: Dict[str, Decimal] = Field(
        default_factory=lambda: dict(DEFAULT_WEIGHTS),
        description="Scoring weights"
    )
    exclude_options: List[IPMVPOption] = Field(
        default_factory=list, description="Excluded options"
    )
    prefer_isolation: bool = Field(
        default=True, description="Prefer isolation if possible"
    )
    max_mv_cost_pct: Decimal = Field(
        default=Decimal("10"), ge=0, le=100,
        description="Max M&V cost %"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class OptionScoreCard(BaseModel):
    """Score card for a single IPMVP option.

    Attributes:
        option: IPMVP option.
        suitability_score: ECM-type suitability score (0-1).
        accuracy_score: Accuracy score (0-1).
        cost_effectiveness_score: Cost-effectiveness score (0-1).
        complexity_score: Complexity score (0-1, higher = simpler).
        weighted_total: Weighted total score.
        rank: Rank among options (1 = best).
        boundary: Measurement boundary for this option.
        metering_approach: Metering approach.
        cost_level: Relative cost level.
        accuracy_level: Accuracy level.
        stipulation_required: Whether stipulation is needed.
        estimated_mv_cost_low: Low estimate of M&V cost.
        estimated_mv_cost_high: High estimate of M&V cost.
        is_eligible: Whether option is eligible given constraints.
        ineligibility_reasons: Reasons for ineligibility.
        strengths: Key strengths of this option.
        weaknesses: Key weaknesses of this option.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    option: IPMVPOption = Field(default=IPMVPOption.OPTION_A)
    suitability_score: Decimal = Field(default=Decimal("0"))
    accuracy_score: Decimal = Field(default=Decimal("0"))
    cost_effectiveness_score: Decimal = Field(default=Decimal("0"))
    complexity_score: Decimal = Field(default=Decimal("0"))
    weighted_total: Decimal = Field(default=Decimal("0"))
    rank: int = Field(default=0)
    boundary: MeasurementBoundary = Field(
        default=MeasurementBoundary.RETROFIT_ISOLATION
    )
    metering_approach: MeteringApproach = Field(
        default=MeteringApproach.KEY_PARAMETER
    )
    cost_level: CostLevel = Field(default=CostLevel.LOW)
    accuracy_level: AccuracyLevel = Field(default=AccuracyLevel.MEDIUM)
    stipulation_required: bool = Field(default=False)
    estimated_mv_cost_low: Decimal = Field(default=Decimal("0"))
    estimated_mv_cost_high: Decimal = Field(default=Decimal("0"))
    is_eligible: bool = Field(default=True)
    ineligibility_reasons: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class OptionADetail(BaseModel):
    """Detailed Option A implementation specification.

    Attributes:
        key_parameters: Parameters to be measured.
        stipulated_values: Parameters to be stipulated.
        measurement_frequency: How often to measure key parameter.
        spot_vs_continuous: Whether to use spot or continuous measurement.
        estimated_uncertainty_pct: Estimated uncertainty.
        verification_protocol: Verification steps.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    key_parameters: List[str] = Field(default_factory=list)
    stipulated_values: List[StipulatedValue] = Field(default_factory=list)
    measurement_frequency: str = Field(default="")
    spot_vs_continuous: str = Field(default="spot")
    estimated_uncertainty_pct: Decimal = Field(default=Decimal("20"))
    verification_protocol: List[str] = Field(default_factory=list)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class OptionBDetail(BaseModel):
    """Detailed Option B implementation specification.

    Attributes:
        measured_parameters: All parameters to be measured.
        metering_points: Specific metering points.
        measurement_duration: Duration of measurement.
        continuous_vs_short_term: Continuous or short-term.
        data_collection_frequency: Data collection interval.
        estimated_uncertainty_pct: Estimated uncertainty.
        verification_protocol: Verification steps.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    measured_parameters: List[str] = Field(default_factory=list)
    metering_points: List[str] = Field(default_factory=list)
    measurement_duration: str = Field(default="")
    continuous_vs_short_term: str = Field(default="continuous")
    data_collection_frequency: str = Field(default="15-minute")
    estimated_uncertainty_pct: Decimal = Field(default=Decimal("10"))
    verification_protocol: List[str] = Field(default_factory=list)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class OptionCDetail(BaseModel):
    """Detailed Option C implementation specification.

    Attributes:
        utility_meters: Utility meters to be used.
        regression_model_type: Type of regression model.
        independent_variables: Independent variables for model.
        baseline_period_months: Baseline period length (months).
        reporting_frequency: Reporting frequency.
        min_savings_fraction: Minimum ECM savings fraction.
        estimated_uncertainty_pct: Estimated uncertainty.
        verification_protocol: Verification steps.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    utility_meters: List[str] = Field(default_factory=list)
    regression_model_type: str = Field(default="change_point")
    independent_variables: List[str] = Field(default_factory=list)
    baseline_period_months: int = Field(default=12)
    reporting_frequency: str = Field(default="monthly")
    min_savings_fraction: Decimal = Field(default=Decimal("0.10"))
    estimated_uncertainty_pct: Decimal = Field(default=Decimal("15"))
    verification_protocol: List[str] = Field(default_factory=list)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class OptionDDetail(BaseModel):
    """Detailed Option D implementation specification.

    Attributes:
        simulation_software: Simulation software to use.
        calibration_data: Calibration data/criteria.
        model_complexity: Model complexity level.
        calibration_criteria: ASHRAE 14 calibration criteria.
        weather_file_source: Source of weather data.
        estimated_uncertainty_pct: Estimated uncertainty.
        verification_protocol: Verification steps.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    simulation_software: str = Field(default="EnergyPlus")
    calibration_data: Optional[SimulationCalibrationData] = Field(
        default=None
    )
    model_complexity: str = Field(default="detailed")
    calibration_criteria: Dict[str, Decimal] = Field(
        default_factory=lambda: {
            "monthly_cvrmse_max": Decimal("15"),
            "monthly_nmbe_max": Decimal("5"),
            "hourly_cvrmse_max": Decimal("30"),
            "hourly_nmbe_max": Decimal("10"),
        }
    )
    weather_file_source: str = Field(default="TMY3")
    estimated_uncertainty_pct: Decimal = Field(default=Decimal("8"))
    verification_protocol: List[str] = Field(default_factory=list)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

class OptionSelectionResult(BaseModel):
    """Complete IPMVP option selection result.

    Attributes:
        selection_id: Unique selection identifier.
        project_id: M&V project identifier.
        ecm_id: ECM identifier.
        facility_id: Facility identifier.
        facility_name: Facility name.
        ecm_type: ECM type evaluated.
        recommended_option: Recommended IPMVP option.
        selection_confidence: Confidence in recommendation.
        score_cards: Score cards for all evaluated options.
        option_a_detail: Option A implementation detail.
        option_b_detail: Option B implementation detail.
        option_c_detail: Option C implementation detail.
        option_d_detail: Option D implementation detail.
        selection_rationale: Human-readable rationale.
        scoring_weights_used: Weights used for scoring.
        excluded_options: Options excluded from consideration.
        comparison_summary: Summary comparison table.
        warnings: Warnings generated.
        recommendations: Recommendations.
        processing_time_ms: Processing duration.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 audit hash.
    """
    selection_id: str = Field(default_factory=_new_uuid)
    project_id: str = Field(default="")
    ecm_id: str = Field(default="")
    facility_id: str = Field(default="")
    facility_name: str = Field(default="", max_length=500)
    ecm_type: ECMType = Field(default=ECMType.COMBINED)
    recommended_option: Optional[IPMVPOption] = Field(default=None)
    selection_confidence: SelectionConfidence = Field(
        default=SelectionConfidence.MANUAL
    )
    score_cards: List[OptionScoreCard] = Field(default_factory=list)
    option_a_detail: Optional[OptionADetail] = Field(default=None)
    option_b_detail: Optional[OptionBDetail] = Field(default=None)
    option_c_detail: Optional[OptionCDetail] = Field(default=None)
    option_d_detail: Optional[OptionDDetail] = Field(default=None)
    selection_rationale: str = Field(default="")
    scoring_weights_used: Dict[str, Decimal] = Field(default_factory=dict)
    excluded_options: List[str] = Field(default_factory=list)
    comparison_summary: Dict[str, Dict[str, str]] = Field(
        default_factory=dict
    )
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    processing_time_ms: Decimal = Field(default=Decimal("0"))
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class IPMVPOptionEngine:
    """IPMVP Options A/B/C/D implementation and selection engine.

    Evaluates all four IPMVP options against an ECM description,
    scores each option on suitability, accuracy, cost-effectiveness,
    and complexity, and recommends the optimal option.  Generates
    implementation specifications for each option.

    Usage::

        engine = IPMVPOptionEngine()
        ecm = ECMDescription(
            ecm_type=ECMType.LIGHTING,
            estimated_savings_pct=Decimal("15"),
            can_isolate_retrofit=True,
        )
        config = OptionSelectionConfig(project_id="PRJ-001")
        result = engine.select_option(ecm, config)
        print(f"Recommended: {result.recommended_option.value}")
        for sc in result.score_cards:
            print(f"  {sc.option.value}: {float(sc.weighted_total):.3f}")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise IPMVPOptionEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - scoring_weights (dict): custom scoring weights
                - default_accuracy_req (str): default accuracy requirement
        """
        self.config = config or {}
        self._weights = dict(DEFAULT_WEIGHTS)
        if "scoring_weights" in self.config:
            for k, v in self.config["scoring_weights"].items():
                self._weights[k] = _decimal(v)
        logger.info(
            "IPMVPOptionEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def select_option(
        self,
        ecm: ECMDescription,
        selection_config: OptionSelectionConfig,
    ) -> OptionSelectionResult:
        """Select the optimal IPMVP option for an ECM.

        Evaluates all four options, scores each, and recommends the
        best option with implementation details.

        Args:
            ecm: ECM description for evaluation.
            selection_config: Selection configuration.

        Returns:
            OptionSelectionResult with recommendation and comparisons.
        """
        t0 = time.perf_counter()
        logger.info(
            "Selecting IPMVP option: ECM=%s (%s), type=%s",
            ecm.ecm_id, ecm.ecm_name, ecm.ecm_type.value,
        )

        weights = dict(selection_config.scoring_weights or self._weights)
        excluded = [o.value for o in selection_config.exclude_options]

        # Score all options
        score_cards: List[OptionScoreCard] = []
        for option in IPMVPOption:
            if option.value in excluded:
                continue
            card = self._score_option(option, ecm, selection_config, weights)
            score_cards.append(card)

        # Rank eligible options
        eligible = [sc for sc in score_cards if sc.is_eligible]
        eligible.sort(
            key=lambda sc: float(sc.weighted_total), reverse=True
        )
        for rank, sc in enumerate(eligible, 1):
            sc.rank = rank

        # Assign ranks to ineligible
        ineligible = [sc for sc in score_cards if not sc.is_eligible]
        for sc in ineligible:
            sc.rank = len(eligible) + 1

        # Determine recommendation
        recommended: Optional[IPMVPOption] = None
        confidence = SelectionConfidence.MANUAL
        if eligible:
            recommended = eligible[0].option
            confidence = self._assess_confidence(eligible)

        # Generate implementation details
        option_a_detail = self._build_option_a_detail(ecm)
        option_b_detail = self._build_option_b_detail(ecm)
        option_c_detail = self._build_option_c_detail(ecm)
        option_d_detail = self._build_option_d_detail(ecm)

        # Build comparison summary
        comparison = self._build_comparison_summary(score_cards)

        # Build rationale
        rationale = self._build_rationale(
            recommended, eligible, ecm, confidence
        )

        # Warnings and recommendations
        warnings = self._generate_warnings(
            recommended, score_cards, ecm, selection_config
        )
        recommendations = self._generate_recommendations(
            recommended, eligible, ecm
        )

        elapsed_ms = _decimal((time.perf_counter() - t0) * 1000.0)

        result = OptionSelectionResult(
            project_id=selection_config.project_id,
            ecm_id=ecm.ecm_id,
            facility_id=selection_config.facility_id,
            facility_name=selection_config.facility_name,
            ecm_type=ecm.ecm_type,
            recommended_option=recommended,
            selection_confidence=confidence,
            score_cards=score_cards,
            option_a_detail=option_a_detail,
            option_b_detail=option_b_detail,
            option_c_detail=option_c_detail,
            option_d_detail=option_d_detail,
            selection_rationale=rationale,
            scoring_weights_used=weights,
            excluded_options=excluded,
            comparison_summary=comparison,
            warnings=warnings,
            recommendations=recommendations,
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Option selected: %s (confidence=%s, score=%.3f), "
            "hash=%s (%.1f ms)",
            recommended.value if recommended else "none",
            confidence.value,
            float(eligible[0].weighted_total) if eligible else 0.0,
            result.provenance_hash[:16], float(elapsed_ms),
        )
        return result

    def evaluate_option_a(
        self,
        ecm: ECMDescription,
        stipulated_values: Optional[List[StipulatedValue]] = None,
    ) -> OptionADetail:
        """Generate detailed Option A implementation specification.

        Args:
            ecm: ECM description.
            stipulated_values: Optional pre-defined stipulated values.

        Returns:
            OptionADetail with implementation specification.
        """
        t0 = time.perf_counter()
        logger.info("Evaluating Option A for ECM %s", ecm.ecm_id)

        detail = self._build_option_a_detail(ecm, stipulated_values)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Option A evaluated: %d key params, %d stipulated, "
            "hash=%s (%.1f ms)",
            len(detail.key_parameters), len(detail.stipulated_values),
            detail.provenance_hash[:16], elapsed,
        )
        return detail

    def evaluate_option_b(
        self,
        ecm: ECMDescription,
    ) -> OptionBDetail:
        """Generate detailed Option B implementation specification.

        Args:
            ecm: ECM description.

        Returns:
            OptionBDetail with implementation specification.
        """
        t0 = time.perf_counter()
        logger.info("Evaluating Option B for ECM %s", ecm.ecm_id)

        detail = self._build_option_b_detail(ecm)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Option B evaluated: %d params, hash=%s (%.1f ms)",
            len(detail.measured_parameters),
            detail.provenance_hash[:16], elapsed,
        )
        return detail

    def evaluate_option_c(
        self,
        ecm: ECMDescription,
    ) -> OptionCDetail:
        """Generate detailed Option C implementation specification.

        Args:
            ecm: ECM description.

        Returns:
            OptionCDetail with implementation specification.
        """
        t0 = time.perf_counter()
        logger.info("Evaluating Option C for ECM %s", ecm.ecm_id)

        detail = self._build_option_c_detail(ecm)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Option C evaluated: model=%s, hash=%s (%.1f ms)",
            detail.regression_model_type,
            detail.provenance_hash[:16], elapsed,
        )
        return detail

    def evaluate_option_d(
        self,
        ecm: ECMDescription,
        calibration_data: Optional[SimulationCalibrationData] = None,
    ) -> OptionDDetail:
        """Generate detailed Option D implementation specification.

        Args:
            ecm: ECM description.
            calibration_data: Optional calibration data.

        Returns:
            OptionDDetail with implementation specification.
        """
        t0 = time.perf_counter()
        logger.info("Evaluating Option D for ECM %s", ecm.ecm_id)

        detail = self._build_option_d_detail(ecm, calibration_data)

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Option D evaluated: software=%s, hash=%s (%.1f ms)",
            detail.simulation_software,
            detail.provenance_hash[:16], elapsed,
        )
        return detail

    def validate_simulation_calibration(
        self,
        calibration_data: SimulationCalibrationData,
    ) -> Tuple[bool, List[str]]:
        """Validate simulation model calibration against ASHRAE 14 criteria.

        Args:
            calibration_data: Simulation calibration data.

        Returns:
            Tuple of (is_calibrated, list_of_messages).
        """
        t0 = time.perf_counter()
        logger.info("Validating simulation calibration")

        messages: List[str] = []
        is_calibrated = True

        level = calibration_data.calibration_level.lower()
        criteria = SIMULATION_CALIBRATION_CRITERIA.get(level, {})

        cvrmse_max = criteria.get("cvrmse_max", Decimal("15"))
        nmbe_max = criteria.get("nmbe_max", Decimal("5"))

        if abs(calibration_data.cvrmse_pct) > cvrmse_max:
            is_calibrated = False
            messages.append(
                f"CVRMSE ({float(calibration_data.cvrmse_pct):.1f}%) exceeds "
                f"{level} limit ({float(cvrmse_max):.0f}%)."
            )
        else:
            messages.append(
                f"CVRMSE ({float(calibration_data.cvrmse_pct):.1f}%) meets "
                f"{level} criterion (<={float(cvrmse_max):.0f}%)."
            )

        if abs(calibration_data.nmbe_pct) > nmbe_max:
            is_calibrated = False
            messages.append(
                f"NMBE ({float(calibration_data.nmbe_pct):.1f}%) exceeds "
                f"{level} limit (+/-{float(nmbe_max):.0f}%)."
            )
        else:
            messages.append(
                f"NMBE ({float(calibration_data.nmbe_pct):.1f}%) meets "
                f"{level} criterion (+/-{float(nmbe_max):.0f}%)."
            )

        if calibration_data.months_of_data < 12:
            messages.append(
                f"Only {calibration_data.months_of_data} months of data. "
                "ASHRAE 14 recommends 12 months for calibration."
            )

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Calibration validation: %s (%d messages) (%.1f ms)",
            "PASS" if is_calibrated else "FAIL",
            len(messages), elapsed,
        )
        return is_calibrated, messages

    def generate_comparison_matrix(
        self,
        ecm: ECMDescription,
        selection_config: Optional[OptionSelectionConfig] = None,
    ) -> Dict[str, Dict[str, str]]:
        """Generate a comparison matrix for all IPMVP options.

        Args:
            ecm: ECM description.
            selection_config: Optional selection configuration.

        Returns:
            Nested dict of {option: {aspect: value}}.
        """
        t0 = time.perf_counter()
        logger.info("Generating comparison matrix")

        config = selection_config or OptionSelectionConfig()
        result = self.select_option(ecm, config)

        matrix = result.comparison_summary

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Comparison matrix generated: %d options (%.1f ms)",
            len(matrix), elapsed,
        )
        return matrix

    # ------------------------------------------------------------------ #
    # Private: Scoring                                                     #
    # ------------------------------------------------------------------ #

    def _score_option(
        self,
        option: IPMVPOption,
        ecm: ECMDescription,
        config: OptionSelectionConfig,
        weights: Dict[str, Decimal],
    ) -> OptionScoreCard:
        """Score a single option against the ECM."""
        chars = OPTION_CHARACTERISTICS.get(option.value, {})
        ecm_suitability = ECM_OPTION_SUITABILITY.get(
            ecm.ecm_type.value, {}
        )

        # Suitability score
        suitability = ecm_suitability.get(option.value, Decimal("0.50"))

        # Adjust suitability based on additional factors
        suitability = self._adjust_suitability(
            suitability, option, ecm, chars
        )

        # Accuracy score
        accuracy = _decimal(chars.get("accuracy_score", Decimal("0.50")))

        # Cost-effectiveness score
        cost_eff = _decimal(chars.get("cost_score", Decimal("0.50")))

        # Complexity score (higher = simpler)
        complexity = _decimal(chars.get("complexity_score", Decimal("0.50")))

        # Weighted total
        w_suit = _decimal(weights.get("suitability", Decimal("0.35")))
        w_acc = _decimal(weights.get("accuracy", Decimal("0.25")))
        w_cost = _decimal(weights.get("cost_effectiveness", Decimal("0.25")))
        w_comp = _decimal(weights.get("complexity", Decimal("0.15")))

        total = (
            suitability * w_suit
            + accuracy * w_acc
            + cost_eff * w_cost
            + complexity * w_comp
        )

        # Eligibility check
        is_eligible, ineligibility = self._check_eligibility(
            option, ecm, config, chars
        )

        # M&V cost estimate
        cost_range = MV_COST_RANGES.get(
            option.value, (Decimal("1"), Decimal("5"))
        )
        mv_cost_low = ecm.project_cost * cost_range[0] / Decimal("100")
        mv_cost_high = ecm.project_cost * cost_range[1] / Decimal("100")

        # Strengths and weaknesses
        strengths = self._option_strengths(option, ecm)
        weaknesses = self._option_weaknesses(option, ecm)

        card = OptionScoreCard(
            option=option,
            suitability_score=_round_val(suitability, 4),
            accuracy_score=_round_val(accuracy, 4),
            cost_effectiveness_score=_round_val(cost_eff, 4),
            complexity_score=_round_val(complexity, 4),
            weighted_total=_round_val(total, 4),
            boundary=chars.get("boundary", MeasurementBoundary.RETROFIT_ISOLATION),
            metering_approach=chars.get("metering", MeteringApproach.KEY_PARAMETER),
            cost_level=chars.get("cost", CostLevel.LOW),
            accuracy_level=chars.get("accuracy", AccuracyLevel.MEDIUM),
            stipulation_required=(option == IPMVPOption.OPTION_A),
            estimated_mv_cost_low=_round_val(mv_cost_low, 2),
            estimated_mv_cost_high=_round_val(mv_cost_high, 2),
            is_eligible=is_eligible,
            ineligibility_reasons=ineligibility,
            strengths=strengths,
            weaknesses=weaknesses,
        )
        card.provenance_hash = _compute_hash(card)
        return card

    def _adjust_suitability(
        self,
        base_score: Decimal,
        option: IPMVPOption,
        ecm: ECMDescription,
        chars: Dict[str, Any],
    ) -> Decimal:
        """Adjust suitability score based on ECM characteristics."""
        score = base_score

        # Savings fraction check for Option C
        min_frac = _decimal(chars.get("min_savings_fraction", Decimal("0")))
        if min_frac > Decimal("0"):
            savings_frac = _safe_divide(
                ecm.estimated_savings_pct, Decimal("100")
            )
            if savings_frac < min_frac:
                # Penalise Option C if savings are too small
                score *= Decimal("0.5")

        # Isolation bonus/penalty
        if option in (IPMVPOption.OPTION_A, IPMVPOption.OPTION_B):
            if not ecm.can_isolate_retrofit:
                score *= Decimal("0.3")

        # Multiple ECMs favour Option C
        if ecm.number_of_ecms > 1 and option == IPMVPOption.OPTION_C:
            score = min(Decimal("1"), score * Decimal("1.2"))

        # Simulation model available boosts Option D
        if ecm.simulation_model_available and option == IPMVPOption.OPTION_D:
            score = min(Decimal("1"), score * Decimal("1.3"))

        # Interactive effects favour whole-facility options
        if ecm.interactive_effects and option in (
            IPMVPOption.OPTION_A, IPMVPOption.OPTION_B
        ):
            score *= Decimal("0.7")

        return max(Decimal("0"), min(Decimal("1"), score))

    def _check_eligibility(
        self,
        option: IPMVPOption,
        ecm: ECMDescription,
        config: OptionSelectionConfig,
        chars: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Check whether an option is eligible given constraints."""
        reasons: List[str] = []

        # Cannot isolate retrofit for Options A/B
        if option in (IPMVPOption.OPTION_A, IPMVPOption.OPTION_B):
            if not ecm.can_isolate_retrofit:
                reasons.append(
                    "Retrofit cannot be isolated from rest of facility."
                )

        # Option C requires savings > ~10% of facility
        if option == IPMVPOption.OPTION_C:
            if ecm.estimated_savings_pct < Decimal("5"):
                reasons.append(
                    f"Estimated savings ({float(ecm.estimated_savings_pct):.1f}%) "
                    "may be too small to detect at whole-facility level."
                )

        # Option D requires simulation model
        if option == IPMVPOption.OPTION_D:
            if not ecm.simulation_model_available and ecm.ecm_type != ECMType.NEW_CONSTRUCTION:
                reasons.append(
                    "No energy simulation model available and not new "
                    "construction. Option D requires a calibrated model."
                )

        # Budget constraint
        if ecm.budget_constraint is not None:
            cost_range = MV_COST_RANGES.get(
                option.value, (Decimal("1"), Decimal("5"))
            )
            min_cost = ecm.project_cost * cost_range[0] / Decimal("100")
            if min_cost > ecm.budget_constraint:
                reasons.append(
                    f"Minimum M&V cost (${float(min_cost):.0f}) exceeds "
                    f"budget constraint (${float(ecm.budget_constraint):.0f})."
                )

        # Accuracy requirement
        if ecm.accuracy_requirement:
            option_accuracy = chars.get("accuracy", AccuracyLevel.MEDIUM)
            accuracy_ranks = {
                AccuracyLevel.LOW: 1,
                AccuracyLevel.MEDIUM: 2,
                AccuracyLevel.HIGH: 3,
                AccuracyLevel.VERY_HIGH: 4,
            }
            if accuracy_ranks.get(option_accuracy, 0) < accuracy_ranks.get(
                ecm.accuracy_requirement, 0
            ):
                reasons.append(
                    f"Option accuracy ({option_accuracy.value}) does not "
                    f"meet requirement ({ecm.accuracy_requirement.value})."
                )

        is_eligible = len(reasons) == 0
        return is_eligible, reasons

    def _assess_confidence(
        self,
        eligible: List[OptionScoreCard],
    ) -> SelectionConfidence:
        """Assess confidence in the automated selection."""
        if len(eligible) < 2:
            return SelectionConfidence.MEDIUM

        gap = eligible[0].weighted_total - eligible[1].weighted_total
        if gap >= CONFIDENCE_HIGH_THRESHOLD:
            return SelectionConfidence.HIGH
        elif gap >= CONFIDENCE_MEDIUM_THRESHOLD:
            return SelectionConfidence.MEDIUM
        else:
            return SelectionConfidence.LOW

    # ------------------------------------------------------------------ #
    # Private: Option Implementation Details                               #
    # ------------------------------------------------------------------ #

    def _build_option_a_detail(
        self,
        ecm: ECMDescription,
        stipulated_values: Optional[List[StipulatedValue]] = None,
    ) -> OptionADetail:
        """Build Option A implementation detail."""
        key_params, stip_vals = self._determine_option_a_params(ecm)
        if stipulated_values:
            stip_vals = stipulated_values

        protocol = [
            "1. Install metering equipment for key parameter(s).",
            "2. Collect baseline key parameter data (min. 2 weeks).",
            "3. Document stipulated values with engineering justification.",
            "4. Implement ECM and verify installation.",
            "5. Measure post-retrofit key parameter(s).",
            "6. Calculate savings using measured key + stipulated values.",
            "7. Document uncertainty of stipulated values.",
        ]

        detail = OptionADetail(
            key_parameters=key_params,
            stipulated_values=stip_vals,
            measurement_frequency="continuous" if ecm.ecm_type in (
                ECMType.HVAC_EQUIPMENT, ECMType.MOTORS_DRIVES
            ) else "spot_with_annual_check",
            spot_vs_continuous="spot" if ecm.ecm_type == ECMType.LIGHTING else "continuous",
            estimated_uncertainty_pct=Decimal("20"),
            verification_protocol=protocol,
        )
        detail.provenance_hash = _compute_hash(detail)
        return detail

    def _build_option_b_detail(
        self,
        ecm: ECMDescription,
    ) -> OptionBDetail:
        """Build Option B implementation detail."""
        params, points = self._determine_option_b_params(ecm)

        protocol = [
            "1. Define measurement boundary around the retrofit.",
            "2. Install metering for all energy-affecting parameters.",
            "3. Commission meters and verify accuracy.",
            "4. Collect baseline measurements (min. 2 weeks continuous).",
            "5. Implement ECM and verify installation.",
            "6. Collect post-retrofit measurements (same duration).",
            "7. Calculate savings from measured pre/post data.",
        ]

        detail = OptionBDetail(
            measured_parameters=params,
            metering_points=points,
            measurement_duration="12 months continuous",
            continuous_vs_short_term="continuous",
            data_collection_frequency="15-minute",
            estimated_uncertainty_pct=Decimal("10"),
            verification_protocol=protocol,
        )
        detail.provenance_hash = _compute_hash(detail)
        return detail

    def _build_option_c_detail(
        self,
        ecm: ECMDescription,
    ) -> OptionCDetail:
        """Build Option C implementation detail."""
        ind_vars = ["temperature"]
        if ecm.is_production_dependent:
            ind_vars.append("production_volume")
        if ecm.ecm_type in (ECMType.BEHAVIOURAL, ECMType.RETRO_COMMISSIONING):
            ind_vars.append("occupancy")

        model_type = "change_point"
        if ecm.is_weather_dependent:
            model_type = "5p_change_point"
        elif ecm.is_production_dependent:
            model_type = "multivariate_linear"

        protocol = [
            "1. Collect 12+ months of utility billing data (baseline).",
            "2. Collect weather data for the baseline period.",
            "3. Develop regression model and validate per ASHRAE 14.",
            "4. Implement ECM and verify installation.",
            "5. Collect reporting-period utility and weather data.",
            "6. Apply routine adjustments (weather normalisation).",
            "7. Apply non-routine adjustments (if any).",
            "8. Calculate savings: adjusted baseline - actual.",
        ]

        detail = OptionCDetail(
            utility_meters=ecm.existing_meters or ["electric_utility_meter"],
            regression_model_type=model_type,
            independent_variables=ind_vars,
            baseline_period_months=12,
            reporting_frequency="monthly",
            min_savings_fraction=Decimal("0.10"),
            estimated_uncertainty_pct=Decimal("15"),
            verification_protocol=protocol,
        )
        detail.provenance_hash = _compute_hash(detail)
        return detail

    def _build_option_d_detail(
        self,
        ecm: ECMDescription,
        calibration_data: Optional[SimulationCalibrationData] = None,
    ) -> OptionDDetail:
        """Build Option D implementation detail."""
        protocol = [
            "1. Develop energy simulation model of the facility.",
            "2. Calibrate model to 12+ months of utility data.",
            "3. Verify calibration meets ASHRAE 14 criteria.",
            "4. Simulate baseline energy use.",
            "5. Modify model to reflect ECM implementation.",
            "6. Simulate post-retrofit energy use.",
            "7. Calculate savings: simulated baseline - simulated post.",
            "8. Validate simulation against actual post data.",
        ]

        detail = OptionDDetail(
            simulation_software="EnergyPlus",
            calibration_data=calibration_data,
            model_complexity="detailed" if ecm.ecm_type == ECMType.NEW_CONSTRUCTION else "standard",
            weather_file_source="TMY3",
            estimated_uncertainty_pct=Decimal("8"),
            verification_protocol=protocol,
        )
        detail.provenance_hash = _compute_hash(detail)
        return detail

    def _determine_option_a_params(
        self,
        ecm: ECMDescription,
    ) -> Tuple[List[str], List[StipulatedValue]]:
        """Determine key and stipulated parameters for Option A."""
        key_params: List[str] = []
        stip_vals: List[StipulatedValue] = []

        if ecm.ecm_type == ECMType.LIGHTING:
            key_params = ["operating_hours"]
            stip_vals = [
                StipulatedValue(
                    parameter_name="wattage_reduction_per_fixture",
                    unit="watts",
                    source="manufacturer_spec",
                    confidence_pct=Decimal("95"),
                    justification="Manufacturer rated wattage, verified by spot measurement.",
                ),
                StipulatedValue(
                    parameter_name="number_of_fixtures",
                    unit="count",
                    source="installation_count",
                    confidence_pct=Decimal("99"),
                    justification="Verified count during installation.",
                ),
            ]
        elif ecm.ecm_type == ECMType.MOTORS_DRIVES:
            key_params = ["motor_power_kw", "operating_hours"]
            stip_vals = [
                StipulatedValue(
                    parameter_name="load_factor",
                    unit="fraction",
                    source="engineering_estimate",
                    confidence_pct=Decimal("80"),
                    justification="Based on process loading analysis.",
                ),
            ]
        elif ecm.ecm_type == ECMType.HVAC_EQUIPMENT:
            key_params = ["equipment_runtime_hours"]
            stip_vals = [
                StipulatedValue(
                    parameter_name="efficiency_improvement",
                    unit="COP_or_EER",
                    source="manufacturer_spec",
                    confidence_pct=Decimal("90"),
                    justification="Manufacturer rated efficiency at standard conditions.",
                ),
            ]
        else:
            key_params = ["energy_consumption"]
            stip_vals = [
                StipulatedValue(
                    parameter_name="operating_conditions",
                    unit="various",
                    source="baseline_measurements",
                    confidence_pct=Decimal("80"),
                    justification="Based on baseline period measurements.",
                ),
            ]

        return key_params, stip_vals

    def _determine_option_b_params(
        self,
        ecm: ECMDescription,
    ) -> Tuple[List[str], List[str]]:
        """Determine measured parameters and metering points for Option B."""
        params: List[str] = []
        points: List[str] = []

        if ecm.ecm_type in (ECMType.HVAC_EQUIPMENT, ECMType.HVAC_CONTROLS):
            params = [
                "electrical_power_kw", "thermal_output_kw",
                "flow_rate", "supply_temperature", "return_temperature",
                "operating_hours",
            ]
            points = [
                "equipment_electrical_panel", "supply_pipe",
                "return_pipe", "flow_meter",
            ]
        elif ecm.ecm_type == ECMType.MOTORS_DRIVES:
            params = [
                "motor_power_kw", "motor_speed_rpm",
                "operating_hours", "load_profile",
            ]
            points = [
                "motor_electrical_panel", "vfd_output",
            ]
        elif ecm.ecm_type == ECMType.COMPRESSED_AIR:
            params = [
                "compressor_power_kw", "air_flow_cfm",
                "system_pressure_psi", "operating_hours",
            ]
            points = [
                "compressor_panel", "flow_meter", "pressure_sensor",
            ]
        else:
            params = [
                "energy_consumption_kwh", "power_demand_kw",
                "operating_hours",
            ]
            points = ["sub_meter"]

        return params, points

    # ------------------------------------------------------------------ #
    # Private: Comparison & Rationale                                      #
    # ------------------------------------------------------------------ #

    def _build_comparison_summary(
        self,
        score_cards: List[OptionScoreCard],
    ) -> Dict[str, Dict[str, str]]:
        """Build a comparison summary dictionary."""
        summary: Dict[str, Dict[str, str]] = {}
        for sc in score_cards:
            opt_key = sc.option.value.upper().replace("_", " ")
            summary[opt_key] = {
                "boundary": sc.boundary.value,
                "metering": sc.metering_approach.value,
                "cost_level": sc.cost_level.value,
                "accuracy": sc.accuracy_level.value,
                "stipulation_required": "yes" if sc.stipulation_required else "no",
                "suitability_score": f"{float(sc.suitability_score):.2f}",
                "weighted_total": f"{float(sc.weighted_total):.3f}",
                "rank": str(sc.rank),
                "eligible": "yes" if sc.is_eligible else "no",
                "mv_cost_range": (
                    f"${float(sc.estimated_mv_cost_low):,.0f}"
                    f" - ${float(sc.estimated_mv_cost_high):,.0f}"
                ),
            }
        return summary

    def _build_rationale(
        self,
        recommended: Optional[IPMVPOption],
        eligible: List[OptionScoreCard],
        ecm: ECMDescription,
        confidence: SelectionConfidence,
    ) -> str:
        """Build human-readable selection rationale."""
        if not recommended:
            return (
                "No option could be recommended. All options were either "
                "excluded or ineligible for this ECM configuration."
            )

        best = eligible[0]
        rationale = (
            f"Recommended IPMVP {recommended.value.upper().replace('_', ' ')} "
            f"for ECM type '{ecm.ecm_type.value}' based on weighted scoring "
            f"(suitability={float(best.suitability_score):.2f}, "
            f"accuracy={float(best.accuracy_score):.2f}, "
            f"cost={float(best.cost_effectiveness_score):.2f}, "
            f"complexity={float(best.complexity_score):.2f}, "
            f"total={float(best.weighted_total):.3f}). "
        )

        if len(eligible) > 1:
            runner_up = eligible[1]
            gap = best.weighted_total - runner_up.weighted_total
            rationale += (
                f"Runner-up: {runner_up.option.value.upper().replace('_', ' ')} "
                f"(score={float(runner_up.weighted_total):.3f}, "
                f"gap={float(gap):.3f}). "
            )

        rationale += f"Selection confidence: {confidence.value}."
        return rationale

    # ------------------------------------------------------------------ #
    # Private: Strengths & Weaknesses                                      #
    # ------------------------------------------------------------------ #

    def _option_strengths(
        self,
        option: IPMVPOption,
        ecm: ECMDescription,
    ) -> List[str]:
        """Identify strengths of an option for this ECM."""
        strengths: List[str] = []

        if option == IPMVPOption.OPTION_A:
            strengths.append("Lowest M&V cost; simple implementation.")
            if ecm.ecm_type == ECMType.LIGHTING:
                strengths.append("Ideal for lighting with stable wattage.")

        elif option == IPMVPOption.OPTION_B:
            strengths.append("High accuracy through complete measurement.")
            strengths.append("No stipulation risk.")

        elif option == IPMVPOption.OPTION_C:
            strengths.append("Uses existing utility meters (low incremental cost).")
            if ecm.number_of_ecms > 1:
                strengths.append("Captures interactive effects of multiple ECMs.")

        elif option == IPMVPOption.OPTION_D:
            strengths.append("Highest accuracy for complex retrofits.")
            strengths.append("Can model scenarios that do not yet exist.")

        return strengths

    def _option_weaknesses(
        self,
        option: IPMVPOption,
        ecm: ECMDescription,
    ) -> List[str]:
        """Identify weaknesses of an option for this ECM."""
        weaknesses: List[str] = []

        if option == IPMVPOption.OPTION_A:
            weaknesses.append("Stipulated values introduce uncertainty.")
            if ecm.interactive_effects:
                weaknesses.append("Cannot capture interactive effects.")

        elif option == IPMVPOption.OPTION_B:
            weaknesses.append("Higher metering cost and complexity.")
            weaknesses.append("Requires dedicated sub-metering installation.")

        elif option == IPMVPOption.OPTION_C:
            if ecm.estimated_savings_pct < Decimal("10"):
                weaknesses.append(
                    "Small savings may be lost in whole-facility meter noise."
                )
            weaknesses.append("Requires regression model development.")

        elif option == IPMVPOption.OPTION_D:
            weaknesses.append("Highest cost; requires simulation expertise.")
            weaknesses.append("Model calibration is time-consuming.")

        return weaknesses

    # ------------------------------------------------------------------ #
    # Private: Warnings & Recommendations                                  #
    # ------------------------------------------------------------------ #

    def _generate_warnings(
        self,
        recommended: Optional[IPMVPOption],
        score_cards: List[OptionScoreCard],
        ecm: ECMDescription,
        config: OptionSelectionConfig,
    ) -> List[str]:
        """Generate warnings for option selection."""
        warnings: List[str] = []

        if not recommended:
            warnings.append(
                "No eligible IPMVP option found. Review ECM description "
                "and constraints."
            )

        eligible = [sc for sc in score_cards if sc.is_eligible]
        if len(eligible) == 1:
            warnings.append(
                "Only one option is eligible. Limited alternative if "
                "recommended option proves unsuitable."
            )

        if recommended == IPMVPOption.OPTION_A and ecm.interactive_effects:
            warnings.append(
                "Option A is recommended but ECM has interactive effects. "
                "Stipulated values may not capture interaction."
            )

        if (recommended == IPMVPOption.OPTION_C
                and ecm.estimated_savings_pct < Decimal("10")):
            warnings.append(
                "Option C selected but estimated savings are below 10% "
                "of facility energy. Savings may not be detectable."
            )

        return warnings

    def _generate_recommendations(
        self,
        recommended: Optional[IPMVPOption],
        eligible: List[OptionScoreCard],
        ecm: ECMDescription,
    ) -> List[str]:
        """Generate recommendations for option selection."""
        recs: List[str] = []

        if recommended:
            recs.append(
                f"Proceed with {recommended.value.upper().replace('_', ' ')} "
                "implementation per the verification protocol."
            )
            recs.append(
                "Document the option selection rationale in the M&V Plan "
                "per IPMVP Core Concepts 2022, Chapter 3."
            )

        if len(eligible) >= 2:
            recs.append(
                f"Consider {eligible[1].option.value.upper().replace('_', ' ')} "
                "as a viable alternative if constraints change."
            )

        if ecm.number_of_ecms > 1 and recommended != IPMVPOption.OPTION_C:
            recs.append(
                "With multiple ECMs, also evaluate Option C for its "
                "ability to capture interactive effects."
            )

        return recs
