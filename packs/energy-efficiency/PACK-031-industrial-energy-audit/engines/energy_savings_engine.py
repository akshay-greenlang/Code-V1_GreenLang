# -*- coding: utf-8 -*-
"""
EnergySavingsEngine - PACK-031 Industrial Energy Audit Engine 5
================================================================

Identifies, quantifies, and prioritises energy conservation measures (ECMs)
for industrial facilities. Implements IPMVP Options A-D for measurement
and verification planning, performs comprehensive financial analysis
(NPV, IRR, simple/discounted payback, LCOE), models measure interaction
effects, generates Marginal Abatement Cost Curves (MACC), and builds
prioritised implementation roadmaps.

Calculation Methodology:
    Financial Analysis:
        NPV = sum( (savings_t - maintenance_t) / (1 + r)^t ) - capex
        IRR = r such that NPV(r) = 0  (bisection method)
        Simple Payback = capex / annual_net_savings
        Discounted Payback = t where cumulative_discounted_savings >= capex
        LCOE = total_cost / total_savings_kwh (levelised cost of energy saved)
        ROI = (total_net_savings - capex) / capex * 100

    Risk-Adjusted Savings:
        expected_savings = base_savings * confidence_factor
        lower_bound = expected * (1 - uncertainty / 2)
        upper_bound = expected * (1 + uncertainty / 2)

    Measure Interactions:
        When ECM_A and ECM_B affect the same system:
        combined_savings = savings_A + savings_B * adjustment_factor
        (adjustment_factor < 1.0 for competing measures)

    MACC (Marginal Abatement Cost Curve):
        abatement_cost = annualised_cost / annual_savings_kwh
        Sorted ascending by abatement_cost for waterfall chart.

    IPMVP Options:
        A: Retrofit Isolation - Key Parameter Measurement
        B: Retrofit Isolation - All Parameter Measurement
        C: Whole Facility (utility bills before/after)
        D: Calibrated Simulation

Regulatory References:
    - IPMVP Core Concepts (Efficiency Valuation Organization, 2022)
    - ISO 50001:2018 - Energy management systems
    - ISO 50006:2023 - Energy baseline and EnPI methodology
    - ISO 50015:2014 - Measurement and verification
    - EN 16247-1:2022 - Energy audits (general requirements)
    - EN 16247-3:2022 - Energy audits (processes)
    - EU EED Article 8 - Mandatory energy audits

Zero-Hallucination:
    - All financial formulas are standard engineering economics
    - IPMVP options from EVO published criteria
    - ECM library sourced from IEA/DOE published databases
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-031 Industrial Energy Audit
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

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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


def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ECMCategory(str, Enum):
    """Energy Conservation Measure category.

    LIGHTING: Lighting upgrades (LED, controls, daylighting).
    HVAC: HVAC system improvements.
    MOTORS: Motor and drive upgrades.
    COMPRESSED_AIR: Compressed air system optimization.
    PROCESS_HEAT: Process heating improvements.
    BOILER: Boiler and steam system upgrades.
    BUILDING_ENVELOPE: Insulation, windows, air sealing.
    CONTROLS: Building/process automation and controls.
    POWER_QUALITY: Power factor correction, harmonics.
    RENEWABLE: On-site renewable energy.
    WASTE_HEAT: Waste heat recovery measures.
    WATER: Water heating and conservation.
    REFRIGERATION: Refrigeration system improvements.
    MATERIAL_HANDLING: Conveyor, crane, elevator efficiency.
    SCHEDULING: Production scheduling optimization.
    """
    LIGHTING = "lighting"
    HVAC = "hvac"
    MOTORS = "motors"
    COMPRESSED_AIR = "compressed_air"
    PROCESS_HEAT = "process_heat"
    BOILER = "boiler"
    BUILDING_ENVELOPE = "building_envelope"
    CONTROLS = "controls"
    POWER_QUALITY = "power_quality"
    RENEWABLE = "renewable"
    WASTE_HEAT = "waste_heat"
    WATER = "water"
    REFRIGERATION = "refrigeration"
    MATERIAL_HANDLING = "material_handling"
    SCHEDULING = "scheduling"


class IPMVPOption(str, Enum):
    """IPMVP measurement and verification options.

    OPTION_A: Retrofit Isolation - Key Parameter Measurement.
    OPTION_B: Retrofit Isolation - All Parameter Measurement.
    OPTION_C: Whole Facility (utility data analysis).
    OPTION_D: Calibrated Simulation.
    """
    OPTION_A = "option_a"
    OPTION_B = "option_b"
    OPTION_C = "option_c"
    OPTION_D = "option_d"


class ImplementationComplexity(str, Enum):
    """Complexity level for ECM implementation.

    LOW: Simple retrofit, no production disruption.
    MEDIUM: Moderate complexity, some planning required.
    HIGH: Complex installation, production impact possible.
    VERY_HIGH: Major capital project, engineering design required.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PriorityLevel(str, Enum):
    """ECM priority level based on multi-criteria analysis.

    CRITICAL: Immediate action, high savings, low cost.
    HIGH: Near-term implementation, strong ROI.
    MEDIUM: Planned implementation, moderate ROI.
    LOW: Future consideration, long payback.
    DEFERRED: Not recommended at this time.
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DEFERRED = "deferred"


class MeasureStatus(str, Enum):
    """ECM implementation status.

    IDENTIFIED: Identified during audit.
    QUANTIFIED: Savings quantified and verified.
    APPROVED: Approved for implementation.
    IN_PROGRESS: Under implementation.
    COMPLETED: Implementation complete.
    VERIFIED: Savings verified post-implementation.
    """
    IDENTIFIED = "identified"
    QUANTIFIED = "quantified"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DISCOUNT_RATE: Decimal = Decimal("0.08")
DEFAULT_ENERGY_PRICE_ESCALATION: Decimal = Decimal("0.03")
DEFAULT_CO2_FACTOR_KG_KWH: Decimal = Decimal("0.4")
MAX_IRR_ITERATIONS: int = 100
IRR_TOLERANCE: Decimal = Decimal("0.0001")

# IPMVP Option selection criteria.
IPMVP_OPTION_CRITERIA: Dict[str, Dict[str, Any]] = {
    IPMVPOption.OPTION_A.value: {
        "name": "Retrofit Isolation - Key Parameter Measurement",
        "description": "Savings determined by field measurement of key parameters "
                       "defining energy use, with some parameters estimated.",
        "best_for": ["lighting", "motors", "compressed_air"],
        "accuracy": "moderate",
        "cost": "low",
        "min_savings_kwh": 10000,
    },
    IPMVPOption.OPTION_B.value: {
        "name": "Retrofit Isolation - All Parameter Measurement",
        "description": "Savings determined by field measurement of all parameters "
                       "defining energy use of the affected system.",
        "best_for": ["process_heat", "boiler", "hvac", "refrigeration"],
        "accuracy": "high",
        "cost": "medium",
        "min_savings_kwh": 50000,
    },
    IPMVPOption.OPTION_C.value: {
        "name": "Whole Facility",
        "description": "Savings determined by measuring energy use at the whole "
                       "facility level using utility meter data.",
        "best_for": ["building_envelope", "controls", "multiple_measures"],
        "accuracy": "moderate",
        "cost": "low",
        "min_savings_kwh": 100000,
    },
    IPMVPOption.OPTION_D.value: {
        "name": "Calibrated Simulation",
        "description": "Savings determined through simulation of the facility, "
                       "calibrated against actual utility data.",
        "best_for": ["complex_systems", "new_construction", "deep_retrofit"],
        "accuracy": "high",
        "cost": "high",
        "min_savings_kwh": 200000,
    },
}

# Complexity scoring factors.
COMPLEXITY_SCORES: Dict[str, Decimal] = {
    ImplementationComplexity.LOW.value: Decimal("1"),
    ImplementationComplexity.MEDIUM.value: Decimal("2"),
    ImplementationComplexity.HIGH.value: Decimal("3"),
    ImplementationComplexity.VERY_HIGH.value: Decimal("4"),
}

# ECM Library: 50+ common industrial energy conservation measures.
# Each entry: (name, category, typical_savings_pct, typical_payback_years, complexity).
ECM_LIBRARY: List[Dict[str, Any]] = [
    # Lighting
    {"id": "ECM-L01", "name": "LED retrofit (fluorescent to LED)", "category": "lighting",
     "savings_pct": Decimal("50"), "payback_yr": Decimal("2.0"), "complexity": "low"},
    {"id": "ECM-L02", "name": "LED retrofit (HID to LED)", "category": "lighting",
     "savings_pct": Decimal("60"), "payback_yr": Decimal("2.5"), "complexity": "low"},
    {"id": "ECM-L03", "name": "Occupancy sensor controls", "category": "lighting",
     "savings_pct": Decimal("25"), "payback_yr": Decimal("1.5"), "complexity": "low"},
    {"id": "ECM-L04", "name": "Daylight harvesting controls", "category": "lighting",
     "savings_pct": Decimal("30"), "payback_yr": Decimal("3.0"), "complexity": "medium"},
    {"id": "ECM-L05", "name": "Task lighting optimization", "category": "lighting",
     "savings_pct": Decimal("15"), "payback_yr": Decimal("1.0"), "complexity": "low"},
    # Motors
    {"id": "ECM-M01", "name": "IE3/IE4 motor replacement", "category": "motors",
     "savings_pct": Decimal("5"), "payback_yr": Decimal("3.0"), "complexity": "medium"},
    {"id": "ECM-M02", "name": "Variable speed drive (VSD) installation", "category": "motors",
     "savings_pct": Decimal("25"), "payback_yr": Decimal("2.5"), "complexity": "medium"},
    {"id": "ECM-M03", "name": "Motor right-sizing", "category": "motors",
     "savings_pct": Decimal("10"), "payback_yr": Decimal("2.0"), "complexity": "medium"},
    {"id": "ECM-M04", "name": "Synchronous belt replacement", "category": "motors",
     "savings_pct": Decimal("5"), "payback_yr": Decimal("0.5"), "complexity": "low"},
    {"id": "ECM-M05", "name": "Power factor correction", "category": "power_quality",
     "savings_pct": Decimal("3"), "payback_yr": Decimal("2.0"), "complexity": "low"},
    # Compressed Air
    {"id": "ECM-C01", "name": "Compressed air leak repair", "category": "compressed_air",
     "savings_pct": Decimal("20"), "payback_yr": Decimal("0.5"), "complexity": "low"},
    {"id": "ECM-C02", "name": "VSD compressor retrofit", "category": "compressed_air",
     "savings_pct": Decimal("25"), "payback_yr": Decimal("3.0"), "complexity": "medium"},
    {"id": "ECM-C03", "name": "Pressure reduction optimization", "category": "compressed_air",
     "savings_pct": Decimal("14"), "payback_yr": Decimal("0.5"), "complexity": "low"},
    {"id": "ECM-C04", "name": "Compressed air heat recovery", "category": "compressed_air",
     "savings_pct": Decimal("15"), "payback_yr": Decimal("3.5"), "complexity": "medium"},
    {"id": "ECM-C05", "name": "Air receiver optimization", "category": "compressed_air",
     "savings_pct": Decimal("5"), "payback_yr": Decimal("1.0"), "complexity": "low"},
    {"id": "ECM-C06", "name": "Eliminate inappropriate compressed air use", "category": "compressed_air",
     "savings_pct": Decimal("10"), "payback_yr": Decimal("0.3"), "complexity": "low"},
    # Boiler / Steam
    {"id": "ECM-B01", "name": "Boiler economizer installation", "category": "boiler",
     "savings_pct": Decimal("5"), "payback_yr": Decimal("3.0"), "complexity": "medium"},
    {"id": "ECM-B02", "name": "Combustion optimization (O2 trim)", "category": "boiler",
     "savings_pct": Decimal("3"), "payback_yr": Decimal("1.5"), "complexity": "medium"},
    {"id": "ECM-B03", "name": "Blowdown heat recovery", "category": "boiler",
     "savings_pct": Decimal("2"), "payback_yr": Decimal("2.0"), "complexity": "medium"},
    {"id": "ECM-B04", "name": "Steam trap survey and repair", "category": "boiler",
     "savings_pct": Decimal("10"), "payback_yr": Decimal("0.5"), "complexity": "low"},
    {"id": "ECM-B05", "name": "Condensate return improvement", "category": "boiler",
     "savings_pct": Decimal("5"), "payback_yr": Decimal("2.5"), "complexity": "medium"},
    {"id": "ECM-B06", "name": "Steam pipe insulation repair", "category": "boiler",
     "savings_pct": Decimal("5"), "payback_yr": Decimal("1.0"), "complexity": "low"},
    {"id": "ECM-B07", "name": "Condensing boiler replacement", "category": "boiler",
     "savings_pct": Decimal("12"), "payback_yr": Decimal("5.0"), "complexity": "high"},
    # HVAC
    {"id": "ECM-H01", "name": "HVAC VSD on AHU fans", "category": "hvac",
     "savings_pct": Decimal("20"), "payback_yr": Decimal("3.0"), "complexity": "medium"},
    {"id": "ECM-H02", "name": "Chiller plant optimization", "category": "hvac",
     "savings_pct": Decimal("15"), "payback_yr": Decimal("4.0"), "complexity": "high"},
    {"id": "ECM-H03", "name": "Free cooling economizer", "category": "hvac",
     "savings_pct": Decimal("10"), "payback_yr": Decimal("3.5"), "complexity": "medium"},
    {"id": "ECM-H04", "name": "Chiller replacement (high-efficiency)", "category": "hvac",
     "savings_pct": Decimal("25"), "payback_yr": Decimal("6.0"), "complexity": "very_high"},
    {"id": "ECM-H05", "name": "Heat recovery from exhaust air", "category": "hvac",
     "savings_pct": Decimal("12"), "payback_yr": Decimal("4.0"), "complexity": "medium"},
    # Process Heat
    {"id": "ECM-P01", "name": "Furnace/kiln insulation upgrade", "category": "process_heat",
     "savings_pct": Decimal("8"), "payback_yr": Decimal("2.5"), "complexity": "medium"},
    {"id": "ECM-P02", "name": "Regenerative burner retrofit", "category": "process_heat",
     "savings_pct": Decimal("20"), "payback_yr": Decimal("4.0"), "complexity": "high"},
    {"id": "ECM-P03", "name": "Combustion air preheating", "category": "process_heat",
     "savings_pct": Decimal("10"), "payback_yr": Decimal("3.0"), "complexity": "medium"},
    {"id": "ECM-P04", "name": "Batch-to-continuous process", "category": "process_heat",
     "savings_pct": Decimal("15"), "payback_yr": Decimal("5.0"), "complexity": "very_high"},
    # Building Envelope
    {"id": "ECM-E01", "name": "Roof insulation upgrade", "category": "building_envelope",
     "savings_pct": Decimal("8"), "payback_yr": Decimal("4.0"), "complexity": "medium"},
    {"id": "ECM-E02", "name": "Wall insulation improvement", "category": "building_envelope",
     "savings_pct": Decimal("6"), "payback_yr": Decimal("5.0"), "complexity": "medium"},
    {"id": "ECM-E03", "name": "High-performance glazing", "category": "building_envelope",
     "savings_pct": Decimal("5"), "payback_yr": Decimal("7.0"), "complexity": "high"},
    {"id": "ECM-E04", "name": "Air sealing and weatherization", "category": "building_envelope",
     "savings_pct": Decimal("4"), "payback_yr": Decimal("2.0"), "complexity": "low"},
    {"id": "ECM-E05", "name": "Loading dock seals/shelters", "category": "building_envelope",
     "savings_pct": Decimal("3"), "payback_yr": Decimal("1.5"), "complexity": "low"},
    # Controls
    {"id": "ECM-A01", "name": "Building Management System (BMS) upgrade", "category": "controls",
     "savings_pct": Decimal("12"), "payback_yr": Decimal("3.5"), "complexity": "high"},
    {"id": "ECM-A02", "name": "Setpoint optimization", "category": "controls",
     "savings_pct": Decimal("5"), "payback_yr": Decimal("0.5"), "complexity": "low"},
    {"id": "ECM-A03", "name": "Scheduling optimization", "category": "scheduling",
     "savings_pct": Decimal("8"), "payback_yr": Decimal("0.5"), "complexity": "low"},
    {"id": "ECM-A04", "name": "Energy monitoring and targeting", "category": "controls",
     "savings_pct": Decimal("5"), "payback_yr": Decimal("2.0"), "complexity": "medium"},
    # Waste Heat Recovery
    {"id": "ECM-W01", "name": "Flue gas economizer", "category": "waste_heat",
     "savings_pct": Decimal("5"), "payback_yr": Decimal("3.0"), "complexity": "medium"},
    {"id": "ECM-W02", "name": "Condensate heat recovery", "category": "waste_heat",
     "savings_pct": Decimal("4"), "payback_yr": Decimal("2.5"), "complexity": "medium"},
    {"id": "ECM-W03", "name": "ORC waste heat to power", "category": "waste_heat",
     "savings_pct": Decimal("10"), "payback_yr": Decimal("6.0"), "complexity": "very_high"},
    {"id": "ECM-W04", "name": "Heat pump for low-grade waste heat", "category": "waste_heat",
     "savings_pct": Decimal("15"), "payback_yr": Decimal("5.0"), "complexity": "high"},
    # Refrigeration
    {"id": "ECM-R01", "name": "Floating head pressure control", "category": "refrigeration",
     "savings_pct": Decimal("10"), "payback_yr": Decimal("2.0"), "complexity": "medium"},
    {"id": "ECM-R02", "name": "EC fan motors on condensers", "category": "refrigeration",
     "savings_pct": Decimal("8"), "payback_yr": Decimal("3.0"), "complexity": "medium"},
    {"id": "ECM-R03", "name": "Strip curtain installation on cold rooms", "category": "refrigeration",
     "savings_pct": Decimal("5"), "payback_yr": Decimal("0.5"), "complexity": "low"},
    # Renewable
    {"id": "ECM-N01", "name": "Rooftop solar PV installation", "category": "renewable",
     "savings_pct": Decimal("15"), "payback_yr": Decimal("6.0"), "complexity": "high"},
    {"id": "ECM-N02", "name": "Solar thermal for process heat", "category": "renewable",
     "savings_pct": Decimal("10"), "payback_yr": Decimal("7.0"), "complexity": "high"},
    {"id": "ECM-N03", "name": "Battery energy storage system", "category": "renewable",
     "savings_pct": Decimal("5"), "payback_yr": Decimal("8.0"), "complexity": "very_high"},
]

# Confidence factors by data quality.
CONFIDENCE_FACTORS: Dict[str, Decimal] = {
    "measured": Decimal("0.95"),
    "calculated": Decimal("0.85"),
    "estimated": Decimal("0.70"),
    "benchmarked": Decimal("0.60"),
}

# Uncertainty ranges by confidence level.
UNCERTAINTY_RANGES: Dict[str, Decimal] = {
    "measured": Decimal("0.10"),
    "calculated": Decimal("0.20"),
    "estimated": Decimal("0.35"),
    "benchmarked": Decimal("0.50"),
}


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class EnergySavingsMeasure(BaseModel):
    """Input data for an energy conservation measure (ECM).

    Attributes:
        measure_id: Unique measure identifier.
        name: Measure name / description.
        category: ECM category.
        system_affected: System or equipment affected.
        baseline_kwh: Baseline annual energy consumption (kWh).
        expected_savings_kwh: Expected annual savings (kWh).
        savings_pct: Savings as percentage of baseline.
        confidence_level: Savings confidence (measured/calculated/estimated/benchmarked).
        implementation_cost_eur: Total implementation cost (EUR).
        annual_maintenance_eur: Annual incremental maintenance cost (EUR).
        lifetime_years: Measure economic lifetime (years).
        complexity: Implementation complexity.
        status: Current measure status.
        ecm_library_id: Reference to ECM library entry.
        notes: Additional notes.
    """
    measure_id: str = Field(default_factory=_new_uuid, description="Measure ID")
    name: str = Field(default="", max_length=500, description="Measure name")
    category: str = Field(
        default=ECMCategory.CONTROLS.value,
        description="ECM category"
    )
    system_affected: str = Field(default="", max_length=200, description="System affected")
    baseline_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Baseline consumption (kWh)"
    )
    expected_savings_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Expected savings (kWh)"
    )
    savings_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Savings percentage"
    )
    confidence_level: str = Field(
        default="estimated",
        description="Savings confidence (measured/calculated/estimated/benchmarked)"
    )
    implementation_cost_eur: Decimal = Field(
        default=Decimal("0"), ge=0, description="Implementation cost (EUR)"
    )
    annual_maintenance_eur: Decimal = Field(
        default=Decimal("0"), ge=0, description="Annual maintenance (EUR)"
    )
    lifetime_years: int = Field(
        default=10, ge=1, le=30, description="Economic lifetime (years)"
    )
    complexity: str = Field(
        default=ImplementationComplexity.MEDIUM.value,
        description="Implementation complexity"
    )
    status: str = Field(
        default=MeasureStatus.IDENTIFIED.value,
        description="Measure status"
    )
    ecm_library_id: str = Field(
        default="", description="ECM library reference"
    )
    notes: str = Field(default="", description="Additional notes")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        valid = {c.value for c in ECMCategory}
        if v not in valid:
            raise ValueError(f"Unknown ECM category '{v}'. Must be one of: {sorted(valid)}")
        return v

    @field_validator("complexity")
    @classmethod
    def validate_complexity(cls, v: str) -> str:
        valid = {c.value for c in ImplementationComplexity}
        if v not in valid:
            raise ValueError(f"Unknown complexity '{v}'. Must be one of: {sorted(valid)}")
        return v

    @field_validator("confidence_level")
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        valid = {"measured", "calculated", "estimated", "benchmarked"}
        if v not in valid:
            raise ValueError(f"Unknown confidence '{v}'. Must be one of: {sorted(valid)}")
        return v


class InteractionEffect(BaseModel):
    """Interaction effect between two ECMs.

    When two measures affect the same system, their combined savings
    may be less (or occasionally more) than the sum of individual savings.

    Attributes:
        measure_a_id: First measure identifier.
        measure_b_id: Second measure identifier.
        combined_savings_adjustment_pct: Adjustment factor (-100 to +20 pct).
        reason: Explanation of interaction.
    """
    measure_a_id: str = Field(default="", description="First measure ID")
    measure_b_id: str = Field(default="", description="Second measure ID")
    combined_savings_adjustment_pct: Decimal = Field(
        default=Decimal("-15"), ge=Decimal("-100"), le=Decimal("20"),
        description="Savings adjustment (%)"
    )
    reason: str = Field(default="", description="Interaction explanation")


class EnergySavingsInput(BaseModel):
    """Complete input for energy savings analysis.

    Attributes:
        facility_id: Facility identifier.
        facility_name: Facility name.
        total_baseline_kwh: Total facility baseline energy (kWh/year).
        total_baseline_cost_eur: Total facility energy cost (EUR/year).
        energy_price_eur_kwh: Energy unit price (EUR/kWh).
        measures: List of identified ECMs.
        interaction_effects: Known measure interaction effects.
        discount_rate: Discount rate for NPV/IRR.
        energy_price_escalation: Annual energy price escalation rate.
        co2_factor_kg_kwh: Grid CO2 emission factor (kg/kWh).
        carbon_price_eur_tonne: Carbon price (EUR/tCO2e).
        include_carbon_savings: Whether to include carbon cost savings.
        ipmvp_required: Whether IPMVP plan is required.
        max_payback_years: Maximum acceptable payback period.
        budget_eur: Available budget for implementation.
    """
    facility_id: str = Field(default="", description="Facility ID")
    facility_name: str = Field(default="", max_length=300, description="Facility name")
    total_baseline_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total baseline (kWh/yr)"
    )
    total_baseline_cost_eur: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total baseline cost (EUR/yr)"
    )
    energy_price_eur_kwh: Decimal = Field(
        default=Decimal("0.15"), ge=0, description="Energy price (EUR/kWh)"
    )
    measures: List[EnergySavingsMeasure] = Field(
        default_factory=list, description="ECM list"
    )
    interaction_effects: List[InteractionEffect] = Field(
        default_factory=list, description="Measure interactions"
    )
    discount_rate: Decimal = Field(
        default=DEFAULT_DISCOUNT_RATE, ge=0, le=Decimal("0.30"),
        description="Discount rate"
    )
    energy_price_escalation: Decimal = Field(
        default=DEFAULT_ENERGY_PRICE_ESCALATION, ge=0, le=Decimal("0.15"),
        description="Energy price escalation rate"
    )
    co2_factor_kg_kwh: Decimal = Field(
        default=DEFAULT_CO2_FACTOR_KG_KWH, ge=0,
        description="Grid CO2 factor (kg/kWh)"
    )
    carbon_price_eur_tonne: Decimal = Field(
        default=Decimal("50"), ge=0,
        description="Carbon price (EUR/tCO2e)"
    )
    include_carbon_savings: bool = Field(
        default=True, description="Include carbon cost savings"
    )
    ipmvp_required: bool = Field(
        default=True, description="Generate IPMVP plans"
    )
    max_payback_years: Decimal = Field(
        default=Decimal("7"), ge=0, description="Max acceptable payback (years)"
    )
    budget_eur: Decimal = Field(
        default=Decimal("0"), ge=0, description="Available budget (EUR)"
    )


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class FinancialAnalysis(BaseModel):
    """Financial analysis for an ECM.

    Attributes:
        npv_eur: Net present value (EUR).
        irr_pct: Internal rate of return (%).
        simple_payback_years: Simple payback period (years).
        discounted_payback_years: Discounted payback period (years).
        lcoe_eur_kwh: Levelised cost of energy saved (EUR/kWh).
        roi_pct: Return on investment (%).
        discount_rate: Discount rate used.
        total_lifetime_savings_eur: Total undiscounted savings (EUR).
        total_lifetime_cost_eur: Total undiscounted cost (EUR).
    """
    npv_eur: Decimal = Field(default=Decimal("0"))
    irr_pct: Decimal = Field(default=Decimal("0"))
    simple_payback_years: Decimal = Field(default=Decimal("0"))
    discounted_payback_years: Decimal = Field(default=Decimal("0"))
    lcoe_eur_kwh: Decimal = Field(default=Decimal("0"))
    roi_pct: Decimal = Field(default=Decimal("0"))
    discount_rate: Decimal = Field(default=Decimal("0"))
    total_lifetime_savings_eur: Decimal = Field(default=Decimal("0"))
    total_lifetime_cost_eur: Decimal = Field(default=Decimal("0"))


class IPMVPPlan(BaseModel):
    """IPMVP measurement and verification plan for an ECM.

    Attributes:
        option: IPMVP option selected.
        option_name: Human-readable option name.
        measurement_boundary: Measurement boundary description.
        baseline_period: Recommended baseline period.
        post_period: Recommended post-implementation period.
        key_parameters: Parameters to be measured.
        sampling_approach: Measurement sampling approach.
        mv_cost_eur: Estimated M&V cost (EUR).
        rationale: Rationale for option selection.
    """
    option: str = Field(default="")
    option_name: str = Field(default="")
    measurement_boundary: str = Field(default="")
    baseline_period: str = Field(default="12 months pre-implementation")
    post_period: str = Field(default="12 months post-implementation")
    key_parameters: List[str] = Field(default_factory=list)
    sampling_approach: str = Field(default="")
    mv_cost_eur: Decimal = Field(default=Decimal("0"))
    rationale: str = Field(default="")


class MACCPoint(BaseModel):
    """Single point on the Marginal Abatement Cost Curve.

    Attributes:
        measure_id: Measure identifier.
        measure_name: Measure name.
        abatement_kwh: Annual energy abated (kWh).
        cost_per_kwh_saved: Cost per kWh saved (EUR/kWh).
        cumulative_abatement_kwh: Cumulative abatement (kWh).
        co2_abatement_tco2e: Annual CO2 abatement (tCO2e).
    """
    measure_id: str = Field(default="")
    measure_name: str = Field(default="")
    abatement_kwh: Decimal = Field(default=Decimal("0"))
    cost_per_kwh_saved: Decimal = Field(default=Decimal("0"))
    cumulative_abatement_kwh: Decimal = Field(default=Decimal("0"))
    co2_abatement_tco2e: Decimal = Field(default=Decimal("0"))


class MeasureResult(BaseModel):
    """Analysis result for a single ECM.

    Attributes:
        measure_id: Measure identifier.
        name: Measure name.
        category: ECM category.
        adjusted_savings_kwh: Savings after interaction adjustments (kWh).
        risk_adjusted_savings_kwh: Risk-adjusted savings (kWh).
        savings_lower_bound_kwh: Lower savings bound (kWh).
        savings_upper_bound_kwh: Upper savings bound (kWh).
        annual_cost_savings_eur: Annual cost savings (EUR).
        annual_carbon_savings_tco2e: Annual carbon savings (tCO2e).
        financial: Financial analysis.
        ipmvp_plan: IPMVP M&V plan.
        priority: Priority level.
        priority_score: Numeric priority score (0-100).
        implementation_phase: Recommended phase (1=immediate, 2=near-term, 3=medium).
        within_budget: Whether within available budget.
    """
    measure_id: str = Field(default="")
    name: str = Field(default="")
    category: str = Field(default="")
    adjusted_savings_kwh: Decimal = Field(default=Decimal("0"))
    risk_adjusted_savings_kwh: Decimal = Field(default=Decimal("0"))
    savings_lower_bound_kwh: Decimal = Field(default=Decimal("0"))
    savings_upper_bound_kwh: Decimal = Field(default=Decimal("0"))
    annual_cost_savings_eur: Decimal = Field(default=Decimal("0"))
    annual_carbon_savings_tco2e: Decimal = Field(default=Decimal("0"))
    financial: FinancialAnalysis = Field(default_factory=FinancialAnalysis)
    ipmvp_plan: Optional[IPMVPPlan] = Field(default=None)
    priority: str = Field(default=PriorityLevel.MEDIUM.value)
    priority_score: Decimal = Field(default=Decimal("50"))
    implementation_phase: int = Field(default=2)
    within_budget: bool = Field(default=True)


class RoadmapPhase(BaseModel):
    """Implementation roadmap phase.

    Attributes:
        phase: Phase number (1-3).
        phase_name: Phase description.
        measures: Measure IDs in this phase.
        total_cost_eur: Total implementation cost for phase.
        total_savings_kwh: Total annual savings for phase.
        total_savings_eur: Total annual cost savings for phase.
        cumulative_savings_kwh: Cumulative savings including prior phases.
    """
    phase: int = Field(default=1)
    phase_name: str = Field(default="")
    measures: List[str] = Field(default_factory=list)
    total_cost_eur: Decimal = Field(default=Decimal("0"))
    total_savings_kwh: Decimal = Field(default=Decimal("0"))
    total_savings_eur: Decimal = Field(default=Decimal("0"))
    cumulative_savings_kwh: Decimal = Field(default=Decimal("0"))


class EnergySavingsResult(BaseModel):
    """Complete energy savings analysis result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        facility_id: Facility identifier.
        facility_name: Facility name.
        total_measures: Number of measures analysed.
        total_savings_kwh: Total annual savings (kWh).
        total_savings_pct: Savings as percentage of baseline.
        total_cost_eur: Total implementation cost (EUR).
        total_annual_savings_eur: Total annual cost savings (EUR).
        total_carbon_savings_tco2e: Total annual carbon savings (tCO2e).
        portfolio_npv_eur: Portfolio-level NPV (EUR).
        portfolio_irr_pct: Portfolio-level IRR (%).
        portfolio_payback_years: Portfolio-level simple payback (years).
        measure_results: Per-measure analysis results.
        macc_data: MACC curve data points.
        interaction_adjustments: Summary of interaction adjustments.
        implementation_roadmap: Phased implementation roadmap.
        within_budget: Whether total cost is within budget.
        budget_utilization_pct: Percentage of budget used.
        recommendations: Overall recommendations.
        warnings: Non-blocking warnings.
        errors: Blocking errors.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=_utcnow)
    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    total_measures: int = Field(default=0)
    total_savings_kwh: Decimal = Field(default=Decimal("0"))
    total_savings_pct: Decimal = Field(default=Decimal("0"))
    total_cost_eur: Decimal = Field(default=Decimal("0"))
    total_annual_savings_eur: Decimal = Field(default=Decimal("0"))
    total_carbon_savings_tco2e: Decimal = Field(default=Decimal("0"))
    portfolio_npv_eur: Decimal = Field(default=Decimal("0"))
    portfolio_irr_pct: Decimal = Field(default=Decimal("0"))
    portfolio_payback_years: Decimal = Field(default=Decimal("0"))
    measure_results: List[MeasureResult] = Field(default_factory=list)
    macc_data: List[MACCPoint] = Field(default_factory=list)
    interaction_adjustments: List[Dict[str, Any]] = Field(default_factory=list)
    implementation_roadmap: List[RoadmapPhase] = Field(default_factory=list)
    within_budget: bool = Field(default=True)
    budget_utilization_pct: Decimal = Field(default=Decimal("0"))
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class EnergySavingsEngine:
    """Energy savings opportunity identification and prioritisation engine.

    Quantifies ECMs with IPMVP-aligned M&V planning, performs NPV/IRR/payback
    financial analysis, models measure interactions, generates MACC curves,
    and builds phased implementation roadmaps.

    Usage::

        engine = EnergySavingsEngine()
        result = engine.analyze(input_data)
        print(f"Total savings: {result.total_savings_kwh} kWh")
        print(f"Portfolio NPV: {result.portfolio_npv_eur} EUR")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise EnergySavingsEngine.

        Args:
            config: Optional overrides. Supported keys:
                - discount_rate (Decimal): default discount rate
                - escalation_rate (Decimal): default energy price escalation
                - max_payback (Decimal): default max payback
        """
        self.config = config or {}
        self._discount_rate = _decimal(
            self.config.get("discount_rate", DEFAULT_DISCOUNT_RATE)
        )
        self._escalation = _decimal(
            self.config.get("escalation_rate", DEFAULT_ENERGY_PRICE_ESCALATION)
        )
        logger.info(
            "EnergySavingsEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def analyze(
        self, data: EnergySavingsInput,
    ) -> EnergySavingsResult:
        """Perform complete energy savings analysis.

        Args:
            data: Validated energy savings input.

        Returns:
            EnergySavingsResult with complete analysis.
        """
        t0 = time.perf_counter()
        logger.info(
            "Energy savings analysis: facility=%s, measures=%d",
            data.facility_name, len(data.measures),
        )

        warnings: List[str] = []
        errors: List[str] = []
        recommendations: List[str] = []

        if not data.measures:
            errors.append("No energy conservation measures provided.")

        discount = data.discount_rate if data.discount_rate > Decimal("0") else self._discount_rate
        escalation = data.energy_price_escalation
        price = data.energy_price_eur_kwh

        # Build interaction map
        interaction_map: Dict[str, List[InteractionEffect]] = {}
        for ie in data.interaction_effects:
            interaction_map.setdefault(ie.measure_a_id, []).append(ie)
            interaction_map.setdefault(ie.measure_b_id, []).append(ie)

        # Step 1: Analyze each measure
        measure_results: List[MeasureResult] = []
        interaction_log: List[Dict[str, Any]] = []

        for m in data.measures:
            # Calculate savings (resolve savings_kwh vs savings_pct)
            base_savings = m.expected_savings_kwh
            if base_savings <= Decimal("0") and m.savings_pct > Decimal("0"):
                base_savings = m.baseline_kwh * m.savings_pct / Decimal("100")

            # Apply interaction adjustments
            adjusted_savings = base_savings
            for ie in interaction_map.get(m.measure_id, []):
                adjustment = ie.combined_savings_adjustment_pct / Decimal("100")
                adjusted_savings = adjusted_savings * (Decimal("1") + adjustment)
                interaction_log.append({
                    "measure_id": m.measure_id,
                    "interacts_with": ie.measure_b_id if ie.measure_a_id == m.measure_id else ie.measure_a_id,
                    "adjustment_pct": str(ie.combined_savings_adjustment_pct),
                    "reason": ie.reason,
                })

            # Risk-adjusted savings
            confidence = CONFIDENCE_FACTORS.get(m.confidence_level, Decimal("0.70"))
            uncertainty = UNCERTAINTY_RANGES.get(m.confidence_level, Decimal("0.35"))
            risk_adjusted = adjusted_savings * confidence
            lower = risk_adjusted * (Decimal("1") - uncertainty / Decimal("2"))
            upper = risk_adjusted * (Decimal("1") + uncertainty / Decimal("2"))

            # Cost savings
            annual_cost_savings = risk_adjusted * price
            if data.include_carbon_savings:
                co2_savings = risk_adjusted * data.co2_factor_kg_kwh / Decimal("1000")
                carbon_cost_savings = co2_savings * data.carbon_price_eur_tonne
                annual_cost_savings = annual_cost_savings + carbon_cost_savings
            else:
                co2_savings = risk_adjusted * data.co2_factor_kg_kwh / Decimal("1000")

            # Financial analysis
            financial = self._financial_analysis(
                capex=m.implementation_cost_eur,
                annual_savings=annual_cost_savings,
                annual_maintenance=m.annual_maintenance_eur,
                lifetime=m.lifetime_years,
                discount_rate=discount,
                escalation=escalation,
                savings_kwh=risk_adjusted,
            )

            # IPMVP plan
            ipmvp: Optional[IPMVPPlan] = None
            if data.ipmvp_required:
                ipmvp = self._generate_ipmvp_plan(m, risk_adjusted)

            # Priority scoring
            priority_score = self._calculate_priority(
                financial, m, risk_adjusted, data.max_payback_years
            )
            priority = self._classify_priority(priority_score)
            phase = self._assign_phase(priority, financial.simple_payback_years)

            # Budget check
            within_budget = True
            if data.budget_eur > Decimal("0"):
                within_budget = m.implementation_cost_eur <= data.budget_eur

            measure_results.append(MeasureResult(
                measure_id=m.measure_id,
                name=m.name,
                category=m.category,
                adjusted_savings_kwh=_round_val(adjusted_savings, 2),
                risk_adjusted_savings_kwh=_round_val(risk_adjusted, 2),
                savings_lower_bound_kwh=_round_val(lower, 2),
                savings_upper_bound_kwh=_round_val(upper, 2),
                annual_cost_savings_eur=_round_val(annual_cost_savings, 2),
                annual_carbon_savings_tco2e=_round_val(co2_savings, 3),
                financial=financial,
                ipmvp_plan=ipmvp,
                priority=priority,
                priority_score=_round_val(priority_score, 2),
                implementation_phase=phase,
                within_budget=within_budget,
            ))

        # Step 2: Totals
        total_savings = sum((mr.risk_adjusted_savings_kwh for mr in measure_results), Decimal("0"))
        total_cost = sum((m.implementation_cost_eur for m in data.measures), Decimal("0"))
        total_annual_eur = sum((mr.annual_cost_savings_eur for mr in measure_results), Decimal("0"))
        total_co2 = sum((mr.annual_carbon_savings_tco2e for mr in measure_results), Decimal("0"))
        savings_pct = _safe_pct(total_savings, data.total_baseline_kwh)

        # Step 3: Portfolio-level financial
        portfolio_npv = sum((mr.financial.npv_eur for mr in measure_results), Decimal("0"))
        portfolio_payback = _safe_divide(total_cost, total_annual_eur, Decimal("99"))

        # Portfolio IRR (simplified: use weighted average)
        weighted_irr = Decimal("0")
        if total_cost > Decimal("0"):
            for mr in measure_results:
                weight = _safe_divide(
                    _decimal(next(
                        (m.implementation_cost_eur for m in data.measures
                         if m.measure_id == mr.measure_id), Decimal("0")
                    )),
                    total_cost
                )
                weighted_irr += mr.financial.irr_pct * weight

        # Step 4: MACC generation
        macc_data = self._generate_macc(measure_results)

        # Step 5: Implementation roadmap
        roadmap = self._build_roadmap(measure_results, data)

        # Step 6: Budget
        within_budget = True
        budget_util = Decimal("0")
        if data.budget_eur > Decimal("0"):
            within_budget = total_cost <= data.budget_eur
            budget_util = _safe_pct(total_cost, data.budget_eur)
            if not within_budget:
                warnings.append(
                    f"Total cost ({_round_val(total_cost, 0)} EUR) exceeds "
                    f"budget ({_round_val(data.budget_eur, 0)} EUR). "
                    f"Consider phased implementation."
                )

        # Step 7: Recommendations
        if total_savings > Decimal("0"):
            recommendations.append(
                f"Total identified savings: {_round_val(total_savings, 0)} kWh/year "
                f"({_round_val(savings_pct, 1)}% of baseline), "
                f"worth {_round_val(total_annual_eur, 0)} EUR/year."
            )

        quick_wins = [mr for mr in measure_results if mr.financial.simple_payback_years <= Decimal("2")]
        if quick_wins:
            qw_savings = sum((mr.risk_adjusted_savings_kwh for mr in quick_wins), Decimal("0"))
            recommendations.append(
                f"{len(quick_wins)} quick-win measures with payback under 2 years, "
                f"delivering {_round_val(qw_savings, 0)} kWh/year savings."
            )

        neg_cost = [mr for mr in measure_results if mr.financial.lcoe_eur_kwh < Decimal("0")]
        if neg_cost:
            recommendations.append(
                f"{len(neg_cost)} measures have negative cost of saved energy "
                f"(savings exceed implementation cost within first year)."
            )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = EnergySavingsResult(
            facility_id=data.facility_id,
            facility_name=data.facility_name,
            total_measures=len(measure_results),
            total_savings_kwh=_round_val(total_savings, 2),
            total_savings_pct=_round_val(savings_pct, 2),
            total_cost_eur=_round_val(total_cost, 2),
            total_annual_savings_eur=_round_val(total_annual_eur, 2),
            total_carbon_savings_tco2e=_round_val(total_co2, 3),
            portfolio_npv_eur=_round_val(portfolio_npv, 2),
            portfolio_irr_pct=_round_val(weighted_irr, 2),
            portfolio_payback_years=_round_val(portfolio_payback, 2),
            measure_results=measure_results,
            macc_data=macc_data,
            interaction_adjustments=interaction_log,
            implementation_roadmap=roadmap,
            within_budget=within_budget,
            budget_utilization_pct=_round_val(budget_util, 2),
            recommendations=recommendations,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Energy savings analysis complete: %d measures, %.0f kWh saved (%.1f%%), "
            "NPV=%.0f EUR, hash=%s",
            len(measure_results), float(total_savings), float(savings_pct),
            float(portfolio_npv), result.provenance_hash[:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # Financial Analysis                                                   #
    # ------------------------------------------------------------------ #

    def _financial_analysis(
        self,
        capex: Decimal,
        annual_savings: Decimal,
        annual_maintenance: Decimal,
        lifetime: int,
        discount_rate: Decimal,
        escalation: Decimal,
        savings_kwh: Decimal,
    ) -> FinancialAnalysis:
        """Perform NPV, IRR, payback, LCOE financial analysis.

        Args:
            capex: Implementation cost (EUR).
            annual_savings: Annual cost savings (EUR).
            annual_maintenance: Annual maintenance cost (EUR).
            lifetime: Economic lifetime (years).
            discount_rate: Discount rate.
            escalation: Annual energy price escalation.
            savings_kwh: Annual energy savings (kWh).

        Returns:
            FinancialAnalysis with all metrics.
        """
        net_annual = annual_savings - annual_maintenance

        # Simple payback
        simple_payback = _safe_divide(capex, net_annual, Decimal("99"))

        # NPV and discounted payback
        npv = -capex
        cumulative_discounted = Decimal("0")
        discounted_payback = _decimal(lifetime)
        total_undiscounted_savings = Decimal("0")
        payback_found = False

        for t in range(1, lifetime + 1):
            escalated_savings = net_annual * (Decimal("1") + escalation) ** _decimal(t - 1)
            discount_factor = (Decimal("1") + discount_rate) ** _decimal(t)
            pv = _safe_divide(escalated_savings, discount_factor)
            npv += pv
            cumulative_discounted += pv
            total_undiscounted_savings += escalated_savings

            if not payback_found and cumulative_discounted >= capex:
                # Interpolate discounted payback year
                prev_cum = cumulative_discounted - pv
                remaining = capex - prev_cum
                fraction = _safe_divide(remaining, pv)
                discounted_payback = _decimal(t - 1) + fraction
                payback_found = True

        # IRR via bisection
        irr = self._calculate_irr(capex, net_annual, escalation, lifetime)

        # LCOE
        total_cost = capex + annual_maintenance * _decimal(lifetime)
        total_kwh_saved = savings_kwh * _decimal(lifetime)
        lcoe = _safe_divide(total_cost, total_kwh_saved)

        # ROI
        roi = _safe_pct(total_undiscounted_savings - capex, capex)

        return FinancialAnalysis(
            npv_eur=_round_val(npv, 2),
            irr_pct=_round_val(irr, 2),
            simple_payback_years=_round_val(simple_payback, 2),
            discounted_payback_years=_round_val(discounted_payback, 2),
            lcoe_eur_kwh=_round_val(lcoe, 4),
            roi_pct=_round_val(roi, 2),
            discount_rate=discount_rate,
            total_lifetime_savings_eur=_round_val(total_undiscounted_savings, 2),
            total_lifetime_cost_eur=_round_val(total_cost, 2),
        )

    def _calculate_irr(
        self,
        capex: Decimal,
        annual_net: Decimal,
        escalation: Decimal,
        lifetime: int,
    ) -> Decimal:
        """Calculate IRR using bisection method.

        Finds the rate r such that NPV(r) = 0.

        Args:
            capex: Initial investment.
            annual_net: Annual net savings.
            escalation: Annual escalation rate.
            lifetime: Number of years.

        Returns:
            IRR as percentage.
        """
        if capex <= Decimal("0") or annual_net <= Decimal("0"):
            return Decimal("0")

        lo = Decimal("-0.50")
        hi = Decimal("5.00")

        for _ in range(MAX_IRR_ITERATIONS):
            mid = (lo + hi) / Decimal("2")
            npv_mid = -capex
            for t in range(1, lifetime + 1):
                cf = annual_net * (Decimal("1") + escalation) ** _decimal(t - 1)
                denom = (Decimal("1") + mid) ** _decimal(t)
                if denom != Decimal("0"):
                    npv_mid += _safe_divide(cf, denom)

            if abs(npv_mid) < Decimal("1"):  # Close enough
                return mid * Decimal("100")
            elif npv_mid > Decimal("0"):
                lo = mid
            else:
                hi = mid

            if abs(hi - lo) < IRR_TOLERANCE:
                break

        return ((lo + hi) / Decimal("2")) * Decimal("100")

    # ------------------------------------------------------------------ #
    # IPMVP Plan                                                           #
    # ------------------------------------------------------------------ #

    def _generate_ipmvp_plan(
        self, measure: EnergySavingsMeasure, savings_kwh: Decimal,
    ) -> IPMVPPlan:
        """Generate IPMVP measurement and verification plan.

        Selects the most appropriate IPMVP option based on ECM category,
        savings magnitude, and complexity.

        Args:
            measure: ECM input data.
            savings_kwh: Expected annual savings (kWh).

        Returns:
            IPMVPPlan.
        """
        # Select option based on category and savings magnitude
        category = measure.category
        savings_f = float(savings_kwh)

        selected = IPMVPOption.OPTION_A.value
        for opt_key, criteria in IPMVP_OPTION_CRITERIA.items():
            if category in criteria["best_for"] and savings_f >= criteria["min_savings_kwh"]:
                selected = opt_key

        # Override for very large or complex measures
        if savings_f >= 200000 or measure.complexity == ImplementationComplexity.VERY_HIGH.value:
            selected = IPMVPOption.OPTION_D.value
        elif savings_f >= 100000 and measure.complexity in (
            ImplementationComplexity.HIGH.value, ImplementationComplexity.VERY_HIGH.value
        ):
            selected = IPMVPOption.OPTION_C.value

        opt_info = IPMVP_OPTION_CRITERIA.get(selected, {})

        # Key parameters by category
        params_by_category: Dict[str, List[str]] = {
            "lighting": ["lighting power density (W/m2)", "operating hours", "lumen output"],
            "motors": ["motor current (A)", "power factor", "operating hours", "speed (rpm)"],
            "compressed_air": ["flow rate (m3/min)", "pressure (bar)", "power (kW)", "leak rate"],
            "hvac": ["supply air temperature", "return air temperature", "flow rate", "runtime hours"],
            "boiler": ["fuel consumption", "steam output", "stack temperature", "excess O2"],
            "process_heat": ["process temperature", "fuel input", "product throughput"],
            "building_envelope": ["indoor temperature", "outdoor temperature", "energy consumption"],
            "controls": ["setpoints", "schedules", "energy consumption"],
            "waste_heat": ["exhaust temperature", "flow rate", "recovered energy"],
            "refrigeration": ["suction pressure", "discharge pressure", "compressor power", "temperature"],
        }
        key_params = params_by_category.get(category, ["energy consumption", "operating hours"])

        # M&V cost estimate (typically 3-10% of savings value)
        mv_cost_factor = Decimal("0.05")
        if selected == IPMVPOption.OPTION_D.value:
            mv_cost_factor = Decimal("0.10")
        elif selected == IPMVPOption.OPTION_B.value:
            mv_cost_factor = Decimal("0.07")
        mv_cost = measure.implementation_cost_eur * mv_cost_factor
        mv_cost = max(mv_cost, Decimal("1000"))  # Minimum M&V cost

        return IPMVPPlan(
            option=selected,
            option_name=opt_info.get("name", selected),
            measurement_boundary=f"{measure.system_affected or measure.name} system boundary",
            baseline_period="12 months pre-implementation",
            post_period="12 months post-implementation",
            key_parameters=key_params,
            sampling_approach="Continuous metering" if selected in (
                IPMVPOption.OPTION_B.value, IPMVPOption.OPTION_C.value
            ) else "Spot measurements with stipulated values",
            mv_cost_eur=_round_val(mv_cost, 2),
            rationale=opt_info.get("description", ""),
        )

    # ------------------------------------------------------------------ #
    # Priority Scoring                                                     #
    # ------------------------------------------------------------------ #

    def _calculate_priority(
        self,
        financial: FinancialAnalysis,
        measure: EnergySavingsMeasure,
        savings_kwh: Decimal,
        max_payback: Decimal,
    ) -> Decimal:
        """Calculate multi-criteria priority score (0-100).

        Scoring weights:
            NPV contribution:       25%
            Simple payback:         25%
            Complexity (inverse):   20%
            Savings magnitude:      15%
            Confidence:             15%

        Args:
            financial: Financial analysis results.
            measure: ECM input data.
            savings_kwh: Risk-adjusted savings.
            max_payback: Maximum acceptable payback.

        Returns:
            Priority score (0-100).
        """
        score = Decimal("0")

        # NPV score (25 pts): positive NPV = good
        if financial.npv_eur > Decimal("0"):
            npv_score = min(
                financial.npv_eur / measure.implementation_cost_eur * Decimal("25")
                if measure.implementation_cost_eur > Decimal("0")
                else Decimal("25"),
                Decimal("25")
            )
        else:
            npv_score = Decimal("0")
        score += npv_score

        # Payback score (25 pts): shorter = better
        if financial.simple_payback_years <= Decimal("0"):
            payback_score = Decimal("0")
        elif financial.simple_payback_years <= Decimal("1"):
            payback_score = Decimal("25")
        elif financial.simple_payback_years <= max_payback:
            payback_score = Decimal("25") * (
                Decimal("1") - financial.simple_payback_years / max_payback
            )
        else:
            payback_score = Decimal("0")
        score += payback_score

        # Complexity score (20 pts): lower complexity = higher score
        complexity_val = COMPLEXITY_SCORES.get(measure.complexity, Decimal("2"))
        complexity_score = Decimal("20") * (Decimal("1") - (complexity_val - Decimal("1")) / Decimal("3"))
        score += complexity_score

        # Savings magnitude score (15 pts)
        if savings_kwh > Decimal("500000"):
            mag_score = Decimal("15")
        elif savings_kwh > Decimal("100000"):
            mag_score = Decimal("12")
        elif savings_kwh > Decimal("50000"):
            mag_score = Decimal("9")
        elif savings_kwh > Decimal("10000"):
            mag_score = Decimal("6")
        else:
            mag_score = Decimal("3")
        score += mag_score

        # Confidence score (15 pts)
        conf = CONFIDENCE_FACTORS.get(measure.confidence_level, Decimal("0.70"))
        conf_score = conf * Decimal("15")
        score += conf_score

        return min(score, Decimal("100"))

    def _classify_priority(self, score: Decimal) -> str:
        """Classify priority from score.

        Args:
            score: Priority score (0-100).

        Returns:
            PriorityLevel value.
        """
        if score >= Decimal("80"):
            return PriorityLevel.CRITICAL.value
        elif score >= Decimal("60"):
            return PriorityLevel.HIGH.value
        elif score >= Decimal("40"):
            return PriorityLevel.MEDIUM.value
        elif score >= Decimal("20"):
            return PriorityLevel.LOW.value
        else:
            return PriorityLevel.DEFERRED.value

    def _assign_phase(self, priority: str, payback: Decimal) -> int:
        """Assign implementation phase.

        Phase 1: Immediate (0-6 months) - quick wins.
        Phase 2: Near-term (6-18 months) - moderate projects.
        Phase 3: Medium-term (18-36 months) - capital projects.

        Args:
            priority: Priority level.
            payback: Simple payback (years).

        Returns:
            Phase number (1-3).
        """
        if priority in (PriorityLevel.CRITICAL.value,) or payback <= Decimal("1"):
            return 1
        elif priority in (PriorityLevel.HIGH.value,) or payback <= Decimal("3"):
            return 2
        else:
            return 3

    # ------------------------------------------------------------------ #
    # MACC Generation                                                      #
    # ------------------------------------------------------------------ #

    def _generate_macc(
        self, results: List[MeasureResult],
    ) -> List[MACCPoint]:
        """Generate Marginal Abatement Cost Curve data.

        MACC sorts measures by cost per unit of energy saved (ascending),
        creating a waterfall chart of cumulative abatement.

        Args:
            results: Measure analysis results.

        Returns:
            List of MACCPoint sorted by cost per kWh.
        """
        points: List[MACCPoint] = []
        for mr in results:
            if mr.risk_adjusted_savings_kwh > Decimal("0"):
                cost_per_kwh = _safe_divide(
                    mr.financial.lcoe_eur_kwh,
                    Decimal("1"),
                    Decimal("0")
                )
                points.append(MACCPoint(
                    measure_id=mr.measure_id,
                    measure_name=mr.name,
                    abatement_kwh=mr.risk_adjusted_savings_kwh,
                    cost_per_kwh_saved=_round_val(cost_per_kwh, 4),
                    co2_abatement_tco2e=mr.annual_carbon_savings_tco2e,
                ))

        # Sort by cost per kWh (ascending)
        points.sort(key=lambda p: p.cost_per_kwh_saved)

        # Calculate cumulative
        cumulative = Decimal("0")
        for p in points:
            cumulative += p.abatement_kwh
            p.cumulative_abatement_kwh = _round_val(cumulative, 2)

        return points

    # ------------------------------------------------------------------ #
    # Implementation Roadmap                                               #
    # ------------------------------------------------------------------ #

    def _build_roadmap(
        self,
        results: List[MeasureResult],
        data: EnergySavingsInput,
    ) -> List[RoadmapPhase]:
        """Build phased implementation roadmap.

        Args:
            results: Measure results with assigned phases.
            data: Input data.

        Returns:
            List of RoadmapPhase.
        """
        phase_names = {
            1: "Quick Wins (0-6 months)",
            2: "Near-Term Projects (6-18 months)",
            3: "Capital Projects (18-36 months)",
        }

        measure_costs: Dict[str, Decimal] = {
            m.measure_id: m.implementation_cost_eur for m in data.measures
        }

        phases: List[RoadmapPhase] = []
        cumulative = Decimal("0")

        for phase_num in (1, 2, 3):
            phase_measures = [
                mr for mr in results if mr.implementation_phase == phase_num
            ]
            if not phase_measures:
                continue

            phase_cost = sum(
                (measure_costs.get(mr.measure_id, Decimal("0")) for mr in phase_measures),
                Decimal("0")
            )
            phase_savings = sum(
                (mr.risk_adjusted_savings_kwh for mr in phase_measures),
                Decimal("0")
            )
            phase_savings_eur = sum(
                (mr.annual_cost_savings_eur for mr in phase_measures),
                Decimal("0")
            )
            cumulative += phase_savings

            phases.append(RoadmapPhase(
                phase=phase_num,
                phase_name=phase_names.get(phase_num, f"Phase {phase_num}"),
                measures=[mr.measure_id for mr in phase_measures],
                total_cost_eur=_round_val(phase_cost, 2),
                total_savings_kwh=_round_val(phase_savings, 2),
                total_savings_eur=_round_val(phase_savings_eur, 2),
                cumulative_savings_kwh=_round_val(cumulative, 2),
            ))

        return phases
