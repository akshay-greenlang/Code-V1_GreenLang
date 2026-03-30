# -*- coding: utf-8 -*-
"""
QuickWinsScannerEngine - PACK-033 Quick Wins Identifier Engine 1
=================================================================

Automated facility/process scanning for quick-win energy-efficiency
opportunities.  Maintains a library of 80+ pre-defined quick-win
actions across 15 categories, scores each action against a facility
profile and equipment survey, estimates energy/cost/CO2e savings,
and returns a prioritised list with full provenance tracking.

Calculation Methodology:
    Applicability Score (0-100):
        Base score from building-type match and equipment-survey flags,
        adjusted by equipment age and operating-hours multipliers.

    Savings Estimation:
        estimated_savings_kwh = annual_energy_kwh
                                * category_share
                                * typical_savings_pct / 100
        estimated_savings_cost = estimated_savings_kwh * unit_energy_cost

    CO2e Reduction:
        co2e_kg = estimated_savings_kwh * grid_emission_factor_kg_kwh

    Payback:
        payback_months = implementation_cost / (monthly_savings_cost)

    Priority Assignment:
        Multi-criteria ranking on applicability, savings magnitude,
        payback period, and disruption level.

Regulatory References:
    - ISO 50001:2018 - Energy management systems
    - EN 16247-1:2022 - Energy audits (general requirements)
    - EN 16247-2:2022 - Energy audits (buildings)
    - EU EED Article 8 - Mandatory energy audits
    - ASHRAE Level 1 Walk-Through Analysis
    - DOE Better Buildings - Quick Win guidance

Zero-Hallucination:
    - All savings percentages sourced from DOE/ASHRAE published data
    - Deterministic Decimal arithmetic throughout
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-033 Quick Wins Identifier
Engine:  1 of 8
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
            if k not in ("calculated_at", "scan_duration_ms", "provenance_hash")
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

class BuildingType(str, Enum):
    """Building / facility type.

    OFFICE: Commercial office space.
    MANUFACTURING: Industrial manufacturing facility.
    RETAIL: Retail store / shopping centre.
    WAREHOUSE: Storage / distribution warehouse.
    HEALTHCARE: Hospital or medical facility.
    EDUCATION: School, college, or university.
    DATA_CENTER: Data centre / server farm.
    HOSPITALITY: Hotel or resort.
    RESTAURANT: Food-service / restaurant.
    GROCERY: Grocery or supermarket.
    MULTIFAMILY: Multi-family residential.
    LABORATORY: Research or testing laboratory.
    WORSHIP: Place of worship.
    FITNESS: Gym or fitness centre.
    MIXED_USE: Mixed-use building.
    """
    OFFICE = "office"
    MANUFACTURING = "manufacturing"
    RETAIL = "retail"
    WAREHOUSE = "warehouse"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    DATA_CENTER = "data_center"
    HOSPITALITY = "hospitality"
    RESTAURANT = "restaurant"
    GROCERY = "grocery"
    MULTIFAMILY = "multifamily"
    LABORATORY = "laboratory"
    WORSHIP = "worship"
    FITNESS = "fitness"
    MIXED_USE = "mixed_use"

class ActionCategory(str, Enum):
    """Quick-win action category.

    LIGHTING: Lighting upgrades and controls.
    HVAC: HVAC system improvements.
    ENVELOPE: Building envelope measures.
    PLUG_LOADS: Plug load management.
    WATER_HEATING: Water heating efficiency.
    COMPRESSED_AIR: Compressed air optimisation.
    MOTORS_DRIVES: Motor and drive efficiency.
    REFRIGERATION: Refrigeration improvements.
    KITCHEN: Commercial kitchen efficiency.
    LAUNDRY: Laundry system efficiency.
    CONTROLS: Building automation and controls.
    BEHAVIORAL: Behavioural / occupant engagement.
    MAINTENANCE: Preventive maintenance actions.
    RENEWABLE: Renewable energy opportunities.
    PROCESS: Process optimisation.
    """
    LIGHTING = "lighting"
    HVAC = "hvac"
    ENVELOPE = "envelope"
    PLUG_LOADS = "plug_loads"
    WATER_HEATING = "water_heating"
    COMPRESSED_AIR = "compressed_air"
    MOTORS_DRIVES = "motors_drives"
    REFRIGERATION = "refrigeration"
    KITCHEN = "kitchen"
    LAUNDRY = "laundry"
    CONTROLS = "controls"
    BEHAVIORAL = "behavioral"
    MAINTENANCE = "maintenance"
    RENEWABLE = "renewable"
    PROCESS = "process"

class ActionComplexity(str, Enum):
    """Implementation complexity / cost tier.

    NO_COST: Zero or negligible cost (behavioural change).
    LOW_COST: Under 5 000 EUR - minor materials or labour.
    MEDIUM_COST: 5 000 - 50 000 EUR - equipment or contractor work.
    CAPITAL: Over 50 000 EUR - significant capital expenditure.
    """
    NO_COST = "no_cost"
    LOW_COST = "low_cost"
    MEDIUM_COST = "medium_cost"
    CAPITAL = "capital"

class ActionPriority(str, Enum):
    """Quick-win action priority.

    CRITICAL: Immediate implementation recommended (<3 month payback).
    HIGH: Near-term implementation (3-12 month payback).
    MEDIUM: Planned implementation (12-24 month payback).
    LOW: Future consideration (>24 month payback).
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ScanStatus(str, Enum):
    """Facility scan status.

    PENDING: Scan queued but not started.
    IN_PROGRESS: Scan currently running.
    COMPLETED: Scan finished successfully.
    FAILED: Scan failed with errors.
    """
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class DisruptionLevel(str, Enum):
    """Operational disruption level during implementation.

    NONE: No disruption to operations.
    MINIMAL: Brief interruption (< 1 hour).
    MODERATE: Partial disruption (hours to 1 day).
    SIGNIFICANT: Major disruption (multiple days / shutdown).
    """
    NONE = "none"
    MINIMAL = "minimal"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CO2_FACTOR_KG_KWH: Decimal = Decimal("0.4")
DEFAULT_ENERGY_PRICE_EUR_KWH: Decimal = Decimal("0.15")

# Typical energy category share by building type (fraction of total energy).
# Source: DOE Commercial Buildings Energy Consumption Survey (CBECS) /
#         EIA Manufacturing Energy Consumption Survey (MECS).
CATEGORY_ENERGY_SHARE: Dict[str, Dict[str, Decimal]] = {
    BuildingType.OFFICE.value: {
        ActionCategory.LIGHTING.value: Decimal("0.25"),
        ActionCategory.HVAC.value: Decimal("0.35"),
        ActionCategory.PLUG_LOADS.value: Decimal("0.20"),
        ActionCategory.WATER_HEATING.value: Decimal("0.05"),
        ActionCategory.ENVELOPE.value: Decimal("0.05"),
        ActionCategory.CONTROLS.value: Decimal("0.05"),
        ActionCategory.BEHAVIORAL.value: Decimal("0.03"),
        ActionCategory.MAINTENANCE.value: Decimal("0.02"),
    },
    BuildingType.MANUFACTURING.value: {
        ActionCategory.LIGHTING.value: Decimal("0.08"),
        ActionCategory.HVAC.value: Decimal("0.12"),
        ActionCategory.COMPRESSED_AIR.value: Decimal("0.15"),
        ActionCategory.MOTORS_DRIVES.value: Decimal("0.25"),
        ActionCategory.PROCESS.value: Decimal("0.20"),
        ActionCategory.MAINTENANCE.value: Decimal("0.08"),
        ActionCategory.CONTROLS.value: Decimal("0.05"),
        ActionCategory.ENVELOPE.value: Decimal("0.04"),
        ActionCategory.BEHAVIORAL.value: Decimal("0.03"),
    },
    BuildingType.RETAIL.value: {
        ActionCategory.LIGHTING.value: Decimal("0.30"),
        ActionCategory.HVAC.value: Decimal("0.30"),
        ActionCategory.PLUG_LOADS.value: Decimal("0.10"),
        ActionCategory.REFRIGERATION.value: Decimal("0.10"),
        ActionCategory.WATER_HEATING.value: Decimal("0.05"),
        ActionCategory.ENVELOPE.value: Decimal("0.05"),
        ActionCategory.CONTROLS.value: Decimal("0.05"),
        ActionCategory.BEHAVIORAL.value: Decimal("0.03"),
        ActionCategory.MAINTENANCE.value: Decimal("0.02"),
    },
    BuildingType.WAREHOUSE.value: {
        ActionCategory.LIGHTING.value: Decimal("0.35"),
        ActionCategory.HVAC.value: Decimal("0.20"),
        ActionCategory.ENVELOPE.value: Decimal("0.15"),
        ActionCategory.MOTORS_DRIVES.value: Decimal("0.10"),
        ActionCategory.PLUG_LOADS.value: Decimal("0.05"),
        ActionCategory.CONTROLS.value: Decimal("0.05"),
        ActionCategory.MAINTENANCE.value: Decimal("0.05"),
        ActionCategory.BEHAVIORAL.value: Decimal("0.03"),
        ActionCategory.WATER_HEATING.value: Decimal("0.02"),
    },
    BuildingType.HEALTHCARE.value: {
        ActionCategory.HVAC.value: Decimal("0.35"),
        ActionCategory.LIGHTING.value: Decimal("0.20"),
        ActionCategory.PLUG_LOADS.value: Decimal("0.10"),
        ActionCategory.WATER_HEATING.value: Decimal("0.10"),
        ActionCategory.LAUNDRY.value: Decimal("0.05"),
        ActionCategory.KITCHEN.value: Decimal("0.05"),
        ActionCategory.CONTROLS.value: Decimal("0.05"),
        ActionCategory.ENVELOPE.value: Decimal("0.04"),
        ActionCategory.MAINTENANCE.value: Decimal("0.03"),
        ActionCategory.BEHAVIORAL.value: Decimal("0.03"),
    },
    BuildingType.EDUCATION.value: {
        ActionCategory.HVAC.value: Decimal("0.35"),
        ActionCategory.LIGHTING.value: Decimal("0.25"),
        ActionCategory.PLUG_LOADS.value: Decimal("0.10"),
        ActionCategory.WATER_HEATING.value: Decimal("0.08"),
        ActionCategory.ENVELOPE.value: Decimal("0.07"),
        ActionCategory.CONTROLS.value: Decimal("0.05"),
        ActionCategory.KITCHEN.value: Decimal("0.04"),
        ActionCategory.BEHAVIORAL.value: Decimal("0.04"),
        ActionCategory.MAINTENANCE.value: Decimal("0.02"),
    },
    BuildingType.DATA_CENTER.value: {
        ActionCategory.HVAC.value: Decimal("0.40"),
        ActionCategory.PLUG_LOADS.value: Decimal("0.30"),
        ActionCategory.LIGHTING.value: Decimal("0.05"),
        ActionCategory.CONTROLS.value: Decimal("0.10"),
        ActionCategory.ENVELOPE.value: Decimal("0.05"),
        ActionCategory.MAINTENANCE.value: Decimal("0.05"),
        ActionCategory.BEHAVIORAL.value: Decimal("0.03"),
        ActionCategory.WATER_HEATING.value: Decimal("0.02"),
    },
    BuildingType.HOSPITALITY.value: {
        ActionCategory.HVAC.value: Decimal("0.30"),
        ActionCategory.LIGHTING.value: Decimal("0.15"),
        ActionCategory.WATER_HEATING.value: Decimal("0.15"),
        ActionCategory.LAUNDRY.value: Decimal("0.10"),
        ActionCategory.KITCHEN.value: Decimal("0.10"),
        ActionCategory.PLUG_LOADS.value: Decimal("0.05"),
        ActionCategory.ENVELOPE.value: Decimal("0.05"),
        ActionCategory.CONTROLS.value: Decimal("0.04"),
        ActionCategory.BEHAVIORAL.value: Decimal("0.04"),
        ActionCategory.MAINTENANCE.value: Decimal("0.02"),
    },
    BuildingType.RESTAURANT.value: {
        ActionCategory.KITCHEN.value: Decimal("0.30"),
        ActionCategory.HVAC.value: Decimal("0.25"),
        ActionCategory.LIGHTING.value: Decimal("0.15"),
        ActionCategory.WATER_HEATING.value: Decimal("0.10"),
        ActionCategory.REFRIGERATION.value: Decimal("0.10"),
        ActionCategory.PLUG_LOADS.value: Decimal("0.04"),
        ActionCategory.CONTROLS.value: Decimal("0.03"),
        ActionCategory.BEHAVIORAL.value: Decimal("0.02"),
        ActionCategory.MAINTENANCE.value: Decimal("0.01"),
    },
    BuildingType.GROCERY.value: {
        ActionCategory.REFRIGERATION.value: Decimal("0.40"),
        ActionCategory.LIGHTING.value: Decimal("0.20"),
        ActionCategory.HVAC.value: Decimal("0.15"),
        ActionCategory.KITCHEN.value: Decimal("0.05"),
        ActionCategory.PLUG_LOADS.value: Decimal("0.05"),
        ActionCategory.WATER_HEATING.value: Decimal("0.05"),
        ActionCategory.CONTROLS.value: Decimal("0.04"),
        ActionCategory.ENVELOPE.value: Decimal("0.03"),
        ActionCategory.MAINTENANCE.value: Decimal("0.02"),
        ActionCategory.BEHAVIORAL.value: Decimal("0.01"),
    },
    BuildingType.MULTIFAMILY.value: {
        ActionCategory.HVAC.value: Decimal("0.30"),
        ActionCategory.WATER_HEATING.value: Decimal("0.20"),
        ActionCategory.LIGHTING.value: Decimal("0.15"),
        ActionCategory.PLUG_LOADS.value: Decimal("0.10"),
        ActionCategory.ENVELOPE.value: Decimal("0.10"),
        ActionCategory.LAUNDRY.value: Decimal("0.05"),
        ActionCategory.CONTROLS.value: Decimal("0.04"),
        ActionCategory.BEHAVIORAL.value: Decimal("0.04"),
        ActionCategory.MAINTENANCE.value: Decimal("0.02"),
    },
    BuildingType.LABORATORY.value: {
        ActionCategory.HVAC.value: Decimal("0.40"),
        ActionCategory.PLUG_LOADS.value: Decimal("0.20"),
        ActionCategory.LIGHTING.value: Decimal("0.12"),
        ActionCategory.WATER_HEATING.value: Decimal("0.08"),
        ActionCategory.CONTROLS.value: Decimal("0.07"),
        ActionCategory.ENVELOPE.value: Decimal("0.05"),
        ActionCategory.MAINTENANCE.value: Decimal("0.04"),
        ActionCategory.BEHAVIORAL.value: Decimal("0.04"),
    },
    BuildingType.WORSHIP.value: {
        ActionCategory.HVAC.value: Decimal("0.40"),
        ActionCategory.LIGHTING.value: Decimal("0.30"),
        ActionCategory.ENVELOPE.value: Decimal("0.10"),
        ActionCategory.PLUG_LOADS.value: Decimal("0.05"),
        ActionCategory.WATER_HEATING.value: Decimal("0.05"),
        ActionCategory.CONTROLS.value: Decimal("0.04"),
        ActionCategory.BEHAVIORAL.value: Decimal("0.04"),
        ActionCategory.MAINTENANCE.value: Decimal("0.02"),
    },
    BuildingType.FITNESS.value: {
        ActionCategory.HVAC.value: Decimal("0.30"),
        ActionCategory.WATER_HEATING.value: Decimal("0.20"),
        ActionCategory.LIGHTING.value: Decimal("0.15"),
        ActionCategory.PLUG_LOADS.value: Decimal("0.10"),
        ActionCategory.LAUNDRY.value: Decimal("0.08"),
        ActionCategory.ENVELOPE.value: Decimal("0.05"),
        ActionCategory.CONTROLS.value: Decimal("0.05"),
        ActionCategory.BEHAVIORAL.value: Decimal("0.04"),
        ActionCategory.MAINTENANCE.value: Decimal("0.03"),
    },
    BuildingType.MIXED_USE.value: {
        ActionCategory.HVAC.value: Decimal("0.30"),
        ActionCategory.LIGHTING.value: Decimal("0.20"),
        ActionCategory.PLUG_LOADS.value: Decimal("0.12"),
        ActionCategory.WATER_HEATING.value: Decimal("0.08"),
        ActionCategory.ENVELOPE.value: Decimal("0.08"),
        ActionCategory.CONTROLS.value: Decimal("0.06"),
        ActionCategory.REFRIGERATION.value: Decimal("0.05"),
        ActionCategory.BEHAVIORAL.value: Decimal("0.04"),
        ActionCategory.KITCHEN.value: Decimal("0.04"),
        ActionCategory.MAINTENANCE.value: Decimal("0.03"),
    },
}

# Complexity to typical implementation cost (EUR per m2 of floor area).
COMPLEXITY_COST_PER_M2: Dict[str, Decimal] = {
    ActionComplexity.NO_COST.value: Decimal("0"),
    ActionComplexity.LOW_COST.value: Decimal("2"),
    ActionComplexity.MEDIUM_COST.value: Decimal("10"),
    ActionComplexity.CAPITAL.value: Decimal("40"),
}

# Disruption numeric weight for priority scoring.
DISRUPTION_WEIGHT: Dict[str, Decimal] = {
    DisruptionLevel.NONE.value: Decimal("1.0"),
    DisruptionLevel.MINIMAL.value: Decimal("0.9"),
    DisruptionLevel.MODERATE.value: Decimal("0.7"),
    DisruptionLevel.SIGNIFICANT.value: Decimal("0.4"),
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class FacilityProfile(BaseModel):
    """Facility profile for quick-win scanning.

    Attributes:
        facility_id: Unique facility identifier.
        name: Facility name.
        building_type: Building type classification.
        floor_area_m2: Gross floor area in square metres.
        operating_hours: Annual operating hours.
        occupancy: Typical occupancy count (persons).
        energy_carriers: List of energy sources (electricity, gas, etc.).
        annual_energy_kwh: Total annual energy consumption (kWh).
        annual_energy_cost: Total annual energy cost (EUR).
        climate_zone: ASHRAE / Koppen climate zone identifier.
        year_built: Year the building was constructed.
        equipment_age_years: Average age of major equipment (years).
    """
    facility_id: str = Field(default_factory=_new_uuid, description="Facility ID")
    name: str = Field(default="", max_length=300, description="Facility name")
    building_type: str = Field(
        default=BuildingType.OFFICE.value, description="Building type"
    )
    floor_area_m2: Decimal = Field(
        default=Decimal("1000"), ge=0, description="Floor area (m2)"
    )
    operating_hours: int = Field(
        default=2500, ge=0, le=8760, description="Annual operating hours"
    )
    occupancy: int = Field(default=50, ge=0, description="Typical occupancy")
    energy_carriers: List[str] = Field(
        default_factory=lambda: ["electricity"],
        description="Energy sources",
    )
    annual_energy_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Annual energy (kWh)"
    )
    annual_energy_cost: Decimal = Field(
        default=Decimal("0"), ge=0, description="Annual energy cost (EUR)"
    )
    climate_zone: str = Field(default="4A", max_length=10, description="Climate zone")
    year_built: int = Field(default=2000, ge=1800, le=2030, description="Year built")
    equipment_age_years: int = Field(
        default=10, ge=0, le=100, description="Average equipment age (years)"
    )

    @field_validator("building_type")
    @classmethod
    def validate_building_type(cls, v: str) -> str:
        valid = {bt.value for bt in BuildingType}
        if v not in valid:
            raise ValueError(
                f"Unknown building type '{v}'. Must be one of: {sorted(valid)}"
            )
        return v

class EquipmentSurvey(BaseModel):
    """Equipment survey answers for applicability scoring.

    Attributes:
        has_led_lighting: Whether facility already has LED lighting.
        pct_led: Percentage of lighting that is already LED (0-100).
        has_vsd_hvac: Whether HVAC fans/pumps have VSDs.
        has_programmable_thermostats: Whether programmable thermostats exist.
        has_occupancy_sensors: Whether occupancy sensors are installed.
        has_bms: Whether a Building Management System is in place.
        has_power_factor_correction: Whether PFC is installed.
        has_compressed_air: Whether facility uses compressed air.
        compressed_air_age_years: Age of compressed air system (years).
        has_steam_system: Whether a steam / boiler system exists.
        hvac_type: HVAC system type description.
        hvac_age_years: HVAC system age (years).
        boiler_type: Boiler type description.
        boiler_age_years: Boiler age (years).
        refrigeration_type: Refrigeration system description.
        roof_insulation_r_value: Roof insulation R-value (m2K/W).
        wall_insulation_r_value: Wall insulation R-value (m2K/W).
        window_type: Window type description.
        window_u_value: Window U-value (W/m2K).
    """
    has_led_lighting: bool = Field(default=False, description="LED lighting present")
    pct_led: Decimal = Field(
        default=Decimal("0"), ge=0, le=100, description="Pct LED (0-100)"
    )
    has_vsd_hvac: bool = Field(default=False, description="VSD on HVAC")
    has_programmable_thermostats: bool = Field(
        default=False, description="Programmable thermostats"
    )
    has_occupancy_sensors: bool = Field(
        default=False, description="Occupancy sensors installed"
    )
    has_bms: bool = Field(default=False, description="BMS present")
    has_power_factor_correction: bool = Field(default=False, description="PFC present")
    has_compressed_air: bool = Field(
        default=False, description="Compressed air system"
    )
    compressed_air_age_years: int = Field(
        default=0, ge=0, le=100, description="Compressed air age (years)"
    )
    has_steam_system: bool = Field(default=False, description="Steam/boiler system")
    hvac_type: str = Field(default="split_system", max_length=100, description="HVAC type")
    hvac_age_years: int = Field(default=10, ge=0, le=100, description="HVAC age (years)")
    boiler_type: str = Field(default="", max_length=100, description="Boiler type")
    boiler_age_years: int = Field(default=0, ge=0, le=100, description="Boiler age (years)")
    refrigeration_type: str = Field(
        default="", max_length=100, description="Refrigeration type"
    )
    roof_insulation_r_value: Decimal = Field(
        default=Decimal("2.0"), ge=0, description="Roof R-value (m2K/W)"
    )
    wall_insulation_r_value: Decimal = Field(
        default=Decimal("1.5"), ge=0, description="Wall R-value (m2K/W)"
    )
    window_type: str = Field(
        default="single_glazed", max_length=100, description="Window type"
    )
    window_u_value: Decimal = Field(
        default=Decimal("5.0"), ge=0, description="Window U-value (W/m2K)"
    )

# ---------------------------------------------------------------------------
# Quick-Win Action Model
# ---------------------------------------------------------------------------

class QuickWinAction(BaseModel):
    """Pre-defined quick-win action from the library.

    Attributes:
        action_id: Unique action identifier.
        action_code: Short code (e.g. QW-LT-001).
        category: Action category.
        title: Action title.
        description: Detailed description.
        typical_savings_pct: Typical savings as percentage of category energy.
        typical_payback_months: Typical payback period in months.
        complexity: Implementation complexity tier.
        disruption: Operational disruption level.
        applicable_building_types: List of building types this applies to.
        is_behavioral: Whether this is a behavioural (no-cost) action.
        co_benefits: Non-energy co-benefits.
        prerequisites: Pre-conditions for applicability.
    """
    action_id: str = Field(default_factory=_new_uuid, description="Action ID")
    action_code: str = Field(default="", max_length=20, description="Action code")
    category: str = Field(
        default=ActionCategory.LIGHTING.value, description="Category"
    )
    title: str = Field(default="", max_length=200, description="Action title")
    description: str = Field(default="", max_length=2000, description="Description")
    typical_savings_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=100, description="Typical savings (%)"
    )
    typical_payback_months: int = Field(
        default=0, ge=0, le=120, description="Typical payback (months)"
    )
    complexity: str = Field(
        default=ActionComplexity.LOW_COST.value, description="Complexity"
    )
    disruption: str = Field(
        default=DisruptionLevel.MINIMAL.value, description="Disruption level"
    )
    applicable_building_types: List[str] = Field(
        default_factory=list, description="Applicable building types"
    )
    is_behavioral: bool = Field(default=False, description="Behavioural action flag")
    co_benefits: List[str] = Field(
        default_factory=list, description="Non-energy co-benefits"
    )
    prerequisites: List[str] = Field(
        default_factory=list, description="Prerequisites"
    )

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        valid = {c.value for c in ActionCategory}
        if v not in valid:
            raise ValueError(
                f"Unknown category '{v}'. Must be one of: {sorted(valid)}"
            )
        return v

    @field_validator("complexity")
    @classmethod
    def validate_complexity(cls, v: str) -> str:
        valid = {c.value for c in ActionComplexity}
        if v not in valid:
            raise ValueError(
                f"Unknown complexity '{v}'. Must be one of: {sorted(valid)}"
            )
        return v

    @field_validator("disruption")
    @classmethod
    def validate_disruption(cls, v: str) -> str:
        valid = {d.value for d in DisruptionLevel}
        if v not in valid:
            raise ValueError(
                f"Unknown disruption '{v}'. Must be one of: {sorted(valid)}"
            )
        return v

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class ScanResult(BaseModel):
    """Result for a single quick-win action evaluated against a facility.

    Attributes:
        result_id: Unique result identifier.
        action: The quick-win action evaluated.
        applicability_score: Score from 0-100 indicating fit.
        estimated_savings_kwh: Estimated annual energy savings (kWh).
        estimated_savings_cost: Estimated annual cost savings (EUR).
        estimated_co2e_reduction: Estimated annual CO2e reduction (kg).
        implementation_cost: Estimated implementation cost (EUR).
        payback_months: Estimated payback period (months).
        priority: Assigned priority level.
        confidence_pct: Confidence in the estimate (0-100).
        notes: Additional notes or caveats.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    action: QuickWinAction = Field(..., description="Quick-win action")
    applicability_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100, description="Applicability (0-100)"
    )
    estimated_savings_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Savings (kWh/yr)"
    )
    estimated_savings_cost: Decimal = Field(
        default=Decimal("0"), ge=0, description="Cost savings (EUR/yr)"
    )
    estimated_co2e_reduction: Decimal = Field(
        default=Decimal("0"), ge=0, description="CO2e reduction (kg/yr)"
    )
    implementation_cost: Decimal = Field(
        default=Decimal("0"), ge=0, description="Implementation cost (EUR)"
    )
    payback_months: Decimal = Field(
        default=Decimal("0"), ge=0, description="Payback (months)"
    )
    priority: str = Field(
        default=ActionPriority.MEDIUM.value, description="Priority"
    )
    confidence_pct: Decimal = Field(
        default=Decimal("50"), ge=0, le=100, description="Confidence (0-100)"
    )
    notes: str = Field(default="", max_length=2000, description="Notes")

class QuickWinsScanResult(BaseModel):
    """Complete scan result for a facility.

    Attributes:
        scan_id: Unique scan identifier.
        facility: Facility profile that was scanned.
        total_actions_scanned: Number of library actions evaluated.
        applicable_actions: Actions that passed applicability threshold.
        total_savings_kwh: Sum of estimated savings (kWh/yr).
        total_savings_cost: Sum of estimated cost savings (EUR/yr).
        total_co2e_reduction: Sum of estimated CO2e reduction (kg/yr).
        total_implementation_cost: Sum of implementation costs (EUR).
        scan_duration_ms: Scan processing time (milliseconds).
        calculated_at: Timestamp of calculation.
        provenance_hash: SHA-256 provenance hash.
    """
    scan_id: str = Field(default_factory=_new_uuid, description="Scan ID")
    facility: FacilityProfile = Field(..., description="Facility scanned")
    total_actions_scanned: int = Field(
        default=0, ge=0, description="Actions scanned"
    )
    applicable_actions: List[ScanResult] = Field(
        default_factory=list, description="Applicable actions"
    )
    total_savings_kwh: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total savings (kWh/yr)"
    )
    total_savings_cost: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total cost savings (EUR/yr)"
    )
    total_co2e_reduction: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total CO2e reduction (kg/yr)"
    )
    total_implementation_cost: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total implementation cost (EUR)"
    )
    scan_duration_ms: float = Field(default=0.0, description="Duration (ms)")
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Calculation timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Quick Wins Library -- 84 pre-defined actions
# ---------------------------------------------------------------------------

_ALL_BUILDING_TYPES: List[str] = [bt.value for bt in BuildingType]

_COMMERCIAL: List[str] = [
    BuildingType.OFFICE.value, BuildingType.RETAIL.value,
    BuildingType.EDUCATION.value, BuildingType.HEALTHCARE.value,
    BuildingType.HOSPITALITY.value, BuildingType.WORSHIP.value,
    BuildingType.FITNESS.value, BuildingType.MIXED_USE.value,
    BuildingType.MULTIFAMILY.value, BuildingType.LABORATORY.value,
]

_INDUSTRIAL: List[str] = [
    BuildingType.MANUFACTURING.value, BuildingType.WAREHOUSE.value,
    BuildingType.DATA_CENTER.value,
]

_FOOD_SERVICE: List[str] = [
    BuildingType.RESTAURANT.value, BuildingType.GROCERY.value,
    BuildingType.HOSPITALITY.value, BuildingType.HEALTHCARE.value,
]

_LAUNDRY_TYPES: List[str] = [
    BuildingType.HOSPITALITY.value, BuildingType.HEALTHCARE.value,
    BuildingType.FITNESS.value, BuildingType.MULTIFAMILY.value,
]

QUICK_WINS_LIBRARY: List[QuickWinAction] = [
    # ---------------------------------------------------------------
    # LIGHTING (8 actions)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-LT-001", category=ActionCategory.LIGHTING.value,
        title="LED retrofit - linear fluorescent to LED",
        description="Replace T8/T12 fluorescent tubes with LED tubes or LED troffer fixtures. Typical 40-60% lighting energy reduction with improved colour rendering.",
        typical_savings_pct=Decimal("50"), typical_payback_months=18,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Improved light quality", "Reduced maintenance", "Lower heat gain"],
        prerequisites=["Non-LED fluorescent fixtures in use"],
    ),
    QuickWinAction(
        action_code="QW-LT-002", category=ActionCategory.LIGHTING.value,
        title="Delamping over-lit areas",
        description="Remove excess lamps in over-lit spaces while maintaining adequate illumination per EN 12464-1 requirements.",
        typical_savings_pct=Decimal("15"), typical_payback_months=0,
        complexity=ActionComplexity.NO_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        is_behavioral=True,
        co_benefits=["Reduced glare", "Lower heat gain"],
        prerequisites=["Illumination levels exceed EN 12464-1 requirements"],
    ),
    QuickWinAction(
        action_code="QW-LT-003", category=ActionCategory.LIGHTING.value,
        title="Occupancy sensor installation",
        description="Install occupancy/vacancy sensors in intermittently occupied spaces (toilets, corridors, meeting rooms, storage).",
        typical_savings_pct=Decimal("25"), typical_payback_months=12,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Convenience", "Extended lamp life"],
        prerequisites=["No existing occupancy sensors"],
    ),
    QuickWinAction(
        action_code="QW-LT-004", category=ActionCategory.LIGHTING.value,
        title="Daylight harvesting controls",
        description="Install photocell dimming controls near windows to reduce artificial lighting when daylight is sufficient.",
        typical_savings_pct=Decimal("20"), typical_payback_months=24,
        complexity=ActionComplexity.MEDIUM_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=_COMMERCIAL,
        co_benefits=["Improved occupant comfort", "Circadian health benefits"],
        prerequisites=["Perimeter zones with glazing", "Dimmable lighting"],
    ),
    QuickWinAction(
        action_code="QW-LT-005", category=ActionCategory.LIGHTING.value,
        title="Exterior lighting timers and photocells",
        description="Install astronomical timers or photocells on exterior, signage, and car park lighting to prevent daytime operation.",
        typical_savings_pct=Decimal("30"), typical_payback_months=6,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Reduced light pollution", "Extended lamp life"],
        prerequisites=["Exterior lighting without automatic controls"],
    ),
    QuickWinAction(
        action_code="QW-LT-006", category=ActionCategory.LIGHTING.value,
        title="Task lighting deployment",
        description="Provide desk-level task lights and reduce ambient overhead lighting levels in office environments.",
        typical_savings_pct=Decimal("15"), typical_payback_months=8,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=[
            BuildingType.OFFICE.value, BuildingType.EDUCATION.value,
            BuildingType.LABORATORY.value, BuildingType.MIXED_USE.value,
        ],
        co_benefits=["Personalised illumination", "Improved productivity"],
        prerequisites=[],
    ),
    QuickWinAction(
        action_code="QW-LT-007", category=ActionCategory.LIGHTING.value,
        title="Parking garage LED retrofit",
        description="Replace HID or fluorescent fixtures in parking garages with LED and bi-level controls.",
        typical_savings_pct=Decimal("60"), typical_payback_months=24,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=[
            BuildingType.OFFICE.value, BuildingType.RETAIL.value,
            BuildingType.HEALTHCARE.value, BuildingType.MIXED_USE.value,
            BuildingType.HOSPITALITY.value,
        ],
        co_benefits=["Improved safety", "Reduced maintenance"],
        prerequisites=["Multi-level parking structure with non-LED fixtures"],
    ),
    QuickWinAction(
        action_code="QW-LT-008", category=ActionCategory.LIGHTING.value,
        title="LED exit sign retrofit",
        description="Replace incandescent or compact fluorescent exit signs with LED versions (2-5W vs 15-40W each).",
        typical_savings_pct=Decimal("80"), typical_payback_months=6,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["10-year lamp life", "Reduced maintenance"],
        prerequisites=["Non-LED exit signs in use"],
    ),

    # ---------------------------------------------------------------
    # HVAC (9 actions)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-HV-001", category=ActionCategory.HVAC.value,
        title="Thermostat setback / setup schedule",
        description="Implement night and weekend temperature setbacks (heating) and setups (cooling) of 3-5 degC during unoccupied periods.",
        typical_savings_pct=Decimal("10"), typical_payback_months=0,
        complexity=ActionComplexity.NO_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        is_behavioral=True,
        co_benefits=["Extended equipment life"],
        prerequisites=["Programmable or manual thermostats"],
    ),
    QuickWinAction(
        action_code="QW-HV-002", category=ActionCategory.HVAC.value,
        title="Air filter replacement programme",
        description="Establish regular filter replacement schedule. Dirty filters increase fan energy by 5-15% and reduce indoor air quality.",
        typical_savings_pct=Decimal("5"), typical_payback_months=1,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Improved IAQ", "Extended equipment life"],
        prerequisites=[],
    ),
    QuickWinAction(
        action_code="QW-HV-003", category=ActionCategory.HVAC.value,
        title="Economiser repair and commissioning",
        description="Inspect and repair outside air economisers. Stuck dampers or failed sensors waste 10-20% of HVAC energy.",
        typical_savings_pct=Decimal("12"), typical_payback_months=6,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Improved ventilation", "Better humidity control"],
        prerequisites=["Economiser-equipped AHUs"],
    ),
    QuickWinAction(
        action_code="QW-HV-004", category=ActionCategory.HVAC.value,
        title="VSD retrofit on AHU supply fans",
        description="Install variable speed drives on constant-volume AHU supply fans. Savings follow fan affinity laws (cube law).",
        typical_savings_pct=Decimal("25"), typical_payback_months=30,
        complexity=ActionComplexity.MEDIUM_COST.value,
        disruption=DisruptionLevel.MODERATE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Reduced noise", "Better comfort control"],
        prerequisites=["Constant-volume AHUs without VSDs"],
    ),
    QuickWinAction(
        action_code="QW-HV-005", category=ActionCategory.HVAC.value,
        title="Duct sealing and insulation",
        description="Seal duct joints and insulate uninsulated ductwork in unconditioned spaces. Typical duct leakage is 15-25% of airflow.",
        typical_savings_pct=Decimal("8"), typical_payback_months=18,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Improved comfort", "Better air distribution"],
        prerequisites=["Accessible ductwork in unconditioned spaces"],
    ),
    QuickWinAction(
        action_code="QW-HV-006", category=ActionCategory.HVAC.value,
        title="Condenser coil cleaning",
        description="Clean air-cooled condenser coils on rooftop units and split systems. Dirty coils increase energy use by 5-15%.",
        typical_savings_pct=Decimal("8"), typical_payback_months=1,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Extended compressor life", "Better cooling capacity"],
        prerequisites=["Air-cooled condensers"],
    ),
    QuickWinAction(
        action_code="QW-HV-007", category=ActionCategory.HVAC.value,
        title="Chiller optimisation and sequencing",
        description="Optimise chiller staging, condenser water setpoints, and chilled water reset schedules.",
        typical_savings_pct=Decimal("15"), typical_payback_months=12,
        complexity=ActionComplexity.MEDIUM_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=[
            BuildingType.OFFICE.value, BuildingType.HEALTHCARE.value,
            BuildingType.DATA_CENTER.value, BuildingType.LABORATORY.value,
            BuildingType.MIXED_USE.value,
        ],
        co_benefits=["Extended chiller life", "Reduced peak demand"],
        prerequisites=["Multiple chiller plant"],
    ),
    QuickWinAction(
        action_code="QW-HV-008", category=ActionCategory.HVAC.value,
        title="Cooling tower optimisation",
        description="Optimise cooling tower fan staging, water treatment, and approach temperature setpoints.",
        typical_savings_pct=Decimal("10"), typical_payback_months=8,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=[
            BuildingType.OFFICE.value, BuildingType.HEALTHCARE.value,
            BuildingType.DATA_CENTER.value, BuildingType.MANUFACTURING.value,
            BuildingType.MIXED_USE.value,
        ],
        co_benefits=["Reduced water consumption", "Legionella risk management"],
        prerequisites=["Water-cooled chiller plant with cooling towers"],
    ),
    QuickWinAction(
        action_code="QW-HV-009", category=ActionCategory.HVAC.value,
        title="Destratification fans in high-bay spaces",
        description="Install ceiling destratification fans to push warm air from ceiling level back to the occupied zone in heated warehouses.",
        typical_savings_pct=Decimal("20"), typical_payback_months=18,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=[
            BuildingType.WAREHOUSE.value, BuildingType.MANUFACTURING.value,
            BuildingType.WORSHIP.value, BuildingType.FITNESS.value,
        ],
        co_benefits=["Improved comfort at floor level"],
        prerequisites=["Ceiling height > 4m with heating"],
    ),

    # ---------------------------------------------------------------
    # ENVELOPE (6 actions)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-EN-001", category=ActionCategory.ENVELOPE.value,
        title="Weather stripping on doors and windows",
        description="Replace worn or missing weather stripping on external doors and operable windows to reduce air infiltration.",
        typical_savings_pct=Decimal("5"), typical_payback_months=3,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Improved comfort", "Reduced draughts", "Pest prevention"],
        prerequisites=[],
    ),
    QuickWinAction(
        action_code="QW-EN-002", category=ActionCategory.ENVELOPE.value,
        title="Caulking and sealing penetrations",
        description="Seal gaps around pipes, cables, and penetrations through the building envelope with appropriate sealant.",
        typical_savings_pct=Decimal("3"), typical_payback_months=2,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Reduced moisture ingress", "Pest prevention"],
        prerequisites=[],
    ),
    QuickWinAction(
        action_code="QW-EN-003", category=ActionCategory.ENVELOPE.value,
        title="Window film installation",
        description="Apply solar control window film to reduce solar heat gain in cooling-dominated climates. Typical SHGC reduction of 30-50%.",
        typical_savings_pct=Decimal("8"), typical_payback_months=24,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=_COMMERCIAL,
        co_benefits=["Reduced glare", "UV protection for furnishings"],
        prerequisites=["Significant solar-exposed glazing"],
    ),
    QuickWinAction(
        action_code="QW-EN-004", category=ActionCategory.ENVELOPE.value,
        title="Cool roof coating application",
        description="Apply reflective roof coating to reduce solar absorption. Typical roof surface temperature reduction of 20-30 degC.",
        typical_savings_pct=Decimal("10"), typical_payback_months=36,
        complexity=ActionComplexity.MEDIUM_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Extended roof life", "Reduced urban heat island"],
        prerequisites=["Dark-coloured flat or low-slope roof"],
    ),
    QuickWinAction(
        action_code="QW-EN-005", category=ActionCategory.ENVELOPE.value,
        title="Loading dock door seals and curtains",
        description="Install dock seals, dock shelters, or strip curtains on loading dock openings to minimise conditioned air loss.",
        typical_savings_pct=Decimal("15"), typical_payback_months=12,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=[
            BuildingType.WAREHOUSE.value, BuildingType.MANUFACTURING.value,
            BuildingType.RETAIL.value, BuildingType.GROCERY.value,
        ],
        co_benefits=["Improved comfort", "Pest prevention", "Noise reduction"],
        prerequisites=["Loading dock openings without seals"],
    ),
    QuickWinAction(
        action_code="QW-EN-006", category=ActionCategory.ENVELOPE.value,
        title="Pipe and duct insulation in unconditioned spaces",
        description="Insulate exposed hot water pipes, steam pipes, and HVAC ducts running through unconditioned spaces.",
        typical_savings_pct=Decimal("5"), typical_payback_months=8,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Reduced condensation risk", "Safety (burn prevention)"],
        prerequisites=["Uninsulated pipes or ducts in unconditioned spaces"],
    ),

    # ---------------------------------------------------------------
    # PLUG LOADS (4 actions)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-PL-001", category=ActionCategory.PLUG_LOADS.value,
        title="Smart power strips at workstations",
        description="Install advanced power strips with occupancy or load-sensing at workstations to eliminate phantom loads from monitors, chargers, and peripherals.",
        typical_savings_pct=Decimal("15"), typical_payback_months=12,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=[
            BuildingType.OFFICE.value, BuildingType.EDUCATION.value,
            BuildingType.LABORATORY.value, BuildingType.MIXED_USE.value,
        ],
        co_benefits=["Surge protection", "Equipment longevity"],
        prerequisites=[],
    ),
    QuickWinAction(
        action_code="QW-PL-002", category=ActionCategory.PLUG_LOADS.value,
        title="Vending machine controllers",
        description="Install occupancy-based controllers on refrigerated vending machines to power down compressors and lighting during unoccupied hours.",
        typical_savings_pct=Decimal("40"), typical_payback_months=10,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Extended vending machine life"],
        prerequisites=["Refrigerated vending machines present"],
    ),
    QuickWinAction(
        action_code="QW-PL-003", category=ActionCategory.PLUG_LOADS.value,
        title="Computer power management (GPO)",
        description="Deploy group-policy or endpoint-management sleep/hibernate settings across all workstations and laptops.",
        typical_savings_pct=Decimal("20"), typical_payback_months=0,
        complexity=ActionComplexity.NO_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=[
            BuildingType.OFFICE.value, BuildingType.EDUCATION.value,
            BuildingType.HEALTHCARE.value, BuildingType.LABORATORY.value,
            BuildingType.MIXED_USE.value,
        ],
        is_behavioral=True,
        co_benefits=["Extended hardware life", "Improved cyber security"],
        prerequisites=["Networked PCs without power management"],
    ),
    QuickWinAction(
        action_code="QW-PL-004", category=ActionCategory.PLUG_LOADS.value,
        title="Printer consolidation and scheduling",
        description="Consolidate personal printers to shared MFDs and enable auto-sleep. Typical office has 1 printer per 4 employees vs best-practice 1 per 15.",
        typical_savings_pct=Decimal("10"), typical_payback_months=3,
        complexity=ActionComplexity.NO_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=[
            BuildingType.OFFICE.value, BuildingType.EDUCATION.value,
            BuildingType.HEALTHCARE.value, BuildingType.MIXED_USE.value,
        ],
        is_behavioral=True,
        co_benefits=["Reduced paper use", "Lower toner costs"],
        prerequisites=["Multiple personal printers in use"],
    ),

    # ---------------------------------------------------------------
    # WATER HEATING (5 actions)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-WH-001", category=ActionCategory.WATER_HEATING.value,
        title="Hot water tank insulation jacket",
        description="Install insulation jacket on uninsulated or poorly insulated hot water storage tanks. Reduces standby losses by 25-45%.",
        typical_savings_pct=Decimal("10"), typical_payback_months=3,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Faster recovery time"],
        prerequisites=["Uninsulated or poorly insulated hot water tank"],
    ),
    QuickWinAction(
        action_code="QW-WH-002", category=ActionCategory.WATER_HEATING.value,
        title="Hot water pipe insulation",
        description="Insulate all accessible hot water distribution pipes, particularly in unconditioned spaces.",
        typical_savings_pct=Decimal("8"), typical_payback_months=6,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Reduced heat loss to corridors", "Safety"],
        prerequisites=["Uninsulated hot water pipes"],
    ),
    QuickWinAction(
        action_code="QW-WH-003", category=ActionCategory.WATER_HEATING.value,
        title="Low-flow fixtures and aerators",
        description="Install low-flow showerheads (< 9 L/min) and tap aerators (< 6 L/min) to reduce hot water demand.",
        typical_savings_pct=Decimal("15"), typical_payback_months=4,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Water conservation", "Reduced sewer costs"],
        prerequisites=[],
    ),
    QuickWinAction(
        action_code="QW-WH-004", category=ActionCategory.WATER_HEATING.value,
        title="Timer on hot water recirculation pump",
        description="Install a timer on the recirculation pump to stop circulation during unoccupied hours.",
        typical_savings_pct=Decimal("20"), typical_payback_months=2,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Pump life extension"],
        prerequisites=["Continuous recirculation pump without timer"],
    ),
    QuickWinAction(
        action_code="QW-WH-005", category=ActionCategory.WATER_HEATING.value,
        title="Hot water temperature setback",
        description="Reduce hot water storage temperature from 70 degC to 60 degC (minimum for Legionella control) to cut standby losses.",
        typical_savings_pct=Decimal("8"), typical_payback_months=0,
        complexity=ActionComplexity.NO_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        is_behavioral=True,
        co_benefits=["Reduced scaling", "Safety"],
        prerequisites=["Hot water stored above 65 degC"],
    ),

    # ---------------------------------------------------------------
    # COMPRESSED AIR (5 actions)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-CA-001", category=ActionCategory.COMPRESSED_AIR.value,
        title="Compressed air leak detection and repair",
        description="Conduct ultrasonic leak survey and repair leaks. Typical plant loses 20-30% of compressed air through leaks.",
        typical_savings_pct=Decimal("20"), typical_payback_months=3,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=[
            BuildingType.MANUFACTURING.value, BuildingType.WAREHOUSE.value,
        ],
        co_benefits=["Improved system pressure stability"],
        prerequisites=["Compressed air system in use"],
    ),
    QuickWinAction(
        action_code="QW-CA-002", category=ActionCategory.COMPRESSED_AIR.value,
        title="System pressure reduction",
        description="Reduce system pressure by 1 bar (saves approx 7% per bar). Many systems operate 1-2 bar above actual requirements.",
        typical_savings_pct=Decimal("14"), typical_payback_months=0,
        complexity=ActionComplexity.NO_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=[
            BuildingType.MANUFACTURING.value, BuildingType.WAREHOUSE.value,
        ],
        is_behavioral=True,
        co_benefits=["Reduced leak rate", "Less wear on equipment"],
        prerequisites=["Compressed air system operated above minimum required pressure"],
    ),
    QuickWinAction(
        action_code="QW-CA-003", category=ActionCategory.COMPRESSED_AIR.value,
        title="Air receiver tank optimisation",
        description="Optimise or add air receiver capacity to reduce compressor cycling and improve pressure stability.",
        typical_savings_pct=Decimal("5"), typical_payback_months=12,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=[
            BuildingType.MANUFACTURING.value, BuildingType.WAREHOUSE.value,
        ],
        co_benefits=["Better pressure stability", "Reduced compressor wear"],
        prerequisites=["Undersized receiver capacity"],
    ),
    QuickWinAction(
        action_code="QW-CA-004", category=ActionCategory.COMPRESSED_AIR.value,
        title="Compressor intake air cooling",
        description="Route compressor intake to the coolest available location. Every 3 degC intake temperature reduction saves 1% compressor energy.",
        typical_savings_pct=Decimal("4"), typical_payback_months=6,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=[
            BuildingType.MANUFACTURING.value, BuildingType.WAREHOUSE.value,
        ],
        co_benefits=["Better air quality to compressor"],
        prerequisites=["Compressor intake in hot location"],
    ),
    QuickWinAction(
        action_code="QW-CA-005", category=ActionCategory.COMPRESSED_AIR.value,
        title="VSD compressor retrofit",
        description="Replace fixed-speed compressor with VSD (variable speed drive) compressor for variable-demand applications.",
        typical_savings_pct=Decimal("25"), typical_payback_months=36,
        complexity=ActionComplexity.CAPITAL.value,
        disruption=DisruptionLevel.MODERATE.value,
        applicable_building_types=[
            BuildingType.MANUFACTURING.value, BuildingType.WAREHOUSE.value,
        ],
        co_benefits=["Better pressure control", "Reduced noise at part load"],
        prerequisites=["Fixed-speed compressor with variable demand"],
    ),

    # ---------------------------------------------------------------
    # MOTORS & DRIVES (4 actions)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-MD-001", category=ActionCategory.MOTORS_DRIVES.value,
        title="VSD retrofit on pumps",
        description="Install variable speed drives on constant-speed circulation pumps with throttling valves. Affinity laws yield cubic savings.",
        typical_savings_pct=Decimal("25"), typical_payback_months=24,
        complexity=ActionComplexity.MEDIUM_COST.value,
        disruption=DisruptionLevel.MODERATE.value,
        applicable_building_types=[
            BuildingType.MANUFACTURING.value, BuildingType.OFFICE.value,
            BuildingType.HEALTHCARE.value, BuildingType.DATA_CENTER.value,
            BuildingType.MIXED_USE.value,
        ],
        co_benefits=["Reduced water hammer", "Better flow control"],
        prerequisites=["Constant-speed pumps with throttling valves"],
    ),
    QuickWinAction(
        action_code="QW-MD-002", category=ActionCategory.MOTORS_DRIVES.value,
        title="Motor right-sizing",
        description="Replace oversized motors (operating below 40% load) with correctly sized IE3/IE4 motors.",
        typical_savings_pct=Decimal("10"), typical_payback_months=30,
        complexity=ActionComplexity.MEDIUM_COST.value,
        disruption=DisruptionLevel.MODERATE.value,
        applicable_building_types=[
            BuildingType.MANUFACTURING.value, BuildingType.WAREHOUSE.value,
        ],
        co_benefits=["Improved power factor", "Extended bearing life"],
        prerequisites=["Motors operating below 40% rated load"],
    ),
    QuickWinAction(
        action_code="QW-MD-003", category=ActionCategory.MOTORS_DRIVES.value,
        title="Belt-to-direct-drive conversion",
        description="Replace V-belt drives with direct-drive or synchronous belt couplings to eliminate belt slip losses (3-5%).",
        typical_savings_pct=Decimal("5"), typical_payback_months=8,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=[
            BuildingType.MANUFACTURING.value, BuildingType.WAREHOUSE.value,
        ],
        co_benefits=["Reduced maintenance", "Less noise"],
        prerequisites=["V-belt driven fans or pumps"],
    ),
    QuickWinAction(
        action_code="QW-MD-004", category=ActionCategory.MOTORS_DRIVES.value,
        title="Premium efficiency motor replacement",
        description="Replace failed or rewound motors with IE4 premium efficiency motors at next opportunity.",
        typical_savings_pct=Decimal("5"), typical_payback_months=18,
        complexity=ActionComplexity.MEDIUM_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=[
            BuildingType.MANUFACTURING.value, BuildingType.WAREHOUSE.value,
            BuildingType.DATA_CENTER.value,
        ],
        co_benefits=["Extended motor life", "Lower operating temperature"],
        prerequisites=["Motors below IE3 efficiency class"],
    ),

    # ---------------------------------------------------------------
    # REFRIGERATION (5 actions)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-RF-001", category=ActionCategory.REFRIGERATION.value,
        title="Strip curtains on cold room doors",
        description="Install PVC strip curtains on cold room and freezer door openings to reduce warm air infiltration during access.",
        typical_savings_pct=Decimal("10"), typical_payback_months=3,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=[
            BuildingType.GROCERY.value, BuildingType.RESTAURANT.value,
            BuildingType.MANUFACTURING.value, BuildingType.HOSPITALITY.value,
            BuildingType.WAREHOUSE.value,
        ],
        co_benefits=["Temperature stability", "Food safety"],
        prerequisites=["Cold rooms without strip curtains"],
    ),
    QuickWinAction(
        action_code="QW-RF-002", category=ActionCategory.REFRIGERATION.value,
        title="Door gasket replacement on reach-ins",
        description="Replace worn or damaged door gaskets on display cases and reach-in coolers. Failed gaskets increase infiltration by 10-20%.",
        typical_savings_pct=Decimal("5"), typical_payback_months=2,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=[
            BuildingType.GROCERY.value, BuildingType.RESTAURANT.value,
            BuildingType.HOSPITALITY.value,
        ],
        co_benefits=["Improved food safety", "Better temperature control"],
        prerequisites=["Worn or damaged gaskets on cooler/freezer doors"],
    ),
    QuickWinAction(
        action_code="QW-RF-003", category=ActionCategory.REFRIGERATION.value,
        title="Floating head pressure control",
        description="Allow condenser pressure to float down with ambient temperature rather than fixed high setpoint.",
        typical_savings_pct=Decimal("10"), typical_payback_months=6,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=[
            BuildingType.GROCERY.value, BuildingType.MANUFACTURING.value,
            BuildingType.WAREHOUSE.value,
        ],
        co_benefits=["Reduced compressor wear"],
        prerequisites=["Fixed head pressure refrigeration system"],
    ),
    QuickWinAction(
        action_code="QW-RF-004", category=ActionCategory.REFRIGERATION.value,
        title="Anti-sweat heater controls",
        description="Install humidity-based controls on anti-sweat heaters in glass-door display cases to reduce heater operation by 50-75%.",
        typical_savings_pct=Decimal("8"), typical_payback_months=12,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=[
            BuildingType.GROCERY.value, BuildingType.RESTAURANT.value,
        ],
        co_benefits=["Reduced condensation", "Better product visibility"],
        prerequisites=["Glass-door display cases with always-on anti-sweat heaters"],
    ),
    QuickWinAction(
        action_code="QW-RF-005", category=ActionCategory.REFRIGERATION.value,
        title="EC motors on evaporator fans",
        description="Replace shaded-pole evaporator fan motors with electronically commutated (EC) motors (60-70% motor energy reduction).",
        typical_savings_pct=Decimal("12"), typical_payback_months=18,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=[
            BuildingType.GROCERY.value, BuildingType.RESTAURANT.value,
            BuildingType.MANUFACTURING.value, BuildingType.WAREHOUSE.value,
        ],
        co_benefits=["Reduced heat load in case", "Quieter operation"],
        prerequisites=["Shaded-pole evaporator fan motors"],
    ),

    # ---------------------------------------------------------------
    # KITCHEN (3 actions)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-KT-001", category=ActionCategory.KITCHEN.value,
        title="Demand-controlled kitchen exhaust hoods",
        description="Install variable-speed or demand-controlled ventilation on kitchen exhaust hoods using temperature/smoke sensors.",
        typical_savings_pct=Decimal("30"), typical_payback_months=24,
        complexity=ActionComplexity.MEDIUM_COST.value,
        disruption=DisruptionLevel.MODERATE.value,
        applicable_building_types=_FOOD_SERVICE,
        co_benefits=["Reduced HVAC make-up air costs", "Improved kitchen comfort"],
        prerequisites=["Constant-speed kitchen exhaust hoods"],
    ),
    QuickWinAction(
        action_code="QW-KT-002", category=ActionCategory.KITCHEN.value,
        title="Pre-rinse spray valve replacement",
        description="Replace high-flow pre-rinse spray valves (> 6 L/min) with low-flow models (< 4.5 L/min).",
        typical_savings_pct=Decimal("10"), typical_payback_months=1,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_FOOD_SERVICE,
        co_benefits=["Water savings", "Reduced sewer costs", "Lower hot water demand"],
        prerequisites=["High-flow pre-rinse spray valves"],
    ),
    QuickWinAction(
        action_code="QW-KT-003", category=ActionCategory.KITCHEN.value,
        title="ENERGY STAR commercial kitchen equipment",
        description="Replace ageing fryers, steamers, and ovens with ENERGY STAR rated models at end of life.",
        typical_savings_pct=Decimal("20"), typical_payback_months=36,
        complexity=ActionComplexity.CAPITAL.value,
        disruption=DisruptionLevel.MODERATE.value,
        applicable_building_types=_FOOD_SERVICE,
        co_benefits=["Better food quality", "Reduced maintenance"],
        prerequisites=["Non-ENERGY-STAR kitchen equipment at end of life"],
    ),

    # ---------------------------------------------------------------
    # LAUNDRY (2 actions)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-LY-001", category=ActionCategory.LAUNDRY.value,
        title="Ozone laundry system",
        description="Install ozone injection system for commercial laundry to wash at lower temperatures (cold water) while maintaining sanitation.",
        typical_savings_pct=Decimal("30"), typical_payback_months=24,
        complexity=ActionComplexity.MEDIUM_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=_LAUNDRY_TYPES,
        co_benefits=["Water savings", "Reduced chemical use", "Linen longevity"],
        prerequisites=["Commercial laundry with hot water wash cycles"],
    ),
    QuickWinAction(
        action_code="QW-LY-002", category=ActionCategory.LAUNDRY.value,
        title="Drain water heat recovery on laundry",
        description="Install heat exchanger on laundry drain to preheat incoming cold water using waste heat from drain water.",
        typical_savings_pct=Decimal("20"), typical_payback_months=18,
        complexity=ActionComplexity.MEDIUM_COST.value,
        disruption=DisruptionLevel.MODERATE.value,
        applicable_building_types=_LAUNDRY_TYPES,
        co_benefits=["Reduced water heating demand", "Lower sewer temperature"],
        prerequisites=["High-volume laundry with accessible drain"],
    ),

    # ---------------------------------------------------------------
    # CONTROLS (4 actions)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-CT-001", category=ActionCategory.CONTROLS.value,
        title="BMS re-commissioning and optimisation",
        description="Re-commission existing BMS: fix overridden setpoints, repair failed sensors, restore original sequences of operation.",
        typical_savings_pct=Decimal("15"), typical_payback_months=6,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Improved comfort", "Extended equipment life"],
        prerequisites=["Existing BMS with deferred maintenance"],
    ),
    QuickWinAction(
        action_code="QW-CT-002", category=ActionCategory.CONTROLS.value,
        title="HVAC scheduling optimisation",
        description="Review and optimise HVAC start/stop schedules to match actual occupancy rather than worst-case assumptions.",
        typical_savings_pct=Decimal("10"), typical_payback_months=0,
        complexity=ActionComplexity.NO_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        is_behavioral=True,
        co_benefits=["Reduced equipment runtime"],
        prerequisites=["BMS or time-clock controlled HVAC"],
    ),
    QuickWinAction(
        action_code="QW-CT-003", category=ActionCategory.CONTROLS.value,
        title="Demand-controlled ventilation (CO2 sensors)",
        description="Install CO2 sensors to modulate outside air ventilation based on actual occupancy rather than fixed minimum rates.",
        typical_savings_pct=Decimal("20"), typical_payback_months=18,
        complexity=ActionComplexity.MEDIUM_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=[
            BuildingType.OFFICE.value, BuildingType.EDUCATION.value,
            BuildingType.RETAIL.value, BuildingType.WORSHIP.value,
            BuildingType.FITNESS.value, BuildingType.MIXED_USE.value,
        ],
        co_benefits=["Improved IAQ monitoring", "Better comfort"],
        prerequisites=["Variable occupancy spaces without DCV"],
    ),
    QuickWinAction(
        action_code="QW-CT-004", category=ActionCategory.CONTROLS.value,
        title="Chilled/hot water temperature reset",
        description="Implement outside-air-temperature reset on chilled water and hot water supply setpoints to reduce energy at part load.",
        typical_savings_pct=Decimal("8"), typical_payback_months=3,
        complexity=ActionComplexity.NO_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=[
            BuildingType.OFFICE.value, BuildingType.HEALTHCARE.value,
            BuildingType.DATA_CENTER.value, BuildingType.LABORATORY.value,
            BuildingType.MIXED_USE.value,
        ],
        is_behavioral=True,
        co_benefits=["Improved part-load efficiency"],
        prerequisites=["Hydronic HVAC system with BMS"],
    ),

    # ---------------------------------------------------------------
    # BEHAVIORAL (4 actions)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-BH-001", category=ActionCategory.BEHAVIORAL.value,
        title="Power-down policy at end of day",
        description="Implement and enforce a last-person-out policy to power down non-essential equipment, lights, and HVAC at end of business.",
        typical_savings_pct=Decimal("5"), typical_payback_months=0,
        complexity=ActionComplexity.NO_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        is_behavioral=True,
        co_benefits=["Fire safety", "Security"],
        prerequisites=[],
    ),
    QuickWinAction(
        action_code="QW-BH-002", category=ActionCategory.BEHAVIORAL.value,
        title="Thermostat awareness campaign",
        description="Educate occupants on energy impact of thermostat adjustments. Each 1 degC setpoint change saves approx 3% of HVAC energy.",
        typical_savings_pct=Decimal("6"), typical_payback_months=0,
        complexity=ActionComplexity.NO_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        is_behavioral=True,
        co_benefits=["Occupant engagement", "Culture change"],
        prerequisites=[],
    ),
    QuickWinAction(
        action_code="QW-BH-003", category=ActionCategory.BEHAVIORAL.value,
        title="Turn-off-the-lights campaign",
        description="Run awareness campaign with signage and feedback to encourage manual switch-off of lights in unoccupied areas.",
        typical_savings_pct=Decimal("5"), typical_payback_months=0,
        complexity=ActionComplexity.NO_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        is_behavioral=True,
        co_benefits=["Occupant engagement", "Extended lamp life"],
        prerequisites=[],
    ),
    QuickWinAction(
        action_code="QW-BH-004", category=ActionCategory.BEHAVIORAL.value,
        title="Print reduction and paperless initiative",
        description="Set default double-sided printing, encourage digital workflows, and deploy print-release stations to reduce print volumes by 30-50%.",
        typical_savings_pct=Decimal("3"), typical_payback_months=0,
        complexity=ActionComplexity.NO_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=[
            BuildingType.OFFICE.value, BuildingType.EDUCATION.value,
            BuildingType.HEALTHCARE.value, BuildingType.MIXED_USE.value,
        ],
        is_behavioral=True,
        co_benefits=["Paper cost savings", "Environmental benefit"],
        prerequisites=[],
    ),

    # ---------------------------------------------------------------
    # MAINTENANCE (5 actions)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-MN-001", category=ActionCategory.MAINTENANCE.value,
        title="Steam trap survey and repair",
        description="Conduct annual steam trap survey using ultrasonic or temperature methods. Replace failed-open traps (typical failure rate 15-30%).",
        typical_savings_pct=Decimal("10"), typical_payback_months=4,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=[
            BuildingType.MANUFACTURING.value, BuildingType.HEALTHCARE.value,
            BuildingType.HOSPITALITY.value,
        ],
        co_benefits=["Improved steam quality", "Reduced water hammer"],
        prerequisites=["Steam system with steam traps"],
    ),
    QuickWinAction(
        action_code="QW-MN-002", category=ActionCategory.MAINTENANCE.value,
        title="Belt tensioning and alignment",
        description="Check and correct belt tension and sheave alignment on belt-driven fans and pumps. Misalignment wastes 2-5% of motor energy.",
        typical_savings_pct=Decimal("3"), typical_payback_months=1,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=[
            BuildingType.MANUFACTURING.value, BuildingType.WAREHOUSE.value,
            BuildingType.OFFICE.value, BuildingType.MIXED_USE.value,
        ],
        co_benefits=["Extended belt life", "Reduced vibration"],
        prerequisites=["Belt-driven rotating equipment"],
    ),
    QuickWinAction(
        action_code="QW-MN-003", category=ActionCategory.MAINTENANCE.value,
        title="Bearing lubrication programme",
        description="Implement scheduled bearing lubrication for motors, fans, and pumps to reduce friction losses.",
        typical_savings_pct=Decimal("2"), typical_payback_months=1,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=[
            BuildingType.MANUFACTURING.value, BuildingType.WAREHOUSE.value,
        ],
        co_benefits=["Extended bearing life", "Reduced downtime risk"],
        prerequisites=["Rotating equipment without scheduled lubrication"],
    ),
    QuickWinAction(
        action_code="QW-MN-004", category=ActionCategory.MAINTENANCE.value,
        title="Heat exchanger cleaning",
        description="Clean fouled heat exchangers (condensers, evaporators, economisers) to restore design heat transfer rates.",
        typical_savings_pct=Decimal("5"), typical_payback_months=2,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Better temperature control", "Extended equipment life"],
        prerequisites=["Heat exchangers with fouling indicators"],
    ),
    QuickWinAction(
        action_code="QW-MN-005", category=ActionCategory.MAINTENANCE.value,
        title="Boiler combustion tuning",
        description="Tune boiler combustion to optimise excess air ratio. Each 1% reduction in excess O2 saves 0.5-1% fuel.",
        typical_savings_pct=Decimal("3"), typical_payback_months=2,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=[
            BuildingType.MANUFACTURING.value, BuildingType.HEALTHCARE.value,
            BuildingType.HOSPITALITY.value, BuildingType.EDUCATION.value,
        ],
        co_benefits=["Reduced NOx emissions", "Extended boiler life"],
        prerequisites=["Boiler without recent combustion analysis"],
    ),

    # ---------------------------------------------------------------
    # RENEWABLE (3 actions)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-RN-001", category=ActionCategory.RENEWABLE.value,
        title="Rooftop solar PV feasibility assessment",
        description="Commission professional solar PV feasibility study including structural, shading, and financial analysis for rooftop PV installation.",
        typical_savings_pct=Decimal("15"), typical_payback_months=60,
        complexity=ActionComplexity.CAPITAL.value,
        disruption=DisruptionLevel.MODERATE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Green credentials", "Energy independence", "Hedge against price rises"],
        prerequisites=["Suitable unshaded roof area"],
    ),
    QuickWinAction(
        action_code="QW-RN-002", category=ActionCategory.RENEWABLE.value,
        title="Solar thermal for domestic hot water",
        description="Install solar thermal collectors to preheat domestic hot water. Effective in buildings with high hot water demand.",
        typical_savings_pct=Decimal("40"), typical_payback_months=48,
        complexity=ActionComplexity.CAPITAL.value,
        disruption=DisruptionLevel.MODERATE.value,
        applicable_building_types=[
            BuildingType.HOSPITALITY.value, BuildingType.HEALTHCARE.value,
            BuildingType.FITNESS.value, BuildingType.MULTIFAMILY.value,
        ],
        co_benefits=["Reduced gas/oil dependency"],
        prerequisites=["High hot water demand and suitable roof space"],
    ),
    QuickWinAction(
        action_code="QW-RN-003", category=ActionCategory.RENEWABLE.value,
        title="Green power purchasing agreement",
        description="Enter into a green power purchase agreement (PPA) or buy renewable energy certificates (RECs) to reduce Scope 2 emissions.",
        typical_savings_pct=Decimal("0"), typical_payback_months=0,
        complexity=ActionComplexity.NO_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        is_behavioral=True,
        co_benefits=["Scope 2 emission reduction", "Green branding"],
        prerequisites=["Electricity procurement flexibility"],
    ),

    # ---------------------------------------------------------------
    # PROCESS (6 actions)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-PR-001", category=ActionCategory.PROCESS.value,
        title="Waste heat recovery from exhaust streams",
        description="Install heat exchangers to capture waste heat from process exhaust streams for preheating combustion air, feedwater, or space heating.",
        typical_savings_pct=Decimal("12"), typical_payback_months=30,
        complexity=ActionComplexity.MEDIUM_COST.value,
        disruption=DisruptionLevel.MODERATE.value,
        applicable_building_types=[
            BuildingType.MANUFACTURING.value,
        ],
        co_benefits=["Reduced stack emissions", "Improved process efficiency"],
        prerequisites=["High-temperature exhaust streams (> 150 degC)"],
    ),
    QuickWinAction(
        action_code="QW-PR-002", category=ActionCategory.PROCESS.value,
        title="Production scheduling optimisation",
        description="Optimise production scheduling to consolidate operations into fewer shifts and reduce idle equipment run time.",
        typical_savings_pct=Decimal("8"), typical_payback_months=0,
        complexity=ActionComplexity.NO_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=[
            BuildingType.MANUFACTURING.value,
        ],
        is_behavioral=True,
        co_benefits=["Reduced labour costs", "Equipment life extension"],
        prerequisites=["Multi-shift operation with variable demand"],
    ),
    QuickWinAction(
        action_code="QW-PR-003", category=ActionCategory.PROCESS.value,
        title="Batch process consolidation",
        description="Consolidate partial batch runs into full batches to improve energy intensity per unit of production.",
        typical_savings_pct=Decimal("6"), typical_payback_months=0,
        complexity=ActionComplexity.NO_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=[
            BuildingType.MANUFACTURING.value,
        ],
        is_behavioral=True,
        co_benefits=["Improved yield", "Reduced material waste"],
        prerequisites=["Batch processes running at partial capacity"],
    ),
    QuickWinAction(
        action_code="QW-PR-004", category=ActionCategory.PROCESS.value,
        title="Compressed air substitution for blowing/cleaning",
        description="Replace compressed air for part blowing, cleaning, or drying with blowers, brushes, or vacuum systems (10x more efficient).",
        typical_savings_pct=Decimal("10"), typical_payback_months=6,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=[
            BuildingType.MANUFACTURING.value,
        ],
        co_benefits=["Reduced noise", "Better cleaning efficacy"],
        prerequisites=["Compressed air used for open blowing or cleaning"],
    ),
    QuickWinAction(
        action_code="QW-PR-005", category=ActionCategory.PROCESS.value,
        title="Insulation of process equipment and piping",
        description="Install or repair insulation on hot process equipment, tanks, valves, and piping operating above 50 degC.",
        typical_savings_pct=Decimal("5"), typical_payback_months=8,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=[
            BuildingType.MANUFACTURING.value,
        ],
        co_benefits=["Safety (burn prevention)", "Temperature stability"],
        prerequisites=["Uninsulated hot process surfaces"],
    ),
    QuickWinAction(
        action_code="QW-PR-006", category=ActionCategory.PROCESS.value,
        title="Power factor correction capacitors",
        description="Install automatic power factor correction capacitor bank to reduce reactive power demand charges and distribution losses.",
        typical_savings_pct=Decimal("3"), typical_payback_months=18,
        complexity=ActionComplexity.MEDIUM_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=[
            BuildingType.MANUFACTURING.value, BuildingType.DATA_CENTER.value,
            BuildingType.WAREHOUSE.value,
        ],
        co_benefits=["Reduced demand charges", "Voltage improvement"],
        prerequisites=["Power factor below 0.90"],
    ),

    # ---------------------------------------------------------------
    # Additional HVAC (2 more actions)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-HV-010", category=ActionCategory.HVAC.value,
        title="Eliminate simultaneous heating and cooling",
        description="Identify and correct HVAC zones where heating and cooling operate simultaneously. Common in dual-duct and multi-zone systems.",
        typical_savings_pct=Decimal("12"), typical_payback_months=0,
        complexity=ActionComplexity.NO_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        is_behavioral=True,
        co_benefits=["Improved comfort", "Reduced equipment wear"],
        prerequisites=["Dual-duct or multi-zone HVAC system"],
    ),
    QuickWinAction(
        action_code="QW-HV-011", category=ActionCategory.HVAC.value,
        title="Close outdoor air dampers during unoccupied hours",
        description="Programme BMS or install timer to close outdoor air dampers during unoccupied hours to eliminate unnecessary ventilation energy.",
        typical_savings_pct=Decimal("6"), typical_payback_months=1,
        complexity=ActionComplexity.NO_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        is_behavioral=True,
        co_benefits=["Reduced heating/cooling load"],
        prerequisites=["AHU with motorised outdoor air dampers"],
    ),

    # ---------------------------------------------------------------
    # Additional LIGHTING (2 more actions)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-LT-009", category=ActionCategory.LIGHTING.value,
        title="Clean light fixtures and lenses",
        description="Clean accumulated dust and dirt from light fixtures, lenses, and reflectors to restore light output without additional energy.",
        typical_savings_pct=Decimal("10"), typical_payback_months=0,
        complexity=ActionComplexity.NO_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        is_behavioral=True,
        co_benefits=["Improved illumination quality"],
        prerequisites=["Dirty or dusty light fixtures"],
    ),
    QuickWinAction(
        action_code="QW-LT-010", category=ActionCategory.LIGHTING.value,
        title="Reduce lighting in non-critical areas",
        description="Reduce lighting levels in corridors, stairwells, and storage areas to minimum safe levels per local codes.",
        typical_savings_pct=Decimal("12"), typical_payback_months=0,
        complexity=ActionComplexity.NO_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        is_behavioral=True,
        co_benefits=["Extended lamp life"],
        prerequisites=["Over-lit non-critical areas"],
    ),

    # ---------------------------------------------------------------
    # Additional ENVELOPE (2 more actions)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-EN-007", category=ActionCategory.ENVELOPE.value,
        title="Automatic door closers on external doors",
        description="Install self-closing mechanisms on frequently used external doors to reduce air infiltration from doors left open.",
        typical_savings_pct=Decimal("4"), typical_payback_months=4,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Security improvement", "Pest prevention"],
        prerequisites=["Frequently used external doors without closers"],
    ),
    QuickWinAction(
        action_code="QW-EN-008", category=ActionCategory.ENVELOPE.value,
        title="Insulate hot water storage tanks and valves",
        description="Add removable insulation covers to uninsulated valves, flanges, and fittings on hot water and steam distribution systems.",
        typical_savings_pct=Decimal("6"), typical_payback_months=6,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=[
            BuildingType.MANUFACTURING.value, BuildingType.HEALTHCARE.value,
            BuildingType.HOSPITALITY.value, BuildingType.EDUCATION.value,
        ],
        co_benefits=["Safety (burn prevention)", "Reduced condensation"],
        prerequisites=["Uninsulated valves and fittings above 50 degC"],
    ),

    # ---------------------------------------------------------------
    # Additional CONTROLS (1 more action)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-CT-005", category=ActionCategory.CONTROLS.value,
        title="Install sub-metering on major loads",
        description="Install energy sub-meters on major HVAC, lighting, and process loads to enable measurement-based energy management.",
        typical_savings_pct=Decimal("5"), typical_payback_months=12,
        complexity=ActionComplexity.MEDIUM_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Visibility into consumption patterns", "ISO 50001 readiness"],
        prerequisites=["Absence of load-level sub-metering"],
    ),

    # ---------------------------------------------------------------
    # Additional MAINTENANCE (2 more actions)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-MN-006", category=ActionCategory.MAINTENANCE.value,
        title="Refrigerant charge verification",
        description="Verify and correct refrigerant charge levels in DX cooling systems. Overcharge or undercharge reduces efficiency by 5-20%.",
        typical_savings_pct=Decimal("7"), typical_payback_months=2,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Extended compressor life", "Better cooling capacity"],
        prerequisites=["DX cooling systems without recent refrigerant check"],
    ),
    QuickWinAction(
        action_code="QW-MN-007", category=ActionCategory.MAINTENANCE.value,
        title="Calibrate sensors and thermostats",
        description="Calibrate temperature, humidity, pressure, and flow sensors. Drifted sensors cause comfort complaints and energy waste.",
        typical_savings_pct=Decimal("4"), typical_payback_months=2,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=_ALL_BUILDING_TYPES,
        co_benefits=["Improved comfort", "Better BMS performance"],
        prerequisites=["Sensors not calibrated in past 2 years"],
    ),

    # ---------------------------------------------------------------
    # Additional WATER HEATING (1 more action)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-WH-006", category=ActionCategory.WATER_HEATING.value,
        title="Point-of-use water heaters for remote fixtures",
        description="Install small point-of-use electric water heaters for isolated fixtures far from central plant to eliminate distribution losses.",
        typical_savings_pct=Decimal("12"), typical_payback_months=18,
        complexity=ActionComplexity.LOW_COST.value,
        disruption=DisruptionLevel.MINIMAL.value,
        applicable_building_types=[
            BuildingType.OFFICE.value, BuildingType.WAREHOUSE.value,
            BuildingType.MANUFACTURING.value, BuildingType.RETAIL.value,
        ],
        co_benefits=["Instant hot water", "Reduced pipe heat losses"],
        prerequisites=["Remote fixtures with long pipe runs from central plant"],
    ),

    # ---------------------------------------------------------------
    # Additional PLUG LOADS (1 more action)
    # ---------------------------------------------------------------
    QuickWinAction(
        action_code="QW-PL-005", category=ActionCategory.PLUG_LOADS.value,
        title="ENERGY STAR office equipment procurement policy",
        description="Establish procurement policy requiring ENERGY STAR certification for all new monitors, computers, copiers, and appliances.",
        typical_savings_pct=Decimal("10"), typical_payback_months=0,
        complexity=ActionComplexity.NO_COST.value,
        disruption=DisruptionLevel.NONE.value,
        applicable_building_types=[
            BuildingType.OFFICE.value, BuildingType.EDUCATION.value,
            BuildingType.HEALTHCARE.value, BuildingType.MIXED_USE.value,
        ],
        is_behavioral=True,
        co_benefits=["Lower lifecycle cost", "Consistent quality"],
        prerequisites=[],
    ),
]

# Validate library count at module load.
assert len(QUICK_WINS_LIBRARY) >= 80, (
    f"QUICK_WINS_LIBRARY must have >= 80 entries, found {len(QUICK_WINS_LIBRARY)}"
)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class QuickWinsScannerEngine:
    """Automated facility scanner for quick-win energy efficiency opportunities.

    Evaluates a facility profile and equipment survey against the 84-action
    quick-wins library.  For each action, calculates an applicability score,
    estimates energy/cost/CO2e savings, and assigns a priority.  Results are
    sorted by priority and savings magnitude, with a SHA-256 provenance hash.

    Usage::

        engine = QuickWinsScannerEngine()
        result = engine.scan_facility(facility, equipment)
        for action in result.applicable_actions:
            print(f"{action.action.title}: {action.estimated_savings_kwh} kWh/yr")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise QuickWinsScannerEngine.

        Args:
            config: Optional overrides.  Supported keys:
                - co2_factor_kg_kwh (Decimal): grid emission factor
                - energy_price_eur_kwh (Decimal): default energy price
                - min_applicability (Decimal): minimum applicability threshold (0-100)
                - min_confidence (Decimal): minimum confidence threshold (0-100)
        """
        self.config = config or {}
        self._co2_factor = _decimal(
            self.config.get("co2_factor_kg_kwh", DEFAULT_CO2_FACTOR_KG_KWH)
        )
        self._energy_price = _decimal(
            self.config.get("energy_price_eur_kwh", DEFAULT_ENERGY_PRICE_EUR_KWH)
        )
        self._min_applicability = _decimal(
            self.config.get("min_applicability", Decimal("20"))
        )
        self._min_confidence = _decimal(
            self.config.get("min_confidence", Decimal("10"))
        )
        logger.info(
            "QuickWinsScannerEngine v%s initialised (library=%d actions)",
            self.engine_version, len(QUICK_WINS_LIBRARY),
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def scan_facility(
        self,
        facility: FacilityProfile,
        equipment: EquipmentSurvey,
    ) -> QuickWinsScanResult:
        """Scan a facility for applicable quick-win actions.

        Filters the library by building-type, scores applicability against
        the equipment survey, estimates savings, and returns prioritised
        results with provenance hash.

        Args:
            facility: Facility profile with energy and building data.
            equipment: Equipment survey answers.

        Returns:
            QuickWinsScanResult with all applicable actions.
        """
        t0 = time.perf_counter()
        logger.info(
            "Quick-win scan: facility=%s, type=%s, area=%.0f m2",
            facility.name, facility.building_type, float(facility.floor_area_m2),
        )

        # Derive unit energy price from facility data or config default.
        unit_price = self._derive_unit_price(facility)

        # Step 1: Filter by building type.
        type_filtered = self._filter_by_building_type(
            QUICK_WINS_LIBRARY, facility.building_type
        )
        total_scanned = len(type_filtered)

        # Step 2: Score, estimate, and prioritise each action.
        scan_results: List[ScanResult] = []
        for action in type_filtered:
            applicability = self._calculate_applicability(
                action, facility, equipment
            )
            if applicability < self._min_applicability:
                continue

            savings_kwh, savings_cost = self._estimate_savings(
                action, facility, unit_price
            )
            co2e_reduction = self._estimate_co2e(savings_kwh, facility)
            impl_cost = self._estimate_implementation_cost(
                action, facility
            )
            payback = self._estimate_payback(impl_cost, savings_cost)
            priority = self._calculate_priority(
                applicability, savings_cost, payback, action.disruption
            )
            confidence = self._estimate_confidence(
                action, facility, equipment
            )

            if confidence < self._min_confidence:
                continue

            notes = self._generate_notes(action, facility, equipment)

            scan_results.append(ScanResult(
                action=action,
                applicability_score=_round_val(applicability, 1),
                estimated_savings_kwh=_round_val(savings_kwh, 2),
                estimated_savings_cost=_round_val(savings_cost, 2),
                estimated_co2e_reduction=_round_val(co2e_reduction, 2),
                implementation_cost=_round_val(impl_cost, 2),
                payback_months=_round_val(payback, 1),
                priority=priority,
                confidence_pct=_round_val(confidence, 1),
                notes=notes,
            ))

        # Step 3: Sort by priority rank then savings descending.
        priority_rank = {
            ActionPriority.CRITICAL.value: 0,
            ActionPriority.HIGH.value: 1,
            ActionPriority.MEDIUM.value: 2,
            ActionPriority.LOW.value: 3,
        }
        scan_results.sort(
            key=lambda r: (
                priority_rank.get(r.priority, 99),
                -r.estimated_savings_cost,
            )
        )

        # Step 4: Aggregate totals.
        total_savings_kwh = sum(
            (r.estimated_savings_kwh for r in scan_results), Decimal("0")
        )
        total_savings_cost = sum(
            (r.estimated_savings_cost for r in scan_results), Decimal("0")
        )
        total_co2e = sum(
            (r.estimated_co2e_reduction for r in scan_results), Decimal("0")
        )
        total_impl_cost = sum(
            (r.implementation_cost for r in scan_results), Decimal("0")
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        result = QuickWinsScanResult(
            facility=facility,
            total_actions_scanned=total_scanned,
            applicable_actions=scan_results,
            total_savings_kwh=_round_val(total_savings_kwh, 2),
            total_savings_cost=_round_val(total_savings_cost, 2),
            total_co2e_reduction=_round_val(total_co2e, 2),
            total_implementation_cost=_round_val(total_impl_cost, 2),
            scan_duration_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Quick-win scan complete: %d/%d applicable, "
            "%.0f kWh saved, %.0f EUR saved, hash=%s",
            len(scan_results), total_scanned,
            float(total_savings_kwh), float(total_savings_cost),
            result.provenance_hash[:16],
        )
        return result

    def get_actions_by_category(
        self, category: ActionCategory,
    ) -> List[QuickWinAction]:
        """Return all library actions for a given category.

        Args:
            category: The action category to filter by.

        Returns:
            List of QuickWinAction matching the category.
        """
        return [
            a for a in QUICK_WINS_LIBRARY
            if a.category == category.value
        ]

    def get_actions_by_complexity(
        self, complexity: ActionComplexity,
    ) -> List[QuickWinAction]:
        """Return all library actions for a given complexity tier.

        Args:
            complexity: The complexity level to filter by.

        Returns:
            List of QuickWinAction matching the complexity.
        """
        return [
            a for a in QUICK_WINS_LIBRARY
            if a.complexity == complexity.value
        ]

    def get_behavioral_actions(self) -> List[QuickWinAction]:
        """Return all behavioural (no-cost, human-change) actions.

        Returns:
            List of QuickWinAction flagged as behavioural.
        """
        return [a for a in QUICK_WINS_LIBRARY if a.is_behavioral]

    def get_library_stats(self) -> Dict[str, Any]:
        """Return summary statistics of the quick-wins library.

        Returns:
            Dictionary with category counts, complexity distribution,
            behavioural count, and total count.
        """
        category_counts: Dict[str, int] = {}
        complexity_counts: Dict[str, int] = {}
        behavioral_count = 0

        for action in QUICK_WINS_LIBRARY:
            category_counts[action.category] = (
                category_counts.get(action.category, 0) + 1
            )
            complexity_counts[action.complexity] = (
                complexity_counts.get(action.complexity, 0) + 1
            )
            if action.is_behavioral:
                behavioral_count += 1

        return {
            "total_actions": len(QUICK_WINS_LIBRARY),
            "categories": len(category_counts),
            "category_counts": category_counts,
            "complexity_counts": complexity_counts,
            "behavioral_count": behavioral_count,
            "non_behavioral_count": len(QUICK_WINS_LIBRARY) - behavioral_count,
            "engine_version": self.engine_version,
        }

    # ------------------------------------------------------------------ #
    # Filtering                                                           #
    # ------------------------------------------------------------------ #

    def _filter_by_building_type(
        self,
        actions: List[QuickWinAction],
        building_type: str,
    ) -> List[QuickWinAction]:
        """Filter actions applicable to a building type.

        Args:
            actions: Full action library.
            building_type: Building type value string.

        Returns:
            Filtered list of applicable actions.
        """
        return [
            a for a in actions
            if not a.applicable_building_types
            or building_type in a.applicable_building_types
        ]

    # ------------------------------------------------------------------ #
    # Applicability Scoring                                               #
    # ------------------------------------------------------------------ #

    def _calculate_applicability(
        self,
        action: QuickWinAction,
        facility: FacilityProfile,
        equipment: EquipmentSurvey,
    ) -> Decimal:
        """Calculate applicability score (0-100) for an action.

        Scoring combines:
            - Base building-type match (30 pts)
            - Equipment survey alignment (40 pts)
            - Equipment age bonus (15 pts)
            - Operating hours bonus (15 pts)

        Args:
            action: Quick-win action to evaluate.
            facility: Facility profile.
            equipment: Equipment survey.

        Returns:
            Applicability score as Decimal (0-100).
        """
        score = Decimal("0")

        # Base score: building type is applicable (30 pts).
        if (not action.applicable_building_types
                or facility.building_type in action.applicable_building_types):
            score += Decimal("30")

        # Equipment alignment score (40 pts).
        equip_score = self._score_equipment_alignment(
            action, equipment
        )
        score += equip_score

        # Equipment age bonus (15 pts): older equipment = more opportunity.
        age_bonus = self._score_equipment_age(
            action, facility, equipment
        )
        score += age_bonus

        # Operating hours bonus (15 pts): more hours = more savings.
        hours_bonus = self._score_operating_hours(
            action, facility
        )
        score += hours_bonus

        return min(score, Decimal("100"))

    def _score_equipment_alignment(
        self,
        action: QuickWinAction,
        equipment: EquipmentSurvey,
    ) -> Decimal:
        """Score equipment survey alignment for an action (0-40).

        Args:
            action: Quick-win action.
            equipment: Equipment survey data.

        Returns:
            Equipment alignment score (0-40).
        """
        category = action.category
        max_pts = Decimal("40")

        # LIGHTING: higher score if not already fully LED.
        if category == ActionCategory.LIGHTING.value:
            if action.action_code in ("QW-LT-001", "QW-LT-007", "QW-LT-008"):
                # LED retrofit: score inversely to existing LED percentage.
                led_gap = Decimal("100") - equipment.pct_led
                return max_pts * led_gap / Decimal("100")
            if action.action_code == "QW-LT-003":
                # Occupancy sensors: higher if none installed.
                return max_pts if not equipment.has_occupancy_sensors else Decimal("10")
            if action.action_code == "QW-LT-004":
                return max_pts if not equipment.has_occupancy_sensors else Decimal("15")
            # Default lighting score.
            return max_pts * (Decimal("100") - equipment.pct_led) / Decimal("100")

        # HVAC: score based on HVAC configuration.
        if category == ActionCategory.HVAC.value:
            if action.action_code == "QW-HV-001":
                return max_pts if not equipment.has_programmable_thermostats else Decimal("15")
            if action.action_code == "QW-HV-004":
                return max_pts if not equipment.has_vsd_hvac else Decimal("5")
            if action.action_code == "QW-HV-009":
                return max_pts  # always applicable for qualifying building types
            # General HVAC: older system = more opportunity.
            age_factor = min(_decimal(equipment.hvac_age_years) / Decimal("20"), Decimal("1"))
            return max_pts * age_factor

        # ENVELOPE: score based on insulation values.
        if category == ActionCategory.ENVELOPE.value:
            if action.action_code in ("QW-EN-003", "QW-EN-004"):
                # Window film / roof coating: score by U-value / R-value.
                if action.action_code == "QW-EN-003":
                    u_factor = min(equipment.window_u_value / Decimal("6"), Decimal("1"))
                    return max_pts * u_factor
                roof_gap = max(Decimal("5") - equipment.roof_insulation_r_value, Decimal("0"))
                return max_pts * min(roof_gap / Decimal("5"), Decimal("1"))
            return max_pts * Decimal("0.7")  # general envelope

        # COMPRESSED AIR: only if facility has compressed air.
        if category == ActionCategory.COMPRESSED_AIR.value:
            if not equipment.has_compressed_air:
                return Decimal("0")
            age_factor = min(
                _decimal(equipment.compressed_air_age_years) / Decimal("15"),
                Decimal("1"),
            )
            return max_pts * max(age_factor, Decimal("0.5"))

        # MOTORS_DRIVES: manufacturing bonus.
        if category == ActionCategory.MOTORS_DRIVES.value:
            return max_pts * Decimal("0.7")

        # CONTROLS: higher if no BMS.
        if category == ActionCategory.CONTROLS.value:
            if action.action_code == "QW-CT-001":
                return max_pts if equipment.has_bms else Decimal("0")
            if action.action_code == "QW-CT-002":
                return max_pts if equipment.has_bms else Decimal("10")
            return max_pts * Decimal("0.6")

        # WATER_HEATING: general applicability.
        if category == ActionCategory.WATER_HEATING.value:
            return max_pts * Decimal("0.6")

        # PLUG_LOADS: general applicability.
        if category == ActionCategory.PLUG_LOADS.value:
            return max_pts * Decimal("0.6")

        # REFRIGERATION: based on refrigeration type presence.
        if category == ActionCategory.REFRIGERATION.value:
            if equipment.refrigeration_type:
                return max_pts * Decimal("0.8")
            return max_pts * Decimal("0.3")

        # KITCHEN: food-service types get full score.
        if category == ActionCategory.KITCHEN.value:
            return max_pts * Decimal("0.7")

        # LAUNDRY: based on building type.
        if category == ActionCategory.LAUNDRY.value:
            return max_pts * Decimal("0.6")

        # BEHAVIORAL: always applicable.
        if category == ActionCategory.BEHAVIORAL.value:
            return max_pts * Decimal("0.8")

        # MAINTENANCE: always somewhat applicable.
        if category == ActionCategory.MAINTENANCE.value:
            if action.action_code == "QW-MN-001":
                return max_pts if equipment.has_steam_system else Decimal("0")
            return max_pts * Decimal("0.6")

        # RENEWABLE: moderate applicability.
        if category == ActionCategory.RENEWABLE.value:
            return max_pts * Decimal("0.5")

        # PROCESS: manufacturing-specific.
        if category == ActionCategory.PROCESS.value:
            return max_pts * Decimal("0.6")

        return max_pts * Decimal("0.5")

    def _score_equipment_age(
        self,
        action: QuickWinAction,
        facility: FacilityProfile,
        equipment: EquipmentSurvey,
    ) -> Decimal:
        """Score equipment age bonus (0-15).

        Older equipment generally presents more quick-win opportunity.

        Args:
            action: Quick-win action.
            facility: Facility profile.
            equipment: Equipment survey.

        Returns:
            Age bonus score (0-15).
        """
        max_pts = Decimal("15")
        category = action.category

        if category == ActionCategory.HVAC.value:
            age = _decimal(equipment.hvac_age_years)
        elif category == ActionCategory.COMPRESSED_AIR.value:
            age = _decimal(equipment.compressed_air_age_years)
        else:
            age = _decimal(facility.equipment_age_years)

        # Linear scale: 0 years = 0 pts, 20+ years = 15 pts.
        factor = min(age / Decimal("20"), Decimal("1"))
        return max_pts * factor

    def _score_operating_hours(
        self,
        action: QuickWinAction,
        facility: FacilityProfile,
    ) -> Decimal:
        """Score operating hours bonus (0-15).

        Higher operating hours amplify savings opportunity.

        Args:
            action: Quick-win action.
            facility: Facility profile.

        Returns:
            Operating hours bonus (0-15).
        """
        max_pts = Decimal("15")
        hours = _decimal(facility.operating_hours)
        # Scale: 2000 hrs = 5 pts, 4000 hrs = 10 pts, 8760 hrs = 15 pts.
        factor = min(hours / Decimal("8760"), Decimal("1"))
        return max_pts * factor

    # ------------------------------------------------------------------ #
    # Savings Estimation                                                  #
    # ------------------------------------------------------------------ #

    def _estimate_savings(
        self,
        action: QuickWinAction,
        facility: FacilityProfile,
        unit_price: Decimal,
    ) -> Tuple[Decimal, Decimal]:
        """Estimate annual energy and cost savings for an action.

        savings_kwh = annual_energy * category_share * typical_savings_pct / 100
        savings_cost = savings_kwh * unit_price

        Args:
            action: Quick-win action with typical savings percentage.
            facility: Facility profile with energy data.
            unit_price: Energy unit price (EUR/kWh).

        Returns:
            Tuple of (savings_kwh, savings_cost).
        """
        annual_kwh = facility.annual_energy_kwh
        if annual_kwh <= Decimal("0"):
            return Decimal("0"), Decimal("0")

        # Get category energy share for this building type.
        shares = CATEGORY_ENERGY_SHARE.get(facility.building_type, {})
        category_share = shares.get(action.category, Decimal("0.05"))

        # Calculate category energy.
        category_energy = annual_kwh * category_share

        # Apply typical savings percentage.
        savings_kwh = category_energy * action.typical_savings_pct / Decimal("100")

        # Cost savings.
        savings_cost = savings_kwh * unit_price

        return savings_kwh, savings_cost

    def _estimate_co2e(
        self,
        savings_kwh: Decimal,
        facility: FacilityProfile,
    ) -> Decimal:
        """Estimate CO2e reduction from energy savings.

        Args:
            savings_kwh: Annual energy savings (kWh).
            facility: Facility profile (unused currently, for future
                      per-carrier emission factors).

        Returns:
            Estimated CO2e reduction in kg/year.
        """
        return savings_kwh * self._co2_factor

    def _estimate_implementation_cost(
        self,
        action: QuickWinAction,
        facility: FacilityProfile,
    ) -> Decimal:
        """Estimate implementation cost based on complexity and floor area.

        Args:
            action: Quick-win action with complexity tier.
            facility: Facility profile with floor area.

        Returns:
            Estimated implementation cost (EUR).
        """
        if action.complexity == ActionComplexity.NO_COST.value:
            return Decimal("0")

        cost_per_m2 = COMPLEXITY_COST_PER_M2.get(
            action.complexity, Decimal("5")
        )
        # Scale by floor area but apply reasonable bounds.
        raw_cost = cost_per_m2 * facility.floor_area_m2

        # Apply minimum and maximum bounds by complexity.
        if action.complexity == ActionComplexity.LOW_COST.value:
            return max(min(raw_cost, Decimal("10000")), Decimal("200"))
        elif action.complexity == ActionComplexity.MEDIUM_COST.value:
            return max(min(raw_cost, Decimal("75000")), Decimal("2000"))
        elif action.complexity == ActionComplexity.CAPITAL.value:
            return max(min(raw_cost, Decimal("500000")), Decimal("10000"))

        return raw_cost

    def _estimate_payback(
        self,
        implementation_cost: Decimal,
        annual_savings_cost: Decimal,
    ) -> Decimal:
        """Estimate payback period in months.

        Args:
            implementation_cost: Estimated cost (EUR).
            annual_savings_cost: Annual cost savings (EUR).

        Returns:
            Payback in months.
        """
        if implementation_cost <= Decimal("0"):
            return Decimal("0")
        monthly_savings = _safe_divide(annual_savings_cost, Decimal("12"))
        if monthly_savings <= Decimal("0"):
            return Decimal("999")
        return _safe_divide(implementation_cost, monthly_savings)

    # ------------------------------------------------------------------ #
    # Priority Calculation                                                #
    # ------------------------------------------------------------------ #

    def _calculate_priority(
        self,
        applicability: Decimal,
        savings_cost: Decimal,
        payback_months: Decimal,
        disruption: str,
    ) -> str:
        """Calculate action priority from multi-criteria scoring.

        Scoring weights:
            Applicability contribution:     25%
            Savings magnitude:              25%
            Payback speed (inverse):        30%
            Disruption (inverse):           20%

        Args:
            applicability: Applicability score (0-100).
            savings_cost: Annual cost savings (EUR).
            payback_months: Payback period (months).
            disruption: Disruption level value string.

        Returns:
            ActionPriority value string.
        """
        score = Decimal("0")

        # Applicability (25 pts).
        score += applicability * Decimal("25") / Decimal("100")

        # Savings magnitude (25 pts): log scale.
        if savings_cost > Decimal("0"):
            log_savings = _decimal(math.log10(max(float(savings_cost), 1.0)))
            # Scale: 100 EUR = 2 -> 10 pts, 1000 EUR = 3 -> 15 pts,
            #        10000 EUR = 4 -> 20 pts, 100000 = 5 -> 25 pts
            savings_pts = min(log_savings * Decimal("5"), Decimal("25"))
            score += max(savings_pts, Decimal("0"))

        # Payback speed (30 pts): shorter = better.
        if payback_months <= Decimal("0"):
            score += Decimal("30")  # Immediate / no cost.
        elif payback_months <= Decimal("6"):
            score += Decimal("27")
        elif payback_months <= Decimal("12"):
            score += Decimal("22")
        elif payback_months <= Decimal("24"):
            score += Decimal("15")
        elif payback_months <= Decimal("36"):
            score += Decimal("8")
        elif payback_months <= Decimal("60"):
            score += Decimal("4")
        # else: 0 pts for > 60 months payback.

        # Disruption (20 pts): less disruption = higher score.
        disruption_factor = DISRUPTION_WEIGHT.get(disruption, Decimal("0.5"))
        score += Decimal("20") * disruption_factor

        # Classify.
        if score >= Decimal("75"):
            return ActionPriority.CRITICAL.value
        elif score >= Decimal("55"):
            return ActionPriority.HIGH.value
        elif score >= Decimal("35"):
            return ActionPriority.MEDIUM.value
        else:
            return ActionPriority.LOW.value

    # ------------------------------------------------------------------ #
    # Confidence Estimation                                               #
    # ------------------------------------------------------------------ #

    def _estimate_confidence(
        self,
        action: QuickWinAction,
        facility: FacilityProfile,
        equipment: EquipmentSurvey,
    ) -> Decimal:
        """Estimate confidence in the savings estimate (0-100).

        Higher confidence when:
            - Facility has detailed energy data.
            - Action is a well-proven measure (high typical_savings_pct data).
            - Equipment survey provides specific information.
            - Behavioural actions have lower inherent confidence.

        Args:
            action: Quick-win action.
            facility: Facility profile.
            equipment: Equipment survey.

        Returns:
            Confidence percentage (0-100).
        """
        confidence = Decimal("50")  # Base confidence.

        # Energy data quality bonus (up to 20 pts).
        if facility.annual_energy_kwh > Decimal("0"):
            confidence += Decimal("10")
        if facility.annual_energy_cost > Decimal("0"):
            confidence += Decimal("10")

        # Action specificity bonus (up to 15 pts).
        if action.typical_savings_pct > Decimal("0"):
            confidence += Decimal("10")
        if action.typical_payback_months > 0:
            confidence += Decimal("5")

        # Behavioural penalty (savings depend on human behaviour).
        if action.is_behavioral:
            confidence -= Decimal("15")

        # Complexity penalty (higher complexity = more uncertainty).
        if action.complexity == ActionComplexity.CAPITAL.value:
            confidence -= Decimal("10")
        elif action.complexity == ActionComplexity.MEDIUM_COST.value:
            confidence -= Decimal("5")

        return min(max(confidence, Decimal("0")), Decimal("100"))

    # ------------------------------------------------------------------ #
    # Notes Generation                                                    #
    # ------------------------------------------------------------------ #

    def _generate_notes(
        self,
        action: QuickWinAction,
        facility: FacilityProfile,
        equipment: EquipmentSurvey,
    ) -> str:
        """Generate contextual notes for a scan result.

        Args:
            action: Quick-win action.
            facility: Facility profile.
            equipment: Equipment survey.

        Returns:
            Contextual note string.
        """
        notes_parts: List[str] = []

        # Prerequisite check.
        if action.prerequisites:
            notes_parts.append(
                f"Prerequisites: {'; '.join(action.prerequisites)}."
            )

        # Co-benefits.
        if action.co_benefits:
            notes_parts.append(
                f"Co-benefits: {', '.join(action.co_benefits)}."
            )

        # Category-specific notes.
        if (action.category == ActionCategory.LIGHTING.value
                and equipment.pct_led > Decimal("80")):
            notes_parts.append(
                "Facility already has >80% LED; residual savings may be limited."
            )

        if (action.category == ActionCategory.COMPRESSED_AIR.value
                and not equipment.has_compressed_air):
            notes_parts.append(
                "No compressed air system reported; verify before proceeding."
            )

        if (action.category == ActionCategory.HVAC.value
                and equipment.hvac_age_years < 5):
            notes_parts.append(
                "Relatively new HVAC equipment; verify measure applicability."
            )

        return " ".join(notes_parts)

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _derive_unit_price(self, facility: FacilityProfile) -> Decimal:
        """Derive energy unit price from facility data or config.

        Args:
            facility: Facility profile.

        Returns:
            Energy unit price (EUR/kWh).
        """
        if (facility.annual_energy_cost > Decimal("0")
                and facility.annual_energy_kwh > Decimal("0")):
            return _safe_divide(
                facility.annual_energy_cost,
                facility.annual_energy_kwh,
                self._energy_price,
            )
        return self._energy_price
