# -*- coding: utf-8 -*-
"""
EquipmentEfficiencyEngine - PACK-031 Industrial Energy Audit Engine 4
======================================================================

Calculates equipment-level energy efficiency for industrial equipment
including motors, pumps, compressors, boilers, HVAC systems, furnaces,
and steam turbines. Identifies efficiency gaps against best-practice
benchmarks and calculates replacement ROI for each equipment class.

Supports motor efficiency classification per IEC 60034-30-1 (IE1-IE5),
pump affinity laws, compressor specific power analysis, boiler direct
and indirect method per EN 12953, HVAC seasonal performance metrics,
and equipment degradation tracking over time.

Calculation Methodology:
    Motor System Efficiency:
        system_eff = motor_eff * vsd_eff * transmission_eff * driven_eff
        energy_waste = rated_power * hours * (1 - system_eff / best_practice_eff)

    Pump Affinity Laws (ISO 9906):
        flow ~ speed,  head ~ speed^2,  power ~ speed^3
        savings_pct = 1 - (reduced_speed / full_speed)^3

    Compressor Specific Power (ISO 1217):
        specific_power = input_power_kw / fad_m3min
        gap = actual_specific_power - benchmark_specific_power

    Boiler Efficiency (EN 12953 / ASME PTC 4):
        direct:   eff = (steam_output / fuel_input) * 100
        indirect: eff = 100 - sum(losses_pct)
        losses:   stack, blowdown, radiation, unburned, moisture

    HVAC System Efficiency:
        COP = heating_output / electrical_input
        EER = cooling_output_btu / electrical_input_wh
        SEER = seasonal_cooling / seasonal_input

    Equipment Degradation:
        degraded_eff = base_eff * (1 - degradation_rate_per_year * age_years)
        min_degraded = base_eff * 0.70  (floor at 70% of original)

Regulatory References:
    - IEC 60034-30-1:2014 - Motor efficiency classes (IE1-IE5)
    - IEC 61800-9-2:2017 - Power drive systems efficiency
    - ISO 9906:2012 - Rotodynamic pumps (hydraulic performance)
    - ISO 1217:2009 - Displacement compressors acceptance tests
    - EN 12953:2012 - Shell boilers
    - ASME PTC 4:2013 - Fired steam generators
    - ISO 5151:2017 - Non-ducted air conditioners testing
    - EU Ecodesign Regulation (EU) 2019/1781 - Motors and VSD
    - EU Regulation (EU) 2015/1095 - Professional refrigeration

Zero-Hallucination:
    - Motor efficiency values from IEC 60034-30-1 Tables
    - Pump benchmarks from Hydraulic Institute standards
    - Compressor specific power from Compressed Air & Gas Institute
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-031 Industrial Energy Audit
Engine:  4 of 10
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


class EquipmentType(str, Enum):
    """Industrial equipment types for energy audit.

    MOTOR: Electric motor (induction, synchronous, PM).
    PUMP: Centrifugal, positive displacement, or submersible pump.
    COMPRESSOR: Air or gas compressor.
    BOILER: Steam or hot water boiler.
    HVAC: Heating, ventilation, and air conditioning system.
    FURNACE: Industrial furnace or kiln.
    STEAM_TURBINE: Steam turbine or turbo-generator.
    FAN: Industrial fan or blower.
    COOLING_TOWER: Evaporative or dry cooling tower.
    TRANSFORMER: Electrical transformer.
    """
    MOTOR = "motor"
    PUMP = "pump"
    COMPRESSOR = "compressor"
    BOILER = "boiler"
    HVAC = "hvac"
    FURNACE = "furnace"
    STEAM_TURBINE = "steam_turbine"
    FAN = "fan"
    COOLING_TOWER = "cooling_tower"
    TRANSFORMER = "transformer"


class MotorEfficiencyClass(str, Enum):
    """Motor efficiency classes per IEC 60034-30-1:2014.

    IE1: Standard efficiency.
    IE2: High efficiency.
    IE3: Premium efficiency (EU minimum since 2015 for >7.5kW).
    IE4: Super premium efficiency.
    IE5: Ultra premium efficiency (synchronous reluctance / PM).
    """
    IE1 = "IE1"
    IE2 = "IE2"
    IE3 = "IE3"
    IE4 = "IE4"
    IE5 = "IE5"


class CompressorType(str, Enum):
    """Compressor technology types.

    SCREW_FIXED: Rotary screw with fixed speed.
    SCREW_VSD: Rotary screw with variable speed drive.
    RECIPROCATING: Piston/reciprocating compressor.
    CENTRIFUGAL: Dynamic centrifugal compressor.
    SCROLL: Scroll compressor (small capacity).
    """
    SCREW_FIXED = "screw_fixed"
    SCREW_VSD = "screw_vsd"
    RECIPROCATING = "reciprocating"
    CENTRIFUGAL = "centrifugal"
    SCROLL = "scroll"


class BoilerType(str, Enum):
    """Boiler technology types.

    FIRE_TUBE: Shell/fire-tube boiler.
    WATER_TUBE: Water-tube boiler.
    CONDENSING: Condensing boiler (flue gas heat recovery).
    ELECTRIC: Electric resistance or electrode boiler.
    BIOMASS: Biomass-fired boiler.
    """
    FIRE_TUBE = "fire_tube"
    WATER_TUBE = "water_tube"
    CONDENSING = "condensing"
    ELECTRIC = "electric"
    BIOMASS = "biomass"


class FuelType(str, Enum):
    """Fuel types for combustion equipment.

    NATURAL_GAS: Natural gas / methane.
    DIESEL: Diesel / gas oil.
    HEAVY_FUEL_OIL: Heavy fuel oil (HFO / residual).
    LPG: Liquefied petroleum gas.
    COAL: Bituminous coal.
    BIOMASS_WOOD: Wood pellets / chips.
    BIOMASS_WASTE: Agricultural or organic waste.
    ELECTRICITY: Electrical energy input.
    """
    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    HEAVY_FUEL_OIL = "heavy_fuel_oil"
    LPG = "lpg"
    COAL = "coal"
    BIOMASS_WOOD = "biomass_wood"
    BIOMASS_WASTE = "biomass_waste"
    ELECTRICITY = "electricity"


class HVACType(str, Enum):
    """HVAC system types.

    SPLIT_SYSTEM: Split air conditioning system.
    CHILLER_AIR: Air-cooled chiller.
    CHILLER_WATER: Water-cooled chiller.
    VRF: Variable refrigerant flow system.
    HEAT_PUMP_AIR: Air-source heat pump.
    HEAT_PUMP_GROUND: Ground-source heat pump.
    ROOFTOP: Packaged rooftop unit.
    AHU: Air handling unit.
    """
    SPLIT_SYSTEM = "split_system"
    CHILLER_AIR = "chiller_air"
    CHILLER_WATER = "chiller_water"
    VRF = "vrf"
    HEAT_PUMP_AIR = "heat_pump_air"
    HEAT_PUMP_GROUND = "heat_pump_ground"
    ROOFTOP = "rooftop"
    AHU = "ahu"


# ---------------------------------------------------------------------------
# Constants -- Motor Efficiency Standards (IEC 60034-30-1, 50Hz, 4-pole)
# ---------------------------------------------------------------------------

# Minimum efficiency (%) by class and rated power (kW).
# Keys: power range lower bound in kW.
MOTOR_EFFICIENCY_STANDARDS: Dict[str, Dict[str, Decimal]] = {
    MotorEfficiencyClass.IE1.value: {
        "0.75": Decimal("72.1"), "1.1": Decimal("75.0"), "1.5": Decimal("77.2"),
        "2.2": Decimal("79.7"), "3": Decimal("81.5"), "4": Decimal("83.1"),
        "5.5": Decimal("84.7"), "7.5": Decimal("86.0"), "11": Decimal("87.6"),
        "15": Decimal("88.7"), "18.5": Decimal("89.3"), "22": Decimal("89.9"),
        "30": Decimal("90.7"), "37": Decimal("91.2"), "45": Decimal("91.7"),
        "55": Decimal("92.1"), "75": Decimal("92.7"), "90": Decimal("93.0"),
        "110": Decimal("93.3"), "132": Decimal("93.5"), "160": Decimal("93.8"),
        "200": Decimal("94.0"), "250": Decimal("94.0"), "315": Decimal("94.0"),
        "355": Decimal("94.0"),
    },
    MotorEfficiencyClass.IE2.value: {
        "0.75": Decimal("77.4"), "1.1": Decimal("79.6"), "1.5": Decimal("81.3"),
        "2.2": Decimal("83.2"), "3": Decimal("84.6"), "4": Decimal("85.8"),
        "5.5": Decimal("87.0"), "7.5": Decimal("88.1"), "11": Decimal("89.4"),
        "15": Decimal("90.3"), "18.5": Decimal("90.9"), "22": Decimal("91.3"),
        "30": Decimal("92.0"), "37": Decimal("92.5"), "45": Decimal("92.9"),
        "55": Decimal("93.2"), "75": Decimal("93.8"), "90": Decimal("94.1"),
        "110": Decimal("94.3"), "132": Decimal("94.6"), "160": Decimal("94.8"),
        "200": Decimal("95.0"), "250": Decimal("95.0"), "315": Decimal("95.0"),
        "355": Decimal("95.0"),
    },
    MotorEfficiencyClass.IE3.value: {
        "0.75": Decimal("80.7"), "1.1": Decimal("82.7"), "1.5": Decimal("84.2"),
        "2.2": Decimal("85.9"), "3": Decimal("87.1"), "4": Decimal("88.1"),
        "5.5": Decimal("89.2"), "7.5": Decimal("90.1"), "11": Decimal("91.2"),
        "15": Decimal("91.9"), "18.5": Decimal("92.4"), "22": Decimal("92.7"),
        "30": Decimal("93.3"), "37": Decimal("93.7"), "45": Decimal("94.0"),
        "55": Decimal("94.3"), "75": Decimal("94.7"), "90": Decimal("95.0"),
        "110": Decimal("95.2"), "132": Decimal("95.4"), "160": Decimal("95.6"),
        "200": Decimal("95.8"), "250": Decimal("95.8"), "315": Decimal("95.8"),
        "355": Decimal("95.8"),
    },
    MotorEfficiencyClass.IE4.value: {
        "0.75": Decimal("82.5"), "1.1": Decimal("84.1"), "1.5": Decimal("85.3"),
        "2.2": Decimal("86.7"), "3": Decimal("87.7"), "4": Decimal("88.6"),
        "5.5": Decimal("89.6"), "7.5": Decimal("90.4"), "11": Decimal("91.4"),
        "15": Decimal("92.1"), "18.5": Decimal("92.6"), "22": Decimal("93.0"),
        "30": Decimal("93.6"), "37": Decimal("94.0"), "45": Decimal("94.3"),
        "55": Decimal("94.6"), "75": Decimal("95.0"), "90": Decimal("95.2"),
        "110": Decimal("95.4"), "132": Decimal("95.6"), "160": Decimal("95.8"),
        "200": Decimal("96.0"), "250": Decimal("96.0"), "315": Decimal("96.0"),
        "355": Decimal("96.0"),
    },
    MotorEfficiencyClass.IE5.value: {
        "0.75": Decimal("85.5"), "1.1": Decimal("87.0"), "1.5": Decimal("88.0"),
        "2.2": Decimal("89.2"), "3": Decimal("90.0"), "4": Decimal("90.7"),
        "5.5": Decimal("91.5"), "7.5": Decimal("92.1"), "11": Decimal("92.9"),
        "15": Decimal("93.4"), "18.5": Decimal("93.8"), "22": Decimal("94.1"),
        "30": Decimal("94.5"), "37": Decimal("94.9"), "45": Decimal("95.1"),
        "55": Decimal("95.4"), "75": Decimal("95.7"), "90": Decimal("95.9"),
        "110": Decimal("96.1"), "132": Decimal("96.3"), "160": Decimal("96.5"),
        "200": Decimal("96.7"), "250": Decimal("96.7"), "315": Decimal("96.7"),
        "355": Decimal("96.7"),
    },
}

# VSD efficiency factor at various load points (IEC 61800-9-2).
VSD_EFFICIENCY_BY_LOAD: Dict[str, Decimal] = {
    "25": Decimal("0.90"), "50": Decimal("0.94"), "75": Decimal("0.96"),
    "100": Decimal("0.97"),
}

# Transmission efficiency by drive type.
TRANSMISSION_EFFICIENCY: Dict[str, Decimal] = {
    "direct_coupling": Decimal("0.99"),
    "v_belt": Decimal("0.93"),
    "synchronous_belt": Decimal("0.98"),
    "gear_single": Decimal("0.96"),
    "gear_multi": Decimal("0.92"),
    "chain": Decimal("0.95"),
}

# ---------------------------------------------------------------------------
# Constants -- Pump Benchmarks (Hydraulic Institute)
# ---------------------------------------------------------------------------

# Best-practice pump efficiency (%) by rated power range (kW).
PUMP_EFFICIENCY_BENCHMARKS: Dict[str, Decimal] = {
    "1": Decimal("60.0"), "5": Decimal("68.0"), "10": Decimal("73.0"),
    "20": Decimal("78.0"), "50": Decimal("82.0"), "100": Decimal("85.0"),
    "200": Decimal("87.0"), "500": Decimal("89.0"),
}

# ---------------------------------------------------------------------------
# Constants -- Compressor Benchmarks (Compressed Air & Gas Institute)
# ---------------------------------------------------------------------------

# Best-practice specific power (kW per m3/min at 7 bar) by type.
COMPRESSOR_SPECIFIC_POWER_BENCHMARKS: Dict[str, Decimal] = {
    CompressorType.SCREW_FIXED.value: Decimal("6.5"),
    CompressorType.SCREW_VSD.value: Decimal("6.0"),
    CompressorType.RECIPROCATING.value: Decimal("5.5"),
    CompressorType.CENTRIFUGAL.value: Decimal("5.8"),
    CompressorType.SCROLL.value: Decimal("7.5"),
}

# Typical isentropic efficiency range by compressor type.
COMPRESSOR_ISENTROPIC_EFF: Dict[str, Decimal] = {
    CompressorType.SCREW_FIXED.value: Decimal("72.0"),
    CompressorType.SCREW_VSD.value: Decimal("75.0"),
    CompressorType.RECIPROCATING.value: Decimal("80.0"),
    CompressorType.CENTRIFUGAL.value: Decimal("78.0"),
    CompressorType.SCROLL.value: Decimal("65.0"),
}

# ---------------------------------------------------------------------------
# Constants -- Boiler Benchmarks
# ---------------------------------------------------------------------------

# Best-practice boiler efficiency (%) by fuel type (GCV basis).
BOILER_EFFICIENCY_BENCHMARKS: Dict[str, Dict[str, Decimal]] = {
    FuelType.NATURAL_GAS.value: {
        BoilerType.FIRE_TUBE.value: Decimal("84.0"),
        BoilerType.WATER_TUBE.value: Decimal("85.0"),
        BoilerType.CONDENSING.value: Decimal("95.0"),
    },
    FuelType.DIESEL.value: {
        BoilerType.FIRE_TUBE.value: Decimal("83.0"),
        BoilerType.WATER_TUBE.value: Decimal("84.0"),
    },
    FuelType.HEAVY_FUEL_OIL.value: {
        BoilerType.FIRE_TUBE.value: Decimal("82.0"),
        BoilerType.WATER_TUBE.value: Decimal("83.0"),
    },
    FuelType.COAL.value: {
        BoilerType.FIRE_TUBE.value: Decimal("78.0"),
        BoilerType.WATER_TUBE.value: Decimal("80.0"),
    },
    FuelType.BIOMASS_WOOD.value: {
        BoilerType.FIRE_TUBE.value: Decimal("75.0"),
        BoilerType.WATER_TUBE.value: Decimal("77.0"),
    },
    FuelType.LPG.value: {
        BoilerType.FIRE_TUBE.value: Decimal("84.0"),
        BoilerType.WATER_TUBE.value: Decimal("85.0"),
        BoilerType.CONDENSING.value: Decimal("94.0"),
    },
}

# Stack loss coefficients per fuel type: loss_pct = coeff * (stack_temp - ambient) / CO2_pct
# Siegert's formula coefficients.
STACK_LOSS_COEFFICIENTS: Dict[str, Decimal] = {
    FuelType.NATURAL_GAS.value: Decimal("0.037"),
    FuelType.DIESEL.value: Decimal("0.040"),
    FuelType.HEAVY_FUEL_OIL.value: Decimal("0.042"),
    FuelType.LPG.value: Decimal("0.038"),
    FuelType.COAL.value: Decimal("0.045"),
    FuelType.BIOMASS_WOOD.value: Decimal("0.050"),
}

# Theoretical CO2 percentage in flue gas by fuel type.
THEORETICAL_CO2_PCT: Dict[str, Decimal] = {
    FuelType.NATURAL_GAS.value: Decimal("11.7"),
    FuelType.DIESEL.value: Decimal("15.2"),
    FuelType.HEAVY_FUEL_OIL.value: Decimal("15.8"),
    FuelType.LPG.value: Decimal("13.8"),
    FuelType.COAL.value: Decimal("18.5"),
    FuelType.BIOMASS_WOOD.value: Decimal("20.0"),
}

# ---------------------------------------------------------------------------
# Constants -- HVAC COP Benchmarks
# ---------------------------------------------------------------------------

# Best-practice COP/EER by HVAC type and age category.
HVAC_COP_BENCHMARKS: Dict[str, Dict[str, Decimal]] = {
    HVACType.SPLIT_SYSTEM.value: {
        "new": Decimal("4.5"), "5yr": Decimal("3.8"), "10yr": Decimal("3.2"),
        "15yr": Decimal("2.8"), "20yr": Decimal("2.4"),
    },
    HVACType.CHILLER_AIR.value: {
        "new": Decimal("3.5"), "5yr": Decimal("3.1"), "10yr": Decimal("2.7"),
        "15yr": Decimal("2.3"), "20yr": Decimal("2.0"),
    },
    HVACType.CHILLER_WATER.value: {
        "new": Decimal("6.0"), "5yr": Decimal("5.5"), "10yr": Decimal("5.0"),
        "15yr": Decimal("4.5"), "20yr": Decimal("4.0"),
    },
    HVACType.VRF.value: {
        "new": Decimal("5.5"), "5yr": Decimal("4.8"), "10yr": Decimal("4.2"),
        "15yr": Decimal("3.6"), "20yr": Decimal("3.0"),
    },
    HVACType.HEAT_PUMP_AIR.value: {
        "new": Decimal("4.0"), "5yr": Decimal("3.5"), "10yr": Decimal("3.0"),
        "15yr": Decimal("2.6"), "20yr": Decimal("2.2"),
    },
    HVACType.HEAT_PUMP_GROUND.value: {
        "new": Decimal("5.0"), "5yr": Decimal("4.6"), "10yr": Decimal("4.2"),
        "15yr": Decimal("3.8"), "20yr": Decimal("3.4"),
    },
    HVACType.ROOFTOP.value: {
        "new": Decimal("3.8"), "5yr": Decimal("3.3"), "10yr": Decimal("2.9"),
        "15yr": Decimal("2.5"), "20yr": Decimal("2.1"),
    },
    HVACType.AHU.value: {
        "new": Decimal("3.0"), "5yr": Decimal("2.7"), "10yr": Decimal("2.4"),
        "15yr": Decimal("2.1"), "20yr": Decimal("1.8"),
    },
}

# Default energy price (EUR/kWh) for cost calculations.
DEFAULT_ENERGY_PRICE_EUR_KWH: Decimal = Decimal("0.15")

# Equipment degradation rate per year (fraction).
DEFAULT_DEGRADATION_RATE_PER_YEAR: Decimal = Decimal("0.005")

# Minimum degraded efficiency floor (fraction of original).
MIN_DEGRADATION_FLOOR: Decimal = Decimal("0.70")

# Furnace/kiln wall loss and opening loss coefficients.
FURNACE_WALL_LOSS_COEFFICIENT: Decimal = Decimal("0.003")  # per degree C above ambient
FURNACE_OPENING_LOSS_PER_M2: Decimal = Decimal("15.0")  # kW per m2 opening area


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class Equipment(BaseModel):
    """Base equipment data for any industrial equipment asset.

    Attributes:
        equipment_id: Unique equipment identifier.
        name: Equipment name / tag.
        equipment_type: Type classification.
        manufacturer: Equipment manufacturer.
        model: Model number.
        year_installed: Year of installation.
        rated_power_kw: Nameplate rated power (kW).
        operating_hours: Annual operating hours.
        load_factor_pct: Average load factor (%).
        location: Facility / area location.
        notes: Additional notes.
    """
    equipment_id: str = Field(default_factory=_new_uuid, description="Equipment ID")
    name: str = Field(default="", max_length=300, description="Equipment name")
    equipment_type: str = Field(
        default=EquipmentType.MOTOR.value,
        description="Equipment type classification"
    )
    manufacturer: str = Field(default="", max_length=200, description="Manufacturer")
    model: str = Field(default="", max_length=200, description="Model number")
    year_installed: int = Field(default=2020, ge=1950, le=2030, description="Year installed")
    rated_power_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Rated power (kW)"
    )
    operating_hours: int = Field(default=0, ge=0, le=8760, description="Annual operating hours")
    load_factor_pct: Decimal = Field(
        default=Decimal("75"), ge=0, le=Decimal("100"),
        description="Average load factor (%)"
    )
    location: str = Field(default="", max_length=200, description="Location")
    notes: str = Field(default="", description="Additional notes")

    @field_validator("equipment_type")
    @classmethod
    def validate_equipment_type(cls, v: str) -> str:
        valid = {t.value for t in EquipmentType}
        if v not in valid:
            raise ValueError(f"Unknown equipment type '{v}'. Must be one of: {sorted(valid)}")
        return v


class MotorData(BaseModel):
    """Motor-specific data for efficiency analysis.

    Attributes:
        efficiency_class: IEC 60034-30-1 efficiency class.
        rated_power_kw: Motor nameplate power (kW).
        poles: Number of poles (2, 4, 6, 8).
        voltage: Rated voltage (V).
        frequency: Supply frequency (50 or 60 Hz).
        actual_load_pct: Measured average load percentage.
        has_vsd: Whether motor has variable speed drive.
        transmission_type: Mechanical transmission type.
        driven_equipment_eff_pct: Driven equipment efficiency (%).
    """
    efficiency_class: str = Field(
        default=MotorEfficiencyClass.IE2.value,
        description="IEC motor efficiency class"
    )
    rated_power_kw: Decimal = Field(
        default=Decimal("11"), ge=Decimal("0.12"), le=Decimal("1000"),
        description="Motor rated power (kW)"
    )
    poles: int = Field(default=4, description="Number of poles")
    voltage: Decimal = Field(default=Decimal("400"), ge=0, description="Rated voltage (V)")
    frequency: int = Field(default=50, description="Supply frequency (Hz)")
    actual_load_pct: Decimal = Field(
        default=Decimal("75"), ge=0, le=Decimal("120"),
        description="Average load (%)"
    )
    has_vsd: bool = Field(default=False, description="Has variable speed drive")
    transmission_type: str = Field(
        default="direct_coupling", description="Transmission type"
    )
    driven_equipment_eff_pct: Decimal = Field(
        default=Decimal("85"), ge=0, le=Decimal("100"),
        description="Driven equipment efficiency (%)"
    )

    @field_validator("efficiency_class")
    @classmethod
    def validate_eff_class(cls, v: str) -> str:
        valid = {c.value for c in MotorEfficiencyClass}
        if v not in valid:
            raise ValueError(f"Unknown motor class '{v}'. Must be one of: {sorted(valid)}")
        return v


class PumpData(BaseModel):
    """Pump-specific data for efficiency analysis.

    Attributes:
        flow_m3h: Design / measured flow rate (m3/h).
        head_m: Design / measured head (m).
        pump_efficiency_pct: Measured pump efficiency (%).
        system_curve_coefficient: System curve coefficient (H = k * Q^2).
        operating_speed_pct: Current speed as percentage of full speed.
        impeller_diameter_mm: Current impeller diameter (mm).
        design_impeller_mm: Design impeller diameter (mm).
    """
    flow_m3h: Decimal = Field(
        default=Decimal("0"), ge=0, description="Flow rate (m3/h)"
    )
    head_m: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total head (m)"
    )
    pump_efficiency_pct: Decimal = Field(
        default=Decimal("65"), ge=0, le=Decimal("100"),
        description="Pump efficiency (%)"
    )
    system_curve_coefficient: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="System curve coefficient H = k * Q^2"
    )
    operating_speed_pct: Decimal = Field(
        default=Decimal("100"), ge=0, le=Decimal("100"),
        description="Operating speed (%)"
    )
    impeller_diameter_mm: Decimal = Field(
        default=Decimal("0"), ge=0, description="Current impeller diameter (mm)"
    )
    design_impeller_mm: Decimal = Field(
        default=Decimal("0"), ge=0, description="Design impeller diameter (mm)"
    )


class CompressorData(BaseModel):
    """Compressor-specific data for efficiency analysis.

    Attributes:
        compressor_type: Compressor technology type.
        fad_m3min: Free air delivery (m3/min).
        pressure_bar: Discharge pressure (bar gauge).
        specific_power: Measured specific power (kW per m3/min).
        load_pct: Average load percentage.
        has_vsd: Whether has variable speed drive.
        unload_power_pct: Unloaded power as percentage of full load.
    """
    compressor_type: str = Field(
        default=CompressorType.SCREW_FIXED.value,
        description="Compressor type"
    )
    fad_m3min: Decimal = Field(
        default=Decimal("0"), ge=0, description="Free air delivery (m3/min)"
    )
    pressure_bar: Decimal = Field(
        default=Decimal("7"), ge=0, le=Decimal("40"),
        description="Discharge pressure (bar gauge)"
    )
    specific_power: Decimal = Field(
        default=Decimal("0"), ge=0, description="Specific power (kW/m3/min)"
    )
    load_pct: Decimal = Field(
        default=Decimal("75"), ge=0, le=Decimal("100"),
        description="Average load (%)"
    )
    has_vsd: bool = Field(default=False, description="Has variable speed drive")
    unload_power_pct: Decimal = Field(
        default=Decimal("25"), ge=0, le=Decimal("100"),
        description="Unloaded power as % of full load"
    )

    @field_validator("compressor_type")
    @classmethod
    def validate_compressor_type(cls, v: str) -> str:
        valid = {t.value for t in CompressorType}
        if v not in valid:
            raise ValueError(f"Unknown compressor type '{v}'. Must be one of: {sorted(valid)}")
        return v


class BoilerData(BaseModel):
    """Boiler-specific data for efficiency analysis.

    Attributes:
        boiler_type: Boiler technology type.
        capacity_kw: Rated thermal capacity (kW).
        fuel_type: Primary fuel type.
        stack_temp_c: Measured flue gas temperature (C).
        ambient_temp_c: Ambient temperature (C).
        excess_air_pct: Measured excess air (%).
        blowdown_pct: Blowdown rate (% of steam flow).
        blowdown_heat_recovery: Whether blowdown heat recovery is installed.
        radiation_loss_pct: Estimated radiation/convection loss (%).
        steam_pressure_bar: Operating steam pressure (bar gauge).
        feedwater_temp_c: Feedwater temperature (C).
    """
    boiler_type: str = Field(
        default=BoilerType.FIRE_TUBE.value,
        description="Boiler type"
    )
    capacity_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Rated thermal capacity (kW)"
    )
    fuel_type: str = Field(
        default=FuelType.NATURAL_GAS.value,
        description="Fuel type"
    )
    stack_temp_c: Decimal = Field(
        default=Decimal("200"), ge=0, le=Decimal("600"),
        description="Stack temperature (C)"
    )
    ambient_temp_c: Decimal = Field(
        default=Decimal("20"), ge=Decimal("-40"), le=Decimal("60"),
        description="Ambient temperature (C)"
    )
    excess_air_pct: Decimal = Field(
        default=Decimal("15"), ge=0, le=Decimal("200"),
        description="Excess air (%)"
    )
    blowdown_pct: Decimal = Field(
        default=Decimal("5"), ge=0, le=Decimal("25"),
        description="Blowdown rate (%)"
    )
    blowdown_heat_recovery: bool = Field(
        default=False, description="Has blowdown heat recovery"
    )
    radiation_loss_pct: Decimal = Field(
        default=Decimal("1.5"), ge=0, le=Decimal("10"),
        description="Radiation loss (%)"
    )
    steam_pressure_bar: Decimal = Field(
        default=Decimal("10"), ge=0, le=Decimal("100"),
        description="Steam pressure (bar gauge)"
    )
    feedwater_temp_c: Decimal = Field(
        default=Decimal("80"), ge=0, le=Decimal("200"),
        description="Feedwater temperature (C)"
    )

    @field_validator("boiler_type")
    @classmethod
    def validate_boiler_type(cls, v: str) -> str:
        valid = {t.value for t in BoilerType}
        if v not in valid:
            raise ValueError(f"Unknown boiler type '{v}'. Must be one of: {sorted(valid)}")
        return v

    @field_validator("fuel_type")
    @classmethod
    def validate_fuel_type(cls, v: str) -> str:
        valid = {t.value for t in FuelType}
        if v not in valid:
            raise ValueError(f"Unknown fuel type '{v}'. Must be one of: {sorted(valid)}")
        return v


class HVACData(BaseModel):
    """HVAC-specific data for efficiency analysis.

    Attributes:
        hvac_type: HVAC system type.
        cooling_capacity_kw: Nominal cooling capacity (kW).
        heating_capacity_kw: Nominal heating capacity (kW).
        cop: Measured or rated COP.
        eer: Measured or rated EER (BTU/Wh).
        seer: Seasonal EER if available.
        scop: Seasonal COP if available.
        refrigerant: Refrigerant type (R410A, R32, R134a, etc.).
        age_years: Equipment age in years.
    """
    hvac_type: str = Field(
        default=HVACType.SPLIT_SYSTEM.value,
        description="HVAC system type"
    )
    cooling_capacity_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Cooling capacity (kW)"
    )
    heating_capacity_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Heating capacity (kW)"
    )
    cop: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("10"),
        description="Coefficient of Performance"
    )
    eer: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("30"),
        description="Energy Efficiency Ratio (BTU/Wh)"
    )
    seer: Decimal = Field(
        default=Decimal("0"), ge=0, description="Seasonal EER"
    )
    scop: Decimal = Field(
        default=Decimal("0"), ge=0, description="Seasonal COP"
    )
    refrigerant: str = Field(default="R410A", max_length=20, description="Refrigerant type")
    age_years: int = Field(default=0, ge=0, le=50, description="Equipment age (years)")

    @field_validator("hvac_type")
    @classmethod
    def validate_hvac_type(cls, v: str) -> str:
        valid = {t.value for t in HVACType}
        if v not in valid:
            raise ValueError(f"Unknown HVAC type '{v}'. Must be one of: {sorted(valid)}")
        return v


class EquipmentEfficiencyInput(BaseModel):
    """Complete input for equipment efficiency analysis.

    Attributes:
        facility_id: Facility identifier.
        facility_name: Facility name.
        equipment: Base equipment data.
        motor_data: Motor-specific data (if motor or motor-driven).
        pump_data: Pump-specific data (if pump).
        compressor_data: Compressor-specific data (if compressor).
        boiler_data: Boiler-specific data (if boiler).
        hvac_data: HVAC-specific data (if HVAC).
        energy_price_eur_kwh: Energy price for cost calculations.
        include_upgrade_analysis: Whether to calculate upgrade ROI.
    """
    facility_id: str = Field(default="", description="Facility ID")
    facility_name: str = Field(default="", max_length=300, description="Facility name")
    equipment: Equipment = Field(default_factory=Equipment, description="Equipment data")
    motor_data: Optional[MotorData] = Field(default=None, description="Motor data")
    pump_data: Optional[PumpData] = Field(default=None, description="Pump data")
    compressor_data: Optional[CompressorData] = Field(default=None, description="Compressor data")
    boiler_data: Optional[BoilerData] = Field(default=None, description="Boiler data")
    hvac_data: Optional[HVACData] = Field(default=None, description="HVAC data")
    energy_price_eur_kwh: Decimal = Field(
        default=DEFAULT_ENERGY_PRICE_EUR_KWH, ge=0,
        description="Energy price (EUR/kWh)"
    )
    include_upgrade_analysis: bool = Field(
        default=True, description="Include upgrade ROI analysis"
    )


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class UpgradeOption(BaseModel):
    """An equipment upgrade recommendation.

    Attributes:
        upgrade_id: Unique upgrade identifier.
        description: Description of the upgrade.
        estimated_savings_kwh: Annual energy savings (kWh).
        estimated_savings_eur: Annual cost savings (EUR).
        estimated_cost_eur: Implementation cost (EUR).
        simple_payback_years: Simple payback period (years).
        new_efficiency_pct: Expected efficiency after upgrade.
        co2_reduction_tco2e: Annual CO2 reduction (tCO2e).
    """
    upgrade_id: str = Field(default_factory=_new_uuid)
    description: str = Field(default="")
    estimated_savings_kwh: Decimal = Field(default=Decimal("0"))
    estimated_savings_eur: Decimal = Field(default=Decimal("0"))
    estimated_cost_eur: Decimal = Field(default=Decimal("0"))
    simple_payback_years: Decimal = Field(default=Decimal("0"))
    new_efficiency_pct: Decimal = Field(default=Decimal("0"))
    co2_reduction_tco2e: Decimal = Field(default=Decimal("0"))


class EquipmentEfficiencyResult(BaseModel):
    """Complete equipment efficiency analysis result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        equipment_id: Equipment identifier.
        equipment_name: Equipment name.
        equipment_type: Equipment type.
        current_efficiency_pct: Current equipment efficiency (%).
        best_practice_efficiency_pct: Best-practice benchmark efficiency (%).
        efficiency_gap_pct: Gap between current and best practice (%).
        degraded_efficiency_pct: Efficiency accounting for degradation (%).
        annual_energy_consumption_kwh: Current annual energy consumption (kWh).
        annual_energy_waste_kwh: Annual wasted energy vs best practice (kWh).
        annual_energy_cost_eur: Current annual energy cost (EUR).
        annual_waste_cost_eur: Annual wasted energy cost (EUR).
        upgrade_options: Available upgrade options with ROI.
        best_upgrade_savings_kwh: Best single upgrade savings (kWh).
        best_upgrade_cost_eur: Best single upgrade cost (EUR).
        best_upgrade_payback_years: Best single upgrade payback (years).
        efficiency_rating: Rating (excellent/good/fair/poor/critical).
        recommendations: Action recommendations.
        calculation_details: Step-by-step calculation details.
        warnings: Non-blocking warnings.
        errors: Blocking errors.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=_utcnow)
    equipment_id: str = Field(default="")
    equipment_name: str = Field(default="")
    equipment_type: str = Field(default="")
    current_efficiency_pct: Decimal = Field(default=Decimal("0"))
    best_practice_efficiency_pct: Decimal = Field(default=Decimal("0"))
    efficiency_gap_pct: Decimal = Field(default=Decimal("0"))
    degraded_efficiency_pct: Decimal = Field(default=Decimal("0"))
    annual_energy_consumption_kwh: Decimal = Field(default=Decimal("0"))
    annual_energy_waste_kwh: Decimal = Field(default=Decimal("0"))
    annual_energy_cost_eur: Decimal = Field(default=Decimal("0"))
    annual_waste_cost_eur: Decimal = Field(default=Decimal("0"))
    upgrade_options: List[UpgradeOption] = Field(default_factory=list)
    best_upgrade_savings_kwh: Decimal = Field(default=Decimal("0"))
    best_upgrade_cost_eur: Decimal = Field(default=Decimal("0"))
    best_upgrade_payback_years: Decimal = Field(default=Decimal("0"))
    efficiency_rating: str = Field(default="fair")
    recommendations: List[str] = Field(default_factory=list)
    calculation_details: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class EquipmentEfficiencyEngine:
    """Industrial equipment efficiency analysis engine.

    Calculates equipment-level energy efficiency for motors, pumps,
    compressors, boilers, HVAC systems, furnaces, and steam turbines.
    Identifies efficiency gaps against best-practice benchmarks and
    calculates replacement ROI for each equipment class.

    Usage::

        engine = EquipmentEfficiencyEngine()
        result = engine.analyze(input_data)
        print(f"Efficiency: {result.current_efficiency_pct}%")
        print(f"Gap: {result.efficiency_gap_pct}%")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise EquipmentEfficiencyEngine.

        Args:
            config: Optional overrides. Supported keys:
                - energy_price_eur_kwh (Decimal): default energy price
                - degradation_rate (Decimal): annual degradation rate
                - co2_factor_kg_per_kwh (Decimal): grid CO2 factor
        """
        self.config = config or {}
        self._energy_price = _decimal(
            self.config.get("energy_price_eur_kwh", DEFAULT_ENERGY_PRICE_EUR_KWH)
        )
        self._degradation_rate = _decimal(
            self.config.get("degradation_rate", DEFAULT_DEGRADATION_RATE_PER_YEAR)
        )
        self._co2_factor = _decimal(
            self.config.get("co2_factor_kg_per_kwh", Decimal("0.4"))
        )
        logger.info(
            "EquipmentEfficiencyEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def analyze(
        self, data: EquipmentEfficiencyInput,
    ) -> EquipmentEfficiencyResult:
        """Perform complete equipment efficiency analysis.

        Routes to the appropriate analysis method based on equipment type,
        calculates efficiency gap, energy waste, degradation, and upgrade
        options with ROI.

        Args:
            data: Validated equipment efficiency input.

        Returns:
            EquipmentEfficiencyResult with complete analysis.
        """
        t0 = time.perf_counter()
        eq = data.equipment
        logger.info(
            "Equipment efficiency analysis: id=%s, type=%s, power=%.1f kW",
            eq.equipment_id, eq.equipment_type, float(eq.rated_power_kw),
        )

        warnings: List[str] = []
        errors: List[str] = []
        details: Dict[str, Any] = {}
        recommendations: List[str] = []

        price = data.energy_price_eur_kwh if data.energy_price_eur_kwh > Decimal("0") else self._energy_price
        current_eff = Decimal("0")
        best_eff = Decimal("0")

        # Route to equipment-specific analysis
        etype = eq.equipment_type
        if etype == EquipmentType.MOTOR.value and data.motor_data:
            current_eff, best_eff, details = self._analyze_motor(
                eq, data.motor_data, warnings
            )
        elif etype == EquipmentType.PUMP.value:
            current_eff, best_eff, details = self._analyze_pump(
                eq, data.motor_data, data.pump_data, warnings
            )
        elif etype == EquipmentType.COMPRESSOR.value and data.compressor_data:
            current_eff, best_eff, details = self._analyze_compressor(
                eq, data.compressor_data, warnings
            )
        elif etype == EquipmentType.BOILER.value and data.boiler_data:
            current_eff, best_eff, details = self._analyze_boiler(
                eq, data.boiler_data, warnings
            )
        elif etype == EquipmentType.HVAC.value and data.hvac_data:
            current_eff, best_eff, details = self._analyze_hvac(
                eq, data.hvac_data, warnings
            )
        elif etype == EquipmentType.FURNACE.value:
            current_eff, best_eff, details = self._analyze_furnace(
                eq, data.boiler_data, warnings
            )
        elif etype == EquipmentType.STEAM_TURBINE.value:
            current_eff, best_eff, details = self._analyze_steam_turbine(
                eq, warnings
            )
        else:
            errors.append(
                f"Equipment type '{etype}' analysis requires matching data input. "
                f"Provide motor_data, pump_data, compressor_data, boiler_data, or hvac_data."
            )
            current_eff = eq.load_factor_pct
            best_eff = Decimal("95")

        # Calculate efficiency gap
        gap = best_eff - current_eff

        # Apply degradation
        current_year = _utcnow().year
        age = max(0, current_year - eq.year_installed)
        degraded = self._apply_degradation(current_eff, age)

        # Annual energy consumption
        annual_kwh = eq.rated_power_kw * _decimal(eq.operating_hours) * (eq.load_factor_pct / Decimal("100"))

        # Energy waste vs best practice
        if best_eff > Decimal("0") and current_eff > Decimal("0"):
            waste_ratio = Decimal("1") - _safe_divide(current_eff, best_eff)
            waste_kwh = annual_kwh * max(waste_ratio, Decimal("0"))
        else:
            waste_kwh = Decimal("0")

        annual_cost = annual_kwh * price
        waste_cost = waste_kwh * price

        # Efficiency rating
        rating = self._rate_efficiency(current_eff, best_eff)

        # Upgrade options
        upgrade_options: List[UpgradeOption] = []
        if data.include_upgrade_analysis:
            upgrade_options = self._generate_upgrade_options(
                eq, data, current_eff, best_eff, annual_kwh, price,
                recommendations, warnings
            )

        best_savings = Decimal("0")
        best_cost = Decimal("0")
        best_payback = Decimal("0")
        if upgrade_options:
            best_opt = max(upgrade_options, key=lambda o: o.estimated_savings_kwh)
            best_savings = best_opt.estimated_savings_kwh
            best_cost = best_opt.estimated_cost_eur
            best_payback = best_opt.simple_payback_years

        # General recommendations
        if gap > Decimal("10"):
            recommendations.append(
                f"Equipment efficiency gap of {_round_val(gap, 1)}% exceeds 10%. "
                f"Priority upgrade recommended."
            )
        if age > 15:
            recommendations.append(
                f"Equipment age ({age} years) exceeds 15 years. "
                f"Consider life-cycle cost analysis for replacement."
            )
        if eq.load_factor_pct < Decimal("40"):
            recommendations.append(
                f"Low load factor ({eq.load_factor_pct}%). "
                f"Consider right-sizing or consolidation with other loads."
            )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = EquipmentEfficiencyResult(
            equipment_id=eq.equipment_id,
            equipment_name=eq.name,
            equipment_type=eq.equipment_type,
            current_efficiency_pct=_round_val(current_eff, 2),
            best_practice_efficiency_pct=_round_val(best_eff, 2),
            efficiency_gap_pct=_round_val(gap, 2),
            degraded_efficiency_pct=_round_val(degraded, 2),
            annual_energy_consumption_kwh=_round_val(annual_kwh, 2),
            annual_energy_waste_kwh=_round_val(waste_kwh, 2),
            annual_energy_cost_eur=_round_val(annual_cost, 2),
            annual_waste_cost_eur=_round_val(waste_cost, 2),
            upgrade_options=upgrade_options,
            best_upgrade_savings_kwh=_round_val(best_savings, 2),
            best_upgrade_cost_eur=_round_val(best_cost, 2),
            best_upgrade_payback_years=_round_val(best_payback, 2),
            efficiency_rating=rating,
            recommendations=recommendations,
            calculation_details=details,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Equipment analysis complete: id=%s, eff=%.1f%%, gap=%.1f%%, "
            "waste=%.0f kWh, rating=%s, hash=%s",
            eq.equipment_id, float(current_eff), float(gap),
            float(waste_kwh), rating, result.provenance_hash[:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # Motor Analysis                                                       #
    # ------------------------------------------------------------------ #

    def _analyze_motor(
        self, eq: Equipment, motor: MotorData, warnings: List[str],
    ) -> Tuple[Decimal, Decimal, Dict[str, Any]]:
        """Analyze motor system efficiency.

        Calculates system efficiency as:
            system_eff = motor_eff * vsd_eff * transmission_eff * driven_eff

        Args:
            eq: Base equipment data.
            motor: Motor-specific data.
            warnings: Warning list.

        Returns:
            Tuple of (current_efficiency, best_practice_efficiency, details).
        """
        # Look up motor efficiency from standards table
        motor_eff = self._lookup_motor_efficiency(
            motor.efficiency_class, motor.rated_power_kw
        )

        # Part-load motor efficiency adjustment
        # Motors below 50% load have reduced efficiency
        load = motor.actual_load_pct
        if load < Decimal("50") and load > Decimal("0"):
            load_penalty = (Decimal("50") - load) / Decimal("100") * Decimal("3")
            motor_eff = motor_eff - load_penalty
            warnings.append(
                f"Motor operating at {load}% load. Efficiency reduced by "
                f"{_round_val(load_penalty, 1)}% due to part-load operation."
            )

        # VSD efficiency
        vsd_eff = Decimal("1")
        if motor.has_vsd:
            load_key = str(int(_round_val(load / Decimal("25"), 0) * 25))
            if load_key in VSD_EFFICIENCY_BY_LOAD:
                vsd_eff = VSD_EFFICIENCY_BY_LOAD[load_key]
            else:
                vsd_eff = Decimal("0.96")
        else:
            vsd_eff = Decimal("1")

        # Transmission efficiency
        trans_eff = TRANSMISSION_EFFICIENCY.get(
            motor.transmission_type, Decimal("0.95")
        )

        # Driven equipment efficiency
        driven_eff = motor.driven_equipment_eff_pct / Decimal("100")

        # System efficiency
        system_eff = (motor_eff / Decimal("100")) * vsd_eff * trans_eff * driven_eff
        current_pct = system_eff * Decimal("100")

        # Best practice: IE5 motor + VSD + synchronous belt + best driven
        best_motor = self._lookup_motor_efficiency(
            MotorEfficiencyClass.IE5.value, motor.rated_power_kw
        )
        best_vsd = Decimal("0.97")
        best_trans = TRANSMISSION_EFFICIENCY["synchronous_belt"]
        best_driven = Decimal("0.92")
        best_pct = (best_motor / Decimal("100")) * best_vsd * best_trans * best_driven * Decimal("100")

        details = {
            "motor_efficiency_pct": str(_round_val(motor_eff, 2)),
            "vsd_efficiency": str(_round_val(vsd_eff, 4)),
            "transmission_efficiency": str(_round_val(trans_eff, 4)),
            "driven_equipment_efficiency": str(_round_val(driven_eff, 4)),
            "system_efficiency_pct": str(_round_val(current_pct, 2)),
            "motor_class": motor.efficiency_class,
            "has_vsd": motor.has_vsd,
            "transmission_type": motor.transmission_type,
            "actual_load_pct": str(motor.actual_load_pct),
        }

        return current_pct, best_pct, details

    def _lookup_motor_efficiency(
        self, eff_class: str, power_kw: Decimal,
    ) -> Decimal:
        """Look up motor efficiency from IEC 60034-30-1 tables.

        Finds the closest power rating at or below the specified power.
        Deterministic database lookup -- no LLM.

        Args:
            eff_class: IEC efficiency class (IE1-IE5).
            power_kw: Motor rated power (kW).

        Returns:
            Motor efficiency (%).
        """
        table = MOTOR_EFFICIENCY_STANDARDS.get(eff_class, {})
        if not table:
            return Decimal("85")

        # Find closest power rating at or below
        best_key = "0.75"
        best_diff = Decimal("999999")
        for key_str in table:
            key_val = _decimal(key_str)
            diff = power_kw - key_val
            if diff >= Decimal("0") and diff < best_diff:
                best_diff = diff
                best_key = key_str

        return table.get(best_key, Decimal("85"))

    # ------------------------------------------------------------------ #
    # Pump Analysis                                                        #
    # ------------------------------------------------------------------ #

    def _analyze_pump(
        self, eq: Equipment, motor: Optional[MotorData],
        pump: Optional[PumpData], warnings: List[str],
    ) -> Tuple[Decimal, Decimal, Dict[str, Any]]:
        """Analyze pump system efficiency using affinity laws.

        Affinity laws (ISO 9906):
            flow ~ speed,  head ~ speed^2,  power ~ speed^3
            savings = 1 - (reduced_speed / full_speed)^3

        Args:
            eq: Base equipment data.
            motor: Optional motor data for motor-driven pump.
            pump: Pump-specific data.
            warnings: Warning list.

        Returns:
            Tuple of (current_efficiency, best_practice_efficiency, details).
        """
        if pump is None:
            return Decimal("65"), Decimal("85"), {"note": "No pump data provided"}

        current_eff = pump.pump_efficiency_pct
        best_eff = self._lookup_pump_benchmark(eq.rated_power_kw)

        # If motor data available, include motor in system efficiency
        if motor:
            motor_eff = self._lookup_motor_efficiency(
                motor.efficiency_class, motor.rated_power_kw
            ) / Decimal("100")
            current_eff = pump.pump_efficiency_pct * motor_eff
            best_eff = best_eff * Decimal("0.96")  # IE5 motor factor

        # Affinity law savings if speed reduced
        speed_pct = pump.operating_speed_pct
        affinity_savings_pct = Decimal("0")
        if speed_pct < Decimal("100") and speed_pct > Decimal("0"):
            speed_ratio = speed_pct / Decimal("100")
            power_ratio = speed_ratio * speed_ratio * speed_ratio
            affinity_savings_pct = (Decimal("1") - power_ratio) * Decimal("100")

        details = {
            "pump_efficiency_pct": str(_round_val(pump.pump_efficiency_pct, 2)),
            "best_practice_pump_eff_pct": str(_round_val(best_eff, 2)),
            "operating_speed_pct": str(speed_pct),
            "affinity_law_savings_pct": str(_round_val(affinity_savings_pct, 2)),
            "flow_m3h": str(pump.flow_m3h),
            "head_m": str(pump.head_m),
        }

        return current_eff, best_eff, details

    def _lookup_pump_benchmark(self, power_kw: Decimal) -> Decimal:
        """Look up best-practice pump efficiency benchmark.

        Args:
            power_kw: Pump rated power (kW).

        Returns:
            Best-practice efficiency (%).
        """
        best_key = "1"
        best_diff = Decimal("999999")
        for key_str in PUMP_EFFICIENCY_BENCHMARKS:
            key_val = _decimal(key_str)
            diff = power_kw - key_val
            if diff >= Decimal("0") and diff < best_diff:
                best_diff = diff
                best_key = key_str

        return PUMP_EFFICIENCY_BENCHMARKS.get(best_key, Decimal("75"))

    # ------------------------------------------------------------------ #
    # Compressor Analysis                                                  #
    # ------------------------------------------------------------------ #

    def _analyze_compressor(
        self, eq: Equipment, comp: CompressorData, warnings: List[str],
    ) -> Tuple[Decimal, Decimal, Dict[str, Any]]:
        """Analyze compressor efficiency via specific power.

        Specific power = input_kW / FAD_m3min (at reference conditions).
        Lower is better.

        Args:
            eq: Base equipment data.
            comp: Compressor-specific data.
            warnings: Warning list.

        Returns:
            Tuple of (current_efficiency, best_practice_efficiency, details).
        """
        benchmark_sp = COMPRESSOR_SPECIFIC_POWER_BENCHMARKS.get(
            comp.compressor_type, Decimal("6.5")
        )
        actual_sp = comp.specific_power

        if actual_sp <= Decimal("0") and comp.fad_m3min > Decimal("0"):
            actual_sp = _safe_divide(eq.rated_power_kw, comp.fad_m3min)
        elif actual_sp <= Decimal("0"):
            actual_sp = benchmark_sp * Decimal("1.3")

        # Normalize to 7 bar reference
        if comp.pressure_bar != Decimal("7") and comp.pressure_bar > Decimal("0"):
            pressure_ratio = Decimal("7") / comp.pressure_bar
            actual_sp = actual_sp * pressure_ratio

        # Convert specific power to efficiency-like metric
        # Current efficiency = benchmark / actual * 100 (capped at 100)
        if actual_sp > Decimal("0"):
            current_eff = min(
                _safe_divide(benchmark_sp, actual_sp) * Decimal("100"),
                Decimal("100")
            )
        else:
            current_eff = Decimal("50")

        best_eff = Decimal("100")  # benchmark defines 100%

        # Isentropic efficiency
        isentropic = COMPRESSOR_ISENTROPIC_EFF.get(
            comp.compressor_type, Decimal("72")
        )

        # Part-load penalty for fixed-speed compressors
        if not comp.has_vsd and comp.load_pct < Decimal("70"):
            unload_waste = (Decimal("100") - comp.load_pct) / Decimal("100") * comp.unload_power_pct
            warnings.append(
                f"Fixed-speed compressor at {comp.load_pct}% load. "
                f"Estimated {_round_val(unload_waste, 1)}% power wasted during unloading."
            )

        details = {
            "actual_specific_power_kw_m3min": str(_round_val(actual_sp, 2)),
            "benchmark_specific_power_kw_m3min": str(_round_val(benchmark_sp, 2)),
            "specific_power_gap": str(_round_val(actual_sp - benchmark_sp, 2)),
            "isentropic_efficiency_pct": str(isentropic),
            "compressor_type": comp.compressor_type,
            "fad_m3min": str(comp.fad_m3min),
            "pressure_bar": str(comp.pressure_bar),
            "has_vsd": comp.has_vsd,
            "load_pct": str(comp.load_pct),
        }

        return current_eff, best_eff, details

    # ------------------------------------------------------------------ #
    # Boiler Analysis                                                      #
    # ------------------------------------------------------------------ #

    def _analyze_boiler(
        self, eq: Equipment, boiler: BoilerData, warnings: List[str],
    ) -> Tuple[Decimal, Decimal, Dict[str, Any]]:
        """Analyze boiler efficiency using indirect (heat loss) method.

        Indirect method per EN 12953 / ASME PTC 4:
            efficiency = 100 - (stack_loss + blowdown_loss + radiation_loss
                                + unburned_loss + moisture_loss)

        Args:
            eq: Base equipment data.
            boiler: Boiler-specific data.
            warnings: Warning list.

        Returns:
            Tuple of (current_efficiency, best_practice_efficiency, details).
        """
        # Stack loss (Siegert's formula)
        coeff = STACK_LOSS_COEFFICIENTS.get(boiler.fuel_type, Decimal("0.040"))
        theo_co2 = THEORETICAL_CO2_PCT.get(boiler.fuel_type, Decimal("12"))
        # Actual CO2% adjusted for excess air
        actual_co2 = _safe_divide(
            theo_co2,
            Decimal("1") + boiler.excess_air_pct / Decimal("100")
        )
        temp_diff = boiler.stack_temp_c - boiler.ambient_temp_c
        stack_loss = coeff * temp_diff / max(actual_co2, Decimal("1")) * Decimal("100")
        stack_loss = max(stack_loss, Decimal("0"))

        # Blowdown loss
        blowdown_loss = boiler.blowdown_pct
        if boiler.blowdown_heat_recovery:
            blowdown_loss = blowdown_loss * Decimal("0.2")

        # Radiation loss (surface loss)
        radiation_loss = boiler.radiation_loss_pct

        # Unburned carbon loss (typical for solid fuels)
        unburned_loss = Decimal("0")
        if boiler.fuel_type in (FuelType.COAL.value, FuelType.BIOMASS_WOOD.value):
            unburned_loss = Decimal("1.5")

        # Moisture loss (hydrogen in fuel)
        moisture_loss = Decimal("0")
        if boiler.fuel_type == FuelType.NATURAL_GAS.value:
            moisture_loss = Decimal("10.5")  # High hydrogen content
        elif boiler.fuel_type in (FuelType.DIESEL.value, FuelType.HEAVY_FUEL_OIL.value):
            moisture_loss = Decimal("6.5")
        elif boiler.fuel_type == FuelType.BIOMASS_WOOD.value:
            moisture_loss = Decimal("8.0")
        elif boiler.fuel_type == FuelType.COAL.value:
            moisture_loss = Decimal("4.0")
        elif boiler.fuel_type == FuelType.LPG.value:
            moisture_loss = Decimal("9.0")

        total_losses = stack_loss + blowdown_loss + radiation_loss + unburned_loss + moisture_loss
        current_eff = max(Decimal("100") - total_losses, Decimal("20"))

        # Best practice benchmark
        fuel_benchmarks = BOILER_EFFICIENCY_BENCHMARKS.get(boiler.fuel_type, {})
        best_eff = fuel_benchmarks.get(boiler.boiler_type, Decimal("85"))

        # Warnings for poor operation
        if boiler.excess_air_pct > Decimal("30"):
            warnings.append(
                f"Excess air at {boiler.excess_air_pct}% is above optimal range "
                f"(10-20% for gas, 15-25% for oil). Reduce to lower stack losses."
            )
        if boiler.stack_temp_c > Decimal("250"):
            warnings.append(
                f"Stack temperature {boiler.stack_temp_c}C exceeds 250C. "
                f"Consider economizer to recover flue gas heat."
            )
        if boiler.blowdown_pct > Decimal("8") and not boiler.blowdown_heat_recovery:
            warnings.append(
                f"Blowdown rate {boiler.blowdown_pct}% without heat recovery. "
                f"Install flash vessel and heat exchanger for blowdown recovery."
            )

        details = {
            "stack_loss_pct": str(_round_val(stack_loss, 2)),
            "blowdown_loss_pct": str(_round_val(blowdown_loss, 2)),
            "radiation_loss_pct": str(_round_val(radiation_loss, 2)),
            "unburned_loss_pct": str(_round_val(unburned_loss, 2)),
            "moisture_loss_pct": str(_round_val(moisture_loss, 2)),
            "total_losses_pct": str(_round_val(total_losses, 2)),
            "actual_co2_pct": str(_round_val(actual_co2, 2)),
            "fuel_type": boiler.fuel_type,
            "boiler_type": boiler.boiler_type,
            "stack_temp_c": str(boiler.stack_temp_c),
            "excess_air_pct": str(boiler.excess_air_pct),
            "blowdown_pct": str(boiler.blowdown_pct),
        }

        return current_eff, best_eff, details

    # ------------------------------------------------------------------ #
    # HVAC Analysis                                                        #
    # ------------------------------------------------------------------ #

    def _analyze_hvac(
        self, eq: Equipment, hvac: HVACData, warnings: List[str],
    ) -> Tuple[Decimal, Decimal, Dict[str, Any]]:
        """Analyze HVAC system efficiency via COP/EER benchmarking.

        COP = heating_output / electrical_input
        EER = cooling_output_BTU / electrical_input_Wh
        SEER = seasonal total cooling / seasonal total input

        Args:
            eq: Base equipment data.
            hvac: HVAC-specific data.
            warnings: Warning list.

        Returns:
            Tuple of (current_efficiency, best_practice_efficiency, details).
        """
        # Determine primary metric
        measured_cop = hvac.cop
        if measured_cop <= Decimal("0") and hvac.eer > Decimal("0"):
            # EER to COP conversion: COP = EER / 3.412
            measured_cop = hvac.eer / Decimal("3.412")
        if measured_cop <= Decimal("0") and hvac.seer > Decimal("0"):
            measured_cop = hvac.seer / Decimal("3.412")

        # Best practice COP by type and age
        type_benchmarks = HVAC_COP_BENCHMARKS.get(hvac.hvac_type, {})
        best_cop = type_benchmarks.get("new", Decimal("4.0"))

        # Age-appropriate benchmark
        age = hvac.age_years
        if age <= 2:
            age_key = "new"
        elif age <= 7:
            age_key = "5yr"
        elif age <= 12:
            age_key = "10yr"
        elif age <= 17:
            age_key = "15yr"
        else:
            age_key = "20yr"
        age_benchmark = type_benchmarks.get(age_key, best_cop)

        # Convert COP to efficiency percentage for gap analysis
        # Normalize: current COP / best COP * 100
        if best_cop > Decimal("0"):
            current_eff = min(
                _safe_divide(measured_cop, best_cop) * Decimal("100"),
                Decimal("100")
            )
        else:
            current_eff = Decimal("50")

        best_eff = Decimal("100")

        # Warnings
        if measured_cop < age_benchmark * Decimal("0.7"):
            warnings.append(
                f"COP of {_round_val(measured_cop, 2)} is more than 30% below "
                f"age-adjusted benchmark of {_round_val(age_benchmark, 2)}. "
                f"Maintenance or replacement recommended."
            )
        if age > 15:
            warnings.append(
                f"HVAC unit age ({age} years) exceeds typical economic life (15 years). "
                f"New equipment with higher COP available."
            )

        details = {
            "measured_cop": str(_round_val(measured_cop, 3)),
            "best_practice_cop": str(_round_val(best_cop, 3)),
            "age_adjusted_benchmark_cop": str(_round_val(age_benchmark, 3)),
            "hvac_type": hvac.hvac_type,
            "cooling_capacity_kw": str(hvac.cooling_capacity_kw),
            "heating_capacity_kw": str(hvac.heating_capacity_kw),
            "eer": str(hvac.eer),
            "seer": str(hvac.seer),
            "scop": str(hvac.scop),
            "refrigerant": hvac.refrigerant,
            "age_years": hvac.age_years,
        }

        return current_eff, best_eff, details

    # ------------------------------------------------------------------ #
    # Furnace/Kiln Analysis                                                #
    # ------------------------------------------------------------------ #

    def _analyze_furnace(
        self, eq: Equipment, boiler: Optional[BoilerData],
        warnings: List[str],
    ) -> Tuple[Decimal, Decimal, Dict[str, Any]]:
        """Analyze furnace/kiln efficiency.

        Considers combustion efficiency, wall losses, and opening losses.

        Args:
            eq: Base equipment data.
            boiler: Boiler/fuel data reused for furnace combustion analysis.
            warnings: Warning list.

        Returns:
            Tuple of (current_efficiency, best_practice_efficiency, details).
        """
        if boiler is None:
            return Decimal("55"), Decimal("75"), {"note": "No combustion data provided"}

        # Combustion efficiency (simplified from boiler analysis)
        coeff = STACK_LOSS_COEFFICIENTS.get(boiler.fuel_type, Decimal("0.040"))
        theo_co2 = THEORETICAL_CO2_PCT.get(boiler.fuel_type, Decimal("12"))
        actual_co2 = _safe_divide(
            theo_co2,
            Decimal("1") + boiler.excess_air_pct / Decimal("100")
        )
        temp_diff = boiler.stack_temp_c - boiler.ambient_temp_c
        stack_loss = coeff * temp_diff / max(actual_co2, Decimal("1")) * Decimal("100")

        # Wall losses (higher for furnaces due to higher temperatures)
        wall_loss = FURNACE_WALL_LOSS_COEFFICIENT * temp_diff
        wall_loss = min(wall_loss, Decimal("15"))

        # Radiation loss
        radiation_loss = boiler.radiation_loss_pct * Decimal("2")  # Higher for furnaces

        total_losses = stack_loss + wall_loss + radiation_loss
        current_eff = max(Decimal("100") - total_losses, Decimal("20"))

        # Best practice for industrial furnaces
        best_eff = Decimal("75")
        if boiler.fuel_type == FuelType.NATURAL_GAS.value:
            best_eff = Decimal("80")

        if boiler.stack_temp_c > Decimal("400"):
            warnings.append(
                f"Furnace exhaust at {boiler.stack_temp_c}C. "
                f"Regenerative or recuperative heat recovery should be evaluated."
            )

        details = {
            "stack_loss_pct": str(_round_val(stack_loss, 2)),
            "wall_loss_pct": str(_round_val(wall_loss, 2)),
            "radiation_loss_pct": str(_round_val(radiation_loss, 2)),
            "total_losses_pct": str(_round_val(total_losses, 2)),
            "fuel_type": boiler.fuel_type,
            "stack_temp_c": str(boiler.stack_temp_c),
        }

        return current_eff, best_eff, details

    # ------------------------------------------------------------------ #
    # Steam Turbine Analysis                                               #
    # ------------------------------------------------------------------ #

    def _analyze_steam_turbine(
        self, eq: Equipment, warnings: List[str],
    ) -> Tuple[Decimal, Decimal, Dict[str, Any]]:
        """Analyze steam turbine efficiency.

        Uses rated power and load factor to estimate isentropic efficiency.

        Args:
            eq: Base equipment data.
            warnings: Warning list.

        Returns:
            Tuple of (current_efficiency, best_practice_efficiency, details).
        """
        # Typical steam turbine efficiency ranges
        if eq.rated_power_kw > Decimal("10000"):
            best_isentropic = Decimal("85")
        elif eq.rated_power_kw > Decimal("1000"):
            best_isentropic = Decimal("78")
        else:
            best_isentropic = Decimal("70")

        # Estimate current from age degradation
        current_year = _utcnow().year
        age = max(0, current_year - eq.year_installed)
        degradation = min(
            _decimal(age) * Decimal("0.003"),  # 0.3% per year
            Decimal("0.15")
        )
        current_eff = best_isentropic * (Decimal("1") - degradation)

        # Part-load penalty
        if eq.load_factor_pct < Decimal("50"):
            part_load_penalty = (Decimal("50") - eq.load_factor_pct) / Decimal("100") * Decimal("5")
            current_eff = current_eff - part_load_penalty

        current_eff = max(current_eff, Decimal("40"))

        details = {
            "estimated_isentropic_eff_pct": str(_round_val(current_eff, 2)),
            "best_practice_isentropic_pct": str(_round_val(best_isentropic, 2)),
            "age_years": age,
            "degradation_factor": str(_round_val(degradation, 4)),
            "load_factor_pct": str(eq.load_factor_pct),
        }

        return current_eff, best_isentropic, details

    # ------------------------------------------------------------------ #
    # Degradation                                                          #
    # ------------------------------------------------------------------ #

    def _apply_degradation(self, base_eff: Decimal, age_years: int) -> Decimal:
        """Apply equipment degradation over time.

        degraded_eff = base_eff * (1 - degradation_rate * age)
        Floor at 70% of original efficiency.

        Args:
            base_eff: Base efficiency (%).
            age_years: Equipment age in years.

        Returns:
            Degraded efficiency (%).
        """
        factor = Decimal("1") - self._degradation_rate * _decimal(age_years)
        floor = MIN_DEGRADATION_FLOOR
        factor = max(factor, floor)
        return base_eff * factor

    # ------------------------------------------------------------------ #
    # Rating                                                               #
    # ------------------------------------------------------------------ #

    def _rate_efficiency(self, current: Decimal, best: Decimal) -> str:
        """Rate equipment efficiency against benchmark.

        Args:
            current: Current efficiency (%).
            best: Best practice efficiency (%).

        Returns:
            Rating string: excellent/good/fair/poor/critical.
        """
        if best <= Decimal("0"):
            return "unknown"
        ratio = _safe_divide(current, best)
        if ratio >= Decimal("0.95"):
            return "excellent"
        elif ratio >= Decimal("0.85"):
            return "good"
        elif ratio >= Decimal("0.70"):
            return "fair"
        elif ratio >= Decimal("0.50"):
            return "poor"
        else:
            return "critical"

    # ------------------------------------------------------------------ #
    # Upgrade Options                                                      #
    # ------------------------------------------------------------------ #

    def _generate_upgrade_options(
        self,
        eq: Equipment,
        data: EquipmentEfficiencyInput,
        current_eff: Decimal,
        best_eff: Decimal,
        annual_kwh: Decimal,
        price: Decimal,
        recommendations: List[str],
        warnings: List[str],
    ) -> List[UpgradeOption]:
        """Generate equipment upgrade options with ROI.

        Args:
            eq: Equipment data.
            data: Full input data.
            current_eff: Current efficiency (%).
            best_eff: Best practice efficiency (%).
            annual_kwh: Annual energy consumption (kWh).
            price: Energy price (EUR/kWh).
            recommendations: Recommendation list.
            warnings: Warning list.

        Returns:
            List of UpgradeOption.
        """
        options: List[UpgradeOption] = []
        gap = best_eff - current_eff

        if gap <= Decimal("2"):
            return options  # Already near best practice

        # Option 1: Upgrade to best-practice efficiency
        savings_ratio = _safe_divide(gap, Decimal("100"))
        savings_kwh = annual_kwh * savings_ratio
        savings_eur = savings_kwh * price
        co2_saved = savings_kwh * self._co2_factor / Decimal("1000")

        # Estimate cost per kW by equipment type
        cost_per_kw: Dict[str, Decimal] = {
            EquipmentType.MOTOR.value: Decimal("150"),
            EquipmentType.PUMP.value: Decimal("250"),
            EquipmentType.COMPRESSOR.value: Decimal("400"),
            EquipmentType.BOILER.value: Decimal("200"),
            EquipmentType.HVAC.value: Decimal("350"),
            EquipmentType.FURNACE.value: Decimal("500"),
            EquipmentType.STEAM_TURBINE.value: Decimal("600"),
            EquipmentType.FAN.value: Decimal("120"),
            EquipmentType.COOLING_TOWER.value: Decimal("300"),
            EquipmentType.TRANSFORMER.value: Decimal("180"),
        }
        unit_cost = cost_per_kw.get(eq.equipment_type, Decimal("250"))
        total_cost = eq.rated_power_kw * unit_cost

        payback = _safe_divide(total_cost, savings_eur, Decimal("99"))

        options.append(UpgradeOption(
            description=f"Replace with best-practice {eq.equipment_type} equipment",
            estimated_savings_kwh=_round_val(savings_kwh, 2),
            estimated_savings_eur=_round_val(savings_eur, 2),
            estimated_cost_eur=_round_val(total_cost, 2),
            simple_payback_years=_round_val(payback, 2),
            new_efficiency_pct=_round_val(best_eff, 2),
            co2_reduction_tco2e=_round_val(co2_saved, 3),
        ))

        # Option 2: VSD retrofit (for motors, pumps, compressors, fans)
        if eq.equipment_type in (
            EquipmentType.MOTOR.value, EquipmentType.PUMP.value,
            EquipmentType.FAN.value,
        ):
            motor = data.motor_data
            if motor and not motor.has_vsd and motor.actual_load_pct < Decimal("85"):
                vsd_savings_pct = Decimal("20")  # Conservative 20% savings estimate
                vsd_savings_kwh = annual_kwh * vsd_savings_pct / Decimal("100")
                vsd_savings_eur = vsd_savings_kwh * price
                vsd_cost = eq.rated_power_kw * Decimal("120")  # EUR/kW for VSD
                vsd_payback = _safe_divide(vsd_cost, vsd_savings_eur, Decimal("99"))
                vsd_co2 = vsd_savings_kwh * self._co2_factor / Decimal("1000")

                options.append(UpgradeOption(
                    description="Install variable speed drive (VSD)",
                    estimated_savings_kwh=_round_val(vsd_savings_kwh, 2),
                    estimated_savings_eur=_round_val(vsd_savings_eur, 2),
                    estimated_cost_eur=_round_val(vsd_cost, 2),
                    simple_payback_years=_round_val(vsd_payback, 2),
                    new_efficiency_pct=_round_val(
                        current_eff + vsd_savings_pct, 2
                    ),
                    co2_reduction_tco2e=_round_val(vsd_co2, 3),
                ))
                recommendations.append(
                    "Install VSD for variable-load operation. "
                    f"Estimated {_round_val(vsd_savings_pct, 0)}% energy savings."
                )

        # Option 3: Boiler economizer
        if eq.equipment_type == EquipmentType.BOILER.value and data.boiler_data:
            bd = data.boiler_data
            if bd.stack_temp_c > Decimal("200"):
                econ_savings_pct = min(
                    (bd.stack_temp_c - Decimal("150")) * Decimal("0.04"),
                    Decimal("8")
                )
                econ_savings_kwh = annual_kwh * econ_savings_pct / Decimal("100")
                econ_savings_eur = econ_savings_kwh * price
                econ_cost = bd.capacity_kw * Decimal("30")
                econ_payback = _safe_divide(econ_cost, econ_savings_eur, Decimal("99"))
                econ_co2 = econ_savings_kwh * self._co2_factor / Decimal("1000")

                options.append(UpgradeOption(
                    description="Install flue gas economizer",
                    estimated_savings_kwh=_round_val(econ_savings_kwh, 2),
                    estimated_savings_eur=_round_val(econ_savings_eur, 2),
                    estimated_cost_eur=_round_val(econ_cost, 2),
                    simple_payback_years=_round_val(econ_payback, 2),
                    new_efficiency_pct=_round_val(
                        current_eff + econ_savings_pct, 2
                    ),
                    co2_reduction_tco2e=_round_val(econ_co2, 3),
                ))
                recommendations.append(
                    f"Install economizer to reduce stack temperature from "
                    f"{bd.stack_temp_c}C to ~150C. "
                    f"Estimated {_round_val(econ_savings_pct, 1)}% efficiency gain."
                )

        return options
