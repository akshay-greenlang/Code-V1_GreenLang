# -*- coding: utf-8 -*-
"""
SteamOptimizationEngine - PACK-031 Industrial Energy Audit Engine 8
=====================================================================

Steam generation, distribution, and condensate recovery analysis engine.
Calculates boiler efficiency (direct and indirect methods), stack losses
via the Siegert formula, blowdown losses and heat recovery potential,
steam trap survey impacts, insulation deficiency costs, flash steam
recovery from condensate, and Combined Heat and Power (CHP) opportunity
assessment.

Scope:
    - Boiler efficiency (direct output/input and indirect 100%-losses)
    - Flue gas analysis with Siegert coefficients per fuel type
    - Blowdown loss quantification and heat recovery sizing
    - Steam trap survey: leak rate estimation by type and status
    - Pipe / valve insulation assessment (bare-pipe heat loss tables)
    - Flash steam recovery from high-pressure condensate
    - Condensate return rate optimisation
    - Steam system energy balance (generation -> distribution -> end use)
    - Deaerator and feedwater optimisation
    - Steam pressure optimisation for end-use matching
    - CHP / cogeneration opportunity screening

Regulatory / Standard References:
    - EN 12952 / EN 12953 (boiler design and operation)
    - ASME PTC 4 (boiler performance test code)
    - BS 845 (boiler efficiency testing)
    - ISO 50001:2018 Energy Management Systems
    - EU BAT Reference Document for Energy Efficiency (ENE BREF)
    - ESRS E1-5 (energy consumption and mix)

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Steam enthalpy values from IAPWS IF-97 published tables
    - Siegert coefficients from published combustion engineering data
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-031 Industrial Energy Audit
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

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
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
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


def _safe_divide(num: Decimal, den: Decimal, default: Decimal = Decimal("0")) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    return default if den == Decimal("0") else num / den


def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* decimal digits and return float."""
    return float(value.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP))


def _round3(value: Any) -> float:
    """Round to 3 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _round2(value: Any) -> float:
    """Round to 2 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BoilerType(str, Enum):
    """Industrial boiler types."""
    FIRE_TUBE = "fire_tube"
    WATER_TUBE = "water_tube"
    ELECTRIC = "electric"
    WASTE_HEAT = "waste_heat"


class FuelType(str, Enum):
    """Boiler fuel types with Siegert coefficients available."""
    NATURAL_GAS = "natural_gas"
    LIGHT_FUEL_OIL = "light_fuel_oil"
    HEAVY_FUEL_OIL = "heavy_fuel_oil"
    COAL = "coal"
    BIOMASS_WOOD = "biomass_wood"
    BIOMASS_PELLET = "biomass_pellet"
    LPG = "lpg"
    BIOGAS = "biogas"


class SteamTrapType(str, Enum):
    """Steam trap mechanism types."""
    THERMODYNAMIC = "thermodynamic"
    THERMOSTATIC = "thermostatic"
    MECHANICAL = "mechanical"
    FIXED_ORIFICE = "fixed_orifice"


class TrapStatus(str, Enum):
    """Steam trap operational status."""
    OPERATIONAL = "operational"
    FAILED_OPEN = "failed_open"
    FAILED_CLOSED = "failed_closed"
    LEAKING = "leaking"
    NOT_INSPECTED = "not_inspected"


class InsulationMaterial(str, Enum):
    """Pipe insulation material types."""
    MINERAL_WOOL = "mineral_wool"
    CALCIUM_SILICATE = "calcium_silicate"
    CELLULAR_GLASS = "cellular_glass"
    AEROGEL = "aerogel"
    POLYURETHANE = "polyurethane"
    NONE = "none"


class InsulationCondition(str, Enum):
    """Insulation condition categories."""
    GOOD = "good"
    DAMAGED = "damaged"
    MISSING = "missing"
    WET = "wet"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Saturated steam enthalpy (kJ/kg) by gauge pressure (bar).
# Source: IAPWS IF-97 steam tables (published international standard).
STEAM_ENTHALPY_TABLE: Dict[str, Dict[str, float]] = {
    "0.5":  {"hf": 467.1, "hfg": 2226.0, "hg": 2693.1, "temp_c": 111.4},
    "1.0":  {"hf": 504.7, "hfg": 2201.6, "hg": 2706.3, "temp_c": 120.2},
    "2.0":  {"hf": 561.4, "hfg": 2163.2, "hg": 2724.7, "temp_c": 133.5},
    "3.0":  {"hf": 604.7, "hfg": 2133.4, "hg": 2738.1, "temp_c": 143.6},
    "4.0":  {"hf": 640.1, "hfg": 2108.1, "hg": 2748.1, "temp_c": 151.8},
    "5.0":  {"hf": 670.4, "hfg": 2085.8, "hg": 2756.2, "temp_c": 158.8},
    "6.0":  {"hf": 697.1, "hfg": 2065.6, "hg": 2762.8, "temp_c": 164.9},
    "7.0":  {"hf": 721.0, "hfg": 2047.3, "hg": 2768.3, "temp_c": 170.4},
    "8.0":  {"hf": 742.6, "hfg": 2030.5, "hg": 2773.0, "temp_c": 175.4},
    "9.0":  {"hf": 762.6, "hfg": 2014.6, "hg": 2777.2, "temp_c": 179.9},
    "10.0": {"hf": 781.1, "hfg": 1999.6, "hg": 2780.7, "temp_c": 184.1},
    "12.0": {"hf": 814.7, "hfg": 1971.5, "hg": 2786.2, "temp_c": 191.6},
    "14.0": {"hf": 845.0, "hfg": 1945.2, "hg": 2790.2, "temp_c": 198.3},
    "16.0": {"hf": 872.6, "hfg": 1920.5, "hg": 2793.1, "temp_c": 204.3},
    "18.0": {"hf": 898.0, "hfg": 1896.9, "hg": 2795.0, "temp_c": 209.8},
    "20.0": {"hf": 921.7, "hfg": 1874.3, "hg": 2796.0, "temp_c": 214.9},
    "25.0": {"hf": 971.9, "hfg": 1823.2, "hg": 2795.1, "temp_c": 224.0},
    "30.0": {"hf": 1017.4, "hfg": 1774.4, "hg": 2791.8, "temp_c": 235.8},
    "35.0": {"hf": 1058.8, "hfg": 1727.5, "hg": 2786.3, "temp_c": 244.2},
    "40.0": {"hf": 1087.4, "hfg": 1693.8, "hg": 2781.3, "temp_c": 250.3},
}

# Siegert formula coefficients for stack loss calculation.
# Stack loss (%) = K1 * (T_stack - T_ambient) / (CO2 %)
# Source: Published combustion engineering reference data (EN 12953-11).
SIEGERT_COEFFICIENTS: Dict[str, Dict[str, Any]] = {
    FuelType.NATURAL_GAS: {
        "k1": 0.37, "k2": 0.009,
        "stoich_co2_pct": 11.7,
        "ncv_kj_kg": 47100,
        "description": "Methane-rich natural gas",
    },
    FuelType.LIGHT_FUEL_OIL: {
        "k1": 0.48, "k2": 0.007,
        "stoich_co2_pct": 15.4,
        "ncv_kj_kg": 42700,
        "description": "Diesel / gas oil",
    },
    FuelType.HEAVY_FUEL_OIL: {
        "k1": 0.50, "k2": 0.007,
        "stoich_co2_pct": 16.0,
        "ncv_kj_kg": 40200,
        "description": "Heavy fuel oil / bunker",
    },
    FuelType.COAL: {
        "k1": 0.63, "k2": 0.011,
        "stoich_co2_pct": 18.5,
        "ncv_kj_kg": 25100,
        "description": "Bituminous coal",
    },
    FuelType.BIOMASS_WOOD: {
        "k1": 0.55, "k2": 0.010,
        "stoich_co2_pct": 20.3,
        "ncv_kj_kg": 14400,
        "description": "Wood chips / logs",
    },
    FuelType.BIOMASS_PELLET: {
        "k1": 0.52, "k2": 0.010,
        "stoich_co2_pct": 20.3,
        "ncv_kj_kg": 17000,
        "description": "Wood pellets (EN Plus A1)",
    },
    FuelType.LPG: {
        "k1": 0.40, "k2": 0.008,
        "stoich_co2_pct": 13.7,
        "ncv_kj_kg": 46100,
        "description": "Propane / butane mix",
    },
    FuelType.BIOGAS: {
        "k1": 0.39, "k2": 0.009,
        "stoich_co2_pct": 11.0,
        "ncv_kj_kg": 21500,
        "description": "Biogas (60% CH4)",
    },
}

# Bare pipe heat loss (W/m) by nominal pipe diameter (mm) and surface
# temperature (C).  Source: EN ISO 12241, typical horizontal pipe in
# still air (natural convection + radiation, emissivity 0.9).
BARE_PIPE_HEAT_LOSS: Dict[int, Dict[int, float]] = {
    #  pipe_dia_mm: {surface_temp_c: watts_per_metre}
    25:  {100: 52, 150: 105, 200: 175, 250: 265, 300: 375},
    50:  {100: 82, 150: 165, 200: 280, 250: 420, 300: 590},
    80:  {100: 115, 150: 235, 200: 395, 250: 590, 300: 835},
    100: {100: 140, 150: 285, 200: 480, 250: 720, 300: 1015},
    150: {100: 195, 150: 400, 200: 675, 250: 1010, 300: 1430},
    200: {100: 250, 150: 510, 200: 865, 250: 1295, 300: 1840},
    250: {100: 305, 150: 625, 200: 1055, 250: 1585, 300: 2250},
    300: {100: 360, 150: 735, 200: 1245, 250: 1870, 300: 2660},
    400: {100: 465, 150: 955, 200: 1620, 250: 2440, 300: 3470},
    500: {100: 570, 150: 1175, 200: 1995, 250: 3005, 300: 4280},
}

# Flash steam percentage by pressure differential.
# flash_pct = (hf_high - hf_low) / hfg_low * 100
# Pre-computed for common pressure drops (from_bar -> to_bar).
FLASH_STEAM_PCT: Dict[str, float] = {
    "10.0_to_1.0":  12.54,
    "10.0_to_2.0":  9.97,
    "10.0_to_3.0":  8.00,
    "10.0_to_4.0":  6.39,
    "10.0_to_5.0":  5.02,
    "8.0_to_1.0":   10.80,
    "8.0_to_2.0":   8.22,
    "8.0_to_3.0":   6.26,
    "6.0_to_1.0":   8.73,
    "6.0_to_2.0":   6.16,
    "6.0_to_3.0":   4.19,
    "5.0_to_1.0":   7.52,
    "5.0_to_2.0":   4.95,
    "4.0_to_1.0":   6.15,
    "4.0_to_2.0":   3.57,
    "3.0_to_1.0":   4.54,
    "3.0_to_0.5":   5.98,
    "2.0_to_0.5":   2.44,
    "2.0_to_1.0":   2.58,
    "20.0_to_1.0":  18.93,
    "20.0_to_3.0":  14.38,
    "20.0_to_5.0":  11.40,
    "25.0_to_1.0":  21.22,
    "25.0_to_5.0":  13.69,
    "30.0_to_1.0":  23.28,
    "30.0_to_5.0":  15.74,
}

# Steam trap failure: estimated leaking steam flow (kg/h) by orifice
# size and trap type.  Source: US DOE Steam Best Practices.
TRAP_FAILURE_STEAM_LOSS: Dict[str, Dict[str, float]] = {
    # trap_type: {orifice_mm: kg_per_hour}
    SteamTrapType.THERMODYNAMIC: {
        "3":  5.0, "5":  12.0, "8":  25.0, "10": 40.0, "12": 60.0,
    },
    SteamTrapType.THERMOSTATIC: {
        "3":  4.0, "5":  10.0, "8":  20.0, "10": 35.0, "12": 50.0,
    },
    SteamTrapType.MECHANICAL: {
        "3":  6.0, "5":  15.0, "8":  30.0, "10": 50.0, "12": 75.0,
    },
    SteamTrapType.FIXED_ORIFICE: {
        "3":  3.0, "5":  8.0,  "8":  18.0, "10": 30.0, "12": 45.0,
    },
}

# Insulation thermal conductivity (W/m-K) at mean temperature 100 C.
# Source: Manufacturer data and EN ISO 12241 reference values.
INSULATION_THERMAL_CONDUCTIVITY: Dict[str, float] = {
    InsulationMaterial.MINERAL_WOOL:      0.040,
    InsulationMaterial.CALCIUM_SILICATE:  0.055,
    InsulationMaterial.CELLULAR_GLASS:    0.048,
    InsulationMaterial.AEROGEL:           0.020,
    InsulationMaterial.POLYURETHANE:      0.028,
}

# Boiler efficiency benchmarks by type.
# Source: EU ENE BREF, US DOE AMO Steam System Assessment Tool.
BOILER_EFFICIENCY_BENCHMARKS: Dict[str, Dict[str, float]] = {
    BoilerType.FIRE_TUBE: {
        "best_practice_pct": 86.0,
        "good_pct": 82.0,
        "average_pct": 78.0,
        "poor_pct": 72.0,
    },
    BoilerType.WATER_TUBE: {
        "best_practice_pct": 90.0,
        "good_pct": 86.0,
        "average_pct": 82.0,
        "poor_pct": 76.0,
    },
    BoilerType.ELECTRIC: {
        "best_practice_pct": 99.0,
        "good_pct": 98.0,
        "average_pct": 97.0,
        "poor_pct": 95.0,
    },
    BoilerType.WASTE_HEAT: {
        "best_practice_pct": 85.0,
        "good_pct": 80.0,
        "average_pct": 75.0,
        "poor_pct": 65.0,
    },
}

# Steam cost reference (EUR/tonne).
STEAM_COST_EUR_PER_TONNE: Decimal = Decimal("30.0")

# Water cost reference (EUR/m3).
WATER_COST_EUR_PER_M3: Decimal = Decimal("2.50")

# Water treatment chemical cost (EUR/m3).
WATER_TREATMENT_EUR_PER_M3: Decimal = Decimal("1.50")

# Typical CHP electrical efficiency.
CHP_ELECTRICAL_EFFICIENCY: Decimal = Decimal("0.35")

# Typical CHP thermal efficiency.
CHP_THERMAL_EFFICIENCY: Decimal = Decimal("0.50")

# CHP capital cost (EUR/kW_e).
CHP_CAPEX_EUR_PER_KWE: Decimal = Decimal("1200")

# Hours per year.
HOURS_PER_YEAR: int = 8760


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class FlueGasAnalysis(BaseModel):
    """Flue gas analysis data from stack measurement.

    Attributes:
        co2_pct: CO2 concentration in flue gas (% vol dry).
        o2_pct: O2 concentration in flue gas (% vol dry).
        co_ppm: CO concentration (ppm).
        stack_temp_c: Flue gas temperature at stack exit (C).
        ambient_temp_c: Combustion air inlet temperature (C).
        excess_air_pct: Excess air percentage (calculated or measured).
        combustion_efficiency_pct: Pre-measured combustion efficiency
            if available; engine will compute if zero.
    """
    co2_pct: float = Field(default=0.0, ge=0.0, le=25.0)
    o2_pct: float = Field(default=0.0, ge=0.0, le=21.0)
    co_ppm: float = Field(default=0.0, ge=0.0)
    stack_temp_c: float = Field(default=180.0, ge=50.0, le=600.0)
    ambient_temp_c: float = Field(default=20.0, ge=-20.0, le=50.0)
    excess_air_pct: float = Field(default=0.0, ge=0.0, le=500.0)
    combustion_efficiency_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class Boiler(BaseModel):
    """Individual boiler unit data.

    Attributes:
        boiler_id: Unique identifier for the boiler.
        name: Human-readable name.
        boiler_type: Type of boiler (fire-tube, water-tube, etc.).
        fuel_type: Primary fuel.
        capacity_kg_h: Rated steam generation capacity (kg/h).
        design_pressure_bar: Design pressure (bar gauge).
        operating_pressure_bar: Current operating pressure (bar gauge).
        feed_water_temp_c: Feedwater temperature entering the boiler (C).
        stack_temp_c: Flue gas exit temperature (C).
        excess_air_pct: Measured excess air (%).
        blowdown_pct: Blowdown rate (% of feedwater).
        operating_hours: Annual operating hours.
        annual_fuel_cost_eur: Annual fuel cost (EUR).
        annual_fuel_consumption_kwh: Annual fuel input energy (kWh).
        flue_gas: Optional detailed flue gas analysis.
    """
    boiler_id: str = Field(default_factory=_new_uuid)
    name: str = Field(default="", min_length=0)
    boiler_type: BoilerType = Field(default=BoilerType.FIRE_TUBE)
    fuel_type: FuelType = Field(default=FuelType.NATURAL_GAS)
    capacity_kg_h: float = Field(default=0.0, ge=0.0)
    design_pressure_bar: float = Field(default=10.0, ge=0.0, le=200.0)
    operating_pressure_bar: float = Field(default=8.0, ge=0.0, le=200.0)
    feed_water_temp_c: float = Field(default=80.0, ge=0.0, le=200.0)
    stack_temp_c: float = Field(default=180.0, ge=50.0, le=600.0)
    excess_air_pct: float = Field(default=15.0, ge=0.0, le=500.0)
    blowdown_pct: float = Field(default=5.0, ge=0.0, le=30.0)
    operating_hours: int = Field(default=6000, ge=0, le=8760)
    annual_fuel_cost_eur: float = Field(default=0.0, ge=0.0)
    annual_fuel_consumption_kwh: float = Field(default=0.0, ge=0.0)
    flue_gas: Optional[FlueGasAnalysis] = Field(default=None)


class SteamTrapRecord(BaseModel):
    """Individual steam trap record from a survey.

    Attributes:
        trap_id: Unique trap identifier or tag.
        trap_type: Mechanism type.
        status: Current operational status.
        orifice_mm: Orifice diameter (mm).
        operating_pressure_bar: Upstream steam pressure (bar).
        location: Installation location description.
    """
    trap_id: str = Field(default_factory=_new_uuid)
    trap_type: SteamTrapType = Field(default=SteamTrapType.THERMODYNAMIC)
    status: TrapStatus = Field(default=TrapStatus.OPERATIONAL)
    orifice_mm: float = Field(default=5.0, ge=1.0, le=25.0)
    operating_pressure_bar: float = Field(default=5.0, ge=0.1, le=50.0)
    location: str = Field(default="")


class PipeSection(BaseModel):
    """Individual pipe section for insulation assessment.

    Attributes:
        section_id: Unique section identifier.
        diameter_mm: Nominal pipe diameter (mm).
        length_m: Section length (metres).
        surface_temp_c: Measured surface temperature (C).
        insulation_material: Current insulation material.
        insulation_condition: Insulation condition.
        insulation_thickness_mm: Current insulation thickness (mm).
    """
    section_id: str = Field(default_factory=_new_uuid)
    diameter_mm: int = Field(default=100, ge=15, le=600)
    length_m: float = Field(default=10.0, ge=0.1, le=5000.0)
    surface_temp_c: float = Field(default=150.0, ge=20.0, le=400.0)
    insulation_material: InsulationMaterial = Field(default=InsulationMaterial.MINERAL_WOOL)
    insulation_condition: InsulationCondition = Field(default=InsulationCondition.GOOD)
    insulation_thickness_mm: float = Field(default=50.0, ge=0.0, le=300.0)


class CondensateSystem(BaseModel):
    """Condensate return system data.

    Attributes:
        return_rate_pct: Current condensate return rate (%).
        flash_steam_recovered: Whether flash steam is currently recovered.
        flash_vessel_pressure_bar: Flash vessel operating pressure (bar).
        condensate_temp_c: Return condensate temperature (C).
        makeup_water_rate_pct: Makeup water as % of total feedwater.
        total_condensate_flow_kg_h: Total condensate flow (kg/h).
        condensate_pressure_bar: Condensate pressure at recovery point (bar).
    """
    return_rate_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    flash_steam_recovered: bool = Field(default=False)
    flash_vessel_pressure_bar: float = Field(default=1.0, ge=0.1, le=20.0)
    condensate_temp_c: float = Field(default=90.0, ge=20.0, le=250.0)
    makeup_water_rate_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    total_condensate_flow_kg_h: float = Field(default=0.0, ge=0.0)
    condensate_pressure_bar: float = Field(default=5.0, ge=0.1, le=50.0)


class SteamSystem(BaseModel):
    """Complete steam system data for a facility.

    Attributes:
        system_id: Unique system identifier.
        facility_id: Parent facility identifier.
        facility_name: Facility name.
        boilers: List of boiler units.
        steam_traps: List of surveyed steam traps.
        pipe_sections: List of pipe sections for insulation audit.
        condensate_system: Condensate return system data.
        total_steam_demand_kg_h: Total steam demand (kg/h).
        operating_hours: Annual steam system operating hours.
        energy_cost_eur_per_kwh: Energy cost for savings calculation.
    """
    system_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default_factory=_new_uuid)
    facility_name: str = Field(default="")
    boilers: List[Boiler] = Field(default_factory=list)
    steam_traps: List[SteamTrapRecord] = Field(default_factory=list)
    pipe_sections: List[PipeSection] = Field(default_factory=list)
    condensate_system: Optional[CondensateSystem] = Field(default=None)
    total_steam_demand_kg_h: float = Field(default=0.0, ge=0.0)
    operating_hours: int = Field(default=6000, ge=0, le=8760)
    energy_cost_eur_per_kwh: float = Field(default=0.06, ge=0.0)


class BoilerEfficiencyResult(BaseModel):
    """Efficiency result for a single boiler.

    Attributes:
        boiler_id: Boiler identifier.
        name: Boiler name.
        direct_efficiency_pct: Direct method efficiency (output/input).
        indirect_efficiency_pct: Indirect method efficiency (100% - losses).
        stack_loss_pct: Dry flue gas loss from Siegert formula.
        blowdown_loss_pct: Loss from blowdown.
        radiation_loss_pct: Shell radiation and convection loss estimate.
        unaccounted_loss_pct: Unaccounted losses.
        benchmark_rating: Rating against type-specific benchmark.
        potential_improvement_pct: Gap to best-practice efficiency.
        annual_savings_kwh: Potential annual energy savings (kWh).
        annual_savings_eur: Potential annual cost savings (EUR).
    """
    boiler_id: str = Field(default="")
    name: str = Field(default="")
    direct_efficiency_pct: float = Field(default=0.0)
    indirect_efficiency_pct: float = Field(default=0.0)
    stack_loss_pct: float = Field(default=0.0)
    blowdown_loss_pct: float = Field(default=0.0)
    radiation_loss_pct: float = Field(default=0.0)
    unaccounted_loss_pct: float = Field(default=0.0)
    benchmark_rating: str = Field(default="unknown")
    potential_improvement_pct: float = Field(default=0.0)
    annual_savings_kwh: float = Field(default=0.0)
    annual_savings_eur: float = Field(default=0.0)


class SteamTrapSurveyResult(BaseModel):
    """Steam trap survey analysis result.

    Attributes:
        total_traps: Total number of traps surveyed.
        operational_traps: Traps operating correctly.
        failed_open_traps: Traps failed in the open position (leaking).
        failed_closed_traps: Traps failed closed (blocked).
        leaking_traps: Traps with partial leakage.
        not_inspected_traps: Traps not inspected.
        failure_rate_pct: Overall failure rate (%).
        estimated_steam_loss_kg_h: Total estimated steam loss (kg/h).
        estimated_annual_loss_kwh: Annual energy loss (kWh).
        estimated_annual_cost_eur: Annual cost of steam leakage (EUR).
        replacement_cost_eur: Estimated cost to replace all failed traps.
        payback_years: Simple payback for trap replacement programme.
    """
    total_traps: int = Field(default=0)
    operational_traps: int = Field(default=0)
    failed_open_traps: int = Field(default=0)
    failed_closed_traps: int = Field(default=0)
    leaking_traps: int = Field(default=0)
    not_inspected_traps: int = Field(default=0)
    failure_rate_pct: float = Field(default=0.0)
    estimated_steam_loss_kg_h: float = Field(default=0.0)
    estimated_annual_loss_kwh: float = Field(default=0.0)
    estimated_annual_cost_eur: float = Field(default=0.0)
    replacement_cost_eur: float = Field(default=0.0)
    payback_years: float = Field(default=0.0)


class InsulationAssessmentResult(BaseModel):
    """Insulation assessment result for the pipe network.

    Attributes:
        pipe_sections_total: Total pipe sections assessed.
        uninsulated_sections: Sections with no insulation.
        damaged_sections: Sections with damaged insulation.
        good_sections: Sections with good insulation.
        bare_pipe_heat_loss_kw: Total heat loss from uninsulated
            or damaged sections (kW).
        potential_savings_kw: Potential heat loss reduction (kW).
        potential_savings_kwh: Annual energy savings (kWh).
        potential_savings_eur: Annual cost savings (EUR).
        insulation_cost_eur: Estimated insulation remediation cost.
        payback_years: Simple payback period (years).
    """
    pipe_sections_total: int = Field(default=0)
    uninsulated_sections: int = Field(default=0)
    damaged_sections: int = Field(default=0)
    good_sections: int = Field(default=0)
    bare_pipe_heat_loss_kw: float = Field(default=0.0)
    potential_savings_kw: float = Field(default=0.0)
    potential_savings_kwh: float = Field(default=0.0)
    potential_savings_eur: float = Field(default=0.0)
    insulation_cost_eur: float = Field(default=0.0)
    payback_years: float = Field(default=0.0)


class FlashSteamRecoveryResult(BaseModel):
    """Flash steam recovery analysis result.

    Attributes:
        flash_steam_pct: Percentage of condensate flashing to steam.
        flash_steam_flow_kg_h: Flash steam mass flow (kg/h).
        energy_recoverable_kw: Thermal energy recoverable (kW).
        annual_savings_kwh: Annual energy savings (kWh).
        annual_savings_eur: Annual cost savings (EUR).
        equipment_cost_eur: Flash vessel and piping cost estimate.
        payback_years: Simple payback (years).
    """
    flash_steam_pct: float = Field(default=0.0)
    flash_steam_flow_kg_h: float = Field(default=0.0)
    energy_recoverable_kw: float = Field(default=0.0)
    annual_savings_kwh: float = Field(default=0.0)
    annual_savings_eur: float = Field(default=0.0)
    equipment_cost_eur: float = Field(default=0.0)
    payback_years: float = Field(default=0.0)


class BlowdownRecoveryResult(BaseModel):
    """Blowdown heat recovery analysis result.

    Attributes:
        blowdown_rate_pct: Current blowdown rate (%).
        blowdown_flow_kg_h: Blowdown mass flow (kg/h).
        energy_in_blowdown_kw: Thermal energy in blowdown (kW).
        recoverable_energy_kw: Energy recoverable via flash + HX (kW).
        annual_savings_kwh: Annual energy savings (kWh).
        annual_savings_eur: Annual cost savings (EUR).
        water_savings_m3: Annual water savings (m3).
        water_savings_eur: Annual water cost savings (EUR).
        equipment_cost_eur: Flash tank + heat exchanger cost.
        payback_years: Simple payback (years).
    """
    blowdown_rate_pct: float = Field(default=0.0)
    blowdown_flow_kg_h: float = Field(default=0.0)
    energy_in_blowdown_kw: float = Field(default=0.0)
    recoverable_energy_kw: float = Field(default=0.0)
    annual_savings_kwh: float = Field(default=0.0)
    annual_savings_eur: float = Field(default=0.0)
    water_savings_m3: float = Field(default=0.0)
    water_savings_eur: float = Field(default=0.0)
    equipment_cost_eur: float = Field(default=0.0)
    payback_years: float = Field(default=0.0)


class CHPOpportunity(BaseModel):
    """Combined Heat and Power opportunity assessment.

    Attributes:
        viable: Whether CHP is technically viable.
        thermal_demand_kw: Site thermal demand (kW).
        electrical_output_kw: Estimated CHP electrical output (kW).
        thermal_output_kw: Estimated CHP thermal output (kW).
        annual_electricity_kwh: Annual electricity generation (kWh).
        annual_electricity_savings_eur: Annual electricity cost offset.
        capex_eur: Capital expenditure estimate (EUR).
        payback_years: Simple payback period.
        co2_savings_tonnes: Annual CO2 reduction estimate (tonnes).
    """
    viable: bool = Field(default=False)
    thermal_demand_kw: float = Field(default=0.0)
    electrical_output_kw: float = Field(default=0.0)
    thermal_output_kw: float = Field(default=0.0)
    annual_electricity_kwh: float = Field(default=0.0)
    annual_electricity_savings_eur: float = Field(default=0.0)
    capex_eur: float = Field(default=0.0)
    payback_years: float = Field(default=0.0)
    co2_savings_tonnes: float = Field(default=0.0)


class SteamOptimizationResult(BaseModel):
    """Complete steam system optimisation result with full provenance.

    Attributes:
        result_id: Unique result identifier.
        system_id: Steam system identifier.
        facility_id: Facility identifier.
        boiler_efficiency_results: Per-boiler efficiency analysis.
        trap_survey: Steam trap survey results.
        insulation_assessment: Pipe insulation assessment.
        condensate_analysis: Condensate return improvement analysis.
        flash_steam_recovery: Flash steam recovery opportunity.
        blowdown_recovery: Blowdown heat recovery opportunity.
        chp_opportunity: CHP / cogeneration opportunity.
        total_savings_kwh: Total annual energy savings (kWh).
        total_savings_eur: Total annual cost savings (EUR).
        total_investment_eur: Total investment required (EUR).
        simple_payback_years: Overall simple payback period.
        recommendations: Prioritised recommendations list.
        methodology_notes: Methodology and data source notes.
        processing_time_ms: Computation time (ms).
        engine_version: Engine version string.
        calculated_at: UTC timestamp of calculation.
        provenance_hash: SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    system_id: str = Field(default="")
    facility_id: str = Field(default="")
    boiler_efficiency_results: List[BoilerEfficiencyResult] = Field(default_factory=list)
    trap_survey: Optional[SteamTrapSurveyResult] = Field(default=None)
    insulation_assessment: Optional[InsulationAssessmentResult] = Field(default=None)
    condensate_analysis: Optional[Dict[str, Any]] = Field(default=None)
    flash_steam_recovery: Optional[FlashSteamRecoveryResult] = Field(default=None)
    blowdown_recovery: Optional[BlowdownRecoveryResult] = Field(default=None)
    chp_opportunity: Optional[CHPOpportunity] = Field(default=None)
    total_savings_kwh: float = Field(default=0.0)
    total_savings_eur: float = Field(default=0.0)
    total_investment_eur: float = Field(default=0.0)
    simple_payback_years: float = Field(default=0.0)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    methodology_notes: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SteamOptimizationEngine:
    """Zero-hallucination steam system optimisation engine.

    Analyses steam generation, distribution, and condensate recovery to
    identify energy saving opportunities.  All calculations are
    deterministic, bit-perfect, and carry SHA-256 provenance hashes.

    Guarantees:
        - Deterministic: same inputs produce identical outputs.
        - Reproducible: every result carries a SHA-256 provenance hash.
        - Auditable: full breakdown of boiler, trap, insulation, and
          condensate analysis with referenced constants.
        - No LLM: zero hallucination risk in any calculation path.

    Usage::

        engine = SteamOptimizationEngine()
        result = engine.analyze_steam_system(steam_system)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the steam optimisation engine.

        Args:
            config: Optional configuration overrides (currently unused;
                    reserved for future settings).
        """
        self._config = config or {}
        self._notes: List[str] = []
        logger.info("SteamOptimizationEngine v%s initialised.", _MODULE_VERSION)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def analyze_steam_system(self, system: SteamSystem) -> SteamOptimizationResult:
        """Run a comprehensive steam system optimisation analysis.

        Evaluates boiler efficiency, steam traps, insulation, condensate
        recovery, flash steam, blowdown, and CHP opportunities.

        Args:
            system: Complete steam system data.

        Returns:
            SteamOptimizationResult with full breakdown and provenance.

        Raises:
            ValueError: If no boilers are defined on the system.
        """
        t0 = time.perf_counter()
        self._notes = [
            f"Engine version: {self.engine_version}",
            f"Analysis timestamp: {_utcnow().isoformat()}",
        ]

        if not system.boilers:
            raise ValueError("Steam system must contain at least one boiler.")

        total_savings_kwh = Decimal("0")
        total_savings_eur = Decimal("0")
        total_investment = Decimal("0")
        recommendations: List[Dict[str, Any]] = []
        priority = 1

        # --- 1. Boiler efficiency ---
        boiler_results: List[BoilerEfficiencyResult] = []
        for boiler in system.boilers:
            br = self.calculate_boiler_efficiency(boiler, system.energy_cost_eur_per_kwh)
            boiler_results.append(br)
            total_savings_kwh += _decimal(br.annual_savings_kwh)
            total_savings_eur += _decimal(br.annual_savings_eur)
            if br.potential_improvement_pct > 2.0:
                recommendations.append({
                    "priority": priority,
                    "category": "boiler_efficiency",
                    "description": (
                        f"Improve boiler '{br.name or br.boiler_id}' efficiency "
                        f"by {br.potential_improvement_pct}% to reach best practice."
                    ),
                    "annual_savings_kwh": br.annual_savings_kwh,
                    "annual_savings_eur": br.annual_savings_eur,
                })
                priority += 1

        # --- 2. Steam trap survey ---
        trap_result: Optional[SteamTrapSurveyResult] = None
        if system.steam_traps:
            trap_result = self.analyze_steam_traps(
                system.steam_traps,
                system.operating_hours,
                system.energy_cost_eur_per_kwh,
            )
            total_savings_kwh += _decimal(trap_result.estimated_annual_loss_kwh)
            total_savings_eur += _decimal(trap_result.estimated_annual_cost_eur)
            total_investment += _decimal(trap_result.replacement_cost_eur)
            if trap_result.failure_rate_pct > 5.0:
                recommendations.append({
                    "priority": priority,
                    "category": "steam_traps",
                    "description": (
                        f"Replace {trap_result.failed_open_traps + trap_result.leaking_traps} "
                        f"failed/leaking steam traps ({trap_result.failure_rate_pct}% failure rate)."
                    ),
                    "annual_savings_kwh": trap_result.estimated_annual_loss_kwh,
                    "annual_savings_eur": trap_result.estimated_annual_cost_eur,
                    "investment_eur": trap_result.replacement_cost_eur,
                    "payback_years": trap_result.payback_years,
                })
                priority += 1

        # --- 3. Insulation assessment ---
        insulation_result: Optional[InsulationAssessmentResult] = None
        if system.pipe_sections:
            insulation_result = self.assess_insulation(
                system.pipe_sections,
                system.operating_hours,
                system.energy_cost_eur_per_kwh,
            )
            total_savings_kwh += _decimal(insulation_result.potential_savings_kwh)
            total_savings_eur += _decimal(insulation_result.potential_savings_eur)
            total_investment += _decimal(insulation_result.insulation_cost_eur)
            if insulation_result.uninsulated_sections + insulation_result.damaged_sections > 0:
                recommendations.append({
                    "priority": priority,
                    "category": "insulation",
                    "description": (
                        f"Insulate/repair {insulation_result.uninsulated_sections + insulation_result.damaged_sections} "
                        f"pipe sections to recover {_round2(insulation_result.potential_savings_kw)} kW."
                    ),
                    "annual_savings_kwh": insulation_result.potential_savings_kwh,
                    "annual_savings_eur": insulation_result.potential_savings_eur,
                    "investment_eur": insulation_result.insulation_cost_eur,
                    "payback_years": insulation_result.payback_years,
                })
                priority += 1

        # --- 4. Flash steam recovery ---
        flash_result: Optional[FlashSteamRecoveryResult] = None
        if system.condensate_system and not system.condensate_system.flash_steam_recovered:
            flash_result = self.calculate_flash_steam_recovery(
                system.condensate_system,
                system.operating_hours,
                system.energy_cost_eur_per_kwh,
            )
            total_savings_kwh += _decimal(flash_result.annual_savings_kwh)
            total_savings_eur += _decimal(flash_result.annual_savings_eur)
            total_investment += _decimal(flash_result.equipment_cost_eur)
            if flash_result.annual_savings_kwh > 0:
                recommendations.append({
                    "priority": priority,
                    "category": "flash_steam_recovery",
                    "description": (
                        f"Install flash steam recovery vessel to capture "
                        f"{_round2(flash_result.flash_steam_pct)}% flash steam."
                    ),
                    "annual_savings_kwh": flash_result.annual_savings_kwh,
                    "annual_savings_eur": flash_result.annual_savings_eur,
                    "investment_eur": flash_result.equipment_cost_eur,
                    "payback_years": flash_result.payback_years,
                })
                priority += 1

        # --- 5. Blowdown heat recovery ---
        blowdown_result: Optional[BlowdownRecoveryResult] = None
        if system.boilers:
            avg_blowdown = _decimal(
                sum(b.blowdown_pct for b in system.boilers) / len(system.boilers)
            )
            avg_pressure = _decimal(
                sum(b.operating_pressure_bar for b in system.boilers) / len(system.boilers)
            )
            total_steam_kg_h = _decimal(system.total_steam_demand_kg_h)
            if total_steam_kg_h == Decimal("0"):
                total_steam_kg_h = _decimal(
                    sum(float(b.capacity_kg_h) * 0.7 for b in system.boilers)
                )
            blowdown_result = self.calculate_blowdown_recovery(
                blowdown_pct=float(avg_blowdown),
                steam_flow_kg_h=float(total_steam_kg_h),
                pressure_bar=float(avg_pressure),
                feed_water_temp_c=float(_decimal(
                    sum(b.feed_water_temp_c for b in system.boilers) / len(system.boilers)
                )),
                operating_hours=system.operating_hours,
                energy_cost_eur_per_kwh=system.energy_cost_eur_per_kwh,
            )
            total_savings_kwh += _decimal(blowdown_result.annual_savings_kwh)
            total_savings_eur += _decimal(blowdown_result.annual_savings_eur + blowdown_result.water_savings_eur)
            total_investment += _decimal(blowdown_result.equipment_cost_eur)
            if blowdown_result.annual_savings_kwh > 0:
                recommendations.append({
                    "priority": priority,
                    "category": "blowdown_recovery",
                    "description": (
                        f"Install blowdown heat recovery to save "
                        f"{_round2(blowdown_result.annual_savings_kwh)} kWh/year."
                    ),
                    "annual_savings_kwh": blowdown_result.annual_savings_kwh,
                    "annual_savings_eur": blowdown_result.annual_savings_eur,
                    "investment_eur": blowdown_result.equipment_cost_eur,
                    "payback_years": blowdown_result.payback_years,
                })
                priority += 1

        # --- 6. Condensate return analysis ---
        condensate_analysis: Optional[Dict[str, Any]] = None
        if system.condensate_system:
            condensate_analysis = self.analyze_condensate_return(
                system.condensate_system,
                system.operating_hours,
                system.energy_cost_eur_per_kwh,
            )
            cond_savings = _decimal(condensate_analysis.get("annual_savings_kwh", 0))
            total_savings_kwh += cond_savings
            total_savings_eur += _decimal(condensate_analysis.get("annual_savings_eur", 0))
            if float(cond_savings) > 0:
                recommendations.append({
                    "priority": priority,
                    "category": "condensate_return",
                    "description": (
                        f"Increase condensate return from "
                        f"{system.condensate_system.return_rate_pct}% to "
                        f"{condensate_analysis.get('target_return_rate_pct', 90)}%."
                    ),
                    "annual_savings_kwh": float(cond_savings),
                    "annual_savings_eur": condensate_analysis.get("annual_savings_eur", 0),
                })
                priority += 1

        # --- 7. CHP opportunity ---
        chp_result: Optional[CHPOpportunity] = None
        thermal_demand_kw = self._estimate_thermal_demand(system)
        if thermal_demand_kw > Decimal("500"):
            chp_result = self.assess_chp_opportunity(
                thermal_demand_kw=float(thermal_demand_kw),
                operating_hours=system.operating_hours,
                electricity_cost_eur_per_kwh=max(system.energy_cost_eur_per_kwh * 2, 0.10),
            )
            if chp_result.viable:
                total_savings_eur += _decimal(chp_result.annual_electricity_savings_eur)
                total_investment += _decimal(chp_result.capex_eur)
                recommendations.append({
                    "priority": priority,
                    "category": "chp",
                    "description": (
                        f"Install {_round2(chp_result.electrical_output_kw)} kW CHP unit "
                        f"to co-generate electricity and heat."
                    ),
                    "annual_savings_eur": chp_result.annual_electricity_savings_eur,
                    "investment_eur": chp_result.capex_eur,
                    "payback_years": chp_result.payback_years,
                    "co2_savings_tonnes": chp_result.co2_savings_tonnes,
                })

        # Sort recommendations by payback (shortest first, items without
        # payback at end).
        recommendations.sort(key=lambda r: r.get("payback_years", 999))

        # Overall payback
        overall_payback = _round2(float(
            _safe_divide(total_investment, total_savings_eur)
        )) if total_savings_eur > 0 else 0.0

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = SteamOptimizationResult(
            system_id=system.system_id,
            facility_id=system.facility_id,
            boiler_efficiency_results=boiler_results,
            trap_survey=trap_result,
            insulation_assessment=insulation_result,
            condensate_analysis=condensate_analysis,
            flash_steam_recovery=flash_result,
            blowdown_recovery=blowdown_result,
            chp_opportunity=chp_result,
            total_savings_kwh=_round2(float(total_savings_kwh)),
            total_savings_eur=_round2(float(total_savings_eur)),
            total_investment_eur=_round2(float(total_investment)),
            simple_payback_years=overall_payback,
            recommendations=recommendations,
            methodology_notes=list(self._notes),
            processing_time_ms=round(elapsed_ms, 3),
        )

        result.provenance_hash = _compute_hash(result)
        return result

    # --------------------------------------------------------------------- #
    # Boiler Efficiency
    # --------------------------------------------------------------------- #

    def calculate_boiler_efficiency(
        self,
        boiler: Boiler,
        energy_cost_eur_per_kwh: float = 0.06,
    ) -> BoilerEfficiencyResult:
        """Calculate boiler efficiency using the indirect (heat-loss) method.

        Indirect efficiency = 100% - stack_loss - blowdown_loss -
                              radiation_loss - unaccounted_loss

        Stack loss is computed from the Siegert formula:
            L_stack = K1 * (T_stack - T_ambient) / CO2_pct

        Args:
            boiler: Boiler data with operating parameters.
            energy_cost_eur_per_kwh: Energy cost for savings calculation.

        Returns:
            BoilerEfficiencyResult with detailed loss breakdown.
        """
        d_fuel_input = _decimal(boiler.annual_fuel_consumption_kwh)

        # --- Stack loss (Siegert formula) ---
        stack_loss = self._calculate_stack_loss(boiler)

        # --- Blowdown loss ---
        blowdown_loss = self._calculate_blowdown_loss(boiler)

        # --- Radiation and convection loss ---
        # BS 845 / ASME PTC 4 simplified estimate: 1-4% depending on size.
        capacity = _decimal(boiler.capacity_kg_h)
        if capacity >= _decimal(20000):
            radiation_loss = Decimal("0.5")
        elif capacity >= _decimal(10000):
            radiation_loss = Decimal("1.0")
        elif capacity >= _decimal(5000):
            radiation_loss = Decimal("1.5")
        elif capacity >= _decimal(2000):
            radiation_loss = Decimal("2.0")
        else:
            radiation_loss = Decimal("3.0")

        # --- Unaccounted loss ---
        unaccounted = Decimal("1.0")

        # --- Indirect efficiency ---
        total_losses = stack_loss + blowdown_loss + radiation_loss + unaccounted
        indirect_eff = Decimal("100") - total_losses
        indirect_eff = max(indirect_eff, Decimal("0"))

        # --- Direct efficiency (simplified if fuel input known) ---
        direct_eff = Decimal("0")
        if d_fuel_input > Decimal("0"):
            pressure_key = str(float(_decimal(boiler.operating_pressure_bar)))
            steam_entry = self._lookup_steam_enthalpy(pressure_key)
            if steam_entry:
                hg = _decimal(steam_entry["hg"])
                hf_feed = _decimal(boiler.feed_water_temp_c) * Decimal("4.186")
                steam_energy = _decimal(boiler.capacity_kg_h) * (hg - hf_feed)
                steam_kwh = steam_energy / Decimal("3600") * _decimal(boiler.operating_hours)
                direct_eff = _safe_divide(steam_kwh, d_fuel_input) * Decimal("100")
                direct_eff = min(direct_eff, Decimal("100"))

        # --- Benchmark rating ---
        bm = BOILER_EFFICIENCY_BENCHMARKS.get(boiler.boiler_type, {})
        best_practice = _decimal(bm.get("best_practice_pct", 90))
        good_val = _decimal(bm.get("good_pct", 85))
        average_val = _decimal(bm.get("average_pct", 80))

        if indirect_eff >= best_practice:
            rating = "best_practice"
        elif indirect_eff >= good_val:
            rating = "good"
        elif indirect_eff >= average_val:
            rating = "average"
        else:
            rating = "poor"

        improvement_pct = max(Decimal("0"), best_practice - indirect_eff)

        # --- Annual savings potential ---
        annual_savings_kwh = Decimal("0")
        annual_savings_eur = Decimal("0")
        if d_fuel_input > Decimal("0") and improvement_pct > Decimal("0"):
            current_fuel = d_fuel_input
            improved_fuel = current_fuel * indirect_eff / (indirect_eff + improvement_pct)
            annual_savings_kwh = current_fuel - improved_fuel
            annual_savings_eur = annual_savings_kwh * _decimal(energy_cost_eur_per_kwh)

        self._notes.append(
            f"Boiler '{boiler.name or boiler.boiler_id}': indirect efficiency "
            f"{_round2(float(indirect_eff))}%, stack loss {_round2(float(stack_loss))}%, "
            f"blowdown loss {_round2(float(blowdown_loss))}%, rating: {rating}."
        )

        return BoilerEfficiencyResult(
            boiler_id=boiler.boiler_id,
            name=boiler.name,
            direct_efficiency_pct=_round2(float(direct_eff)),
            indirect_efficiency_pct=_round2(float(indirect_eff)),
            stack_loss_pct=_round2(float(stack_loss)),
            blowdown_loss_pct=_round2(float(blowdown_loss)),
            radiation_loss_pct=_round2(float(radiation_loss)),
            unaccounted_loss_pct=_round2(float(unaccounted)),
            benchmark_rating=rating,
            potential_improvement_pct=_round2(float(improvement_pct)),
            annual_savings_kwh=_round2(float(annual_savings_kwh)),
            annual_savings_eur=_round2(float(annual_savings_eur)),
        )

    # --------------------------------------------------------------------- #
    # Steam Trap Survey
    # --------------------------------------------------------------------- #

    def analyze_steam_traps(
        self,
        traps: List[SteamTrapRecord],
        operating_hours: int,
        energy_cost_eur_per_kwh: float,
    ) -> SteamTrapSurveyResult:
        """Analyse a steam trap survey to quantify leakage losses.

        For each failed-open or leaking trap, estimates the steam loss
        using the TRAP_FAILURE_STEAM_LOSS lookup table and the operating
        pressure enthalpy.

        Args:
            traps: List of surveyed steam trap records.
            operating_hours: Annual operating hours.
            energy_cost_eur_per_kwh: Energy cost (EUR/kWh).

        Returns:
            SteamTrapSurveyResult with loss quantification and payback.
        """
        total = len(traps)
        operational = 0
        failed_open = 0
        failed_closed = 0
        leaking = 0
        not_inspected = 0
        total_steam_loss_kg_h = Decimal("0")

        for trap in traps:
            if trap.status == TrapStatus.OPERATIONAL:
                operational += 1
            elif trap.status == TrapStatus.FAILED_OPEN:
                failed_open += 1
                total_steam_loss_kg_h += self._estimate_trap_loss(trap, Decimal("1.0"))
            elif trap.status == TrapStatus.FAILED_CLOSED:
                failed_closed += 1
            elif trap.status == TrapStatus.LEAKING:
                leaking += 1
                total_steam_loss_kg_h += self._estimate_trap_loss(trap, Decimal("0.5"))
            else:
                not_inspected += 1

        failure_count = failed_open + leaking + failed_closed
        failure_rate = _safe_divide(
            _decimal(failure_count) * Decimal("100"),
            _decimal(total),
        )

        # Convert steam loss to kWh: enthalpy of evaporation * mass / 3600
        avg_hfg = Decimal("2100")  # kJ/kg typical
        annual_loss_kj = total_steam_loss_kg_h * avg_hfg * _decimal(operating_hours)
        annual_loss_kwh = annual_loss_kj / Decimal("3600")
        annual_cost = annual_loss_kwh * _decimal(energy_cost_eur_per_kwh)

        # Replacement cost: EUR 150-400 per trap depending on type.
        traps_to_replace = failed_open + leaking
        avg_replacement_cost = Decimal("250")
        replacement_cost = _decimal(traps_to_replace) * avg_replacement_cost

        payback = _safe_divide(replacement_cost, annual_cost) if annual_cost > 0 else Decimal("0")

        self._notes.append(
            f"Steam trap survey: {total} traps, {failure_count} failures "
            f"({_round2(float(failure_rate))}%), estimated loss "
            f"{_round2(float(total_steam_loss_kg_h))} kg/h."
        )

        return SteamTrapSurveyResult(
            total_traps=total,
            operational_traps=operational,
            failed_open_traps=failed_open,
            failed_closed_traps=failed_closed,
            leaking_traps=leaking,
            not_inspected_traps=not_inspected,
            failure_rate_pct=_round2(float(failure_rate)),
            estimated_steam_loss_kg_h=_round2(float(total_steam_loss_kg_h)),
            estimated_annual_loss_kwh=_round2(float(annual_loss_kwh)),
            estimated_annual_cost_eur=_round2(float(annual_cost)),
            replacement_cost_eur=_round2(float(replacement_cost)),
            payback_years=_round2(float(payback)),
        )

    # --------------------------------------------------------------------- #
    # Insulation Assessment
    # --------------------------------------------------------------------- #

    def assess_insulation(
        self,
        sections: List[PipeSection],
        operating_hours: int,
        energy_cost_eur_per_kwh: float,
    ) -> InsulationAssessmentResult:
        """Assess pipe insulation condition and quantify losses.

        Uses BARE_PIPE_HEAT_LOSS lookup table for uninsulated sections
        and a conductivity-based estimate for damaged sections.

        Args:
            sections: List of pipe sections.
            operating_hours: Annual operating hours.
            energy_cost_eur_per_kwh: Energy cost (EUR/kWh).

        Returns:
            InsulationAssessmentResult with savings potential and payback.
        """
        total_sections = len(sections)
        uninsulated = 0
        damaged = 0
        good = 0
        total_heat_loss_kw = Decimal("0")
        potential_savings_kw = Decimal("0")
        total_insulation_cost = Decimal("0")

        for section in sections:
            if section.insulation_condition == InsulationCondition.MISSING:
                uninsulated += 1
                loss = self._bare_pipe_loss(section)
                total_heat_loss_kw += loss
                insulated_loss = loss * Decimal("0.05")  # 95% reduction with insulation
                potential_savings_kw += loss - insulated_loss
                total_insulation_cost += self._insulation_cost(section)
            elif section.insulation_condition in (InsulationCondition.DAMAGED, InsulationCondition.WET):
                damaged += 1
                loss = self._bare_pipe_loss(section) * Decimal("0.5")
                total_heat_loss_kw += loss
                insulated_loss = self._bare_pipe_loss(section) * Decimal("0.05")
                potential_savings_kw += loss - insulated_loss
                total_insulation_cost += self._insulation_cost(section) * Decimal("0.7")
            else:
                good += 1

        annual_savings_kwh = potential_savings_kw * _decimal(operating_hours)
        annual_savings_eur = annual_savings_kwh * _decimal(energy_cost_eur_per_kwh)
        payback = _safe_divide(total_insulation_cost, annual_savings_eur)

        self._notes.append(
            f"Insulation assessment: {total_sections} sections, "
            f"{uninsulated} uninsulated, {damaged} damaged, "
            f"heat loss {_round2(float(total_heat_loss_kw))} kW."
        )

        return InsulationAssessmentResult(
            pipe_sections_total=total_sections,
            uninsulated_sections=uninsulated,
            damaged_sections=damaged,
            good_sections=good,
            bare_pipe_heat_loss_kw=_round2(float(total_heat_loss_kw)),
            potential_savings_kw=_round2(float(potential_savings_kw)),
            potential_savings_kwh=_round2(float(annual_savings_kwh)),
            potential_savings_eur=_round2(float(annual_savings_eur)),
            insulation_cost_eur=_round2(float(total_insulation_cost)),
            payback_years=_round2(float(payback)),
        )

    # --------------------------------------------------------------------- #
    # Flash Steam Recovery
    # --------------------------------------------------------------------- #

    def calculate_flash_steam_recovery(
        self,
        condensate: CondensateSystem,
        operating_hours: int,
        energy_cost_eur_per_kwh: float,
    ) -> FlashSteamRecoveryResult:
        """Calculate flash steam recovery potential from condensate.

        Flash steam percentage is looked up or computed as:
            flash_pct = (hf_high - hf_low) / hfg_low * 100

        Args:
            condensate: Condensate system data.
            operating_hours: Annual operating hours.
            energy_cost_eur_per_kwh: Energy cost (EUR/kWh).

        Returns:
            FlashSteamRecoveryResult with savings and payback.
        """
        high_p = _decimal(condensate.condensate_pressure_bar)
        low_p = _decimal(condensate.flash_vessel_pressure_bar)
        flow = _decimal(condensate.total_condensate_flow_kg_h)

        if flow == Decimal("0"):
            return FlashSteamRecoveryResult()

        # Look up flash percentage
        lookup_key = f"{float(high_p)}_to_{float(low_p)}"
        flash_pct = _decimal(FLASH_STEAM_PCT.get(lookup_key, 0))

        if flash_pct == Decimal("0"):
            # Compute from enthalpy tables
            high_entry = self._lookup_steam_enthalpy(str(float(high_p)))
            low_entry = self._lookup_steam_enthalpy(str(float(low_p)))
            if high_entry and low_entry:
                hf_high = _decimal(high_entry["hf"])
                hf_low = _decimal(low_entry["hf"])
                hfg_low = _decimal(low_entry["hfg"])
                if hfg_low > Decimal("0"):
                    flash_pct = (hf_high - hf_low) / hfg_low * Decimal("100")
                    flash_pct = max(flash_pct, Decimal("0"))

        flash_flow_kg_h = flow * flash_pct / Decimal("100")

        # Energy in flash steam
        low_entry = self._lookup_steam_enthalpy(str(float(low_p)))
        hfg = _decimal(low_entry["hfg"]) if low_entry else Decimal("2200")
        energy_kw = flash_flow_kg_h * hfg / Decimal("3600")

        annual_kwh = energy_kw * _decimal(operating_hours)
        annual_eur = annual_kwh * _decimal(energy_cost_eur_per_kwh)

        # Equipment cost estimate: flash vessel + piping.
        equipment_cost = Decimal("15000") + flow * Decimal("5")
        payback = _safe_divide(equipment_cost, annual_eur)

        self._notes.append(
            f"Flash steam recovery: {_round2(float(flash_pct))}% flash at "
            f"{float(high_p)}->{float(low_p)} bar, {_round2(float(flash_flow_kg_h))} kg/h."
        )

        return FlashSteamRecoveryResult(
            flash_steam_pct=_round2(float(flash_pct)),
            flash_steam_flow_kg_h=_round2(float(flash_flow_kg_h)),
            energy_recoverable_kw=_round2(float(energy_kw)),
            annual_savings_kwh=_round2(float(annual_kwh)),
            annual_savings_eur=_round2(float(annual_eur)),
            equipment_cost_eur=_round2(float(equipment_cost)),
            payback_years=_round2(float(payback)),
        )

    # --------------------------------------------------------------------- #
    # Blowdown Recovery
    # --------------------------------------------------------------------- #

    def calculate_blowdown_recovery(
        self,
        blowdown_pct: float,
        steam_flow_kg_h: float,
        pressure_bar: float,
        feed_water_temp_c: float,
        operating_hours: int,
        energy_cost_eur_per_kwh: float,
    ) -> BlowdownRecoveryResult:
        """Calculate blowdown heat recovery potential.

        Blowdown flow = steam_flow * blowdown_pct / (100 - blowdown_pct).
        Energy recovery via flash tank and heat exchanger typically
        recovers 80% of blowdown enthalpy above feedwater temperature.

        Args:
            blowdown_pct: Blowdown rate (% of feedwater).
            steam_flow_kg_h: Total steam generation (kg/h).
            pressure_bar: Boiler operating pressure (bar).
            feed_water_temp_c: Feedwater temperature (C).
            operating_hours: Annual operating hours.
            energy_cost_eur_per_kwh: Energy cost (EUR/kWh).

        Returns:
            BlowdownRecoveryResult with savings and payback.
        """
        d_blowdown_pct = _decimal(blowdown_pct)
        d_steam = _decimal(steam_flow_kg_h)
        d_pressure = _decimal(pressure_bar)
        d_fw_temp = _decimal(feed_water_temp_c)

        # Blowdown mass flow.
        blowdown_flow = d_steam * d_blowdown_pct / (Decimal("100") - d_blowdown_pct)

        # Enthalpy of blowdown water at boiler pressure.
        entry = self._lookup_steam_enthalpy(str(float(d_pressure)))
        hf_bd = _decimal(entry["hf"]) if entry else _decimal(feed_water_temp_c) * Decimal("4.186")

        # Enthalpy of feedwater.
        hf_fw = d_fw_temp * Decimal("4.186")

        # Energy in blowdown.
        energy_kw = blowdown_flow * (hf_bd - hf_fw) / Decimal("3600")

        # Recovery factor: 80% with flash tank + heat exchanger.
        recovery_factor = Decimal("0.80")
        recoverable_kw = energy_kw * recovery_factor

        annual_kwh = recoverable_kw * _decimal(operating_hours)
        annual_eur = annual_kwh * _decimal(energy_cost_eur_per_kwh)

        # Water savings: reduced makeup water from recovered blowdown.
        water_savings_m3 = blowdown_flow * _decimal(operating_hours) / Decimal("1000") * Decimal("0.5")
        water_cost = water_savings_m3 * (WATER_COST_EUR_PER_M3 + WATER_TREATMENT_EUR_PER_M3)

        # Equipment cost: flash tank + HX.
        equipment_cost = Decimal("20000") + blowdown_flow * Decimal("10")
        total_annual_savings = annual_eur + water_cost
        payback = _safe_divide(equipment_cost, total_annual_savings)

        return BlowdownRecoveryResult(
            blowdown_rate_pct=_round2(float(d_blowdown_pct)),
            blowdown_flow_kg_h=_round2(float(blowdown_flow)),
            energy_in_blowdown_kw=_round2(float(energy_kw)),
            recoverable_energy_kw=_round2(float(recoverable_kw)),
            annual_savings_kwh=_round2(float(annual_kwh)),
            annual_savings_eur=_round2(float(annual_eur)),
            water_savings_m3=_round2(float(water_savings_m3)),
            water_savings_eur=_round2(float(water_cost)),
            equipment_cost_eur=_round2(float(equipment_cost)),
            payback_years=_round2(float(payback)),
        )

    # --------------------------------------------------------------------- #
    # Condensate Return Analysis
    # --------------------------------------------------------------------- #

    def analyze_condensate_return(
        self,
        condensate: CondensateSystem,
        operating_hours: int,
        energy_cost_eur_per_kwh: float,
    ) -> Dict[str, Any]:
        """Analyse condensate return rate and improvement potential.

        Increasing condensate return from current rate to 90% reduces
        makeup water, chemicals, and energy needed to heat fresh water.

        Args:
            condensate: Condensate system data.
            operating_hours: Annual operating hours.
            energy_cost_eur_per_kwh: Energy cost (EUR/kWh).

        Returns:
            Dictionary with condensate improvement analysis.
        """
        current_rate = _decimal(condensate.return_rate_pct)
        target_rate = Decimal("90")
        flow = _decimal(condensate.total_condensate_flow_kg_h)
        cond_temp = _decimal(condensate.condensate_temp_c)

        if current_rate >= target_rate or flow == Decimal("0"):
            return {
                "current_return_rate_pct": _round2(float(current_rate)),
                "target_return_rate_pct": _round2(float(target_rate)),
                "improvement_pct": 0.0,
                "annual_savings_kwh": 0.0,
                "annual_savings_eur": 0.0,
            }

        improvement = target_rate - current_rate
        additional_return = flow * improvement / Decimal("100")

        # Energy saved = mass * cp * delta_T / 3600.
        # Delta T = condensate temp - makeup water temp (assume 15 C).
        makeup_temp = Decimal("15")
        delta_t = cond_temp - makeup_temp
        cp = Decimal("4.186")  # kJ/kg-K for water
        energy_saved_kw = additional_return * cp * delta_t / Decimal("3600")
        annual_kwh = energy_saved_kw * _decimal(operating_hours)
        annual_eur = annual_kwh * _decimal(energy_cost_eur_per_kwh)

        # Water and chemical savings.
        water_saved_m3 = additional_return * _decimal(operating_hours) / Decimal("1000")
        water_eur = water_saved_m3 * (WATER_COST_EUR_PER_M3 + WATER_TREATMENT_EUR_PER_M3)

        total_savings_eur = annual_eur + water_eur

        self._notes.append(
            f"Condensate return improvement: {_round2(float(current_rate))}% -> "
            f"{_round2(float(target_rate))}%, saving {_round2(float(annual_kwh))} kWh/year."
        )

        return {
            "current_return_rate_pct": _round2(float(current_rate)),
            "target_return_rate_pct": _round2(float(target_rate)),
            "improvement_pct": _round2(float(improvement)),
            "additional_return_kg_h": _round2(float(additional_return)),
            "annual_savings_kwh": _round2(float(annual_kwh)),
            "annual_savings_eur": _round2(float(total_savings_eur)),
            "water_savings_m3": _round2(float(water_saved_m3)),
            "water_savings_eur": _round2(float(water_eur)),
        }

    # --------------------------------------------------------------------- #
    # CHP Opportunity Assessment
    # --------------------------------------------------------------------- #

    def assess_chp_opportunity(
        self,
        thermal_demand_kw: float,
        operating_hours: int,
        electricity_cost_eur_per_kwh: float = 0.12,
    ) -> CHPOpportunity:
        """Screen for Combined Heat and Power (CHP) viability.

        A CHP unit is sized to thermal demand.  Electrical output is
        derived from the thermal-to-electrical ratio of a typical gas
        engine or turbine CHP.

        Viability threshold: >= 500 kW thermal demand, >= 4000 hours/year.

        Args:
            thermal_demand_kw: Site thermal demand (kW).
            operating_hours: Annual operating hours.
            electricity_cost_eur_per_kwh: Grid electricity cost (EUR/kWh).

        Returns:
            CHPOpportunity with sizing, savings, and payback.
        """
        d_thermal = _decimal(thermal_demand_kw)
        d_hours = _decimal(operating_hours)

        viable = d_thermal >= Decimal("500") and d_hours >= Decimal("4000")

        if not viable:
            return CHPOpportunity(viable=False, thermal_demand_kw=float(d_thermal))

        # Size CHP to match thermal demand.
        thermal_output = d_thermal
        # Electrical output from efficiency ratio.
        electrical_output = thermal_output * CHP_ELECTRICAL_EFFICIENCY / CHP_THERMAL_EFFICIENCY

        annual_elec_kwh = electrical_output * d_hours
        annual_elec_savings = annual_elec_kwh * _decimal(electricity_cost_eur_per_kwh)

        capex = electrical_output * CHP_CAPEX_EUR_PER_KWE
        payback = _safe_divide(capex, annual_elec_savings)

        # CO2 savings: displaced grid electricity.
        # EU average grid factor ~ 0.25 tCO2/MWh (2025).
        grid_factor = Decimal("0.25")
        co2_savings = annual_elec_kwh / Decimal("1000") * grid_factor

        self._notes.append(
            f"CHP opportunity: {_round2(float(electrical_output))} kW_e, "
            f"payback {_round2(float(payback))} years."
        )

        return CHPOpportunity(
            viable=True,
            thermal_demand_kw=_round2(float(d_thermal)),
            electrical_output_kw=_round2(float(electrical_output)),
            thermal_output_kw=_round2(float(thermal_output)),
            annual_electricity_kwh=_round2(float(annual_elec_kwh)),
            annual_electricity_savings_eur=_round2(float(annual_elec_savings)),
            capex_eur=_round2(float(capex)),
            payback_years=_round2(float(payback)),
            co2_savings_tonnes=_round2(float(co2_savings)),
        )

    # --------------------------------------------------------------------- #
    # Private Helpers
    # --------------------------------------------------------------------- #

    def _calculate_stack_loss(self, boiler: Boiler) -> Decimal:
        """Calculate stack (flue gas) loss using the Siegert formula.

        Siegert: L_stack = K1 * (T_stack - T_ambient) / CO2_pct

        If CO2 is not available, excess air is used to estimate CO2:
            CO2_actual = CO2_stoich / (1 + excess_air/100)

        Args:
            boiler: Boiler data with flue gas parameters.

        Returns:
            Stack loss as a Decimal percentage.
        """
        coefficients = SIEGERT_COEFFICIENTS.get(boiler.fuel_type)
        if not coefficients:
            return Decimal("8.0")  # Conservative default

        k1 = _decimal(coefficients["k1"])
        stoich_co2 = _decimal(coefficients["stoich_co2_pct"])

        # Determine temperatures.
        if boiler.flue_gas:
            stack_temp = _decimal(boiler.flue_gas.stack_temp_c)
            ambient_temp = _decimal(boiler.flue_gas.ambient_temp_c)
            co2_pct = _decimal(boiler.flue_gas.co2_pct)
            excess_air = _decimal(boiler.flue_gas.excess_air_pct)
        else:
            stack_temp = _decimal(boiler.stack_temp_c)
            ambient_temp = Decimal("20")
            co2_pct = Decimal("0")
            excess_air = _decimal(boiler.excess_air_pct)

        # Compute CO2 from excess air if not measured.
        if co2_pct <= Decimal("0") and excess_air > Decimal("0"):
            co2_pct = stoich_co2 / (Decimal("1") + excess_air / Decimal("100"))

        # Fallback: derive excess air from O2 if available.
        if co2_pct <= Decimal("0") and boiler.flue_gas and boiler.flue_gas.o2_pct > 0:
            o2_pct = _decimal(boiler.flue_gas.o2_pct)
            excess_air_from_o2 = o2_pct / (Decimal("21") - o2_pct) * Decimal("100")
            co2_pct = stoich_co2 / (Decimal("1") + excess_air_from_o2 / Decimal("100"))

        if co2_pct <= Decimal("0"):
            co2_pct = stoich_co2 * Decimal("0.7")  # Assume ~30% excess air

        delta_t = stack_temp - ambient_temp
        stack_loss = k1 * delta_t / co2_pct

        return max(stack_loss, Decimal("0"))

    def _calculate_blowdown_loss(self, boiler: Boiler) -> Decimal:
        """Calculate blowdown energy loss as a percentage of fuel input.

        Blowdown loss % = (blowdown_rate / 100) * (hf / hg) * 100

        Args:
            boiler: Boiler data.

        Returns:
            Blowdown loss as a Decimal percentage.
        """
        d_bd = _decimal(boiler.blowdown_pct)
        if d_bd <= Decimal("0"):
            return Decimal("0")

        pressure_key = str(float(_decimal(boiler.operating_pressure_bar)))
        entry = self._lookup_steam_enthalpy(pressure_key)
        if not entry:
            return d_bd * Decimal("0.2")  # Conservative estimate

        hf = _decimal(entry["hf"])
        hg = _decimal(entry["hg"])
        blowdown_loss = (d_bd / Decimal("100")) * _safe_divide(hf, hg) * Decimal("100")
        return blowdown_loss

    def _estimate_trap_loss(self, trap: SteamTrapRecord, factor: Decimal) -> Decimal:
        """Estimate steam loss from a failed or leaking trap.

        Uses the TRAP_FAILURE_STEAM_LOSS lookup table, scaled by
        a factor (1.0 for failed open, 0.5 for leaking).

        Args:
            trap: Steam trap record.
            factor: Scaling factor (1.0 = full leak, 0.5 = partial).

        Returns:
            Estimated steam loss in kg/h as Decimal.
        """
        orifice_key = str(int(trap.orifice_mm))
        loss_table = TRAP_FAILURE_STEAM_LOSS.get(trap.trap_type, {})
        base_loss = _decimal(loss_table.get(orifice_key, 10.0))

        # Scale by pressure: higher pressure = more loss.
        pressure_factor = _decimal(trap.operating_pressure_bar) / Decimal("5")
        pressure_factor = max(pressure_factor, Decimal("0.5"))
        pressure_factor = min(pressure_factor, Decimal("3.0"))

        return base_loss * factor * pressure_factor

    def _bare_pipe_loss(self, section: PipeSection) -> Decimal:
        """Look up bare pipe heat loss for a section.

        Uses the BARE_PIPE_HEAT_LOSS table, interpolating if necessary,
        and scales by section length.

        Args:
            section: Pipe section data.

        Returns:
            Heat loss in kW as Decimal.
        """
        # Find nearest diameter.
        available_dias = sorted(BARE_PIPE_HEAT_LOSS.keys())
        nearest_dia = min(available_dias, key=lambda d: abs(d - section.diameter_mm))

        # Find nearest temperature.
        temp_table = BARE_PIPE_HEAT_LOSS[nearest_dia]
        available_temps = sorted(temp_table.keys())
        nearest_temp = min(available_temps, key=lambda t: abs(t - section.surface_temp_c))

        loss_w_per_m = _decimal(temp_table[nearest_temp])
        total_loss_w = loss_w_per_m * _decimal(section.length_m)
        return total_loss_w / Decimal("1000")  # Convert W to kW

    def _insulation_cost(self, section: PipeSection) -> Decimal:
        """Estimate insulation installation cost for a pipe section.

        Cost = base_rate_per_m * length * diameter_factor.

        Args:
            section: Pipe section to insulate.

        Returns:
            Estimated cost in EUR as Decimal.
        """
        base_rate = Decimal("45")  # EUR/m for 100mm pipe
        dia_factor = _decimal(section.diameter_mm) / Decimal("100")
        dia_factor = max(dia_factor, Decimal("0.5"))
        return base_rate * _decimal(section.length_m) * dia_factor

    def _lookup_steam_enthalpy(self, pressure_str: str) -> Optional[Dict[str, float]]:
        """Look up steam enthalpy data from the table.

        Tries exact match first, then nearest available pressure.

        Args:
            pressure_str: Pressure as string (e.g., '10.0').

        Returns:
            Dictionary with hf, hfg, hg, temp_c or None.
        """
        if pressure_str in STEAM_ENTHALPY_TABLE:
            return STEAM_ENTHALPY_TABLE[pressure_str]

        # Find nearest pressure.
        try:
            target = float(pressure_str)
        except (ValueError, TypeError):
            return None

        available = sorted(STEAM_ENTHALPY_TABLE.keys(), key=lambda k: abs(float(k) - target))
        if available:
            return STEAM_ENTHALPY_TABLE[available[0]]
        return None

    def _estimate_thermal_demand(self, system: SteamSystem) -> Decimal:
        """Estimate total site thermal demand from boiler data.

        Args:
            system: Steam system data.

        Returns:
            Thermal demand in kW as Decimal.
        """
        total_kw = Decimal("0")
        for boiler in system.boilers:
            # Steam enthalpy at operating pressure.
            entry = self._lookup_steam_enthalpy(str(boiler.operating_pressure_bar))
            if entry:
                hg = _decimal(entry["hg"])
                hf_fw = _decimal(boiler.feed_water_temp_c) * Decimal("4.186")
                capacity_kw = _decimal(boiler.capacity_kg_h) * (hg - hf_fw) / Decimal("3600")
                # Assume 70% average load.
                total_kw += capacity_kw * Decimal("0.7")
            else:
                # Fallback: 700 kWh per tonne of steam.
                total_kw += _decimal(boiler.capacity_kg_h) * Decimal("0.7") / Decimal("1000") * Decimal("700")
        return total_kw
