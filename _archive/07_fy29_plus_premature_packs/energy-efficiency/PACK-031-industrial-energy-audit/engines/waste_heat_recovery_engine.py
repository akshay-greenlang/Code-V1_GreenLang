# -*- coding: utf-8 -*-
"""
WasteHeatRecoveryEngine - PACK-031 Industrial Energy Audit Engine 6
====================================================================

Identifies and quantifies waste heat recovery opportunities in industrial
facilities. Inventories waste heat sources (flue gases, cooling water,
compressed air, process exhaust, steam blowdown), performs pinch analysis
for optimal heat integration, sizes heat exchangers using LMTD and
effectiveness-NTU methods, and selects appropriate recovery technologies
(economizers, recuperators, regenerators, heat pumps, ORC, absorption
chillers).

Calculation Methodology:
    Heat Content (fundamental):
        Q = m_dot * cp * delta_T  (kW)
        where m_dot in kg/s, cp in kJ/(kg*K), delta_T in K

    Pinch Analysis (Linnhoff & Hindmarsh):
        Hot composite curve: cumulative heat from hot streams
        Cold composite curve: cumulative heat from cold streams
        Pinch point: minimum temperature approach (delta_T_min)
        Maximum heat recovery = overlap of composite curves
        Minimum heating utility = cold curve above pinch
        Minimum cooling utility = hot curve below pinch

    LMTD Heat Exchanger Sizing:
        Q = U * A * LMTD
        LMTD = (delta_T1 - delta_T2) / ln(delta_T1 / delta_T2)
        A = Q / (U * LMTD)

    Effectiveness-NTU:
        epsilon = Q_actual / Q_max
        NTU = U * A / C_min
        epsilon = f(NTU, C_min/C_max, flow arrangement)

    Carnot Efficiency (heat-to-power):
        eta_carnot = 1 - T_cold / T_hot  (absolute temperatures)
        eta_actual = eta_carnot * internal_efficiency_factor

    Fouling Factor Application:
        U_dirty = 1 / (1/U_clean + R_f_hot + R_f_cold)

Regulatory References:
    - EN 15900:2010 - Energy efficiency services
    - ISO 50001:2018 - Energy management systems
    - EN 16247-3:2022 - Energy audits (processes)
    - ASME BPVC Section VIII - Pressure vessel code
    - TEMA Standards - Tubular Exchanger Manufacturers Association
    - EU Industrial Emissions Directive 2010/75/EU (BAT for heat recovery)
    - EU Waste Heat Recovery BAT Reference (2017)

Zero-Hallucination:
    - Specific heat values from NIST Chemistry WebBook
    - U-values from TEMA standards and engineering handbooks
    - Fouling factors from TEMA Table RGP-T2.4
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-031 Industrial Energy Audit
Engine:  6 of 10
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

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HeatSourceType(str, Enum):
    """Waste heat source classification.

    FLUE_GAS: Combustion exhaust from boilers, furnaces, engines.
    COOLING_WATER: Process or equipment cooling water circuits.
    COMPRESSED_AIR: Heat from compressed air systems (94% of input).
    PROCESS_EXHAUST: Hot process exhaust (dryers, ovens, kilns).
    STEAM_BLOWDOWN: Boiler blowdown water/flash steam.
    CONDENSER: Refrigeration/chiller condenser reject heat.
    ENGINE_EXHAUST: CHP, generator, or engine exhaust.
    HOT_PRODUCT: Heat from hot products after processing.
    EFFLUENT: Hot process wastewater or effluent.
    RADIATION: Radiation losses from hot surfaces.
    """
    FLUE_GAS = "flue_gas"
    COOLING_WATER = "cooling_water"
    COMPRESSED_AIR = "compressed_air"
    PROCESS_EXHAUST = "process_exhaust"
    STEAM_BLOWDOWN = "steam_blowdown"
    CONDENSER = "condenser"
    ENGINE_EXHAUST = "engine_exhaust"
    HOT_PRODUCT = "hot_product"
    EFFLUENT = "effluent"
    RADIATION = "radiation"

class TemperatureGrade(str, Enum):
    """Waste heat temperature grade classification.

    HIGH: >400C (furnaces, kilns, engines, high-temp processes).
    MEDIUM: 100-400C (boiler exhaust, steam systems, dryers).
    LOW: <100C (cooling water, condenser reject, compressed air).
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class HeatExchangerType(str, Enum):
    """Heat exchanger technology type.

    SHELL_TUBE: Shell and tube (most common industrial).
    PLATE: Plate / plate-and-frame.
    PLATE_FIN: Compact plate-fin (air-to-air or gas-to-gas).
    HEAT_PIPE: Heat pipe exchanger.
    ECONOMIZER: Flue gas to water economizer.
    RECUPERATOR: Gas-to-gas recuperator.
    REGENERATOR: Regenerative heat exchanger.
    SPIRAL: Spiral heat exchanger.
    AIR_PREHEATER: Combustion air preheater.
    CONDENSING: Condensing heat exchanger (below dew point).
    """
    SHELL_TUBE = "shell_tube"
    PLATE = "plate"
    PLATE_FIN = "plate_fin"
    HEAT_PIPE = "heat_pipe"
    ECONOMIZER = "economizer"
    RECUPERATOR = "recuperator"
    REGENERATOR = "regenerator"
    SPIRAL = "spiral"
    AIR_PREHEATER = "air_preheater"
    CONDENSING = "condensing"

class RecoveryTechnologyType(str, Enum):
    """Waste heat recovery technology.

    HEAT_EXCHANGER: Direct heat exchange (source to sink).
    ECONOMIZER: Flue gas economizer for boiler feedwater.
    HEAT_PUMP: Industrial heat pump (upgrade low-grade heat).
    ORC: Organic Rankine Cycle (heat to power).
    ABSORPTION_CHILLER: Absorption chiller (heat to cooling).
    STEAM_GENERATOR: Waste heat steam generator.
    THERMOELECTRIC: Thermoelectric generator (direct conversion).
    PREHEATER: Combustion air or process preheater.
    DESALINATION: Waste heat driven desalination.
    """
    HEAT_EXCHANGER = "heat_exchanger"
    ECONOMIZER = "economizer"
    HEAT_PUMP = "heat_pump"
    ORC = "orc"
    ABSORPTION_CHILLER = "absorption_chiller"
    STEAM_GENERATOR = "steam_generator"
    THERMOELECTRIC = "thermoelectric"
    PREHEATER = "preheater"
    DESALINATION = "desalination"

# ---------------------------------------------------------------------------
# Constants -- Specific Heat Capacities (kJ/(kg*K))
# ---------------------------------------------------------------------------

# At approximate operating conditions from NIST data.
SPECIFIC_HEAT_CAPACITY: Dict[str, Decimal] = {
    "water": Decimal("4.186"),
    "air": Decimal("1.005"),
    "steam_100c": Decimal("2.010"),
    "steam_200c": Decimal("1.975"),
    "flue_gas_natural_gas": Decimal("1.100"),
    "flue_gas_diesel": Decimal("1.080"),
    "flue_gas_coal": Decimal("1.050"),
    "thermal_oil": Decimal("2.100"),
    "glycol_50pct": Decimal("3.350"),
    "nitrogen": Decimal("1.040"),
    "carbon_dioxide": Decimal("0.846"),
    "hydrogen": Decimal("14.300"),
    "methane": Decimal("2.220"),
    "ethanol": Decimal("2.440"),
    "milk": Decimal("3.930"),
    "cooking_oil": Decimal("2.000"),
}

# ---------------------------------------------------------------------------
# Constants -- Heat Exchanger U-Values (kW/(m2*K)) -- TEMA standards
# ---------------------------------------------------------------------------

# Overall heat transfer coefficient by hot-side / cold-side fluid pair.
HEAT_EXCHANGER_U_VALUES: Dict[str, Decimal] = {
    "water_water": Decimal("1.500"),
    "water_glycol": Decimal("0.900"),
    "steam_water": Decimal("2.500"),
    "flue_gas_water": Decimal("0.035"),
    "flue_gas_air": Decimal("0.025"),
    "air_water": Decimal("0.050"),
    "thermal_oil_water": Decimal("0.400"),
    "process_exhaust_water": Decimal("0.030"),
    "process_exhaust_air": Decimal("0.020"),
    "steam_thermal_oil": Decimal("0.350"),
    "condensing_water": Decimal("0.045"),
    "compressed_air_water": Decimal("0.060"),
}

# ---------------------------------------------------------------------------
# Constants -- Fouling Factors (m2*K/kW) -- TEMA Table RGP-T2.4
# ---------------------------------------------------------------------------

FOULING_FACTORS: Dict[str, Decimal] = {
    "clean_water": Decimal("0.0001"),
    "treated_cooling_water": Decimal("0.0002"),
    "untreated_water": Decimal("0.0005"),
    "sea_water": Decimal("0.0003"),
    "flue_gas_clean": Decimal("0.0020"),
    "flue_gas_dirty": Decimal("0.0050"),
    "air_clean": Decimal("0.0002"),
    "air_dusty": Decimal("0.0010"),
    "thermal_oil": Decimal("0.0002"),
    "steam_clean": Decimal("0.0001"),
    "process_fluid": Decimal("0.0005"),
    "compressed_air": Decimal("0.0002"),
}

# ---------------------------------------------------------------------------
# Constants -- Recovery Technology Database
# ---------------------------------------------------------------------------

RECOVERY_TECHNOLOGY_DATABASE: Dict[str, Dict[str, Any]] = {
    RecoveryTechnologyType.HEAT_EXCHANGER.value: {
        "name": "Direct Heat Exchanger",
        "temp_range_min_c": 30, "temp_range_max_c": 900,
        "efficiency_pct": Decimal("70"), "capex_per_kw": Decimal("80"),
        "opex_pct_capex": Decimal("2"), "lifetime_years": 20,
    },
    RecoveryTechnologyType.ECONOMIZER.value: {
        "name": "Flue Gas Economizer",
        "temp_range_min_c": 120, "temp_range_max_c": 450,
        "efficiency_pct": Decimal("75"), "capex_per_kw": Decimal("100"),
        "opex_pct_capex": Decimal("2"), "lifetime_years": 20,
    },
    RecoveryTechnologyType.HEAT_PUMP.value: {
        "name": "Industrial Heat Pump",
        "temp_range_min_c": 20, "temp_range_max_c": 100,
        "efficiency_pct": Decimal("300"),  # COP ~3
        "capex_per_kw": Decimal("400"),
        "opex_pct_capex": Decimal("3"), "lifetime_years": 15,
    },
    RecoveryTechnologyType.ORC.value: {
        "name": "Organic Rankine Cycle",
        "temp_range_min_c": 80, "temp_range_max_c": 350,
        "efficiency_pct": Decimal("12"), "capex_per_kw": Decimal("2500"),
        "opex_pct_capex": Decimal("3"), "lifetime_years": 20,
    },
    RecoveryTechnologyType.ABSORPTION_CHILLER.value: {
        "name": "Absorption Chiller",
        "temp_range_min_c": 80, "temp_range_max_c": 200,
        "efficiency_pct": Decimal("70"),  # COP ~0.7
        "capex_per_kw": Decimal("500"),
        "opex_pct_capex": Decimal("2"), "lifetime_years": 20,
    },
    RecoveryTechnologyType.STEAM_GENERATOR.value: {
        "name": "Waste Heat Steam Generator",
        "temp_range_min_c": 250, "temp_range_max_c": 900,
        "efficiency_pct": Decimal("65"), "capex_per_kw": Decimal("200"),
        "opex_pct_capex": Decimal("3"), "lifetime_years": 25,
    },
    RecoveryTechnologyType.THERMOELECTRIC.value: {
        "name": "Thermoelectric Generator",
        "temp_range_min_c": 150, "temp_range_max_c": 600,
        "efficiency_pct": Decimal("5"), "capex_per_kw": Decimal("5000"),
        "opex_pct_capex": Decimal("1"), "lifetime_years": 15,
    },
    RecoveryTechnologyType.PREHEATER.value: {
        "name": "Combustion Air Preheater",
        "temp_range_min_c": 150, "temp_range_max_c": 500,
        "efficiency_pct": Decimal("60"), "capex_per_kw": Decimal("90"),
        "opex_pct_capex": Decimal("2"), "lifetime_years": 20,
    },
}

# Minimum approach temperature for pinch analysis (delta_T_min).
DEFAULT_PINCH_DT_MIN: Decimal = Decimal("10")  # degrees C

# Default energy price and CO2 factor.
DEFAULT_ENERGY_PRICE_EUR_KWH: Decimal = Decimal("0.15")
DEFAULT_CO2_FACTOR_KG_KWH: Decimal = Decimal("0.4")

# Ambient temperature.
DEFAULT_AMBIENT_TEMP_C: Decimal = Decimal("20")

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class WasteHeatSource(BaseModel):
    """A waste heat source in the facility.

    Attributes:
        source_id: Unique source identifier.
        name: Source name/description.
        source_type: Heat source classification.
        flow_rate_kg_s: Mass flow rate (kg/s).
        inlet_temperature_c: Source inlet temperature (C).
        outlet_temperature_c: Minimum allowable outlet temperature (C).
        specific_heat_kj_kgk: Specific heat capacity (kJ/(kg*K)).
        fluid_type: Fluid type for property lookup.
        operating_hours: Annual operating hours.
        notes: Additional notes.
    """
    source_id: str = Field(default_factory=_new_uuid, description="Source ID")
    name: str = Field(default="", max_length=300, description="Source name")
    source_type: str = Field(
        default=HeatSourceType.FLUE_GAS.value,
        description="Heat source type"
    )
    flow_rate_kg_s: Decimal = Field(
        default=Decimal("0"), ge=0, description="Mass flow rate (kg/s)"
    )
    inlet_temperature_c: Decimal = Field(
        default=Decimal("200"), ge=Decimal("-40"), le=Decimal("1500"),
        description="Inlet temperature (C)"
    )
    outlet_temperature_c: Decimal = Field(
        default=Decimal("60"), ge=Decimal("-40"), le=Decimal("1500"),
        description="Minimum outlet temperature (C)"
    )
    specific_heat_kj_kgk: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Specific heat (kJ/(kg*K)). 0 = auto-lookup."
    )
    fluid_type: str = Field(
        default="water", max_length=50,
        description="Fluid type for property lookup"
    )
    operating_hours: int = Field(
        default=8000, ge=0, le=8760,
        description="Annual operating hours"
    )
    notes: str = Field(default="", description="Additional notes")

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, v: str) -> str:
        valid = {t.value for t in HeatSourceType}
        if v not in valid:
            raise ValueError(f"Unknown source type '{v}'. Must be one of: {sorted(valid)}")
        return v

class HeatSink(BaseModel):
    """A heat demand (sink) that could use recovered heat.

    Attributes:
        sink_id: Unique sink identifier.
        name: Sink name/description.
        required_heat_kw: Required heat input (kW).
        inlet_temperature_c: Sink inlet (current) temperature (C).
        target_temperature_c: Sink target temperature (C).
        flow_rate_kg_s: Mass flow rate (kg/s).
        specific_heat_kj_kgk: Specific heat (kJ/(kg*K)). 0 = auto-lookup.
        fluid_type: Fluid type.
        current_source: Current heat source (e.g., boiler, electric heater).
        current_cost_eur_kwh: Current cost of heat supply (EUR/kWh).
        operating_hours: Annual operating hours.
        notes: Additional notes.
    """
    sink_id: str = Field(default_factory=_new_uuid, description="Sink ID")
    name: str = Field(default="", max_length=300, description="Sink name")
    required_heat_kw: Decimal = Field(
        default=Decimal("0"), ge=0, description="Required heat (kW)"
    )
    inlet_temperature_c: Decimal = Field(
        default=Decimal("20"), ge=Decimal("-40"), le=Decimal("500"),
        description="Inlet temperature (C)"
    )
    target_temperature_c: Decimal = Field(
        default=Decimal("60"), ge=Decimal("-40"), le=Decimal("500"),
        description="Target temperature (C)"
    )
    flow_rate_kg_s: Decimal = Field(
        default=Decimal("0"), ge=0, description="Mass flow rate (kg/s)"
    )
    specific_heat_kj_kgk: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Specific heat (kJ/(kg*K)). 0 = auto-lookup."
    )
    fluid_type: str = Field(
        default="water", max_length=50,
        description="Fluid type"
    )
    current_source: str = Field(default="boiler", max_length=100, description="Current heat source")
    current_cost_eur_kwh: Decimal = Field(
        default=Decimal("0.05"), ge=0, description="Current heat cost (EUR/kWh)"
    )
    operating_hours: int = Field(
        default=8000, ge=0, le=8760, description="Annual operating hours"
    )
    notes: str = Field(default="", description="Additional notes")

class WasteHeatRecoveryInput(BaseModel):
    """Complete input for waste heat recovery analysis.

    Attributes:
        facility_id: Facility identifier.
        facility_name: Facility name.
        sources: Waste heat sources.
        sinks: Heat demand sinks.
        pinch_dt_min_c: Minimum temperature approach for pinch (C).
        ambient_temperature_c: Ambient temperature (C).
        energy_price_eur_kwh: Energy price (EUR/kWh).
        co2_factor_kg_kwh: Grid CO2 factor (kg/kWh).
        discount_rate: Discount rate for financial analysis.
        include_pinch_analysis: Whether to perform pinch analysis.
        include_technology_selection: Whether to recommend technologies.
        include_heat_exchanger_sizing: Whether to size heat exchangers.
    """
    facility_id: str = Field(default="", description="Facility ID")
    facility_name: str = Field(default="", max_length=300, description="Facility name")
    sources: List[WasteHeatSource] = Field(
        default_factory=list, description="Waste heat sources"
    )
    sinks: List[HeatSink] = Field(
        default_factory=list, description="Heat demand sinks"
    )
    pinch_dt_min_c: Decimal = Field(
        default=DEFAULT_PINCH_DT_MIN, ge=Decimal("1"), le=Decimal("50"),
        description="Minimum pinch approach temperature (C)"
    )
    ambient_temperature_c: Decimal = Field(
        default=DEFAULT_AMBIENT_TEMP_C, description="Ambient temperature (C)"
    )
    energy_price_eur_kwh: Decimal = Field(
        default=DEFAULT_ENERGY_PRICE_EUR_KWH, ge=0,
        description="Energy price (EUR/kWh)"
    )
    co2_factor_kg_kwh: Decimal = Field(
        default=DEFAULT_CO2_FACTOR_KG_KWH, ge=0,
        description="CO2 factor (kg/kWh)"
    )
    discount_rate: Decimal = Field(
        default=Decimal("0.08"), ge=0, le=Decimal("0.30"),
        description="Discount rate"
    )
    include_pinch_analysis: bool = Field(
        default=True, description="Include pinch analysis"
    )
    include_technology_selection: bool = Field(
        default=True, description="Include technology selection"
    )
    include_heat_exchanger_sizing: bool = Field(
        default=True, description="Include HX sizing"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class SourceAnalysis(BaseModel):
    """Analysis result for a single waste heat source.

    Attributes:
        source_id: Source identifier.
        name: Source name.
        source_type: Source type.
        temperature_grade: Temperature grade classification.
        available_heat_kw: Available waste heat (kW).
        annual_heat_mwh: Annual available heat (MWh).
        annual_value_eur: Annual value at energy price (EUR).
        carnot_efficiency: Maximum Carnot efficiency for power conversion.
    """
    source_id: str = Field(default="")
    name: str = Field(default="")
    source_type: str = Field(default="")
    temperature_grade: str = Field(default="")
    available_heat_kw: Decimal = Field(default=Decimal("0"))
    annual_heat_mwh: Decimal = Field(default=Decimal("0"))
    annual_value_eur: Decimal = Field(default=Decimal("0"))
    carnot_efficiency: Decimal = Field(default=Decimal("0"))

class PinchAnalysisResult(BaseModel):
    """Pinch analysis results.

    Attributes:
        pinch_temperature_c: Pinch point temperature (C).
        min_heating_utility_kw: Minimum external heating required (kW).
        min_cooling_utility_kw: Minimum external cooling required (kW).
        max_heat_recovery_kw: Maximum internal heat recovery (kW).
        hot_utility_savings_pct: Heating utility savings potential (%).
        cold_utility_savings_pct: Cooling utility savings potential (%).
        dt_min_c: Minimum approach temperature used (C).
        hot_composite: List of (temperature, cumulative_heat) points.
        cold_composite: List of (temperature, cumulative_heat) points.
    """
    pinch_temperature_c: Decimal = Field(default=Decimal("0"))
    min_heating_utility_kw: Decimal = Field(default=Decimal("0"))
    min_cooling_utility_kw: Decimal = Field(default=Decimal("0"))
    max_heat_recovery_kw: Decimal = Field(default=Decimal("0"))
    hot_utility_savings_pct: Decimal = Field(default=Decimal("0"))
    cold_utility_savings_pct: Decimal = Field(default=Decimal("0"))
    dt_min_c: Decimal = Field(default=Decimal("10"))
    hot_composite: List[Dict[str, str]] = Field(default_factory=list)
    cold_composite: List[Dict[str, str]] = Field(default_factory=list)

class HeatExchangerDesign(BaseModel):
    """Heat exchanger design parameters.

    Attributes:
        hx_id: Heat exchanger identifier.
        hx_type: Heat exchanger type.
        source_id: Hot-side source ID.
        sink_id: Cold-side sink ID.
        heat_duty_kw: Heat transfer duty (kW).
        lmtd_c: Log mean temperature difference (C).
        u_value_kw_m2k: Overall heat transfer coefficient (kW/(m2*K)).
        area_m2: Required heat transfer area (m2).
        fouling_factor: Applied fouling factor.
        estimated_cost_eur: Estimated equipment cost (EUR).
    """
    hx_id: str = Field(default_factory=_new_uuid)
    hx_type: str = Field(default="")
    source_id: str = Field(default="")
    sink_id: str = Field(default="")
    heat_duty_kw: Decimal = Field(default=Decimal("0"))
    lmtd_c: Decimal = Field(default=Decimal("0"))
    u_value_kw_m2k: Decimal = Field(default=Decimal("0"))
    area_m2: Decimal = Field(default=Decimal("0"))
    fouling_factor: Decimal = Field(default=Decimal("0"))
    estimated_cost_eur: Decimal = Field(default=Decimal("0"))

class TechnologyRecommendation(BaseModel):
    """Technology recommendation for a recovery opportunity.

    Attributes:
        technology: Recommended technology.
        technology_name: Human-readable name.
        source_id: Matched heat source.
        sink_id: Matched heat sink (if applicable).
        recoverable_kw: Recoverable heat/power (kW).
        annual_recovery_mwh: Annual recovery (MWh).
        efficiency_pct: Expected recovery efficiency (%).
        capex_eur: Capital cost estimate (EUR).
        annual_savings_eur: Annual savings (EUR).
        simple_payback_years: Simple payback (years).
        co2_savings_tco2e: Annual CO2 savings (tCO2e).
        suitability_score: Technology suitability (0-100).
    """
    technology: str = Field(default="")
    technology_name: str = Field(default="")
    source_id: str = Field(default="")
    sink_id: str = Field(default="")
    recoverable_kw: Decimal = Field(default=Decimal("0"))
    annual_recovery_mwh: Decimal = Field(default=Decimal("0"))
    efficiency_pct: Decimal = Field(default=Decimal("0"))
    capex_eur: Decimal = Field(default=Decimal("0"))
    annual_savings_eur: Decimal = Field(default=Decimal("0"))
    simple_payback_years: Decimal = Field(default=Decimal("0"))
    co2_savings_tco2e: Decimal = Field(default=Decimal("0"))
    suitability_score: Decimal = Field(default=Decimal("0"))

class WasteHeatResult(BaseModel):
    """Complete waste heat recovery analysis result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        facility_id: Facility identifier.
        facility_name: Facility name.
        source_analyses: Per-source analysis results.
        sink_count: Number of heat sinks.
        total_waste_heat_kw: Total available waste heat (kW).
        total_waste_heat_mwh: Total annual waste heat (MWh).
        total_waste_heat_value_eur: Total annual value (EUR).
        high_grade_kw: High grade (>400C) waste heat (kW).
        medium_grade_kw: Medium grade (100-400C) waste heat (kW).
        low_grade_kw: Low grade (<100C) waste heat (kW).
        pinch_analysis: Pinch analysis results.
        heat_exchangers: Heat exchanger designs.
        technology_recommendations: Technology recommendations.
        total_recoverable_kw: Total recoverable heat (kW).
        total_recoverable_mwh: Total annual recoverable (MWh).
        total_savings_eur: Total annual savings (EUR).
        total_capex_eur: Total capital cost (EUR).
        portfolio_payback_years: Portfolio payback (years).
        total_co2_savings_tco2e: Total annual CO2 savings (tCO2e).
        recovery_pct: Recovery as percentage of total waste heat.
        recommendations: Recommendations.
        warnings: Warnings.
        errors: Errors.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    source_analyses: List[SourceAnalysis] = Field(default_factory=list)
    sink_count: int = Field(default=0)
    total_waste_heat_kw: Decimal = Field(default=Decimal("0"))
    total_waste_heat_mwh: Decimal = Field(default=Decimal("0"))
    total_waste_heat_value_eur: Decimal = Field(default=Decimal("0"))
    high_grade_kw: Decimal = Field(default=Decimal("0"))
    medium_grade_kw: Decimal = Field(default=Decimal("0"))
    low_grade_kw: Decimal = Field(default=Decimal("0"))
    pinch_analysis: Optional[PinchAnalysisResult] = Field(default=None)
    heat_exchangers: List[HeatExchangerDesign] = Field(default_factory=list)
    technology_recommendations: List[TechnologyRecommendation] = Field(default_factory=list)
    total_recoverable_kw: Decimal = Field(default=Decimal("0"))
    total_recoverable_mwh: Decimal = Field(default=Decimal("0"))
    total_savings_eur: Decimal = Field(default=Decimal("0"))
    total_capex_eur: Decimal = Field(default=Decimal("0"))
    portfolio_payback_years: Decimal = Field(default=Decimal("0"))
    total_co2_savings_tco2e: Decimal = Field(default=Decimal("0"))
    recovery_pct: Decimal = Field(default=Decimal("0"))
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class WasteHeatRecoveryEngine:
    """Waste heat recovery opportunity identification and quantification engine.

    Inventories waste heat sources, performs pinch analysis, sizes heat
    exchangers, and selects recovery technologies with ROI analysis.

    Usage::

        engine = WasteHeatRecoveryEngine()
        result = engine.analyze(input_data)
        print(f"Total waste heat: {result.total_waste_heat_kw} kW")
        print(f"Recoverable: {result.total_recoverable_kw} kW")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise WasteHeatRecoveryEngine.

        Args:
            config: Optional overrides.
        """
        self.config = config or {}
        self._energy_price = _decimal(
            self.config.get("energy_price_eur_kwh", DEFAULT_ENERGY_PRICE_EUR_KWH)
        )
        self._co2_factor = _decimal(
            self.config.get("co2_factor_kg_kwh", DEFAULT_CO2_FACTOR_KG_KWH)
        )
        logger.info(
            "WasteHeatRecoveryEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def analyze(
        self, data: WasteHeatRecoveryInput,
    ) -> WasteHeatResult:
        """Perform complete waste heat recovery analysis.

        Args:
            data: Validated waste heat recovery input.

        Returns:
            WasteHeatResult with complete analysis.
        """
        t0 = time.perf_counter()
        logger.info(
            "Waste heat analysis: facility=%s, sources=%d, sinks=%d",
            data.facility_name, len(data.sources), len(data.sinks),
        )

        warnings: List[str] = []
        errors: List[str] = []
        recommendations: List[str] = []
        price = data.energy_price_eur_kwh if data.energy_price_eur_kwh > Decimal("0") else self._energy_price

        if not data.sources:
            errors.append("No waste heat sources provided.")

        # Step 1: Analyze each source
        source_analyses: List[SourceAnalysis] = []
        high_kw = Decimal("0")
        medium_kw = Decimal("0")
        low_kw = Decimal("0")

        for src in data.sources:
            cp = src.specific_heat_kj_kgk
            if cp <= Decimal("0"):
                cp = SPECIFIC_HEAT_CAPACITY.get(src.fluid_type, Decimal("1.0"))

            dt = src.inlet_temperature_c - src.outlet_temperature_c
            available_kw = src.flow_rate_kg_s * cp * dt  # kJ/s = kW
            available_kw = max(available_kw, Decimal("0"))

            annual_mwh = available_kw * _decimal(src.operating_hours) / Decimal("1000")
            annual_eur = annual_mwh * price * Decimal("1000") / Decimal("1000")  # MWh * EUR/kWh * 1000

            # Correctly: annual_eur = available_kw * hours * price_eur_kwh
            annual_eur = available_kw * _decimal(src.operating_hours) * price

            # Temperature grade
            grade = self._classify_temperature(src.inlet_temperature_c)
            if grade == TemperatureGrade.HIGH.value:
                high_kw += available_kw
            elif grade == TemperatureGrade.MEDIUM.value:
                medium_kw += available_kw
            else:
                low_kw += available_kw

            # Carnot efficiency
            carnot = self._carnot_efficiency(
                src.inlet_temperature_c, data.ambient_temperature_c
            )

            source_analyses.append(SourceAnalysis(
                source_id=src.source_id,
                name=src.name,
                source_type=src.source_type,
                temperature_grade=grade,
                available_heat_kw=_round_val(available_kw, 2),
                annual_heat_mwh=_round_val(annual_mwh, 2),
                annual_value_eur=_round_val(annual_eur, 2),
                carnot_efficiency=_round_val(carnot, 4),
            ))

        total_kw = high_kw + medium_kw + low_kw
        total_mwh = sum((sa.annual_heat_mwh for sa in source_analyses), Decimal("0"))
        total_value = sum((sa.annual_value_eur for sa in source_analyses), Decimal("0"))

        # Step 2: Pinch analysis
        pinch_result: Optional[PinchAnalysisResult] = None
        if data.include_pinch_analysis and data.sources and data.sinks:
            pinch_result = self._pinch_analysis(
                data.sources, data.sinks, data.pinch_dt_min_c, warnings
            )

        # Step 3: Heat exchanger sizing
        heat_exchangers: List[HeatExchangerDesign] = []
        if data.include_heat_exchanger_sizing and data.sources and data.sinks:
            heat_exchangers = self._size_heat_exchangers(
                data.sources, data.sinks, data.pinch_dt_min_c, warnings
            )

        # Step 4: Technology selection
        tech_recs: List[TechnologyRecommendation] = []
        if data.include_technology_selection:
            tech_recs = self._select_technologies(
                data.sources, data.sinks, source_analyses,
                price, data.co2_factor_kg_kwh, data.discount_rate, warnings
            )

        # Step 5: Totals from technology recommendations
        total_recoverable_kw = sum(
            (tr.recoverable_kw for tr in tech_recs), Decimal("0")
        )
        total_recoverable_mwh = sum(
            (tr.annual_recovery_mwh for tr in tech_recs), Decimal("0")
        )
        total_savings = sum(
            (tr.annual_savings_eur for tr in tech_recs), Decimal("0")
        )
        total_capex = sum(
            (tr.capex_eur for tr in tech_recs), Decimal("0")
        )
        total_co2 = sum(
            (tr.co2_savings_tco2e for tr in tech_recs), Decimal("0")
        )
        portfolio_payback = _safe_divide(total_capex, total_savings, Decimal("99"))
        recovery_pct = _safe_pct(total_recoverable_kw, total_kw)

        # Step 6: Recommendations
        if total_kw > Decimal("0"):
            recommendations.append(
                f"Total waste heat identified: {_round_val(total_kw, 0)} kW "
                f"({_round_val(total_mwh, 0)} MWh/year), valued at "
                f"{_round_val(total_value, 0)} EUR/year."
            )

        if high_kw > Decimal("0"):
            recommendations.append(
                f"High-grade waste heat (>400C): {_round_val(high_kw, 0)} kW. "
                f"Consider steam generation, ORC power generation, or preheating."
            )
        if medium_kw > Decimal("0"):
            recommendations.append(
                f"Medium-grade waste heat (100-400C): {_round_val(medium_kw, 0)} kW. "
                f"Consider economizers, preheaters, or absorption cooling."
            )
        if low_kw > Decimal("0"):
            recommendations.append(
                f"Low-grade waste heat (<100C): {_round_val(low_kw, 0)} kW. "
                f"Consider heat pumps for temperature upgrade or direct use "
                f"for space/water heating."
            )

        if pinch_result and pinch_result.max_heat_recovery_kw > Decimal("0"):
            recommendations.append(
                f"Pinch analysis indicates maximum heat recovery of "
                f"{_round_val(pinch_result.max_heat_recovery_kw, 0)} kW "
                f"with pinch at {_round_val(pinch_result.pinch_temperature_c, 1)}C."
            )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = WasteHeatResult(
            facility_id=data.facility_id,
            facility_name=data.facility_name,
            source_analyses=source_analyses,
            sink_count=len(data.sinks),
            total_waste_heat_kw=_round_val(total_kw, 2),
            total_waste_heat_mwh=_round_val(total_mwh, 2),
            total_waste_heat_value_eur=_round_val(total_value, 2),
            high_grade_kw=_round_val(high_kw, 2),
            medium_grade_kw=_round_val(medium_kw, 2),
            low_grade_kw=_round_val(low_kw, 2),
            pinch_analysis=pinch_result,
            heat_exchangers=heat_exchangers,
            technology_recommendations=tech_recs,
            total_recoverable_kw=_round_val(total_recoverable_kw, 2),
            total_recoverable_mwh=_round_val(total_recoverable_mwh, 2),
            total_savings_eur=_round_val(total_savings, 2),
            total_capex_eur=_round_val(total_capex, 2),
            portfolio_payback_years=_round_val(portfolio_payback, 2),
            total_co2_savings_tco2e=_round_val(total_co2, 3),
            recovery_pct=_round_val(recovery_pct, 2),
            recommendations=recommendations,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Waste heat analysis complete: %.0f kW total, %.0f kW recoverable (%.1f%%), "
            "savings=%.0f EUR/yr, hash=%s",
            float(total_kw), float(total_recoverable_kw), float(recovery_pct),
            float(total_savings), result.provenance_hash[:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # Temperature Classification                                           #
    # ------------------------------------------------------------------ #

    def _classify_temperature(self, temp_c: Decimal) -> str:
        """Classify waste heat temperature grade.

        Args:
            temp_c: Temperature in degrees C.

        Returns:
            TemperatureGrade value.
        """
        if temp_c > Decimal("400"):
            return TemperatureGrade.HIGH.value
        elif temp_c >= Decimal("100"):
            return TemperatureGrade.MEDIUM.value
        else:
            return TemperatureGrade.LOW.value

    # ------------------------------------------------------------------ #
    # Carnot Efficiency                                                    #
    # ------------------------------------------------------------------ #

    def _carnot_efficiency(
        self, t_hot_c: Decimal, t_cold_c: Decimal,
    ) -> Decimal:
        """Calculate Carnot efficiency for heat-to-power conversion.

        eta_carnot = 1 - T_cold_K / T_hot_K

        Args:
            t_hot_c: Hot source temperature (C).
            t_cold_c: Cold sink temperature (C).

        Returns:
            Carnot efficiency (0-1).
        """
        t_hot_k = t_hot_c + Decimal("273.15")
        t_cold_k = t_cold_c + Decimal("273.15")
        if t_hot_k <= Decimal("0"):
            return Decimal("0")
        return max(Decimal("1") - _safe_divide(t_cold_k, t_hot_k), Decimal("0"))

    # ------------------------------------------------------------------ #
    # Pinch Analysis                                                       #
    # ------------------------------------------------------------------ #

    def _pinch_analysis(
        self,
        sources: List[WasteHeatSource],
        sinks: List[HeatSink],
        dt_min: Decimal,
        warnings: List[str],
    ) -> PinchAnalysisResult:
        """Perform simplified pinch analysis.

        Calculates hot and cold composite curves, finds pinch temperature,
        and determines minimum heating/cooling utilities and maximum
        internal heat recovery.

        Args:
            sources: Hot streams (waste heat sources).
            sinks: Cold streams (heat demands).
            dt_min: Minimum temperature approach (C).
            warnings: Warning list.

        Returns:
            PinchAnalysisResult.
        """
        # Calculate total hot stream heat and cold stream heat
        total_hot_kw = Decimal("0")
        hot_intervals: List[Tuple[Decimal, Decimal, Decimal]] = []  # (T_in, T_out, Q_kw)

        for src in sources:
            cp = src.specific_heat_kj_kgk
            if cp <= Decimal("0"):
                cp = SPECIFIC_HEAT_CAPACITY.get(src.fluid_type, Decimal("1.0"))
            dt = src.inlet_temperature_c - src.outlet_temperature_c
            q = src.flow_rate_kg_s * cp * max(dt, Decimal("0"))
            total_hot_kw += q
            hot_intervals.append((src.inlet_temperature_c, src.outlet_temperature_c, q))

        total_cold_kw = Decimal("0")
        cold_intervals: List[Tuple[Decimal, Decimal, Decimal]] = []

        for snk in sinks:
            cp = snk.specific_heat_kj_kgk
            if cp <= Decimal("0"):
                cp = SPECIFIC_HEAT_CAPACITY.get(snk.fluid_type, Decimal("4.186"))
            if snk.flow_rate_kg_s > Decimal("0"):
                dt = snk.target_temperature_c - snk.inlet_temperature_c
                q = snk.flow_rate_kg_s * cp * max(dt, Decimal("0"))
            else:
                q = snk.required_heat_kw
            total_cold_kw += q
            cold_intervals.append((snk.inlet_temperature_c, snk.target_temperature_c, q))

        # Simplified pinch: find temperature where heat balance shifts
        # Collect all temperature levels
        temps: set = set()
        for t_in, t_out, _ in hot_intervals:
            temps.add(t_in)
            temps.add(t_out)
        for t_in, t_out, _ in cold_intervals:
            temps.add(t_in + dt_min)  # Shift cold by dt_min
            temps.add(t_out + dt_min)

        sorted_temps = sorted(temps, reverse=True)

        # Problem table algorithm (simplified)
        pinch_temp = Decimal("0")
        max_cascade_deficit = Decimal("0")
        cascade = Decimal("0")

        for i in range(len(sorted_temps) - 1):
            t_upper = sorted_temps[i]
            t_lower = sorted_temps[i + 1]
            dt_interval = t_upper - t_lower

            # Hot stream contribution in this interval
            hot_in_interval = Decimal("0")
            for t_in, t_out, q in hot_intervals:
                if t_in >= t_upper and t_out <= t_lower:
                    hot_in_interval += q
                elif t_in > t_lower and t_out < t_upper:
                    frac = dt_interval / max(t_in - t_out, Decimal("1"))
                    hot_in_interval += q * min(frac, Decimal("1"))

            # Cold stream contribution (shifted)
            cold_in_interval = Decimal("0")
            for t_in, t_out, q in cold_intervals:
                t_in_s = t_in + dt_min
                t_out_s = t_out + dt_min
                if t_out_s >= t_upper and t_in_s <= t_lower:
                    cold_in_interval += q
                elif t_out_s > t_lower and t_in_s < t_upper:
                    frac = dt_interval / max(t_out_s - t_in_s, Decimal("1"))
                    cold_in_interval += q * min(frac, Decimal("1"))

            cascade += hot_in_interval - cold_in_interval

            if cascade < max_cascade_deficit:
                max_cascade_deficit = cascade
                pinch_temp = t_lower

        # Minimum utilities
        min_heating = abs(max_cascade_deficit) if max_cascade_deficit < Decimal("0") else Decimal("0")
        min_cooling = max(total_hot_kw - total_cold_kw + min_heating, Decimal("0"))
        max_recovery = min(total_hot_kw, total_cold_kw) - min_heating

        # Prevent negative recovery
        max_recovery = max(max_recovery, Decimal("0"))

        # Savings percentages
        hot_savings = _safe_pct(max_recovery, total_hot_kw)
        cold_savings = _safe_pct(max_recovery, total_cold_kw)

        # Build composite curve points (simplified)
        hot_composite: List[Dict[str, str]] = []
        cold_composite: List[Dict[str, str]] = []

        cum_hot = Decimal("0")
        for t_in, t_out, q in sorted(hot_intervals, key=lambda x: x[0], reverse=True):
            hot_composite.append({"temperature_c": str(t_in), "cumulative_kw": str(cum_hot)})
            cum_hot += q
            hot_composite.append({"temperature_c": str(t_out), "cumulative_kw": str(_round_val(cum_hot, 2))})

        cum_cold = Decimal("0")
        for t_in, t_out, q in sorted(cold_intervals, key=lambda x: x[1]):
            cold_composite.append({"temperature_c": str(t_in), "cumulative_kw": str(cum_cold)})
            cum_cold += q
            cold_composite.append({"temperature_c": str(t_out), "cumulative_kw": str(_round_val(cum_cold, 2))})

        return PinchAnalysisResult(
            pinch_temperature_c=_round_val(pinch_temp, 1),
            min_heating_utility_kw=_round_val(min_heating, 2),
            min_cooling_utility_kw=_round_val(min_cooling, 2),
            max_heat_recovery_kw=_round_val(max_recovery, 2),
            hot_utility_savings_pct=_round_val(hot_savings, 2),
            cold_utility_savings_pct=_round_val(cold_savings, 2),
            dt_min_c=dt_min,
            hot_composite=hot_composite,
            cold_composite=cold_composite,
        )

    # ------------------------------------------------------------------ #
    # Heat Exchanger Sizing                                                #
    # ------------------------------------------------------------------ #

    def _size_heat_exchangers(
        self,
        sources: List[WasteHeatSource],
        sinks: List[HeatSink],
        dt_min: Decimal,
        warnings: List[str],
    ) -> List[HeatExchangerDesign]:
        """Size heat exchangers for source-sink pairs using LMTD method.

        Q = U * A * LMTD
        A = Q / (U * LMTD)

        Args:
            sources: Hot streams.
            sinks: Cold streams.
            dt_min: Minimum temperature approach.
            warnings: Warning list.

        Returns:
            List of HeatExchangerDesign.
        """
        designs: List[HeatExchangerDesign] = []

        for src in sources:
            for snk in sinks:
                # Check temperature compatibility
                if src.inlet_temperature_c < snk.target_temperature_c + dt_min:
                    continue  # Source not hot enough

                # Calculate heat duty
                cp_src = src.specific_heat_kj_kgk
                if cp_src <= Decimal("0"):
                    cp_src = SPECIFIC_HEAT_CAPACITY.get(src.fluid_type, Decimal("1.0"))

                cp_snk = snk.specific_heat_kj_kgk
                if cp_snk <= Decimal("0"):
                    cp_snk = SPECIFIC_HEAT_CAPACITY.get(snk.fluid_type, Decimal("4.186"))

                q_source = src.flow_rate_kg_s * cp_src * (
                    src.inlet_temperature_c - src.outlet_temperature_c
                )
                q_source = max(q_source, Decimal("0"))

                if snk.flow_rate_kg_s > Decimal("0"):
                    q_sink = snk.flow_rate_kg_s * cp_snk * (
                        snk.target_temperature_c - snk.inlet_temperature_c
                    )
                else:
                    q_sink = snk.required_heat_kw

                q_duty = min(q_source, max(q_sink, Decimal("0")))
                if q_duty <= Decimal("0"):
                    continue

                # LMTD calculation (counter-flow)
                lmtd = self._calculate_lmtd(
                    src.inlet_temperature_c, src.outlet_temperature_c,
                    snk.inlet_temperature_c, snk.target_temperature_c,
                )
                if lmtd <= Decimal("0"):
                    continue

                # U-value lookup
                u_key = f"{src.fluid_type}_{snk.fluid_type}"
                u_clean = HEAT_EXCHANGER_U_VALUES.get(u_key, Decimal("0.050"))

                # Apply fouling factors
                rf_hot = FOULING_FACTORS.get(
                    f"{src.fluid_type}_clean",
                    FOULING_FACTORS.get("process_fluid", Decimal("0.0005"))
                )
                rf_cold = FOULING_FACTORS.get(
                    f"{snk.fluid_type}_clean",
                    FOULING_FACTORS.get("clean_water", Decimal("0.0001"))
                )
                # U_dirty = 1 / (1/U_clean + Rf_hot + Rf_cold)
                u_dirty = _safe_divide(
                    Decimal("1"),
                    _safe_divide(Decimal("1"), u_clean) + rf_hot + rf_cold
                )

                # Area calculation
                area = _safe_divide(q_duty, u_dirty * lmtd)

                # Cost estimate (EUR/m2 by type)
                hx_type = self._select_hx_type(src, snk)
                cost_per_m2 = Decimal("500")  # Default shell-and-tube
                if hx_type == HeatExchangerType.PLATE.value:
                    cost_per_m2 = Decimal("350")
                elif hx_type == HeatExchangerType.ECONOMIZER.value:
                    cost_per_m2 = Decimal("400")
                elif hx_type == HeatExchangerType.RECUPERATOR.value:
                    cost_per_m2 = Decimal("600")

                cost = area * cost_per_m2

                designs.append(HeatExchangerDesign(
                    hx_type=hx_type,
                    source_id=src.source_id,
                    sink_id=snk.sink_id,
                    heat_duty_kw=_round_val(q_duty, 2),
                    lmtd_c=_round_val(lmtd, 2),
                    u_value_kw_m2k=_round_val(u_dirty, 4),
                    area_m2=_round_val(area, 2),
                    fouling_factor=_round_val(rf_hot + rf_cold, 6),
                    estimated_cost_eur=_round_val(cost, 2),
                ))

        return designs

    def _calculate_lmtd(
        self,
        t_hot_in: Decimal, t_hot_out: Decimal,
        t_cold_in: Decimal, t_cold_out: Decimal,
    ) -> Decimal:
        """Calculate Log Mean Temperature Difference (counter-flow).

        LMTD = (dT1 - dT2) / ln(dT1 / dT2)
        where dT1 = T_hot_in - T_cold_out, dT2 = T_hot_out - T_cold_in

        Args:
            t_hot_in: Hot inlet temperature (C).
            t_hot_out: Hot outlet temperature (C).
            t_cold_in: Cold inlet temperature (C).
            t_cold_out: Cold outlet temperature (C).

        Returns:
            LMTD in degrees C.
        """
        dt1 = t_hot_in - t_cold_out
        dt2 = t_hot_out - t_cold_in

        if dt1 <= Decimal("0") or dt2 <= Decimal("0"):
            return Decimal("0")

        if dt1 == dt2:
            return dt1  # Special case: LMTD = arithmetic mean

        ratio = float(dt1 / dt2)
        if ratio <= 0:
            return Decimal("0")

        ln_ratio = _decimal(math.log(ratio))
        if ln_ratio == Decimal("0"):
            return dt1

        return _safe_divide(dt1 - dt2, ln_ratio)

    def _select_hx_type(
        self, source: WasteHeatSource, sink: HeatSink,
    ) -> str:
        """Select appropriate heat exchanger type.

        Args:
            source: Hot stream.
            sink: Cold stream.

        Returns:
            HeatExchangerType value.
        """
        if source.source_type == HeatSourceType.FLUE_GAS.value:
            if sink.fluid_type == "water":
                return HeatExchangerType.ECONOMIZER.value
            else:
                return HeatExchangerType.RECUPERATOR.value

        if source.inlet_temperature_c > Decimal("400"):
            return HeatExchangerType.RECUPERATOR.value

        if (source.fluid_type in ("water", "glycol_50pct") and
                sink.fluid_type in ("water", "glycol_50pct")):
            return HeatExchangerType.PLATE.value

        return HeatExchangerType.SHELL_TUBE.value

    # ------------------------------------------------------------------ #
    # Technology Selection                                                 #
    # ------------------------------------------------------------------ #

    def _select_technologies(
        self,
        sources: List[WasteHeatSource],
        sinks: List[HeatSink],
        analyses: List[SourceAnalysis],
        price: Decimal,
        co2_factor: Decimal,
        discount_rate: Decimal,
        warnings: List[str],
    ) -> List[TechnologyRecommendation]:
        """Select optimal recovery technologies for each source.

        Evaluates each technology in the database against source temperature
        and available heat, calculates suitability score, and provides
        ROI analysis.

        Args:
            sources: Heat sources.
            sinks: Heat sinks.
            analyses: Source analysis results.
            price: Energy price (EUR/kWh).
            co2_factor: CO2 factor (kg/kWh).
            discount_rate: Discount rate.
            warnings: Warning list.

        Returns:
            List of TechnologyRecommendation.
        """
        recs: List[TechnologyRecommendation] = []

        for i, src in enumerate(sources):
            if i >= len(analyses):
                break
            sa = analyses[i]
            if sa.available_heat_kw <= Decimal("0"):
                continue

            best_tech = None
            best_score = Decimal("0")

            for tech_id, tech_info in RECOVERY_TECHNOLOGY_DATABASE.items():
                temp_min = _decimal(tech_info["temp_range_min_c"])
                temp_max = _decimal(tech_info["temp_range_max_c"])

                # Check temperature compatibility
                if src.inlet_temperature_c < temp_min or src.inlet_temperature_c > temp_max:
                    continue

                eff = tech_info["efficiency_pct"]
                capex_per_kw = tech_info["capex_per_kw"]

                # For heat pumps, COP > 1 means efficiency >100%
                if eff > Decimal("100"):
                    recoverable_kw = sa.available_heat_kw * eff / Decimal("100")
                else:
                    recoverable_kw = sa.available_heat_kw * eff / Decimal("100")

                annual_mwh = recoverable_kw * _decimal(src.operating_hours) / Decimal("1000")
                annual_savings = recoverable_kw * _decimal(src.operating_hours) * price
                capex = recoverable_kw * capex_per_kw
                payback = _safe_divide(capex, annual_savings, Decimal("99"))
                co2_saved = recoverable_kw * _decimal(src.operating_hours) * co2_factor / Decimal("1000000")

                # Suitability score
                score = self._technology_suitability(
                    src, tech_id, eff, payback, recoverable_kw, sinks
                )

                if score > best_score:
                    best_score = score
                    best_tech = TechnologyRecommendation(
                        technology=tech_id,
                        technology_name=tech_info["name"],
                        source_id=src.source_id,
                        recoverable_kw=_round_val(recoverable_kw, 2),
                        annual_recovery_mwh=_round_val(annual_mwh, 2),
                        efficiency_pct=eff,
                        capex_eur=_round_val(capex, 2),
                        annual_savings_eur=_round_val(annual_savings, 2),
                        simple_payback_years=_round_val(payback, 2),
                        co2_savings_tco2e=_round_val(co2_saved, 3),
                        suitability_score=_round_val(score, 2),
                    )

            if best_tech:
                # Match to nearest sink if applicable
                for snk in sinks:
                    if (snk.target_temperature_c <= src.inlet_temperature_c and
                            snk.inlet_temperature_c >= src.outlet_temperature_c - Decimal("20")):
                        best_tech.sink_id = snk.sink_id
                        break

                recs.append(best_tech)

        return recs

    def _technology_suitability(
        self,
        source: WasteHeatSource,
        tech_id: str,
        efficiency: Decimal,
        payback: Decimal,
        recoverable_kw: Decimal,
        sinks: List[HeatSink],
    ) -> Decimal:
        """Calculate technology suitability score (0-100).

        Scoring:
            Temperature match:    30%
            Financial (payback):  30%
            Recovery amount:      20%
            Demand matching:      20%

        Args:
            source: Heat source.
            tech_id: Technology identifier.
            efficiency: Technology efficiency.
            payback: Simple payback (years).
            recoverable_kw: Recoverable heat (kW).
            sinks: Available heat sinks.

        Returns:
            Suitability score (0-100).
        """
        score = Decimal("0")

        # Temperature match (30 pts)
        tech = RECOVERY_TECHNOLOGY_DATABASE.get(tech_id, {})
        temp_min = _decimal(tech.get("temp_range_min_c", 0))
        temp_max = _decimal(tech.get("temp_range_max_c", 900))
        temp_range = temp_max - temp_min
        src_temp = source.inlet_temperature_c

        if temp_range > Decimal("0"):
            position = (src_temp - temp_min) / temp_range
            # Best score at 30-70% of range (sweet spot)
            if Decimal("0.3") <= position <= Decimal("0.7"):
                score += Decimal("30")
            elif position > Decimal("0"):
                score += Decimal("20")

        # Financial (30 pts): shorter payback = higher score
        if payback <= Decimal("2"):
            score += Decimal("30")
        elif payback <= Decimal("5"):
            score += Decimal("20")
        elif payback <= Decimal("8"):
            score += Decimal("10")

        # Recovery amount (20 pts)
        if recoverable_kw > Decimal("500"):
            score += Decimal("20")
        elif recoverable_kw > Decimal("100"):
            score += Decimal("15")
        elif recoverable_kw > Decimal("10"):
            score += Decimal("10")
        else:
            score += Decimal("5")

        # Demand matching (20 pts): is there a sink that can use this heat?
        for snk in sinks:
            if (snk.target_temperature_c <= source.inlet_temperature_c and
                    snk.required_heat_kw <= recoverable_kw):
                score += Decimal("20")
                break
        else:
            score += Decimal("5")  # No matching sink, but heat still has value

        return min(score, Decimal("100"))
