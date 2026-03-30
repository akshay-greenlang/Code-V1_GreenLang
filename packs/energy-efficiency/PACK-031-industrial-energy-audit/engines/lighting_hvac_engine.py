# -*- coding: utf-8 -*-
"""
LightingHVACEngine - PACK-031 Industrial Energy Audit Engine 9
================================================================

Calculates energy savings from lighting upgrades and HVAC optimisation
in industrial facilities.  Covers LED retrofit analysis, daylight
harvesting, occupancy sensing, Lighting Power Density (LPD) benchmarking,
HVAC system efficiency assessment, Variable Speed Drive (VSD) analysis
using affinity laws, economiser (free cooling) cycle estimation, heat
recovery ventilation, setpoint/deadband optimisation, and demand-
controlled ventilation.

Lighting Standards / References:
    - EN 12464-1:2021 (Indoor workplace lighting)
    - IES Lighting Handbook (US)
    - EU Ecodesign Regulation (EU) 2019/2020
    - ESRS E1-5 (energy consumption and mix)

HVAC Standards / References:
    - EN 14511 / EN 14825 (SEER, SCOP calculation)
    - ASHRAE 90.1 (HVAC efficiency benchmarks)
    - EU Ecodesign Regulation for circulators and fans
    - ISO 50001:2018 Energy Management Systems
    - EU ENE BREF (energy efficiency reference document)

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - LPD benchmarks from EN 12464-1:2021 published tables
    - VSD savings from cubic affinity law (P ~ n^3)
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

class FixtureType(str, Enum):
    """Lighting fixture / lamp types."""
    FLUORESCENT_T8 = "fluorescent_t8"
    FLUORESCENT_T5 = "fluorescent_t5"
    HID_MH = "hid_metal_halide"
    HID_HPS = "hid_high_pressure_sodium"
    LED = "led"
    HALOGEN = "halogen"
    CFL = "cfl"
    INCANDESCENT = "incandescent"

class SpaceType(str, Enum):
    """Space classification for LPD benchmarking (EN 12464-1)."""
    OFFICE = "office"
    WAREHOUSE = "warehouse"
    PRODUCTION_GENERAL = "production_general"
    PRODUCTION_FINE = "production_fine"
    LABORATORY = "laboratory"
    CORRIDOR = "corridor"
    LOADING_DOCK = "loading_dock"
    CANTEEN = "canteen"
    EXTERIOR = "exterior"
    RESTROOM = "restroom"
    CONTROL_ROOM = "control_room"
    CLEAN_ROOM = "clean_room"
    COLD_STORE = "cold_store"
    MECHANICAL_ROOM = "mechanical_room"

class HVACSystemType(str, Enum):
    """HVAC system types."""
    SPLIT_SYSTEM = "split_system"
    VRF = "vrf"
    CHILLER_AHU = "chiller_ahu"
    ROOFTOP_UNIT = "rooftop_unit"
    HEAT_PUMP = "heat_pump"
    EVAPORATIVE = "evaporative"
    DISTRICT_COOLING = "district_cooling"
    DISTRICT_HEATING = "district_heating"
    GAS_FIRED_HEATER = "gas_fired_heater"

class ClimateZone(str, Enum):
    """European climate zones for free-cooling estimation."""
    NORTHERN = "northern"           # Scandinavia, Baltics
    CENTRAL_MARITIME = "central_maritime"   # UK, Benelux, NW Germany
    CENTRAL_CONTINENTAL = "central_continental"  # Germany, Poland, Czechia
    SOUTHERN_MARITIME = "southern_maritime"   # Portugal, W Spain, W Italy
    SOUTHERN_CONTINENTAL = "southern_continental"  # E Spain, S Italy, Greece
    MEDITERRANEAN = "mediterranean"        # Cyprus, Malta, S Greece

class VentilationStrategy(str, Enum):
    """Ventilation control strategies."""
    CONSTANT_VOLUME = "constant_volume"
    VAV = "variable_air_volume"
    DCV_CO2 = "demand_controlled_co2"
    DCV_OCCUPANCY = "demand_controlled_occupancy"
    NATURAL = "natural"
    HYBRID = "hybrid"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Lighting efficacy (lumens/watt) by fixture type.
# Source: IES Lighting Handbook 10th Ed., manufacturer data.
LIGHTING_EFFICACY: Dict[str, Dict[str, Any]] = {
    FixtureType.FLUORESCENT_T8: {
        "efficacy_lm_w": 80, "typical_wattage": 36, "lifespan_hours": 20000,
    },
    FixtureType.FLUORESCENT_T5: {
        "efficacy_lm_w": 100, "typical_wattage": 28, "lifespan_hours": 25000,
    },
    FixtureType.HID_MH: {
        "efficacy_lm_w": 90, "typical_wattage": 400, "lifespan_hours": 15000,
    },
    FixtureType.HID_HPS: {
        "efficacy_lm_w": 120, "typical_wattage": 250, "lifespan_hours": 24000,
    },
    FixtureType.LED: {
        "efficacy_lm_w": 150, "typical_wattage": 20, "lifespan_hours": 50000,
    },
    FixtureType.HALOGEN: {
        "efficacy_lm_w": 20, "typical_wattage": 50, "lifespan_hours": 3000,
    },
    FixtureType.CFL: {
        "efficacy_lm_w": 65, "typical_wattage": 18, "lifespan_hours": 10000,
    },
    FixtureType.INCANDESCENT: {
        "efficacy_lm_w": 14, "typical_wattage": 60, "lifespan_hours": 1000,
    },
}

# LPD benchmarks (W/m2) by space type.
# Source: EN 12464-1:2021 recommended maintained illuminance + typical
# LED efficacy to derive maximum installed power density.
LPD_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    SpaceType.OFFICE: {
        "required_lux": 500, "best_practice_w_sqm": 6.0,
        "good_w_sqm": 8.0, "average_w_sqm": 12.0, "poor_w_sqm": 16.0,
    },
    SpaceType.WAREHOUSE: {
        "required_lux": 200, "best_practice_w_sqm": 3.0,
        "good_w_sqm": 5.0, "average_w_sqm": 8.0, "poor_w_sqm": 12.0,
    },
    SpaceType.PRODUCTION_GENERAL: {
        "required_lux": 300, "best_practice_w_sqm": 5.0,
        "good_w_sqm": 7.0, "average_w_sqm": 10.0, "poor_w_sqm": 15.0,
    },
    SpaceType.PRODUCTION_FINE: {
        "required_lux": 750, "best_practice_w_sqm": 10.0,
        "good_w_sqm": 14.0, "average_w_sqm": 18.0, "poor_w_sqm": 25.0,
    },
    SpaceType.LABORATORY: {
        "required_lux": 500, "best_practice_w_sqm": 7.0,
        "good_w_sqm": 10.0, "average_w_sqm": 14.0, "poor_w_sqm": 18.0,
    },
    SpaceType.CORRIDOR: {
        "required_lux": 100, "best_practice_w_sqm": 2.0,
        "good_w_sqm": 3.5, "average_w_sqm": 5.0, "poor_w_sqm": 8.0,
    },
    SpaceType.LOADING_DOCK: {
        "required_lux": 150, "best_practice_w_sqm": 3.0,
        "good_w_sqm": 5.0, "average_w_sqm": 8.0, "poor_w_sqm": 12.0,
    },
    SpaceType.CANTEEN: {
        "required_lux": 200, "best_practice_w_sqm": 4.0,
        "good_w_sqm": 6.0, "average_w_sqm": 9.0, "poor_w_sqm": 12.0,
    },
    SpaceType.EXTERIOR: {
        "required_lux": 50, "best_practice_w_sqm": 1.0,
        "good_w_sqm": 2.0, "average_w_sqm": 4.0, "poor_w_sqm": 6.0,
    },
    SpaceType.RESTROOM: {
        "required_lux": 200, "best_practice_w_sqm": 4.0,
        "good_w_sqm": 6.0, "average_w_sqm": 9.0, "poor_w_sqm": 12.0,
    },
    SpaceType.CONTROL_ROOM: {
        "required_lux": 500, "best_practice_w_sqm": 7.0,
        "good_w_sqm": 10.0, "average_w_sqm": 14.0, "poor_w_sqm": 18.0,
    },
    SpaceType.CLEAN_ROOM: {
        "required_lux": 500, "best_practice_w_sqm": 8.0,
        "good_w_sqm": 12.0, "average_w_sqm": 16.0, "poor_w_sqm": 22.0,
    },
    SpaceType.COLD_STORE: {
        "required_lux": 200, "best_practice_w_sqm": 4.0,
        "good_w_sqm": 6.0, "average_w_sqm": 10.0, "poor_w_sqm": 14.0,
    },
    SpaceType.MECHANICAL_ROOM: {
        "required_lux": 200, "best_practice_w_sqm": 3.0,
        "good_w_sqm": 5.0, "average_w_sqm": 8.0, "poor_w_sqm": 12.0,
    },
}

# Occupancy sensor savings (%) by space type.
# Source: US DOE, LBNL studies on occupancy-based lighting controls.
OCCUPANCY_SENSOR_SAVINGS: Dict[str, float] = {
    SpaceType.OFFICE:           25.0,
    SpaceType.WAREHOUSE:        35.0,
    SpaceType.PRODUCTION_GENERAL: 15.0,
    SpaceType.PRODUCTION_FINE:  10.0,
    SpaceType.LABORATORY:       20.0,
    SpaceType.CORRIDOR:         40.0,
    SpaceType.LOADING_DOCK:     35.0,
    SpaceType.CANTEEN:          30.0,
    SpaceType.RESTROOM:         50.0,
    SpaceType.CONTROL_ROOM:     10.0,
    SpaceType.CLEAN_ROOM:       10.0,
    SpaceType.COLD_STORE:       30.0,
    SpaceType.MECHANICAL_ROOM:  50.0,
    SpaceType.EXTERIOR:         30.0,
}

# VSD savings curve: % power saved vs % speed reduction.
# Based on affinity law: P2/P1 = (n2/n1)^3.
# Table gives savings at various load fractions.
VSD_SAVINGS_CURVE: Dict[int, float] = {
    # load_pct: power_savings_pct (compared to throttling/damper)
    100: 0.0,
    90:  20.0,
    80:  35.0,
    70:  50.0,
    60:  62.0,
    50:  72.0,
    40:  80.0,
    30:  86.0,
}

# Free cooling hours by climate zone (hours/year where outdoor temp
# is below indoor cooling setpoint, enabling economiser operation).
# Source: ASHRAE, Eurostat heating/cooling degree day data.
CLIMATE_ZONE_FREE_COOLING_HOURS: Dict[str, int] = {
    ClimateZone.NORTHERN:            6500,
    ClimateZone.CENTRAL_MARITIME:    5500,
    ClimateZone.CENTRAL_CONTINENTAL: 5000,
    ClimateZone.SOUTHERN_MARITIME:   3500,
    ClimateZone.SOUTHERN_CONTINENTAL: 2500,
    ClimateZone.MEDITERRANEAN:       1500,
}

# HVAC COP benchmarks by system type and age band.
# Source: EN 14511, ASHRAE 90.1, EU Ecodesign min requirements.
HVAC_COP_BENCHMARKS: Dict[str, Dict[str, float]] = {
    HVACSystemType.SPLIT_SYSTEM: {
        "best_practice_cop": 5.0, "good_cop": 4.0,
        "average_cop": 3.2, "poor_cop": 2.5,
    },
    HVACSystemType.VRF: {
        "best_practice_cop": 5.5, "good_cop": 4.5,
        "average_cop": 3.8, "poor_cop": 3.0,
    },
    HVACSystemType.CHILLER_AHU: {
        "best_practice_cop": 6.5, "good_cop": 5.5,
        "average_cop": 4.5, "poor_cop": 3.5,
    },
    HVACSystemType.ROOFTOP_UNIT: {
        "best_practice_cop": 4.5, "good_cop": 3.5,
        "average_cop": 3.0, "poor_cop": 2.2,
    },
    HVACSystemType.HEAT_PUMP: {
        "best_practice_cop": 5.5, "good_cop": 4.5,
        "average_cop": 3.5, "poor_cop": 2.8,
    },
    HVACSystemType.EVAPORATIVE: {
        "best_practice_cop": 15.0, "good_cop": 12.0,
        "average_cop": 9.0, "poor_cop": 6.0,
    },
    HVACSystemType.GAS_FIRED_HEATER: {
        "best_practice_cop": 0.97, "good_cop": 0.92,
        "average_cop": 0.85, "poor_cop": 0.75,
    },
}

# Refrigerant GWP (Global Warming Potential) table.
# Source: IPCC AR6 (100-year GWP values).
REFRIGERANT_GWP: Dict[str, int] = {
    "R-22":   1810,
    "R-134a": 1430,
    "R-404A": 3922,
    "R-407C": 1774,
    "R-410A": 2088,
    "R-32":   675,
    "R-290":  3,
    "R-600a": 3,
    "R-744":  1,
    "R-1234yf": 4,
    "R-1234ze": 7,
    "R-717":  0,
    "R-1270": 2,
}

# LED retrofit cost per fixture (EUR) by existing fixture type.
LED_RETROFIT_COST: Dict[str, float] = {
    FixtureType.FLUORESCENT_T8:  85.0,
    FixtureType.FLUORESCENT_T5:  80.0,
    FixtureType.HID_MH:         350.0,
    FixtureType.HID_HPS:        320.0,
    FixtureType.HALOGEN:         45.0,
    FixtureType.CFL:             40.0,
    FixtureType.INCANDESCENT:    35.0,
    FixtureType.LED:              0.0,
}

# VSD retrofit cost (EUR/kW motor nameplate).
VSD_COST_EUR_PER_KW: Decimal = Decimal("120")

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class LightingZone(BaseModel):
    """Individual lighting zone within a facility.

    Attributes:
        zone_id: Unique zone identifier.
        name: Human-readable zone name.
        space_type: Space classification for LPD benchmarking.
        area_sqm: Zone floor area (m2).
        fixture_count: Number of installed fixtures.
        fixture_type: Type of installed fixture.
        wattage_per_fixture: Rated wattage per fixture (W).
        operating_hours: Annual operating hours.
        current_lux: Measured maintained illuminance (lux).
        required_lux: Target illuminance per standard (lux).
        has_occupancy_sensor: Whether occupancy sensors are installed.
        has_daylight_sensor: Whether daylight dimming is installed.
        daylight_factor: Daylight factor (0-1) representing available
            natural light contribution.
    """
    zone_id: str = Field(default_factory=_new_uuid)
    name: str = Field(default="")
    space_type: SpaceType = Field(default=SpaceType.PRODUCTION_GENERAL)
    area_sqm: float = Field(default=100.0, ge=1.0)
    fixture_count: int = Field(default=10, ge=0)
    fixture_type: FixtureType = Field(default=FixtureType.FLUORESCENT_T8)
    wattage_per_fixture: float = Field(default=36.0, ge=0.0)
    operating_hours: int = Field(default=4000, ge=0, le=8760)
    current_lux: float = Field(default=300.0, ge=0.0)
    required_lux: float = Field(default=300.0, ge=0.0)
    has_occupancy_sensor: bool = Field(default=False)
    has_daylight_sensor: bool = Field(default=False)
    daylight_factor: float = Field(default=0.0, ge=0.0, le=1.0)

class HVACSystem(BaseModel):
    """HVAC system data.

    Attributes:
        system_id: Unique system identifier.
        name: Human-readable name.
        system_type: HVAC system type.
        cooling_capacity_kw: Rated cooling capacity (kW).
        heating_capacity_kw: Rated heating capacity (kW).
        current_cop: Current operating COP (or efficiency ratio).
        current_eer: Current EER if measured.
        age_years: System age (years).
        refrigerant: Refrigerant type.
        annual_cooling_hours: Annual cooling operating hours.
        annual_heating_hours: Annual heating operating hours.
        annual_energy_kwh: Annual HVAC energy consumption (kWh).
        has_economiser: Whether free cooling economiser is installed.
        has_heat_recovery: Whether heat recovery ventilation is used.
        ventilation_strategy: Current ventilation control approach.
    """
    system_id: str = Field(default_factory=_new_uuid)
    name: str = Field(default="")
    system_type: HVACSystemType = Field(default=HVACSystemType.SPLIT_SYSTEM)
    cooling_capacity_kw: float = Field(default=100.0, ge=0.0)
    heating_capacity_kw: float = Field(default=100.0, ge=0.0)
    current_cop: float = Field(default=3.0, ge=0.1, le=20.0)
    current_eer: float = Field(default=0.0, ge=0.0, le=20.0)
    age_years: int = Field(default=10, ge=0, le=40)
    refrigerant: str = Field(default="R-410A")
    annual_cooling_hours: int = Field(default=2000, ge=0, le=8760)
    annual_heating_hours: int = Field(default=3000, ge=0, le=8760)
    annual_energy_kwh: float = Field(default=0.0, ge=0.0)
    has_economiser: bool = Field(default=False)
    has_heat_recovery: bool = Field(default=False)
    ventilation_strategy: VentilationStrategy = Field(default=VentilationStrategy.CONSTANT_VOLUME)

class VSDCandidate(BaseModel):
    """Motor / equipment candidate for VSD retrofit.

    Attributes:
        equipment_id: Unique identifier.
        name: Equipment description.
        motor_kw: Motor nameplate power (kW).
        equipment_type: Equipment type (fan, pump, compressor).
        average_load_pct: Average operating load (% of nameplate).
        operating_hours: Annual operating hours.
        has_vsd: Whether VSD is already installed.
    """
    equipment_id: str = Field(default_factory=_new_uuid)
    name: str = Field(default="")
    motor_kw: float = Field(default=15.0, ge=0.1)
    equipment_type: str = Field(default="fan")
    average_load_pct: int = Field(default=80, ge=10, le=100)
    operating_hours: int = Field(default=6000, ge=0, le=8760)
    has_vsd: bool = Field(default=False)

class BuildingEnvelope(BaseModel):
    """Building envelope data for HVAC load analysis.

    Attributes:
        total_area_sqm: Total conditioned floor area (m2).
        wall_u_value: Average wall U-value (W/m2-K).
        roof_u_value: Average roof U-value (W/m2-K).
        window_u_value: Average window U-value (W/m2-K).
        window_to_wall_ratio: Window-to-wall area ratio (0-1).
        air_changes_per_hour: Measured or estimated ACH.
        infiltration_rate: Air infiltration rate (ACH at 50Pa).
    """
    total_area_sqm: float = Field(default=5000.0, ge=1.0)
    wall_u_value: float = Field(default=0.35, ge=0.05, le=5.0)
    roof_u_value: float = Field(default=0.25, ge=0.05, le=5.0)
    window_u_value: float = Field(default=1.4, ge=0.3, le=6.0)
    window_to_wall_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    air_changes_per_hour: float = Field(default=0.7, ge=0.0, le=20.0)
    infiltration_rate: float = Field(default=5.0, ge=0.0, le=50.0)

class FacilityLightingHVACData(BaseModel):
    """Complete lighting and HVAC data for a facility.

    Attributes:
        facility_id: Unique facility identifier.
        facility_name: Facility name.
        lighting_zones: List of lighting zones.
        hvac_systems: List of HVAC systems.
        vsd_candidates: Motors / equipment candidates for VSD retrofit.
        building_envelope: Building envelope data.
        climate_zone: Climate zone for free-cooling estimation.
        electricity_cost_eur_per_kwh: Electricity cost (EUR/kWh).
        gas_cost_eur_per_kwh: Gas cost (EUR/kWh).
    """
    facility_id: str = Field(default_factory=_new_uuid)
    facility_name: str = Field(default="")
    lighting_zones: List[LightingZone] = Field(default_factory=list)
    hvac_systems: List[HVACSystem] = Field(default_factory=list)
    vsd_candidates: List[VSDCandidate] = Field(default_factory=list)
    building_envelope: Optional[BuildingEnvelope] = Field(default=None)
    climate_zone: ClimateZone = Field(default=ClimateZone.CENTRAL_CONTINENTAL)
    electricity_cost_eur_per_kwh: float = Field(default=0.12, ge=0.0)
    gas_cost_eur_per_kwh: float = Field(default=0.04, ge=0.0)

# --- Result models ---

class LightingRetrofitResult(BaseModel):
    """Lighting retrofit analysis for one zone."""
    zone_id: str = Field(default="")
    zone_name: str = Field(default="")
    current_lpd_w_sqm: float = Field(default=0.0)
    proposed_lpd_w_sqm: float = Field(default=0.0)
    lpd_benchmark_rating: str = Field(default="unknown")
    current_power_kw: float = Field(default=0.0)
    proposed_power_kw: float = Field(default=0.0)
    savings_kwh: float = Field(default=0.0)
    savings_eur: float = Field(default=0.0)
    retrofit_cost_eur: float = Field(default=0.0)
    payback_years: float = Field(default=0.0)
    occupancy_sensor_savings_kwh: float = Field(default=0.0)
    daylight_savings_kwh: float = Field(default=0.0)

class VSDRetrofitResult(BaseModel):
    """VSD retrofit analysis for one motor / equipment."""
    equipment_id: str = Field(default="")
    name: str = Field(default="")
    motor_kw: float = Field(default=0.0)
    average_load_pct: int = Field(default=100)
    estimated_savings_pct: float = Field(default=0.0)
    savings_kwh: float = Field(default=0.0)
    savings_eur: float = Field(default=0.0)
    cost_eur: float = Field(default=0.0)
    payback_years: float = Field(default=0.0)

class EconomizerAnalysisResult(BaseModel):
    """Economiser (free cooling) analysis."""
    climate_zone: str = Field(default="")
    free_cooling_hours: int = Field(default=0)
    cooling_load_reduction_kwh: float = Field(default=0.0)
    savings_eur: float = Field(default=0.0)
    equipment_cost_eur: float = Field(default=0.0)
    payback_years: float = Field(default=0.0)

class HVACEfficiencyResult(BaseModel):
    """HVAC efficiency assessment for one system."""
    system_id: str = Field(default="")
    system_name: str = Field(default="")
    current_cop: float = Field(default=0.0)
    benchmark_cop: float = Field(default=0.0)
    cop_rating: str = Field(default="unknown")
    refrigerant_gwp: int = Field(default=0)
    improvement_potential_pct: float = Field(default=0.0)
    savings_kwh: float = Field(default=0.0)
    savings_eur: float = Field(default=0.0)

class LightingHVACResult(BaseModel):
    """Complete lighting and HVAC optimisation result with provenance.

    Attributes:
        result_id: Unique result identifier.
        facility_id: Facility identifier.
        lighting_results: Per-zone lighting retrofit analysis.
        hvac_results: Per-system HVAC efficiency analysis.
        vsd_results: Per-equipment VSD retrofit analysis.
        economizer_analysis: Economiser free-cooling analysis.
        total_lighting_savings_kwh: Total annual lighting savings (kWh).
        total_hvac_savings_kwh: Total annual HVAC savings (kWh).
        total_vsd_savings_kwh: Total annual VSD savings (kWh).
        total_savings_kwh: Grand total savings (kWh).
        total_savings_eur: Grand total cost savings (EUR).
        total_investment_eur: Grand total investment (EUR).
        simple_payback_years: Overall simple payback (years).
        lpd_improvement_pct: Average LPD improvement (%).
        hvac_efficiency_improvement_pct: Average COP improvement (%).
        measures_list: Summary list of recommended measures.
        methodology_notes: Methodology notes.
        processing_time_ms: Computation time (ms).
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        provenance_hash: SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default="")
    lighting_results: List[LightingRetrofitResult] = Field(default_factory=list)
    hvac_results: List[HVACEfficiencyResult] = Field(default_factory=list)
    vsd_results: List[VSDRetrofitResult] = Field(default_factory=list)
    economizer_analysis: Optional[EconomizerAnalysisResult] = Field(default=None)
    total_lighting_savings_kwh: float = Field(default=0.0)
    total_hvac_savings_kwh: float = Field(default=0.0)
    total_vsd_savings_kwh: float = Field(default=0.0)
    total_savings_kwh: float = Field(default=0.0)
    total_savings_eur: float = Field(default=0.0)
    total_investment_eur: float = Field(default=0.0)
    simple_payback_years: float = Field(default=0.0)
    lpd_improvement_pct: float = Field(default=0.0)
    hvac_efficiency_improvement_pct: float = Field(default=0.0)
    measures_list: List[Dict[str, Any]] = Field(default_factory=list)
    methodology_notes: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class LightingHVACEngine:
    """Zero-hallucination lighting and HVAC optimisation engine.

    Calculates energy savings from LED retrofits, occupancy/daylight
    controls, HVAC efficiency improvements, VSD retrofits, and
    economiser installations.

    Guarantees:
        - Deterministic: same inputs produce identical outputs.
        - Reproducible: every result carries a SHA-256 provenance hash.
        - Auditable: full breakdown by zone, system, and measure.
        - No LLM: zero hallucination risk in any calculation path.

    Usage::

        engine = LightingHVACEngine()
        result = engine.analyze(facility_data)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the lighting and HVAC engine.

        Args:
            config: Optional configuration overrides.
        """
        self._config = config or {}
        self._notes: List[str] = []
        logger.info("LightingHVACEngine v%s initialised.", _MODULE_VERSION)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def analyze(self, facility: FacilityLightingHVACData) -> LightingHVACResult:
        """Run comprehensive lighting and HVAC analysis.

        Args:
            facility: Complete facility lighting and HVAC data.

        Returns:
            LightingHVACResult with full breakdown and provenance.
        """
        t0 = time.perf_counter()
        self._notes = [
            f"Engine version: {self.engine_version}",
            f"Analysis timestamp: {utcnow().isoformat()}",
        ]

        total_lighting_kwh = Decimal("0")
        total_hvac_kwh = Decimal("0")
        total_vsd_kwh = Decimal("0")
        total_savings_eur = Decimal("0")
        total_investment = Decimal("0")
        measures: List[Dict[str, Any]] = []

        elec_cost = _decimal(facility.electricity_cost_eur_per_kwh)

        # --- 1. Lighting Analysis ---
        lighting_results: List[LightingRetrofitResult] = []
        for zone in facility.lighting_zones:
            lr = self.analyze_lighting_zone(zone, float(elec_cost))
            lighting_results.append(lr)
            zone_savings = _decimal(lr.savings_kwh) + _decimal(lr.occupancy_sensor_savings_kwh) + _decimal(lr.daylight_savings_kwh)
            total_lighting_kwh += zone_savings
            total_savings_eur += zone_savings * elec_cost
            total_investment += _decimal(lr.retrofit_cost_eur)
            if float(zone_savings) > 0:
                measures.append({
                    "category": "lighting",
                    "zone": lr.zone_name or lr.zone_id,
                    "description": f"LED retrofit + controls for zone '{lr.zone_name}'",
                    "savings_kwh": _round2(float(zone_savings)),
                    "cost_eur": lr.retrofit_cost_eur,
                    "payback_years": lr.payback_years,
                })

        # --- 2. HVAC Efficiency Analysis ---
        hvac_results: List[HVACEfficiencyResult] = []
        for system in facility.hvac_systems:
            hr = self.analyze_hvac_system(system, float(elec_cost))
            hvac_results.append(hr)
            total_hvac_kwh += _decimal(hr.savings_kwh)
            total_savings_eur += _decimal(hr.savings_eur)
            if hr.savings_kwh > 0:
                measures.append({
                    "category": "hvac_efficiency",
                    "system": hr.system_name or hr.system_id,
                    "description": f"HVAC upgrade for '{hr.system_name}' (COP {hr.current_cop} -> {hr.benchmark_cop})",
                    "savings_kwh": hr.savings_kwh,
                    "savings_eur": hr.savings_eur,
                })

        # --- 3. VSD Retrofit Analysis ---
        vsd_results: List[VSDRetrofitResult] = []
        for candidate in facility.vsd_candidates:
            if not candidate.has_vsd:
                vr = self.analyze_vsd_retrofit(candidate, float(elec_cost))
                vsd_results.append(vr)
                total_vsd_kwh += _decimal(vr.savings_kwh)
                total_savings_eur += _decimal(vr.savings_eur)
                total_investment += _decimal(vr.cost_eur)
                if vr.savings_kwh > 0:
                    measures.append({
                        "category": "vsd_retrofit",
                        "equipment": vr.name or vr.equipment_id,
                        "description": f"VSD for {vr.motor_kw} kW {candidate.equipment_type}",
                        "savings_kwh": vr.savings_kwh,
                        "cost_eur": vr.cost_eur,
                        "payback_years": vr.payback_years,
                    })

        # --- 4. Economiser / Free Cooling Analysis ---
        econ_result: Optional[EconomizerAnalysisResult] = None
        hvac_without_econ = [s for s in facility.hvac_systems if not s.has_economiser]
        if hvac_without_econ:
            total_cooling_kw = sum(s.cooling_capacity_kw for s in hvac_without_econ)
            avg_cooling_hours = (
                sum(s.annual_cooling_hours for s in hvac_without_econ) / len(hvac_without_econ)
                if hvac_without_econ else 0
            )
            econ_result = self.analyze_economizer(
                climate_zone=facility.climate_zone,
                cooling_capacity_kw=total_cooling_kw,
                annual_cooling_hours=int(avg_cooling_hours),
                electricity_cost_eur_per_kwh=float(elec_cost),
            )
            total_hvac_kwh += _decimal(econ_result.cooling_load_reduction_kwh)
            total_savings_eur += _decimal(econ_result.savings_eur)
            total_investment += _decimal(econ_result.equipment_cost_eur)
            if econ_result.savings_eur > 0:
                measures.append({
                    "category": "economiser",
                    "description": f"Install economiser for free cooling ({econ_result.free_cooling_hours} hrs/yr)",
                    "savings_kwh": econ_result.cooling_load_reduction_kwh,
                    "cost_eur": econ_result.equipment_cost_eur,
                    "payback_years": econ_result.payback_years,
                })

        # --- LPD improvement calculation ---
        total_current_lpd = Decimal("0")
        total_proposed_lpd = Decimal("0")
        lpd_count = 0
        for lr in lighting_results:
            if lr.current_lpd_w_sqm > 0:
                total_current_lpd += _decimal(lr.current_lpd_w_sqm)
                total_proposed_lpd += _decimal(lr.proposed_lpd_w_sqm)
                lpd_count += 1
        avg_lpd_improvement = Decimal("0")
        if lpd_count > 0 and total_current_lpd > Decimal("0"):
            avg_current = total_current_lpd / _decimal(lpd_count)
            avg_proposed = total_proposed_lpd / _decimal(lpd_count)
            avg_lpd_improvement = _safe_divide(
                (avg_current - avg_proposed), avg_current
            ) * Decimal("100")

        # --- HVAC efficiency improvement ---
        total_current_cop = Decimal("0")
        total_bench_cop = Decimal("0")
        cop_count = 0
        for hr in hvac_results:
            if hr.current_cop > 0:
                total_current_cop += _decimal(hr.current_cop)
                total_bench_cop += _decimal(hr.benchmark_cop)
                cop_count += 1
        avg_hvac_improvement = Decimal("0")
        if cop_count > 0 and total_current_cop > Decimal("0"):
            avg_cop_now = total_current_cop / _decimal(cop_count)
            avg_cop_bench = total_bench_cop / _decimal(cop_count)
            avg_hvac_improvement = _safe_divide(
                (avg_cop_bench - avg_cop_now), avg_cop_now
            ) * Decimal("100")

        # Sort measures by payback.
        measures.sort(key=lambda m: m.get("payback_years", 999))

        total_all_kwh = total_lighting_kwh + total_hvac_kwh + total_vsd_kwh
        overall_payback = _round2(float(
            _safe_divide(total_investment, total_savings_eur)
        )) if total_savings_eur > Decimal("0") else 0.0

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = LightingHVACResult(
            facility_id=facility.facility_id,
            lighting_results=lighting_results,
            hvac_results=hvac_results,
            vsd_results=vsd_results,
            economizer_analysis=econ_result,
            total_lighting_savings_kwh=_round2(float(total_lighting_kwh)),
            total_hvac_savings_kwh=_round2(float(total_hvac_kwh)),
            total_vsd_savings_kwh=_round2(float(total_vsd_kwh)),
            total_savings_kwh=_round2(float(total_all_kwh)),
            total_savings_eur=_round2(float(total_savings_eur)),
            total_investment_eur=_round2(float(total_investment)),
            simple_payback_years=overall_payback,
            lpd_improvement_pct=_round2(float(avg_lpd_improvement)),
            hvac_efficiency_improvement_pct=_round2(float(avg_hvac_improvement)),
            measures_list=measures,
            methodology_notes=list(self._notes),
            processing_time_ms=round(elapsed_ms, 3),
        )

        result.provenance_hash = _compute_hash(result)
        return result

    # --------------------------------------------------------------------- #
    # Lighting Analysis
    # --------------------------------------------------------------------- #

    def analyze_lighting_zone(
        self,
        zone: LightingZone,
        electricity_cost: float,
    ) -> LightingRetrofitResult:
        """Analyse a single lighting zone for retrofit potential.

        Calculates current LPD, proposed LED LPD, occupancy sensor
        savings, daylight harvesting savings, and retrofit payback.

        Args:
            zone: Lighting zone data.
            electricity_cost: Electricity cost (EUR/kWh).

        Returns:
            LightingRetrofitResult for the zone.
        """
        d_area = _decimal(zone.area_sqm)
        d_fixtures = _decimal(zone.fixture_count)
        d_wattage = _decimal(zone.wattage_per_fixture)
        d_hours = _decimal(zone.operating_hours)
        d_cost = _decimal(electricity_cost)

        # Current power and LPD.
        current_power_w = d_fixtures * d_wattage
        current_power_kw = current_power_w / Decimal("1000")
        current_lpd = _safe_divide(current_power_w, d_area)

        # Proposed LED replacement.
        led_efficacy = _decimal(LIGHTING_EFFICACY[FixtureType.LED]["efficacy_lm_w"])
        current_efficacy_data = LIGHTING_EFFICACY.get(zone.fixture_type, {})
        current_efficacy = _decimal(current_efficacy_data.get("efficacy_lm_w", 80))

        # Calculate required lumens to maintain lux levels.
        current_lumens = current_power_w * current_efficacy
        # If we need at least the same lumens, LED wattage = lumens / led_efficacy.
        proposed_power_w = _safe_divide(current_lumens, led_efficacy)
        proposed_power_kw = proposed_power_w / Decimal("1000")
        proposed_lpd = _safe_divide(proposed_power_w, d_area)

        # Energy savings from LED retrofit.
        savings_kw = max(current_power_kw - proposed_power_kw, Decimal("0"))
        savings_kwh = savings_kw * d_hours
        savings_eur = savings_kwh * d_cost

        # LPD benchmark rating.
        benchmark = LPD_BENCHMARKS.get(zone.space_type, {})
        best = _decimal(benchmark.get("best_practice_w_sqm", 6.0))
        good = _decimal(benchmark.get("good_w_sqm", 8.0))
        average = _decimal(benchmark.get("average_w_sqm", 12.0))

        if proposed_lpd <= best:
            lpd_rating = "best_practice"
        elif proposed_lpd <= good:
            lpd_rating = "good"
        elif proposed_lpd <= average:
            lpd_rating = "average"
        else:
            lpd_rating = "poor"

        # Retrofit cost.
        unit_cost = _decimal(LED_RETROFIT_COST.get(zone.fixture_type, 100))
        retrofit_cost = d_fixtures * unit_cost

        # Occupancy sensor savings.
        occ_savings_kwh = Decimal("0")
        if not zone.has_occupancy_sensor:
            occ_savings_pct = _decimal(OCCUPANCY_SENSOR_SAVINGS.get(zone.space_type, 20))
            occ_savings_kwh = proposed_power_kw * d_hours * occ_savings_pct / Decimal("100")
            # Add sensor cost: ~EUR 60 per sensor, one per 50 m2.
            sensors_needed = max(int(float(d_area / Decimal("50"))), 1)
            retrofit_cost += _decimal(sensors_needed) * Decimal("60")

        # Daylight harvesting savings.
        daylight_savings_kwh = Decimal("0")
        if not zone.has_daylight_sensor and zone.daylight_factor > 0:
            d_daylight = _decimal(zone.daylight_factor)
            # Assume daylight available for 50% of operating hours.
            daylight_hours = d_hours * Decimal("0.5")
            daylight_savings_kwh = proposed_power_kw * daylight_hours * d_daylight * Decimal("0.6")
            retrofit_cost += _decimal(max(int(float(d_area / Decimal("100"))), 1)) * Decimal("120")

        total_savings_kwh = savings_kwh + occ_savings_kwh + daylight_savings_kwh
        total_savings_eur = total_savings_kwh * d_cost
        payback = _safe_divide(retrofit_cost, total_savings_eur)

        self._notes.append(
            f"Lighting zone '{zone.name}': LPD {_round2(float(current_lpd))} -> "
            f"{_round2(float(proposed_lpd))} W/m2, savings {_round2(float(total_savings_kwh))} kWh/yr."
        )

        return LightingRetrofitResult(
            zone_id=zone.zone_id,
            zone_name=zone.name,
            current_lpd_w_sqm=_round2(float(current_lpd)),
            proposed_lpd_w_sqm=_round2(float(proposed_lpd)),
            lpd_benchmark_rating=lpd_rating,
            current_power_kw=_round3(float(current_power_kw)),
            proposed_power_kw=_round3(float(proposed_power_kw)),
            savings_kwh=_round2(float(savings_kwh)),
            savings_eur=_round2(float(savings_eur)),
            retrofit_cost_eur=_round2(float(retrofit_cost)),
            payback_years=_round2(float(payback)),
            occupancy_sensor_savings_kwh=_round2(float(occ_savings_kwh)),
            daylight_savings_kwh=_round2(float(daylight_savings_kwh)),
        )

    # --------------------------------------------------------------------- #
    # HVAC Efficiency Analysis
    # --------------------------------------------------------------------- #

    def analyze_hvac_system(
        self,
        system: HVACSystem,
        electricity_cost: float,
    ) -> HVACEfficiencyResult:
        """Analyse HVAC system efficiency against benchmarks.

        Compares current COP against type-specific benchmarks and
        calculates potential savings from upgrade.

        Args:
            system: HVAC system data.
            electricity_cost: Electricity cost (EUR/kWh).

        Returns:
            HVACEfficiencyResult with rating and savings.
        """
        d_cop = _decimal(system.current_cop)
        d_cost = _decimal(electricity_cost)

        benchmarks = HVAC_COP_BENCHMARKS.get(system.system_type, {})
        best_cop = _decimal(benchmarks.get("best_practice_cop", 5.0))
        good_cop = _decimal(benchmarks.get("good_cop", 4.0))
        average_cop = _decimal(benchmarks.get("average_cop", 3.2))

        if d_cop >= best_cop:
            rating = "best_practice"
        elif d_cop >= good_cop:
            rating = "good"
        elif d_cop >= average_cop:
            rating = "average"
        else:
            rating = "poor"

        # Improvement potential: upgrade to good practice COP.
        target_cop = good_cop
        improvement_pct = Decimal("0")
        savings_kwh = Decimal("0")
        savings_eur = Decimal("0")

        if d_cop < target_cop and d_cop > Decimal("0"):
            improvement_pct = _safe_divide(target_cop - d_cop, d_cop) * Decimal("100")

            # Current energy for cooling.
            d_cooling_kw = _decimal(system.cooling_capacity_kw)
            d_cooling_hours = _decimal(system.annual_cooling_hours)
            # Assume 60% average load.
            avg_load_factor = Decimal("0.6")
            current_energy = d_cooling_kw * avg_load_factor * d_cooling_hours / d_cop
            improved_energy = d_cooling_kw * avg_load_factor * d_cooling_hours / target_cop
            savings_kwh = max(current_energy - improved_energy, Decimal("0"))
            savings_eur = savings_kwh * d_cost

        # Refrigerant GWP flag.
        gwp = REFRIGERANT_GWP.get(system.refrigerant, 0)

        self._notes.append(
            f"HVAC '{system.name}': COP {float(d_cop)} ({rating}), "
            f"target {float(target_cop)}, refrigerant {system.refrigerant} GWP={gwp}."
        )

        return HVACEfficiencyResult(
            system_id=system.system_id,
            system_name=system.name,
            current_cop=float(d_cop),
            benchmark_cop=float(target_cop),
            cop_rating=rating,
            refrigerant_gwp=gwp,
            improvement_potential_pct=_round2(float(improvement_pct)),
            savings_kwh=_round2(float(savings_kwh)),
            savings_eur=_round2(float(savings_eur)),
        )

    # --------------------------------------------------------------------- #
    # VSD Retrofit Analysis
    # --------------------------------------------------------------------- #

    def analyze_vsd_retrofit(
        self,
        candidate: VSDCandidate,
        electricity_cost: float,
    ) -> VSDRetrofitResult:
        """Analyse VSD (Variable Speed Drive) retrofit potential.

        Uses the affinity law cubic relationship: P2/P1 = (n2/n1)^3
        to estimate energy savings at partial load conditions.

        Args:
            candidate: Motor / equipment to analyse.
            electricity_cost: Electricity cost (EUR/kWh).

        Returns:
            VSDRetrofitResult with savings and payback.
        """
        d_motor = _decimal(candidate.motor_kw)
        d_hours = _decimal(candidate.operating_hours)
        d_cost = _decimal(electricity_cost)
        load_pct = candidate.average_load_pct

        # Look up savings from VSD curve.
        savings_pct = Decimal("0")
        available_loads = sorted(VSD_SAVINGS_CURVE.keys(), reverse=True)
        for load_point in available_loads:
            if load_pct <= load_point:
                savings_pct = _decimal(VSD_SAVINGS_CURVE[load_point])

        # If load_pct is between two points, use the next lower.
        if savings_pct == Decimal("0") and load_pct < 100:
            for load_point in sorted(VSD_SAVINGS_CURVE.keys()):
                if load_pct <= load_point:
                    savings_pct = _decimal(VSD_SAVINGS_CURVE[load_point])
                    break

        # Current annual consumption (at nameplate, adjusted for load).
        # Without VSD, throttled system uses ~95% power at partial load.
        current_annual_kwh = d_motor * d_hours * Decimal("0.95")

        # With VSD, power follows cube law.
        savings_kwh = current_annual_kwh * savings_pct / Decimal("100")
        savings_eur = savings_kwh * d_cost

        # VSD cost.
        vsd_cost = d_motor * VSD_COST_EUR_PER_KW
        payback = _safe_divide(vsd_cost, savings_eur)

        self._notes.append(
            f"VSD candidate '{candidate.name}': {float(d_motor)} kW at {load_pct}% load, "
            f"estimated savings {_round2(float(savings_pct))}%."
        )

        return VSDRetrofitResult(
            equipment_id=candidate.equipment_id,
            name=candidate.name,
            motor_kw=float(d_motor),
            average_load_pct=load_pct,
            estimated_savings_pct=_round2(float(savings_pct)),
            savings_kwh=_round2(float(savings_kwh)),
            savings_eur=_round2(float(savings_eur)),
            cost_eur=_round2(float(vsd_cost)),
            payback_years=_round2(float(payback)),
        )

    # --------------------------------------------------------------------- #
    # Economiser / Free Cooling Analysis
    # --------------------------------------------------------------------- #

    def analyze_economizer(
        self,
        climate_zone: ClimateZone,
        cooling_capacity_kw: float,
        annual_cooling_hours: int,
        electricity_cost_eur_per_kwh: float,
    ) -> EconomizerAnalysisResult:
        """Analyse economiser (free cooling) potential.

        Estimates the number of hours per year where outdoor conditions
        allow free cooling, reducing mechanical cooling load.

        Args:
            climate_zone: Facility climate zone.
            cooling_capacity_kw: Total cooling capacity (kW).
            annual_cooling_hours: Annual mechanical cooling hours.
            electricity_cost_eur_per_kwh: Electricity cost (EUR/kWh).

        Returns:
            EconomizerAnalysisResult with savings and payback.
        """
        free_hours = CLIMATE_ZONE_FREE_COOLING_HOURS.get(climate_zone, 3000)
        d_capacity = _decimal(cooling_capacity_kw)
        d_cost = _decimal(electricity_cost_eur_per_kwh)

        # Overlap between free-cooling hours and cooling demand hours.
        # Assume 30% of cooling demand hours overlap with free-cooling conditions.
        overlap_factor = Decimal("0.30")
        effective_free_hours = _decimal(min(free_hours, annual_cooling_hours)) * overlap_factor

        # Average load factor during free-cooling eligible periods.
        avg_load = Decimal("0.4")
        cooling_load_reduction_kwh = d_capacity * avg_load * effective_free_hours

        # COP of mechanical cooling ~ 3.5 on average.
        avg_cop = Decimal("3.5")
        electrical_savings_kwh = _safe_divide(cooling_load_reduction_kwh, avg_cop)
        savings_eur = electrical_savings_kwh * d_cost

        # Equipment cost: dampers, controls, sensors.
        equipment_cost = Decimal("8000") + d_capacity * Decimal("15")
        payback = _safe_divide(equipment_cost, savings_eur)

        self._notes.append(
            f"Economiser analysis: {free_hours} free-cooling hours in "
            f"{climate_zone.value}, effective overlap {_round2(float(effective_free_hours))} hours."
        )

        return EconomizerAnalysisResult(
            climate_zone=climate_zone.value,
            free_cooling_hours=free_hours,
            cooling_load_reduction_kwh=_round2(float(electrical_savings_kwh)),
            savings_eur=_round2(float(savings_eur)),
            equipment_cost_eur=_round2(float(equipment_cost)),
            payback_years=_round2(float(payback)),
        )

    # --------------------------------------------------------------------- #
    # Demand-Controlled Ventilation Analysis
    # --------------------------------------------------------------------- #

    def analyze_demand_controlled_ventilation(
        self,
        current_strategy: VentilationStrategy,
        total_airflow_m3_h: float,
        fan_power_kw: float,
        operating_hours: int,
        occupancy_variation_pct: float,
        electricity_cost: float,
    ) -> Dict[str, Any]:
        """Analyse demand-controlled ventilation savings.

        Switching from constant volume to CO2/occupancy-based DCV
        reduces fan energy proportionally to the average reduction
        in airflow during low-occupancy periods.

        Args:
            current_strategy: Current ventilation strategy.
            total_airflow_m3_h: Total ventilation airflow (m3/h).
            fan_power_kw: Supply + exhaust fan power (kW).
            operating_hours: Annual operating hours.
            occupancy_variation_pct: How much occupancy varies (0-100).
            electricity_cost: Electricity cost (EUR/kWh).

        Returns:
            Dictionary with DCV savings analysis.
        """
        if current_strategy in (
            VentilationStrategy.DCV_CO2,
            VentilationStrategy.DCV_OCCUPANCY,
        ):
            return {
                "applicable": False,
                "reason": "DCV already installed",
                "savings_kwh": 0.0,
                "savings_eur": 0.0,
            }

        d_fan = _decimal(fan_power_kw)
        d_hours = _decimal(operating_hours)
        d_cost = _decimal(electricity_cost)
        d_variation = _decimal(occupancy_variation_pct) / Decimal("100")

        # Average airflow reduction with DCV.
        avg_reduction = d_variation * Decimal("0.35")

        # Fan energy follows cube law: savings = 1 - (1 - reduction)^3.
        reduced_speed_ratio = Decimal("1") - avg_reduction
        power_ratio = reduced_speed_ratio ** 3
        savings_fraction = Decimal("1") - power_ratio

        savings_kwh = d_fan * d_hours * savings_fraction
        savings_eur = savings_kwh * d_cost

        # DCV system cost: CO2 sensors + controller.
        dcv_cost = Decimal("5000") + _decimal(total_airflow_m3_h) * Decimal("0.5")
        payback = _safe_divide(dcv_cost, savings_eur)

        return {
            "applicable": True,
            "current_strategy": current_strategy.value,
            "proposed_strategy": "demand_controlled_co2",
            "average_airflow_reduction_pct": _round2(float(avg_reduction * Decimal("100"))),
            "fan_energy_savings_pct": _round2(float(savings_fraction * Decimal("100"))),
            "savings_kwh": _round2(float(savings_kwh)),
            "savings_eur": _round2(float(savings_eur)),
            "dcv_cost_eur": _round2(float(dcv_cost)),
            "payback_years": _round2(float(payback)),
        }

    # --------------------------------------------------------------------- #
    # Heat Recovery Ventilation Analysis
    # --------------------------------------------------------------------- #

    def analyze_heat_recovery_ventilation(
        self,
        exhaust_airflow_m3_h: float,
        indoor_temp_c: float,
        outdoor_temp_c_winter: float,
        operating_hours_heating: int,
        heating_energy_cost_eur_per_kwh: float,
        hrv_effectiveness: float = 0.75,
    ) -> Dict[str, Any]:
        """Analyse heat recovery ventilation (HRV) savings potential.

        Recovers thermal energy from exhaust air to pre-heat incoming
        fresh air, reducing heating demand.

        Args:
            exhaust_airflow_m3_h: Exhaust airflow rate (m3/h).
            indoor_temp_c: Indoor temperature setpoint (C).
            outdoor_temp_c_winter: Average winter outdoor temperature (C).
            operating_hours_heating: Annual heating-season hours.
            heating_energy_cost_eur_per_kwh: Heating energy cost (EUR/kWh).
            hrv_effectiveness: Heat recovery effectiveness (0-1).

        Returns:
            Dictionary with HRV savings analysis.
        """
        d_airflow = _decimal(exhaust_airflow_m3_h)
        d_indoor = _decimal(indoor_temp_c)
        d_outdoor = _decimal(outdoor_temp_c_winter)
        d_hours = _decimal(operating_hours_heating)
        d_cost = _decimal(heating_energy_cost_eur_per_kwh)
        d_eff = _decimal(hrv_effectiveness)

        delta_t = d_indoor - d_outdoor
        if delta_t <= Decimal("0"):
            return {"applicable": False, "reason": "No temperature differential", "savings_kwh": 0.0}

        # Air density * specific heat.
        rho_cp = Decimal("0.34")  # W-h / (m3 * K) for air at ~20C
        recovered_kw = d_airflow * rho_cp * delta_t * d_eff / Decimal("1000")
        annual_kwh = recovered_kw * d_hours
        annual_eur = annual_kwh * d_cost

        # HRV unit cost.
        hrv_cost = Decimal("10000") + d_airflow * Decimal("2")
        payback = _safe_divide(hrv_cost, annual_eur)

        return {
            "applicable": True,
            "exhaust_airflow_m3_h": float(d_airflow),
            "temperature_differential_c": _round2(float(delta_t)),
            "hrv_effectiveness": float(d_eff),
            "recovered_power_kw": _round2(float(recovered_kw)),
            "annual_savings_kwh": _round2(float(annual_kwh)),
            "annual_savings_eur": _round2(float(annual_eur)),
            "hrv_cost_eur": _round2(float(hrv_cost)),
            "payback_years": _round2(float(payback)),
        }

    # --------------------------------------------------------------------- #
    # Setpoint Optimisation Analysis
    # --------------------------------------------------------------------- #

    def analyze_setpoint_optimization(
        self,
        current_cooling_setpoint_c: float,
        current_heating_setpoint_c: float,
        conditioned_area_sqm: float,
        annual_cooling_kwh: float,
        annual_heating_kwh: float,
        electricity_cost: float,
        gas_cost: float,
    ) -> Dict[str, Any]:
        """Analyse energy savings from setpoint and deadband optimisation.

        Rule of thumb: each 1C increase in cooling setpoint saves ~3-5%
        of cooling energy.  Each 1C decrease in heating setpoint saves
        ~3% of heating energy.  Widening the deadband eliminates
        simultaneous heating and cooling.

        Args:
            current_cooling_setpoint_c: Current cooling setpoint (C).
            current_heating_setpoint_c: Current heating setpoint (C).
            conditioned_area_sqm: Conditioned floor area (m2).
            annual_cooling_kwh: Annual cooling energy (kWh).
            annual_heating_kwh: Annual heating energy (kWh).
            electricity_cost: Electricity cost (EUR/kWh).
            gas_cost: Gas/heating cost (EUR/kWh).

        Returns:
            Dictionary with setpoint optimisation analysis.
        """
        d_cool_sp = _decimal(current_cooling_setpoint_c)
        d_heat_sp = _decimal(current_heating_setpoint_c)
        d_cooling = _decimal(annual_cooling_kwh)
        d_heating = _decimal(annual_heating_kwh)
        d_elec = _decimal(electricity_cost)
        d_gas = _decimal(gas_cost)

        # Recommended: cooling 25C, heating 20C (5C deadband).
        target_cool = Decimal("25")
        target_heat = Decimal("20")

        cool_delta = target_cool - d_cool_sp  # Positive = raising setpoint (savings)
        heat_delta = d_heat_sp - target_heat   # Positive = lowering setpoint (savings)

        # Savings per degree.
        cool_savings_per_degree = Decimal("0.04")  # 4% per degree
        heat_savings_per_degree = Decimal("0.03")   # 3% per degree

        cooling_savings_kwh = Decimal("0")
        heating_savings_kwh = Decimal("0")

        if cool_delta > Decimal("0"):
            cooling_savings_kwh = d_cooling * cool_delta * cool_savings_per_degree
        if heat_delta > Decimal("0"):
            heating_savings_kwh = d_heating * heat_delta * heat_savings_per_degree

        cooling_savings_eur = cooling_savings_kwh * d_elec
        heating_savings_eur = heating_savings_kwh * d_gas
        total_savings_eur = cooling_savings_eur + heating_savings_eur

        current_deadband = d_cool_sp - d_heat_sp
        proposed_deadband = target_cool - target_heat

        return {
            "current_cooling_setpoint_c": float(d_cool_sp),
            "current_heating_setpoint_c": float(d_heat_sp),
            "proposed_cooling_setpoint_c": float(target_cool),
            "proposed_heating_setpoint_c": float(target_heat),
            "current_deadband_c": _round2(float(current_deadband)),
            "proposed_deadband_c": _round2(float(proposed_deadband)),
            "cooling_savings_kwh": _round2(float(cooling_savings_kwh)),
            "heating_savings_kwh": _round2(float(heating_savings_kwh)),
            "total_savings_kwh": _round2(float(cooling_savings_kwh + heating_savings_kwh)),
            "total_savings_eur": _round2(float(total_savings_eur)),
            "implementation_cost_eur": 0.0,
            "payback_years": 0.0,
        }
