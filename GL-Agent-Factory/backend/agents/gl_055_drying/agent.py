"""GL-055: Drying Process Optimizer Agent (DRYING).

Optimizes industrial drying processes for energy efficiency and product quality.
Implements psychrometric calculations, drying rate curves, and mass/energy balances.

Standards: ISO 13061 (Wood), ASABE S448.1 (Grain), ASTM E96 (Water Vapor Transmission)

Physics-Based Calculations:
- Psychrometric properties (ASHRAE Fundamentals)
- Drying rate curves (constant and falling rate periods)
- Heat and mass transfer coefficients
- Lewis number correlation
- Critical moisture content determination

Zero-Hallucination: All calculations use deterministic thermodynamic equations.
"""
import hashlib
import json
import logging
import math
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class DryerType(str, Enum):
    """Industrial dryer classification."""
    CONVECTIVE = "CONVECTIVE"           # Hot air dryers (spray, flash, rotary)
    CONDUCTIVE = "CONDUCTIVE"           # Drum, cylinder dryers
    RADIANT = "RADIANT"                 # Infrared dryers
    MICROWAVE = "MICROWAVE"             # Dielectric heating
    VACUUM = "VACUUM"                   # Low pressure drying
    FLUIDIZED_BED = "FLUIDIZED_BED"     # Suspended particle drying
    FREEZE = "FREEZE"                   # Lyophilization
    SPRAY = "SPRAY"                     # Atomized liquid drying
    ROTARY = "ROTARY"                   # Tumbling drum dryer
    TRAY = "TRAY"                       # Batch tray dryers


class MaterialType(str, Enum):
    """Material classification for drying properties."""
    WOOD = "WOOD"                       # Lumber, veneer
    GRAIN = "GRAIN"                     # Corn, wheat, rice
    PAPER = "PAPER"                     # Pulp, paper products
    CERAMIC = "CERAMIC"                 # Clay, brick, tiles
    CHEMICAL = "CHEMICAL"               # Salts, powders
    FOOD = "FOOD"                       # Fruits, vegetables
    TEXTILE = "TEXTILE"                 # Fabric, yarn
    PHARMACEUTICAL = "PHARMACEUTICAL"   # API, excipients
    SLUDGE = "SLUDGE"                   # Wastewater biosolids
    BIOMASS = "BIOMASS"                 # Wood chips, pellets


class DryingPhase(str, Enum):
    """Drying rate period classification."""
    PREHEATING = "PREHEATING"           # Initial heating, minimal moisture loss
    CONSTANT_RATE = "CONSTANT_RATE"     # Surface moisture evaporation
    FIRST_FALLING = "FIRST_FALLING"     # Unsaturated surface drying
    SECOND_FALLING = "SECOND_FALLING"   # Internal diffusion limited


class AirFlowPattern(str, Enum):
    """Air flow configuration in dryer."""
    PARALLEL = "PARALLEL"               # Co-current
    COUNTER = "COUNTER"                 # Counter-current
    CROSS = "CROSS"                     # Perpendicular flow
    THROUGH = "THROUGH"                 # Air through material bed


# =============================================================================
# PYDANTIC INPUT/OUTPUT MODELS
# =============================================================================

class MaterialProperties(BaseModel):
    """Material-specific drying properties."""
    critical_moisture_db: float = Field(
        default=0.3,
        ge=0, le=5,
        description="Critical moisture content (dry basis kg/kg)"
    )
    equilibrium_moisture_db: float = Field(
        default=0.05,
        ge=0, le=1,
        description="Equilibrium moisture content (dry basis kg/kg)"
    )
    bulk_density_kg_m3: float = Field(
        default=500,
        gt=0,
        description="Bulk density of material"
    )
    specific_heat_kj_kg_k: float = Field(
        default=2.0,
        gt=0,
        description="Specific heat of dry material"
    )
    thermal_conductivity_w_mk: float = Field(
        default=0.2,
        gt=0,
        description="Thermal conductivity"
    )
    particle_diameter_mm: float = Field(
        default=5.0,
        gt=0,
        description="Characteristic particle/piece diameter"
    )
    max_temp_c: float = Field(
        default=100,
        gt=0,
        description="Maximum allowable temperature (quality limit)"
    )


class DryerGeometry(BaseModel):
    """Dryer physical dimensions."""
    length_m: float = Field(default=10.0, gt=0, description="Dryer length")
    width_m: float = Field(default=2.0, gt=0, description="Dryer width")
    height_m: float = Field(default=1.5, gt=0, description="Dryer height or diameter")
    bed_depth_m: float = Field(default=0.1, gt=0, description="Material bed depth")
    exposed_area_m2: Optional[float] = Field(
        default=None,
        description="Total exposed drying area (calculated if not provided)"
    )


class DryingInput(BaseModel):
    """Comprehensive input model for drying process optimization."""

    # Equipment identification
    equipment_id: str = Field(..., min_length=1, description="Unique equipment identifier")
    equipment_name: Optional[str] = Field(default=None, description="Equipment name/tag")

    # Dryer configuration
    dryer_type: DryerType = Field(default=DryerType.CONVECTIVE)
    air_flow_pattern: AirFlowPattern = Field(default=AirFlowPattern.CROSS)

    # Material properties
    material_type: MaterialType = Field(default=MaterialType.WOOD)
    material_properties: Optional[MaterialProperties] = Field(default=None)

    # Throughput and moisture
    throughput_kg_hr: float = Field(..., gt=0, description="Wet material feed rate")
    initial_moisture_wb: float = Field(
        ..., ge=0, le=99,
        description="Initial moisture content (wet basis %)"
    )
    target_moisture_wb: float = Field(
        ..., ge=0, le=99,
        description="Target moisture content (wet basis %)"
    )

    # Inlet air conditions
    inlet_air_temp_c: float = Field(
        default=80, gt=0, le=500,
        description="Drying air temperature"
    )
    inlet_air_humidity_rh: float = Field(
        default=20, ge=0, le=100,
        description="Inlet air relative humidity (%)"
    )
    ambient_temp_c: float = Field(
        default=25, ge=-40, le=50,
        description="Ambient temperature"
    )
    ambient_rh_pct: float = Field(
        default=50, ge=0, le=100,
        description="Ambient relative humidity (%)"
    )
    atmospheric_pressure_kpa: float = Field(
        default=101.325, gt=50, le=110,
        description="Local atmospheric pressure"
    )

    # Air flow
    air_velocity_m_s: float = Field(
        default=2.0, gt=0, le=30,
        description="Drying air velocity"
    )
    air_mass_flow_kg_s: Optional[float] = Field(
        default=None, gt=0,
        description="Total air mass flow rate"
    )
    recirculation_ratio: float = Field(
        default=0.0, ge=0, le=0.9,
        description="Fraction of exhaust air recirculated"
    )

    # Dryer geometry
    geometry: Optional[DryerGeometry] = Field(default=None)
    residence_time_min: float = Field(
        default=60, gt=0,
        description="Average material residence time"
    )

    # Energy parameters
    energy_source: str = Field(default="natural_gas")
    heater_efficiency_pct: float = Field(
        default=85, gt=0, le=100,
        description="Heating system efficiency"
    )
    fan_power_kw: Optional[float] = Field(
        default=None, ge=0,
        description="Total fan power consumption"
    )
    heat_recovery_efficiency: float = Field(
        default=0, ge=0, le=0.8,
        description="Heat recovery system efficiency"
    )

    # Economic parameters
    energy_cost_kwh: float = Field(default=0.05, ge=0)
    operating_hours_year: int = Field(default=6000, gt=0, le=8760)

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('target_moisture_wb')
    @classmethod
    def validate_target_moisture(cls, v, info):
        """Ensure target moisture is less than initial."""
        if 'initial_moisture_wb' in info.data and v >= info.data['initial_moisture_wb']:
            raise ValueError("Target moisture must be less than initial moisture")
        return v


class PsychrometricState(BaseModel):
    """Psychrometric properties of moist air."""
    dry_bulb_temp_c: float = Field(..., description="Dry bulb temperature")
    wet_bulb_temp_c: float = Field(..., description="Wet bulb temperature")
    dew_point_temp_c: float = Field(..., description="Dew point temperature")
    relative_humidity_pct: float = Field(..., ge=0, le=100)
    humidity_ratio_kg_kg: float = Field(..., ge=0, description="kg water/kg dry air")
    specific_enthalpy_kj_kg: float = Field(..., description="kJ/kg dry air")
    specific_volume_m3_kg: float = Field(..., gt=0, description="m3/kg dry air")
    vapor_pressure_kpa: float = Field(..., ge=0)
    saturation_pressure_kpa: float = Field(..., gt=0)


class DryingRateAnalysis(BaseModel):
    """Drying rate curve analysis results."""
    current_phase: DryingPhase = Field(..., description="Current drying phase")
    constant_rate_kg_m2_hr: float = Field(..., ge=0)
    actual_rate_kg_m2_hr: float = Field(..., ge=0)
    critical_moisture_db: float = Field(..., ge=0)
    equilibrium_moisture_db: float = Field(..., ge=0)
    normalized_moisture: float = Field(..., ge=0, le=1)
    mass_transfer_coeff_m_s: float = Field(..., ge=0)
    heat_transfer_coeff_w_m2k: float = Field(..., ge=0)
    lewis_number: float = Field(..., gt=0)
    estimated_drying_time_hr: float = Field(..., ge=0)


class EnergyBalance(BaseModel):
    """Energy balance for drying process."""
    # Heat inputs
    air_heating_kw: float = Field(..., ge=0, description="Heat to raise air temperature")
    latent_heat_kw: float = Field(..., ge=0, description="Heat for moisture evaporation")
    material_heating_kw: float = Field(..., ge=0, description="Sensible heat to material")
    total_heat_input_kw: float = Field(..., ge=0)

    # Heat recovery and losses
    heat_recovery_kw: float = Field(default=0, ge=0)
    wall_losses_kw: float = Field(default=0, ge=0)
    exhaust_losses_kw: float = Field(default=0, ge=0)
    net_heat_required_kw: float = Field(..., ge=0)

    # Efficiency metrics
    thermal_efficiency_pct: float = Field(..., ge=0, le=100)
    specific_energy_kj_kg_water: float = Field(..., ge=0)
    specific_energy_kwh_kg_water: float = Field(..., ge=0)

    # Fuel/energy consumption
    fuel_input_kw: float = Field(..., ge=0)
    fan_power_kw: float = Field(default=0, ge=0)
    total_power_kw: float = Field(..., ge=0)


class OptimizationRecommendation(BaseModel):
    """Structured optimization recommendation."""
    category: str = Field(..., description="Recommendation category")
    priority: str = Field(..., description="HIGH, MEDIUM, LOW")
    current_value: str = Field(...)
    recommended_value: str = Field(...)
    estimated_savings_pct: float = Field(default=0, ge=0)
    payback_months: Optional[float] = Field(default=None, ge=0)
    description: str = Field(...)


class DryingOutput(BaseModel):
    """Comprehensive output model for drying optimization."""

    # Identification
    equipment_id: str
    agent_id: str = Field(default="GL-055")
    agent_version: str = Field(default="1.0.0")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Mass balance results
    dry_solids_rate_kg_hr: float = Field(..., description="Dry material throughput")
    water_removed_kg_hr: float = Field(..., description="Evaporation rate")
    product_rate_kg_hr: float = Field(..., description="Dry product output rate")
    initial_moisture_db: float = Field(..., description="Initial moisture dry basis")
    final_moisture_db: float = Field(..., description="Final moisture dry basis")

    # Psychrometric analysis
    inlet_air_state: PsychrometricState
    outlet_air_state: PsychrometricState
    air_mass_flow_kg_s: float = Field(..., gt=0)
    air_volume_flow_m3_s: float = Field(..., gt=0)

    # Drying rate analysis
    drying_rate_analysis: DryingRateAnalysis
    exposed_area_m2: float = Field(..., gt=0)

    # Energy analysis
    energy_balance: EnergyBalance
    annual_energy_kwh: float = Field(..., ge=0)
    annual_energy_cost_usd: float = Field(..., ge=0)

    # Optimization results
    optimal_air_temp_c: float = Field(..., gt=0)
    optimal_air_velocity_m_s: float = Field(..., gt=0)
    optimal_recirculation_ratio: float = Field(..., ge=0, le=0.9)
    potential_energy_savings_pct: float = Field(default=0, ge=0)

    # Quality indicators
    product_quality_risk: str = Field(..., description="LOW, MEDIUM, HIGH")
    case_hardening_risk: bool = Field(default=False)
    over_drying_risk: bool = Field(default=False)

    # Recommendations
    recommendations: List[OptimizationRecommendation] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # Audit trail
    calculation_hash: str = Field(..., description="SHA-256 provenance hash")
    input_hash: str = Field(..., description="Hash of input parameters")
    calculation_method: str = Field(default="psychrometric_balance")
    standards_applied: List[str] = Field(default_factory=list)


# =============================================================================
# MATERIAL PROPERTY DATABASE
# =============================================================================

MATERIAL_DEFAULTS: Dict[MaterialType, Dict[str, float]] = {
    MaterialType.WOOD: {
        "critical_moisture_db": 0.30,
        "equilibrium_moisture_db": 0.08,
        "bulk_density_kg_m3": 450,
        "specific_heat_kj_kg_k": 1.7,
        "thermal_conductivity_w_mk": 0.15,
        "particle_diameter_mm": 25,
        "max_temp_c": 80,  # Prevent case hardening
    },
    MaterialType.GRAIN: {
        "critical_moisture_db": 0.20,
        "equilibrium_moisture_db": 0.12,
        "bulk_density_kg_m3": 720,
        "specific_heat_kj_kg_k": 1.5,
        "thermal_conductivity_w_mk": 0.12,
        "particle_diameter_mm": 5,
        "max_temp_c": 55,  # Preserve germination
    },
    MaterialType.PAPER: {
        "critical_moisture_db": 0.15,
        "equilibrium_moisture_db": 0.06,
        "bulk_density_kg_m3": 700,
        "specific_heat_kj_kg_k": 1.4,
        "thermal_conductivity_w_mk": 0.06,
        "particle_diameter_mm": 0.2,
        "max_temp_c": 120,
    },
    MaterialType.CERAMIC: {
        "critical_moisture_db": 0.10,
        "equilibrium_moisture_db": 0.01,
        "bulk_density_kg_m3": 1800,
        "specific_heat_kj_kg_k": 0.9,
        "thermal_conductivity_w_mk": 0.7,
        "particle_diameter_mm": 10,
        "max_temp_c": 200,
    },
    MaterialType.CHEMICAL: {
        "critical_moisture_db": 0.08,
        "equilibrium_moisture_db": 0.02,
        "bulk_density_kg_m3": 800,
        "specific_heat_kj_kg_k": 1.2,
        "thermal_conductivity_w_mk": 0.3,
        "particle_diameter_mm": 2,
        "max_temp_c": 150,
    },
    MaterialType.FOOD: {
        "critical_moisture_db": 0.50,
        "equilibrium_moisture_db": 0.10,
        "bulk_density_kg_m3": 600,
        "specific_heat_kj_kg_k": 2.5,
        "thermal_conductivity_w_mk": 0.4,
        "particle_diameter_mm": 10,
        "max_temp_c": 70,
    },
    MaterialType.TEXTILE: {
        "critical_moisture_db": 0.20,
        "equilibrium_moisture_db": 0.06,
        "bulk_density_kg_m3": 300,
        "specific_heat_kj_kg_k": 1.3,
        "thermal_conductivity_w_mk": 0.05,
        "particle_diameter_mm": 1,
        "max_temp_c": 100,
    },
    MaterialType.PHARMACEUTICAL: {
        "critical_moisture_db": 0.05,
        "equilibrium_moisture_db": 0.02,
        "bulk_density_kg_m3": 600,
        "specific_heat_kj_kg_k": 1.0,
        "thermal_conductivity_w_mk": 0.2,
        "particle_diameter_mm": 0.5,
        "max_temp_c": 50,
    },
    MaterialType.SLUDGE: {
        "critical_moisture_db": 2.0,
        "equilibrium_moisture_db": 0.10,
        "bulk_density_kg_m3": 1000,
        "specific_heat_kj_kg_k": 3.5,
        "thermal_conductivity_w_mk": 0.5,
        "particle_diameter_mm": 5,
        "max_temp_c": 90,
    },
    MaterialType.BIOMASS: {
        "critical_moisture_db": 0.40,
        "equilibrium_moisture_db": 0.10,
        "bulk_density_kg_m3": 250,
        "specific_heat_kj_kg_k": 1.8,
        "thermal_conductivity_w_mk": 0.1,
        "particle_diameter_mm": 15,
        "max_temp_c": 100,
    },
}

# Dryer efficiency baselines by type
DRYER_EFFICIENCY_BASELINE: Dict[DryerType, float] = {
    DryerType.CONVECTIVE: 0.45,
    DryerType.CONDUCTIVE: 0.65,
    DryerType.RADIANT: 0.55,
    DryerType.MICROWAVE: 0.70,
    DryerType.VACUUM: 0.75,
    DryerType.FLUIDIZED_BED: 0.50,
    DryerType.FREEZE: 0.25,
    DryerType.SPRAY: 0.40,
    DryerType.ROTARY: 0.50,
    DryerType.TRAY: 0.55,
}


# =============================================================================
# PSYCHROMETRIC CALCULATION FUNCTIONS
# =============================================================================

def calculate_saturation_pressure(temp_c: float) -> float:
    """
    Calculate saturation vapor pressure using Magnus-Tetens equation.

    ASHRAE Fundamentals validated formula.
    Valid range: -40C to 100C

    Args:
        temp_c: Temperature in Celsius

    Returns:
        Saturation vapor pressure in kPa
    """
    if temp_c >= 0:
        # For positive temperatures (water)
        a = 17.27
        b = 237.7
    else:
        # For negative temperatures (ice)
        a = 21.875
        b = 265.5

    exponent = (a * temp_c) / (b + temp_c)
    p_sat = 0.61078 * math.exp(exponent)
    return p_sat


def calculate_humidity_ratio(
    dry_bulb_c: float,
    rh_pct: float,
    p_atm_kpa: float = 101.325
) -> float:
    """
    Calculate humidity ratio (specific humidity).

    W = 0.622 * Pv / (P - Pv)

    Args:
        dry_bulb_c: Dry bulb temperature (C)
        rh_pct: Relative humidity (%)
        p_atm_kpa: Atmospheric pressure (kPa)

    Returns:
        Humidity ratio (kg water / kg dry air)
    """
    p_sat = calculate_saturation_pressure(dry_bulb_c)
    p_v = (rh_pct / 100) * p_sat

    # Prevent division by zero
    if p_atm_kpa <= p_v:
        return 1.0  # Maximum physically possible

    w = 0.622 * p_v / (p_atm_kpa - p_v)
    return w


def calculate_wet_bulb_temp(
    dry_bulb_c: float,
    rh_pct: float,
    p_atm_kpa: float = 101.325
) -> float:
    """
    Calculate wet bulb temperature using iterative psychrometric relation.

    Uses the psychrometric equation and iterates to find Twb.

    Args:
        dry_bulb_c: Dry bulb temperature
        rh_pct: Relative humidity
        p_atm_kpa: Atmospheric pressure

    Returns:
        Wet bulb temperature (C)
    """
    w_actual = calculate_humidity_ratio(dry_bulb_c, rh_pct, p_atm_kpa)

    # Iterative solution using bisection
    t_wb_low = -40
    t_wb_high = dry_bulb_c

    for _ in range(50):  # Max iterations
        t_wb = (t_wb_low + t_wb_high) / 2

        # Saturated humidity ratio at wet bulb
        w_sat = calculate_humidity_ratio(t_wb, 100, p_atm_kpa)

        # Psychrometric equation
        # W = ((2501 - 2.326*Twb)*Wsat - 1.006*(Tdb - Twb)) / (2501 + 1.86*Tdb - 4.186*Twb)
        numerator = (2501 - 2.326 * t_wb) * w_sat - 1.006 * (dry_bulb_c - t_wb)
        denominator = 2501 + 1.86 * dry_bulb_c - 4.186 * t_wb

        w_calc = numerator / denominator

        if abs(w_calc - w_actual) < 1e-6:
            return t_wb
        elif w_calc > w_actual:
            t_wb_high = t_wb
        else:
            t_wb_low = t_wb

    return t_wb


def calculate_dew_point(
    dry_bulb_c: float,
    rh_pct: float,
    p_atm_kpa: float = 101.325
) -> float:
    """
    Calculate dew point temperature.

    Magnus formula inversion.

    Args:
        dry_bulb_c: Dry bulb temperature
        rh_pct: Relative humidity
        p_atm_kpa: Atmospheric pressure

    Returns:
        Dew point temperature (C)
    """
    if rh_pct <= 0:
        return -40  # Minimum practical value

    p_sat = calculate_saturation_pressure(dry_bulb_c)
    p_v = (rh_pct / 100) * p_sat

    # Invert Magnus equation
    # Pv = 0.61078 * exp(17.27*Tdp / (237.7 + Tdp))
    # ln(Pv/0.61078) = 17.27*Tdp / (237.7 + Tdp)

    if p_v <= 0:
        return -40

    alpha = math.log(p_v / 0.61078)
    t_dp = 237.7 * alpha / (17.27 - alpha)

    return t_dp


def calculate_enthalpy(
    dry_bulb_c: float,
    humidity_ratio: float
) -> float:
    """
    Calculate specific enthalpy of moist air.

    h = Cpa*T + W*(hfg + Cpv*T)
    h = 1.006*T + W*(2501 + 1.86*T) [kJ/kg dry air]

    Args:
        dry_bulb_c: Dry bulb temperature
        humidity_ratio: kg water / kg dry air

    Returns:
        Specific enthalpy (kJ/kg dry air)
    """
    cpa = 1.006  # kJ/kg.K - dry air specific heat
    cpv = 1.86   # kJ/kg.K - water vapor specific heat
    hfg = 2501   # kJ/kg - latent heat at 0C

    h = cpa * dry_bulb_c + humidity_ratio * (hfg + cpv * dry_bulb_c)
    return h


def calculate_specific_volume(
    dry_bulb_c: float,
    humidity_ratio: float,
    p_atm_kpa: float = 101.325
) -> float:
    """
    Calculate specific volume of moist air.

    v = Ra*T*(1 + 1.608*W) / P

    Args:
        dry_bulb_c: Dry bulb temperature
        humidity_ratio: kg water / kg dry air
        p_atm_kpa: Atmospheric pressure

    Returns:
        Specific volume (m3/kg dry air)
    """
    Ra = 0.287  # kJ/kg.K - gas constant for air
    T_k = dry_bulb_c + 273.15

    v = Ra * T_k * (1 + 1.608 * humidity_ratio) / p_atm_kpa
    return v


def calculate_psychrometric_state(
    dry_bulb_c: float,
    rh_pct: float,
    p_atm_kpa: float = 101.325
) -> PsychrometricState:
    """
    Calculate complete psychrometric state of moist air.

    Args:
        dry_bulb_c: Dry bulb temperature
        rh_pct: Relative humidity
        p_atm_kpa: Atmospheric pressure

    Returns:
        Complete psychrometric state
    """
    p_sat = calculate_saturation_pressure(dry_bulb_c)
    p_v = (rh_pct / 100) * p_sat
    w = calculate_humidity_ratio(dry_bulb_c, rh_pct, p_atm_kpa)
    t_wb = calculate_wet_bulb_temp(dry_bulb_c, rh_pct, p_atm_kpa)
    t_dp = calculate_dew_point(dry_bulb_c, rh_pct, p_atm_kpa)
    h = calculate_enthalpy(dry_bulb_c, w)
    v = calculate_specific_volume(dry_bulb_c, w, p_atm_kpa)

    return PsychrometricState(
        dry_bulb_temp_c=round(dry_bulb_c, 2),
        wet_bulb_temp_c=round(t_wb, 2),
        dew_point_temp_c=round(t_dp, 2),
        relative_humidity_pct=round(rh_pct, 1),
        humidity_ratio_kg_kg=round(w, 6),
        specific_enthalpy_kj_kg=round(h, 2),
        specific_volume_m3_kg=round(v, 4),
        vapor_pressure_kpa=round(p_v, 4),
        saturation_pressure_kpa=round(p_sat, 4)
    )


# =============================================================================
# DRYING RATE CALCULATIONS
# =============================================================================

def convert_moisture_wb_to_db(moisture_wb: float) -> float:
    """
    Convert moisture from wet basis to dry basis.

    X_db = X_wb / (1 - X_wb)

    Args:
        moisture_wb: Moisture content wet basis (fraction or %)

    Returns:
        Moisture content dry basis (same units)
    """
    if moisture_wb >= 100:
        return float('inf')
    if moisture_wb < 0:
        return 0

    x_wb = moisture_wb / 100 if moisture_wb > 1 else moisture_wb
    x_db = x_wb / (1 - x_wb)

    return x_db * 100 if moisture_wb > 1 else x_db


def convert_moisture_db_to_wb(moisture_db: float) -> float:
    """
    Convert moisture from dry basis to wet basis.

    X_wb = X_db / (1 + X_db)

    Args:
        moisture_db: Moisture content dry basis (fraction)

    Returns:
        Moisture content wet basis (fraction)
    """
    return moisture_db / (1 + moisture_db)


def calculate_mass_transfer_coefficient(
    air_velocity: float,
    particle_diameter: float,
    air_temp_c: float,
    humidity_ratio: float
) -> float:
    """
    Calculate convective mass transfer coefficient.

    Uses Ranz-Marshall correlation for spherical particles:
    Sh = 2 + 0.6 * Re^0.5 * Sc^0.33

    Args:
        air_velocity: Air velocity (m/s)
        particle_diameter: Characteristic dimension (m)
        air_temp_c: Air temperature (C)
        humidity_ratio: kg water / kg dry air

    Returns:
        Mass transfer coefficient (m/s)
    """
    # Air properties at temperature
    T_k = air_temp_c + 273.15

    # Air density (ideal gas)
    p_atm = 101325  # Pa
    R_air = 287  # J/kg.K
    rho_air = p_atm / (R_air * T_k) * (1 + humidity_ratio) / (1 + 1.608 * humidity_ratio)

    # Dynamic viscosity (Sutherland)
    mu_ref = 1.716e-5  # Pa.s at 273.15 K
    T_ref = 273.15
    S = 110.4  # Sutherland constant
    mu = mu_ref * (T_k / T_ref) ** 1.5 * (T_ref + S) / (T_k + S)

    # Kinematic viscosity
    nu = mu / rho_air

    # Diffusivity of water vapor in air
    D_ab = 2.26e-5 * (T_k / 273.15) ** 1.81  # m2/s

    # Dimensionless numbers
    Re = air_velocity * particle_diameter / nu
    Sc = nu / D_ab

    # Sherwood number (Ranz-Marshall)
    Sh = 2 + 0.6 * Re ** 0.5 * Sc ** 0.33

    # Mass transfer coefficient
    k_m = Sh * D_ab / particle_diameter

    return k_m


def calculate_heat_transfer_coefficient(
    air_velocity: float,
    particle_diameter: float,
    air_temp_c: float
) -> float:
    """
    Calculate convective heat transfer coefficient.

    Uses Ranz-Marshall correlation:
    Nu = 2 + 0.6 * Re^0.5 * Pr^0.33

    Args:
        air_velocity: Air velocity (m/s)
        particle_diameter: Characteristic dimension (m)
        air_temp_c: Air temperature (C)

    Returns:
        Heat transfer coefficient (W/m2.K)
    """
    T_k = air_temp_c + 273.15

    # Air properties
    p_atm = 101325
    R_air = 287
    rho_air = p_atm / (R_air * T_k)

    # Dynamic viscosity
    mu_ref = 1.716e-5
    T_ref = 273.15
    S = 110.4
    mu = mu_ref * (T_k / T_ref) ** 1.5 * (T_ref + S) / (T_k + S)

    nu = mu / rho_air

    # Thermal conductivity of air
    k_air = 0.0241 * (T_k / 273.15) ** 0.81  # W/m.K

    # Specific heat of air
    cp_air = 1006  # J/kg.K

    # Prandtl number
    Pr = cp_air * mu / k_air

    # Reynolds number
    Re = air_velocity * particle_diameter / nu

    # Nusselt number
    Nu = 2 + 0.6 * Re ** 0.5 * Pr ** 0.33

    # Heat transfer coefficient
    h = Nu * k_air / particle_diameter

    return h


def calculate_constant_drying_rate(
    h: float,
    air_temp_c: float,
    wet_bulb_c: float,
    latent_heat: float = 2260
) -> float:
    """
    Calculate constant rate period drying rate.

    Nc = h * (T_air - T_wb) / lambda

    During constant rate period, surface is saturated and
    drying rate is controlled by external heat/mass transfer.

    Args:
        h: Heat transfer coefficient (W/m2.K)
        air_temp_c: Drying air temperature
        wet_bulb_c: Wet bulb temperature
        latent_heat: Latent heat of vaporization (kJ/kg)

    Returns:
        Constant drying rate (kg/m2.s)
    """
    delta_t = air_temp_c - wet_bulb_c
    Nc = h * delta_t / (latent_heat * 1000)  # Convert kJ to J
    return max(0, Nc)


def calculate_falling_rate_drying(
    Nc: float,
    current_moisture_db: float,
    critical_moisture_db: float,
    equilibrium_moisture_db: float
) -> Tuple[float, DryingPhase]:
    """
    Calculate falling rate period drying rate.

    First falling rate (linear):
    N = Nc * (X - Xe) / (Xc - Xe)

    Second falling rate (diffusion controlled):
    N = Nc * ((X - Xe) / (Xc - Xe))^2

    Args:
        Nc: Constant drying rate
        current_moisture_db: Current moisture (dry basis)
        critical_moisture_db: Critical moisture content
        equilibrium_moisture_db: Equilibrium moisture content

    Returns:
        Tuple of (drying rate, drying phase)
    """
    X = current_moisture_db
    Xc = critical_moisture_db
    Xe = equilibrium_moisture_db

    if X >= Xc:
        return Nc, DryingPhase.CONSTANT_RATE

    # Normalized moisture
    if (Xc - Xe) <= 0:
        return 0, DryingPhase.SECOND_FALLING

    phi = (X - Xe) / (Xc - Xe)
    phi = max(0, min(1, phi))

    # Transition point between falling rate periods
    phi_transition = 0.5

    if phi > phi_transition:
        # First falling rate period
        N = Nc * phi
        phase = DryingPhase.FIRST_FALLING
    else:
        # Second falling rate period
        N = Nc * phi ** 2
        phase = DryingPhase.SECOND_FALLING

    return N, phase


def estimate_drying_time(
    initial_moisture_db: float,
    final_moisture_db: float,
    critical_moisture_db: float,
    equilibrium_moisture_db: float,
    Nc: float,
    dry_solids_kg: float,
    area_m2: float
) -> float:
    """
    Estimate total drying time using analytical integration.

    Constant rate: t1 = (ms/A) * (Xi - Xc) / Nc
    First falling: t2 = (ms/A) * (Xc - Xe) / Nc * ln((Xc - Xe)/(Xf - Xe))

    Args:
        initial_moisture_db: Initial moisture (dry basis)
        final_moisture_db: Final moisture (dry basis)
        critical_moisture_db: Critical moisture content
        equilibrium_moisture_db: Equilibrium moisture
        Nc: Constant drying rate (kg/m2.s)
        dry_solids_kg: Mass of dry solids
        area_m2: Exposed surface area

    Returns:
        Estimated drying time (hours)
    """
    if Nc <= 0 or area_m2 <= 0:
        return float('inf')

    Xi = initial_moisture_db
    Xf = final_moisture_db
    Xc = critical_moisture_db
    Xe = equilibrium_moisture_db

    total_time_s = 0

    # Constant rate period
    if Xi > Xc:
        t1 = (dry_solids_kg / area_m2) * (Xi - Xc) / Nc
        total_time_s += t1
        Xi = Xc  # Continue from critical moisture

    # Falling rate period
    if Xi > Xf and Xi <= Xc:
        if (Xc - Xe) > 0 and (Xf - Xe) > 0:
            t2 = (dry_solids_kg / area_m2) * (Xc - Xe) / Nc * math.log((Xi - Xe) / (Xf - Xe))
            total_time_s += t2

    return total_time_s / 3600  # Convert to hours


# =============================================================================
# DRYING AGENT CLASS
# =============================================================================

class DryingAgent:
    """
    GL-055 Drying Process Optimizer Agent.

    Optimizes industrial drying processes using psychrometric analysis,
    drying rate curve modeling, and energy balance calculations.

    Standards Applied:
    - ISO 13061: Wood moisture measurement
    - ASABE S448.1: Grain drying
    - ASHRAE Fundamentals: Psychrometrics
    - ASTM E96: Water vapor transmission

    Zero-Hallucination Implementation:
    - All calculations use physics-based deterministic equations
    - No ML/LLM in calculation path
    - Psychrometric calculations per ASHRAE formulas
    - Mass and energy balances for audit trail
    """

    AGENT_ID = "GL-055"
    AGENT_NAME = "DRYING"
    VERSION = "1.0.0"

    # Physical constants
    LATENT_HEAT_KJ_KG = 2260  # Latent heat of vaporization at 100C
    CP_AIR = 1.006  # kJ/kg.K - dry air specific heat
    CP_VAPOR = 1.86  # kJ/kg.K - water vapor specific heat
    CP_WATER = 4.18  # kJ/kg.K - liquid water specific heat

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize DryingAgent with optional configuration."""
        self.config = config or {}
        self.material_db = MATERIAL_DEFAULTS.copy()
        self.efficiency_baseline = DRYER_EFFICIENCY_BASELINE.copy()
        logger.info(f"{self.AGENT_NAME} Agent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous entry point for drying optimization.

        Args:
            input_data: Dictionary of input parameters

        Returns:
            Dictionary with optimization results
        """
        validated = DryingInput(**input_data)
        result = self._process(validated)
        return result.model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Async entry point (delegates to sync)."""
        return self.run(input_data)

    def _process(self, inp: DryingInput) -> DryingOutput:
        """
        Main processing method for drying optimization.

        Implements:
        1. Mass balance calculations
        2. Psychrometric analysis
        3. Drying rate curve determination
        4. Energy balance
        5. Optimization recommendations

        Args:
            inp: Validated input parameters

        Returns:
            Comprehensive drying optimization output
        """
        start_time = datetime.utcnow()
        warnings: List[str] = []
        recommendations: List[OptimizationRecommendation] = []

        # Get material properties
        mat_props = self._get_material_properties(inp)

        # Get dryer geometry
        geometry = self._get_dryer_geometry(inp)

        # =================================================================
        # MASS BALANCE
        # =================================================================

        # Convert moisture to dry basis
        initial_moisture_db = convert_moisture_wb_to_db(inp.initial_moisture_wb) / 100
        final_moisture_db = convert_moisture_wb_to_db(inp.target_moisture_wb) / 100

        # Calculate dry solids rate
        # wet_feed = dry_solids + water
        # wet_feed = dry_solids * (1 + X_initial)
        dry_solids_rate = inp.throughput_kg_hr / (1 + initial_moisture_db)

        # Water removed
        water_in = dry_solids_rate * initial_moisture_db
        water_out = dry_solids_rate * final_moisture_db
        water_removed_hr = water_in - water_out
        water_removed_s = water_removed_hr / 3600

        # Product rate
        product_rate = dry_solids_rate * (1 + final_moisture_db)

        # =================================================================
        # PSYCHROMETRIC ANALYSIS
        # =================================================================

        # Inlet air state
        inlet_air = calculate_psychrometric_state(
            inp.inlet_air_temp_c,
            inp.inlet_air_humidity_rh,
            inp.atmospheric_pressure_kpa
        )

        # Calculate required air flow for moisture pickup
        # Energy balance: ma * (h2 - h1) = mw * lambda + sensible_loads

        # First estimate: air can absorb moisture to ~80% RH at exit
        # Iterative solution for outlet conditions
        outlet_air, air_mass_flow = self._calculate_outlet_air_state(
            inp, inlet_air, water_removed_s, dry_solids_rate, mat_props
        )

        # Air volume flow
        air_volume_flow = air_mass_flow * inlet_air.specific_volume_m3_kg

        # Apply recirculation
        if inp.recirculation_ratio > 0:
            # Mixed inlet conditions
            w_mixed = (1 - inp.recirculation_ratio) * inlet_air.humidity_ratio_kg_kg + \
                      inp.recirculation_ratio * outlet_air.humidity_ratio_kg_kg
            h_mixed = (1 - inp.recirculation_ratio) * inlet_air.specific_enthalpy_kj_kg + \
                      inp.recirculation_ratio * outlet_air.specific_enthalpy_kj_kg

            # Recirculation increases humidity but reduces fresh air heating
            warnings.append(f"Recirculation ratio {inp.recirculation_ratio*100:.0f}% - "
                          f"monitor exit humidity to prevent re-wetting")

        # =================================================================
        # DRYING RATE ANALYSIS
        # =================================================================

        # Exposed area
        exposed_area = geometry.exposed_area_m2 or (geometry.length_m * geometry.width_m)

        # Heat and mass transfer coefficients
        particle_d = mat_props.particle_diameter_mm / 1000
        h_transfer = calculate_heat_transfer_coefficient(
            inp.air_velocity_m_s, particle_d, inp.inlet_air_temp_c
        )
        k_mass = calculate_mass_transfer_coefficient(
            inp.air_velocity_m_s, particle_d, inp.inlet_air_temp_c,
            inlet_air.humidity_ratio_kg_kg
        )

        # Lewis number check
        # Le = h / (k * rho * cp) should be ~1 for psychrometric equilibrium
        rho_air = 1 / inlet_air.specific_volume_m3_kg
        Le = h_transfer / (k_mass * rho_air * self.CP_AIR * 1000)

        # Constant drying rate
        Nc = calculate_constant_drying_rate(
            h_transfer, inp.inlet_air_temp_c,
            inlet_air.wet_bulb_temp_c, self.LATENT_HEAT_KJ_KG
        )
        Nc_kg_m2_hr = Nc * 3600

        # Average moisture during drying
        avg_moisture_db = (initial_moisture_db + final_moisture_db) / 2

        # Current drying rate (based on average conditions)
        actual_rate, drying_phase = calculate_falling_rate_drying(
            Nc, avg_moisture_db,
            mat_props.critical_moisture_db,
            mat_props.equilibrium_moisture_db
        )
        actual_rate_kg_m2_hr = actual_rate * 3600

        # Normalized moisture
        if (mat_props.critical_moisture_db - mat_props.equilibrium_moisture_db) > 0:
            normalized = (avg_moisture_db - mat_props.equilibrium_moisture_db) / \
                        (mat_props.critical_moisture_db - mat_props.equilibrium_moisture_db)
        else:
            normalized = 0
        normalized = max(0, min(1, normalized))

        # Estimate drying time
        dry_solids_in_dryer = dry_solids_rate * (inp.residence_time_min / 60)
        est_drying_time = estimate_drying_time(
            initial_moisture_db, final_moisture_db,
            mat_props.critical_moisture_db, mat_props.equilibrium_moisture_db,
            Nc, dry_solids_in_dryer, exposed_area
        )

        drying_rate_analysis = DryingRateAnalysis(
            current_phase=drying_phase,
            constant_rate_kg_m2_hr=round(Nc_kg_m2_hr, 4),
            actual_rate_kg_m2_hr=round(actual_rate_kg_m2_hr, 4),
            critical_moisture_db=mat_props.critical_moisture_db,
            equilibrium_moisture_db=mat_props.equilibrium_moisture_db,
            normalized_moisture=round(normalized, 3),
            mass_transfer_coeff_m_s=round(k_mass, 6),
            heat_transfer_coeff_w_m2k=round(h_transfer, 2),
            lewis_number=round(Le, 3),
            estimated_drying_time_hr=round(est_drying_time, 2)
        )

        # =================================================================
        # ENERGY BALANCE
        # =================================================================

        energy_balance = self._calculate_energy_balance(
            inp, inlet_air, outlet_air, air_mass_flow,
            water_removed_hr, dry_solids_rate, mat_props
        )

        # Annual metrics
        annual_energy = energy_balance.total_power_kw * inp.operating_hours_year
        annual_cost = annual_energy * inp.energy_cost_kwh

        # =================================================================
        # OPTIMIZATION
        # =================================================================

        optimal_temp, optimal_velocity, optimal_recirc, savings_pct = \
            self._calculate_optimal_conditions(inp, mat_props, energy_balance)

        # Quality risk assessment
        quality_risk, case_hardening, over_drying = self._assess_quality_risks(
            inp, mat_props, drying_rate_analysis
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            inp, mat_props, energy_balance, drying_rate_analysis,
            optimal_temp, optimal_velocity, optimal_recirc,
            quality_risk, case_hardening, over_drying
        )

        # Warnings
        if inp.inlet_air_temp_c > mat_props.max_temp_c:
            warnings.append(f"Air temperature {inp.inlet_air_temp_c}C exceeds material limit "
                          f"{mat_props.max_temp_c}C - quality degradation risk")

        if abs(Le - 1) > 0.3:
            warnings.append(f"Lewis number {Le:.2f} deviates from unity - "
                          "psychrometric equilibrium assumption may be invalid")

        if final_moisture_db < mat_props.equilibrium_moisture_db:
            warnings.append(f"Target moisture {final_moisture_db:.3f} below equilibrium "
                          f"{mat_props.equilibrium_moisture_db:.3f} - may be unachievable")

        if energy_balance.thermal_efficiency_pct < 30:
            warnings.append(f"Very low thermal efficiency {energy_balance.thermal_efficiency_pct:.1f}% - "
                          "significant opportunity for improvement")

        # =================================================================
        # PROVENANCE
        # =================================================================

        input_hash = hashlib.sha256(
            json.dumps(inp.model_dump(), default=str, sort_keys=True).encode()
        ).hexdigest()[:16]

        calc_hash = hashlib.sha256(json.dumps({
            "equipment_id": inp.equipment_id,
            "water_removed_kg_hr": round(water_removed_hr, 4),
            "specific_energy_kwh_kg": round(energy_balance.specific_energy_kwh_kg_water, 4),
            "thermal_efficiency": round(energy_balance.thermal_efficiency_pct, 2),
            "inlet_air_w": round(inlet_air.humidity_ratio_kg_kg, 6),
            "outlet_air_w": round(outlet_air.humidity_ratio_kg_kg, 6),
            "air_mass_flow": round(air_mass_flow, 4),
            "drying_phase": drying_phase.value,
            "input_hash": input_hash
        }, sort_keys=True).encode()).hexdigest()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(f"GL-055 processing completed in {processing_time:.1f}ms")

        return DryingOutput(
            equipment_id=inp.equipment_id,
            agent_id=self.AGENT_ID,
            agent_version=self.VERSION,

            # Mass balance
            dry_solids_rate_kg_hr=round(dry_solids_rate, 2),
            water_removed_kg_hr=round(water_removed_hr, 2),
            product_rate_kg_hr=round(product_rate, 2),
            initial_moisture_db=round(initial_moisture_db, 4),
            final_moisture_db=round(final_moisture_db, 4),

            # Psychrometrics
            inlet_air_state=inlet_air,
            outlet_air_state=outlet_air,
            air_mass_flow_kg_s=round(air_mass_flow, 4),
            air_volume_flow_m3_s=round(air_volume_flow, 4),

            # Drying rate
            drying_rate_analysis=drying_rate_analysis,
            exposed_area_m2=round(exposed_area, 2),

            # Energy
            energy_balance=energy_balance,
            annual_energy_kwh=round(annual_energy, 0),
            annual_energy_cost_usd=round(annual_cost, 2),

            # Optimization
            optimal_air_temp_c=round(optimal_temp, 1),
            optimal_air_velocity_m_s=round(optimal_velocity, 2),
            optimal_recirculation_ratio=round(optimal_recirc, 2),
            potential_energy_savings_pct=round(savings_pct, 1),

            # Quality
            product_quality_risk=quality_risk,
            case_hardening_risk=case_hardening,
            over_drying_risk=over_drying,

            # Recommendations
            recommendations=recommendations,
            warnings=warnings,

            # Provenance
            calculation_hash=calc_hash,
            input_hash=input_hash,
            calculation_method="psychrometric_balance",
            standards_applied=["ASHRAE Fundamentals", "ISO 13061", "ASABE S448.1"]
        )

    def _get_material_properties(self, inp: DryingInput) -> MaterialProperties:
        """Get material properties from input or defaults."""
        if inp.material_properties:
            return inp.material_properties

        defaults = self.material_db.get(inp.material_type, {})
        return MaterialProperties(**defaults)

    def _get_dryer_geometry(self, inp: DryingInput) -> DryerGeometry:
        """Get dryer geometry from input or estimate."""
        if inp.geometry:
            return inp.geometry

        # Estimate geometry based on throughput
        # Rule of thumb: ~100 kg/hr per m2 of dryer area
        area_needed = inp.throughput_kg_hr / 100
        length = math.sqrt(area_needed * 5)  # L/W ratio ~5
        width = area_needed / length

        return DryerGeometry(
            length_m=max(2, length),
            width_m=max(1, width),
            height_m=1.5,
            bed_depth_m=0.1,
            exposed_area_m2=area_needed
        )

    def _calculate_outlet_air_state(
        self,
        inp: DryingInput,
        inlet_air: PsychrometricState,
        water_removed_s: float,
        dry_solids_rate: float,
        mat_props: MaterialProperties
    ) -> Tuple[PsychrometricState, float]:
        """
        Calculate outlet air conditions and required air flow.

        Uses mass balance: ma * (W2 - W1) = mw
        Energy balance for outlet temperature.

        Args:
            inp: Input parameters
            inlet_air: Inlet psychrometric state
            water_removed_s: Water removal rate (kg/s)
            dry_solids_rate: Dry solids throughput (kg/hr)
            mat_props: Material properties

        Returns:
            Tuple of (outlet_air_state, air_mass_flow_kg_s)
        """
        # Target outlet conditions
        # Practical limit: 70-80% RH at outlet to prevent condensation
        target_outlet_rh = 70

        # Estimate outlet temperature
        # Adiabatic saturation: outlet follows constant wet bulb
        # But with heat losses, outlet is typically 10-20C above wet bulb

        # First estimate: 80% of temperature drop for convective dryers
        efficiency_factor = self.efficiency_baseline.get(inp.dryer_type, 0.5)

        # Outlet temperature depends on air flow rate
        # Higher flow = higher outlet temp (less cooling)
        # Lower flow = more temperature drop (but may not achieve target RH)

        # Iterative solution
        air_mass_flow = inp.air_mass_flow_kg_s

        if air_mass_flow is None:
            # Estimate based on moisture pickup
            # Maximum humidity at outlet (80% RH, temperature drop of 30C)
            est_outlet_temp = max(inlet_air.wet_bulb_temp_c + 5,
                                 inp.inlet_air_temp_c - 30)
            w_sat_outlet = calculate_humidity_ratio(est_outlet_temp, 100,
                                                    inp.atmospheric_pressure_kpa)
            w_max_outlet = w_sat_outlet * 0.8  # 80% RH limit

            delta_w = w_max_outlet - inlet_air.humidity_ratio_kg_kg

            if delta_w > 0:
                air_mass_flow = water_removed_s / delta_w
            else:
                # Very humid inlet air - use high flow rate
                air_mass_flow = water_removed_s / 0.005

        # Minimum air flow for adequate heat transfer
        min_flow = inp.throughput_kg_hr / 3600 * 5  # 5:1 air to material ratio
        air_mass_flow = max(air_mass_flow, min_flow)

        # Calculate outlet conditions with this flow
        # Moisture balance
        w_outlet = inlet_air.humidity_ratio_kg_kg + water_removed_s / air_mass_flow

        # Energy balance for outlet temperature
        # Q_in = ma * cp_air * (T1 - T2) + ma * W1 * cp_v * (T1 - T2)
        # Q_out = mw * lambda + m_solids * cp_s * (T_material - T_ambient)

        # Latent heat load
        Q_latent = water_removed_s * self.LATENT_HEAT_KJ_KG  # kW

        # Sensible heat to material (heating from ambient)
        material_rate_s = dry_solids_rate / 3600
        Q_material = material_rate_s * mat_props.specific_heat_kj_kg_k * \
                    (inp.inlet_air_temp_c * 0.7 - inp.ambient_temp_c)  # Material reaches ~70% of air temp

        # Total heat extracted from air
        Q_total = Q_latent + Q_material

        # Temperature drop
        cp_moist = self.CP_AIR + inlet_air.humidity_ratio_kg_kg * self.CP_VAPOR
        delta_T = Q_total / (air_mass_flow * cp_moist)

        T_outlet = inp.inlet_air_temp_c - delta_T
        T_outlet = max(T_outlet, inlet_air.wet_bulb_temp_c + 2)  # Can't go below wet bulb

        # Calculate outlet RH
        p_sat_outlet = calculate_saturation_pressure(T_outlet)
        p_v_outlet = w_outlet * inp.atmospheric_pressure_kpa / (0.622 + w_outlet)
        rh_outlet = min(95, (p_v_outlet / p_sat_outlet) * 100)

        outlet_air = calculate_psychrometric_state(
            T_outlet, rh_outlet, inp.atmospheric_pressure_kpa
        )

        return outlet_air, air_mass_flow

    def _calculate_energy_balance(
        self,
        inp: DryingInput,
        inlet_air: PsychrometricState,
        outlet_air: PsychrometricState,
        air_mass_flow: float,
        water_removed_hr: float,
        dry_solids_rate: float,
        mat_props: MaterialProperties
    ) -> EnergyBalance:
        """
        Calculate complete energy balance for the dryer.

        Args:
            inp: Input parameters
            inlet_air: Inlet air psychrometric state
            outlet_air: Outlet air state
            air_mass_flow: Air mass flow rate (kg/s)
            water_removed_hr: Water removal rate (kg/hr)
            dry_solids_rate: Dry solids rate (kg/hr)
            mat_props: Material properties

        Returns:
            Complete energy balance
        """
        # Theoretical minimum energy = latent heat only
        latent_heat_kw = (water_removed_hr / 3600) * self.LATENT_HEAT_KJ_KG

        # Air heating load (from ambient to inlet)
        ambient_state = calculate_psychrometric_state(
            inp.ambient_temp_c, inp.ambient_rh_pct, inp.atmospheric_pressure_kpa
        )

        # Fresh air heating
        fresh_air_flow = air_mass_flow * (1 - inp.recirculation_ratio)
        delta_h_air = inlet_air.specific_enthalpy_kj_kg - ambient_state.specific_enthalpy_kj_kg
        air_heating_kw = fresh_air_flow * delta_h_air

        # Material sensible heating
        material_rate_s = dry_solids_rate / 3600
        T_material_exit = inlet_air.wet_bulb_temp_c + 5  # Material exits near wet bulb
        material_heating_kw = material_rate_s * mat_props.specific_heat_kj_kg_k * \
                             (T_material_exit - inp.ambient_temp_c)

        # Total heat requirement
        total_heat_input = air_heating_kw + latent_heat_kw + material_heating_kw

        # Wall/radiation losses (estimate 5-15% of total)
        wall_loss_factor = 0.10 if inp.dryer_type in [DryerType.ROTARY, DryerType.SPRAY] else 0.05
        wall_losses = total_heat_input * wall_loss_factor

        # Exhaust losses (sensible heat in exit air)
        exhaust_losses = air_mass_flow * self.CP_AIR * (outlet_air.dry_bulb_temp_c - inp.ambient_temp_c)

        # Heat recovery
        heat_recovery = inp.heat_recovery_efficiency * exhaust_losses

        # Net heat required
        net_heat = total_heat_input + wall_losses - heat_recovery

        # Fuel input (accounting for heater efficiency)
        fuel_input = net_heat / (inp.heater_efficiency_pct / 100)

        # Fan power (estimate if not provided)
        fan_power = inp.fan_power_kw
        if fan_power is None:
            # Estimate: 0.5-2 kW per 1000 m3/hr air flow
            air_volume_m3_hr = air_mass_flow * inlet_air.specific_volume_m3_kg * 3600
            fan_power = air_volume_m3_hr / 1000 * 1.0  # 1 kW per 1000 m3/hr

        # Total power
        total_power = fuel_input + fan_power

        # Efficiency metrics
        thermal_efficiency = (latent_heat_kw / total_heat_input * 100) if total_heat_input > 0 else 0

        # Specific energy
        water_rate_s = water_removed_hr / 3600
        if water_rate_s > 0:
            specific_energy_kj = total_power / water_rate_s
            specific_energy_kwh = total_power / water_removed_hr
        else:
            specific_energy_kj = 0
            specific_energy_kwh = 0

        return EnergyBalance(
            air_heating_kw=round(air_heating_kw, 2),
            latent_heat_kw=round(latent_heat_kw, 2),
            material_heating_kw=round(material_heating_kw, 2),
            total_heat_input_kw=round(total_heat_input, 2),
            heat_recovery_kw=round(heat_recovery, 2),
            wall_losses_kw=round(wall_losses, 2),
            exhaust_losses_kw=round(exhaust_losses, 2),
            net_heat_required_kw=round(net_heat, 2),
            thermal_efficiency_pct=round(thermal_efficiency, 1),
            specific_energy_kj_kg_water=round(specific_energy_kj, 1),
            specific_energy_kwh_kg_water=round(specific_energy_kwh, 3),
            fuel_input_kw=round(fuel_input, 2),
            fan_power_kw=round(fan_power, 2),
            total_power_kw=round(total_power, 2)
        )

    def _calculate_optimal_conditions(
        self,
        inp: DryingInput,
        mat_props: MaterialProperties,
        current_energy: EnergyBalance
    ) -> Tuple[float, float, float, float]:
        """
        Calculate optimal operating conditions.

        Returns:
            Tuple of (optimal_temp, optimal_velocity, optimal_recirc, savings_pct)
        """
        # Optimal temperature: maximize efficiency while respecting quality limits
        # Higher temp = faster drying but quality risk
        optimal_temp = min(
            mat_props.max_temp_c,
            inp.inlet_air_temp_c + 10  # Can often increase 10C
        )

        # For temperature-sensitive materials, be conservative
        if inp.material_type in [MaterialType.FOOD, MaterialType.PHARMACEUTICAL, MaterialType.GRAIN]:
            optimal_temp = min(optimal_temp, mat_props.max_temp_c - 5)

        # Optimal velocity: 2-3 m/s typically optimal
        # Lower = inadequate heat transfer
        # Higher = excessive fan power
        if inp.air_velocity_m_s < 1.5:
            optimal_velocity = 2.0
        elif inp.air_velocity_m_s > 4:
            optimal_velocity = 3.0
        else:
            optimal_velocity = inp.air_velocity_m_s

        # Optimal recirculation: depends on inlet humidity
        # Low humidity inlet: more recirculation possible
        # High humidity inlet: less recirculation
        if inp.inlet_air_humidity_rh < 30:
            optimal_recirc = min(0.6, inp.recirculation_ratio + 0.2)
        elif inp.inlet_air_humidity_rh < 50:
            optimal_recirc = min(0.4, inp.recirculation_ratio + 0.1)
        else:
            optimal_recirc = inp.recirculation_ratio

        # If no heat recovery, recommend it
        if inp.heat_recovery_efficiency == 0:
            potential_recovery = 0.5  # 50% of exhaust heat recoverable
        else:
            potential_recovery = inp.heat_recovery_efficiency

        # Estimate savings
        savings_components = []

        # Temperature optimization savings (faster drying, less time)
        if optimal_temp > inp.inlet_air_temp_c:
            savings_components.append(5)  # ~5% from higher temp

        # Velocity optimization
        if abs(optimal_velocity - inp.air_velocity_m_s) > 0.5:
            savings_components.append(3)  # ~3% from velocity optimization

        # Recirculation savings (reduce fresh air heating)
        recirc_increase = optimal_recirc - inp.recirculation_ratio
        if recirc_increase > 0.1:
            savings_components.append(recirc_increase * 30)  # ~3% per 0.1 increase

        # Heat recovery savings
        if inp.heat_recovery_efficiency < potential_recovery:
            recovery_improvement = potential_recovery - inp.heat_recovery_efficiency
            # Exhaust losses are typically 30-40% of input
            savings_components.append(recovery_improvement * 35)

        savings_pct = min(40, sum(savings_components))  # Cap at 40%

        return optimal_temp, optimal_velocity, optimal_recirc, savings_pct

    def _assess_quality_risks(
        self,
        inp: DryingInput,
        mat_props: MaterialProperties,
        drying_rate: DryingRateAnalysis
    ) -> Tuple[str, bool, bool]:
        """
        Assess product quality risks.

        Returns:
            Tuple of (risk_level, case_hardening_risk, over_drying_risk)
        """
        risk_score = 0

        # Temperature risk
        if inp.inlet_air_temp_c > mat_props.max_temp_c:
            risk_score += 3
        elif inp.inlet_air_temp_c > mat_props.max_temp_c * 0.9:
            risk_score += 1

        # Case hardening risk (surface dries too fast)
        # High for: high temp, low humidity, falling rate period
        case_hardening = False
        if inp.material_type in [MaterialType.WOOD, MaterialType.FOOD, MaterialType.CERAMIC]:
            if inp.inlet_air_temp_c > mat_props.max_temp_c * 0.8:
                if inp.inlet_air_humidity_rh < 30:
                    if drying_rate.current_phase in [DryingPhase.FIRST_FALLING, DryingPhase.SECOND_FALLING]:
                        case_hardening = True
                        risk_score += 2

        # Over-drying risk
        final_db = convert_moisture_wb_to_db(inp.target_moisture_wb) / 100
        over_drying = final_db < mat_props.equilibrium_moisture_db * 1.2
        if over_drying:
            risk_score += 1

        # Drying rate too high
        if drying_rate.actual_rate_kg_m2_hr > drying_rate.constant_rate_kg_m2_hr * 1.2:
            risk_score += 1

        if risk_score >= 4:
            quality_risk = "HIGH"
        elif risk_score >= 2:
            quality_risk = "MEDIUM"
        else:
            quality_risk = "LOW"

        return quality_risk, case_hardening, over_drying

    def _generate_recommendations(
        self,
        inp: DryingInput,
        mat_props: MaterialProperties,
        energy: EnergyBalance,
        drying_rate: DryingRateAnalysis,
        optimal_temp: float,
        optimal_velocity: float,
        optimal_recirc: float,
        quality_risk: str,
        case_hardening: bool,
        over_drying: bool
    ) -> List[OptimizationRecommendation]:
        """Generate actionable optimization recommendations."""
        recommendations = []

        # Temperature optimization
        if optimal_temp != inp.inlet_air_temp_c:
            direction = "Increase" if optimal_temp > inp.inlet_air_temp_c else "Decrease"
            recommendations.append(OptimizationRecommendation(
                category="Temperature",
                priority="MEDIUM",
                current_value=f"{inp.inlet_air_temp_c}C",
                recommended_value=f"{optimal_temp}C",
                estimated_savings_pct=5,
                description=f"{direction} drying air temperature to improve efficiency "
                           f"while staying within material quality limits"
            ))

        # Air velocity optimization
        if abs(optimal_velocity - inp.air_velocity_m_s) > 0.3:
            recommendations.append(OptimizationRecommendation(
                category="Air Velocity",
                priority="MEDIUM",
                current_value=f"{inp.air_velocity_m_s} m/s",
                recommended_value=f"{optimal_velocity} m/s",
                estimated_savings_pct=3,
                description="Optimize air velocity for better heat/mass transfer balance"
            ))

        # Recirculation
        if optimal_recirc > inp.recirculation_ratio + 0.1:
            recommendations.append(OptimizationRecommendation(
                category="Air Recirculation",
                priority="HIGH",
                current_value=f"{inp.recirculation_ratio*100:.0f}%",
                recommended_value=f"{optimal_recirc*100:.0f}%",
                estimated_savings_pct=(optimal_recirc - inp.recirculation_ratio) * 30,
                payback_months=6,
                description="Increase exhaust air recirculation to reduce fresh air heating load"
            ))

        # Heat recovery
        if inp.heat_recovery_efficiency < 0.3:
            recommendations.append(OptimizationRecommendation(
                category="Heat Recovery",
                priority="HIGH",
                current_value=f"{inp.heat_recovery_efficiency*100:.0f}%",
                recommended_value="50%",
                estimated_savings_pct=15,
                payback_months=18,
                description="Install air-to-air heat exchanger on exhaust to preheat inlet air"
            ))

        # Efficiency concerns
        if energy.thermal_efficiency_pct < 40:
            recommendations.append(OptimizationRecommendation(
                category="Thermal Efficiency",
                priority="HIGH",
                current_value=f"{energy.thermal_efficiency_pct:.1f}%",
                recommended_value="50%+",
                estimated_savings_pct=10,
                description="Overall thermal efficiency is low - review insulation, air leaks, and heat recovery"
            ))

        # Specific energy concerns
        if energy.specific_energy_kwh_kg_water > 1.5:
            recommendations.append(OptimizationRecommendation(
                category="Specific Energy",
                priority="MEDIUM",
                current_value=f"{energy.specific_energy_kwh_kg_water:.2f} kWh/kg",
                recommended_value="<1.0 kWh/kg",
                estimated_savings_pct=20,
                description="High specific energy consumption - consider dryer type change or operating optimization"
            ))

        # Dryer type upgrade
        if inp.dryer_type == DryerType.CONVECTIVE:
            baseline_eff = self.efficiency_baseline[DryerType.CONVECTIVE]
            if energy.thermal_efficiency_pct < baseline_eff * 100 * 0.8:
                recommendations.append(OptimizationRecommendation(
                    category="Equipment Upgrade",
                    priority="LOW",
                    current_value=inp.dryer_type.value,
                    recommended_value="VACUUM or MICROWAVE",
                    estimated_savings_pct=25,
                    payback_months=36,
                    description="Consider more efficient dryer technology for significant energy savings"
                ))

        # Case hardening mitigation
        if case_hardening:
            recommendations.append(OptimizationRecommendation(
                category="Quality Control",
                priority="HIGH",
                current_value="Risk present",
                recommended_value="Mitigated",
                estimated_savings_pct=0,
                description="Reduce case hardening risk by increasing inlet humidity or using "
                           "intermittent drying schedule"
            ))

        # Over-drying
        if over_drying:
            recommendations.append(OptimizationRecommendation(
                category="Moisture Control",
                priority="MEDIUM",
                current_value=f"{inp.target_moisture_wb}% wb",
                recommended_value=f"{mat_props.equilibrium_moisture_db*100:.1f}% db min",
                estimated_savings_pct=5,
                description="Target moisture is near/below equilibrium - consider relaxing target "
                           "to reduce energy consumption"
            ))

        # Batch vs continuous
        if inp.dryer_type == DryerType.TRAY and inp.throughput_kg_hr > 1000:
            recommendations.append(OptimizationRecommendation(
                category="Process Mode",
                priority="MEDIUM",
                current_value="Batch (Tray)",
                recommended_value="Continuous",
                estimated_savings_pct=15,
                payback_months=24,
                description="High throughput may benefit from continuous dryer for better efficiency"
            ))

        return recommendations


# =============================================================================
# PACK SPECIFICATION
# =============================================================================

PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-055",
    "name": "DRYING",
    "display_name": "Drying Process Optimizer",
    "version": "1.0.0",
    "category": "Process Optimization",
    "tier": "Standard",
    "summary": "Optimizes industrial drying processes using psychrometric analysis, "
               "drying rate curves, and energy balance calculations for maximum efficiency",
    "description": (
        "The Drying Process Optimizer Agent provides comprehensive analysis and optimization "
        "of industrial drying operations. It implements physics-based psychrometric calculations "
        "per ASHRAE Fundamentals, models constant and falling rate drying periods, and performs "
        "complete mass and energy balances. The agent supports multiple dryer types and material "
        "categories with material-specific property databases."
    ),
    "standards": [
        {"ref": "ISO 13061", "title": "Wood moisture content determination"},
        {"ref": "ASABE S448.1", "title": "Thin-layer drying equations"},
        {"ref": "ASHRAE Fundamentals", "title": "Psychrometric calculations"},
        {"ref": "ASTM E96", "title": "Water vapor transmission"},
    ],
    "capabilities": [
        "Psychrometric state calculations (dry/wet bulb, dew point, enthalpy)",
        "Drying rate curve analysis (constant and falling rate periods)",
        "Heat and mass transfer coefficient estimation",
        "Energy balance with heat recovery assessment",
        "Multi-dryer type support (10 configurations)",
        "Material property database (10 material types)",
        "Quality risk assessment (case hardening, over-drying)",
        "Optimization recommendations with payback analysis",
    ],
    "inputs": {
        "equipment_id": "Unique dryer identifier",
        "dryer_type": "CONVECTIVE, VACUUM, SPRAY, etc.",
        "material_type": "WOOD, GRAIN, FOOD, etc.",
        "throughput_kg_hr": "Wet material feed rate",
        "initial_moisture_wb": "Initial moisture (wet basis %)",
        "target_moisture_wb": "Target moisture (wet basis %)",
        "inlet_air_temp_c": "Drying air temperature",
        "inlet_air_humidity_rh": "Inlet air relative humidity",
    },
    "outputs": {
        "water_removed_kg_hr": "Evaporation rate",
        "thermal_efficiency_pct": "Dryer thermal efficiency",
        "specific_energy_kwh_kg_water": "Energy per kg water removed",
        "drying_rate_analysis": "Drying curve analysis with phase determination",
        "inlet_air_state": "Complete inlet psychrometric state",
        "outlet_air_state": "Complete outlet psychrometric state",
        "recommendations": "Optimization recommendations with savings estimates",
    },
    "provenance": {
        "calculation_method": "psychrometric_balance",
        "enable_audit": True,
        "sha256_hash": True,
        "zero_hallucination": True,
    },
    "performance": {
        "typical_latency_ms": 50,
        "max_throughput_rps": 100,
    },
    "dependencies": ["pydantic", "hashlib", "math"],
    "entry_point": "DryingAgent",
    "author": "GreenLang Team",
    "license": "Apache-2.0",
}
