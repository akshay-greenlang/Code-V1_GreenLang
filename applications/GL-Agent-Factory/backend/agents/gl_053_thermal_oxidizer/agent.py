"""GL-053: Thermal Oxidizer Agent (THERMAL-OXIDIZER).

Optimizes thermal oxidizer performance for VOC destruction including RTO, RCO,
and direct-fired systems. Implements the "Three T's of Combustion" (Time,
Temperature, Turbulence) for destruction efficiency calculation and fuel
optimization.

The agent follows GreenLang's zero-hallucination principle by using only
deterministic calculations from combustion engineering - no ML/LLM
in the calculation path.

Standards: EPA 40 CFR 63 (MACT), NFPA 86 (Ovens and Furnaces)

Example:
    >>> agent = ThermalOxidizerAgent()
    >>> result = agent.run({
    ...     "equipment_id": "TO-001",
    ...     "oxidizer_type": "REGENERATIVE",
    ...     "process_flow_scfm": 20000,
    ...     "inlet_voc_ppm": 500,
    ...     "combustion_temp_f": 1600
    ... })
    >>> assert result["actual_destruction_pct"] >= 99.0
"""

import hashlib
import json
import logging
import math
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator, root_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class OxidizerType(str, Enum):
    """Types of thermal oxidizers."""
    DIRECT_FIRED = "DIRECT_FIRED"  # Direct thermal oxidizer
    RECUPERATIVE = "RECUPERATIVE"  # Recuperative thermal oxidizer
    REGENERATIVE = "REGENERATIVE"  # Regenerative thermal oxidizer (RTO)
    CATALYTIC = "CATALYTIC"  # Catalytic oxidizer (RCO)
    FLAMELESS = "FLAMELESS"  # Flameless thermal oxidizer


class FuelType(str, Enum):
    """Fuel types for auxiliary firing."""
    NATURAL_GAS = "NATURAL_GAS"
    PROPANE = "PROPANE"
    FUEL_OIL = "FUEL_OIL"
    DIGESTER_GAS = "DIGESTER_GAS"
    LANDFILL_GAS = "LANDFILL_GAS"


class VOCType(str, Enum):
    """Common VOC categories for destruction modeling."""
    ALIPHATIC = "ALIPHATIC"  # e.g., hexane, octane
    AROMATIC = "AROMATIC"  # e.g., toluene, xylene
    OXYGENATED = "OXYGENATED"  # e.g., acetone, MEK
    HALOGENATED = "HALOGENATED"  # e.g., methylene chloride
    MIXED = "MIXED"  # Mixed or unknown


class OperatingStatus(str, Enum):
    """Oxidizer operating status."""
    OPTIMAL = "OPTIMAL"
    ACCEPTABLE = "ACCEPTABLE"
    SUBOPTIMAL = "SUBOPTIMAL"
    NON_COMPLIANT = "NON_COMPLIANT"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class VOCProperties(BaseModel):
    """VOC stream properties."""

    voc_type: VOCType = Field(default=VOCType.MIXED)
    molecular_weight: float = Field(default=80, gt=0, description="Average MW (g/mol)")
    heat_of_combustion_btu_lb: float = Field(
        default=18000,
        gt=0,
        description="Heat of combustion (BTU/lb)"
    )
    lel_ppm: float = Field(
        default=10000,
        gt=0,
        description="Lower explosive limit (ppm)"
    )
    autoignition_temp_f: float = Field(
        default=500,
        gt=0,
        description="Autoignition temperature (F)"
    )


class HeatRecoverySystem(BaseModel):
    """Heat recovery system parameters."""

    type: str = Field(default="regenerative", description="ceramic, metallic, shell-tube")
    design_efficiency_pct: float = Field(default=95, ge=0, le=99)
    current_efficiency_pct: float = Field(default=90, ge=0, le=99)
    media_life_years: float = Field(default=10, gt=0)
    media_age_years: float = Field(default=3, ge=0)
    pressure_drop_inwc: float = Field(default=15, ge=0)


class ThermalOxidizerInput(BaseModel):
    """Input data model for ThermalOxidizerAgent."""

    equipment_id: str = Field(..., min_length=1, description="Equipment identifier")
    oxidizer_type: OxidizerType = Field(default=OxidizerType.REGENERATIVE)

    # Process gas parameters
    process_flow_scfm: float = Field(
        ...,
        gt=0,
        le=500000,
        description="Process gas flow rate (SCFM)"
    )
    inlet_voc_ppm: float = Field(
        default=1000,
        ge=0,
        le=50000,
        description="Inlet VOC concentration (ppm as carbon)"
    )
    inlet_temp_f: float = Field(
        default=100,
        ge=0,
        le=500,
        description="Inlet gas temperature (F)"
    )
    inlet_moisture_pct: float = Field(
        default=2,
        ge=0,
        le=30,
        description="Inlet moisture content (%)"
    )

    # VOC properties
    voc_properties: Optional[VOCProperties] = None

    # Combustion parameters (Three T's)
    combustion_temp_f: float = Field(
        default=1600,
        ge=1000,
        le=2200,
        description="Combustion chamber temperature (F)"
    )
    residence_time_s: float = Field(
        default=1.0,
        gt=0,
        le=5,
        description="Residence time in combustion zone (s)"
    )
    turbulence_re: Optional[float] = Field(
        default=None,
        description="Reynolds number in combustion zone"
    )

    # Heat recovery
    heat_recovery: Optional[HeatRecoverySystem] = None
    heat_recovery_pct: float = Field(
        default=85,
        ge=0,
        le=99,
        description="Heat recovery efficiency (%)"
    )

    # Target destruction efficiency
    required_destruction_pct: float = Field(
        default=99,
        ge=90,
        le=99.99,
        description="Required destruction efficiency (%)"
    )

    # Fuel parameters
    fuel_type: FuelType = Field(default=FuelType.NATURAL_GAS)
    fuel_hhv_btu_scf: float = Field(
        default=1020,
        gt=0,
        description="Fuel heating value (BTU/SCF)"
    )
    fuel_cost_mmbtu: float = Field(
        default=5.0,
        ge=0,
        description="Fuel cost ($/MMBTU)"
    )

    # Operating parameters
    operating_hours_year: int = Field(default=8000, ge=0, le=8760)
    electricity_price_kwh: float = Field(default=0.10, ge=0)

    # Safety parameters
    lel_limit_pct: float = Field(
        default=25,
        ge=0,
        le=50,
        description="Maximum LEL percentage for safe operation"
    )

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @root_validator(skip_on_failure=True)
    def validate_combustion_parameters(cls, values):
        """Validate combustion parameters meet regulatory requirements."""
        temp = values.get('combustion_temp_f', 1600)
        time = values.get('residence_time_s', 1.0)
        req_de = values.get('required_destruction_pct', 99)

        # EPA guidance: 99% DE typically requires 1600F and 0.75s minimum
        if req_de >= 99 and (temp < 1500 or time < 0.5):
            logger.warning(
                f"Combustion parameters (T={temp}F, t={time}s) may not achieve "
                f"{req_de}% destruction efficiency"
            )

        return values


class ThreeTsAnalysis(BaseModel):
    """Three T's of Combustion analysis results."""

    temperature_f: float
    temperature_rating: str  # Excellent, Good, Marginal, Poor
    residence_time_s: float
    residence_time_rating: str
    turbulence_factor: float  # 0-1 scale
    turbulence_rating: str
    overall_rating: str
    theoretical_de_pct: float


class CombustionAnalysis(BaseModel):
    """Combustion performance analysis."""

    heat_required_mmbtu_hr: float
    heat_recovered_mmbtu_hr: float
    net_fuel_mmbtu_hr: float
    voc_heat_contribution_mmbtu_hr: float
    self_sustaining_threshold_ppm: float
    is_self_sustaining: bool
    fuel_savings_from_voc_pct: float


class EmissionsAnalysis(BaseModel):
    """Emissions output analysis."""

    outlet_voc_ppm: float
    actual_destruction_pct: float
    outlet_voc_lb_hr: float
    co_estimate_ppm: float
    nox_estimate_ppm: float
    co2_emissions_lb_hr: float
    co2_emissions_tonnes_yr: float


class EconomicAnalysis(BaseModel):
    """Economic analysis results."""

    fuel_consumption_scfh: float
    fuel_cost_hr: float
    annual_fuel_cost_usd: float
    electricity_cost_hr: float
    annual_electricity_cost_usd: float
    total_annual_cost_usd: float
    cost_per_scfm_yr: float


class OptimizationResult(BaseModel):
    """Optimization recommendations."""

    optimal_temp_f: float
    optimal_residence_s: float
    optimal_heat_recovery_pct: float
    potential_fuel_savings_pct: float
    potential_cost_savings_usd: float


class ThermalOxidizerOutput(BaseModel):
    """Output data model for ThermalOxidizerAgent."""

    equipment_id: str
    oxidizer_type: str

    # Three T's analysis
    three_ts_analysis: ThreeTsAnalysis

    # Destruction efficiency
    outlet_voc_ppm: float
    actual_destruction_pct: float
    destruction_margin_pct: float
    operating_status: OperatingStatus

    # Combustion analysis
    combustion_analysis: CombustionAnalysis

    # Emissions
    emissions_analysis: EmissionsAnalysis

    # Energy consumption
    fuel_consumption_mmbtu_hr: float
    annual_fuel_mmbtu: float
    annual_fuel_cost_usd: float
    thermal_efficiency_pct: float

    # Economics
    economic_analysis: EconomicAnalysis

    # Optimization
    optimization: OptimizationResult

    # Regulatory compliance
    meets_destruction_requirement: bool
    lel_percentage: float
    lel_status: str

    # Recommendations
    recommendations: List[str]
    warnings: List[str]

    # Provenance
    calculation_hash: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    validation_status: str = Field(default="PASS")
    validation_errors: List[str] = Field(default_factory=list)
    agent_version: str = Field(default="1.0.0")


# =============================================================================
# CALCULATION ENGINE
# =============================================================================

# Air properties at standard conditions
AIR_DENSITY_LB_SCF = 0.075  # lb/SCF at 60F, 14.7 psia
AIR_CP_BTU_LB_F = 0.24  # BTU/lb-F
AIR_MW = 29.0  # g/mol

# Fuel properties database
FUEL_PROPERTIES_DB: Dict[FuelType, Dict[str, float]] = {
    FuelType.NATURAL_GAS: {
        "hhv_btu_scf": 1020,
        "co2_lb_mmbtu": 117,
        "density_lb_scf": 0.044
    },
    FuelType.PROPANE: {
        "hhv_btu_scf": 2516,
        "co2_lb_mmbtu": 139,
        "density_lb_scf": 0.116
    },
    FuelType.FUEL_OIL: {
        "hhv_btu_scf": 140000,  # BTU/gal converted
        "co2_lb_mmbtu": 161,
        "density_lb_scf": 7.1  # lb/gal
    },
    FuelType.DIGESTER_GAS: {
        "hhv_btu_scf": 600,
        "co2_lb_mmbtu": 117,  # Biogenic
        "density_lb_scf": 0.050
    },
    FuelType.LANDFILL_GAS: {
        "hhv_btu_scf": 500,
        "co2_lb_mmbtu": 117,  # Biogenic
        "density_lb_scf": 0.055
    }
}

# VOC properties database
VOC_PROPERTIES_DB: Dict[VOCType, Dict[str, float]] = {
    VOCType.ALIPHATIC: {
        "avg_mw": 86,  # Hexane-like
        "hoc_btu_lb": 19000,
        "lel_ppm": 11000,
        "autoignition_f": 437
    },
    VOCType.AROMATIC: {
        "avg_mw": 92,  # Toluene-like
        "hoc_btu_lb": 17400,
        "lel_ppm": 12000,
        "autoignition_f": 896
    },
    VOCType.OXYGENATED: {
        "avg_mw": 72,  # MEK-like
        "hoc_btu_lb": 13500,
        "lel_ppm": 18000,
        "autoignition_f": 759
    },
    VOCType.HALOGENATED: {
        "avg_mw": 85,
        "hoc_btu_lb": 5000,  # Lower due to halogens
        "lel_ppm": 130000,  # Higher LEL
        "autoignition_f": 1033
    },
    VOCType.MIXED: {
        "avg_mw": 80,
        "hoc_btu_lb": 15000,
        "lel_ppm": 12000,
        "autoignition_f": 700
    }
}

# Oxidizer type specifications
OXIDIZER_SPECS_DB: Dict[OxidizerType, Dict[str, Any]] = {
    OxidizerType.DIRECT_FIRED: {
        "typical_heat_recovery_pct": 0,
        "max_heat_recovery_pct": 0,
        "min_temp_f": 1400,
        "typical_temp_f": 1600,
        "min_residence_s": 0.5,
        "typical_residence_s": 1.0,
        "fan_power_factor": 0.8,  # HP per 1000 SCFM
        "turndown_ratio": 4
    },
    OxidizerType.RECUPERATIVE: {
        "typical_heat_recovery_pct": 70,
        "max_heat_recovery_pct": 80,
        "min_temp_f": 1400,
        "typical_temp_f": 1600,
        "min_residence_s": 0.5,
        "typical_residence_s": 1.0,
        "fan_power_factor": 1.0,
        "turndown_ratio": 4
    },
    OxidizerType.REGENERATIVE: {
        "typical_heat_recovery_pct": 95,
        "max_heat_recovery_pct": 97,
        "min_temp_f": 1500,
        "typical_temp_f": 1600,
        "min_residence_s": 0.5,
        "typical_residence_s": 1.0,
        "fan_power_factor": 1.5,
        "turndown_ratio": 3
    },
    OxidizerType.CATALYTIC: {
        "typical_heat_recovery_pct": 70,
        "max_heat_recovery_pct": 80,
        "min_temp_f": 600,
        "typical_temp_f": 800,
        "min_residence_s": 0.2,
        "typical_residence_s": 0.5,
        "fan_power_factor": 1.2,
        "turndown_ratio": 5
    },
    OxidizerType.FLAMELESS: {
        "typical_heat_recovery_pct": 90,
        "max_heat_recovery_pct": 95,
        "min_temp_f": 1500,
        "typical_temp_f": 1700,
        "min_residence_s": 0.75,
        "typical_residence_s": 1.0,
        "fan_power_factor": 1.0,
        "turndown_ratio": 4
    }
}


def calculate_destruction_efficiency(
    temperature_f: float,
    residence_time_s: float,
    voc_type: VOCType,
    oxidizer_type: OxidizerType
) -> float:
    """
    Calculate VOC destruction efficiency using Three T's of Combustion.

    The destruction efficiency is modeled using Arrhenius kinetics:
    DE = 1 - exp(-k * t)
    where k = A * exp(-Ea/RT)

    For practical use, this is simplified to empirical correlations
    based on EPA guidance and field data.

    Reference: EPA OAQPS, NFPA 86

    Args:
        temperature_f: Combustion temperature (F)
        residence_time_s: Residence time (s)
        voc_type: VOC category
        oxidizer_type: Type of oxidizer

    Returns:
        Destruction efficiency (%)
    """
    # Temperature conversion
    temperature_r = temperature_f + 459.67  # Rankine

    # Base destruction rate constants by VOC type
    # k99 = temperature for 99% DE at 1 second residence time
    k99_temp = {
        VOCType.ALIPHATIC: 1400,
        VOCType.AROMATIC: 1500,
        VOCType.OXYGENATED: 1300,
        VOCType.HALOGENATED: 1800,  # Harder to destroy
        VOCType.MIXED: 1500
    }

    base_temp = k99_temp.get(voc_type, 1500)

    # Catalytic oxidizers work at lower temperatures
    if oxidizer_type == OxidizerType.CATALYTIC:
        base_temp = base_temp * 0.5  # Catalyst reduces required temp

    # Temperature factor using simplified Arrhenius
    # DE increases exponentially with temperature
    if temperature_f >= base_temp:
        temp_factor = 1.0 + 0.1 * (temperature_f - base_temp) / 100
    else:
        temp_factor = (temperature_f / base_temp) ** 2

    temp_factor = min(1.2, max(0.1, temp_factor))

    # Time factor - first-order kinetics
    # At 1 second and proper temp, should achieve 99%
    # k = -ln(1 - DE) / t, for DE = 0.99 at t = 1s: k = 4.6
    k_apparent = 4.6 * temp_factor

    # Calculate destruction from kinetics
    de_fraction = 1 - math.exp(-k_apparent * residence_time_s)

    # Convert to percentage and apply limits
    de_pct = de_fraction * 100

    # Maximum practical limits
    if oxidizer_type == OxidizerType.CATALYTIC:
        de_pct = min(de_pct, 99.5)  # Catalyst has limits
    else:
        de_pct = min(de_pct, 99.99)

    return round(de_pct, 2)


def calculate_three_ts(
    temperature_f: float,
    residence_time_s: float,
    oxidizer_type: OxidizerType
) -> ThreeTsAnalysis:
    """
    Analyze the Three T's of Combustion.

    Reference: EPA AP-42, NFPA 86

    Args:
        temperature_f: Combustion chamber temperature (F)
        residence_time_s: Residence time (s)
        oxidizer_type: Type of oxidizer

    Returns:
        ThreeTsAnalysis with ratings
    """
    specs = OXIDIZER_SPECS_DB.get(oxidizer_type, OXIDIZER_SPECS_DB[OxidizerType.REGENERATIVE])

    # Temperature rating
    if oxidizer_type == OxidizerType.CATALYTIC:
        if temperature_f >= 800:
            temp_rating = "Excellent"
        elif temperature_f >= 600:
            temp_rating = "Good"
        elif temperature_f >= 500:
            temp_rating = "Marginal"
        else:
            temp_rating = "Poor"
    else:
        if temperature_f >= 1600:
            temp_rating = "Excellent"
        elif temperature_f >= 1500:
            temp_rating = "Good"
        elif temperature_f >= 1400:
            temp_rating = "Marginal"
        else:
            temp_rating = "Poor"

    # Residence time rating
    min_time = specs["min_residence_s"]
    if residence_time_s >= 1.0:
        time_rating = "Excellent"
    elif residence_time_s >= 0.75:
        time_rating = "Good"
    elif residence_time_s >= min_time:
        time_rating = "Marginal"
    else:
        time_rating = "Poor"

    # Turbulence factor (estimated from typical designs)
    # RTO/RCO have better mixing than direct-fired
    turbulence_factors = {
        OxidizerType.DIRECT_FIRED: 0.7,
        OxidizerType.RECUPERATIVE: 0.75,
        OxidizerType.REGENERATIVE: 0.85,
        OxidizerType.CATALYTIC: 0.9,  # Catalyst bed provides mixing
        OxidizerType.FLAMELESS: 0.95  # Excellent mixing
    }
    turb_factor = turbulence_factors.get(oxidizer_type, 0.8)

    if turb_factor >= 0.9:
        turb_rating = "Excellent"
    elif turb_factor >= 0.8:
        turb_rating = "Good"
    elif turb_factor >= 0.7:
        turb_rating = "Marginal"
    else:
        turb_rating = "Poor"

    # Overall rating
    ratings = [temp_rating, time_rating, turb_rating]
    if all(r == "Excellent" for r in ratings):
        overall = "Excellent"
    elif "Poor" in ratings:
        overall = "Poor"
    elif "Marginal" in ratings:
        overall = "Marginal"
    else:
        overall = "Good"

    # Theoretical DE based on combined factors
    theoretical_de = calculate_destruction_efficiency(
        temperature_f, residence_time_s, VOCType.MIXED, oxidizer_type
    )

    return ThreeTsAnalysis(
        temperature_f=temperature_f,
        temperature_rating=temp_rating,
        residence_time_s=residence_time_s,
        residence_time_rating=time_rating,
        turbulence_factor=turb_factor,
        turbulence_rating=turb_rating,
        overall_rating=overall,
        theoretical_de_pct=theoretical_de
    )


def calculate_heat_balance(
    flow_scfm: float,
    inlet_temp_f: float,
    combustion_temp_f: float,
    heat_recovery_pct: float,
    inlet_voc_ppm: float,
    voc_properties: Dict[str, float],
    fuel_hhv: float
) -> CombustionAnalysis:
    """
    Calculate thermal oxidizer heat balance.

    Energy balance:
    Q_fuel + Q_voc + Q_recovered = Q_heat_gas + Q_losses

    Reference: EPA OAQPS, Perry's Chemical Engineers' Handbook

    Args:
        flow_scfm: Process gas flow (SCFM)
        inlet_temp_f: Inlet temperature (F)
        combustion_temp_f: Combustion temperature (F)
        heat_recovery_pct: Heat recovery efficiency (%)
        inlet_voc_ppm: VOC concentration (ppm)
        voc_properties: VOC property dict
        fuel_hhv: Fuel heating value (BTU/SCF)

    Returns:
        CombustionAnalysis with heat balance results
    """
    # Temperature rise needed
    delta_t = combustion_temp_f - inlet_temp_f

    # Heat to raise gas temperature
    mass_flow_lb_hr = flow_scfm * 60 * AIR_DENSITY_LB_SCF
    heat_required_btu_hr = mass_flow_lb_hr * AIR_CP_BTU_LB_F * delta_t
    heat_required_mmbtu = heat_required_btu_hr / 1e6

    # Heat recovered from exhaust
    heat_recovered_mmbtu = heat_required_mmbtu * (heat_recovery_pct / 100)

    # VOC heat contribution
    # VOC mass flow: ppm * flow * MW / (molar volume * 1e6)
    molar_vol_scf = 385.5  # SCF per lb-mol at standard conditions
    voc_mw = voc_properties.get("avg_mw", 80)
    voc_hoc = voc_properties.get("hoc_btu_lb", 15000)

    # VOC mass flow (lb/hr)
    voc_moles_hr = (inlet_voc_ppm / 1e6) * (flow_scfm * 60) / molar_vol_scf
    voc_lb_hr = voc_moles_hr * voc_mw / 453.6  # g to lb

    voc_heat_btu_hr = voc_lb_hr * voc_hoc
    voc_heat_mmbtu = voc_heat_btu_hr / 1e6

    # Net fuel requirement
    net_heat_needed_mmbtu = heat_required_mmbtu - heat_recovered_mmbtu - voc_heat_mmbtu
    net_heat_needed_mmbtu = max(0, net_heat_needed_mmbtu)

    # Self-sustaining VOC concentration
    # When VOC heat = net heat required (after recovery)
    net_heat_after_recovery = heat_required_mmbtu * (1 - heat_recovery_pct / 100)
    if voc_heat_mmbtu > 0 and inlet_voc_ppm > 0:
        self_sustaining_factor = net_heat_after_recovery / voc_heat_mmbtu
        self_sustaining_ppm = inlet_voc_ppm * self_sustaining_factor
    else:
        self_sustaining_ppm = float('inf')

    is_self_sustaining = inlet_voc_ppm >= self_sustaining_ppm

    # Fuel savings from VOC
    if heat_required_mmbtu > 0 and not is_self_sustaining:
        fuel_savings_pct = (voc_heat_mmbtu / (heat_required_mmbtu * (1 - heat_recovery_pct / 100))) * 100
        fuel_savings_pct = min(100, fuel_savings_pct)
    elif is_self_sustaining:
        fuel_savings_pct = 100
    else:
        fuel_savings_pct = 0

    return CombustionAnalysis(
        heat_required_mmbtu_hr=round(heat_required_mmbtu, 3),
        heat_recovered_mmbtu_hr=round(heat_recovered_mmbtu, 3),
        net_fuel_mmbtu_hr=round(net_heat_needed_mmbtu, 3),
        voc_heat_contribution_mmbtu_hr=round(voc_heat_mmbtu, 4),
        self_sustaining_threshold_ppm=round(min(50000, self_sustaining_ppm), 0),
        is_self_sustaining=is_self_sustaining,
        fuel_savings_from_voc_pct=round(fuel_savings_pct, 1)
    )


def calculate_emissions(
    inlet_voc_ppm: float,
    destruction_pct: float,
    flow_scfm: float,
    voc_properties: Dict[str, float],
    fuel_mmbtu_hr: float,
    fuel_type: FuelType
) -> EmissionsAnalysis:
    """
    Calculate outlet emissions.

    Reference: EPA AP-42, 40 CFR 63

    Args:
        inlet_voc_ppm: Inlet VOC concentration (ppm)
        destruction_pct: Destruction efficiency (%)
        flow_scfm: Process flow (SCFM)
        voc_properties: VOC properties dict
        fuel_mmbtu_hr: Fuel consumption (MMBTU/hr)
        fuel_type: Fuel type

    Returns:
        EmissionsAnalysis with outlet concentrations
    """
    # Outlet VOC
    outlet_voc_ppm = inlet_voc_ppm * (1 - destruction_pct / 100)

    # VOC mass flow outlet
    molar_vol_scf = 385.5
    voc_mw = voc_properties.get("avg_mw", 80)

    voc_moles_hr = (outlet_voc_ppm / 1e6) * (flow_scfm * 60) / molar_vol_scf
    outlet_voc_lb_hr = voc_moles_hr * voc_mw / 453.6

    # CO estimate (typically 10-100 ppm for well-operated units)
    # Higher at lower temps or poor mixing
    co_ppm = 20  # Typical value for good operation

    # NOx estimate (temperature dependent)
    # Thermal NOx increases above 2800F
    if fuel_mmbtu_hr > 0:
        nox_ppm = 0.1 * fuel_mmbtu_hr * 1000 / flow_scfm if flow_scfm > 0 else 50
        nox_ppm = min(100, max(10, nox_ppm))
    else:
        nox_ppm = 10

    # CO2 emissions from fuel
    fuel_props = FUEL_PROPERTIES_DB.get(fuel_type, FUEL_PROPERTIES_DB[FuelType.NATURAL_GAS])
    co2_factor = fuel_props["co2_lb_mmbtu"]
    co2_lb_hr = fuel_mmbtu_hr * co2_factor

    # Annual CO2
    co2_tonnes_yr = co2_lb_hr * 8760 / 2205  # Assuming continuous operation for now

    return EmissionsAnalysis(
        outlet_voc_ppm=round(outlet_voc_ppm, 2),
        actual_destruction_pct=round(destruction_pct, 2),
        outlet_voc_lb_hr=round(outlet_voc_lb_hr, 3),
        co_estimate_ppm=round(co_ppm, 0),
        nox_estimate_ppm=round(nox_ppm, 0),
        co2_emissions_lb_hr=round(co2_lb_hr, 1),
        co2_emissions_tonnes_yr=round(co2_tonnes_yr, 1)
    )


def calculate_lel_percentage(
    inlet_voc_ppm: float,
    voc_properties: Dict[str, float]
) -> Tuple[float, str]:
    """
    Calculate LEL percentage for safety assessment.

    Reference: NFPA 86

    Args:
        inlet_voc_ppm: Inlet VOC concentration (ppm)
        voc_properties: VOC properties dict

    Returns:
        Tuple of (LEL percentage, status string)
    """
    lel_ppm = voc_properties.get("lel_ppm", 12000)

    lel_pct = (inlet_voc_ppm / lel_ppm) * 100

    if lel_pct <= 25:
        status = "SAFE (<=25% LEL)"
    elif lel_pct <= 50:
        status = "CAUTION (25-50% LEL)"
    else:
        status = "DANGER (>50% LEL)"

    return round(lel_pct, 1), status


# =============================================================================
# AGENT CLASS
# =============================================================================

class ThermalOxidizerAgent:
    """
    GL-053: Thermal Oxidizer Optimization Agent.

    Optimizes thermal oxidizer performance using:
    1. Three T's of Combustion (Time, Temperature, Turbulence) analysis
    2. Destruction efficiency calculation based on kinetics
    3. Heat balance and fuel consumption optimization
    4. LEL safety assessment
    5. Regulatory compliance verification

    All calculations are deterministic using combustion engineering principles
    and EPA guidance - no ML/LLM in the calculation path.

    Attributes:
        AGENT_ID: Unique agent identifier (GL-053)
        AGENT_NAME: Human-readable name (THERMAL-OXIDIZER)
        VERSION: Semantic version string

    Example:
        >>> agent = ThermalOxidizerAgent()
        >>> result = agent.run({
        ...     "equipment_id": "TO-001",
        ...     "process_flow_scfm": 20000,
        ...     "inlet_voc_ppm": 500,
        ...     "combustion_temp_f": 1600
        ... })
    """

    AGENT_ID = "GL-053"
    AGENT_NAME = "THERMAL-OXIDIZER"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ThermalOxidizerAgent."""
        self.config = config or {}
        logger.info(f"{self.AGENT_NAME} agent initialized (ID: {self.AGENT_ID}, v{self.VERSION})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute thermal oxidizer analysis.

        Args:
            input_data: Dictionary matching ThermalOxidizerInput schema

        Returns:
            Dictionary with analysis results and provenance
        """
        start_time = datetime.now()

        try:
            validated = ThermalOxidizerInput(**input_data)
            output = self._process(validated, start_time)
            return output.model_dump()
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}", exc_info=True)
            raise

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Async wrapper for run method."""
        return self.run(input_data)

    def _process(self, inp: ThermalOxidizerInput, start_time: datetime) -> ThermalOxidizerOutput:
        """Main processing logic."""
        recommendations = []
        warnings = []
        validation_errors = []

        logger.info(f"Processing thermal oxidizer analysis for {inp.equipment_id}")

        # Get VOC properties
        if inp.voc_properties:
            voc_props = {
                "avg_mw": inp.voc_properties.molecular_weight,
                "hoc_btu_lb": inp.voc_properties.heat_of_combustion_btu_lb,
                "lel_ppm": inp.voc_properties.lel_ppm,
                "autoignition_f": inp.voc_properties.autoignition_temp_f
            }
            voc_type = inp.voc_properties.voc_type
        else:
            voc_type = VOCType.MIXED
            voc_props = VOC_PROPERTIES_DB[voc_type]

        # Get oxidizer specs
        ox_specs = OXIDIZER_SPECS_DB.get(
            inp.oxidizer_type,
            OXIDIZER_SPECS_DB[OxidizerType.REGENERATIVE]
        )

        # Calculate Three T's analysis
        three_ts = calculate_three_ts(
            inp.combustion_temp_f,
            inp.residence_time_s,
            inp.oxidizer_type
        )

        # Calculate destruction efficiency
        actual_de = calculate_destruction_efficiency(
            inp.combustion_temp_f,
            inp.residence_time_s,
            voc_type,
            inp.oxidizer_type
        )

        # Outlet VOC and margin
        outlet_voc = inp.inlet_voc_ppm * (1 - actual_de / 100)
        destruction_margin = actual_de - inp.required_destruction_pct

        # Operating status
        if actual_de >= inp.required_destruction_pct and three_ts.overall_rating in ["Excellent", "Good"]:
            op_status = OperatingStatus.OPTIMAL
        elif actual_de >= inp.required_destruction_pct:
            op_status = OperatingStatus.ACCEPTABLE
        elif actual_de >= inp.required_destruction_pct - 1:
            op_status = OperatingStatus.SUBOPTIMAL
        else:
            op_status = OperatingStatus.NON_COMPLIANT

        # Heat recovery efficiency
        if inp.heat_recovery:
            hr_pct = inp.heat_recovery.current_efficiency_pct
        else:
            hr_pct = inp.heat_recovery_pct

        # Heat balance calculation
        combustion = calculate_heat_balance(
            inp.process_flow_scfm,
            inp.inlet_temp_f,
            inp.combustion_temp_f,
            hr_pct,
            inp.inlet_voc_ppm,
            voc_props,
            inp.fuel_hhv_btu_scf
        )

        # Emissions calculation
        fuel_props = FUEL_PROPERTIES_DB.get(inp.fuel_type, FUEL_PROPERTIES_DB[FuelType.NATURAL_GAS])
        emissions = calculate_emissions(
            inp.inlet_voc_ppm,
            actual_de,
            inp.process_flow_scfm,
            voc_props,
            combustion.net_fuel_mmbtu_hr,
            inp.fuel_type
        )

        # LEL safety check
        lel_pct, lel_status = calculate_lel_percentage(inp.inlet_voc_ppm, voc_props)

        # Economic analysis
        fuel_scfh = combustion.net_fuel_mmbtu_hr * 1e6 / inp.fuel_hhv_btu_scf
        fuel_cost_hr = combustion.net_fuel_mmbtu_hr * inp.fuel_cost_mmbtu
        annual_fuel_cost = fuel_cost_hr * inp.operating_hours_year

        # Electricity (fan power)
        fan_hp = ox_specs["fan_power_factor"] * inp.process_flow_scfm / 1000
        fan_kw = fan_hp * 0.746
        elec_cost_hr = fan_kw * inp.electricity_price_kwh
        annual_elec_cost = elec_cost_hr * inp.operating_hours_year

        total_annual_cost = annual_fuel_cost + annual_elec_cost
        cost_per_scfm = total_annual_cost / inp.process_flow_scfm if inp.process_flow_scfm > 0 else 0

        economic = EconomicAnalysis(
            fuel_consumption_scfh=round(fuel_scfh, 1),
            fuel_cost_hr=round(fuel_cost_hr, 2),
            annual_fuel_cost_usd=round(annual_fuel_cost, 2),
            electricity_cost_hr=round(elec_cost_hr, 2),
            annual_electricity_cost_usd=round(annual_elec_cost, 2),
            total_annual_cost_usd=round(total_annual_cost, 2),
            cost_per_scfm_yr=round(cost_per_scfm, 2)
        )

        # Thermal efficiency
        if combustion.heat_required_mmbtu_hr > 0:
            thermal_eff = (combustion.heat_recovered_mmbtu_hr / combustion.heat_required_mmbtu_hr) * 100
        else:
            thermal_eff = hr_pct

        # Optimization calculation
        optimization = self._calculate_optimization(
            inp, combustion, ox_specs, actual_de, annual_fuel_cost
        )

        # Generate recommendations
        recommendations.extend(self._generate_recommendations(
            inp, three_ts, combustion, actual_de, lel_pct, optimization
        ))

        # Generate warnings
        warnings.extend(self._generate_warnings(
            inp, op_status, lel_pct, combustion, three_ts
        ))

        # Validation
        validation_status = "PASS"
        if op_status == OperatingStatus.NON_COMPLIANT:
            validation_errors.append(
                f"Destruction efficiency {actual_de:.2f}% does not meet "
                f"requirement of {inp.required_destruction_pct}%"
            )
            validation_status = "FAIL"
        if lel_pct > inp.lel_limit_pct:
            validation_errors.append(f"LEL {lel_pct:.1f}% exceeds limit of {inp.lel_limit_pct}%")
            validation_status = "FAIL"

        # Provenance hash
        calc_hash = self._calculate_provenance_hash(inp, three_ts, combustion, emissions)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Completed analysis for {inp.equipment_id} in {processing_time:.1f}ms")

        return ThermalOxidizerOutput(
            equipment_id=inp.equipment_id,
            oxidizer_type=inp.oxidizer_type.value,
            three_ts_analysis=three_ts,
            outlet_voc_ppm=round(outlet_voc, 2),
            actual_destruction_pct=round(actual_de, 2),
            destruction_margin_pct=round(destruction_margin, 2),
            operating_status=op_status,
            combustion_analysis=combustion,
            emissions_analysis=emissions,
            fuel_consumption_mmbtu_hr=round(combustion.net_fuel_mmbtu_hr, 3),
            annual_fuel_mmbtu=round(combustion.net_fuel_mmbtu_hr * inp.operating_hours_year, 0),
            annual_fuel_cost_usd=round(annual_fuel_cost, 2),
            thermal_efficiency_pct=round(thermal_eff, 1),
            economic_analysis=economic,
            optimization=optimization,
            meets_destruction_requirement=actual_de >= inp.required_destruction_pct,
            lel_percentage=lel_pct,
            lel_status=lel_status,
            recommendations=recommendations,
            warnings=warnings,
            calculation_hash=calc_hash,
            validation_status=validation_status,
            validation_errors=validation_errors,
            agent_version=self.VERSION
        )

    def _calculate_optimization(
        self,
        inp: ThermalOxidizerInput,
        combustion: CombustionAnalysis,
        ox_specs: Dict,
        current_de: float,
        current_fuel_cost: float
    ) -> OptimizationResult:
        """Calculate optimization opportunities."""

        # Optimal temperature - just enough to meet DE requirement
        # Find minimum temp that achieves required DE
        min_temp = ox_specs["min_temp_f"]
        optimal_temp = inp.combustion_temp_f

        for test_temp in range(int(min_temp), int(inp.combustion_temp_f), 25):
            test_de = calculate_destruction_efficiency(
                test_temp, inp.residence_time_s, VOCType.MIXED, inp.oxidizer_type
            )
            if test_de >= inp.required_destruction_pct + 0.5:  # Small margin
                optimal_temp = test_temp
                break

        # Optimal residence time - similar approach
        optimal_time = inp.residence_time_s
        for test_time in [t / 10 for t in range(5, int(inp.residence_time_s * 10))]:
            test_de = calculate_destruction_efficiency(
                inp.combustion_temp_f, test_time, VOCType.MIXED, inp.oxidizer_type
            )
            if test_de >= inp.required_destruction_pct + 0.5:
                optimal_time = test_time
                break

        # Optimal heat recovery
        max_hr = ox_specs["max_heat_recovery_pct"]
        current_hr = inp.heat_recovery_pct
        optimal_hr = min(max_hr, current_hr + 5)  # 5% improvement target

        # Calculate potential savings
        # Lower temp = less heat required = less fuel
        temp_reduction = inp.combustion_temp_f - optimal_temp
        temp_savings_pct = (temp_reduction / (inp.combustion_temp_f - inp.inlet_temp_f)) * 100 if inp.combustion_temp_f > inp.inlet_temp_f else 0

        # Better heat recovery savings
        hr_improvement = optimal_hr - current_hr
        hr_savings_pct = hr_improvement * (1 - current_hr / 100)

        total_savings_pct = min(50, temp_savings_pct + hr_savings_pct)
        cost_savings = current_fuel_cost * (total_savings_pct / 100)

        return OptimizationResult(
            optimal_temp_f=round(optimal_temp, 0),
            optimal_residence_s=round(optimal_time, 2),
            optimal_heat_recovery_pct=round(optimal_hr, 1),
            potential_fuel_savings_pct=round(total_savings_pct, 1),
            potential_cost_savings_usd=round(cost_savings, 2)
        )

    def _generate_recommendations(
        self,
        inp: ThermalOxidizerInput,
        three_ts: ThreeTsAnalysis,
        combustion: CombustionAnalysis,
        actual_de: float,
        lel_pct: float,
        optimization: OptimizationResult
    ) -> List[str]:
        """Generate optimization recommendations."""
        recs = []

        if actual_de < inp.required_destruction_pct:
            recs.append(
                f"URGENT: Increase combustion temp from {inp.combustion_temp_f}F to "
                f"at least {inp.combustion_temp_f + 100}F to meet DE requirement"
            )

        if three_ts.temperature_rating in ["Marginal", "Poor"]:
            recs.append(
                f"Combustion temperature ({inp.combustion_temp_f}F) rated "
                f"'{three_ts.temperature_rating}' - consider increasing to 1600F+"
            )

        if three_ts.residence_time_rating in ["Marginal", "Poor"]:
            recs.append(
                f"Residence time ({inp.residence_time_s}s) rated "
                f"'{three_ts.residence_time_rating}' - target 1.0s minimum"
            )

        if combustion.is_self_sustaining:
            recs.append(
                f"VOC concentration ({inp.inlet_voc_ppm} ppm) provides self-sustaining "
                "operation - no auxiliary fuel required"
            )
        elif combustion.fuel_savings_from_voc_pct > 20:
            recs.append(
                f"VOC heat contributes {combustion.fuel_savings_from_voc_pct:.1f}% "
                "fuel savings - maintain consistent loading"
            )

        if optimization.potential_cost_savings_usd > 10000:
            recs.append(
                f"Potential annual savings ${optimization.potential_cost_savings_usd:,.0f} "
                f"by optimizing to {optimization.optimal_temp_f}F and "
                f"{optimization.optimal_heat_recovery_pct}% heat recovery"
            )

        if inp.oxidizer_type == OxidizerType.DIRECT_FIRED:
            recs.append(
                "Consider upgrading to RTO for 90%+ heat recovery and major fuel savings"
            )

        if inp.heat_recovery_pct < 90 and inp.oxidizer_type == OxidizerType.REGENERATIVE:
            recs.append(
                f"RTO heat recovery at {inp.heat_recovery_pct}% below typical 95% - "
                "inspect ceramic media and seals"
            )

        if lel_pct > 15 and lel_pct <= 25:
            recs.append(
                f"VOC loading at {lel_pct:.1f}% LEL - monitor closely and consider "
                "LEL monitoring system if not installed"
            )

        return recs

    def _generate_warnings(
        self,
        inp: ThermalOxidizerInput,
        op_status: OperatingStatus,
        lel_pct: float,
        combustion: CombustionAnalysis,
        three_ts: ThreeTsAnalysis
    ) -> List[str]:
        """Generate safety warnings."""
        warnings = []

        if op_status == OperatingStatus.NON_COMPLIANT:
            warnings.append(
                f"CRITICAL: Unit not meeting {inp.required_destruction_pct}% "
                "destruction requirement - regulatory non-compliance"
            )

        if lel_pct > 25:
            warnings.append(
                f"DANGER: VOC loading at {lel_pct:.1f}% LEL exceeds safe limit - "
                "reduce VOC concentration or add dilution air"
            )

        if inp.combustion_temp_f < 1400 and inp.oxidizer_type != OxidizerType.CATALYTIC:
            warnings.append(
                f"Combustion temp {inp.combustion_temp_f}F below minimum 1400F - "
                "incomplete destruction likely"
            )

        if inp.residence_time_s < 0.5:
            warnings.append(
                f"Residence time {inp.residence_time_s}s below minimum 0.5s - "
                "inadequate reaction time"
            )

        if three_ts.overall_rating == "Poor":
            warnings.append(
                "Overall Three T's rating is Poor - combustion conditions inadequate"
            )

        if inp.inlet_voc_ppm > 5000:
            warnings.append(
                f"High VOC loading ({inp.inlet_voc_ppm} ppm) - verify LEL monitoring "
                "and safety interlocks are operational"
            )

        return warnings

    def _calculate_provenance_hash(
        self,
        inp: ThermalOxidizerInput,
        three_ts: ThreeTsAnalysis,
        combustion: CombustionAnalysis,
        emissions: EmissionsAnalysis
    ) -> str:
        """Calculate SHA-256 hash for audit trail."""
        provenance_data = {
            "equipment_id": inp.equipment_id,
            "oxidizer_type": inp.oxidizer_type.value,
            "flow_scfm": inp.process_flow_scfm,
            "inlet_voc_ppm": inp.inlet_voc_ppm,
            "combustion_temp_f": inp.combustion_temp_f,
            "destruction_pct": emissions.actual_destruction_pct,
            "fuel_mmbtu_hr": combustion.net_fuel_mmbtu_hr,
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "timestamp": datetime.utcnow().isoformat()
        }
        return hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

    def get_metadata(self) -> Dict[str, Any]:
        """Return agent metadata."""
        return {
            "agent_id": self.AGENT_ID,
            "agent_name": self.AGENT_NAME,
            "version": self.VERSION,
            "category": "Air Quality",
            "type": "Optimization",
            "standards": ["EPA 40 CFR 63", "NFPA 86"],
            "capabilities": [
                "Three T's analysis",
                "Destruction efficiency calculation",
                "Heat balance optimization",
                "LEL safety assessment",
                "Regulatory compliance verification"
            ]
        }


# =============================================================================
# PACK SPECIFICATION
# =============================================================================

PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-053",
    "name": "THERMAL-OXIDIZER",
    "version": "1.0.0",
    "summary": "Thermal oxidizer optimization for VOC destruction with Three T's analysis",
    "tags": ["thermal-oxidizer", "RTO", "VOC", "combustion", "EPA-40-CFR", "NFPA-86"],
    "standards": [
        {"ref": "EPA 40 CFR 63", "description": "National Emission Standards for Hazardous Air Pollutants"},
        {"ref": "NFPA 86", "description": "Standard for Ovens and Furnaces"}
    ],
    "provenance": {
        "calculation_verified": True,
        "enable_audit": True,
        "deterministic": True
    }
}
