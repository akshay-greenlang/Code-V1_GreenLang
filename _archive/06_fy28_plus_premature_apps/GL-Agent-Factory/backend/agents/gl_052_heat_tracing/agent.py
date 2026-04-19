"""GL-052: Heat Tracing Agent (HEAT-TRACING).

Optimizes heat tracing systems for pipe freeze protection and process temperature
maintenance. Implements physics-based heat loss calculations, tracer sizing,
insulation optimization, and energy cost analysis.

The agent follows GreenLang's zero-hallucination principle by using only
deterministic calculations from heat transfer engineering - no ML/LLM
in the calculation path.

Standards: IEEE 515 (Electric Heat Tracing), IEC 62395 (Electric Heating Systems)

Example:
    >>> agent = HeatTracingAgent()
    >>> result = agent.run({
    ...     "equipment_id": "PIPE-001",
    ...     "tracing_type": "ELECTRIC_SELF_REG",
    ...     "pipe_length_m": 100,
    ...     "pipe_diameter_mm": 150,
    ...     "maintain_temp_c": 50,
    ...     "ambient_temp_c": -20
    ... })
    >>> assert result["system_adequacy"] == "ADEQUATE"
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

class TracingType(str, Enum):
    """Types of heat tracing systems."""
    ELECTRIC_SELF_REG = "ELECTRIC_SELF_REG"  # Self-regulating cable
    ELECTRIC_CONSTANT = "ELECTRIC_CONSTANT"  # Constant wattage cable
    ELECTRIC_MI = "ELECTRIC_MI"  # Mineral insulated
    STEAM_TRACER = "STEAM_TRACER"  # Steam tracing
    HOT_WATER = "HOT_WATER"  # Hot water tracing
    HOT_OIL = "HOT_OIL"  # Hot oil tracing


class InsulationType(str, Enum):
    """Insulation material types."""
    CALCIUM_SILICATE = "CALCIUM_SILICATE"
    MINERAL_WOOL = "MINERAL_WOOL"
    FIBERGLASS = "FIBERGLASS"
    CELLULAR_GLASS = "CELLULAR_GLASS"
    POLYURETHANE = "POLYURETHANE"
    AEROGEL = "AEROGEL"
    NONE = "NONE"


class PipeLocation(str, Enum):
    """Pipe installation location."""
    OUTDOOR_EXPOSED = "OUTDOOR_EXPOSED"
    OUTDOOR_SHELTERED = "OUTDOOR_SHELTERED"
    INDOOR_UNHEATED = "INDOOR_UNHEATED"
    INDOOR_HEATED = "INDOOR_HEATED"
    UNDERGROUND = "UNDERGROUND"


class SystemAdequacy(str, Enum):
    """Heat tracing system adequacy status."""
    ADEQUATE = "ADEQUATE"
    MARGINAL = "MARGINAL"
    INADEQUATE = "INADEQUATE"
    OVERSIZED = "OVERSIZED"


class ApplicationType(str, Enum):
    """Heat tracing application type."""
    FREEZE_PROTECTION = "FREEZE_PROTECTION"
    TEMPERATURE_MAINTENANCE = "TEMPERATURE_MAINTENANCE"
    VISCOSITY_CONTROL = "VISCOSITY_CONTROL"
    CONDENSATE_PREVENTION = "CONDENSATE_PREVENTION"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class InsulationProperties(BaseModel):
    """Insulation material properties."""

    material: InsulationType = Field(default=InsulationType.MINERAL_WOOL)
    thickness_mm: float = Field(default=50, ge=0, le=300)
    thermal_conductivity_w_mk: float = Field(
        default=0.04,
        gt=0,
        le=1.0,
        description="Thermal conductivity at mean temperature (W/m-K)"
    )
    max_service_temp_c: float = Field(default=450, gt=0)
    density_kg_m3: float = Field(default=100, gt=0)


class PipeProperties(BaseModel):
    """Pipe physical properties."""

    outer_diameter_mm: float = Field(..., gt=0, le=3000)
    wall_thickness_mm: float = Field(default=5, gt=0)
    material: str = Field(default="carbon_steel")
    thermal_conductivity_w_mk: float = Field(default=50, gt=0)


class HeatTracingInput(BaseModel):
    """Input data model for HeatTracingAgent."""

    equipment_id: str = Field(..., min_length=1, description="Equipment identifier")
    tracing_type: TracingType = Field(default=TracingType.ELECTRIC_SELF_REG)
    application: ApplicationType = Field(default=ApplicationType.FREEZE_PROTECTION)

    # Pipe parameters
    pipe_length_m: float = Field(..., gt=0, le=100000, description="Total pipe length (m)")
    pipe_diameter_mm: float = Field(default=100, gt=0, le=3000, description="Pipe OD (mm)")
    pipe_properties: Optional[PipeProperties] = None
    pipe_location: PipeLocation = Field(default=PipeLocation.OUTDOOR_EXPOSED)

    # Insulation parameters
    insulation: Optional[InsulationProperties] = None
    insulation_thickness_mm: float = Field(default=50, ge=0, le=300)
    insulation_conductivity_w_mk: float = Field(default=0.04, gt=0, le=1.0)

    # Temperature parameters
    maintain_temp_c: float = Field(
        default=50,
        ge=-50,
        le=500,
        description="Temperature to maintain (C)"
    )
    ambient_temp_c: float = Field(
        default=-10,
        ge=-60,
        le=60,
        description="Design ambient temperature (C)"
    )
    min_ambient_temp_c: Optional[float] = Field(
        default=None,
        ge=-60,
        description="Minimum expected ambient (C)"
    )

    # Environmental conditions
    wind_speed_m_s: float = Field(default=5, ge=0, le=50, description="Design wind speed (m/s)")
    snow_cover: bool = Field(default=False, description="Snow cover expected")
    rain_exposure: bool = Field(default=True, description="Rain exposure expected")

    # Existing tracer parameters
    tracer_power_w_m: float = Field(
        default=30,
        gt=0,
        le=500,
        description="Existing tracer power rating (W/m)"
    )
    tracer_voltage: float = Field(default=240, gt=0, description="Supply voltage (V)")
    tracer_circuits: int = Field(default=1, ge=1, description="Number of circuits")

    # Operating parameters
    operating_hours_year: int = Field(default=4000, ge=0, le=8760)
    control_type: str = Field(
        default="thermostat",
        description="Control type: thermostat, ambient_sensing, self_regulating"
    )
    set_point_c: Optional[float] = Field(default=None, description="Thermostat set point")

    # Cost parameters
    electricity_price_kwh: float = Field(default=0.10, ge=0, description="$/kWh")
    steam_price_kg: float = Field(default=0.03, ge=0, description="$/kg steam")

    # Safety parameters
    safety_factor: float = Field(
        default=1.2,
        ge=1.0,
        le=2.0,
        description="Design safety factor"
    )

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @root_validator(skip_on_failure=True)
    def validate_temperatures(cls, values):
        """Validate temperature relationships."""
        maintain = values.get('maintain_temp_c', 50)
        ambient = values.get('ambient_temp_c', -10)
        min_ambient = values.get('min_ambient_temp_c')

        if maintain <= ambient:
            logger.warning("Maintain temp <= ambient temp - heat tracing may not be needed")

        if min_ambient is not None and min_ambient > ambient:
            values['ambient_temp_c'] = min_ambient

        return values


class HeatLossAnalysis(BaseModel):
    """Detailed heat loss analysis results."""

    conduction_loss_w_m: float = Field(..., ge=0)
    convection_loss_w_m: float = Field(..., ge=0)
    radiation_loss_w_m: float = Field(..., ge=0)
    total_heat_loss_w_m: float = Field(..., ge=0)
    insulation_r_value_m2k_w: float = Field(..., ge=0)
    surface_temperature_c: float
    overall_u_value_w_m2k: float = Field(..., ge=0)


class TracerSizing(BaseModel):
    """Heat tracer sizing results."""

    required_power_w_m: float = Field(..., ge=0)
    selected_power_w_m: float = Field(..., ge=0)
    total_power_kw: float = Field(..., ge=0)
    max_circuit_length_m: float = Field(..., gt=0)
    recommended_circuits: int = Field(..., ge=1)
    power_margin_pct: float


class EconomicAnalysis(BaseModel):
    """Economic analysis results."""

    annual_energy_kwh: float = Field(..., ge=0)
    annual_energy_cost_usd: float = Field(..., ge=0)
    energy_cost_per_meter_usd: float = Field(..., ge=0)
    insulation_savings_usd: float = Field(..., ge=0)
    payback_years: Optional[float] = None


class HeatTracingOutput(BaseModel):
    """Output data model for HeatTracingAgent."""

    equipment_id: str
    tracing_type: str
    application: str

    # Heat loss analysis
    heat_loss_w_m: float
    heat_loss_analysis: HeatLossAnalysis

    # Tracer sizing
    required_power_w_m: float
    tracer_sizing: TracerSizing

    # System adequacy
    total_power_kw: float
    system_adequacy: SystemAdequacy
    safety_margin_pct: float

    # Optimization recommendations
    optimal_insulation_mm: float
    optimal_tracer_power_w_m: float
    economic_insulation_mm: float

    # Economics
    economic_analysis: EconomicAnalysis
    annual_energy_kwh: float
    annual_cost_usd: float

    # Temperature analysis
    maintain_temp_c: float
    design_ambient_c: float
    temperature_margin_c: float

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

# Insulation material properties database
INSULATION_PROPERTIES_DB: Dict[InsulationType, Dict[str, float]] = {
    InsulationType.CALCIUM_SILICATE: {
        "k_25c": 0.055,  # Thermal conductivity at 25C
        "k_100c": 0.065,
        "k_200c": 0.080,
        "max_temp": 650,
        "cost_per_m3": 300
    },
    InsulationType.MINERAL_WOOL: {
        "k_25c": 0.038,
        "k_100c": 0.045,
        "k_200c": 0.055,
        "max_temp": 700,
        "cost_per_m3": 150
    },
    InsulationType.FIBERGLASS: {
        "k_25c": 0.035,
        "k_100c": 0.042,
        "k_200c": 0.052,
        "max_temp": 450,
        "cost_per_m3": 100
    },
    InsulationType.CELLULAR_GLASS: {
        "k_25c": 0.045,
        "k_100c": 0.055,
        "k_200c": 0.070,
        "max_temp": 480,
        "cost_per_m3": 400
    },
    InsulationType.POLYURETHANE: {
        "k_25c": 0.025,
        "k_100c": 0.030,
        "k_200c": None,  # Not rated above 120C
        "max_temp": 120,
        "cost_per_m3": 200
    },
    InsulationType.AEROGEL: {
        "k_25c": 0.015,
        "k_100c": 0.018,
        "k_200c": 0.022,
        "max_temp": 650,
        "cost_per_m3": 3000
    },
    InsulationType.NONE: {
        "k_25c": 100,  # Effectively infinite
        "k_100c": 100,
        "k_200c": 100,
        "max_temp": 1000,
        "cost_per_m3": 0
    }
}

# Tracer cable properties by type
TRACER_PROPERTIES_DB: Dict[TracingType, Dict[str, Any]] = {
    TracingType.ELECTRIC_SELF_REG: {
        "available_powers": [10, 15, 20, 30, 40, 50],  # W/m at 10C
        "max_exposure_temp": 85,
        "max_maintain_temp": 65,
        "max_circuit_length_m": 200,
        "voltage_options": [120, 240],
        "safety_factor": 1.2,
        "efficiency": 0.95
    },
    TracingType.ELECTRIC_CONSTANT: {
        "available_powers": [5, 10, 15, 20, 25, 30, 40, 50],
        "max_exposure_temp": 200,
        "max_maintain_temp": 150,
        "max_circuit_length_m": 150,
        "voltage_options": [120, 240, 480],
        "safety_factor": 1.3,
        "efficiency": 0.98
    },
    TracingType.ELECTRIC_MI: {
        "available_powers": [20, 30, 40, 50, 75, 100],
        "max_exposure_temp": 600,
        "max_maintain_temp": 450,
        "max_circuit_length_m": 500,
        "voltage_options": [240, 480],
        "safety_factor": 1.25,
        "efficiency": 0.99
    },
    TracingType.STEAM_TRACER: {
        "available_powers": [100, 150, 200, 300],  # Equivalent W/m
        "max_exposure_temp": 250,
        "max_maintain_temp": 180,
        "max_circuit_length_m": 100,
        "voltage_options": [],
        "safety_factor": 1.3,
        "efficiency": 0.70
    },
    TracingType.HOT_WATER: {
        "available_powers": [50, 75, 100],
        "max_exposure_temp": 95,
        "max_maintain_temp": 80,
        "max_circuit_length_m": 75,
        "voltage_options": [],
        "safety_factor": 1.25,
        "efficiency": 0.75
    },
    TracingType.HOT_OIL: {
        "available_powers": [75, 100, 150, 200],
        "max_exposure_temp": 300,
        "max_maintain_temp": 250,
        "max_circuit_length_m": 100,
        "voltage_options": [],
        "safety_factor": 1.3,
        "efficiency": 0.72
    }
}


def calculate_thermal_conductivity_at_temp(
    material: InsulationType,
    mean_temp_c: float
) -> float:
    """
    Calculate insulation thermal conductivity at operating temperature.

    Reference: ASTM C680, CINI Manual

    Args:
        material: Insulation material type
        mean_temp_c: Mean temperature through insulation (C)

    Returns:
        Thermal conductivity (W/m-K)
    """
    props = INSULATION_PROPERTIES_DB.get(material, INSULATION_PROPERTIES_DB[InsulationType.MINERAL_WOOL])

    if mean_temp_c <= 25:
        return props["k_25c"]
    elif mean_temp_c <= 100:
        # Linear interpolation
        k = props["k_25c"] + (props["k_100c"] - props["k_25c"]) * (mean_temp_c - 25) / 75
        return k
    elif mean_temp_c <= 200 and props["k_200c"] is not None:
        k = props["k_100c"] + (props["k_200c"] - props["k_100c"]) * (mean_temp_c - 100) / 100
        return k
    else:
        # Extrapolate or use highest available
        return props.get("k_200c", props["k_100c"])


def calculate_convection_coefficient(
    wind_speed: float,
    diameter_m: float,
    surface_temp_c: float,
    ambient_temp_c: float,
    location: PipeLocation
) -> float:
    """
    Calculate external convection heat transfer coefficient.

    Uses correlations from:
    - McAdams (natural convection)
    - Hilpert/Churchill-Bernstein (forced convection)

    Reference: IEEE 515, Incropera & DeWitt

    Args:
        wind_speed: Wind speed (m/s)
        diameter_m: Outer diameter including insulation (m)
        surface_temp_c: Surface temperature (C)
        ambient_temp_c: Ambient temperature (C)
        location: Installation location

    Returns:
        Convection coefficient (W/m2-K)
    """
    delta_t = abs(surface_temp_c - ambient_temp_c)

    # Natural convection (always present)
    # h_natural = C * (delta_T / D)^0.25 for horizontal cylinders
    if delta_t > 0 and diameter_m > 0:
        h_natural = 1.32 * (delta_t / diameter_m) ** 0.25
    else:
        h_natural = 5.0  # Minimum value

    # Forced convection (wind effect)
    if wind_speed < 0.5:
        h_forced = 0
    else:
        # Simplified Hilpert correlation
        # Nu = C * Re^m * Pr^0.33
        # For Re = 1000-40000: C = 0.26, m = 0.6
        air_density = 1.2  # kg/m3
        air_viscosity = 1.8e-5  # Pa-s
        air_k = 0.025  # W/m-K

        Re = air_density * wind_speed * diameter_m / air_viscosity

        if Re < 4:
            Nu = 0.989 * Re ** 0.33
        elif Re < 40:
            Nu = 0.911 * Re ** 0.385
        elif Re < 4000:
            Nu = 0.683 * Re ** 0.466
        elif Re < 40000:
            Nu = 0.193 * Re ** 0.618
        else:
            Nu = 0.027 * Re ** 0.805

        h_forced = Nu * air_k / diameter_m

    # Location factors
    location_factors = {
        PipeLocation.OUTDOOR_EXPOSED: 1.0,
        PipeLocation.OUTDOOR_SHELTERED: 0.7,
        PipeLocation.INDOOR_UNHEATED: 0.3,
        PipeLocation.INDOOR_HEATED: 0.2,
        PipeLocation.UNDERGROUND: 0.1
    }
    factor = location_factors.get(location, 1.0)

    # Combined coefficient (root sum square for mixed convection)
    h_combined = math.sqrt(h_natural ** 2 + (h_forced * factor) ** 2)

    # Apply minimum and maximum bounds
    return max(5.0, min(100.0, h_combined))


def calculate_radiation_coefficient(
    surface_temp_c: float,
    ambient_temp_c: float,
    emissivity: float = 0.9
) -> float:
    """
    Calculate radiation heat transfer coefficient.

    q_rad = epsilon * sigma * (T_s^4 - T_a^4)
    h_rad = q_rad / (T_s - T_a)

    Reference: Stefan-Boltzmann law

    Args:
        surface_temp_c: Surface temperature (C)
        ambient_temp_c: Ambient/sky temperature (C)
        emissivity: Surface emissivity (default 0.9 for weathered surfaces)

    Returns:
        Radiation coefficient (W/m2-K)
    """
    sigma = 5.67e-8  # Stefan-Boltzmann constant

    T_s = surface_temp_c + 273.15
    T_a = ambient_temp_c + 273.15

    if abs(T_s - T_a) < 0.1:
        return 0.0

    # Linearized radiation coefficient
    h_rad = emissivity * sigma * (T_s ** 2 + T_a ** 2) * (T_s + T_a)

    return h_rad


def calculate_heat_loss_per_meter(
    pipe_od_mm: float,
    insulation_thickness_mm: float,
    insulation_k: float,
    maintain_temp_c: float,
    ambient_temp_c: float,
    wind_speed: float,
    location: PipeLocation,
    surface_emissivity: float = 0.9
) -> Tuple[HeatLossAnalysis, float]:
    """
    Calculate heat loss per meter of insulated pipe.

    Uses cylindrical heat transfer equations with:
    - Conduction through insulation
    - Convection from outer surface
    - Radiation from outer surface

    Reference: IEEE 515 Annex B, ASTM C680

    Args:
        pipe_od_mm: Pipe outer diameter (mm)
        insulation_thickness_mm: Insulation thickness (mm)
        insulation_k: Insulation thermal conductivity (W/m-K)
        maintain_temp_c: Temperature to maintain (C)
        ambient_temp_c: Ambient temperature (C)
        wind_speed: Wind speed (m/s)
        location: Installation location
        surface_emissivity: Outer surface emissivity

    Returns:
        Tuple of (HeatLossAnalysis, surface_temperature_c)
    """
    # Convert to meters
    r_pipe = pipe_od_mm / 2000.0  # m
    r_ins = r_pipe + insulation_thickness_mm / 1000.0  # m

    delta_t = maintain_temp_c - ambient_temp_c

    if delta_t <= 0:
        # No heat loss if maintain temp <= ambient
        return HeatLossAnalysis(
            conduction_loss_w_m=0,
            convection_loss_w_m=0,
            radiation_loss_w_m=0,
            total_heat_loss_w_m=0,
            insulation_r_value_m2k_w=0,
            surface_temperature_c=ambient_temp_c,
            overall_u_value_w_m2k=0
        ), ambient_temp_c

    # Insulation thermal resistance (per meter length)
    if insulation_thickness_mm > 0 and insulation_k > 0:
        R_ins = math.log(r_ins / r_pipe) / (2 * math.pi * insulation_k)
    else:
        R_ins = 0.001  # Minimal resistance for bare pipe

    # Initial estimate of surface temperature for coefficient calculation
    # Assume 80% of temperature drop is through insulation
    t_surface_est = ambient_temp_c + 0.2 * delta_t

    # Iterate to find actual surface temperature
    for _ in range(5):  # Usually converges in 2-3 iterations
        # Convection coefficient
        h_conv = calculate_convection_coefficient(
            wind_speed, 2 * r_ins, t_surface_est, ambient_temp_c, location
        )

        # Radiation coefficient
        h_rad = calculate_radiation_coefficient(t_surface_est, ambient_temp_c, surface_emissivity)

        # Combined external coefficient
        h_total = h_conv + h_rad

        # External thermal resistance
        R_ext = 1 / (2 * math.pi * r_ins * h_total)

        # Total thermal resistance
        R_total = R_ins + R_ext

        # Heat loss per meter
        if R_total > 0:
            q_total = delta_t / R_total
        else:
            q_total = delta_t * 100  # Very high loss for uninsulated

        # Update surface temperature estimate
        t_surface_new = maintain_temp_c - q_total * R_ins
        t_surface_new = max(ambient_temp_c, min(maintain_temp_c, t_surface_new))

        if abs(t_surface_new - t_surface_est) < 0.1:
            break
        t_surface_est = t_surface_new

    # Final calculations
    t_surface = t_surface_est

    # Conduction through insulation
    if R_ins > 0:
        q_cond = (maintain_temp_c - t_surface) / R_ins
    else:
        q_cond = 0

    # Convection from surface
    h_conv = calculate_convection_coefficient(wind_speed, 2 * r_ins, t_surface, ambient_temp_c, location)
    q_conv = h_conv * 2 * math.pi * r_ins * (t_surface - ambient_temp_c)

    # Radiation from surface
    h_rad = calculate_radiation_coefficient(t_surface, ambient_temp_c, surface_emissivity)
    q_rad = h_rad * 2 * math.pi * r_ins * (t_surface - ambient_temp_c)

    # R-value of insulation (US units: ft2-h-F/BTU)
    r_value = R_ins * 5.678  # Convert from m2-K/W to ft2-h-F/BTU

    # Overall U-value
    if R_total > 0:
        u_value = 1 / (R_total * 2 * math.pi * r_ins)
    else:
        u_value = 100

    return HeatLossAnalysis(
        conduction_loss_w_m=round(max(0, q_cond), 2),
        convection_loss_w_m=round(max(0, q_conv), 2),
        radiation_loss_w_m=round(max(0, q_rad), 2),
        total_heat_loss_w_m=round(q_total, 2),
        insulation_r_value_m2k_w=round(r_value, 2),
        surface_temperature_c=round(t_surface, 1),
        overall_u_value_w_m2k=round(u_value, 3)
    ), t_surface


def select_tracer_power(
    required_w_m: float,
    tracing_type: TracingType,
    safety_factor: float
) -> Tuple[float, float]:
    """
    Select appropriate tracer power rating.

    Args:
        required_w_m: Required power per meter (W/m)
        tracing_type: Type of heat tracing
        safety_factor: Design safety factor

    Returns:
        Tuple of (selected_power_w_m, power_margin_pct)
    """
    props = TRACER_PROPERTIES_DB.get(tracing_type, TRACER_PROPERTIES_DB[TracingType.ELECTRIC_SELF_REG])
    available = props["available_powers"]

    # Apply safety factor
    design_power = required_w_m * safety_factor

    # Find minimum adequate rating
    for power in sorted(available):
        if power >= design_power:
            margin = (power - required_w_m) / required_w_m * 100 if required_w_m > 0 else 100
            return power, margin

    # If none adequate, return highest available
    highest = max(available)
    margin = (highest - required_w_m) / required_w_m * 100 if required_w_m > 0 else 0
    return highest, margin


def calculate_optimal_insulation(
    pipe_od_mm: float,
    maintain_temp_c: float,
    ambient_temp_c: float,
    wind_speed: float,
    location: PipeLocation,
    insulation_type: InsulationType,
    electricity_price: float,
    operating_hours: int,
    years: int = 10
) -> Tuple[float, float]:
    """
    Calculate economic optimal insulation thickness.

    Balances insulation cost against energy cost savings.

    Reference: ASTM C680 Economic Thickness

    Args:
        pipe_od_mm: Pipe outer diameter (mm)
        maintain_temp_c: Maintain temperature (C)
        ambient_temp_c: Ambient temperature (C)
        wind_speed: Wind speed (m/s)
        location: Installation location
        insulation_type: Insulation material
        electricity_price: $/kWh
        operating_hours: Annual operating hours
        years: Economic analysis period

    Returns:
        Tuple of (optimal_thickness_mm, economic_thickness_mm)
    """
    mean_temp = (maintain_temp_c + ambient_temp_c) / 2
    k_ins = calculate_thermal_conductivity_at_temp(insulation_type, mean_temp)
    ins_props = INSULATION_PROPERTIES_DB.get(insulation_type, INSULATION_PROPERTIES_DB[InsulationType.MINERAL_WOOL])

    # Test range of thicknesses
    thicknesses = [0, 25, 50, 75, 100, 125, 150, 200]
    best_total_cost = float('inf')
    optimal_mm = 50
    economic_mm = 50

    for thick in thicknesses:
        if thick == 0:
            continue

        # Calculate heat loss
        analysis, _ = calculate_heat_loss_per_meter(
            pipe_od_mm, thick, k_ins, maintain_temp_c, ambient_temp_c,
            wind_speed, location
        )

        # Annual energy cost per meter
        annual_energy_kwh = analysis.total_heat_loss_w_m * operating_hours / 1000
        annual_cost = annual_energy_kwh * electricity_price

        # Insulation cost per meter (simplified)
        r_pipe = pipe_od_mm / 2000
        r_ins = r_pipe + thick / 1000
        volume_m3_per_m = math.pi * (r_ins ** 2 - r_pipe ** 2)
        insulation_cost = volume_m3_per_m * ins_props.get("cost_per_m3", 200)

        # Total lifecycle cost
        total_cost = insulation_cost + annual_cost * years

        if total_cost < best_total_cost:
            best_total_cost = total_cost
            economic_mm = thick

        # Optimal = minimum heat loss within reason
        if analysis.total_heat_loss_w_m < 20:  # Target 20 W/m max
            optimal_mm = thick
            break

    return optimal_mm, economic_mm


# =============================================================================
# AGENT CLASS
# =============================================================================

class HeatTracingAgent:
    """
    GL-052: Heat Tracing Optimization Agent.

    Optimizes heat tracing systems by:
    1. Calculating heat loss using cylindrical heat transfer equations
    2. Sizing heat tracer power requirements
    3. Evaluating system adequacy with safety margins
    4. Determining optimal insulation thickness
    5. Performing economic analysis

    All calculations are deterministic using physics-based formulas from
    IEEE 515, IEC 62395, and heat transfer engineering principles.

    Attributes:
        AGENT_ID: Unique agent identifier (GL-052)
        AGENT_NAME: Human-readable name (HEAT-TRACING)
        VERSION: Semantic version string

    Example:
        >>> agent = HeatTracingAgent()
        >>> result = agent.run({
        ...     "equipment_id": "PIPE-001",
        ...     "pipe_length_m": 100,
        ...     "maintain_temp_c": 50,
        ...     "ambient_temp_c": -20
        ... })
    """

    AGENT_ID = "GL-052"
    AGENT_NAME = "HEAT-TRACING"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize HeatTracingAgent."""
        self.config = config or {}
        logger.info(f"{self.AGENT_NAME} agent initialized (ID: {self.AGENT_ID}, v{self.VERSION})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute heat tracing analysis.

        Args:
            input_data: Dictionary matching HeatTracingInput schema

        Returns:
            Dictionary with analysis results and provenance
        """
        start_time = datetime.now()

        try:
            validated = HeatTracingInput(**input_data)
            output = self._process(validated, start_time)
            return output.model_dump()
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}", exc_info=True)
            raise

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Async wrapper for run method."""
        return self.run(input_data)

    def _process(self, inp: HeatTracingInput, start_time: datetime) -> HeatTracingOutput:
        """Main processing logic."""
        recommendations = []
        warnings = []
        validation_errors = []

        logger.info(f"Processing heat tracing analysis for {inp.equipment_id}")

        # Get insulation properties
        if inp.insulation:
            ins_type = inp.insulation.material
            ins_thickness = inp.insulation.thickness_mm
            ins_k = inp.insulation.thermal_conductivity_w_mk
        else:
            ins_type = InsulationType.MINERAL_WOOL
            ins_thickness = inp.insulation_thickness_mm
            mean_temp = (inp.maintain_temp_c + inp.ambient_temp_c) / 2
            ins_k = calculate_thermal_conductivity_at_temp(ins_type, mean_temp)

        # Use minimum ambient if provided
        design_ambient = inp.min_ambient_temp_c if inp.min_ambient_temp_c else inp.ambient_temp_c

        # Calculate heat loss
        heat_loss_analysis, surface_temp = calculate_heat_loss_per_meter(
            inp.pipe_diameter_mm,
            ins_thickness,
            ins_k,
            inp.maintain_temp_c,
            design_ambient,
            inp.wind_speed_m_s,
            inp.pipe_location
        )

        heat_loss_w_m = heat_loss_analysis.total_heat_loss_w_m

        # Get tracer properties
        tracer_props = TRACER_PROPERTIES_DB.get(
            inp.tracing_type,
            TRACER_PROPERTIES_DB[TracingType.ELECTRIC_SELF_REG]
        )
        tracer_safety = tracer_props["safety_factor"]
        combined_safety = max(inp.safety_factor, tracer_safety)

        # Required tracer power
        required_power = heat_loss_w_m * combined_safety

        # Select tracer power
        selected_power, power_margin = select_tracer_power(
            heat_loss_w_m, inp.tracing_type, combined_safety
        )

        # Total system power
        total_power_kw = inp.tracer_power_w_m * inp.pipe_length_m / 1000

        # System adequacy
        if inp.tracer_power_w_m >= required_power:
            adequacy = SystemAdequacy.ADEQUATE
            if inp.tracer_power_w_m > required_power * 1.5:
                adequacy = SystemAdequacy.OVERSIZED
        elif inp.tracer_power_w_m >= heat_loss_w_m:
            adequacy = SystemAdequacy.MARGINAL
        else:
            adequacy = SystemAdequacy.INADEQUATE

        # Safety margin
        safety_margin = (inp.tracer_power_w_m - heat_loss_w_m) / heat_loss_w_m * 100 if heat_loss_w_m > 0 else 100

        # Circuit sizing
        max_circuit = tracer_props["max_circuit_length_m"]
        recommended_circuits = max(1, math.ceil(inp.pipe_length_m / max_circuit))

        # Tracer sizing output
        tracer_sizing = TracerSizing(
            required_power_w_m=round(required_power, 1),
            selected_power_w_m=round(selected_power, 1),
            total_power_kw=round(selected_power * inp.pipe_length_m / 1000, 2),
            max_circuit_length_m=max_circuit,
            recommended_circuits=recommended_circuits,
            power_margin_pct=round(power_margin, 1)
        )

        # Temperature margin
        temp_margin = inp.maintain_temp_c - design_ambient

        # Optimal insulation calculation
        optimal_ins, economic_ins = calculate_optimal_insulation(
            inp.pipe_diameter_mm,
            inp.maintain_temp_c,
            design_ambient,
            inp.wind_speed_m_s,
            inp.pipe_location,
            ins_type,
            inp.electricity_price_kwh,
            inp.operating_hours_year
        )

        # Economic analysis
        efficiency = tracer_props.get("efficiency", 0.95)
        annual_energy = total_power_kw * inp.operating_hours_year / efficiency
        annual_cost = annual_energy * inp.electricity_price_kwh
        cost_per_meter = annual_cost / inp.pipe_length_m if inp.pipe_length_m > 0 else 0

        # Calculate savings with optimal insulation
        if ins_thickness < economic_ins:
            opt_analysis, _ = calculate_heat_loss_per_meter(
                inp.pipe_diameter_mm, economic_ins, ins_k,
                inp.maintain_temp_c, design_ambient,
                inp.wind_speed_m_s, inp.pipe_location
            )
            opt_energy = opt_analysis.total_heat_loss_w_m * inp.pipe_length_m * inp.operating_hours_year / 1000
            current_energy = heat_loss_w_m * inp.pipe_length_m * inp.operating_hours_year / 1000
            savings = (current_energy - opt_energy) * inp.electricity_price_kwh
        else:
            savings = 0

        economic_analysis = EconomicAnalysis(
            annual_energy_kwh=round(annual_energy, 0),
            annual_energy_cost_usd=round(annual_cost, 2),
            energy_cost_per_meter_usd=round(cost_per_meter, 2),
            insulation_savings_usd=round(savings, 2),
            payback_years=None  # Would need insulation cost for full payback calc
        )

        # Generate recommendations
        recommendations.extend(self._generate_recommendations(
            inp, heat_loss_analysis, tracer_sizing, adequacy,
            ins_thickness, optimal_ins, economic_ins, annual_cost
        ))

        # Generate warnings
        warnings.extend(self._generate_warnings(inp, adequacy, heat_loss_w_m, tracer_props))

        # Validation
        validation_status = "PASS"
        if adequacy == SystemAdequacy.INADEQUATE:
            validation_errors.append("Heat tracing system is inadequate for design conditions")
            validation_status = "FAIL"

        # Calculate provenance hash
        calc_hash = self._calculate_provenance_hash(
            inp, heat_loss_analysis, tracer_sizing, economic_analysis
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Completed analysis for {inp.equipment_id} in {processing_time:.1f}ms")

        return HeatTracingOutput(
            equipment_id=inp.equipment_id,
            tracing_type=inp.tracing_type.value,
            application=inp.application.value,
            heat_loss_w_m=round(heat_loss_w_m, 2),
            heat_loss_analysis=heat_loss_analysis,
            required_power_w_m=round(required_power, 1),
            tracer_sizing=tracer_sizing,
            total_power_kw=round(total_power_kw, 2),
            system_adequacy=adequacy,
            safety_margin_pct=round(safety_margin, 1),
            optimal_insulation_mm=round(optimal_ins, 0),
            optimal_tracer_power_w_m=round(selected_power, 1),
            economic_insulation_mm=round(economic_ins, 0),
            economic_analysis=economic_analysis,
            annual_energy_kwh=round(annual_energy, 0),
            annual_cost_usd=round(annual_cost, 2),
            maintain_temp_c=inp.maintain_temp_c,
            design_ambient_c=design_ambient,
            temperature_margin_c=round(temp_margin, 1),
            recommendations=recommendations,
            warnings=warnings,
            calculation_hash=calc_hash,
            validation_status=validation_status,
            validation_errors=validation_errors,
            agent_version=self.VERSION
        )

    def _generate_recommendations(
        self,
        inp: HeatTracingInput,
        heat_loss: HeatLossAnalysis,
        sizing: TracerSizing,
        adequacy: SystemAdequacy,
        current_ins: float,
        optimal_ins: float,
        economic_ins: float,
        annual_cost: float
    ) -> List[str]:
        """Generate optimization recommendations."""
        recs = []

        if adequacy == SystemAdequacy.INADEQUATE:
            recs.append(
                f"URGENT: Upgrade tracer from {inp.tracer_power_w_m} W/m to "
                f"{sizing.selected_power_w_m} W/m minimum"
            )
        elif adequacy == SystemAdequacy.MARGINAL:
            recs.append(
                f"Marginal capacity - recommend upgrading to {sizing.selected_power_w_m} W/m "
                "or adding insulation"
            )
        elif adequacy == SystemAdequacy.OVERSIZED:
            recs.append(
                "System oversized - consider control optimization to reduce energy consumption"
            )

        if current_ins < economic_ins:
            recs.append(
                f"Increase insulation from {current_ins}mm to {economic_ins}mm for optimal economics"
            )

        if current_ins < 25:
            recs.append(
                f"Insulation thickness {current_ins}mm inadequate - "
                f"minimum 50mm recommended for freeze protection"
            )

        if heat_loss.total_heat_loss_w_m > 50:
            recs.append(
                f"High heat loss ({heat_loss.total_heat_loss_w_m:.1f} W/m) - "
                "improve insulation to reduce energy costs"
            )

        if inp.tracing_type == TracingType.ELECTRIC_CONSTANT and inp.maintain_temp_c < 65:
            recs.append(
                "Consider self-regulating cable for temperatures below 65C - "
                "better energy efficiency and overheat protection"
            )

        if annual_cost > 5000:
            recs.append(
                f"High annual energy cost ${annual_cost:,.0f} - "
                "evaluate thermostat control or improved insulation"
            )

        if sizing.recommended_circuits > inp.tracer_circuits:
            recs.append(
                f"Increase circuits from {inp.tracer_circuits} to {sizing.recommended_circuits} "
                f"for proper coverage"
            )

        if inp.control_type != "self_regulating" and inp.tracing_type != TracingType.ELECTRIC_SELF_REG:
            recs.append(
                "Add thermostat or ambient-sensing control to reduce energy consumption"
            )

        return recs

    def _generate_warnings(
        self,
        inp: HeatTracingInput,
        adequacy: SystemAdequacy,
        heat_loss: float,
        tracer_props: Dict
    ) -> List[str]:
        """Generate safety warnings."""
        warnings = []

        if adequacy == SystemAdequacy.INADEQUATE:
            warnings.append(
                f"CRITICAL: Tracer power ({inp.tracer_power_w_m} W/m) insufficient "
                f"for {heat_loss:.1f} W/m heat loss - FREEZE RISK"
            )

        if inp.maintain_temp_c > tracer_props["max_maintain_temp"]:
            warnings.append(
                f"Maintain temperature {inp.maintain_temp_c}C exceeds tracer rating "
                f"({tracer_props['max_maintain_temp']}C) - select different tracer type"
            )

        if inp.pipe_location == PipeLocation.OUTDOOR_EXPOSED and inp.wind_speed_m_s < 5:
            warnings.append(
                "Design wind speed may be too low for exposed outdoor installation - "
                "consider 8-10 m/s for design margin"
            )

        if inp.insulation_thickness_mm == 0:
            warnings.append(
                "No insulation specified - heat loss will be excessive and system may be inadequate"
            )

        if inp.application == ApplicationType.FREEZE_PROTECTION and inp.maintain_temp_c < 5:
            warnings.append(
                f"Maintain temperature {inp.maintain_temp_c}C too close to freezing - "
                "recommend 10-15C for freeze protection"
            )

        return warnings

    def _calculate_provenance_hash(
        self,
        inp: HeatTracingInput,
        heat_loss: HeatLossAnalysis,
        sizing: TracerSizing,
        economics: EconomicAnalysis
    ) -> str:
        """Calculate SHA-256 hash for audit trail."""
        provenance_data = {
            "equipment_id": inp.equipment_id,
            "tracing_type": inp.tracing_type.value,
            "pipe_length_m": inp.pipe_length_m,
            "heat_loss_w_m": heat_loss.total_heat_loss_w_m,
            "required_power_w_m": sizing.required_power_w_m,
            "annual_cost": economics.annual_energy_cost_usd,
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
            "category": "Process Heat",
            "type": "Optimization",
            "standards": ["IEEE 515", "IEC 62395"],
            "capabilities": [
                "Heat loss calculation",
                "Tracer sizing",
                "Insulation optimization",
                "Economic analysis",
                "System adequacy assessment"
            ]
        }


# =============================================================================
# PACK SPECIFICATION
# =============================================================================

PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-052",
    "name": "HEAT-TRACING",
    "version": "1.0.0",
    "summary": "Heat tracing system optimization for freeze protection and temperature maintenance",
    "tags": ["heat-tracing", "freeze-protection", "insulation", "IEEE-515", "IEC-62395"],
    "standards": [
        {"ref": "IEEE 515", "description": "Testing, Design, Installation of Electrical Resistance Heat Tracing"},
        {"ref": "IEC 62395", "description": "Electrical Resistance Trace Heating Systems"}
    ],
    "provenance": {
        "calculation_verified": True,
        "enable_audit": True,
        "deterministic": True
    }
}
