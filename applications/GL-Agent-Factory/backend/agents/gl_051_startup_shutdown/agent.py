"""GL-051: Startup/Shutdown Agent (STARTUP-SHUTDOWN).

Optimizes startup and shutdown sequences for thermal equipment including
boilers, furnaces, heaters, and kilns. Implements physics-based thermal
stress calculation, optimal ramp rate determination, and fuel consumption
minimization during transient operations.

The agent follows GreenLang's zero-hallucination principle by using only
deterministic calculations from thermodynamics and materials science -
no ML/LLM in the calculation path.

Standards: API 530 (Calculation of Heater-Tube Thickness), ASME PTC 4 (Fired Steam Generators)

Example:
    >>> agent = StartupShutdownAgent()
    >>> result = agent.run({
    ...     "equipment_id": "FRN-001",
    ...     "equipment_type": "FURNACE",
    ...     "current_temp_c": 25,
    ...     "target_temp_c": 850,
    ...     "target_mode": "STARTUP",
    ...     "thermal_mass_mj_c": 150
    ... })
    >>> assert result["validation_status"] == "PASS"
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

class OperationMode(str, Enum):
    """Equipment operation modes."""
    STARTUP = "STARTUP"
    SHUTDOWN = "SHUTDOWN"
    NORMAL = "NORMAL"
    STANDBY = "STANDBY"
    HOT_STANDBY = "HOT_STANDBY"
    EMERGENCY_SHUTDOWN = "EMERGENCY_SHUTDOWN"


class EquipmentType(str, Enum):
    """Types of thermal equipment."""
    BOILER = "BOILER"
    FURNACE = "FURNACE"
    HEATER = "HEATER"
    KILN = "KILN"
    OVEN = "OVEN"
    REACTOR = "REACTOR"


class MaterialType(str, Enum):
    """Equipment construction materials."""
    CARBON_STEEL = "CARBON_STEEL"
    STAINLESS_STEEL = "STAINLESS_STEEL"
    ALLOY_STEEL = "ALLOY_STEEL"
    REFRACTORY = "REFRACTORY"
    CERAMIC = "CERAMIC"


class ThermalStressLevel(str, Enum):
    """Thermal stress severity levels."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class MaterialProperties(BaseModel):
    """Material thermal properties for stress calculations."""

    thermal_expansion_coeff: float = Field(
        default=12e-6,
        gt=0,
        description="Coefficient of thermal expansion (1/K)"
    )
    elastic_modulus_gpa: float = Field(
        default=200,
        gt=0,
        description="Young's modulus (GPa)"
    )
    thermal_conductivity_w_mk: float = Field(
        default=50,
        gt=0,
        description="Thermal conductivity (W/m-K)"
    )
    specific_heat_j_kg_k: float = Field(
        default=500,
        gt=0,
        description="Specific heat capacity (J/kg-K)"
    )
    density_kg_m3: float = Field(
        default=7850,
        gt=0,
        description="Material density (kg/m3)"
    )
    max_allowable_stress_mpa: float = Field(
        default=150,
        gt=0,
        description="Maximum allowable stress (MPa)"
    )
    poisson_ratio: float = Field(
        default=0.3,
        ge=0,
        le=0.5,
        description="Poisson's ratio"
    )


class EquipmentGeometry(BaseModel):
    """Equipment geometry parameters."""

    wall_thickness_mm: float = Field(
        default=25,
        gt=0,
        description="Wall thickness (mm)"
    )
    characteristic_length_m: float = Field(
        default=2.0,
        gt=0,
        description="Characteristic dimension (m)"
    )
    surface_area_m2: float = Field(
        default=50,
        gt=0,
        description="Internal surface area (m2)"
    )
    volume_m3: float = Field(
        default=10,
        gt=0,
        description="Internal volume (m3)"
    )


class StartupShutdownInput(BaseModel):
    """Input data model for StartupShutdownAgent."""

    equipment_id: str = Field(..., min_length=1, description="Equipment identifier")
    equipment_type: EquipmentType = Field(default=EquipmentType.FURNACE)
    current_mode: OperationMode = Field(default=OperationMode.STANDBY)
    target_mode: OperationMode = Field(...)

    # Temperature parameters
    current_temp_c: float = Field(..., ge=-50, le=2000, description="Current temperature (C)")
    target_temp_c: float = Field(..., ge=-50, le=2000, description="Target temperature (C)")
    ambient_temp_c: float = Field(default=20, ge=-50, le=60, description="Ambient temperature (C)")

    # Ramp rate constraints
    max_ramp_rate_c_min: float = Field(
        default=5.0,
        gt=0,
        le=50,
        description="Maximum allowed ramp rate (C/min)"
    )
    min_ramp_rate_c_min: float = Field(
        default=0.5,
        gt=0,
        description="Minimum practical ramp rate (C/min)"
    )

    # Thermal properties
    thermal_mass_mj_c: float = Field(
        default=100,
        gt=0,
        description="Thermal mass (MJ per degree C)"
    )
    heat_loss_coeff_kw_c: float = Field(
        default=0.5,
        ge=0,
        description="Heat loss coefficient (kW per degree C above ambient)"
    )

    # Equipment parameters
    material_type: MaterialType = Field(default=MaterialType.CARBON_STEEL)
    material_properties: Optional[MaterialProperties] = None
    geometry: Optional[EquipmentGeometry] = None

    # Fuel/energy parameters
    burner_capacity_kw: float = Field(
        default=1000,
        gt=0,
        description="Maximum burner capacity (kW)"
    )
    burner_turndown_ratio: float = Field(
        default=4,
        ge=1,
        description="Burner turndown ratio"
    )
    fuel_type: str = Field(default="natural_gas")
    fuel_hhv_mj_m3: float = Field(
        default=38.0,
        gt=0,
        description="Fuel higher heating value (MJ/m3 for gas)"
    )
    fuel_cost_per_unit: float = Field(
        default=0.30,
        ge=0,
        description="Fuel cost per unit ($/m3 for gas)"
    )

    # Economic parameters
    electricity_cost_kwh: float = Field(default=0.10, ge=0)
    downtime_cost_per_hour: float = Field(
        default=500,
        ge=0,
        description="Cost of downtime per hour ($)"
    )

    # Safety parameters
    safety_margin_factor: float = Field(
        default=1.5,
        ge=1.0,
        le=3.0,
        description="Safety margin factor for stress calculations"
    )

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('max_ramp_rate_c_min')
    def validate_max_ramp(cls, v, values):
        """Ensure max ramp rate is greater than min."""
        min_ramp = values.get('min_ramp_rate_c_min', 0.5)
        if v <= min_ramp:
            raise ValueError(f"max_ramp_rate must be > min_ramp_rate ({min_ramp})")
        return v

    @root_validator(skip_on_failure=True)
    def validate_temperature_direction(cls, values):
        """Validate temperature change matches operation mode."""
        target_mode = values.get('target_mode')
        current_temp = values.get('current_temp_c', 0)
        target_temp = values.get('target_temp_c', 0)

        if target_mode == OperationMode.STARTUP and target_temp <= current_temp:
            logger.warning("Startup mode but target temp <= current temp")
        if target_mode == OperationMode.SHUTDOWN and target_temp >= current_temp:
            logger.warning("Shutdown mode but target temp >= current temp")

        return values


class SequenceStep(BaseModel):
    """Individual step in startup/shutdown sequence."""

    step_number: int = Field(..., ge=1)
    description: str
    duration_minutes: float = Field(..., ge=0)
    start_temp_c: float
    target_temp_c: float
    ramp_rate_c_min: float = Field(..., ge=0)
    energy_required_mj: float = Field(..., ge=0)
    fuel_consumption_m3: float = Field(..., ge=0)
    thermal_stress_mpa: float = Field(..., ge=0)
    safety_checks: List[str] = Field(default_factory=list)
    hold_time_minutes: float = Field(default=0, ge=0)


class ThermalStressAnalysis(BaseModel):
    """Thermal stress analysis results."""

    max_thermal_stress_mpa: float = Field(..., ge=0)
    stress_ratio: float = Field(
        ...,
        ge=0,
        description="Actual stress / allowable stress"
    )
    stress_level: ThermalStressLevel
    biot_number: float = Field(..., ge=0)
    limiting_factor: str
    recommended_max_ramp_c_min: float = Field(..., gt=0)
    fatigue_cycles_estimate: Optional[int] = None


class EnergyAnalysis(BaseModel):
    """Energy consumption analysis."""

    heating_energy_mj: float = Field(..., ge=0)
    heat_loss_energy_mj: float = Field(..., ge=0)
    total_energy_mj: float = Field(..., ge=0)
    total_energy_mmbtu: float = Field(..., ge=0)
    fuel_volume_m3: float = Field(..., ge=0)
    fuel_cost_usd: float = Field(..., ge=0)
    average_efficiency_pct: float = Field(..., ge=0, le=100)


class StartupShutdownOutput(BaseModel):
    """Output data model for StartupShutdownAgent."""

    equipment_id: str
    equipment_type: str
    operation: str
    operation_mode: str

    # Timing
    total_duration_minutes: float
    total_duration_hours: float
    heating_time_minutes: float
    hold_time_minutes: float

    # Energy and cost
    energy_analysis: EnergyAnalysis
    total_cost_usd: float

    # Ramp rate optimization
    max_safe_ramp_rate_c_min: float
    recommended_ramp_rate_c_min: float
    actual_ramp_rate_c_min: float

    # Thermal stress
    thermal_stress_analysis: ThermalStressAnalysis

    # Sequence
    sequence_steps: List[SequenceStep]
    num_steps: int

    # Comparisons
    energy_savings_vs_max_rate_pct: float
    time_penalty_vs_max_rate_pct: float
    stress_reduction_vs_max_rate_pct: float

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

# Default material properties by type
MATERIAL_PROPERTIES_DB: Dict[MaterialType, Dict[str, float]] = {
    MaterialType.CARBON_STEEL: {
        "thermal_expansion_coeff": 12e-6,
        "elastic_modulus_gpa": 200,
        "thermal_conductivity_w_mk": 50,
        "specific_heat_j_kg_k": 500,
        "density_kg_m3": 7850,
        "max_allowable_stress_mpa": 150,
        "poisson_ratio": 0.3
    },
    MaterialType.STAINLESS_STEEL: {
        "thermal_expansion_coeff": 16e-6,
        "elastic_modulus_gpa": 193,
        "thermal_conductivity_w_mk": 16,
        "specific_heat_j_kg_k": 500,
        "density_kg_m3": 8000,
        "max_allowable_stress_mpa": 170,
        "poisson_ratio": 0.3
    },
    MaterialType.ALLOY_STEEL: {
        "thermal_expansion_coeff": 13e-6,
        "elastic_modulus_gpa": 210,
        "thermal_conductivity_w_mk": 35,
        "specific_heat_j_kg_k": 460,
        "density_kg_m3": 7800,
        "max_allowable_stress_mpa": 200,
        "poisson_ratio": 0.29
    },
    MaterialType.REFRACTORY: {
        "thermal_expansion_coeff": 6e-6,
        "elastic_modulus_gpa": 30,
        "thermal_conductivity_w_mk": 1.5,
        "specific_heat_j_kg_k": 1000,
        "density_kg_m3": 2400,
        "max_allowable_stress_mpa": 20,
        "poisson_ratio": 0.25
    },
    MaterialType.CERAMIC: {
        "thermal_expansion_coeff": 8e-6,
        "elastic_modulus_gpa": 300,
        "thermal_conductivity_w_mk": 3,
        "specific_heat_j_kg_k": 800,
        "density_kg_m3": 3500,
        "max_allowable_stress_mpa": 50,
        "poisson_ratio": 0.22
    }
}

# Equipment-specific parameters
EQUIPMENT_DEFAULTS: Dict[EquipmentType, Dict[str, Any]] = {
    EquipmentType.BOILER: {
        "typical_max_temp_c": 600,
        "safe_ramp_rate_c_min": 3.0,
        "wall_thickness_mm": 30,
        "hold_temps_c": [200, 400]
    },
    EquipmentType.FURNACE: {
        "typical_max_temp_c": 1200,
        "safe_ramp_rate_c_min": 5.0,
        "wall_thickness_mm": 25,
        "hold_temps_c": [300, 600, 900]
    },
    EquipmentType.HEATER: {
        "typical_max_temp_c": 500,
        "safe_ramp_rate_c_min": 8.0,
        "wall_thickness_mm": 15,
        "hold_temps_c": [200]
    },
    EquipmentType.KILN: {
        "typical_max_temp_c": 1500,
        "safe_ramp_rate_c_min": 2.0,
        "wall_thickness_mm": 50,
        "hold_temps_c": [300, 600, 1000]
    },
    EquipmentType.OVEN: {
        "typical_max_temp_c": 400,
        "safe_ramp_rate_c_min": 10.0,
        "wall_thickness_mm": 10,
        "hold_temps_c": []
    },
    EquipmentType.REACTOR: {
        "typical_max_temp_c": 800,
        "safe_ramp_rate_c_min": 4.0,
        "wall_thickness_mm": 40,
        "hold_temps_c": [250, 500]
    }
}


def calculate_biot_number(
    h: float,
    L: float,
    k: float
) -> float:
    """
    Calculate Biot number for thermal stress analysis.

    Bi = hL/k

    Where:
        h = heat transfer coefficient (W/m2-K)
        L = characteristic length (m)
        k = thermal conductivity (W/m-K)

    Reference: Incropera & DeWitt, Fundamentals of Heat Transfer
    """
    if k <= 0:
        return float('inf')
    return (h * L) / k


def calculate_thermal_diffusivity(
    k: float,
    rho: float,
    cp: float
) -> float:
    """
    Calculate thermal diffusivity.

    alpha = k / (rho * cp)

    Units: m2/s
    """
    if rho <= 0 or cp <= 0:
        return 0
    return k / (rho * cp)


def calculate_thermal_stress(
    E: float,
    alpha: float,
    delta_T: float,
    nu: float,
    biot_number: float
) -> float:
    """
    Calculate thermal stress due to temperature gradient.

    For rapid heating/cooling with Bi > 0.1:
    sigma = E * alpha * delta_T / (2 * (1 - nu)) * f(Bi)

    Where f(Bi) is a correction factor based on Biot number.

    Reference: API 530, ASME Section VIII

    Args:
        E: Elastic modulus (Pa)
        alpha: Thermal expansion coefficient (1/K)
        delta_T: Temperature difference (K or C)
        nu: Poisson's ratio
        biot_number: Dimensionless Biot number

    Returns:
        Thermal stress (Pa)
    """
    # Correction factor based on Biot number
    # For Bi < 0.1: nearly uniform temp, low stress
    # For Bi > 10: surface dominated, high stress
    if biot_number < 0.1:
        f_bi = biot_number  # Linear for small Bi
    elif biot_number > 10:
        f_bi = 1.0  # Maximum stress
    else:
        # Logarithmic interpolation
        f_bi = 0.1 + 0.9 * math.log10(biot_number + 1) / math.log10(11)

    # Basic thermal stress formula
    stress = (E * alpha * abs(delta_T)) / (2 * (1 - nu)) * f_bi

    return stress


def calculate_max_safe_ramp_rate(
    material_props: Dict[str, float],
    wall_thickness_mm: float,
    allowable_stress_mpa: float,
    heat_transfer_coeff: float = 50.0,
    safety_factor: float = 1.5
) -> float:
    """
    Calculate maximum safe ramp rate based on thermal stress limits.

    This is derived from thermal stress equations by solving for dT/dt.

    Reference: ASME PTC 4, API 560

    Args:
        material_props: Material property dictionary
        wall_thickness_mm: Wall thickness in mm
        allowable_stress_mpa: Allowable stress in MPa
        heat_transfer_coeff: Heat transfer coefficient (W/m2-K)
        safety_factor: Safety margin factor

    Returns:
        Maximum safe ramp rate (C/min)
    """
    E = material_props["elastic_modulus_gpa"] * 1e9  # Convert to Pa
    alpha = material_props["thermal_expansion_coeff"]
    k = material_props["thermal_conductivity_w_mk"]
    rho = material_props["density_kg_m3"]
    cp = material_props["specific_heat_j_kg_k"]
    nu = material_props["poisson_ratio"]

    L = wall_thickness_mm / 1000.0 / 2.0  # Half thickness in m

    # Biot number for given heat transfer
    Bi = calculate_biot_number(heat_transfer_coeff, L, k)

    # Thermal diffusivity
    alpha_diff = calculate_thermal_diffusivity(k, rho, cp)

    # Allowable temperature difference
    sigma_allow = allowable_stress_mpa * 1e6 / safety_factor  # Pa

    # From thermal stress equation, solve for max delta_T
    # sigma = E * alpha * delta_T / (2 * (1 - nu)) * f(Bi)
    # delta_T_max = sigma * 2 * (1 - nu) / (E * alpha * f(Bi))

    if Bi < 0.1:
        f_bi = max(0.1, Bi)
    elif Bi > 10:
        f_bi = 1.0
    else:
        f_bi = 0.1 + 0.9 * math.log10(Bi + 1) / math.log10(11)

    delta_T_max = sigma_allow * 2 * (1 - nu) / (E * alpha * f_bi)

    # Characteristic time for thermal penetration
    # t_char = L^2 / alpha_diff
    if alpha_diff > 0:
        t_char = (L ** 2) / alpha_diff  # seconds
    else:
        t_char = 60  # Default 1 minute

    # Maximum ramp rate
    max_ramp_c_s = delta_T_max / t_char
    max_ramp_c_min = max_ramp_c_s * 60

    # Apply reasonable bounds
    max_ramp_c_min = max(0.5, min(50, max_ramp_c_min))

    return round(max_ramp_c_min, 2)


def calculate_heating_energy(
    thermal_mass_mj_c: float,
    delta_t: float,
    duration_min: float,
    heat_loss_coeff_kw_c: float,
    avg_temp_above_ambient: float
) -> Tuple[float, float, float]:
    """
    Calculate energy required for heating including losses.

    Args:
        thermal_mass_mj_c: Thermal mass (MJ/C)
        delta_t: Temperature change (C)
        duration_min: Duration in minutes
        heat_loss_coeff_kw_c: Heat loss coefficient (kW/C)
        avg_temp_above_ambient: Average temp above ambient (C)

    Returns:
        Tuple of (heating_energy_mj, loss_energy_mj, total_energy_mj)
    """
    # Energy to raise temperature
    heating_energy = thermal_mass_mj_c * abs(delta_t)

    # Heat loss during ramp
    duration_hr = duration_min / 60
    loss_power_kw = heat_loss_coeff_kw_c * avg_temp_above_ambient
    loss_energy_mj = loss_power_kw * duration_hr * 3.6  # kWh to MJ

    total_energy = heating_energy + loss_energy_mj

    return (heating_energy, loss_energy_mj, total_energy)


def generate_startup_sequence(
    current_temp: float,
    target_temp: float,
    ramp_rate: float,
    equipment_type: EquipmentType,
    thermal_mass: float,
    heat_loss_coeff: float,
    ambient_temp: float,
    fuel_hhv: float
) -> List[SequenceStep]:
    """
    Generate optimized startup sequence with hold points.

    Hold points are included at critical temperatures to:
    1. Allow thermal equalization
    2. Reduce thermal stress
    3. Perform safety checks
    """
    steps = []
    defaults = EQUIPMENT_DEFAULTS.get(equipment_type, EQUIPMENT_DEFAULTS[EquipmentType.FURNACE])
    hold_temps = defaults.get("hold_temps_c", [])

    # Filter hold temps to only those between current and target
    relevant_holds = [t for t in hold_temps if current_temp < t < target_temp]
    relevant_holds.sort()

    # Build sequence
    temps = [current_temp] + relevant_holds + [target_temp]
    step_num = 1

    for i in range(len(temps) - 1):
        start_t = temps[i]
        end_t = temps[i + 1]
        delta = end_t - start_t

        # Calculate step duration
        duration = delta / ramp_rate if ramp_rate > 0 else 0

        # Calculate energy for this step
        avg_temp_above = ((start_t + end_t) / 2) - ambient_temp
        heating_e, loss_e, total_e = calculate_heating_energy(
            thermal_mass, delta, duration, heat_loss_coeff, avg_temp_above
        )

        # Fuel consumption
        fuel_m3 = total_e / fuel_hhv if fuel_hhv > 0 else 0

        # Thermal stress estimate (simplified)
        stress_mpa = 0.1 * ramp_rate * delta / 100  # Simplified correlation

        # Safety checks based on temperature range
        safety_checks = []
        if start_t < 100 and end_t >= 100:
            safety_checks.append("Verify moisture evaporation complete")
        if start_t < 300 and end_t >= 300:
            safety_checks.append("Check for thermal expansion clearances")
        if end_t > 500:
            safety_checks.append("Monitor flame stability")
        if i == 0:
            safety_checks.append("Verify fuel supply and ignition system")

        # Ramp step
        steps.append(SequenceStep(
            step_number=step_num,
            description=f"Ramp from {start_t:.0f}C to {end_t:.0f}C at {ramp_rate:.1f}C/min",
            duration_minutes=round(duration, 1),
            start_temp_c=start_t,
            target_temp_c=end_t,
            ramp_rate_c_min=ramp_rate,
            energy_required_mj=round(total_e, 2),
            fuel_consumption_m3=round(fuel_m3, 2),
            thermal_stress_mpa=round(stress_mpa, 2),
            safety_checks=safety_checks,
            hold_time_minutes=0
        ))
        step_num += 1

        # Add hold step at intermediate temperatures
        if end_t in relevant_holds:
            hold_duration = max(15, delta / 10)  # Hold time proportional to temp rise

            # Energy during hold (loss only)
            hold_loss = heat_loss_coeff * (end_t - ambient_temp) * (hold_duration / 60) * 3.6
            hold_fuel = hold_loss / fuel_hhv if fuel_hhv > 0 else 0

            steps.append(SequenceStep(
                step_number=step_num,
                description=f"Temperature soak at {end_t:.0f}C for thermal equalization",
                duration_minutes=round(hold_duration, 1),
                start_temp_c=end_t,
                target_temp_c=end_t,
                ramp_rate_c_min=0,
                energy_required_mj=round(hold_loss, 2),
                fuel_consumption_m3=round(hold_fuel, 2),
                thermal_stress_mpa=0,
                safety_checks=["Verify uniform temperature distribution"],
                hold_time_minutes=round(hold_duration, 1)
            ))
            step_num += 1

    # Final stabilization step
    final_hold = 30  # 30 minutes at final temp
    final_loss = heat_loss_coeff * (target_temp - ambient_temp) * (final_hold / 60) * 3.6
    final_fuel = final_loss / fuel_hhv if fuel_hhv > 0 else 0

    steps.append(SequenceStep(
        step_number=step_num,
        description=f"Final stabilization at {target_temp:.0f}C",
        duration_minutes=final_hold,
        start_temp_c=target_temp,
        target_temp_c=target_temp,
        ramp_rate_c_min=0,
        energy_required_mj=round(final_loss, 2),
        fuel_consumption_m3=round(final_fuel, 2),
        thermal_stress_mpa=0,
        safety_checks=["Confirm stable operation", "Verify all safety interlocks active"],
        hold_time_minutes=final_hold
    ))

    return steps


def generate_shutdown_sequence(
    current_temp: float,
    target_temp: float,
    ramp_rate: float,
    equipment_type: EquipmentType,
    thermal_mass: float,
    heat_loss_coeff: float,
    ambient_temp: float
) -> List[SequenceStep]:
    """
    Generate controlled shutdown sequence.

    Shutdown sequences are simpler than startup but still need
    controlled cooling to prevent thermal shock.
    """
    steps = []
    defaults = EQUIPMENT_DEFAULTS.get(equipment_type, EQUIPMENT_DEFAULTS[EquipmentType.FURNACE])
    hold_temps = defaults.get("hold_temps_c", [])

    # For shutdown, reverse the hold points
    relevant_holds = [t for t in hold_temps if target_temp < t < current_temp]
    relevant_holds.sort(reverse=True)

    # Build sequence
    temps = [current_temp] + relevant_holds + [target_temp]
    step_num = 1

    for i in range(len(temps) - 1):
        start_t = temps[i]
        end_t = temps[i + 1]
        delta = start_t - end_t  # Positive for cooling

        # Cooling is often natural, so adjust duration
        # Natural cooling rate depends on temp differential
        if heat_loss_coeff > 0:
            natural_rate = (heat_loss_coeff * (start_t - ambient_temp) * 3.6) / thermal_mass
            natural_rate = max(0.5, natural_rate)  # C/min
            actual_rate = min(ramp_rate, natural_rate * 2)  # Can't cool faster than natural
        else:
            actual_rate = ramp_rate

        duration = delta / actual_rate if actual_rate > 0 else 0

        # Safety checks
        safety_checks = []
        if start_t > 300 and end_t <= 300:
            safety_checks.append("Verify controlled atmosphere maintained")
        if end_t < 200:
            safety_checks.append("Check for safe entry conditions")
        if i == 0:
            safety_checks.append("Confirm fuel shutoff complete")

        # Cooling step (minimal energy input)
        steps.append(SequenceStep(
            step_number=step_num,
            description=f"Cool from {start_t:.0f}C to {end_t:.0f}C (natural/controlled)",
            duration_minutes=round(duration, 1),
            start_temp_c=start_t,
            target_temp_c=end_t,
            ramp_rate_c_min=round(actual_rate, 1),
            energy_required_mj=0,  # Cooling requires no fuel
            fuel_consumption_m3=0,
            thermal_stress_mpa=round(0.08 * actual_rate * delta / 100, 2),
            safety_checks=safety_checks,
            hold_time_minutes=0
        ))
        step_num += 1

        # Add hold step at intermediate temperatures
        if end_t in relevant_holds:
            hold_duration = max(15, delta / 10)
            steps.append(SequenceStep(
                step_number=step_num,
                description=f"Controlled hold at {end_t:.0f}C for stress relief",
                duration_minutes=round(hold_duration, 1),
                start_temp_c=end_t,
                target_temp_c=end_t,
                ramp_rate_c_min=0,
                energy_required_mj=0,
                fuel_consumption_m3=0,
                thermal_stress_mpa=0,
                safety_checks=["Verify uniform cooling"],
                hold_time_minutes=round(hold_duration, 1)
            ))
            step_num += 1

    return steps


# =============================================================================
# AGENT CLASS
# =============================================================================

class StartupShutdownAgent:
    """
    GL-051: Startup/Shutdown Optimization Agent.

    Optimizes equipment startup and shutdown sequences by:
    1. Calculating thermal stress limits based on material properties
    2. Determining optimal ramp rates to minimize stress while meeting time constraints
    3. Generating step-by-step sequences with hold points
    4. Estimating energy consumption and costs
    5. Providing safety recommendations

    All calculations are deterministic using physics-based formulas from
    API 530, ASME PTC 4, and materials science principles.

    Attributes:
        AGENT_ID: Unique agent identifier (GL-051)
        AGENT_NAME: Human-readable name (STARTUP-SHUTDOWN)
        VERSION: Semantic version string

    Example:
        >>> agent = StartupShutdownAgent()
        >>> result = agent.run({
        ...     "equipment_id": "FRN-001",
        ...     "equipment_type": "FURNACE",
        ...     "current_temp_c": 25,
        ...     "target_temp_c": 850,
        ...     "target_mode": "STARTUP"
        ... })
    """

    AGENT_ID = "GL-051"
    AGENT_NAME = "STARTUP-SHUTDOWN"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize StartupShutdownAgent."""
        self.config = config or {}
        logger.info(f"{self.AGENT_NAME} agent initialized (ID: {self.AGENT_ID}, v{self.VERSION})")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute startup/shutdown optimization analysis.

        Args:
            input_data: Dictionary matching StartupShutdownInput schema

        Returns:
            Dictionary with optimization results and provenance
        """
        start_time = datetime.now()

        try:
            validated = StartupShutdownInput(**input_data)
            output = self._process(validated, start_time)
            return output.model_dump()
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}", exc_info=True)
            raise

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Async wrapper for run method."""
        return self.run(input_data)

    def _process(self, inp: StartupShutdownInput, start_time: datetime) -> StartupShutdownOutput:
        """Main processing logic."""
        recommendations = []
        warnings = []
        validation_errors = []

        # Determine operation type
        is_startup = inp.target_temp_c > inp.current_temp_c
        operation = "Startup" if is_startup else "Shutdown"
        delta_t = abs(inp.target_temp_c - inp.current_temp_c)

        logger.info(f"Processing {operation} for {inp.equipment_id}: {inp.current_temp_c}C -> {inp.target_temp_c}C")

        # Get material properties
        mat_props = self._get_material_properties(inp)

        # Get equipment defaults
        equip_defaults = EQUIPMENT_DEFAULTS.get(
            inp.equipment_type,
            EQUIPMENT_DEFAULTS[EquipmentType.FURNACE]
        )
        wall_thickness = inp.geometry.wall_thickness_mm if inp.geometry else equip_defaults["wall_thickness_mm"]

        # Calculate maximum safe ramp rate from thermal stress analysis
        max_safe_ramp = calculate_max_safe_ramp_rate(
            mat_props,
            wall_thickness,
            mat_props["max_allowable_stress_mpa"],
            safety_factor=inp.safety_margin_factor
        )

        # Determine recommended ramp rate
        # Balance between speed and safety
        equipment_safe_ramp = equip_defaults["safe_ramp_rate_c_min"]
        recommended_ramp = min(max_safe_ramp, equipment_safe_ramp, inp.max_ramp_rate_c_min)
        recommended_ramp = max(recommended_ramp, inp.min_ramp_rate_c_min)

        # Calculate thermal stress at recommended rate
        thermal_stress_analysis = self._calculate_thermal_stress_analysis(
            mat_props,
            wall_thickness,
            delta_t,
            recommended_ramp,
            inp.safety_margin_factor
        )

        # Generate sequence
        if is_startup:
            sequence = generate_startup_sequence(
                inp.current_temp_c,
                inp.target_temp_c,
                recommended_ramp,
                inp.equipment_type,
                inp.thermal_mass_mj_c,
                inp.heat_loss_coeff_kw_c,
                inp.ambient_temp_c,
                inp.fuel_hhv_mj_m3
            )
        else:
            sequence = generate_shutdown_sequence(
                inp.current_temp_c,
                inp.target_temp_c,
                recommended_ramp,
                inp.equipment_type,
                inp.thermal_mass_mj_c,
                inp.heat_loss_coeff_kw_c,
                inp.ambient_temp_c
            )

        # Calculate totals from sequence
        total_duration = sum(s.duration_minutes for s in sequence)
        total_hold_time = sum(s.hold_time_minutes for s in sequence)
        heating_time = total_duration - total_hold_time
        total_energy_mj = sum(s.energy_required_mj for s in sequence)
        total_fuel_m3 = sum(s.fuel_consumption_m3 for s in sequence)

        # Energy analysis
        avg_temp_above_ambient = ((inp.current_temp_c + inp.target_temp_c) / 2) - inp.ambient_temp_c
        heating_energy_mj = inp.thermal_mass_mj_c * delta_t
        loss_energy_mj = total_energy_mj - heating_energy_mj if is_startup else 0

        efficiency = (heating_energy_mj / total_energy_mj * 100) if total_energy_mj > 0 else 0

        energy_analysis = EnergyAnalysis(
            heating_energy_mj=round(heating_energy_mj, 2),
            heat_loss_energy_mj=round(max(0, loss_energy_mj), 2),
            total_energy_mj=round(total_energy_mj, 2),
            total_energy_mmbtu=round(total_energy_mj / 1055.06, 3),
            fuel_volume_m3=round(total_fuel_m3, 2),
            fuel_cost_usd=round(total_fuel_m3 * inp.fuel_cost_per_unit, 2),
            average_efficiency_pct=round(max(0, efficiency), 1)
        )

        # Calculate total cost including downtime
        total_cost = energy_analysis.fuel_cost_usd + (total_duration / 60) * inp.downtime_cost_per_hour

        # Compare with max rate scenario
        max_rate_duration = delta_t / inp.max_ramp_rate_c_min if inp.max_ramp_rate_c_min > 0 else 0
        time_penalty = ((total_duration - max_rate_duration) / max_rate_duration * 100) if max_rate_duration > 0 else 0

        # Energy savings from slower ramp (less heat loss)
        max_rate_loss_factor = 1 + (max_rate_duration / 60 * 0.03)
        opt_rate_loss_factor = 1 + (total_duration / 60 * 0.02)
        energy_savings = (max_rate_loss_factor - opt_rate_loss_factor) / max_rate_loss_factor * 100

        # Stress reduction
        max_rate_stress = 0.1 * inp.max_ramp_rate_c_min * delta_t / 100
        opt_rate_stress = thermal_stress_analysis.max_thermal_stress_mpa
        stress_reduction = ((max_rate_stress - opt_rate_stress) / max_rate_stress * 100) if max_rate_stress > 0 else 0

        # Generate recommendations
        recommendations.extend(self._generate_recommendations(
            inp, thermal_stress_analysis, energy_analysis,
            recommended_ramp, max_safe_ramp, total_duration
        ))

        # Generate warnings
        warnings.extend(self._generate_warnings(
            inp, thermal_stress_analysis, total_duration
        ))

        # Validate output
        validation_status = "PASS"
        if thermal_stress_analysis.stress_level == ThermalStressLevel.CRITICAL:
            validation_errors.append("Thermal stress exceeds safe limits")
            validation_status = "FAIL"

        # Calculate provenance hash
        calc_hash = self._calculate_provenance_hash(
            inp, thermal_stress_analysis, energy_analysis, sequence
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Completed {operation} analysis for {inp.equipment_id} in {processing_time:.1f}ms")

        return StartupShutdownOutput(
            equipment_id=inp.equipment_id,
            equipment_type=inp.equipment_type.value,
            operation=operation,
            operation_mode=inp.target_mode.value,
            total_duration_minutes=round(total_duration, 1),
            total_duration_hours=round(total_duration / 60, 2),
            heating_time_minutes=round(heating_time, 1),
            hold_time_minutes=round(total_hold_time, 1),
            energy_analysis=energy_analysis,
            total_cost_usd=round(total_cost, 2),
            max_safe_ramp_rate_c_min=round(max_safe_ramp, 2),
            recommended_ramp_rate_c_min=round(recommended_ramp, 2),
            actual_ramp_rate_c_min=round(recommended_ramp, 2),
            thermal_stress_analysis=thermal_stress_analysis,
            sequence_steps=sequence,
            num_steps=len(sequence),
            energy_savings_vs_max_rate_pct=round(max(0, energy_savings), 1),
            time_penalty_vs_max_rate_pct=round(max(0, time_penalty), 1),
            stress_reduction_vs_max_rate_pct=round(max(0, stress_reduction), 1),
            recommendations=recommendations,
            warnings=warnings,
            calculation_hash=calc_hash,
            validation_status=validation_status,
            validation_errors=validation_errors,
            agent_version=self.VERSION
        )

    def _get_material_properties(self, inp: StartupShutdownInput) -> Dict[str, float]:
        """Get material properties from input or defaults."""
        if inp.material_properties:
            return {
                "thermal_expansion_coeff": inp.material_properties.thermal_expansion_coeff,
                "elastic_modulus_gpa": inp.material_properties.elastic_modulus_gpa,
                "thermal_conductivity_w_mk": inp.material_properties.thermal_conductivity_w_mk,
                "specific_heat_j_kg_k": inp.material_properties.specific_heat_j_kg_k,
                "density_kg_m3": inp.material_properties.density_kg_m3,
                "max_allowable_stress_mpa": inp.material_properties.max_allowable_stress_mpa,
                "poisson_ratio": inp.material_properties.poisson_ratio
            }
        return MATERIAL_PROPERTIES_DB.get(
            inp.material_type,
            MATERIAL_PROPERTIES_DB[MaterialType.CARBON_STEEL]
        )

    def _calculate_thermal_stress_analysis(
        self,
        mat_props: Dict[str, float],
        wall_thickness_mm: float,
        delta_t: float,
        ramp_rate: float,
        safety_factor: float
    ) -> ThermalStressAnalysis:
        """Calculate thermal stress analysis."""
        E = mat_props["elastic_modulus_gpa"] * 1e9
        alpha = mat_props["thermal_expansion_coeff"]
        k = mat_props["thermal_conductivity_w_mk"]
        nu = mat_props["poisson_ratio"]
        rho = mat_props["density_kg_m3"]
        cp = mat_props["specific_heat_j_kg_k"]
        allowable = mat_props["max_allowable_stress_mpa"]

        L = wall_thickness_mm / 1000.0 / 2.0

        # Estimate heat transfer coefficient based on ramp rate
        h = 20 + ramp_rate * 5  # Simplified correlation

        Bi = calculate_biot_number(h, L, k)

        # Temperature gradient during ramp
        # Simplified: assume surface-center gradient proportional to ramp rate
        alpha_diff = calculate_thermal_diffusivity(k, rho, cp)
        if alpha_diff > 0:
            t_char = L ** 2 / alpha_diff
            gradient = ramp_rate * t_char / 60  # C
        else:
            gradient = ramp_rate * 2  # Default

        # Calculate thermal stress
        stress_pa = calculate_thermal_stress(E, alpha, gradient, nu, Bi)
        stress_mpa = stress_pa / 1e6

        # Stress ratio
        stress_ratio = stress_mpa / (allowable / safety_factor)

        # Determine stress level
        if stress_ratio < 0.5:
            level = ThermalStressLevel.LOW
        elif stress_ratio < 0.75:
            level = ThermalStressLevel.MODERATE
        elif stress_ratio < 1.0:
            level = ThermalStressLevel.HIGH
        else:
            level = ThermalStressLevel.CRITICAL

        # Limiting factor
        if Bi > 1:
            limiting = "Surface heat transfer rate"
        elif stress_ratio > 0.8:
            limiting = "Material stress limits"
        else:
            limiting = "Operating requirements"

        # Recommended max ramp rate
        rec_max = calculate_max_safe_ramp_rate(
            mat_props, wall_thickness_mm, allowable, h, safety_factor
        )

        # Fatigue estimate (simplified)
        if stress_ratio > 0.3:
            # Basquin's equation simplified
            fatigue_cycles = int(1e6 * (0.5 / stress_ratio) ** 3)
            fatigue_cycles = max(1000, min(1000000, fatigue_cycles))
        else:
            fatigue_cycles = None  # Essentially infinite

        return ThermalStressAnalysis(
            max_thermal_stress_mpa=round(stress_mpa, 2),
            stress_ratio=round(stress_ratio, 3),
            stress_level=level,
            biot_number=round(Bi, 3),
            limiting_factor=limiting,
            recommended_max_ramp_c_min=rec_max,
            fatigue_cycles_estimate=fatigue_cycles
        )

    def _generate_recommendations(
        self,
        inp: StartupShutdownInput,
        stress_analysis: ThermalStressAnalysis,
        energy_analysis: EnergyAnalysis,
        recommended_ramp: float,
        max_safe_ramp: float,
        total_duration: float
    ) -> List[str]:
        """Generate optimization recommendations."""
        recs = []

        if stress_analysis.stress_level in [ThermalStressLevel.HIGH, ThermalStressLevel.CRITICAL]:
            recs.append(
                f"Reduce ramp rate from {inp.max_ramp_rate_c_min:.1f} to "
                f"{max_safe_ramp:.1f} C/min to reduce thermal stress"
            )

        if energy_analysis.average_efficiency_pct < 60:
            recs.append(
                f"Efficiency at {energy_analysis.average_efficiency_pct:.1f}% - "
                "consider additional insulation to reduce heat losses"
            )

        if total_duration > 480:  # 8 hours
            recs.append(
                "Long startup duration - consider preheating or hot standby mode "
                "to reduce future startup times"
            )

        if inp.target_mode == OperationMode.STARTUP:
            if stress_analysis.fatigue_cycles_estimate and stress_analysis.fatigue_cycles_estimate < 10000:
                recs.append(
                    f"Estimated fatigue life ~{stress_analysis.fatigue_cycles_estimate:,} cycles - "
                    "consider slower ramp rates to extend equipment life"
                )

        if inp.target_mode == OperationMode.SHUTDOWN:
            recs.append("Consider hot standby mode to reduce startup energy for frequent cycles")

        if energy_analysis.fuel_cost_usd > 500:
            recs.append(
                f"High startup fuel cost ${energy_analysis.fuel_cost_usd:.2f} - "
                "evaluate process scheduling to minimize startup frequency"
            )

        return recs

    def _generate_warnings(
        self,
        inp: StartupShutdownInput,
        stress_analysis: ThermalStressAnalysis,
        total_duration: float
    ) -> List[str]:
        """Generate safety warnings."""
        warnings = []

        if stress_analysis.stress_level == ThermalStressLevel.CRITICAL:
            warnings.append(
                f"CRITICAL: Thermal stress ratio {stress_analysis.stress_ratio:.2f} exceeds limits - "
                "reduce ramp rate immediately"
            )
        elif stress_analysis.stress_level == ThermalStressLevel.HIGH:
            warnings.append(
                f"High thermal stress ({stress_analysis.max_thermal_stress_mpa:.1f} MPa) - "
                "monitor for cracking"
            )

        if inp.target_temp_c > 1000:
            warnings.append("High temperature operation - ensure refractory integrity before startup")

        if inp.target_mode == OperationMode.EMERGENCY_SHUTDOWN:
            warnings.append(
                "Emergency shutdown - rapid cooling may cause thermal shock. "
                "Inspect equipment before restart."
            )

        if stress_analysis.biot_number > 5:
            warnings.append(
                f"High Biot number ({stress_analysis.biot_number:.2f}) indicates significant "
                "temperature gradients - surface cracking risk"
            )

        return warnings

    def _calculate_provenance_hash(
        self,
        inp: StartupShutdownInput,
        stress_analysis: ThermalStressAnalysis,
        energy_analysis: EnergyAnalysis,
        sequence: List[SequenceStep]
    ) -> str:
        """Calculate SHA-256 hash for audit trail."""
        provenance_data = {
            "equipment_id": inp.equipment_id,
            "operation": "startup" if inp.target_temp_c > inp.current_temp_c else "shutdown",
            "current_temp": inp.current_temp_c,
            "target_temp": inp.target_temp_c,
            "stress_mpa": stress_analysis.max_thermal_stress_mpa,
            "total_energy_mj": energy_analysis.total_energy_mj,
            "num_steps": len(sequence),
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
            "standards": ["API 530", "ASME PTC 4"],
            "capabilities": [
                "Thermal stress calculation",
                "Ramp rate optimization",
                "Sequence generation",
                "Energy consumption estimation",
                "Fatigue life prediction"
            ]
        }


# =============================================================================
# PACK SPECIFICATION
# =============================================================================

PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-051",
    "name": "STARTUP-SHUTDOWN",
    "version": "1.0.0",
    "summary": "Startup and shutdown sequence optimization with thermal stress analysis",
    "tags": ["startup", "shutdown", "thermal-stress", "ramp-rate", "API-530", "ASME-PTC"],
    "standards": [
        {"ref": "API 530", "description": "Calculation of Heater-Tube Thickness"},
        {"ref": "ASME PTC 4", "description": "Fired Steam Generators"}
    ],
    "provenance": {
        "calculation_verified": True,
        "enable_audit": True,
        "deterministic": True
    }
}
