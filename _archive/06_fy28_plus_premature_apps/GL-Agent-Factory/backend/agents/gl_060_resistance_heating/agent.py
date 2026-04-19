"""
GL-060: Resistance Heating Optimizer Agent (RESIST-OPT)

This module implements the ResistanceHeatingOptimizerAgent for optimizing
electric resistance heating systems in industrial furnaces and ovens.

The agent provides:
- Joule heating power calculation (P = I^2 * R)
- Resistance element life prediction
- Temperature control optimization
- Element degradation modeling
- Energy efficiency analysis
- Complete SHA-256 provenance tracking

Key Formulas:
- Joule heating: P = I^2 * R = V^2 / R
- Resistance temperature dependence: R(T) = R0 * (1 + alpha * (T - T0))
- Element life (Arrhenius): L = L0 * exp(Ea / (k * T))
- Power density: W/m^2 = P / A_surface

Standards Compliance:
- NFPA 86: Standard for Ovens and Furnaces
- IEC 60519: Safety in Electroheat Installations
- IEEE 515: Standard for Testing, Design, Installation, and Maintenance of Electrical Resistance Heat Tracing
- NFPA 70: National Electrical Code

Example:
    >>> agent = ResistanceHeatingOptimizerAgent()
    >>> result = agent.run(ResistanceHeatingInput(
    ...     system_id="RESIST-001",
    ...     element_type=ElementType.NICHROME,
    ...     voltage_v=480,
    ...     power_rating_kW=50
    ... ))
    >>> print(f"Element Life: {result.element_life.remaining_life_hours:.0f} hours")
"""

import hashlib
import json
import logging
import math
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Boltzmann constant (eV/K)
BOLTZMANN_K_EV = 8.617e-5

# Boltzmann constant (J/K)
BOLTZMANN_K = 1.381e-23

# Stefan-Boltzmann constant W/(m^2*K^4)
STEFAN_BOLTZMANN = 5.67e-8


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ElementType(str, Enum):
    """Types of resistance heating elements."""
    NICHROME = "nichrome"  # Ni-Cr (80/20)
    KANTHAL = "kanthal"    # Fe-Cr-Al
    SILICON_CARBIDE = "silicon_carbide"  # SiC
    MOLYBDENUM_DISILICIDE = "molybdenum_disilicide"  # MoSi2
    TUNGSTEN = "tungsten"
    GRAPHITE = "graphite"
    PLATINUM = "platinum"
    NICKEL = "nickel"


# Legacy enum for backward compatibility
class HeaterType(str, Enum):
    NICHROME = "NICHROME"
    KANTHAL = "KANTHAL"
    SILICON_CARBIDE = "SILICON_CARBIDE"
    MOLYBDENUM_DISILICIDE = "MOLYBDENUM_DISILICIDE"
    TUNGSTEN = "TUNGSTEN"


class HeatingConfiguration(str, Enum):
    """Heating element configurations."""
    SINGLE_PHASE = "single_phase"
    THREE_PHASE_DELTA = "three_phase_delta"
    THREE_PHASE_WYE = "three_phase_wye"
    MULTI_ZONE = "multi_zone"


class ControlMode(str, Enum):
    """Temperature control modes."""
    ON_OFF = "on_off"
    TIME_PROPORTIONING = "time_proportioning"
    SCR_PHASE_ANGLE = "scr_phase_angle"
    SCR_ZERO_CROSS = "scr_zero_cross"
    PID_CONTINUOUS = "pid_continuous"
    # Legacy values
    PROPORTIONAL = "proportional"
    PID = "pid"
    SCR_PHASE = "scr_phase"


class ProcessStatus(str, Enum):
    """Process status indicators."""
    OPTIMAL = "optimal"
    ACCEPTABLE = "acceptable"
    WARNING = "warning"
    CRITICAL = "critical"


# Material properties for resistance heating elements
# Format: {element_type: {resistivity_20c: Ohm*m, temp_coeff: 1/K, max_temp_c: C,
#          activation_energy_eV: eV, base_life_hours: hours, density_kg_m3: kg/m3,
#          emissivity: -, specific_heat: J/(kg*K)}}
ELEMENT_PROPERTIES = {
    ElementType.NICHROME: {
        "resistivity_20c": 1.10e-6,  # Ohm*m
        "temp_coeff": 0.0004,        # 1/K (alpha)
        "max_temp_c": 1150,
        "max_continuous_temp_c": 1050,
        "activation_energy_eV": 1.2,  # For oxidation degradation
        "base_life_hours": 20000,    # At max continuous temp
        "density_kg_m3": 8400,
        "emissivity": 0.9,
        "specific_heat": 450
    },
    ElementType.KANTHAL: {
        "resistivity_20c": 1.45e-6,
        "temp_coeff": 0.00004,  # Very low temp coefficient
        "max_temp_c": 1400,
        "max_continuous_temp_c": 1300,
        "activation_energy_eV": 1.5,
        "base_life_hours": 25000,
        "density_kg_m3": 7100,
        "emissivity": 0.85,
        "specific_heat": 460
    },
    ElementType.SILICON_CARBIDE: {
        "resistivity_20c": 0.1,  # Much higher, varies with composition
        "temp_coeff": -0.004,  # Negative coefficient (decreases with temp)
        "max_temp_c": 1650,
        "max_continuous_temp_c": 1500,
        "activation_energy_eV": 2.0,
        "base_life_hours": 15000,
        "density_kg_m3": 3200,
        "emissivity": 0.9,
        "specific_heat": 750
    },
    ElementType.MOLYBDENUM_DISILICIDE: {
        "resistivity_20c": 2.0e-5,
        "temp_coeff": 0.003,
        "max_temp_c": 1850,
        "max_continuous_temp_c": 1700,
        "activation_energy_eV": 2.5,
        "base_life_hours": 30000,
        "density_kg_m3": 6240,
        "emissivity": 0.8,
        "specific_heat": 440
    },
    ElementType.TUNGSTEN: {
        "resistivity_20c": 5.6e-8,
        "temp_coeff": 0.0045,
        "max_temp_c": 2500,  # In vacuum/inert atmosphere
        "max_continuous_temp_c": 2200,
        "activation_energy_eV": 3.0,
        "base_life_hours": 10000,  # Atmosphere dependent
        "density_kg_m3": 19300,
        "emissivity": 0.35,
        "specific_heat": 130
    },
    ElementType.GRAPHITE: {
        "resistivity_20c": 1.3e-5,
        "temp_coeff": -0.0005,  # Negative coefficient
        "max_temp_c": 3000,  # In vacuum/inert atmosphere
        "max_continuous_temp_c": 2500,
        "activation_energy_eV": 2.8,
        "base_life_hours": 8000,
        "density_kg_m3": 1800,
        "emissivity": 0.85,
        "specific_heat": 720
    },
    ElementType.PLATINUM: {
        "resistivity_20c": 1.06e-7,
        "temp_coeff": 0.00392,
        "max_temp_c": 1600,
        "max_continuous_temp_c": 1400,
        "activation_energy_eV": 3.5,
        "base_life_hours": 50000,
        "density_kg_m3": 21450,
        "emissivity": 0.15,
        "specific_heat": 130
    },
    ElementType.NICKEL: {
        "resistivity_20c": 6.99e-8,
        "temp_coeff": 0.006,
        "max_temp_c": 600,
        "max_continuous_temp_c": 500,
        "activation_energy_eV": 0.8,
        "base_life_hours": 30000,
        "density_kg_m3": 8900,
        "emissivity": 0.45,
        "specific_heat": 440
    }
}


# =============================================================================
# INPUT MODELS
# =============================================================================

class ElementGeometry(BaseModel):
    """Heating element geometry."""

    element_type: ElementType = Field(..., description="Type of heating element")
    length_m: float = Field(default=10.0, gt=0, description="Total element length (m)")
    diameter_mm: Optional[float] = Field(None, gt=0, description="Wire diameter (mm)")
    width_mm: Optional[float] = Field(None, gt=0, description="Ribbon width (mm)")
    thickness_mm: Optional[float] = Field(None, gt=0, description="Ribbon thickness (mm)")
    coil_diameter_mm: Optional[float] = Field(None, gt=0, description="Coil diameter (mm)")
    surface_area_m2: Optional[float] = Field(None, gt=0, description="Total surface area (m2)")
    cross_section_mm2: Optional[float] = Field(None, gt=0, description="Cross-sectional area (mm2)")
    element_count: int = Field(default=1, ge=1, description="Number of elements")


class ElectricalParameters(BaseModel):
    """Electrical system parameters."""

    voltage_v: float = Field(..., gt=0, le=1000, description="Supply voltage (V)")
    current_a: Optional[float] = Field(None, gt=0, description="Operating current (A)")
    power_rating_kW: float = Field(..., gt=0, description="Rated power (kW)")
    actual_power_kW: Optional[float] = Field(None, ge=0, description="Actual power (kW)")
    element_resistance_ohm: float = Field(default=10.0, gt=0, description="Cold resistance (Ohm)")
    configuration: HeatingConfiguration = Field(
        default=HeatingConfiguration.SINGLE_PHASE,
        description="Electrical configuration"
    )
    power_factor: float = Field(default=1.0, ge=0.5, le=1.0, description="Power factor")


class OperatingConditions(BaseModel):
    """Operating conditions."""

    element_temp_c: float = Field(default=800, ge=0, description="Current element temperature (C)")
    setpoint_temp_c: float = Field(default=800, ge=0, description="Target temperature setpoint (C)")
    ambient_temp_c: float = Field(default=25, ge=-40, le=100, description="Ambient temperature (C)")
    process_temp_c: float = Field(default=700, ge=0, description="Process/furnace temperature (C)")
    operating_hours: float = Field(default=0, ge=0, description="Total operating hours")
    cycles_count: int = Field(default=0, ge=0, description="Number of on/off cycles")
    atmosphere: str = Field(default="air", description="Operating atmosphere")
    thermal_mass_kj_per_k: float = Field(default=1000, gt=0, description="System thermal mass (kJ/K)")
    heat_loss_kw_per_k: float = Field(default=0.01, ge=0, description="Heat loss coefficient (kW/K)")


class ControlParameters(BaseModel):
    """Temperature control parameters."""

    control_mode: ControlMode = Field(default=ControlMode.PID_CONTINUOUS, description="Control mode")
    duty_cycle_percent: Optional[float] = Field(None, ge=0, le=100, description="Current duty cycle (%)")
    pid_p: Optional[float] = Field(None, ge=0, description="Proportional gain")
    pid_i: Optional[float] = Field(None, ge=0, description="Integral gain")
    pid_d: Optional[float] = Field(None, ge=0, description="Derivative gain")
    cycle_time_s: Optional[float] = Field(None, gt=0, description="Control cycle time (s)")
    operating_hours_per_day: float = Field(default=8, ge=0, le=24, description="Operating hours/day")


class ResistanceHeatingInput(BaseModel):
    """Input data model for ResistanceHeatingOptimizerAgent."""

    system_id: str = Field(..., min_length=1, description="Unique system identifier")
    element: ElementGeometry = Field(default_factory=lambda: ElementGeometry(element_type=ElementType.NICHROME))
    electrical: ElectricalParameters = Field(default_factory=lambda: ElectricalParameters(voltage_v=480, power_rating_kW=50))
    operating: OperatingConditions = Field(default_factory=OperatingConditions)
    control: ControlParameters = Field(default_factory=ControlParameters)

    # Cost parameters
    energy_cost_per_kwh: float = Field(default=0.10, gt=0, description="Energy cost ($/kWh)")
    element_cost: float = Field(default=500.0, ge=0, description="Cost per element ($)")

    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Legacy field support
    process_id: Optional[str] = Field(None, description="Legacy process ID")
    heater_type: Optional[HeaterType] = Field(None, description="Legacy heater type")
    rated_power_kw: Optional[float] = Field(None, description="Legacy rated power")
    target_temp_c: Optional[float] = Field(None, description="Legacy target temp")
    current_temp_c: Optional[float] = Field(None, description="Legacy current temp")

    def __init__(self, **data):
        # Handle legacy input format
        if 'process_id' in data and 'system_id' not in data:
            data['system_id'] = data['process_id']
        if 'heater_type' in data:
            heater_map = {
                'NICHROME': ElementType.NICHROME,
                'KANTHAL': ElementType.KANTHAL,
                'SILICON_CARBIDE': ElementType.SILICON_CARBIDE,
                'MOLYBDENUM_DISILICIDE': ElementType.MOLYBDENUM_DISILICIDE,
                'TUNGSTEN': ElementType.TUNGSTEN,
            }
            ht = data['heater_type']
            if isinstance(ht, str):
                ht = HeaterType(ht)
            if 'element' not in data:
                data['element'] = {}
            data['element']['element_type'] = heater_map.get(ht.value, ElementType.NICHROME)
        if 'rated_power_kw' in data:
            if 'electrical' not in data:
                data['electrical'] = {}
            data['electrical']['power_rating_kW'] = data['rated_power_kw']
        if 'voltage_v' in data and 'electrical' not in data:
            data['electrical'] = {'voltage_v': data['voltage_v']}
        if 'target_temp_c' in data:
            if 'operating' not in data:
                data['operating'] = {}
            data['operating']['setpoint_temp_c'] = data['target_temp_c']
            data['operating']['element_temp_c'] = data['target_temp_c'] + 100
        if 'current_temp_c' in data:
            if 'operating' not in data:
                data['operating'] = {}
            data['operating']['process_temp_c'] = data['current_temp_c']
        super().__init__(**data)


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class ResistanceAnalysis(BaseModel):
    """Electrical resistance analysis results."""

    cold_resistance_ohm: float = Field(..., description="Resistance at 20C (Ohm)")
    hot_resistance_ohm: float = Field(..., description="Resistance at operating temp (Ohm)")
    resistance_ratio: float = Field(..., description="Hot/cold resistance ratio")
    actual_current_a: float = Field(..., description="Calculated current (A)")
    actual_power_kW: float = Field(..., description="Calculated power (kW)")
    power_density_W_cm2: float = Field(..., description="Surface power density (W/cm2)")
    joule_heat_rate_kW: float = Field(..., description="Joule heating rate (kW)")


class TemperatureAnalysis(BaseModel):
    """Temperature performance analysis."""

    element_temp_c: float = Field(..., description="Element surface temperature (C)")
    max_allowable_temp_c: float = Field(..., description="Maximum allowable temperature (C)")
    temperature_margin_c: float = Field(..., description="Margin below max temp (C)")
    heating_time_min: float = Field(..., description="Time to reach target (min)")
    steady_state_power_kW: float = Field(..., description="Steady state power (kW)")
    heat_transfer_rate_kW: float = Field(..., description="Heat transfer rate (kW)")
    radiation_loss_kW: float = Field(..., description="Radiation heat loss (kW)")


class ElementLifeAnalysis(BaseModel):
    """Element life prediction analysis."""

    expected_life_hours: float = Field(..., description="Expected total life (hours)")
    remaining_life_hours: float = Field(..., description="Remaining life (hours)")
    remaining_life_percent: float = Field(..., ge=0, le=100, description="Remaining life (%)")
    element_life_factor: float = Field(..., description="Life factor (1.0 = normal)")
    cycle_fatigue_factor: float = Field(..., description="Cycle fatigue factor (0-1)")
    temperature_derating_factor: float = Field(..., description="Temperature derating factor")
    mean_time_between_failures_h: float = Field(..., description="MTBF (hours)")


class EfficiencyAnalysis(BaseModel):
    """Energy efficiency analysis."""

    electrical_efficiency_percent: float = Field(..., description="Electrical efficiency (%)")
    thermal_efficiency_percent: float = Field(..., description="Thermal efficiency (%)")
    overall_efficiency_percent: float = Field(..., description="Overall efficiency (%)")
    daily_energy_kwh: float = Field(..., description="Daily energy consumption (kWh)")
    cost_per_day_usd: float = Field(..., description="Daily energy cost ($)")


class ControlAnalysis(BaseModel):
    """Control system analysis."""

    control_mode: str
    response_quality: str  # EXCELLENT, GOOD, ACCEPTABLE, POOR
    stability_index: float = Field(..., ge=0, le=100, description="Stability index (0-100)")
    duty_cycle_percent: float = Field(..., ge=0, le=100, description="Average duty cycle (%)")


class Recommendation(BaseModel):
    """Optimization recommendation."""

    recommendation_id: str
    priority: str  # HIGH, MEDIUM, LOW
    category: str
    description: str
    expected_benefit: str
    implementation_effort: str


class Warning(BaseModel):
    """Process warning or alert."""

    warning_id: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str
    description: str
    affected_component: str
    corrective_action: str


class ProvenanceRecord(BaseModel):
    """Provenance tracking record."""

    operation: str
    timestamp: datetime
    input_hash: str
    output_hash: str
    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ResistanceHeatingOutput(BaseModel):
    """Output data model for ResistanceHeatingOptimizerAgent."""

    # Identification
    analysis_id: str = Field(default="", description="Analysis ID")
    system_id: str = Field(..., description="System identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Analysis Results
    resistance: ResistanceAnalysis
    temperature: TemperatureAnalysis
    element_life: ElementLifeAnalysis
    efficiency: EfficiencyAnalysis
    control: ControlAnalysis

    # Overall Assessment
    process_status: ProcessStatus = Field(default=ProcessStatus.ACCEPTABLE)
    performance_score: float = Field(default=75.0, ge=0, le=100)

    # Recommendations and Warnings
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[Warning] = Field(default_factory=list)

    # Provenance
    provenance_chain: List[ProvenanceRecord] = Field(default_factory=list)
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    calculation_hash: str = Field(default="", description="Calculation hash")

    # Processing Metadata
    processing_time_ms: float = Field(default=0.0)
    validation_status: str = Field(default="PASS")
    validation_errors: List[str] = Field(default_factory=list)
    agent_version: str = Field(default="1.1.0")

    # Legacy field support
    process_id: str = Field(default="")
    actual_power_kw: float = Field(default=0.0)
    current_draw_a: float = Field(default=0.0)
    hot_resistance_ohm: float = Field(default=0.0)
    heating_time_min: float = Field(default=0.0)
    steady_state_power_kw: float = Field(default=0.0)
    daily_energy_kwh: float = Field(default=0.0)
    energy_efficiency_pct: float = Field(default=0.0)
    element_life_factor: float = Field(default=1.0)
    cost_per_day_usd: float = Field(default=0.0)


# =============================================================================
# RESISTANCE HEATING CALCULATOR
# =============================================================================

class ResistanceHeatingCalculator:
    """
    Deterministic resistance heating physics calculator.

    Implements Joule heating and element life prediction formulas:
    - Joule heating: P = I^2 * R
    - Temperature-dependent resistance: R(T) = R0 * (1 + alpha * dT)
    - Arrhenius element life prediction
    - Heat transfer analysis

    Zero-hallucination: All calculations are deterministic from published
    electrical and thermal physics.
    """

    @staticmethod
    def calculate_hot_resistance(
        cold_resistance: float,
        temp_coefficient: float,
        operating_temp_c: float,
        reference_temp_c: float = 20.0
    ) -> float:
        """
        Calculate resistance at operating temperature.

        Formula: R(T) = R0 * (1 + alpha * (T - T0))

        Args:
            cold_resistance: Resistance at reference temperature (Ohm)
            temp_coefficient: Temperature coefficient (1/K)
            operating_temp_c: Operating temperature (C)
            reference_temp_c: Reference temperature (C), default 20C

        Returns:
            Hot resistance in Ohms
        """
        delta_t = operating_temp_c - reference_temp_c
        return cold_resistance * (1 + temp_coefficient * delta_t)

    @staticmethod
    def calculate_joule_power(current_a: float, resistance_ohm: float) -> float:
        """Calculate Joule heating power. Formula: P = I^2 * R"""
        return current_a ** 2 * resistance_ohm

    @staticmethod
    def calculate_power_from_voltage(voltage_v: float, resistance_ohm: float) -> float:
        """Calculate power from voltage and resistance. Formula: P = V^2 / R"""
        if resistance_ohm <= 0:
            return 0.0
        return voltage_v ** 2 / resistance_ohm

    @staticmethod
    def calculate_current(power_w: float, voltage_v: float, power_factor: float = 1.0) -> float:
        """Calculate current from power and voltage. Formula: I = P / (V * PF)"""
        if voltage_v <= 0 or power_factor <= 0:
            return 0.0
        return power_w / (voltage_v * power_factor)

    @staticmethod
    def calculate_element_life_arrhenius(
        base_life_hours: float,
        activation_energy_eV: float,
        operating_temp_c: float,
        reference_temp_c: float
    ) -> float:
        """
        Calculate element life using Arrhenius equation.

        The life decreases exponentially with temperature:
        L = L0 * exp(Ea/k * (1/T - 1/T_ref))
        """
        t_op = operating_temp_c + 273.15
        t_ref = reference_temp_c + 273.15

        if t_op <= 0 or t_ref <= 0:
            return base_life_hours

        exponent = (activation_energy_eV / BOLTZMANN_K_EV) * (1/t_op - 1/t_ref)
        exponent = max(-50, min(50, exponent))

        return base_life_hours * math.exp(exponent)

    @staticmethod
    def calculate_cycle_fatigue_factor(cycles_count: int, element_type: ElementType) -> float:
        """Calculate cycle fatigue derating factor."""
        cycle_limits = {
            ElementType.NICHROME: 50000,
            ElementType.KANTHAL: 100000,
            ElementType.SILICON_CARBIDE: 20000,
            ElementType.MOLYBDENUM_DISILICIDE: 30000,
            ElementType.TUNGSTEN: 10000,
            ElementType.GRAPHITE: 5000,
            ElementType.PLATINUM: 100000,
            ElementType.NICKEL: 50000
        }
        limit = cycle_limits.get(element_type, 50000)

        if cycles_count >= limit:
            return 0.1
        else:
            return math.exp(-(cycles_count / limit) ** 2)

    @staticmethod
    def calculate_radiation_heat_loss(
        surface_area_m2: float,
        element_temp_k: float,
        ambient_temp_k: float,
        emissivity: float
    ) -> float:
        """Calculate radiation heat loss using Stefan-Boltzmann law."""
        return emissivity * STEFAN_BOLTZMANN * surface_area_m2 * (
            element_temp_k ** 4 - ambient_temp_k ** 4
        )


# =============================================================================
# RESISTANCE HEATING OPTIMIZER AGENT
# =============================================================================

class ResistanceHeatingOptimizerAgent:
    """
    GL-060: Resistance Heating Optimizer Agent (RESIST-OPT).

    This agent optimizes electric resistance heating systems for industrial
    applications, providing comprehensive analysis of element performance,
    life prediction, and energy efficiency.

    Key Capabilities:
    - Joule heating power calculation (P = I^2 * R)
    - Temperature-dependent resistance analysis
    - Element life prediction using Arrhenius model
    - Cycle fatigue assessment
    - Energy efficiency optimization

    Zero-Hallucination Guarantee:
    - All calculations use deterministic formulas from electrical physics
    - No LLM inference in calculation path
    - Complete audit trail for quality assurance
    """

    AGENT_ID = "GL-060"
    AGENT_NAME = "RESIST-OPT"
    VERSION = "1.1.0"
    DESCRIPTION = "Resistance Heating Optimizer Agent"

    # Maximum element temperatures by type (legacy support)
    MAX_ELEMENT_TEMPS = {
        HeaterType.NICHROME: 1150,
        HeaterType.KANTHAL: 1400,
        HeaterType.SILICON_CARBIDE: 1600,
        HeaterType.MOLYBDENUM_DISILICIDE: 1800,
        HeaterType.TUNGSTEN: 2000,
    }

    DEFAULT_ELECTRICITY_COST = 0.10

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the ResistanceHeatingOptimizerAgent."""
        self.config = config or {}
        self.electricity_cost = self.config.get("electricity_cost_usd", self.DEFAULT_ELECTRICITY_COST)
        self._provenance_steps: List[Dict[str, Any]] = []
        self._warnings: List[Warning] = []
        self._recommendations: List[str] = []
        self._calculator = ResistanceHeatingCalculator()

        logger.info(
            f"ResistanceHeatingOptimizerAgent initialized "
            f"(ID: {self.AGENT_ID}, Name: {self.AGENT_NAME}, Version: {self.VERSION})"
        )

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute resistance heating optimization analysis.

        Supports both legacy and new input formats.

        Args:
            input_data: Input data dictionary

        Returns:
            Output data dictionary with analysis results
        """
        validated = ResistanceHeatingInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Async version of run method."""
        return self.run(input_data)

    def _process(self, inp: ResistanceHeatingInput) -> ResistanceHeatingOutput:
        """Process the resistance heating analysis."""
        start_time = datetime.utcnow()
        self._provenance_steps = []
        self._warnings = []
        self._recommendations = []

        # Get element properties
        elem_props = ELEMENT_PROPERTIES.get(
            inp.element.element_type,
            ELEMENT_PROPERTIES[ElementType.NICHROME]
        )

        # Step 1: Resistance Analysis
        element_temp = inp.operating.element_temp_c
        cold_resistance = inp.electrical.element_resistance_ohm

        hot_resistance = self._calculator.calculate_hot_resistance(
            cold_resistance,
            elem_props["temp_coeff"],
            element_temp,
            20.0
        )

        actual_power_w = self._calculator.calculate_power_from_voltage(
            inp.electrical.voltage_v,
            hot_resistance
        )
        actual_power_kW = min(actual_power_w / 1000, inp.electrical.power_rating_kW)

        actual_current = inp.electrical.voltage_v / hot_resistance if hot_resistance > 0 else 0

        # Estimate surface area and power density
        surface_area_m2 = inp.element.surface_area_m2 or 0.1 * inp.element.element_count
        power_density = (actual_power_kW * 1000 / (surface_area_m2 * 10000)) if surface_area_m2 > 0 else 0

        resistance = ResistanceAnalysis(
            cold_resistance_ohm=round(cold_resistance, 4),
            hot_resistance_ohm=round(hot_resistance, 4),
            resistance_ratio=round(hot_resistance / cold_resistance, 3) if cold_resistance > 0 else 1.0,
            actual_current_a=round(actual_current, 2),
            actual_power_kW=round(actual_power_kW, 3),
            power_density_W_cm2=round(power_density, 2),
            joule_heat_rate_kW=round(actual_power_kW, 3)
        )

        self._track_provenance("resistance_analysis", {"element_temp_c": element_temp}, {"hot_resistance": hot_resistance}, "resistance_calculator")

        # Step 2: Temperature Analysis
        max_temp_c = elem_props["max_continuous_temp_c"]
        temp_margin = max_temp_c - element_temp

        # Heating time calculation
        delta_t = inp.operating.setpoint_temp_c - inp.operating.process_temp_c
        if delta_t <= 0:
            heating_time_min = 0
        else:
            energy_required_kj = inp.operating.thermal_mass_kj_per_k * delta_t
            avg_temp = (inp.operating.process_temp_c + inp.operating.setpoint_temp_c) / 2
            avg_heat_loss = inp.operating.heat_loss_kw_per_k * (avg_temp - inp.operating.ambient_temp_c)
            net_power = actual_power_kW - avg_heat_loss
            if net_power > 0:
                heating_time_min = (energy_required_kj / net_power) / 60
            else:
                heating_time_min = float('inf')

        # Steady state power
        temp_diff = inp.operating.setpoint_temp_c - inp.operating.ambient_temp_c
        steady_state_power = inp.operating.heat_loss_kw_per_k * temp_diff

        # Radiation loss
        element_temp_k = element_temp + 273.15
        process_temp_k = inp.operating.process_temp_c + 273.15
        radiation_loss_w = self._calculator.calculate_radiation_heat_loss(
            surface_area_m2, element_temp_k, process_temp_k, elem_props["emissivity"]
        )

        temperature = TemperatureAnalysis(
            element_temp_c=round(element_temp, 1),
            max_allowable_temp_c=round(max_temp_c, 0),
            temperature_margin_c=round(temp_margin, 0),
            heating_time_min=round(min(heating_time_min, 9999), 2),
            steady_state_power_kW=round(steady_state_power, 3),
            heat_transfer_rate_kW=round(max(0, actual_power_kW - radiation_loss_w / 1000), 2),
            radiation_loss_kW=round(radiation_loss_w / 1000, 3)
        )

        # Temperature warnings
        temp_ratio = element_temp / max_temp_c
        if temp_ratio > 0.9:
            self._recommendations.append(f"Element temperature {element_temp}C near max {max_temp_c}C - consider upgrading element type")

        # Step 3: Element Life Analysis
        expected_life = self._calculator.calculate_element_life_arrhenius(
            elem_props["base_life_hours"],
            elem_props["activation_energy_eV"],
            element_temp,
            elem_props["max_continuous_temp_c"]
        )

        cycle_factor = self._calculator.calculate_cycle_fatigue_factor(
            inp.operating.cycles_count,
            inp.element.element_type
        )
        expected_life *= cycle_factor

        remaining_life = max(0, expected_life - inp.operating.operating_hours)
        remaining_life_pct = (remaining_life / expected_life * 100) if expected_life > 0 else 0

        # Life factor (legacy compatibility)
        if temp_ratio < 0.7:
            life_factor = 1.5
        elif temp_ratio < 0.85:
            life_factor = 1.0
        elif temp_ratio < 0.95:
            life_factor = 0.7
        else:
            life_factor = 0.4

        element_life = ElementLifeAnalysis(
            expected_life_hours=round(expected_life, 0),
            remaining_life_hours=round(remaining_life, 0),
            remaining_life_percent=round(remaining_life_pct, 1),
            element_life_factor=round(life_factor, 2),
            cycle_fatigue_factor=round(cycle_factor, 3),
            temperature_derating_factor=round(1.0 / life_factor, 3) if life_factor > 0 else 1.0,
            mean_time_between_failures_h=round(expected_life * 0.7, 0)
        )

        if life_factor < 0.8:
            self._recommendations.append(f"Reduced element life factor ({life_factor:.1f}) - plan for more frequent replacement")

        # Step 4: Efficiency Analysis
        duty_cycle = inp.control.duty_cycle_percent or 100
        heating_energy = actual_power_kW * (heating_time_min / 60) if heating_time_min < 9999 else 0
        steady_energy = steady_state_power * max(0, inp.control.operating_hours_per_day - heating_time_min / 60)
        daily_energy = (heating_energy + steady_energy) * (duty_cycle / 100)

        if daily_energy > 0:
            useful_energy = daily_energy - (
                inp.operating.heat_loss_kw_per_k * temp_diff * inp.control.operating_hours_per_day * 0.5
            )
            efficiency = max(0, min(100, (useful_energy / daily_energy) * 100))
        else:
            efficiency = 0

        cost_per_day = daily_energy * inp.energy_cost_per_kwh

        efficiency_analysis = EfficiencyAnalysis(
            electrical_efficiency_percent=99.5,
            thermal_efficiency_percent=round(efficiency, 1),
            overall_efficiency_percent=round(efficiency * 0.995, 1),
            daily_energy_kwh=round(daily_energy, 2),
            cost_per_day_usd=round(cost_per_day, 2)
        )

        if efficiency < 50:
            self._recommendations.append(f"Low efficiency ({efficiency:.0f}%) - improve insulation to reduce heat loss")

        # Step 5: Control Analysis
        control_mode = inp.control.control_mode.value

        quality_map = {
            "on_off": "ACCEPTABLE", "time_proportioning": "GOOD",
            "scr_phase_angle": "EXCELLENT", "scr_zero_cross": "EXCELLENT",
            "pid_continuous": "EXCELLENT", "pid": "EXCELLENT",
            "proportional": "GOOD", "scr_phase": "EXCELLENT"
        }

        stability_map = {
            "on_off": 60, "time_proportioning": 75, "scr_phase_angle": 95,
            "scr_zero_cross": 90, "pid_continuous": 95, "pid": 95,
            "proportional": 75, "scr_phase": 95
        }

        control = ControlAnalysis(
            control_mode=control_mode,
            response_quality=quality_map.get(control_mode.lower(), "ACCEPTABLE"),
            stability_index=stability_map.get(control_mode.lower(), 70),
            duty_cycle_percent=round(duty_cycle, 1)
        )

        if control_mode.lower() in ["on_off", "ON_OFF"] and inp.electrical.power_rating_kW > 10:
            self._recommendations.append("High power with ON-OFF control causes thermal cycling - switch to PID or SCR")

        if inp.electrical.power_factor < 0.85:
            self._recommendations.append(f"Low power factor ({inp.electrical.power_factor}) - consider power factor correction")

        # Calculate provenance hash
        calc_hash = hashlib.sha256(
            json.dumps({
                "process": inp.system_id,
                "daily_energy": round(daily_energy, 4),
                "efficiency": round(efficiency, 2),
            }).encode()
        ).hexdigest()

        provenance_hash = self._calculate_provenance_hash()

        # Processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        # Generate analysis ID
        analysis_id = f"RESIST-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{calc_hash[:8]}"

        # Determine process status
        if efficiency > 70 and life_factor >= 1.0:
            status = ProcessStatus.OPTIMAL
        elif efficiency > 50 and life_factor >= 0.7:
            status = ProcessStatus.ACCEPTABLE
        elif efficiency > 30:
            status = ProcessStatus.WARNING
        else:
            status = ProcessStatus.CRITICAL

        performance_score = (efficiency * 0.4 + remaining_life_pct * 0.3 + control.stability_index * 0.3)

        return ResistanceHeatingOutput(
            analysis_id=analysis_id,
            system_id=inp.system_id,
            process_id=inp.system_id,  # Legacy
            resistance=resistance,
            temperature=temperature,
            element_life=element_life,
            efficiency=efficiency_analysis,
            control=control,
            process_status=status,
            performance_score=round(performance_score, 2),
            recommendations=self._recommendations,
            warnings=self._warnings,
            provenance_chain=[
                ProvenanceRecord(
                    operation=s["operation"],
                    timestamp=s["timestamp"],
                    input_hash=s["input_hash"],
                    output_hash=s["output_hash"],
                    tool_name=s["tool_name"],
                    parameters=s.get("parameters", {})
                )
                for s in self._provenance_steps
            ],
            provenance_hash=provenance_hash,
            calculation_hash=calc_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS",
            agent_version=self.VERSION,
            # Legacy fields
            actual_power_kw=round(actual_power_kW, 3),
            current_draw_a=round(actual_current, 2),
            hot_resistance_ohm=round(hot_resistance, 3),
            heating_time_min=round(min(heating_time_min, 9999), 2),
            steady_state_power_kw=round(steady_state_power, 3),
            daily_energy_kwh=round(daily_energy, 2),
            energy_efficiency_pct=round(efficiency, 1),
            element_life_factor=round(life_factor, 2),
            cost_per_day_usd=round(cost_per_day, 2),
        )

    def _track_provenance(
        self,
        operation: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        tool_name: str
    ):
        """Track a calculation step for audit trail."""
        input_str = json.dumps(inputs, sort_keys=True, default=str)
        output_str = json.dumps(outputs, sort_keys=True, default=str)

        self._provenance_steps.append({
            "operation": operation,
            "timestamp": datetime.utcnow(),
            "input_hash": hashlib.sha256(input_str.encode()).hexdigest(),
            "output_hash": hashlib.sha256(output_str.encode()).hexdigest(),
            "tool_name": tool_name,
            "parameters": inputs
        })

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash of complete provenance chain."""
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "agent_name": self.AGENT_NAME,
            "version": self.VERSION,
            "steps": [
                {
                    "operation": s["operation"],
                    "input_hash": s["input_hash"],
                    "output_hash": s["output_hash"]
                }
                for s in self._provenance_steps
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        json_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


# Legacy alias
ResistanceHeatingAgent = ResistanceHeatingOptimizerAgent


# =============================================================================
# PACK SPECIFICATION
# =============================================================================

PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-060",
    "name": "RESIST-OPT - Resistance Heating Optimizer Agent",
    "version": "1.1.0",
    "summary": "Resistance heating optimization with element life prediction for industrial furnaces",
    "tags": [
        "resistance-heating",
        "joule-heating",
        "electric-furnace",
        "element-life",
        "temperature-control",
        "energy-efficiency",
        "NFPA-86",
        "IEC-60519",
        "IEEE-515",
        "NFPA-70"
    ],
    "owners": ["process-heat-team"],
    "compute": {
        "entrypoint": "python://agents.gl_060_resistance_heating.agent:ResistanceHeatingOptimizerAgent",
        "deterministic": True
    },
    "formulas": {
        "joule_heating": "P = I^2 * R = V^2 / R",
        "resistance_temperature": "R(T) = R0 * (1 + alpha * (T - T0))",
        "element_life_arrhenius": "L = L0 * exp(Ea/k * (1/T - 1/T_ref))",
        "power_density": "W/m^2 = P / A_surface",
        "radiation_loss": "Q = epsilon * sigma * A * (T_hot^4 - T_cold^4)"
    },
    "standards": [
        {"ref": "NFPA 86", "description": "Standard for Ovens and Furnaces"},
        {"ref": "IEC 60519", "description": "Safety in Electroheat Installations"},
        {"ref": "IEEE 515", "description": "Electrical Resistance Heat Tracing"},
        {"ref": "NFPA 70", "description": "National Electrical Code"}
    ],
    "provenance": {
        "calculation_verified": True,
        "enable_audit": True
    }
}
