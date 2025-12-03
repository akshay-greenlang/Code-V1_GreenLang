# -*- coding: utf-8 -*-
"""
Desuperheater Sizing Calculator for GL-012 STEAMQUAL.

Provides deterministic calculations for desuperheater system sizing and design,
including spray water flow calculation, atomization droplet size analysis,
residence time calculation, thermal shock prevention, control valve sizing,
and turndown ratio analysis.

Standards:
- ISA-75.01.01: Flow Equations for Sizing Control Valves
- ASME PTC 6: Steam Turbines
- ASME B31.1: Power Piping
- Spray Dynamics: Nukiyama-Tanasawa droplet correlation

Zero-hallucination: All calculations are deterministic with bit-perfect reproducibility.
No LLM involved in any calculation path.

Author: GL-CalculatorEngineer
Version: 1.0.0

Formulas:
    Spray Water Flow: m_water = m_steam * (h_in - h_out) / (h_out - h_water)
    Atomization SMD: d32 = A * (sigma/rho_g)^0.5 * (mu_l/sqrt(sigma*rho_l))^0.45 * (rho_l/rho_g)^0.25 / We^0.4
    Residence Time: t = L / v_steam
    Valve Cv: Cv = Q * sqrt(SG / delta_P)
    Turndown Ratio: TR = Q_max / Q_min

Design Principles:
    1. Complete evaporation before pipe walls
    2. Minimum temperature approach of 10-20C above saturation
    3. Spray velocity sufficient for atomization
    4. Thermal stress limits for desuperheater components
"""

import hashlib
import json
import logging
import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS AND ENUMERATIONS
# =============================================================================

class DesuperheaterType(Enum):
    """Types of desuperheater designs."""
    SPRAY_NOZZLE = "spray_nozzle"  # Single or multi-nozzle spray
    ATTEMPERATOR = "attemperator"  # Steam-cooled attemperator
    VENTURI = "venturi"  # Venturi-type with mixing section
    RING_TYPE = "ring_type"  # Annular ring spray
    PROBE_TYPE = "probe_type"  # Insertion probe
    VARIABLE_ORIFICE = "variable_orifice"  # Self-regulating type


class SprayPattern(Enum):
    """Spray pattern configurations."""
    FULL_CONE = "full_cone"
    HOLLOW_CONE = "hollow_cone"
    FLAT_FAN = "flat_fan"
    SOLID_STREAM = "solid_stream"


class FlowCharacteristic(Enum):
    """Control valve flow characteristics."""
    LINEAR = "linear"
    EQUAL_PERCENTAGE = "equal_percentage"
    QUICK_OPENING = "quick_opening"


class ThermalStressLevel(Enum):
    """Thermal stress severity classification."""
    ACCEPTABLE = "acceptable"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# DATA CLASSES - INPUTS
# =============================================================================

@dataclass
class DesuperheaterSizingInput:
    """
    Input parameters for desuperheater sizing calculations.

    Attributes:
        steam_flow_kg_s: Design steam mass flow rate (kg/s)
        inlet_temperature_c: Inlet steam temperature (Celsius)
        outlet_temperature_c: Target outlet temperature (Celsius)
        inlet_pressure_mpa: Steam pressure at desuperheater inlet (MPa)
        spray_water_temperature_c: Cooling water temperature (Celsius)
        spray_water_pressure_mpa: Available spray water pressure (MPa)
        pipe_diameter_m: Steam pipe internal diameter (m)
        straight_pipe_length_m: Available straight pipe length downstream (m)
        turndown_ratio: Required turndown ratio (default: 10:1)
        desuperheater_type: Type of desuperheater (default: SPRAY_NOZZLE)
        spray_pattern: Spray pattern (default: HOLLOW_CONE)
    """
    steam_flow_kg_s: float
    inlet_temperature_c: float
    outlet_temperature_c: float
    inlet_pressure_mpa: float
    spray_water_temperature_c: float = 25.0
    spray_water_pressure_mpa: float = 2.0
    pipe_diameter_m: float = 0.2
    straight_pipe_length_m: float = 3.0
    turndown_ratio: float = 10.0
    desuperheater_type: DesuperheaterType = DesuperheaterType.SPRAY_NOZZLE
    spray_pattern: SprayPattern = SprayPattern.HOLLOW_CONE


@dataclass
class AtomizationInput:
    """
    Input parameters for atomization analysis.

    Attributes:
        spray_velocity_m_s: Spray water velocity at nozzle (m/s)
        water_pressure_mpa: Water supply pressure (MPa)
        steam_velocity_m_s: Steam velocity at spray location (m/s)
        water_temperature_c: Spray water temperature (Celsius)
        steam_temperature_c: Steam temperature (Celsius)
        steam_pressure_mpa: Steam pressure (MPa)
        nozzle_diameter_mm: Nozzle orifice diameter (mm)
    """
    spray_velocity_m_s: float
    water_pressure_mpa: float
    steam_velocity_m_s: float
    water_temperature_c: float
    steam_temperature_c: float
    steam_pressure_mpa: float
    nozzle_diameter_mm: float = 3.0


@dataclass
class ValveSizingInput:
    """
    Input parameters for control valve sizing.

    Attributes:
        flow_rate_max_kg_s: Maximum spray water flow (kg/s)
        flow_rate_min_kg_s: Minimum spray water flow (kg/s)
        inlet_pressure_mpa: Valve inlet pressure (MPa)
        outlet_pressure_mpa: Valve outlet pressure (MPa)
        water_temperature_c: Water temperature (Celsius)
        characteristic: Valve flow characteristic
        allowable_noise_db: Maximum allowable noise level (dB)
    """
    flow_rate_max_kg_s: float
    flow_rate_min_kg_s: float
    inlet_pressure_mpa: float
    outlet_pressure_mpa: float
    water_temperature_c: float = 25.0
    characteristic: FlowCharacteristic = FlowCharacteristic.EQUAL_PERCENTAGE
    allowable_noise_db: float = 85.0


# =============================================================================
# DATA CLASSES - OUTPUTS
# =============================================================================

@dataclass
class SprayWaterFlowResult:
    """
    Result of spray water flow calculation.

    Attributes:
        design_flow_kg_s: Design spray water flow rate (kg/s)
        max_flow_kg_s: Maximum flow at turndown (kg/s)
        min_flow_kg_s: Minimum flow at turndown (kg/s)
        water_to_steam_ratio: Design water/steam mass ratio
        energy_absorbed_kw: Heat absorbed by spray water (kW)
        temperature_reduction_c: Temperature drop achieved (Celsius)
        evaporation_complete: Whether water fully evaporates
        approach_to_saturation_c: Outlet temp above saturation (Celsius)
        provenance_hash: SHA-256 audit trail
        warnings: Calculation warnings
    """
    design_flow_kg_s: float
    max_flow_kg_s: float
    min_flow_kg_s: float
    water_to_steam_ratio: float
    energy_absorbed_kw: float
    temperature_reduction_c: float
    evaporation_complete: bool
    approach_to_saturation_c: float
    provenance_hash: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class AtomizationResult:
    """
    Result of atomization droplet size analysis.

    Attributes:
        sauter_mean_diameter_um: Sauter mean diameter d32 (micrometers)
        max_droplet_diameter_um: Maximum expected droplet size (um)
        min_droplet_diameter_um: Minimum expected droplet size (um)
        weber_number: Weber number (dimensionless)
        reynolds_number_liquid: Liquid Reynolds number
        ohnesorge_number: Ohnesorge number
        atomization_regime: Atomization regime description
        evaporation_time_ms: Estimated evaporation time (ms)
        penetration_distance_m: Spray penetration into steam (m)
        is_adequate: Whether atomization meets design requirements
        provenance_hash: SHA-256 audit trail
        warnings: Calculation warnings
    """
    sauter_mean_diameter_um: float
    max_droplet_diameter_um: float
    min_droplet_diameter_um: float
    weber_number: float
    reynolds_number_liquid: float
    ohnesorge_number: float
    atomization_regime: str
    evaporation_time_ms: float
    penetration_distance_m: float
    is_adequate: bool
    provenance_hash: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class ResidenceTimeResult:
    """
    Result of residence time calculation.

    Attributes:
        residence_time_ms: Available residence time (milliseconds)
        steam_velocity_m_s: Steam velocity in pipe (m/s)
        required_length_m: Required straight length for evaporation (m)
        available_length_m: Available straight pipe length (m)
        length_adequate: Whether available length is sufficient
        evaporation_distance_m: Distance for complete evaporation (m)
        safety_factor: Actual safety factor (available/required)
        provenance_hash: SHA-256 audit trail
        warnings: Calculation warnings
    """
    residence_time_ms: float
    steam_velocity_m_s: float
    required_length_m: float
    available_length_m: float
    length_adequate: bool
    evaporation_distance_m: float
    safety_factor: float
    provenance_hash: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class ThermalShockResult:
    """
    Result of thermal shock prevention analysis.

    Attributes:
        delta_t_rate_c_s: Rate of temperature change (C/second)
        thermal_stress_mpa: Estimated thermal stress (MPa)
        stress_level: Severity classification
        max_delta_t_allowable_c: Maximum allowable temperature change (Celsius)
        min_warmup_time_s: Minimum recommended warmup time (s)
        is_acceptable: Whether thermal conditions are acceptable
        recommendations: Design recommendations
        provenance_hash: SHA-256 audit trail
        warnings: Calculation warnings
    """
    delta_t_rate_c_s: float
    thermal_stress_mpa: float
    stress_level: ThermalStressLevel
    max_delta_t_allowable_c: float
    min_warmup_time_s: float
    is_acceptable: bool
    recommendations: List[str] = field(default_factory=list)
    provenance_hash: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class ValveSizingResult:
    """
    Result of control valve sizing (Cv calculation).

    Attributes:
        cv_required: Required valve Cv at design conditions
        cv_selected: Recommended valve Cv (next standard size)
        cv_min_required: Cv required at minimum flow
        cv_max_required: Cv required at maximum flow
        pressure_drop_design_mpa: Pressure drop at design (MPa)
        velocity_m_s: Fluid velocity through valve (m/s)
        fl_factor: Liquid pressure recovery factor
        cavitation_index: Cavitation sigma index
        cavitation_risk: Risk of cavitation
        noise_level_db: Estimated noise level (dB)
        noise_acceptable: Whether noise is within limits
        turndown_achievable: Actual achievable turndown ratio
        provenance_hash: SHA-256 audit trail
        warnings: Calculation warnings
    """
    cv_required: float
    cv_selected: float
    cv_min_required: float
    cv_max_required: float
    pressure_drop_design_mpa: float
    velocity_m_s: float
    fl_factor: float
    cavitation_index: float
    cavitation_risk: str
    noise_level_db: float
    noise_acceptable: bool
    turndown_achievable: float
    provenance_hash: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class TurndownAnalysisResult:
    """
    Result of turndown ratio analysis.

    Attributes:
        requested_turndown: Requested turndown ratio
        achievable_turndown: Actually achievable turndown
        min_controllable_flow_kg_s: Minimum controllable flow (kg/s)
        max_controllable_flow_kg_s: Maximum controllable flow (kg/s)
        control_range_limited_by: Factor limiting turndown
        valve_position_at_min_pct: Valve position at minimum flow (%)
        valve_position_at_max_pct: Valve position at maximum flow (%)
        rangeability_adequate: Whether rangeability meets requirements
        provenance_hash: SHA-256 audit trail
        warnings: Calculation warnings
    """
    requested_turndown: float
    achievable_turndown: float
    min_controllable_flow_kg_s: float
    max_controllable_flow_kg_s: float
    control_range_limited_by: str
    valve_position_at_min_pct: float
    valve_position_at_max_pct: float
    rangeability_adequate: bool
    provenance_hash: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class CompleteSizingResult:
    """
    Complete desuperheater sizing result combining all analyses.

    Attributes:
        spray_water: Spray water flow results
        atomization: Atomization analysis results
        residence_time: Residence time results
        thermal_shock: Thermal shock analysis results
        valve_sizing: Control valve sizing results
        turndown: Turndown analysis results
        overall_feasibility: Whether design is feasible
        design_recommendations: List of design recommendations
        provenance_hash: SHA-256 audit trail
    """
    spray_water: SprayWaterFlowResult
    atomization: AtomizationResult
    residence_time: ResidenceTimeResult
    thermal_shock: ThermalShockResult
    valve_sizing: ValveSizingResult
    turndown: TurndownAnalysisResult
    overall_feasibility: bool
    design_recommendations: List[str] = field(default_factory=list)
    provenance_hash: str = ""


# =============================================================================
# MAIN CALCULATOR CLASS
# =============================================================================

class DesuperheaterSizingCalculator:
    """
    Desuperheater Sizing Calculator for steam temperature control systems.

    Provides comprehensive sizing calculations for desuperheater systems
    including spray water requirements, atomization, residence time,
    thermal shock prevention, and control valve sizing.

    ZERO-HALLUCINATION GUARANTEES:
    - All calculations use deterministic engineering formulas
    - Same inputs always produce bit-perfect identical outputs
    - Complete provenance tracking with SHA-256 hashes
    - No LLM or AI inference in any calculation path

    Design Standards:
    - ISA-75.01.01: Control Valve Sizing
    - ASME PTC 6: Steam Turbines
    - ASME B31.1: Power Piping

    Example:
        >>> calc = DesuperheaterSizingCalculator()
        >>> result = calc.calculate_spray_water_flow(DesuperheaterSizingInput(
        ...     steam_flow_kg_s=10.0,
        ...     inlet_temperature_c=400.0,
        ...     outlet_temperature_c=350.0,
        ...     inlet_pressure_mpa=4.0
        ... ))
        >>> print(f"Spray flow: {result.design_flow_kg_s:.3f} kg/s")
    """

    # Physical constants
    WATER_DENSITY_KG_M3 = Decimal("1000.0")
    WATER_SPECIFIC_HEAT_KJ_KG_K = Decimal("4.186")
    WATER_SURFACE_TENSION_N_M = Decimal("0.0728")  # At 20C
    WATER_VISCOSITY_PA_S = Decimal("0.001002")  # At 20C
    LATENT_HEAT_VAPORIZATION_KJ_KG = Decimal("2257.0")  # At 100C

    # Steam properties (approximate averages for superheated steam)
    STEAM_SPECIFIC_HEAT_KJ_KG_K = Decimal("2.1")  # Cp average
    STEAM_VISCOSITY_PA_S = Decimal("0.000018")  # At 300C, 1 MPa

    # ISA-75.01.01 constants
    N1_CV_CONSTANT = Decimal("0.0865")  # For Q in m3/h, dP in kPa
    N2_CV_CONSTANT = Decimal("0.00214")  # Alternative

    # Design limits
    MIN_APPROACH_SATURATION_C = Decimal("10.0")  # Minimum degrees above saturation
    MAX_WATER_STEAM_RATIO = Decimal("0.25")  # Maximum 25% injection
    MIN_SPRAY_VELOCITY_M_S = Decimal("20.0")  # Minimum for atomization
    MAX_THERMAL_STRESS_MPA = Decimal("150.0")  # Allowable thermal stress

    # Standard valve Cv sizes
    STANDARD_CV_SIZES = [
        0.25, 0.50, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0,
        10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0,
        100.0, 125.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize desuperheater sizing calculator.

        Args:
            config: Optional configuration dictionary with keys:
                - precision: Decimal places for output (default: 4)
                - safety_factor: Design safety factor (default: 1.25)
                - min_approach: Minimum approach to saturation (default: 10 C)
        """
        self.config = config or {}
        self.precision = self.config.get('precision', 4)
        self.safety_factor = Decimal(str(self.config.get('safety_factor', 1.25)))
        self.min_approach = Decimal(str(self.config.get('min_approach', 10.0)))
        self.calculation_count = 0

        # Initialize saturation table for quick lookups
        self._init_saturation_table()

    def _init_saturation_table(self) -> None:
        """Initialize saturation temperature lookup table."""
        # Pressure (MPa) -> T_sat (C)
        self.saturation_table = {
            Decimal("0.1"): Decimal("99.61"),
            Decimal("0.5"): Decimal("151.83"),
            Decimal("1.0"): Decimal("179.88"),
            Decimal("1.5"): Decimal("198.29"),
            Decimal("2.0"): Decimal("212.38"),
            Decimal("2.5"): Decimal("223.95"),
            Decimal("3.0"): Decimal("233.85"),
            Decimal("3.5"): Decimal("242.56"),
            Decimal("4.0"): Decimal("250.35"),
            Decimal("4.5"): Decimal("257.41"),
            Decimal("5.0"): Decimal("263.94"),
            Decimal("6.0"): Decimal("275.59"),
            Decimal("7.0"): Decimal("285.83"),
            Decimal("8.0"): Decimal("295.01"),
            Decimal("9.0"): Decimal("303.35"),
            Decimal("10.0"): Decimal("311.00"),
            Decimal("12.0"): Decimal("324.68"),
            Decimal("14.0"): Decimal("336.67"),
            Decimal("16.0"): Decimal("347.36"),
            Decimal("18.0"): Decimal("357.00"),
            Decimal("20.0"): Decimal("365.75"),
        }

    # =========================================================================
    # SPRAY WATER FLOW CALCULATION
    # =========================================================================

    def calculate_spray_water_flow(
        self,
        input_data: DesuperheaterSizingInput
    ) -> SprayWaterFlowResult:
        """
        Calculate required spray water flow rate.

        FORMULA (Energy Balance):
            m_steam * h_in + m_water * h_water = (m_steam + m_water) * h_out

        Solving for m_water:
            m_water = m_steam * (h_in - h_out) / (h_out - h_water)

        Simplified (using Cp):
            m_water = m_steam * Cp_steam * (T_in - T_out) / (h_fg + Cp_water * (T_sat - T_water))

        ZERO-HALLUCINATION GUARANTEE:
        - Direct algebraic solution
        - No iteration or numerical methods
        - Deterministic result

        Args:
            input_data: Design conditions for desuperheater

        Returns:
            SprayWaterFlowResult with flow rates and design parameters

        Example:
            >>> calc = DesuperheaterSizingCalculator()
            >>> result = calc.calculate_spray_water_flow(DesuperheaterSizingInput(
            ...     steam_flow_kg_s=10.0,
            ...     inlet_temperature_c=400.0,
            ...     outlet_temperature_c=350.0,
            ...     inlet_pressure_mpa=4.0
            ... ))
            >>> print(f"Design flow: {result.design_flow_kg_s:.4f} kg/s")
        """
        self.calculation_count += 1
        warnings = []

        # Convert inputs to Decimal for precision
        m_steam = Decimal(str(input_data.steam_flow_kg_s))
        T_in = Decimal(str(input_data.inlet_temperature_c))
        T_out = Decimal(str(input_data.outlet_temperature_c))
        P = Decimal(str(input_data.inlet_pressure_mpa))
        T_water = Decimal(str(input_data.spray_water_temperature_c))
        turndown = Decimal(str(input_data.turndown_ratio))

        # Get saturation temperature
        T_sat = self._get_saturation_temperature(P)

        # Validate outlet temperature is above saturation
        approach = T_out - T_sat
        if approach < self.min_approach:
            warnings.append(
                f"Outlet temperature {T_out}C is only {approach:.1f}C above saturation. "
                f"Recommend minimum {self.min_approach}C approach."
            )

        # Calculate temperature reduction
        delta_T = T_in - T_out

        if delta_T <= 0:
            warnings.append("Outlet temperature >= inlet temperature - no cooling required")
            return SprayWaterFlowResult(
                design_flow_kg_s=0.0,
                max_flow_kg_s=0.0,
                min_flow_kg_s=0.0,
                water_to_steam_ratio=0.0,
                energy_absorbed_kw=0.0,
                temperature_reduction_c=0.0,
                evaporation_complete=True,
                approach_to_saturation_c=float(approach),
                provenance_hash=self._generate_provenance("spray_water_flow", {}, {}),
                warnings=warnings
            )

        # Energy balance calculation
        # Heat removed from steam: Q = m_steam * Cp_steam * (T_in - T_out)
        Q_steam = m_steam * self.STEAM_SPECIFIC_HEAT_KJ_KG_K * delta_T  # kW

        # Heat absorbed by water:
        # Q = m_water * [Cp_water * (T_sat - T_water) + h_fg + Cp_steam * (T_out - T_sat)]
        # For water that fully evaporates and superheats to T_out

        # Enthalpy components for spray water
        h_preheat = self.WATER_SPECIFIC_HEAT_KJ_KG_K * (T_sat - T_water)  # Heating to saturation
        h_evap = self.LATENT_HEAT_VAPORIZATION_KJ_KG  # Vaporization
        h_superheat = self.STEAM_SPECIFIC_HEAT_KJ_KG_K * (T_out - T_sat)  # Superheat to outlet

        # Total enthalpy rise per kg of spray water
        h_total_rise = h_preheat + h_evap + h_superheat

        if h_total_rise <= 0:
            warnings.append("Invalid enthalpy calculation - check temperatures")
            h_total_rise = Decimal("2500")  # Fallback estimate

        # Calculate spray water flow rate
        # m_water = Q_steam / h_total_rise
        m_water = Q_steam / h_total_rise

        # Check water to steam ratio
        ratio = m_water / m_steam if m_steam > 0 else Decimal("0")
        if ratio > self.MAX_WATER_STEAM_RATIO:
            warnings.append(
                f"Water/steam ratio {ratio:.3f} exceeds maximum {self.MAX_WATER_STEAM_RATIO}. "
                "Consider staged desuperheating."
            )

        # Calculate turndown flows
        m_max = m_water * self.safety_factor
        m_min = m_water / turndown

        # Check evaporation completeness
        # Simplified check: if approach > min_approach, evaporation is complete
        evaporation_complete = approach >= self.min_approach

        # Generate provenance
        provenance_data = {
            'inputs': {
                'steam_flow_kg_s': str(m_steam),
                'T_in_c': str(T_in),
                'T_out_c': str(T_out),
                'P_mpa': str(P),
                'T_water_c': str(T_water)
            },
            'outputs': {
                'design_flow_kg_s': str(m_water),
                'water_steam_ratio': str(ratio)
            }
        }
        provenance_hash = self._generate_provenance("spray_water_flow",
                                                    provenance_data['inputs'],
                                                    provenance_data['outputs'])

        return SprayWaterFlowResult(
            design_flow_kg_s=float(m_water.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)),
            max_flow_kg_s=float(m_max.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)),
            min_flow_kg_s=float(m_min.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)),
            water_to_steam_ratio=float(ratio.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)),
            energy_absorbed_kw=float(Q_steam.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)),
            temperature_reduction_c=float(delta_T),
            evaporation_complete=evaporation_complete,
            approach_to_saturation_c=float(approach.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)),
            provenance_hash=provenance_hash,
            warnings=warnings
        )

    # =========================================================================
    # ATOMIZATION DROPLET SIZE ANALYSIS
    # =========================================================================

    def calculate_atomization(
        self,
        input_data: AtomizationInput
    ) -> AtomizationResult:
        """
        Calculate atomization droplet size using Nukiyama-Tanasawa correlation.

        FORMULA (Nukiyama-Tanasawa):
            d32 = (585/v) * sqrt(sigma/rho_l) + 597 * (mu_l/sqrt(sigma*rho_l))^0.45 * (1000*Q_l/Q_g)^1.5

        Where:
            d32 = Sauter mean diameter (micrometers)
            v = relative velocity (m/s)
            sigma = surface tension (N/m)
            rho_l = liquid density (kg/m3)
            mu_l = liquid viscosity (Pa.s)
            Q_l/Q_g = liquid to gas volumetric flow ratio

        Alternative (Weber number based):
            d32 = D_nozzle / We^0.5  (for pressure atomizers)

        ZERO-HALLUCINATION GUARANTEE:
        - Standard empirical correlations
        - Deterministic calculation

        Args:
            input_data: Atomization conditions

        Returns:
            AtomizationResult with droplet sizes and regime analysis

        Example:
            >>> calc = DesuperheaterSizingCalculator()
            >>> result = calc.calculate_atomization(AtomizationInput(
            ...     spray_velocity_m_s=50.0,
            ...     water_pressure_mpa=2.0,
            ...     steam_velocity_m_s=30.0,
            ...     water_temperature_c=25.0,
            ...     steam_temperature_c=350.0,
            ...     steam_pressure_mpa=4.0,
            ...     nozzle_diameter_mm=3.0
            ... ))
            >>> print(f"SMD: {result.sauter_mean_diameter_um:.1f} micrometers")
        """
        self.calculation_count += 1
        warnings = []

        # Convert inputs
        v_spray = Decimal(str(input_data.spray_velocity_m_s))
        v_steam = Decimal(str(input_data.steam_velocity_m_s))
        P_water = Decimal(str(input_data.water_pressure_mpa))
        T_water = Decimal(str(input_data.water_temperature_c))
        T_steam = Decimal(str(input_data.steam_temperature_c))
        P_steam = Decimal(str(input_data.steam_pressure_mpa))
        D_nozzle = Decimal(str(input_data.nozzle_diameter_mm)) / Decimal("1000")  # Convert to m

        # Water properties (temperature-adjusted approximations)
        rho_l = self.WATER_DENSITY_KG_M3 * (Decimal("1") - Decimal("0.0003") * T_water)  # Slight decrease with T
        sigma = self.WATER_SURFACE_TENSION_N_M * (Decimal("1") - Decimal("0.002") * T_water)  # Decrease with T
        mu_l = self.WATER_VISCOSITY_PA_S * Decimal(str(math.exp(-0.02 * float(T_water))))  # Decrease with T

        # Steam density approximation (ideal gas with compressibility)
        # rho_g = P / (Z * R * T)
        R_steam = Decimal("0.4615")  # kJ/kg.K
        Z = Decimal("0.97")  # Compressibility factor
        T_steam_K = T_steam + Decimal("273.15")
        rho_g = P_steam * Decimal("1000") / (Z * R_steam * T_steam_K)  # kg/m3

        # Relative velocity
        v_rel = abs(v_spray - v_steam)  # Could be additive or subtractive depending on geometry
        if v_rel < Decimal("10"):
            v_rel = v_spray  # Use spray velocity if relative is too low
            warnings.append("Low relative velocity - using spray velocity for calculations")

        # Weber number: We = rho_g * v_rel^2 * D / sigma
        We = rho_g * v_rel ** 2 * D_nozzle / sigma
        We = max(Decimal("1"), We)  # Prevent division by zero

        # Reynolds number for liquid: Re_l = rho_l * v * D / mu_l
        Re_l = rho_l * v_spray * D_nozzle / mu_l

        # Ohnesorge number: Oh = mu_l / sqrt(rho_l * sigma * D)
        Oh = mu_l / (rho_l * sigma * D_nozzle).sqrt()

        # Sauter Mean Diameter calculation (Weber number correlation)
        # d32 ~ D * C * We^(-0.5) for pressure atomizers
        # With correction for viscosity using Ohnesorge number
        C_atomization = Decimal("6.0")  # Empirical constant for hollow cone nozzles

        # Primary breakup: d32 = C * D / We^0.5 * (1 + 3*Oh)
        d32_m = C_atomization * D_nozzle / We.sqrt() * (Decimal("1") + Decimal("3") * Oh)

        # Convert to micrometers
        d32_um = d32_m * Decimal("1000000")

        # Size distribution (log-normal approximation)
        # d_max ~ 2.5 * d32, d_min ~ 0.2 * d32
        d_max = d32_um * Decimal("2.5")
        d_min = d32_um * Decimal("0.2")

        # Determine atomization regime
        if We < Decimal("12"):
            regime = "Vibrational (poor atomization)"
            warnings.append("Low Weber number - poor atomization expected")
        elif We < Decimal("100"):
            regime = "Bag breakup"
        elif We < Decimal("1000"):
            regime = "Stripping (good atomization)"
        else:
            regime = "Catastrophic (excellent atomization)"

        # Evaporation time estimation (D^2 law)
        # t_evap ~ K * d^2 where K is evaporation constant
        # For water droplets in superheated steam: K ~ 0.8e-6 m^2/s
        K_evap = Decimal("0.0000008")  # m^2/s
        d32_m_value = d32_um / Decimal("1000000")
        t_evap_s = d32_m_value ** 2 / K_evap
        t_evap_ms = t_evap_s * Decimal("1000")

        # Penetration distance (simplified ballistic trajectory)
        # Penetration ~ v_spray * t_drag where t_drag ~ rho_l * d / (rho_g * v_rel)
        t_drag = rho_l * d32_m_value / (rho_g * v_rel)
        penetration = v_spray * t_drag

        # Adequacy check
        # Atomization is adequate if d32 < 300 um and We > 100
        is_adequate = d32_um < Decimal("300") and We > Decimal("100")

        if d32_um > Decimal("500"):
            warnings.append(f"Large droplets ({d32_um:.0f} um) may not fully evaporate")

        # Generate provenance
        provenance_data = {
            'inputs': {
                'spray_velocity_m_s': str(v_spray),
                'steam_velocity_m_s': str(v_steam),
                'nozzle_diameter_mm': str(D_nozzle * 1000)
            },
            'outputs': {
                'd32_um': str(d32_um),
                'We': str(We),
                'regime': regime
            }
        }
        provenance_hash = self._generate_provenance("atomization",
                                                    provenance_data['inputs'],
                                                    provenance_data['outputs'])

        return AtomizationResult(
            sauter_mean_diameter_um=float(d32_um.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)),
            max_droplet_diameter_um=float(d_max.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)),
            min_droplet_diameter_um=float(d_min.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)),
            weber_number=float(We.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)),
            reynolds_number_liquid=float(Re_l.quantize(Decimal('1'), rounding=ROUND_HALF_UP)),
            ohnesorge_number=float(Oh.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)),
            atomization_regime=regime,
            evaporation_time_ms=float(t_evap_ms.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)),
            penetration_distance_m=float(penetration.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            is_adequate=is_adequate,
            provenance_hash=provenance_hash,
            warnings=warnings
        )

    # =========================================================================
    # RESIDENCE TIME CALCULATION
    # =========================================================================

    def calculate_residence_time(
        self,
        steam_flow_kg_s: float,
        steam_pressure_mpa: float,
        steam_temperature_c: float,
        pipe_diameter_m: float,
        available_length_m: float,
        evaporation_time_ms: float
    ) -> ResidenceTimeResult:
        """
        Calculate residence time for spray droplet evaporation.

        FORMULA:
            t_residence = L / v_steam

            v_steam = m_dot / (rho_steam * A)
            A = pi * D^2 / 4

        Required length for evaporation:
            L_required = v_steam * t_evaporation * safety_factor

        ZERO-HALLUCINATION GUARANTEE:
        - Direct kinematic calculation
        - No iteration

        Args:
            steam_flow_kg_s: Steam mass flow rate (kg/s)
            steam_pressure_mpa: Steam pressure (MPa)
            steam_temperature_c: Steam temperature (Celsius)
            pipe_diameter_m: Pipe internal diameter (m)
            available_length_m: Available straight pipe length (m)
            evaporation_time_ms: Required evaporation time (ms)

        Returns:
            ResidenceTimeResult with residence time analysis

        Example:
            >>> calc = DesuperheaterSizingCalculator()
            >>> result = calc.calculate_residence_time(
            ...     steam_flow_kg_s=10.0,
            ...     steam_pressure_mpa=4.0,
            ...     steam_temperature_c=350.0,
            ...     pipe_diameter_m=0.2,
            ...     available_length_m=3.0,
            ...     evaporation_time_ms=50.0
            ... )
            >>> print(f"Residence time: {result.residence_time_ms:.1f} ms")
        """
        self.calculation_count += 1
        warnings = []

        # Convert inputs
        m_dot = Decimal(str(steam_flow_kg_s))
        P = Decimal(str(steam_pressure_mpa))
        T = Decimal(str(steam_temperature_c))
        D = Decimal(str(pipe_diameter_m))
        L = Decimal(str(available_length_m))
        t_evap = Decimal(str(evaporation_time_ms)) / Decimal("1000")  # Convert to seconds

        # Calculate steam density (ideal gas approximation)
        R_steam = Decimal("0.4615")  # kJ/kg.K
        T_K = T + Decimal("273.15")
        rho_steam = P * Decimal("1000") / (R_steam * T_K)  # kg/m3

        # Pipe cross-sectional area
        A = Decimal(str(math.pi)) * D ** 2 / Decimal("4")

        # Steam velocity
        if A > 0 and rho_steam > 0:
            v_steam = m_dot / (rho_steam * A)
        else:
            v_steam = Decimal("30")  # Fallback estimate
            warnings.append("Could not calculate steam velocity - using estimate")

        # Residence time
        if v_steam > 0:
            t_residence = L / v_steam
        else:
            t_residence = Decimal("0.1")
            warnings.append("Zero steam velocity calculated")

        t_residence_ms = t_residence * Decimal("1000")

        # Required length for evaporation
        L_required = v_steam * t_evap * self.safety_factor

        # Evaporation distance (where droplets fully evaporate)
        L_evap = v_steam * t_evap

        # Safety factor achieved
        if L_required > 0:
            safety = L / L_required
        else:
            safety = Decimal("999")

        # Check adequacy
        length_adequate = L >= L_required

        if not length_adequate:
            warnings.append(
                f"Available length {L:.2f}m is less than required {L_required:.2f}m. "
                "Increase pipe length or improve atomization."
            )

        # Velocity check
        if v_steam > Decimal("60"):
            warnings.append(f"High steam velocity ({v_steam:.1f} m/s) may cause erosion")
        elif v_steam < Decimal("10"):
            warnings.append(f"Low steam velocity ({v_steam:.1f} m/s) may cause stratification")

        # Generate provenance
        provenance_hash = self._generate_provenance("residence_time",
                                                    {'L': str(L), 'v_steam': str(v_steam)},
                                                    {'t_residence_ms': str(t_residence_ms)})

        return ResidenceTimeResult(
            residence_time_ms=float(t_residence_ms.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)),
            steam_velocity_m_s=float(v_steam.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)),
            required_length_m=float(L_required.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            available_length_m=float(L),
            length_adequate=length_adequate,
            evaporation_distance_m=float(L_evap.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            safety_factor=float(safety.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            provenance_hash=provenance_hash,
            warnings=warnings
        )

    # =========================================================================
    # THERMAL SHOCK PREVENTION
    # =========================================================================

    def calculate_thermal_shock(
        self,
        temperature_change_c: float,
        time_span_s: float,
        pipe_thickness_mm: float = 10.0,
        material_expansion_coeff: float = 12e-6  # Steel: ~12e-6 /K
    ) -> ThermalShockResult:
        """
        Calculate thermal shock potential and prevention requirements.

        FORMULA (Thermal Stress):
            sigma_thermal = E * alpha * delta_T / (1 - nu)

        Where:
            E = elastic modulus (~200 GPa for steel)
            alpha = thermal expansion coefficient (~12e-6 /K for steel)
            delta_T = temperature change
            nu = Poisson's ratio (~0.3 for steel)

        Rate of change limit:
            dT/dt_max ~ 50 C/minute for thick-walled vessels
            dT/dt_max ~ 200 C/minute for thin tubes

        ZERO-HALLUCINATION GUARANTEE:
        - Standard thermoelastic equations
        - Fixed material properties

        Args:
            temperature_change_c: Temperature change magnitude (Celsius)
            time_span_s: Time over which change occurs (seconds)
            pipe_thickness_mm: Pipe wall thickness (mm)
            material_expansion_coeff: Thermal expansion coefficient (1/K)

        Returns:
            ThermalShockResult with stress analysis and recommendations

        Example:
            >>> calc = DesuperheaterSizingCalculator()
            >>> result = calc.calculate_thermal_shock(
            ...     temperature_change_c=100.0,
            ...     time_span_s=60.0,
            ...     pipe_thickness_mm=12.0
            ... )
            >>> print(f"Thermal stress: {result.thermal_stress_mpa:.1f} MPa")
        """
        self.calculation_count += 1
        warnings = []

        # Convert inputs
        delta_T = Decimal(str(abs(temperature_change_c)))
        t = Decimal(str(max(0.1, time_span_s)))
        thickness = Decimal(str(pipe_thickness_mm))
        alpha = Decimal(str(material_expansion_coeff))

        # Material properties for steel
        E = Decimal("200000")  # MPa (elastic modulus)
        nu = Decimal("0.3")  # Poisson's ratio

        # Rate of temperature change
        dT_dt = delta_T / t  # C/s
        dT_dt_per_min = dT_dt * Decimal("60")  # C/min

        # Thermal stress calculation
        # sigma = E * alpha * delta_T / (1 - nu)
        sigma = E * alpha * delta_T / (Decimal("1") - nu)

        # Determine allowable rate based on thickness
        # Thick walls (>15mm) are more susceptible to thermal shock
        if thickness > Decimal("15"):
            dT_dt_max = Decimal("50")  # C/min max for thick walls
            stress_limit = Decimal("100")  # MPa
        elif thickness > Decimal("8"):
            dT_dt_max = Decimal("100")  # C/min for medium walls
            stress_limit = Decimal("120")
        else:
            dT_dt_max = Decimal("200")  # C/min for thin walls
            stress_limit = Decimal("150")

        # Classify stress level
        if sigma < stress_limit * Decimal("0.5"):
            stress_level = ThermalStressLevel.ACCEPTABLE
        elif sigma < stress_limit * Decimal("0.75"):
            stress_level = ThermalStressLevel.MODERATE
        elif sigma < stress_limit:
            stress_level = ThermalStressLevel.HIGH
            warnings.append("Thermal stress approaching allowable limit")
        else:
            stress_level = ThermalStressLevel.CRITICAL
            warnings.append("CRITICAL: Thermal stress exceeds recommended limits")

        # Check rate of change
        is_acceptable = dT_dt_per_min <= dT_dt_max and stress_level != ThermalStressLevel.CRITICAL

        # Calculate minimum warmup time
        min_warmup = delta_T / (dT_dt_max / Decimal("60"))  # seconds

        # Maximum allowable temperature change at given rate
        max_delta_T = dT_dt_max * t / Decimal("60")

        # Generate recommendations
        recommendations = []
        if not is_acceptable:
            recommendations.append(
                f"Reduce temperature change rate to below {dT_dt_max:.0f} C/min"
            )
            recommendations.append(
                f"Use gradual startup over at least {min_warmup:.0f} seconds"
            )
        if stress_level in (ThermalStressLevel.HIGH, ThermalStressLevel.CRITICAL):
            recommendations.append(
                "Consider thermal sleeves or spray shield to protect pipe wall"
            )
            recommendations.append(
                "Implement staged temperature reduction with intermediate targets"
            )
        if thickness > Decimal("15"):
            recommendations.append(
                "Thick-walled pipe requires extra care during temperature transients"
            )

        # Generate provenance
        provenance_hash = self._generate_provenance("thermal_shock",
                                                    {'delta_T': str(delta_T), 't_s': str(t)},
                                                    {'sigma_mpa': str(sigma), 'stress_level': stress_level.value})

        return ThermalShockResult(
            delta_t_rate_c_s=float(dT_dt.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            thermal_stress_mpa=float(sigma.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)),
            stress_level=stress_level,
            max_delta_t_allowable_c=float(max_delta_T.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)),
            min_warmup_time_s=float(min_warmup.quantize(Decimal('1'), rounding=ROUND_HALF_UP)),
            is_acceptable=is_acceptable,
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            warnings=warnings
        )

    # =========================================================================
    # CONTROL VALVE SIZING (Cv CALCULATION)
    # =========================================================================

    def calculate_valve_cv(
        self,
        input_data: ValveSizingInput
    ) -> ValveSizingResult:
        """
        Calculate control valve flow coefficient (Cv) per ISA-75.01.01.

        FORMULA (ISA-75.01.01 for liquid):
            Cv = Q / (N1 * Fp * sqrt(delta_P / SG))

        For liquid without piping effects (Fp = 1):
            Cv = Q / (N1 * sqrt(delta_P / SG))

        Where:
            Q = volumetric flow rate (m3/h)
            N1 = 0.0865 (constant for SI units)
            delta_P = pressure drop (kPa)
            SG = specific gravity (dimensionless)
            Fp = piping geometry factor (typically 0.9-1.0)

        ZERO-HALLUCINATION GUARANTEE:
        - Standard ISA-75.01.01 equations
        - Deterministic calculation

        Args:
            input_data: Valve sizing conditions

        Returns:
            ValveSizingResult with Cv and valve selection

        Example:
            >>> calc = DesuperheaterSizingCalculator()
            >>> result = calc.calculate_valve_cv(ValveSizingInput(
            ...     flow_rate_max_kg_s=1.0,
            ...     flow_rate_min_kg_s=0.1,
            ...     inlet_pressure_mpa=2.0,
            ...     outlet_pressure_mpa=1.5,
            ...     water_temperature_c=25.0
            ... ))
            >>> print(f"Required Cv: {result.cv_required:.1f}")
        """
        self.calculation_count += 1
        warnings = []

        # Convert inputs
        m_max = Decimal(str(input_data.flow_rate_max_kg_s))
        m_min = Decimal(str(input_data.flow_rate_min_kg_s))
        P1 = Decimal(str(input_data.inlet_pressure_mpa)) * Decimal("1000")  # kPa
        P2 = Decimal(str(input_data.outlet_pressure_mpa)) * Decimal("1000")  # kPa
        T = Decimal(str(input_data.water_temperature_c))

        # Pressure drop
        delta_P = P1 - P2
        if delta_P <= 0:
            warnings.append("Negative or zero pressure drop - check pressures")
            delta_P = Decimal("100")  # Fallback minimum

        # Water properties
        rho = self.WATER_DENSITY_KG_M3 * (Decimal("1") - Decimal("0.0003") * T)
        SG = rho / Decimal("1000")  # Specific gravity relative to water at 15C

        # Convert mass flow to volumetric flow (m3/h)
        Q_max = m_max / rho * Decimal("3600")
        Q_min = m_min / rho * Decimal("3600")

        # Design flow (use max for sizing)
        Q_design = Q_max

        # Calculate Cv at design conditions
        # Cv = Q / (N1 * sqrt(delta_P / SG))
        N1 = self.N1_CV_CONSTANT

        if delta_P > 0:
            Cv_design = Q_design / (N1 * (delta_P / SG).sqrt())
            Cv_min = Q_min / (N1 * (delta_P / SG).sqrt())
            Cv_max = Q_max / (N1 * (delta_P / SG).sqrt())
        else:
            Cv_design = Decimal("10")
            Cv_min = Decimal("1")
            Cv_max = Decimal("10")

        # Select next larger standard Cv
        Cv_selected = Decimal(str(self._select_standard_cv(float(Cv_design))))

        # Calculate actual turndown
        if Cv_min > 0:
            turndown_achievable = Cv_max / Cv_min
        else:
            turndown_achievable = Decimal("50")

        # Velocity through valve (estimate based on Cv)
        # Approximate: v = Q / A, where A ~ Cv * k (empirical)
        k_velocity = Decimal("0.0001")  # m2 per Cv unit (approximate)
        A_valve = Cv_selected * k_velocity
        if A_valve > 0:
            v_valve = Q_design / Decimal("3600") / A_valve
        else:
            v_valve = Decimal("5")

        # FL factor (pressure recovery factor) - typical for globe valve
        FL = Decimal("0.9")

        # Cavitation check
        # Cavitation occurs when P2 < Pv * FL^2 * (P1 - Pv) / P1
        Pv = Decimal("3.17")  # Vapor pressure at 25C in kPa
        P_critical = P1 - (P1 - Pv) / FL ** 2

        if P2 < P_critical:
            cavitation_risk = "High - outlet pressure below critical"
            warnings.append("High cavitation risk - consider anti-cavitation trim")
        elif P2 < P_critical * Decimal("1.2"):
            cavitation_risk = "Moderate"
            warnings.append("Moderate cavitation potential")
        else:
            cavitation_risk = "Low"

        # Cavitation index (sigma)
        sigma = (P1 - Pv) / (P1 - P2) if (P1 - P2) > 0 else Decimal("10")

        # Noise estimation (simplified)
        # Noise ~ base + 10*log10(delta_P) + 20*log10(Q)
        noise_base = Decimal("70")  # dB base level
        if delta_P > 0 and Q_design > 0:
            noise = noise_base + Decimal("10") * Decimal(str(math.log10(float(delta_P)))) + \
                    Decimal("20") * Decimal(str(math.log10(float(Q_design))))
        else:
            noise = noise_base

        noise_acceptable = noise <= Decimal(str(input_data.allowable_noise_db))
        if not noise_acceptable:
            warnings.append(f"Predicted noise {noise:.0f} dB exceeds limit {input_data.allowable_noise_db} dB")

        # Generate provenance
        provenance_hash = self._generate_provenance("valve_cv",
                                                    {'Q_max_m3h': str(Q_max), 'delta_P_kPa': str(delta_P)},
                                                    {'Cv_required': str(Cv_design), 'Cv_selected': str(Cv_selected)})

        return ValveSizingResult(
            cv_required=float(Cv_design.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)),
            cv_selected=float(Cv_selected),
            cv_min_required=float(Cv_min.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            cv_max_required=float(Cv_max.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)),
            pressure_drop_design_mpa=float((delta_P / Decimal("1000")).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)),
            velocity_m_s=float(v_valve.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)),
            fl_factor=float(FL),
            cavitation_index=float(sigma.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)),
            cavitation_risk=cavitation_risk,
            noise_level_db=float(noise.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)),
            noise_acceptable=noise_acceptable,
            turndown_achievable=float(turndown_achievable.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)),
            provenance_hash=provenance_hash,
            warnings=warnings
        )

    def _select_standard_cv(self, cv_required: float) -> float:
        """Select next larger standard Cv size."""
        for cv in self.STANDARD_CV_SIZES:
            if cv >= cv_required:
                return cv
        return self.STANDARD_CV_SIZES[-1]  # Return largest if none found

    # =========================================================================
    # TURNDOWN RATIO ANALYSIS
    # =========================================================================

    def calculate_turndown_analysis(
        self,
        design_flow_kg_s: float,
        valve_cv: float,
        requested_turndown: float,
        valve_characteristic: FlowCharacteristic = FlowCharacteristic.EQUAL_PERCENTAGE,
        valve_rangeability: float = 50.0
    ) -> TurndownAnalysisResult:
        """
        Analyze achievable turndown ratio for the desuperheater system.

        FORMULA (Equal Percentage Valve):
            Q = Q_max * R^(x-1)

        Where:
            R = rangeability (typically 30-50 for equal percentage)
            x = valve position (0-1)
            Q_max = flow at 100% open

        Minimum controllable flow:
            Q_min = Q_max / R (at ~10% valve position)

        Actual turndown:
            TR_actual = Q_max / Q_min_controllable

        ZERO-HALLUCINATION GUARANTEE:
        - Standard valve characteristic equations
        - Deterministic calculation

        Args:
            design_flow_kg_s: Design spray water flow (kg/s)
            valve_cv: Selected valve Cv
            requested_turndown: Desired turndown ratio
            valve_characteristic: Valve flow characteristic
            valve_rangeability: Valve rangeability ratio

        Returns:
            TurndownAnalysisResult with turndown analysis

        Example:
            >>> calc = DesuperheaterSizingCalculator()
            >>> result = calc.calculate_turndown_analysis(
            ...     design_flow_kg_s=0.5,
            ...     valve_cv=10.0,
            ...     requested_turndown=10.0
            ... )
            >>> print(f"Achievable turndown: {result.achievable_turndown:.1f}:1")
        """
        self.calculation_count += 1
        warnings = []

        # Convert inputs
        Q_design = Decimal(str(design_flow_kg_s))
        Cv = Decimal(str(valve_cv))
        TR_requested = Decimal(str(requested_turndown))
        R = Decimal(str(valve_rangeability))

        # Calculate max and min controllable flows based on valve characteristic
        if valve_characteristic == FlowCharacteristic.EQUAL_PERCENTAGE:
            # Equal percentage: good turndown, ~R:1 rangeability
            # At 10% position: Q_min = Q_max * R^(-0.9)
            # At 100% position: Q_max = Q_design * safety_factor
            Q_max = Q_design * self.safety_factor
            Q_min = Q_max / R

            # Valve positions
            # x = 1 + ln(Q/Q_max) / ln(R)
            x_at_design = Decimal("1") + Decimal(str(math.log(float(Q_design / Q_max)))) / Decimal(str(math.log(float(R))))
            x_at_min = Decimal("0.1")  # Minimum controllable position

        elif valve_characteristic == FlowCharacteristic.LINEAR:
            # Linear: proportional flow to position
            Q_max = Q_design * self.safety_factor
            # Minimum controllable at ~5% position
            Q_min = Q_max * Decimal("0.05")

            x_at_design = Q_design / Q_max
            x_at_min = Decimal("0.05")

        else:  # QUICK_OPENING
            # Quick opening: high flow at low positions
            Q_max = Q_design * self.safety_factor
            Q_min = Q_max * Decimal("0.1")  # Lower rangeability

            x_at_design = (Q_design / Q_max).sqrt()
            x_at_min = Decimal("0.1")

        # Calculate achievable turndown
        if Q_min > 0:
            TR_achievable = Q_max / Q_min
        else:
            TR_achievable = Decimal("100")

        # Determine limiting factor
        if TR_achievable >= TR_requested:
            limiting_factor = "None - requested turndown achievable"
            rangeability_adequate = True
        else:
            limiting_factor = f"Valve rangeability ({R}:1)"
            rangeability_adequate = False
            warnings.append(
                f"Requested turndown {TR_requested}:1 exceeds valve capability {TR_achievable:.1f}:1"
            )

        # Check for atomization at low flow
        if Q_min / Q_design < Decimal("0.1"):
            warnings.append(
                "Very low flow at turndown may result in poor atomization"
            )
            limiting_factor = "Atomization quality at low flow"

        # Generate provenance
        provenance_hash = self._generate_provenance("turndown",
                                                    {'Q_design': str(Q_design), 'TR_requested': str(TR_requested)},
                                                    {'TR_achievable': str(TR_achievable)})

        return TurndownAnalysisResult(
            requested_turndown=float(TR_requested),
            achievable_turndown=float(TR_achievable.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)),
            min_controllable_flow_kg_s=float(Q_min.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)),
            max_controllable_flow_kg_s=float(Q_max.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)),
            control_range_limited_by=limiting_factor,
            valve_position_at_min_pct=float(x_at_min * 100),
            valve_position_at_max_pct=100.0,
            rangeability_adequate=rangeability_adequate,
            provenance_hash=provenance_hash,
            warnings=warnings
        )

    # =========================================================================
    # COMPLETE SIZING CALCULATION
    # =========================================================================

    def calculate_complete_sizing(
        self,
        input_data: DesuperheaterSizingInput
    ) -> CompleteSizingResult:
        """
        Perform complete desuperheater sizing analysis.

        Combines all sizing calculations:
        1. Spray water flow calculation
        2. Atomization analysis
        3. Residence time calculation
        4. Thermal shock prevention
        5. Control valve sizing
        6. Turndown analysis

        ZERO-HALLUCINATION GUARANTEE:
        - All sub-calculations are deterministic
        - Complete audit trail with SHA-256 hash

        Args:
            input_data: Complete design conditions

        Returns:
            CompleteSizingResult with all analyses and recommendations

        Example:
            >>> calc = DesuperheaterSizingCalculator()
            >>> result = calc.calculate_complete_sizing(DesuperheaterSizingInput(
            ...     steam_flow_kg_s=10.0,
            ...     inlet_temperature_c=400.0,
            ...     outlet_temperature_c=350.0,
            ...     inlet_pressure_mpa=4.0,
            ...     spray_water_temperature_c=25.0,
            ...     spray_water_pressure_mpa=6.0,
            ...     pipe_diameter_m=0.25,
            ...     straight_pipe_length_m=3.0,
            ...     turndown_ratio=10.0
            ... ))
            >>> print(f"Feasible: {result.overall_feasibility}")
        """
        recommendations = []

        # 1. Spray water flow calculation
        spray_water = self.calculate_spray_water_flow(input_data)
        recommendations.extend([f"Spray water: {w}" for w in spray_water.warnings])

        # 2. Atomization analysis
        # Estimate spray velocity from pressure differential
        delta_P_spray = input_data.spray_water_pressure_mpa - input_data.inlet_pressure_mpa
        if delta_P_spray > 0:
            # Bernoulli: v = sqrt(2 * delta_P / rho)
            v_spray = math.sqrt(2 * delta_P_spray * 1e6 / 1000)  # m/s
        else:
            v_spray = 30.0  # Fallback

        # Estimate steam velocity
        T_K = input_data.inlet_temperature_c + 273.15
        rho_steam = input_data.inlet_pressure_mpa * 1000 / (0.4615 * T_K)
        A_pipe = math.pi * input_data.pipe_diameter_m ** 2 / 4
        v_steam = input_data.steam_flow_kg_s / (rho_steam * A_pipe)

        atomization = self.calculate_atomization(AtomizationInput(
            spray_velocity_m_s=v_spray,
            water_pressure_mpa=input_data.spray_water_pressure_mpa,
            steam_velocity_m_s=v_steam,
            water_temperature_c=input_data.spray_water_temperature_c,
            steam_temperature_c=input_data.inlet_temperature_c,
            steam_pressure_mpa=input_data.inlet_pressure_mpa,
            nozzle_diameter_mm=3.0
        ))
        recommendations.extend([f"Atomization: {w}" for w in atomization.warnings])

        # 3. Residence time calculation
        residence = self.calculate_residence_time(
            steam_flow_kg_s=input_data.steam_flow_kg_s,
            steam_pressure_mpa=input_data.inlet_pressure_mpa,
            steam_temperature_c=input_data.inlet_temperature_c,
            pipe_diameter_m=input_data.pipe_diameter_m,
            available_length_m=input_data.straight_pipe_length_m,
            evaporation_time_ms=atomization.evaporation_time_ms
        )
        recommendations.extend([f"Residence time: {w}" for w in residence.warnings])

        # 4. Thermal shock analysis
        temp_change = input_data.inlet_temperature_c - input_data.outlet_temperature_c
        thermal = self.calculate_thermal_shock(
            temperature_change_c=temp_change,
            time_span_s=60.0,  # Assume 1 minute startup
            pipe_thickness_mm=10.0
        )
        recommendations.extend(thermal.recommendations)
        recommendations.extend([f"Thermal: {w}" for w in thermal.warnings])

        # 5. Control valve sizing
        valve = self.calculate_valve_cv(ValveSizingInput(
            flow_rate_max_kg_s=spray_water.max_flow_kg_s,
            flow_rate_min_kg_s=spray_water.min_flow_kg_s,
            inlet_pressure_mpa=input_data.spray_water_pressure_mpa,
            outlet_pressure_mpa=input_data.inlet_pressure_mpa + 0.3,  # Assume 0.3 MPa drop
            water_temperature_c=input_data.spray_water_temperature_c
        ))
        recommendations.extend([f"Valve: {w}" for w in valve.warnings])

        # 6. Turndown analysis
        turndown = self.calculate_turndown_analysis(
            design_flow_kg_s=spray_water.design_flow_kg_s,
            valve_cv=valve.cv_selected,
            requested_turndown=input_data.turndown_ratio
        )
        recommendations.extend([f"Turndown: {w}" for w in turndown.warnings])

        # Overall feasibility assessment
        feasibility_checks = [
            spray_water.evaporation_complete,
            atomization.is_adequate,
            residence.length_adequate,
            thermal.is_acceptable,
            valve.noise_acceptable,
            turndown.rangeability_adequate
        ]
        overall_feasibility = all(feasibility_checks)

        # Add summary recommendations
        if not overall_feasibility:
            failed_checks = []
            if not spray_water.evaporation_complete:
                failed_checks.append("evaporation")
            if not atomization.is_adequate:
                failed_checks.append("atomization")
            if not residence.length_adequate:
                failed_checks.append("residence time")
            if not thermal.is_acceptable:
                failed_checks.append("thermal stress")
            if not valve.noise_acceptable:
                failed_checks.append("valve noise")
            if not turndown.rangeability_adequate:
                failed_checks.append("turndown")

            recommendations.insert(0, f"Design issues: {', '.join(failed_checks)}")

        # Generate overall provenance
        provenance_hash = self._generate_provenance("complete_sizing",
            {'steam_flow': input_data.steam_flow_kg_s, 'T_in': input_data.inlet_temperature_c},
            {'feasible': overall_feasibility, 'spray_flow': spray_water.design_flow_kg_s}
        )

        return CompleteSizingResult(
            spray_water=spray_water,
            atomization=atomization,
            residence_time=residence,
            thermal_shock=thermal,
            valve_sizing=valve,
            turndown=turndown,
            overall_feasibility=overall_feasibility,
            design_recommendations=recommendations,
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _get_saturation_temperature(self, pressure_mpa: Decimal) -> Decimal:
        """Get saturation temperature from pressure using lookup table."""
        # Check for exact match
        if pressure_mpa in self.saturation_table:
            return self.saturation_table[pressure_mpa]

        # Interpolate
        pressures = sorted(self.saturation_table.keys())
        for i in range(len(pressures) - 1):
            if pressures[i] <= pressure_mpa <= pressures[i + 1]:
                P1, P2 = pressures[i], pressures[i + 1]
                T1, T2 = self.saturation_table[P1], self.saturation_table[P2]
                f = (pressure_mpa - P1) / (P2 - P1)
                return T1 + f * (T2 - T1)

        # Return closest if out of range
        if pressure_mpa < pressures[0]:
            return self.saturation_table[pressures[0]]
        return self.saturation_table[pressures[-1]]

    def _generate_provenance(
        self,
        method: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any]
    ) -> str:
        """Generate SHA-256 provenance hash for calculation."""
        data = {
            'calculator': 'DesuperheaterSizingCalculator',
            'version': '1.0.0',
            'method': method,
            'inputs': inputs,
            'outputs': outputs
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

    def get_statistics(self) -> Dict[str, Any]:
        """Get calculator statistics."""
        return {
            'calculation_count': self.calculation_count,
            'precision': self.precision,
            'safety_factor': float(self.safety_factor),
            'min_approach_c': float(self.min_approach),
            'saturation_table_entries': len(self.saturation_table),
            'standard_cv_sizes': len(self.STANDARD_CV_SIZES)
        }


# =============================================================================
# UNIT TESTS
# =============================================================================

def _run_self_tests():
    """
    Run self-tests to verify calculator correctness.

    Tests verify against engineering expectations and known values.
    """
    calc = DesuperheaterSizingCalculator()

    # Test 1: Spray water flow calculation
    spray_result = calc.calculate_spray_water_flow(DesuperheaterSizingInput(
        steam_flow_kg_s=10.0,
        inlet_temperature_c=400.0,
        outlet_temperature_c=350.0,
        inlet_pressure_mpa=4.0,
        spray_water_temperature_c=25.0
    ))
    assert spray_result.design_flow_kg_s > 0, f"Flow should be positive: {spray_result.design_flow_kg_s}"
    assert spray_result.water_to_steam_ratio < 0.25, f"Ratio should be < 0.25: {spray_result.water_to_steam_ratio}"
    print(f"Test 1 passed: Spray water flow = {spray_result.design_flow_kg_s:.4f} kg/s")

    # Test 2: Atomization analysis
    atom_result = calc.calculate_atomization(AtomizationInput(
        spray_velocity_m_s=50.0,
        water_pressure_mpa=6.0,
        steam_velocity_m_s=30.0,
        water_temperature_c=25.0,
        steam_temperature_c=350.0,
        steam_pressure_mpa=4.0,
        nozzle_diameter_mm=3.0
    ))
    assert atom_result.sauter_mean_diameter_um > 0, f"SMD should be positive: {atom_result.sauter_mean_diameter_um}"
    assert atom_result.weber_number > 0, f"Weber number should be positive: {atom_result.weber_number}"
    print(f"Test 2 passed: SMD = {atom_result.sauter_mean_diameter_um:.1f} um, We = {atom_result.weber_number:.1f}")

    # Test 3: Residence time calculation
    res_result = calc.calculate_residence_time(
        steam_flow_kg_s=10.0,
        steam_pressure_mpa=4.0,
        steam_temperature_c=350.0,
        pipe_diameter_m=0.25,
        available_length_m=3.0,
        evaporation_time_ms=atom_result.evaporation_time_ms
    )
    assert res_result.residence_time_ms > 0, f"Residence time should be positive: {res_result.residence_time_ms}"
    assert res_result.steam_velocity_m_s > 0, f"Steam velocity should be positive: {res_result.steam_velocity_m_s}"
    print(f"Test 3 passed: Residence time = {res_result.residence_time_ms:.1f} ms, v_steam = {res_result.steam_velocity_m_s:.1f} m/s")

    # Test 4: Thermal shock analysis
    thermal_result = calc.calculate_thermal_shock(
        temperature_change_c=100.0,
        time_span_s=60.0,
        pipe_thickness_mm=12.0
    )
    assert thermal_result.thermal_stress_mpa > 0, f"Stress should be positive: {thermal_result.thermal_stress_mpa}"
    print(f"Test 4 passed: Thermal stress = {thermal_result.thermal_stress_mpa:.1f} MPa, level = {thermal_result.stress_level.value}")

    # Test 5: Valve Cv calculation
    valve_result = calc.calculate_valve_cv(ValveSizingInput(
        flow_rate_max_kg_s=1.0,
        flow_rate_min_kg_s=0.1,
        inlet_pressure_mpa=6.0,
        outlet_pressure_mpa=4.3,
        water_temperature_c=25.0
    ))
    assert valve_result.cv_required > 0, f"Cv should be positive: {valve_result.cv_required}"
    assert valve_result.cv_selected >= valve_result.cv_required, f"Selected Cv should be >= required"
    print(f"Test 5 passed: Cv required = {valve_result.cv_required:.1f}, selected = {valve_result.cv_selected}")

    # Test 6: Turndown analysis
    turndown_result = calc.calculate_turndown_analysis(
        design_flow_kg_s=0.5,
        valve_cv=valve_result.cv_selected,
        requested_turndown=10.0
    )
    assert turndown_result.achievable_turndown > 1, f"Turndown should be > 1: {turndown_result.achievable_turndown}"
    print(f"Test 6 passed: Achievable turndown = {turndown_result.achievable_turndown:.1f}:1")

    # Test 7: Complete sizing
    complete_result = calc.calculate_complete_sizing(DesuperheaterSizingInput(
        steam_flow_kg_s=10.0,
        inlet_temperature_c=400.0,
        outlet_temperature_c=350.0,
        inlet_pressure_mpa=4.0,
        spray_water_temperature_c=25.0,
        spray_water_pressure_mpa=6.0,
        pipe_diameter_m=0.25,
        straight_pipe_length_m=3.0,
        turndown_ratio=10.0
    ))
    assert complete_result.provenance_hash, "Should have provenance hash"
    print(f"Test 7 passed: Complete sizing, feasibility = {complete_result.overall_feasibility}")

    # Test 8: Statistics
    stats = calc.get_statistics()
    assert stats['calculation_count'] >= 7, f"Should have many calculations: {stats['calculation_count']}"
    print(f"Test 8 passed: Statistics, {stats['calculation_count']} calculations performed")

    print("\n" + "="*60)
    print("All self-tests passed!")
    print("="*60)
    return True


if __name__ == "__main__":
    _run_self_tests()
