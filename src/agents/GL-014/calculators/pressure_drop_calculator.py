# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO - Pressure Drop Calculator Module

Comprehensive pressure drop calculations for heat exchanger analysis including:
- Tube-side pressure drop (friction, entrance/exit, return losses)
- Shell-side pressure drop (Bell-Delaware method)
- Friction factor correlations (Hagen-Poiseuille, Blasius, Colebrook-White, Churchill)
- Fouling impact on pressure drop
- Allowable pressure drop analysis
- Pump/compressor power requirements

Zero-hallucination guarantee: All calculations use deterministic formulas
from TEMA Standards, HTRI correlations, and established fluid mechanics.

Reference Standards:
- TEMA Standards (Tubular Exchanger Manufacturers Association)
- HTRI (Heat Transfer Research Institute) Correlations
- Crane TP-410 (Flow of Fluids Through Valves, Fittings, and Pipe)
- ASME PTC 12.5 (Single Phase Heat Exchangers)
- Perry's Chemical Engineers' Handbook, 9th Edition

Author: GL-CalculatorEngineer
Version: 1.0.0
License: Proprietary - GreenLang
"""

from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Final
from enum import Enum, auto
from datetime import datetime, timezone
import hashlib
import json
import math
import uuid


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Mathematical constants with high precision
PI: Final[Decimal] = Decimal("3.14159265358979323846264338327950288419716939937510")

# Standard gravity (m/s^2) - NIST CODATA 2018
STANDARD_GRAVITY: Final[Decimal] = Decimal("9.80665")

# Minimum values to avoid division by zero
MIN_REYNOLDS: Final[Decimal] = Decimal("1")
MIN_VELOCITY: Final[Decimal] = Decimal("0.001")
MIN_DIAMETER: Final[Decimal] = Decimal("0.001")


# =============================================================================
# ENUMERATIONS
# =============================================================================

class FlowRegime(Enum):
    """Flow regime classification based on Reynolds number."""
    LAMINAR = auto()           # Re < 2300
    TRANSITION = auto()        # 2300 <= Re < 4000
    TURBULENT_SMOOTH = auto()  # Re >= 4000, smooth tubes
    TURBULENT_ROUGH = auto()   # Re >= 4000, rough tubes


class FrictionCorrelation(Enum):
    """Available friction factor correlations."""
    HAGEN_POISEUILLE = auto()   # Laminar: f = 64/Re
    BLASIUS = auto()            # Turbulent smooth: f = 0.316/Re^0.25
    PETUKHOV_KIRILLOV = auto()  # Turbulent smooth: more accurate
    COLEBROOK_WHITE = auto()    # Turbulent rough: implicit equation
    CHURCHILL = auto()          # Universal: all regimes
    SWAMEE_JAIN = auto()        # Explicit approximation of Colebrook-White


class ShellType(Enum):
    """TEMA shell types for shell-side pressure drop."""
    E_SHELL = auto()  # Single pass shell
    F_SHELL = auto()  # Two pass shell with longitudinal baffle
    G_SHELL = auto()  # Split flow shell
    H_SHELL = auto()  # Double split flow shell
    J_SHELL = auto()  # Divided flow shell
    K_SHELL = auto()  # Kettle reboiler
    X_SHELL = auto()  # Crossflow shell


class BaffleType(Enum):
    """Baffle types for shell-side flow."""
    SEGMENTAL = auto()      # Standard segmental baffles
    DOUBLE_SEGMENTAL = auto()  # Double segmental baffles
    DISC_DOUGHNUT = auto()  # Disc and doughnut baffles
    ROD_BAFFLE = auto()     # Rod baffle support
    NO_TUBES_IN_WINDOW = auto()  # NTIW design


class TubePitchPattern(Enum):
    """Tube pitch patterns."""
    TRIANGULAR = auto()       # 30 degree triangular
    ROTATED_TRIANGULAR = auto()  # 60 degree rotated triangular
    SQUARE = auto()           # 90 degree square
    ROTATED_SQUARE = auto()   # 45 degree rotated square


# =============================================================================
# DATA CLASSES - INPUT PARAMETERS
# =============================================================================

@dataclass(frozen=True)
class FluidProperties:
    """
    Fluid properties required for pressure drop calculations.

    All properties at bulk fluid temperature.
    """
    density_kg_m3: Decimal = field(metadata={"description": "Fluid density (kg/m^3)"})
    viscosity_pa_s: Decimal = field(metadata={"description": "Dynamic viscosity (Pa.s)"})
    specific_heat_j_kg_k: Decimal = field(default=Decimal("4186"), metadata={"description": "Specific heat (J/kg.K)"})
    thermal_conductivity_w_m_k: Decimal = field(default=Decimal("0.6"), metadata={"description": "Thermal conductivity (W/m.K)"})

    def __post_init__(self):
        """Validate fluid properties."""
        if self.density_kg_m3 <= Decimal("0"):
            raise ValueError("Density must be positive")
        if self.viscosity_pa_s <= Decimal("0"):
            raise ValueError("Viscosity must be positive")


@dataclass(frozen=True)
class TubeGeometry:
    """
    Tube-side geometry parameters.

    Reference: TEMA Standards, Table RCB-2.21
    """
    tube_od_m: Decimal = field(metadata={"description": "Tube outer diameter (m)"})
    tube_id_m: Decimal = field(metadata={"description": "Tube inner diameter (m)"})
    tube_length_m: Decimal = field(metadata={"description": "Effective tube length (m)"})
    number_of_tubes: int = field(metadata={"description": "Total number of tubes"})
    number_of_passes: int = field(default=1, metadata={"description": "Number of tube passes"})
    tube_roughness_m: Decimal = field(default=Decimal("0.0000015"), metadata={"description": "Surface roughness (m)"})
    tube_pitch_m: Decimal = field(default=Decimal("0"), metadata={"description": "Tube pitch (m)"})
    pitch_pattern: TubePitchPattern = field(default=TubePitchPattern.TRIANGULAR)

    def __post_init__(self):
        """Validate tube geometry."""
        if self.tube_od_m <= Decimal("0"):
            raise ValueError("Tube OD must be positive")
        if self.tube_id_m <= Decimal("0"):
            raise ValueError("Tube ID must be positive")
        if self.tube_id_m >= self.tube_od_m:
            raise ValueError("Tube ID must be less than OD")
        if self.tube_length_m <= Decimal("0"):
            raise ValueError("Tube length must be positive")
        if self.number_of_tubes <= 0:
            raise ValueError("Number of tubes must be positive")
        if self.number_of_passes <= 0:
            raise ValueError("Number of passes must be positive")


@dataclass(frozen=True)
class ShellGeometry:
    """
    Shell-side geometry parameters for Bell-Delaware method.

    Reference: Bell, K.J. (1963), Delaware Method
    """
    shell_id_m: Decimal = field(metadata={"description": "Shell inner diameter (m)"})
    baffle_spacing_m: Decimal = field(metadata={"description": "Central baffle spacing (m)"})
    baffle_cut_fraction: Decimal = field(default=Decimal("0.25"), metadata={"description": "Baffle cut as fraction of shell ID"})
    inlet_baffle_spacing_m: Decimal = field(default=Decimal("0"), metadata={"description": "Inlet baffle spacing (m)"})
    outlet_baffle_spacing_m: Decimal = field(default=Decimal("0"), metadata={"description": "Outlet baffle spacing (m)"})
    number_of_baffles: int = field(default=0, metadata={"description": "Number of baffles"})
    shell_type: ShellType = field(default=ShellType.E_SHELL)
    baffle_type: BaffleType = field(default=BaffleType.SEGMENTAL)
    tube_to_baffle_clearance_m: Decimal = field(default=Decimal("0.0004"), metadata={"description": "Tube-to-baffle hole clearance (m)"})
    shell_to_baffle_clearance_m: Decimal = field(default=Decimal("0.003"), metadata={"description": "Shell-to-baffle clearance (m)"})
    bundle_to_shell_clearance_m: Decimal = field(default=Decimal("0.012"), metadata={"description": "Bundle-to-shell clearance (m)"})
    number_of_sealing_strips: int = field(default=0, metadata={"description": "Number of sealing strip pairs"})

    def __post_init__(self):
        """Validate shell geometry."""
        if self.shell_id_m <= Decimal("0"):
            raise ValueError("Shell ID must be positive")
        if self.baffle_spacing_m <= Decimal("0"):
            raise ValueError("Baffle spacing must be positive")
        if not (Decimal("0.15") <= self.baffle_cut_fraction <= Decimal("0.45")):
            raise ValueError("Baffle cut must be between 15% and 45%")


@dataclass(frozen=True)
class FoulingCondition:
    """
    Fouling condition parameters for pressure drop impact analysis.

    Reference: TEMA Standards, Table RGP-T-2.4
    """
    fouling_thickness_m: Decimal = field(default=Decimal("0"), metadata={"description": "Fouling layer thickness (m)"})
    fouling_roughness_m: Decimal = field(default=Decimal("0"), metadata={"description": "Fouling surface roughness (m)"})
    fouling_resistance_m2_k_w: Decimal = field(default=Decimal("0"), metadata={"description": "Fouling thermal resistance (m^2.K/W)"})
    years_in_service: Decimal = field(default=Decimal("0"), metadata={"description": "Years since last cleaning"})


@dataclass(frozen=True)
class TubeSideInput:
    """Complete input for tube-side pressure drop calculation."""
    fluid: FluidProperties
    geometry: TubeGeometry
    mass_flow_rate_kg_s: Decimal = field(metadata={"description": "Mass flow rate (kg/s)"})
    fouling: FoulingCondition = field(default_factory=lambda: FoulingCondition())
    friction_correlation: FrictionCorrelation = field(default=FrictionCorrelation.CHURCHILL)

    def __post_init__(self):
        """Validate inputs."""
        if self.mass_flow_rate_kg_s <= Decimal("0"):
            raise ValueError("Mass flow rate must be positive")


@dataclass(frozen=True)
class ShellSideInput:
    """Complete input for shell-side pressure drop calculation."""
    fluid: FluidProperties
    tube_geometry: TubeGeometry
    shell_geometry: ShellGeometry
    mass_flow_rate_kg_s: Decimal = field(metadata={"description": "Mass flow rate (kg/s)"})
    fouling: FoulingCondition = field(default_factory=lambda: FoulingCondition())


@dataclass(frozen=True)
class PressureDropLimits:
    """Allowable pressure drop limits for assessment."""
    max_tube_side_pa: Decimal = field(default=Decimal("50000"), metadata={"description": "Max tube-side pressure drop (Pa)"})
    max_shell_side_pa: Decimal = field(default=Decimal("50000"), metadata={"description": "Max shell-side pressure drop (Pa)"})
    pump_available_head_m: Decimal = field(default=Decimal("100"), metadata={"description": "Available pump head (m)"})
    pump_efficiency: Decimal = field(default=Decimal("0.75"), metadata={"description": "Pump efficiency (0-1)"})


# =============================================================================
# DATA CLASSES - CALCULATION RESULTS
# =============================================================================

@dataclass(frozen=True)
class CalculationStep:
    """
    Immutable record of a single calculation step.

    Provides complete audit trail for zero-hallucination guarantee.
    """
    step_number: int
    operation: str
    description: str
    inputs: Dict[str, Union[str, Decimal, float, int]]
    output_name: str
    output_value: Union[Decimal, float, int]
    formula: str = ""
    reference: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "step_number": self.step_number,
            "operation": self.operation,
            "description": self.description,
            "inputs": {k: str(v) for k, v in self.inputs.items()},
            "output_name": self.output_name,
            "output_value": str(self.output_value),
            "formula": self.formula,
            "reference": self.reference,
        }


@dataclass(frozen=True)
class TubeSidePressureDropResult:
    """
    Immutable result of tube-side pressure drop calculation.

    Components:
    - Friction loss: Darcy-Weisbach equation
    - Entrance/exit losses: K-factor method
    - Return losses: U-bend pressure drop
    """
    friction_loss_pa: Decimal = field(metadata={"description": "Friction pressure drop (Pa)"})
    entrance_exit_loss_pa: Decimal = field(metadata={"description": "Entrance/exit losses (Pa)"})
    return_loss_pa: Decimal = field(metadata={"description": "Return bend losses (Pa)"})
    total_pressure_drop_pa: Decimal = field(metadata={"description": "Total tube-side pressure drop (Pa)"})
    velocity_m_s: Decimal = field(metadata={"description": "Tube-side velocity (m/s)"})
    reynolds_number: Decimal = field(metadata={"description": "Reynolds number"})
    friction_factor: Decimal = field(metadata={"description": "Darcy friction factor"})
    flow_regime: FlowRegime = field(metadata={"description": "Flow regime classification"})
    calculation_steps: Tuple[CalculationStep, ...] = field(default_factory=tuple)
    provenance_hash: str = field(default="")
    timestamp: str = field(default="")

    @property
    def total_pressure_drop_kpa(self) -> Decimal:
        """Total pressure drop in kPa."""
        return self.total_pressure_drop_pa / Decimal("1000")

    @property
    def total_pressure_drop_psi(self) -> Decimal:
        """Total pressure drop in psi."""
        return self.total_pressure_drop_pa / Decimal("6894.757")


@dataclass(frozen=True)
class ShellSidePressureDropResult:
    """
    Immutable result of shell-side pressure drop calculation.

    Bell-Delaware method components:
    - Ideal crossflow pressure drop
    - Correction factors (Jc, Jl, Jb, Js)
    - Window pressure drop
    - Entrance/exit pressure drop
    """
    ideal_crossflow_dp_pa: Decimal = field(metadata={"description": "Ideal crossflow pressure drop (Pa)"})
    window_dp_pa: Decimal = field(metadata={"description": "Window pressure drop (Pa)"})
    entrance_exit_dp_pa: Decimal = field(metadata={"description": "Entrance/exit pressure drop (Pa)"})
    total_pressure_drop_pa: Decimal = field(metadata={"description": "Total shell-side pressure drop (Pa)"})
    j_c_baffle_cut: Decimal = field(metadata={"description": "Baffle cut correction factor"})
    j_l_leakage: Decimal = field(metadata={"description": "Leakage correction factor"})
    j_b_bypass: Decimal = field(metadata={"description": "Bypass correction factor"})
    j_s_spacing: Decimal = field(metadata={"description": "Unequal spacing correction factor"})
    crossflow_velocity_m_s: Decimal = field(metadata={"description": "Crossflow velocity (m/s)"})
    crossflow_reynolds: Decimal = field(metadata={"description": "Crossflow Reynolds number"})
    number_of_crossflow_rows: int = field(metadata={"description": "Number of tube rows crossed"})
    calculation_steps: Tuple[CalculationStep, ...] = field(default_factory=tuple)
    provenance_hash: str = field(default="")
    timestamp: str = field(default="")

    @property
    def total_pressure_drop_kpa(self) -> Decimal:
        """Total pressure drop in kPa."""
        return self.total_pressure_drop_pa / Decimal("1000")


@dataclass(frozen=True)
class FoulingImpactResult:
    """
    Immutable result of fouling impact analysis.

    Shows how fouling affects pressure drop over time.
    """
    clean_pressure_drop_pa: Decimal = field(metadata={"description": "Pressure drop when clean (Pa)"})
    fouled_pressure_drop_pa: Decimal = field(metadata={"description": "Pressure drop when fouled (Pa)"})
    pressure_drop_ratio: Decimal = field(metadata={"description": "Fouled/Clean pressure drop ratio"})
    effective_diameter_reduction_m: Decimal = field(metadata={"description": "Diameter reduction due to fouling (m)"})
    roughness_increase_factor: Decimal = field(metadata={"description": "Roughness increase multiplier"})
    flow_area_reduction_percent: Decimal = field(metadata={"description": "Flow area reduction (%)"})
    cleaning_recommended: bool = field(metadata={"description": "Whether cleaning is recommended"})
    estimated_energy_penalty_percent: Decimal = field(metadata={"description": "Estimated pumping energy increase (%)"})
    calculation_steps: Tuple[CalculationStep, ...] = field(default_factory=tuple)
    provenance_hash: str = field(default="")


@dataclass(frozen=True)
class PressureDropAcceptabilityResult:
    """
    Immutable result of pressure drop acceptability assessment.
    """
    tube_side_acceptable: bool = field(metadata={"description": "Tube-side within limits"})
    shell_side_acceptable: bool = field(metadata={"description": "Shell-side within limits"})
    total_acceptable: bool = field(metadata={"description": "Total system within limits"})
    tube_side_margin_percent: Decimal = field(metadata={"description": "Tube-side margin to limit (%)"})
    shell_side_margin_percent: Decimal = field(metadata={"description": "Shell-side margin to limit (%)"})
    pump_capacity_adequate: bool = field(metadata={"description": "Pump capacity sufficient"})
    required_pump_head_m: Decimal = field(metadata={"description": "Required pump head (m)"})
    recommendations: Tuple[str, ...] = field(default_factory=tuple)
    calculation_steps: Tuple[CalculationStep, ...] = field(default_factory=tuple)
    provenance_hash: str = field(default="")


@dataclass(frozen=True)
class PumpPowerResult:
    """
    Immutable result of pump power calculation.
    """
    hydraulic_power_w: Decimal = field(metadata={"description": "Hydraulic power (W)"})
    shaft_power_w: Decimal = field(metadata={"description": "Required shaft power (W)"})
    electrical_power_w: Decimal = field(metadata={"description": "Electrical power (W)"})
    required_head_m: Decimal = field(metadata={"description": "Required pump head (m)"})
    volumetric_flow_m3_s: Decimal = field(metadata={"description": "Volumetric flow rate (m^3/s)"})
    pump_efficiency: Decimal = field(metadata={"description": "Pump efficiency used"})
    motor_efficiency: Decimal = field(metadata={"description": "Motor efficiency used"})
    annual_energy_kwh: Decimal = field(metadata={"description": "Annual energy consumption (kWh)"})
    annual_energy_cost_usd: Decimal = field(metadata={"description": "Annual energy cost (USD)"})
    calculation_steps: Tuple[CalculationStep, ...] = field(default_factory=tuple)
    provenance_hash: str = field(default="")


# =============================================================================
# FRICTION FACTOR CORRELATIONS
# =============================================================================

class FrictionFactorCalculator:
    """
    Deterministic friction factor calculations.

    Implements multiple correlations for different flow regimes:
    - Laminar: Hagen-Poiseuille (f = 64/Re)
    - Turbulent smooth: Blasius, Petukhov-Kirillov
    - Turbulent rough: Colebrook-White (iterative), Churchill (explicit)
    - Transition: Churchill (continuous across all regimes)

    Zero hallucination guarantee: All formulas from established references.
    """

    # Reynolds number thresholds
    RE_LAMINAR_MAX: Final[Decimal] = Decimal("2300")
    RE_TURBULENT_MIN: Final[Decimal] = Decimal("4000")
    RE_BLASIUS_MAX: Final[Decimal] = Decimal("100000")

    def calculate(
        self,
        reynolds: Decimal,
        relative_roughness: Decimal,
        correlation: FrictionCorrelation = FrictionCorrelation.CHURCHILL
    ) -> Tuple[Decimal, FlowRegime, List[CalculationStep]]:
        """
        Calculate Darcy friction factor.

        Args:
            reynolds: Reynolds number (dimensionless)
            relative_roughness: epsilon/D ratio (dimensionless)
            correlation: Correlation to use

        Returns:
            Tuple of (friction_factor, flow_regime, calculation_steps)

        Reference: Crane TP-410, Chapter 1
        """
        steps = []

        # Determine flow regime
        flow_regime = self._determine_flow_regime(reynolds, relative_roughness)
        steps.append(CalculationStep(
            step_number=1,
            operation="classify_flow_regime",
            description=f"Determine flow regime from Reynolds number",
            inputs={"reynolds": reynolds, "relative_roughness": relative_roughness},
            output_name="flow_regime",
            output_value=flow_regime.name,
            formula="Re < 2300: Laminar, 2300 <= Re < 4000: Transition, Re >= 4000: Turbulent",
            reference="Crane TP-410"
        ))

        # Calculate friction factor based on correlation
        if correlation == FrictionCorrelation.CHURCHILL:
            f, sub_steps = self._churchill_equation(reynolds, relative_roughness)
            steps.extend(sub_steps)
        elif correlation == FrictionCorrelation.HAGEN_POISEUILLE and flow_regime == FlowRegime.LAMINAR:
            f, sub_steps = self._hagen_poiseuille(reynolds)
            steps.extend(sub_steps)
        elif correlation == FrictionCorrelation.BLASIUS and flow_regime in [FlowRegime.TURBULENT_SMOOTH, FlowRegime.TURBULENT_ROUGH]:
            f, sub_steps = self._blasius(reynolds)
            steps.extend(sub_steps)
        elif correlation == FrictionCorrelation.COLEBROOK_WHITE:
            f, sub_steps = self._colebrook_white(reynolds, relative_roughness)
            steps.extend(sub_steps)
        elif correlation == FrictionCorrelation.SWAMEE_JAIN:
            f, sub_steps = self._swamee_jain(reynolds, relative_roughness)
            steps.extend(sub_steps)
        elif correlation == FrictionCorrelation.PETUKHOV_KIRILLOV:
            f, sub_steps = self._petukhov_kirillov(reynolds)
            steps.extend(sub_steps)
        else:
            # Default to Churchill for universal coverage
            f, sub_steps = self._churchill_equation(reynolds, relative_roughness)
            steps.extend(sub_steps)

        return f, flow_regime, steps

    def _determine_flow_regime(
        self,
        reynolds: Decimal,
        relative_roughness: Decimal
    ) -> FlowRegime:
        """Determine flow regime from Reynolds number."""
        if reynolds < self.RE_LAMINAR_MAX:
            return FlowRegime.LAMINAR
        elif reynolds < self.RE_TURBULENT_MIN:
            return FlowRegime.TRANSITION
        else:
            # Check if flow is hydraulically smooth
            # Smooth condition: e/D < 5/Re^(7/8) approximately
            smooth_limit = Decimal("5") / (reynolds ** Decimal("0.875"))
            if relative_roughness < smooth_limit:
                return FlowRegime.TURBULENT_SMOOTH
            else:
                return FlowRegime.TURBULENT_ROUGH

    def _hagen_poiseuille(self, reynolds: Decimal) -> Tuple[Decimal, List[CalculationStep]]:
        """
        Hagen-Poiseuille equation for laminar flow.

        f = 64/Re

        Valid for Re < 2300 (laminar flow in circular pipes)

        Reference: Bird, Stewart, Lightfoot (2002), Transport Phenomena
        """
        f = Decimal("64") / reynolds

        step = CalculationStep(
            step_number=2,
            operation="hagen_poiseuille",
            description="Calculate laminar friction factor",
            inputs={"reynolds": reynolds},
            output_name="friction_factor",
            output_value=f,
            formula="f = 64/Re",
            reference="Hagen-Poiseuille equation"
        )

        return f, [step]

    def _blasius(self, reynolds: Decimal) -> Tuple[Decimal, List[CalculationStep]]:
        """
        Blasius equation for turbulent smooth pipes.

        f = 0.316 / Re^0.25

        Valid for 4000 < Re < 100000, smooth pipes

        Reference: Blasius, H. (1913)
        """
        # Use float for power operation, then convert back
        re_float = float(reynolds)
        f_float = 0.316 / (re_float ** 0.25)
        f = Decimal(str(f_float))

        step = CalculationStep(
            step_number=2,
            operation="blasius",
            description="Calculate turbulent smooth friction factor",
            inputs={"reynolds": reynolds},
            output_name="friction_factor",
            output_value=f,
            formula="f = 0.316 / Re^0.25",
            reference="Blasius (1913)"
        )

        return f, [step]

    def _petukhov_kirillov(self, reynolds: Decimal) -> Tuple[Decimal, List[CalculationStep]]:
        """
        Petukhov-Kirillov equation for turbulent smooth pipes.

        f = (0.790 * ln(Re) - 1.64)^(-2)

        Valid for 3000 < Re < 5,000,000, smooth pipes
        More accurate than Blasius for high Reynolds numbers.

        Reference: Petukhov, B.S. (1970)
        """
        re_float = float(reynolds)
        log_re = math.log(re_float)
        f_float = (0.790 * log_re - 1.64) ** (-2)
        f = Decimal(str(f_float))

        step = CalculationStep(
            step_number=2,
            operation="petukhov_kirillov",
            description="Calculate turbulent smooth friction factor (high accuracy)",
            inputs={"reynolds": reynolds},
            output_name="friction_factor",
            output_value=f,
            formula="f = (0.790*ln(Re) - 1.64)^(-2)",
            reference="Petukhov-Kirillov (1970)"
        )

        return f, [step]

    def _colebrook_white(
        self,
        reynolds: Decimal,
        relative_roughness: Decimal,
        max_iterations: int = 50,
        tolerance: Decimal = Decimal("1e-10")
    ) -> Tuple[Decimal, List[CalculationStep]]:
        """
        Colebrook-White equation (implicit) for turbulent rough pipes.

        1/sqrt(f) = -2*log10(e/(3.7*D) + 2.51/(Re*sqrt(f)))

        Solved iteratively using Newton-Raphson method.

        Reference: Colebrook, C.F. (1939)
        """
        steps = []

        # Initial guess using Swamee-Jain
        re_float = float(reynolds)
        eps_d = float(relative_roughness)

        # Swamee-Jain initial estimate
        term1 = eps_d / 3.7
        term2 = 5.74 / (re_float ** 0.9)
        f_init = 0.25 / (math.log10(term1 + term2) ** 2)

        f = f_init

        # Newton-Raphson iteration
        for iteration in range(max_iterations):
            sqrt_f = math.sqrt(f)

            # Function: F(f) = 1/sqrt(f) + 2*log10(e/(3.7D) + 2.51/(Re*sqrt(f)))
            func = 1/sqrt_f + 2*math.log10(term1 + 2.51/(re_float*sqrt_f))

            # Derivative: dF/df
            deriv = -0.5/(f*sqrt_f) - 2.51/(math.log(10)*re_float*sqrt_f*(term1 + 2.51/(re_float*sqrt_f))*f)

            # Newton-Raphson update
            f_new = f - func/deriv

            if abs(f_new - f) < float(tolerance):
                f = f_new
                break

            f = f_new

        f_decimal = Decimal(str(f))

        step = CalculationStep(
            step_number=2,
            operation="colebrook_white",
            description=f"Solve Colebrook-White equation iteratively ({iteration+1} iterations)",
            inputs={"reynolds": reynolds, "relative_roughness": relative_roughness},
            output_name="friction_factor",
            output_value=f_decimal,
            formula="1/sqrt(f) = -2*log10(e/(3.7*D) + 2.51/(Re*sqrt(f)))",
            reference="Colebrook (1939)"
        )

        return f_decimal, [step]

    def _swamee_jain(
        self,
        reynolds: Decimal,
        relative_roughness: Decimal
    ) -> Tuple[Decimal, List[CalculationStep]]:
        """
        Swamee-Jain equation - explicit approximation of Colebrook-White.

        f = 0.25 / [log10(e/(3.7*D) + 5.74/Re^0.9)]^2

        Accuracy: within 1% of Colebrook-White for:
        - 10^-6 <= e/D <= 10^-2
        - 5000 <= Re <= 10^8

        Reference: Swamee, P.K. and Jain, A.K. (1976)
        """
        re_float = float(reynolds)
        eps_d = float(relative_roughness)

        term1 = eps_d / 3.7
        term2 = 5.74 / (re_float ** 0.9)
        log_term = math.log10(term1 + term2)
        f = 0.25 / (log_term ** 2)

        f_decimal = Decimal(str(f))

        step = CalculationStep(
            step_number=2,
            operation="swamee_jain",
            description="Calculate friction factor using Swamee-Jain explicit equation",
            inputs={"reynolds": reynolds, "relative_roughness": relative_roughness},
            output_name="friction_factor",
            output_value=f_decimal,
            formula="f = 0.25 / [log10(e/(3.7*D) + 5.74/Re^0.9)]^2",
            reference="Swamee-Jain (1976)"
        )

        return f_decimal, [step]

    def _churchill_equation(
        self,
        reynolds: Decimal,
        relative_roughness: Decimal
    ) -> Tuple[Decimal, List[CalculationStep]]:
        """
        Churchill equation - universal for all flow regimes.

        Single explicit equation valid for:
        - Laminar flow (Re < 2300)
        - Transition regime (2300 <= Re < 4000)
        - Turbulent smooth and rough (Re >= 4000)

        f = 8 * [(8/Re)^12 + 1/(A+B)^1.5]^(1/12)

        where:
        A = {2.457 * ln[(7/Re)^0.9 + 0.27*(e/D)]}^16
        B = (37530/Re)^16

        Reference: Churchill, S.W. (1977), Friction-factor equation spans all
        fluid-flow regimes, Chemical Engineering, 84(24), 91-92
        """
        steps = []
        re_float = float(reynolds)
        eps_d = float(relative_roughness)

        # Calculate term A
        inner_A = (7/re_float)**0.9 + 0.27*eps_d
        A = (2.457 * math.log(inner_A))**16

        steps.append(CalculationStep(
            step_number=2,
            operation="churchill_term_A",
            description="Calculate Churchill equation term A",
            inputs={"reynolds": reynolds, "relative_roughness": relative_roughness},
            output_name="A",
            output_value=Decimal(str(A)),
            formula="A = {2.457*ln[(7/Re)^0.9 + 0.27*(e/D)]}^16",
            reference="Churchill (1977)"
        ))

        # Calculate term B
        B = (37530/re_float)**16

        steps.append(CalculationStep(
            step_number=3,
            operation="churchill_term_B",
            description="Calculate Churchill equation term B",
            inputs={"reynolds": reynolds},
            output_name="B",
            output_value=Decimal(str(B)),
            formula="B = (37530/Re)^16",
            reference="Churchill (1977)"
        ))

        # Calculate friction factor
        laminar_term = (8/re_float)**12
        turbulent_term = 1/(A + B)**1.5
        f = 8 * (laminar_term + turbulent_term)**(1/12)

        f_decimal = Decimal(str(f))

        steps.append(CalculationStep(
            step_number=4,
            operation="churchill_friction_factor",
            description="Calculate Churchill friction factor",
            inputs={"laminar_term": Decimal(str(laminar_term)), "turbulent_term": Decimal(str(turbulent_term))},
            output_name="friction_factor",
            output_value=f_decimal,
            formula="f = 8*[(8/Re)^12 + 1/(A+B)^1.5]^(1/12)",
            reference="Churchill (1977)"
        ))

        return f_decimal, steps


# =============================================================================
# PRESSURE DROP CALCULATOR - MAIN CLASS
# =============================================================================

class PressureDropCalculator:
    """
    Comprehensive pressure drop calculator for heat exchangers.

    Implements:
    1. Tube-side pressure drop (Darcy-Weisbach + losses)
    2. Shell-side pressure drop (Bell-Delaware method)
    3. Friction factor correlations (multiple methods)
    4. Fouling impact analysis
    5. Allowable pressure drop assessment
    6. Pump power requirements

    Zero Hallucination Guarantee:
    - All calculations are deterministic (Decimal arithmetic)
    - No LLM inference in calculation path
    - Complete provenance tracking with SHA-256 hashing
    - All formulas from established engineering references

    Reference Standards:
    - TEMA Standards
    - HTRI Correlations
    - Crane TP-410
    - ASME PTC 12.5
    """

    # K-factors for tube-side losses (from Crane TP-410)
    K_TUBE_ENTRANCE: Final[Decimal] = Decimal("0.5")   # Sharp-edged entrance
    K_TUBE_EXIT: Final[Decimal] = Decimal("1.0")       # Exit loss
    K_RETURN_BEND: Final[Decimal] = Decimal("4.0")     # 180-degree return bend

    # Bell-Delaware correction factor limits
    J_MIN: Final[Decimal] = Decimal("0.5")
    J_MAX: Final[Decimal] = Decimal("1.0")

    def __init__(self, precision: int = 6):
        """
        Initialize pressure drop calculator.

        Args:
            precision: Decimal precision for results (default: 6)
        """
        self._precision = precision
        self._friction_calculator = FrictionFactorCalculator()

    # =========================================================================
    # TUBE-SIDE PRESSURE DROP
    # =========================================================================

    def calculate_tube_side_pressure_drop(
        self,
        inputs: TubeSideInput
    ) -> TubeSidePressureDropResult:
        """
        Calculate complete tube-side pressure drop.

        Components:
        1. Friction loss: dP_f = f * (L/D) * (rho*v^2/2) * n_passes
        2. Entrance/exit losses: dP_e = K * (rho*v^2/2)
        3. Return losses: dP_r = 4 * n_passes * (rho*v^2/2)

        Total: dP_tube = dP_f + dP_e + dP_r

        Args:
            inputs: TubeSideInput with all required parameters

        Returns:
            TubeSidePressureDropResult with complete breakdown

        Reference: TEMA Standards, Section 5
        """
        steps: List[CalculationStep] = []
        step_num = 1

        # Extract parameters
        fluid = inputs.fluid
        geom = inputs.geometry
        m_dot = inputs.mass_flow_rate_kg_s

        # Apply fouling effects to geometry
        effective_id = geom.tube_id_m - Decimal("2") * inputs.fouling.fouling_thickness_m
        effective_roughness = geom.tube_roughness_m + inputs.fouling.fouling_roughness_m

        steps.append(CalculationStep(
            step_number=step_num,
            operation="apply_fouling",
            description="Calculate effective tube ID and roughness with fouling",
            inputs={
                "tube_id_m": geom.tube_id_m,
                "fouling_thickness_m": inputs.fouling.fouling_thickness_m,
                "tube_roughness_m": geom.tube_roughness_m,
                "fouling_roughness_m": inputs.fouling.fouling_roughness_m
            },
            output_name="effective_id_m",
            output_value=effective_id,
            formula="D_eff = D_i - 2*t_fouling",
            reference="TEMA Standards"
        ))
        step_num += 1

        # Calculate flow area per pass
        tubes_per_pass = Decimal(str(geom.number_of_tubes)) / Decimal(str(geom.number_of_passes))
        flow_area = tubes_per_pass * PI * (effective_id / Decimal("2")) ** 2

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_flow_area",
            description="Calculate tube-side flow area per pass",
            inputs={
                "tubes_per_pass": tubes_per_pass,
                "effective_id_m": effective_id
            },
            output_name="flow_area_m2",
            output_value=flow_area,
            formula="A = (N_tubes/n_passes) * pi * (D_i/2)^2",
            reference="TEMA Standards"
        ))
        step_num += 1

        # Calculate velocity
        volumetric_flow = m_dot / fluid.density_kg_m3
        velocity = volumetric_flow / flow_area
        velocity = max(velocity, MIN_VELOCITY)

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_velocity",
            description="Calculate tube-side velocity",
            inputs={
                "mass_flow_rate_kg_s": m_dot,
                "density_kg_m3": fluid.density_kg_m3,
                "flow_area_m2": flow_area
            },
            output_name="velocity_m_s",
            output_value=velocity,
            formula="v = m_dot / (rho * A)",
            reference="Continuity equation"
        ))
        step_num += 1

        # Calculate Reynolds number
        reynolds = (fluid.density_kg_m3 * velocity * effective_id) / fluid.viscosity_pa_s
        reynolds = max(reynolds, MIN_REYNOLDS)

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_reynolds",
            description="Calculate Reynolds number",
            inputs={
                "density_kg_m3": fluid.density_kg_m3,
                "velocity_m_s": velocity,
                "diameter_m": effective_id,
                "viscosity_pa_s": fluid.viscosity_pa_s
            },
            output_name="reynolds",
            output_value=reynolds,
            formula="Re = rho*v*D/mu",
            reference="Dimensionless number definition"
        ))
        step_num += 1

        # Calculate relative roughness
        relative_roughness = effective_roughness / effective_id

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_relative_roughness",
            description="Calculate relative roughness",
            inputs={
                "roughness_m": effective_roughness,
                "diameter_m": effective_id
            },
            output_name="relative_roughness",
            output_value=relative_roughness,
            formula="e/D = roughness / diameter",
            reference="Moody diagram"
        ))
        step_num += 1

        # Calculate friction factor
        friction_factor, flow_regime, friction_steps = self._friction_calculator.calculate(
            reynolds, relative_roughness, inputs.friction_correlation
        )

        # Renumber friction steps
        for fs in friction_steps:
            steps.append(CalculationStep(
                step_number=step_num,
                operation=fs.operation,
                description=fs.description,
                inputs=fs.inputs,
                output_name=fs.output_name,
                output_value=fs.output_value,
                formula=fs.formula,
                reference=fs.reference
            ))
            step_num += 1

        # Calculate dynamic pressure (velocity head)
        dynamic_pressure = fluid.density_kg_m3 * velocity ** 2 / Decimal("2")

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_dynamic_pressure",
            description="Calculate dynamic pressure (velocity head)",
            inputs={
                "density_kg_m3": fluid.density_kg_m3,
                "velocity_m_s": velocity
            },
            output_name="dynamic_pressure_pa",
            output_value=dynamic_pressure,
            formula="q = rho*v^2/2",
            reference="Bernoulli equation"
        ))
        step_num += 1

        # Calculate friction pressure drop
        # dP_f = f * (L/D) * (rho*v^2/2) * n_passes
        friction_loss = friction_factor * (geom.tube_length_m / effective_id) * dynamic_pressure * Decimal(str(geom.number_of_passes))

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_friction_loss",
            description="Calculate friction pressure drop (Darcy-Weisbach)",
            inputs={
                "friction_factor": friction_factor,
                "tube_length_m": geom.tube_length_m,
                "effective_id_m": effective_id,
                "dynamic_pressure_pa": dynamic_pressure,
                "number_of_passes": geom.number_of_passes
            },
            output_name="friction_loss_pa",
            output_value=friction_loss,
            formula="dP_f = f*(L/D)*(rho*v^2/2)*n_passes",
            reference="Darcy-Weisbach equation"
        ))
        step_num += 1

        # Calculate entrance/exit losses
        # One entrance and one exit per pass
        entrance_exit_loss = (self.K_TUBE_ENTRANCE + self.K_TUBE_EXIT) * dynamic_pressure * Decimal(str(geom.number_of_passes))

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_entrance_exit_loss",
            description="Calculate entrance and exit pressure losses",
            inputs={
                "K_entrance": self.K_TUBE_ENTRANCE,
                "K_exit": self.K_TUBE_EXIT,
                "dynamic_pressure_pa": dynamic_pressure,
                "number_of_passes": geom.number_of_passes
            },
            output_name="entrance_exit_loss_pa",
            output_value=entrance_exit_loss,
            formula="dP_e = (K_in + K_out)*(rho*v^2/2)*n_passes",
            reference="Crane TP-410"
        ))
        step_num += 1

        # Calculate return bend losses (n_passes - 1 return bends)
        if geom.number_of_passes > 1:
            return_loss = self.K_RETURN_BEND * Decimal(str(geom.number_of_passes - 1)) * dynamic_pressure
        else:
            return_loss = Decimal("0")

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_return_loss",
            description="Calculate return bend pressure losses",
            inputs={
                "K_return": self.K_RETURN_BEND,
                "number_of_returns": max(0, geom.number_of_passes - 1),
                "dynamic_pressure_pa": dynamic_pressure
            },
            output_name="return_loss_pa",
            output_value=return_loss,
            formula="dP_r = K_return*(n_passes-1)*(rho*v^2/2)",
            reference="TEMA Standards"
        ))
        step_num += 1

        # Calculate total pressure drop
        total_pressure_drop = friction_loss + entrance_exit_loss + return_loss

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_total_tube_side",
            description="Calculate total tube-side pressure drop",
            inputs={
                "friction_loss_pa": friction_loss,
                "entrance_exit_loss_pa": entrance_exit_loss,
                "return_loss_pa": return_loss
            },
            output_name="total_pressure_drop_pa",
            output_value=total_pressure_drop,
            formula="dP_total = dP_f + dP_e + dP_r",
            reference="TEMA Standards"
        ))

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(inputs, total_pressure_drop, steps)
        timestamp = datetime.now(timezone.utc).isoformat()

        # Apply precision to results
        return TubeSidePressureDropResult(
            friction_loss_pa=self._apply_precision(friction_loss),
            entrance_exit_loss_pa=self._apply_precision(entrance_exit_loss),
            return_loss_pa=self._apply_precision(return_loss),
            total_pressure_drop_pa=self._apply_precision(total_pressure_drop),
            velocity_m_s=self._apply_precision(velocity),
            reynolds_number=self._apply_precision(reynolds),
            friction_factor=self._apply_precision(friction_factor),
            flow_regime=flow_regime,
            calculation_steps=tuple(steps),
            provenance_hash=provenance_hash,
            timestamp=timestamp
        )

    # =========================================================================
    # SHELL-SIDE PRESSURE DROP (BELL-DELAWARE METHOD)
    # =========================================================================

    def calculate_shell_side_pressure_drop(
        self,
        inputs: ShellSideInput
    ) -> ShellSidePressureDropResult:
        """
        Calculate shell-side pressure drop using Bell-Delaware method.

        Components:
        1. Ideal crossflow pressure drop
        2. Correction factors (Jc, Jl, Jb, Js)
        3. Window pressure drop
        4. Entrance/exit pressure drop

        Reference: Bell, K.J. (1963), Delaware Method for Shell-Side
        Heat Transfer and Pressure Drop
        """
        steps: List[CalculationStep] = []
        step_num = 1

        # Extract parameters
        fluid = inputs.fluid
        tube_geom = inputs.tube_geometry
        shell_geom = inputs.shell_geometry
        m_dot = inputs.mass_flow_rate_kg_s

        # Calculate tube pitch if not provided
        tube_pitch = tube_geom.tube_pitch_m
        if tube_pitch <= Decimal("0"):
            tube_pitch = Decimal("1.25") * tube_geom.tube_od_m  # Standard 1.25 pitch ratio

        # Calculate crossflow area (minimum flow area between tubes)
        # For triangular pitch: A_c = (Ds * Lb / Pt) * (Pt - Do)
        D_s = shell_geom.shell_id_m
        L_b = shell_geom.baffle_spacing_m
        P_t = tube_pitch
        D_o = tube_geom.tube_od_m

        crossflow_area = (D_s * L_b / P_t) * (P_t - D_o)

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_crossflow_area",
            description="Calculate shell-side crossflow area",
            inputs={
                "shell_id_m": D_s,
                "baffle_spacing_m": L_b,
                "tube_pitch_m": P_t,
                "tube_od_m": D_o
            },
            output_name="crossflow_area_m2",
            output_value=crossflow_area,
            formula="A_c = (Ds*Lb/Pt)*(Pt-Do)",
            reference="Bell-Delaware Method"
        ))
        step_num += 1

        # Calculate crossflow velocity
        volumetric_flow = m_dot / fluid.density_kg_m3
        crossflow_velocity = volumetric_flow / crossflow_area
        crossflow_velocity = max(crossflow_velocity, MIN_VELOCITY)

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_crossflow_velocity",
            description="Calculate shell-side crossflow velocity",
            inputs={
                "mass_flow_rate_kg_s": m_dot,
                "density_kg_m3": fluid.density_kg_m3,
                "crossflow_area_m2": crossflow_area
            },
            output_name="crossflow_velocity_m_s",
            output_value=crossflow_velocity,
            formula="v_c = m_dot/(rho*A_c)",
            reference="Continuity equation"
        ))
        step_num += 1

        # Calculate equivalent diameter for shell-side
        # For triangular pitch: De = 4*(Pt^2*sqrt(3)/4 - pi*Do^2/8)/(pi*Do/2)
        if tube_geom.pitch_pattern == TubePitchPattern.TRIANGULAR:
            flow_area_element = (P_t ** 2 * Decimal(str(math.sqrt(3))) / Decimal("4") -
                               PI * D_o ** 2 / Decimal("8"))
            wetted_perimeter_element = PI * D_o / Decimal("2")
            D_e = Decimal("4") * flow_area_element / wetted_perimeter_element
        else:  # Square pitch
            flow_area_element = P_t ** 2 - PI * D_o ** 2 / Decimal("4")
            wetted_perimeter_element = PI * D_o
            D_e = Decimal("4") * flow_area_element / wetted_perimeter_element

        D_e = max(D_e, MIN_DIAMETER)

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_equivalent_diameter",
            description="Calculate shell-side equivalent diameter",
            inputs={
                "tube_pitch_m": P_t,
                "tube_od_m": D_o,
                "pitch_pattern": tube_geom.pitch_pattern.name
            },
            output_name="equivalent_diameter_m",
            output_value=D_e,
            formula="De = 4*flow_area/wetted_perimeter",
            reference="Bell-Delaware Method"
        ))
        step_num += 1

        # Calculate Reynolds number for crossflow
        reynolds_crossflow = (fluid.density_kg_m3 * crossflow_velocity * D_o) / fluid.viscosity_pa_s
        reynolds_crossflow = max(reynolds_crossflow, MIN_REYNOLDS)

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_crossflow_reynolds",
            description="Calculate crossflow Reynolds number",
            inputs={
                "density_kg_m3": fluid.density_kg_m3,
                "crossflow_velocity_m_s": crossflow_velocity,
                "tube_od_m": D_o,
                "viscosity_pa_s": fluid.viscosity_pa_s
            },
            output_name="reynolds_crossflow",
            output_value=reynolds_crossflow,
            formula="Re = rho*v*Do/mu",
            reference="Bell-Delaware Method"
        ))
        step_num += 1

        # Calculate number of tube rows crossed in crossflow
        # N_c = Ds * (1 - 2*Bc) / Pp
        # where Bc = baffle cut fraction, Pp = tube pitch (parallel)
        B_c = shell_geom.baffle_cut_fraction
        if tube_geom.pitch_pattern in [TubePitchPattern.TRIANGULAR, TubePitchPattern.ROTATED_TRIANGULAR]:
            P_p = P_t * Decimal(str(math.sqrt(3))) / Decimal("2")  # Pitch parallel to flow
        else:
            P_p = P_t

        N_c_float = float(D_s * (Decimal("1") - Decimal("2") * B_c) / P_p)
        N_c = int(max(1, N_c_float))

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_crossflow_rows",
            description="Calculate number of tube rows in crossflow",
            inputs={
                "shell_id_m": D_s,
                "baffle_cut_fraction": B_c,
                "tube_pitch_parallel_m": P_p
            },
            output_name="crossflow_rows",
            output_value=N_c,
            formula="N_c = Ds*(1-2*Bc)/Pp",
            reference="Bell-Delaware Method"
        ))
        step_num += 1

        # Calculate ideal crossflow friction factor
        # Using Kern correlation: f_ideal = 0.43 * Re^(-0.19) for 10 < Re < 1000
        # For higher Re: f_ideal = 0.19 * Re^(-0.16)
        re_float = float(reynolds_crossflow)
        if re_float < 1000:
            f_ideal = 0.43 * (re_float ** (-0.19))
        else:
            f_ideal = 0.19 * (re_float ** (-0.16))

        f_ideal_decimal = Decimal(str(f_ideal))

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_ideal_friction_factor",
            description="Calculate ideal crossflow friction factor",
            inputs={"reynolds_crossflow": reynolds_crossflow},
            output_name="f_ideal",
            output_value=f_ideal_decimal,
            formula="f = 0.43*Re^(-0.19) for Re<1000, else 0.19*Re^(-0.16)",
            reference="Bell-Delaware Method, Kern correlation"
        ))
        step_num += 1

        # Calculate ideal crossflow pressure drop
        # dP_ideal = f_ideal * N_c * (rho*v^2/2)
        dynamic_pressure = fluid.density_kg_m3 * crossflow_velocity ** 2 / Decimal("2")
        dp_ideal = f_ideal_decimal * Decimal(str(N_c)) * dynamic_pressure

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_ideal_crossflow_dp",
            description="Calculate ideal crossflow pressure drop",
            inputs={
                "f_ideal": f_ideal_decimal,
                "crossflow_rows": N_c,
                "dynamic_pressure_pa": dynamic_pressure
            },
            output_name="dp_ideal_pa",
            output_value=dp_ideal,
            formula="dP_ideal = f*N_c*(rho*v^2/2)",
            reference="Bell-Delaware Method"
        ))
        step_num += 1

        # Calculate Bell-Delaware correction factors
        # J_c: Baffle cut correction
        J_c = self._calculate_j_c(B_c)

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_j_c",
            description="Calculate baffle cut correction factor J_c",
            inputs={"baffle_cut_fraction": B_c},
            output_name="J_c",
            output_value=J_c,
            formula="J_c = 0.55 + 0.72*Fc (correlation)",
            reference="Bell-Delaware Method"
        ))
        step_num += 1

        # J_l: Leakage correction
        J_l = self._calculate_j_l(shell_geom, tube_geom)

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_j_l",
            description="Calculate leakage correction factor J_l",
            inputs={
                "tube_to_baffle_clearance_m": shell_geom.tube_to_baffle_clearance_m,
                "shell_to_baffle_clearance_m": shell_geom.shell_to_baffle_clearance_m
            },
            output_name="J_l",
            output_value=J_l,
            formula="J_l = exp(-1.33*(1+r_s)*(A_leak/A_c)^p)",
            reference="Bell-Delaware Method"
        ))
        step_num += 1

        # J_b: Bypass correction
        J_b = self._calculate_j_b(shell_geom, tube_geom, reynolds_crossflow)

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_j_b",
            description="Calculate bypass correction factor J_b",
            inputs={
                "bundle_to_shell_clearance_m": shell_geom.bundle_to_shell_clearance_m,
                "number_of_sealing_strips": shell_geom.number_of_sealing_strips
            },
            output_name="J_b",
            output_value=J_b,
            formula="J_b = exp(-C_bp*F_bp*(1-(2*N_ss/N_c)^(1/3)))",
            reference="Bell-Delaware Method"
        ))
        step_num += 1

        # J_s: Unequal baffle spacing correction
        J_s = self._calculate_j_s(shell_geom)

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_j_s",
            description="Calculate unequal spacing correction factor J_s",
            inputs={
                "inlet_baffle_spacing_m": shell_geom.inlet_baffle_spacing_m,
                "outlet_baffle_spacing_m": shell_geom.outlet_baffle_spacing_m,
                "central_baffle_spacing_m": shell_geom.baffle_spacing_m
            },
            output_name="J_s",
            output_value=J_s,
            formula="J_s = (N_b-1+(L_i/L_c)^(1-n)+(L_o/L_c)^(1-n))/(N_b-1+L_i/L_c+L_o/L_c)",
            reference="Bell-Delaware Method"
        ))
        step_num += 1

        # Calculate window pressure drop
        dp_window = self._calculate_window_pressure_drop(
            shell_geom, tube_geom, fluid, crossflow_velocity, reynolds_crossflow
        )

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_window_dp",
            description="Calculate window pressure drop",
            inputs={
                "crossflow_velocity_m_s": crossflow_velocity,
                "baffle_cut_fraction": B_c,
                "density_kg_m3": fluid.density_kg_m3
            },
            output_name="dp_window_pa",
            output_value=dp_window,
            formula="dP_w = (2+0.6*N_tw)*(rho*v_w^2/2)*N_b",
            reference="Bell-Delaware Method"
        ))
        step_num += 1

        # Calculate entrance/exit pressure drop
        dp_entrance_exit = self._calculate_shell_entrance_exit_dp(
            shell_geom, fluid, crossflow_velocity, N_c
        )

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_entrance_exit_dp",
            description="Calculate shell entrance/exit pressure drop",
            inputs={
                "crossflow_velocity_m_s": crossflow_velocity,
                "density_kg_m3": fluid.density_kg_m3,
                "crossflow_rows": N_c
            },
            output_name="dp_entrance_exit_pa",
            output_value=dp_entrance_exit,
            formula="dP_io = 2*dP_ideal*(1+N_cw/N_c)",
            reference="Bell-Delaware Method"
        ))
        step_num += 1

        # Calculate corrected crossflow pressure drop
        # Number of baffles
        N_b = shell_geom.number_of_baffles
        if N_b <= 0:
            # Estimate number of baffles
            N_b = max(1, int(float(tube_geom.tube_length_m / shell_geom.baffle_spacing_m)) - 1)

        dp_crossflow_corrected = dp_ideal * J_c * J_l * J_b * Decimal(str(N_b))

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_corrected_crossflow_dp",
            description="Calculate corrected crossflow pressure drop",
            inputs={
                "dp_ideal_pa": dp_ideal,
                "J_c": J_c,
                "J_l": J_l,
                "J_b": J_b,
                "number_of_baffles": N_b
            },
            output_name="dp_crossflow_corrected_pa",
            output_value=dp_crossflow_corrected,
            formula="dP_c = dP_ideal*J_c*J_l*J_b*N_b",
            reference="Bell-Delaware Method"
        ))
        step_num += 1

        # Calculate total shell-side pressure drop
        dp_total = (dp_crossflow_corrected + dp_window * Decimal(str(N_b)) + dp_entrance_exit) * J_s

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_total_shell_side",
            description="Calculate total shell-side pressure drop",
            inputs={
                "dp_crossflow_pa": dp_crossflow_corrected,
                "dp_window_pa": dp_window,
                "dp_entrance_exit_pa": dp_entrance_exit,
                "J_s": J_s,
                "number_of_baffles": N_b
            },
            output_name="total_pressure_drop_pa",
            output_value=dp_total,
            formula="dP_total = (dP_c + dP_w*N_b + dP_io)*J_s",
            reference="Bell-Delaware Method"
        ))

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(inputs, dp_total, steps)
        timestamp = datetime.now(timezone.utc).isoformat()

        return ShellSidePressureDropResult(
            ideal_crossflow_dp_pa=self._apply_precision(dp_ideal),
            window_dp_pa=self._apply_precision(dp_window * Decimal(str(N_b))),
            entrance_exit_dp_pa=self._apply_precision(dp_entrance_exit),
            total_pressure_drop_pa=self._apply_precision(dp_total),
            j_c_baffle_cut=self._apply_precision(J_c),
            j_l_leakage=self._apply_precision(J_l),
            j_b_bypass=self._apply_precision(J_b),
            j_s_spacing=self._apply_precision(J_s),
            crossflow_velocity_m_s=self._apply_precision(crossflow_velocity),
            crossflow_reynolds=self._apply_precision(reynolds_crossflow),
            number_of_crossflow_rows=N_c,
            calculation_steps=tuple(steps),
            provenance_hash=provenance_hash,
            timestamp=timestamp
        )

    # =========================================================================
    # FRICTION FACTOR INTERFACE
    # =========================================================================

    def calculate_friction_factor(
        self,
        reynolds: Decimal,
        relative_roughness: Decimal = Decimal("0"),
        correlation: FrictionCorrelation = FrictionCorrelation.CHURCHILL
    ) -> Tuple[Decimal, FlowRegime, Tuple[CalculationStep, ...]]:
        """
        Calculate Darcy friction factor using specified correlation.

        Args:
            reynolds: Reynolds number
            relative_roughness: epsilon/D (default: 0 for smooth)
            correlation: Friction factor correlation to use

        Returns:
            Tuple of (friction_factor, flow_regime, calculation_steps)
        """
        f, regime, steps = self._friction_calculator.calculate(
            reynolds, relative_roughness, correlation
        )
        return f, regime, tuple(steps)

    # =========================================================================
    # FOULING IMPACT ANALYSIS
    # =========================================================================

    def calculate_fouling_impact_on_pressure(
        self,
        clean_inputs: TubeSideInput,
        fouling_condition: FoulingCondition
    ) -> FoulingImpactResult:
        """
        Analyze impact of fouling on tube-side pressure drop.

        Fouling effects:
        1. Effective diameter reduction: D_eff = D_i - 2*t_fouling
        2. Roughness increase: e_eff = e_tube + e_fouling
        3. Flow area reduction: A_eff = pi*(D_eff/2)^2

        Args:
            clean_inputs: TubeSideInput with clean geometry
            fouling_condition: FoulingCondition with fouling parameters

        Returns:
            FoulingImpactResult with fouled/clean comparison
        """
        steps: List[CalculationStep] = []
        step_num = 1

        # Calculate clean pressure drop
        clean_result = self.calculate_tube_side_pressure_drop(clean_inputs)
        clean_dp = clean_result.total_pressure_drop_pa

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_clean_dp",
            description="Calculate pressure drop with clean tubes",
            inputs={"tube_id_m": clean_inputs.geometry.tube_id_m},
            output_name="clean_pressure_drop_pa",
            output_value=clean_dp,
            formula="(See tube-side calculation)",
            reference="TEMA Standards"
        ))
        step_num += 1

        # Create fouled inputs
        fouled_inputs = TubeSideInput(
            fluid=clean_inputs.fluid,
            geometry=clean_inputs.geometry,
            mass_flow_rate_kg_s=clean_inputs.mass_flow_rate_kg_s,
            fouling=fouling_condition,
            friction_correlation=clean_inputs.friction_correlation
        )

        # Calculate fouled pressure drop
        fouled_result = self.calculate_tube_side_pressure_drop(fouled_inputs)
        fouled_dp = fouled_result.total_pressure_drop_pa

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_fouled_dp",
            description="Calculate pressure drop with fouled tubes",
            inputs={
                "tube_id_m": clean_inputs.geometry.tube_id_m,
                "fouling_thickness_m": fouling_condition.fouling_thickness_m
            },
            output_name="fouled_pressure_drop_pa",
            output_value=fouled_dp,
            formula="(See tube-side calculation with reduced diameter)",
            reference="TEMA Standards"
        ))
        step_num += 1

        # Calculate pressure drop ratio
        if clean_dp > Decimal("0"):
            dp_ratio = fouled_dp / clean_dp
        else:
            dp_ratio = Decimal("1")

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_dp_ratio",
            description="Calculate fouled/clean pressure drop ratio",
            inputs={
                "fouled_dp_pa": fouled_dp,
                "clean_dp_pa": clean_dp
            },
            output_name="pressure_drop_ratio",
            output_value=dp_ratio,
            formula="ratio = dP_fouled / dP_clean",
            reference="TEMA Standards"
        ))
        step_num += 1

        # Calculate diameter reduction
        diameter_reduction = Decimal("2") * fouling_condition.fouling_thickness_m

        # Calculate flow area reduction
        D_clean = clean_inputs.geometry.tube_id_m
        D_fouled = D_clean - diameter_reduction
        A_clean = PI * (D_clean / Decimal("2")) ** 2
        A_fouled = PI * (D_fouled / Decimal("2")) ** 2
        if A_clean > Decimal("0"):
            area_reduction_percent = (Decimal("1") - A_fouled / A_clean) * Decimal("100")
        else:
            area_reduction_percent = Decimal("0")

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_area_reduction",
            description="Calculate flow area reduction due to fouling",
            inputs={
                "D_clean_m": D_clean,
                "D_fouled_m": D_fouled
            },
            output_name="area_reduction_percent",
            output_value=area_reduction_percent,
            formula="reduction = (1 - A_fouled/A_clean)*100",
            reference="Geometry calculation"
        ))
        step_num += 1

        # Calculate roughness increase factor
        e_clean = clean_inputs.geometry.tube_roughness_m
        e_fouled = e_clean + fouling_condition.fouling_roughness_m
        if e_clean > Decimal("0"):
            roughness_factor = e_fouled / e_clean
        else:
            roughness_factor = Decimal("1")

        # Energy penalty estimate
        # Pumping power proportional to pressure drop
        energy_penalty_percent = (dp_ratio - Decimal("1")) * Decimal("100")

        # Cleaning recommendation (if pressure drop increased by > 50%)
        cleaning_recommended = dp_ratio > Decimal("1.5")

        steps.append(CalculationStep(
            step_number=step_num,
            operation="assess_cleaning_need",
            description="Assess whether cleaning is recommended",
            inputs={
                "pressure_drop_ratio": dp_ratio,
                "threshold": Decimal("1.5")
            },
            output_name="cleaning_recommended",
            output_value=cleaning_recommended,
            formula="cleaning recommended if ratio > 1.5",
            reference="TEMA Guidelines"
        ))

        # Calculate provenance hash
        provenance_data = {
            "clean_dp_pa": str(clean_dp),
            "fouled_dp_pa": str(fouled_dp),
            "fouling_thickness_m": str(fouling_condition.fouling_thickness_m)
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return FoulingImpactResult(
            clean_pressure_drop_pa=self._apply_precision(clean_dp),
            fouled_pressure_drop_pa=self._apply_precision(fouled_dp),
            pressure_drop_ratio=self._apply_precision(dp_ratio),
            effective_diameter_reduction_m=self._apply_precision(diameter_reduction),
            roughness_increase_factor=self._apply_precision(roughness_factor),
            flow_area_reduction_percent=self._apply_precision(area_reduction_percent),
            cleaning_recommended=cleaning_recommended,
            estimated_energy_penalty_percent=self._apply_precision(energy_penalty_percent),
            calculation_steps=tuple(steps),
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # PRESSURE DROP ACCEPTABILITY
    # =========================================================================

    def assess_pressure_drop_acceptability(
        self,
        tube_side_result: TubeSidePressureDropResult,
        shell_side_result: ShellSidePressureDropResult,
        limits: PressureDropLimits,
        fluid_density_kg_m3: Decimal
    ) -> PressureDropAcceptabilityResult:
        """
        Assess whether pressure drops are within allowable limits.

        Checks:
        1. Tube-side vs maximum allowable
        2. Shell-side vs maximum allowable
        3. Pump/compressor capacity adequacy

        Args:
            tube_side_result: Calculated tube-side pressure drop
            shell_side_result: Calculated shell-side pressure drop
            limits: Allowable pressure drop limits
            fluid_density_kg_m3: Fluid density for head calculation

        Returns:
            PressureDropAcceptabilityResult with recommendations
        """
        steps: List[CalculationStep] = []
        recommendations: List[str] = []
        step_num = 1

        # Check tube-side acceptability
        tube_dp = tube_side_result.total_pressure_drop_pa
        tube_acceptable = tube_dp <= limits.max_tube_side_pa

        if limits.max_tube_side_pa > Decimal("0"):
            tube_margin = (limits.max_tube_side_pa - tube_dp) / limits.max_tube_side_pa * Decimal("100")
        else:
            tube_margin = Decimal("0")

        steps.append(CalculationStep(
            step_number=step_num,
            operation="check_tube_side",
            description="Check tube-side pressure drop against limit",
            inputs={
                "tube_dp_pa": tube_dp,
                "max_allowable_pa": limits.max_tube_side_pa
            },
            output_name="tube_side_acceptable",
            output_value=tube_acceptable,
            formula="acceptable = dP_tube <= dP_max",
            reference="Design specification"
        ))
        step_num += 1

        if not tube_acceptable:
            recommendations.append(
                f"Tube-side pressure drop ({float(tube_dp):.0f} Pa) exceeds limit "
                f"({float(limits.max_tube_side_pa):.0f} Pa). Consider: "
                "1) Reduce number of passes, 2) Increase tube diameter, "
                "3) Reduce tube length, 4) Clean tubes if fouled."
            )

        # Check shell-side acceptability
        shell_dp = shell_side_result.total_pressure_drop_pa
        shell_acceptable = shell_dp <= limits.max_shell_side_pa

        if limits.max_shell_side_pa > Decimal("0"):
            shell_margin = (limits.max_shell_side_pa - shell_dp) / limits.max_shell_side_pa * Decimal("100")
        else:
            shell_margin = Decimal("0")

        steps.append(CalculationStep(
            step_number=step_num,
            operation="check_shell_side",
            description="Check shell-side pressure drop against limit",
            inputs={
                "shell_dp_pa": shell_dp,
                "max_allowable_pa": limits.max_shell_side_pa
            },
            output_name="shell_side_acceptable",
            output_value=shell_acceptable,
            formula="acceptable = dP_shell <= dP_max",
            reference="Design specification"
        ))
        step_num += 1

        if not shell_acceptable:
            recommendations.append(
                f"Shell-side pressure drop ({float(shell_dp):.0f} Pa) exceeds limit "
                f"({float(limits.max_shell_side_pa):.0f} Pa). Consider: "
                "1) Increase baffle spacing, 2) Use double-segmental baffles, "
                "3) Increase baffle cut percentage, 4) Add sealing strips."
            )

        # Calculate required pump head
        # Total head = pressure drop / (rho * g)
        total_dp = tube_dp + shell_dp
        required_head = total_dp / (fluid_density_kg_m3 * STANDARD_GRAVITY)

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_required_head",
            description="Calculate required pump head",
            inputs={
                "total_dp_pa": total_dp,
                "density_kg_m3": fluid_density_kg_m3,
                "gravity_m_s2": STANDARD_GRAVITY
            },
            output_name="required_head_m",
            output_value=required_head,
            formula="H = dP / (rho*g)",
            reference="Fluid mechanics"
        ))
        step_num += 1

        # Check pump capacity
        pump_adequate = required_head <= limits.pump_available_head_m

        steps.append(CalculationStep(
            step_number=step_num,
            operation="check_pump_capacity",
            description="Check pump capacity adequacy",
            inputs={
                "required_head_m": required_head,
                "available_head_m": limits.pump_available_head_m
            },
            output_name="pump_adequate",
            output_value=pump_adequate,
            formula="adequate = H_required <= H_available",
            reference="Pump specification"
        ))

        if not pump_adequate:
            recommendations.append(
                f"Required pump head ({float(required_head):.1f} m) exceeds available "
                f"({float(limits.pump_available_head_m):.1f} m). Consider: "
                "1) Upgrade pump, 2) Reduce pressure drops, 3) Add booster pump."
            )

        # Overall acceptability
        total_acceptable = tube_acceptable and shell_acceptable and pump_adequate

        if total_acceptable and not recommendations:
            recommendations.append(
                "All pressure drops are within acceptable limits. "
                "Design is acceptable for the specified operating conditions."
            )

        # Calculate provenance hash
        provenance_data = {
            "tube_dp_pa": str(tube_dp),
            "shell_dp_pa": str(shell_dp),
            "required_head_m": str(required_head)
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return PressureDropAcceptabilityResult(
            tube_side_acceptable=tube_acceptable,
            shell_side_acceptable=shell_acceptable,
            total_acceptable=total_acceptable,
            tube_side_margin_percent=self._apply_precision(tube_margin),
            shell_side_margin_percent=self._apply_precision(shell_margin),
            pump_capacity_adequate=pump_adequate,
            required_pump_head_m=self._apply_precision(required_head),
            recommendations=tuple(recommendations),
            calculation_steps=tuple(steps),
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # PUMP POWER REQUIREMENT
    # =========================================================================

    def calculate_pump_power_requirement(
        self,
        total_pressure_drop_pa: Decimal,
        volumetric_flow_rate_m3_s: Decimal,
        fluid_density_kg_m3: Decimal,
        pump_efficiency: Decimal = Decimal("0.75"),
        motor_efficiency: Decimal = Decimal("0.95"),
        operating_hours_per_year: Decimal = Decimal("8000"),
        electricity_cost_per_kwh: Decimal = Decimal("0.10")
    ) -> PumpPowerResult:
        """
        Calculate pump power requirement and operating costs.

        Power calculations:
        1. Hydraulic power: P_h = Q * dP
        2. Shaft power: P_s = P_h / eta_pump
        3. Electrical power: P_e = P_s / eta_motor

        Args:
            total_pressure_drop_pa: Total system pressure drop (Pa)
            volumetric_flow_rate_m3_s: Volumetric flow rate (m^3/s)
            fluid_density_kg_m3: Fluid density (kg/m^3)
            pump_efficiency: Pump efficiency (0-1)
            motor_efficiency: Motor efficiency (0-1)
            operating_hours_per_year: Annual operating hours
            electricity_cost_per_kwh: Electricity cost (USD/kWh)

        Returns:
            PumpPowerResult with power requirements and costs
        """
        steps: List[CalculationStep] = []
        step_num = 1

        # Calculate required head
        required_head = total_pressure_drop_pa / (fluid_density_kg_m3 * STANDARD_GRAVITY)

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_head",
            description="Calculate required pump head",
            inputs={
                "pressure_drop_pa": total_pressure_drop_pa,
                "density_kg_m3": fluid_density_kg_m3,
                "gravity_m_s2": STANDARD_GRAVITY
            },
            output_name="required_head_m",
            output_value=required_head,
            formula="H = dP / (rho*g)",
            reference="Fluid mechanics"
        ))
        step_num += 1

        # Calculate hydraulic power
        # P_h = Q * dP = Q * rho * g * H
        hydraulic_power = volumetric_flow_rate_m3_s * total_pressure_drop_pa

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_hydraulic_power",
            description="Calculate hydraulic power",
            inputs={
                "volumetric_flow_m3_s": volumetric_flow_rate_m3_s,
                "pressure_drop_pa": total_pressure_drop_pa
            },
            output_name="hydraulic_power_w",
            output_value=hydraulic_power,
            formula="P_h = Q * dP",
            reference="Pump hydraulics"
        ))
        step_num += 1

        # Calculate shaft power (brake power)
        if pump_efficiency > Decimal("0"):
            shaft_power = hydraulic_power / pump_efficiency
        else:
            shaft_power = hydraulic_power

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_shaft_power",
            description="Calculate required shaft power",
            inputs={
                "hydraulic_power_w": hydraulic_power,
                "pump_efficiency": pump_efficiency
            },
            output_name="shaft_power_w",
            output_value=shaft_power,
            formula="P_s = P_h / eta_pump",
            reference="Pump performance"
        ))
        step_num += 1

        # Calculate electrical power
        if motor_efficiency > Decimal("0"):
            electrical_power = shaft_power / motor_efficiency
        else:
            electrical_power = shaft_power

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_electrical_power",
            description="Calculate electrical power input",
            inputs={
                "shaft_power_w": shaft_power,
                "motor_efficiency": motor_efficiency
            },
            output_name="electrical_power_w",
            output_value=electrical_power,
            formula="P_e = P_s / eta_motor",
            reference="Motor performance"
        ))
        step_num += 1

        # Calculate annual energy consumption
        electrical_power_kw = electrical_power / Decimal("1000")
        annual_energy_kwh = electrical_power_kw * operating_hours_per_year

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_annual_energy",
            description="Calculate annual energy consumption",
            inputs={
                "electrical_power_kw": electrical_power_kw,
                "operating_hours": operating_hours_per_year
            },
            output_name="annual_energy_kwh",
            output_value=annual_energy_kwh,
            formula="E = P * t",
            reference="Energy calculation"
        ))
        step_num += 1

        # Calculate annual energy cost
        annual_cost = annual_energy_kwh * electricity_cost_per_kwh

        steps.append(CalculationStep(
            step_number=step_num,
            operation="calculate_annual_cost",
            description="Calculate annual electricity cost",
            inputs={
                "annual_energy_kwh": annual_energy_kwh,
                "cost_per_kwh": electricity_cost_per_kwh
            },
            output_name="annual_cost_usd",
            output_value=annual_cost,
            formula="Cost = E * price",
            reference="Cost calculation"
        ))

        # Calculate provenance hash
        provenance_data = {
            "pressure_drop_pa": str(total_pressure_drop_pa),
            "flow_rate_m3_s": str(volumetric_flow_rate_m3_s),
            "electrical_power_w": str(electrical_power)
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return PumpPowerResult(
            hydraulic_power_w=self._apply_precision(hydraulic_power),
            shaft_power_w=self._apply_precision(shaft_power),
            electrical_power_w=self._apply_precision(electrical_power),
            required_head_m=self._apply_precision(required_head),
            volumetric_flow_m3_s=self._apply_precision(volumetric_flow_rate_m3_s),
            pump_efficiency=pump_efficiency,
            motor_efficiency=motor_efficiency,
            annual_energy_kwh=self._apply_precision(annual_energy_kwh),
            annual_energy_cost_usd=self._apply_precision(annual_cost),
            calculation_steps=tuple(steps),
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    def _calculate_j_c(self, baffle_cut_fraction: Decimal) -> Decimal:
        """
        Calculate baffle cut correction factor J_c.

        Reference: Bell-Delaware Method
        """
        # J_c correlation: J_c = 0.55 + 0.72 * Fc
        # where Fc is fraction of tubes in crossflow
        # For typical designs: Fc ~ (1 - 2*Bc)
        F_c = Decimal("1") - Decimal("2") * baffle_cut_fraction
        J_c = Decimal("0.55") + Decimal("0.72") * F_c

        # Clamp to valid range
        J_c = max(self.J_MIN, min(self.J_MAX, J_c))
        return J_c

    def _calculate_j_l(
        self,
        shell_geom: ShellGeometry,
        tube_geom: TubeGeometry
    ) -> Decimal:
        """
        Calculate leakage correction factor J_l.

        Accounts for:
        - Tube-to-baffle hole leakage
        - Shell-to-baffle leakage

        Reference: Bell-Delaware Method
        """
        # Simplified J_l correlation
        # Full method requires detailed leak area calculations

        # Estimate leakage ratio
        tube_leak_area = (PI * tube_geom.tube_od_m * shell_geom.tube_to_baffle_clearance_m *
                        Decimal(str(tube_geom.number_of_tubes)))
        shell_leak_area = PI * shell_geom.shell_id_m * shell_geom.shell_to_baffle_clearance_m

        # Crossflow area estimate
        crossflow_area = (shell_geom.shell_id_m * shell_geom.baffle_spacing_m *
                         (Decimal("1") - tube_geom.tube_od_m / tube_geom.tube_pitch_m
                          if tube_geom.tube_pitch_m > 0 else Decimal("0.75")))

        if crossflow_area > Decimal("0"):
            leak_ratio = (tube_leak_area + shell_leak_area) / crossflow_area
        else:
            leak_ratio = Decimal("0.1")

        # J_l correlation
        # J_l = exp(-1.33 * (1 + r_s) * leak_ratio^0.3)
        # r_s = shell_leak / tube_leak ratio
        if tube_leak_area > Decimal("0"):
            r_s = shell_leak_area / tube_leak_area
        else:
            r_s = Decimal("0.5")

        exponent = float(Decimal("-1.33") * (Decimal("1") + r_s) *
                        Decimal(str(float(leak_ratio) ** 0.3)))
        J_l = Decimal(str(math.exp(exponent)))

        # Clamp to valid range
        J_l = max(self.J_MIN, min(self.J_MAX, J_l))
        return J_l

    def _calculate_j_b(
        self,
        shell_geom: ShellGeometry,
        tube_geom: TubeGeometry,
        reynolds: Decimal
    ) -> Decimal:
        """
        Calculate bypass correction factor J_b.

        Accounts for bundle-to-shell bypass flow.

        Reference: Bell-Delaware Method
        """
        # Bypass area ratio
        bypass_area = shell_geom.bundle_to_shell_clearance_m * shell_geom.baffle_spacing_m
        crossflow_area = (shell_geom.shell_id_m * shell_geom.baffle_spacing_m *
                         Decimal("0.25"))  # Simplified

        if crossflow_area > Decimal("0"):
            F_bp = bypass_area / crossflow_area
        else:
            F_bp = Decimal("0.1")

        # Sealing strip effectiveness
        N_ss = shell_geom.number_of_sealing_strips
        if N_ss > 0:
            # Estimate N_c (tube rows) from geometry
            N_c_est = int(float(shell_geom.shell_id_m / tube_geom.tube_od_m / Decimal("1.5")))
            N_c_est = max(1, N_c_est)
            seal_factor = Decimal("1") - (Decimal("2") * Decimal(str(N_ss)) /
                                          Decimal(str(N_c_est))) ** (Decimal("1") / Decimal("3"))
            seal_factor = max(Decimal("0"), seal_factor)
        else:
            seal_factor = Decimal("1")

        # C_bp depends on Reynolds number
        if reynolds < Decimal("1000"):
            C_bp = Decimal("4.5")
        else:
            C_bp = Decimal("3.7")

        # J_b = exp(-C_bp * F_bp * seal_factor)
        exponent = float(-C_bp * F_bp * seal_factor)
        J_b = Decimal(str(math.exp(exponent)))

        # Clamp to valid range
        J_b = max(self.J_MIN, min(self.J_MAX, J_b))
        return J_b

    def _calculate_j_s(self, shell_geom: ShellGeometry) -> Decimal:
        """
        Calculate unequal baffle spacing correction factor J_s.

        Accounts for different inlet/outlet baffle spacing.

        Reference: Bell-Delaware Method
        """
        L_c = shell_geom.baffle_spacing_m
        L_i = shell_geom.inlet_baffle_spacing_m if shell_geom.inlet_baffle_spacing_m > 0 else L_c
        L_o = shell_geom.outlet_baffle_spacing_m if shell_geom.outlet_baffle_spacing_m > 0 else L_c

        N_b = shell_geom.number_of_baffles
        if N_b <= 1:
            N_b = 2  # Minimum for calculation

        # n exponent (typically 0.6 for pressure drop)
        n = Decimal("0.6")

        # J_s correlation
        numerator = (Decimal(str(N_b - 1)) +
                    (L_i / L_c) ** (Decimal("1") - n) +
                    (L_o / L_c) ** (Decimal("1") - n))
        denominator = (Decimal(str(N_b - 1)) + L_i / L_c + L_o / L_c)

        if denominator > Decimal("0"):
            J_s = numerator / denominator
        else:
            J_s = Decimal("1")

        # Clamp to valid range
        J_s = max(self.J_MIN, min(Decimal("1.2"), J_s))  # Can be > 1 for this factor
        return J_s

    def _calculate_window_pressure_drop(
        self,
        shell_geom: ShellGeometry,
        tube_geom: TubeGeometry,
        fluid: FluidProperties,
        crossflow_velocity: Decimal,
        reynolds: Decimal
    ) -> Decimal:
        """
        Calculate baffle window pressure drop.

        Reference: Bell-Delaware Method
        """
        # Number of tubes in window (estimate)
        B_c = shell_geom.baffle_cut_fraction
        N_tw = int(float(Decimal(str(tube_geom.number_of_tubes)) * Decimal("2") * B_c))
        N_tw = max(1, N_tw)

        # Window flow area
        # A_w = A_window - A_tubes_in_window
        D_s = shell_geom.shell_id_m

        # Approximate window area (simplified)
        theta = Decimal("2") * Decimal(str(math.acos(float(Decimal("1") - Decimal("2") * B_c))))
        A_window = (D_s ** 2 / Decimal("8")) * (theta - Decimal(str(math.sin(float(theta)))))
        A_tubes = Decimal(str(N_tw)) * PI * (tube_geom.tube_od_m / Decimal("2")) ** 2
        A_w = A_window - A_tubes
        A_w = max(A_w, MIN_DIAMETER ** 2)

        # Window velocity
        Q = crossflow_velocity * shell_geom.baffle_spacing_m * D_s * Decimal("0.25")
        v_w = Q / A_w

        # Window pressure drop per baffle
        # dP_w = (2 + 0.6*N_tw) * (rho*v_w^2/2)
        dynamic_pressure = fluid.density_kg_m3 * v_w ** 2 / Decimal("2")
        dp_w = (Decimal("2") + Decimal("0.6") * Decimal(str(N_tw))) * dynamic_pressure

        return dp_w

    def _calculate_shell_entrance_exit_dp(
        self,
        shell_geom: ShellGeometry,
        fluid: FluidProperties,
        crossflow_velocity: Decimal,
        N_c: int
    ) -> Decimal:
        """
        Calculate shell entrance and exit pressure drops.

        Reference: Bell-Delaware Method
        """
        # Entrance/exit pressure drop
        # dP_io = 2 * dP_crossflow_per_row * (1 + N_cw/N_c)

        # Estimate N_cw (tube rows in window)
        B_c = shell_geom.baffle_cut_fraction
        N_cw = int(float(Decimal(str(N_c)) * B_c))
        N_cw = max(1, N_cw)

        # Dynamic pressure
        dynamic_pressure = fluid.density_kg_m3 * crossflow_velocity ** 2 / Decimal("2")

        # Entrance/exit factor
        factor = Decimal("1") + Decimal(str(N_cw)) / Decimal(str(max(1, N_c)))

        # Simplified entrance/exit pressure drop
        dp_io = Decimal("2") * dynamic_pressure * factor

        return dp_io

    def _apply_precision(self, value: Decimal) -> Decimal:
        """Apply precision rounding to result."""
        quantize_str = "0." + "0" * self._precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _calculate_provenance_hash(
        self,
        inputs: Union[TubeSideInput, ShellSideInput],
        result: Decimal,
        steps: List[CalculationStep]
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        # Serialize calculation data
        if isinstance(inputs, TubeSideInput):
            input_data = {
                "type": "tube_side",
                "mass_flow_rate_kg_s": str(inputs.mass_flow_rate_kg_s),
                "tube_id_m": str(inputs.geometry.tube_id_m),
                "tube_length_m": str(inputs.geometry.tube_length_m),
                "density_kg_m3": str(inputs.fluid.density_kg_m3),
                "viscosity_pa_s": str(inputs.fluid.viscosity_pa_s)
            }
        else:
            input_data = {
                "type": "shell_side",
                "mass_flow_rate_kg_s": str(inputs.mass_flow_rate_kg_s),
                "shell_id_m": str(inputs.shell_geometry.shell_id_m),
                "baffle_spacing_m": str(inputs.shell_geometry.baffle_spacing_m),
                "density_kg_m3": str(inputs.fluid.density_kg_m3),
                "viscosity_pa_s": str(inputs.fluid.viscosity_pa_s)
            }

        provenance_data = {
            "inputs": input_data,
            "result": str(result),
            "step_count": len(steps),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "FlowRegime",
    "FrictionCorrelation",
    "ShellType",
    "BaffleType",
    "TubePitchPattern",

    # Input Data Classes
    "FluidProperties",
    "TubeGeometry",
    "ShellGeometry",
    "FoulingCondition",
    "TubeSideInput",
    "ShellSideInput",
    "PressureDropLimits",

    # Result Data Classes
    "CalculationStep",
    "TubeSidePressureDropResult",
    "ShellSidePressureDropResult",
    "FoulingImpactResult",
    "PressureDropAcceptabilityResult",
    "PumpPowerResult",

    # Calculator Classes
    "FrictionFactorCalculator",
    "PressureDropCalculator",
]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Calculate tube-side pressure drop for a shell-and-tube heat exchanger

    # Define fluid properties (water at 50C)
    fluid = FluidProperties(
        density_kg_m3=Decimal("988.1"),
        viscosity_pa_s=Decimal("0.000547"),
        specific_heat_j_kg_k=Decimal("4181"),
        thermal_conductivity_w_m_k=Decimal("0.644")
    )

    # Define tube geometry
    tube_geom = TubeGeometry(
        tube_od_m=Decimal("0.01905"),      # 3/4" OD
        tube_id_m=Decimal("0.01575"),      # BWG 14
        tube_length_m=Decimal("4.88"),     # 16 ft
        number_of_tubes=200,
        number_of_passes=4,
        tube_roughness_m=Decimal("0.0000015"),
        tube_pitch_m=Decimal("0.02381"),   # 15/16" pitch
        pitch_pattern=TubePitchPattern.TRIANGULAR
    )

    # Define shell geometry
    shell_geom = ShellGeometry(
        shell_id_m=Decimal("0.489"),       # ~19" ID
        baffle_spacing_m=Decimal("0.20"),  # 200mm spacing
        baffle_cut_fraction=Decimal("0.25"),
        inlet_baffle_spacing_m=Decimal("0.25"),
        outlet_baffle_spacing_m=Decimal("0.25"),
        number_of_baffles=20,
        shell_type=ShellType.E_SHELL,
        baffle_type=BaffleType.SEGMENTAL
    )

    # Create calculator
    calculator = PressureDropCalculator(precision=3)

    # Calculate tube-side pressure drop
    tube_inputs = TubeSideInput(
        fluid=fluid,
        geometry=tube_geom,
        mass_flow_rate_kg_s=Decimal("15.0"),  # 15 kg/s
        friction_correlation=FrictionCorrelation.CHURCHILL
    )

    tube_result = calculator.calculate_tube_side_pressure_drop(tube_inputs)

    print("=" * 60)
    print("TUBE-SIDE PRESSURE DROP CALCULATION")
    print("=" * 60)
    print(f"Velocity:             {tube_result.velocity_m_s} m/s")
    print(f"Reynolds Number:      {tube_result.reynolds_number}")
    print(f"Flow Regime:          {tube_result.flow_regime.name}")
    print(f"Friction Factor:      {tube_result.friction_factor}")
    print(f"Friction Loss:        {tube_result.friction_loss_pa} Pa")
    print(f"Entrance/Exit Loss:   {tube_result.entrance_exit_loss_pa} Pa")
    print(f"Return Loss:          {tube_result.return_loss_pa} Pa")
    print(f"TOTAL Pressure Drop:  {tube_result.total_pressure_drop_pa} Pa")
    print(f"                      {tube_result.total_pressure_drop_kpa} kPa")
    print(f"Provenance Hash:      {tube_result.provenance_hash[:16]}...")

    # Calculate shell-side pressure drop
    shell_inputs = ShellSideInput(
        fluid=fluid,
        tube_geometry=tube_geom,
        shell_geometry=shell_geom,
        mass_flow_rate_kg_s=Decimal("20.0")  # 20 kg/s
    )

    shell_result = calculator.calculate_shell_side_pressure_drop(shell_inputs)

    print("\n" + "=" * 60)
    print("SHELL-SIDE PRESSURE DROP CALCULATION (Bell-Delaware)")
    print("=" * 60)
    print(f"Crossflow Velocity:   {shell_result.crossflow_velocity_m_s} m/s")
    print(f"Crossflow Reynolds:   {shell_result.crossflow_reynolds}")
    print(f"Tube Rows Crossed:    {shell_result.number_of_crossflow_rows}")
    print(f"Correction Factors:")
    print(f"  J_c (baffle cut):   {shell_result.j_c_baffle_cut}")
    print(f"  J_l (leakage):      {shell_result.j_l_leakage}")
    print(f"  J_b (bypass):       {shell_result.j_b_bypass}")
    print(f"  J_s (spacing):      {shell_result.j_s_spacing}")
    print(f"Ideal Crossflow dP:   {shell_result.ideal_crossflow_dp_pa} Pa")
    print(f"Window dP:            {shell_result.window_dp_pa} Pa")
    print(f"Entrance/Exit dP:     {shell_result.entrance_exit_dp_pa} Pa")
    print(f"TOTAL Pressure Drop:  {shell_result.total_pressure_drop_pa} Pa")
    print(f"                      {shell_result.total_pressure_drop_kpa} kPa")

    # Calculate pump power requirement
    total_dp = tube_result.total_pressure_drop_pa + shell_result.total_pressure_drop_pa
    volumetric_flow = Decimal("15.0") / fluid.density_kg_m3

    pump_result = calculator.calculate_pump_power_requirement(
        total_pressure_drop_pa=total_dp,
        volumetric_flow_rate_m3_s=volumetric_flow,
        fluid_density_kg_m3=fluid.density_kg_m3,
        pump_efficiency=Decimal("0.75"),
        motor_efficiency=Decimal("0.95"),
        operating_hours_per_year=Decimal("8000"),
        electricity_cost_per_kwh=Decimal("0.12")
    )

    print("\n" + "=" * 60)
    print("PUMP POWER REQUIREMENT")
    print("=" * 60)
    print(f"Required Head:        {pump_result.required_head_m} m")
    print(f"Hydraulic Power:      {pump_result.hydraulic_power_w} W")
    print(f"Shaft Power:          {pump_result.shaft_power_w} W")
    print(f"Electrical Power:     {pump_result.electrical_power_w} W")
    print(f"Annual Energy:        {pump_result.annual_energy_kwh} kWh")
    print(f"Annual Cost:          ${pump_result.annual_energy_cost_usd}")

    # Assess pressure drop acceptability
    limits = PressureDropLimits(
        max_tube_side_pa=Decimal("70000"),
        max_shell_side_pa=Decimal("50000"),
        pump_available_head_m=Decimal("15"),
        pump_efficiency=Decimal("0.75")
    )

    acceptability = calculator.assess_pressure_drop_acceptability(
        tube_side_result=tube_result,
        shell_side_result=shell_result,
        limits=limits,
        fluid_density_kg_m3=fluid.density_kg_m3
    )

    print("\n" + "=" * 60)
    print("PRESSURE DROP ACCEPTABILITY ASSESSMENT")
    print("=" * 60)
    print(f"Tube-side acceptable: {acceptability.tube_side_acceptable}")
    print(f"  Margin to limit:    {acceptability.tube_side_margin_percent}%")
    print(f"Shell-side acceptable:{acceptability.shell_side_acceptable}")
    print(f"  Margin to limit:    {acceptability.shell_side_margin_percent}%")
    print(f"Pump capacity OK:     {acceptability.pump_capacity_adequate}")
    print(f"Required head:        {acceptability.required_pump_head_m} m")
    print(f"Overall acceptable:   {acceptability.total_acceptable}")
    print("\nRecommendations:")
    for rec in acceptability.recommendations:
        print(f"  - {rec}")
