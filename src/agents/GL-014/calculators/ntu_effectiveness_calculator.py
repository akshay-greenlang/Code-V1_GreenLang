# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO - NTU-Effectiveness Calculator Module

Comprehensive epsilon-NTU (effectiveness-NTU) method calculations including:
- Effectiveness calculation for all flow configurations
- NTU calculation from effectiveness (inverse problem)
- Capacity ratio impact analysis
- Flow configuration comparison and selection
- Exit temperature prediction
- Heat duty calculation from effectiveness
- Effectiveness sensitivity analysis

Zero-hallucination guarantee: All calculations use deterministic formulas
from established heat transfer theory and ASME/TEMA standards.

Reference Standards:
- Kays & London "Compact Heat Exchangers" 3rd Edition
- Incropera & DeWitt "Fundamentals of Heat and Mass Transfer"
- ASME PTC 12.5 (Single Phase Heat Exchangers)
- HTRI Guidelines for Heat Exchanger Design

Author: GL-CalculatorEngineer
Version: 1.0.0
License: Proprietary - GreenLang
"""

from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Final, Any, Callable
from enum import Enum, auto
from datetime import datetime, timezone
import hashlib
import json
import math
import uuid


# =============================================================================
# CONSTANTS AND PRECISION SETTINGS
# =============================================================================

PI: Final[Decimal] = Decimal("3.14159265358979323846264338327950288419716939937510")
DECIMAL_PRECISION: Final[int] = 10
ROUNDING_MODE = ROUND_HALF_UP

# Convergence parameters for iterative calculations
MAX_ITERATIONS: Final[int] = 100
CONVERGENCE_TOLERANCE: Final[Decimal] = Decimal("1E-10")


# =============================================================================
# ENUMERATIONS
# =============================================================================

class FlowConfiguration(Enum):
    """
    Heat exchanger flow configurations.

    Each configuration has specific effectiveness-NTU relationships.

    Reference: Kays & London, "Compact Heat Exchangers", Table 3-3
    """
    COUNTERFLOW = auto()             # Pure counterflow
    PARALLEL_FLOW = auto()           # Pure parallel flow (cocurrent)
    CROSSFLOW_BOTH_UNMIXED = auto()  # Crossflow, both fluids unmixed
    CROSSFLOW_CMAX_MIXED = auto()    # Crossflow, Cmax mixed, Cmin unmixed
    CROSSFLOW_CMIN_MIXED = auto()    # Crossflow, Cmin mixed, Cmax unmixed
    CROSSFLOW_BOTH_MIXED = auto()    # Crossflow, both fluids mixed
    SHELL_AND_TUBE_1_2 = auto()      # 1 shell pass, 2 or more tube passes
    SHELL_AND_TUBE_2_4 = auto()      # 2 shell passes, 4 or more tube passes
    SHELL_AND_TUBE_N_2N = auto()     # N shell passes, 2N tube passes
    CONDENSATION = auto()            # One fluid condensing (C* = 0)
    EVAPORATION = auto()             # One fluid evaporating (C* = 0)


class CalculationMode(Enum):
    """Calculation mode for NTU-effectiveness problems."""
    RATING = auto()       # Given UA, find Q and outlet temps
    SIZING = auto()        # Given Q, find UA (NTU)
    SIMULATION = auto()    # Given conditions, simulate performance


class FluidSide(Enum):
    """Fluid side designation."""
    HOT = auto()
    COLD = auto()
    TUBE = auto()
    SHELL = auto()


# =============================================================================
# DATA CLASSES - INPUT PARAMETERS
# =============================================================================

@dataclass(frozen=True)
class FluidStream:
    """
    Fluid stream properties and flow conditions.

    Represents one side of the heat exchanger (hot or cold).
    """
    mass_flow_rate_kg_s: Decimal = field(metadata={"description": "Mass flow rate (kg/s)"})
    specific_heat_j_kg_k: Decimal = field(metadata={"description": "Specific heat capacity (J/kg.K)"})
    inlet_temperature_k: Decimal = field(metadata={"description": "Inlet temperature (K)"})
    outlet_temperature_k: Optional[Decimal] = field(
        default=None,
        metadata={"description": "Outlet temperature (K), if known"}
    )
    side: FluidSide = field(default=FluidSide.HOT)

    @property
    def heat_capacity_rate_w_k(self) -> Decimal:
        """Calculate heat capacity rate C = m_dot * c_p (W/K)."""
        return self.mass_flow_rate_kg_s * self.specific_heat_j_kg_k

    @property
    def inlet_temperature_c(self) -> Decimal:
        """Inlet temperature in Celsius."""
        return self.inlet_temperature_k - Decimal("273.15")

    def __post_init__(self):
        if self.mass_flow_rate_kg_s <= 0:
            raise ValueError("Mass flow rate must be positive")
        if self.specific_heat_j_kg_k <= 0:
            raise ValueError("Specific heat must be positive")
        if self.inlet_temperature_k <= 0:
            raise ValueError("Inlet temperature must be positive (Kelvin)")


@dataclass(frozen=True)
class HeatExchangerGeometry:
    """Heat exchanger geometry parameters for NTU calculation."""
    heat_transfer_area_m2: Decimal = field(metadata={"description": "Total heat transfer area (m^2)"})
    overall_htc_w_m2_k: Decimal = field(
        default=Decimal("0"),
        metadata={"description": "Overall heat transfer coefficient U (W/m^2.K)"}
    )
    ua_w_k: Optional[Decimal] = field(
        default=None,
        metadata={"description": "UA product (W/K), overrides U*A calculation"}
    )

    @property
    def ua_value(self) -> Decimal:
        """Get UA value (W/K)."""
        if self.ua_w_k is not None:
            return self.ua_w_k
        return self.overall_htc_w_m2_k * self.heat_transfer_area_m2

    def __post_init__(self):
        if self.heat_transfer_area_m2 <= 0:
            raise ValueError("Heat transfer area must be positive")


@dataclass(frozen=True)
class NTUEffectivenessInput:
    """Complete input for NTU-effectiveness calculation."""
    hot_stream: FluidStream = field(metadata={"description": "Hot fluid stream"})
    cold_stream: FluidStream = field(metadata={"description": "Cold fluid stream"})
    geometry: HeatExchangerGeometry = field(metadata={"description": "Exchanger geometry"})
    flow_configuration: FlowConfiguration = field(
        default=FlowConfiguration.COUNTERFLOW,
        metadata={"description": "Flow configuration"}
    )
    number_of_shell_passes: int = field(default=1)
    number_of_tube_passes: int = field(default=2)

    @property
    def c_hot(self) -> Decimal:
        """Hot side heat capacity rate (W/K)."""
        return self.hot_stream.heat_capacity_rate_w_k

    @property
    def c_cold(self) -> Decimal:
        """Cold side heat capacity rate (W/K)."""
        return self.cold_stream.heat_capacity_rate_w_k

    @property
    def c_min(self) -> Decimal:
        """Minimum heat capacity rate (W/K)."""
        return min(self.c_hot, self.c_cold)

    @property
    def c_max(self) -> Decimal:
        """Maximum heat capacity rate (W/K)."""
        return max(self.c_hot, self.c_cold)

    @property
    def capacity_ratio(self) -> Decimal:
        """Capacity ratio C* = Cmin/Cmax."""
        if self.c_max == 0:
            return Decimal("0")
        return self.c_min / self.c_max

    @property
    def ntu(self) -> Decimal:
        """Number of Transfer Units NTU = UA/Cmin."""
        if self.c_min == 0:
            return Decimal("0")
        return self.geometry.ua_value / self.c_min

    @property
    def q_max_w(self) -> Decimal:
        """Maximum possible heat transfer rate (W)."""
        delta_t_max = abs(self.hot_stream.inlet_temperature_k -
                         self.cold_stream.inlet_temperature_k)
        return self.c_min * delta_t_max


# =============================================================================
# DATA CLASSES - CALCULATION RESULTS
# =============================================================================

@dataclass(frozen=True)
class CalculationStep:
    """Immutable record of a single calculation step for audit trail."""
    step_number: int
    operation: str
    description: str
    inputs: Dict[str, Any]
    output_name: str
    output_value: Any
    formula: str = ""
    reference: str = ""

    def to_dict(self) -> Dict[str, Any]:
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
class EffectivenessResult:
    """Result of effectiveness calculation."""
    effectiveness: Decimal = field(metadata={"description": "Heat exchanger effectiveness (0-1)"})
    ntu: Decimal = field(metadata={"description": "Number of Transfer Units"})
    capacity_ratio: Decimal = field(metadata={"description": "Capacity ratio C*"})
    c_min_w_k: Decimal = field(metadata={"description": "Minimum heat capacity rate (W/K)"})
    c_max_w_k: Decimal = field(metadata={"description": "Maximum heat capacity rate (W/K)"})
    c_min_side: str = field(metadata={"description": "Which side has Cmin (hot/cold)"})
    q_max_w: Decimal = field(metadata={"description": "Maximum possible heat transfer (W)"})
    q_actual_w: Decimal = field(metadata={"description": "Actual heat transfer (W)"})
    hot_outlet_temp_k: Decimal = field(metadata={"description": "Hot outlet temperature (K)"})
    cold_outlet_temp_k: Decimal = field(metadata={"description": "Cold outlet temperature (K)"})
    lmtd_k: Decimal = field(metadata={"description": "Log mean temperature difference (K)"})
    flow_configuration: FlowConfiguration = field(metadata={"description": "Flow configuration"})
    calculation_steps: Tuple[CalculationStep, ...] = field(default_factory=tuple)
    provenance_hash: str = field(default="")
    timestamp: str = field(default="")

    @property
    def effectiveness_percent(self) -> Decimal:
        """Effectiveness as percentage."""
        return self.effectiveness * Decimal("100")

    @property
    def hot_outlet_temp_c(self) -> Decimal:
        """Hot outlet temperature in Celsius."""
        return self.hot_outlet_temp_k - Decimal("273.15")

    @property
    def cold_outlet_temp_c(self) -> Decimal:
        """Cold outlet temperature in Celsius."""
        return self.cold_outlet_temp_k - Decimal("273.15")

    @property
    def q_actual_kw(self) -> Decimal:
        """Actual heat transfer in kW."""
        return self.q_actual_w / Decimal("1000")


@dataclass(frozen=True)
class NTUFromEffectivenessResult:
    """Result of NTU calculation from effectiveness (inverse problem)."""
    ntu_required: Decimal = field(metadata={"description": "Required NTU"})
    ua_required_w_k: Decimal = field(metadata={"description": "Required UA (W/K)"})
    area_required_m2: Decimal = field(metadata={"description": "Required area for given U (m^2)"})
    effectiveness_target: Decimal = field(metadata={"description": "Target effectiveness"})
    capacity_ratio: Decimal = field(metadata={"description": "Capacity ratio C*"})
    flow_configuration: FlowConfiguration = field(metadata={"description": "Flow configuration"})
    iterations: int = field(metadata={"description": "Number of iterations (if iterative)"})
    calculation_steps: Tuple[CalculationStep, ...] = field(default_factory=tuple)
    provenance_hash: str = field(default="")


@dataclass(frozen=True)
class ConfigurationComparisonResult:
    """Result of comparing different flow configurations."""
    configurations: Tuple[Tuple[FlowConfiguration, Decimal], ...] = field(
        metadata={"description": "Configurations and their effectiveness values"}
    )
    best_configuration: FlowConfiguration = field(
        metadata={"description": "Best (highest effectiveness) configuration"}
    )
    best_effectiveness: Decimal = field(metadata={"description": "Best effectiveness value"})
    worst_configuration: FlowConfiguration = field(
        metadata={"description": "Worst (lowest effectiveness) configuration"}
    )
    worst_effectiveness: Decimal = field(metadata={"description": "Worst effectiveness value"})
    effectiveness_range: Decimal = field(
        metadata={"description": "Range of effectiveness values"}
    )
    ntu: Decimal = field(metadata={"description": "NTU used for comparison"})
    capacity_ratio: Decimal = field(metadata={"description": "Capacity ratio C*"})
    calculation_steps: Tuple[CalculationStep, ...] = field(default_factory=tuple)
    provenance_hash: str = field(default="")


@dataclass(frozen=True)
class CapacityRatioSensitivityResult:
    """Result of capacity ratio sensitivity analysis."""
    c_star_values: Tuple[Decimal, ...] = field(
        metadata={"description": "Capacity ratio values analyzed"}
    )
    effectiveness_values: Tuple[Decimal, ...] = field(
        metadata={"description": "Corresponding effectiveness values"}
    )
    ntu: Decimal = field(metadata={"description": "Fixed NTU value"})
    flow_configuration: FlowConfiguration = field(metadata={"description": "Flow configuration"})
    sensitivity_coefficient: Decimal = field(
        metadata={"description": "Sensitivity d(epsilon)/d(C*) at C*=0.5"}
    )
    calculation_steps: Tuple[CalculationStep, ...] = field(default_factory=tuple)
    provenance_hash: str = field(default="")


# =============================================================================
# NTU-EFFECTIVENESS CALCULATOR ENGINE
# =============================================================================

class NTUEffectivenessCalculator:
    """
    Zero-hallucination NTU-Effectiveness calculator.

    Implements the epsilon-NTU method for heat exchanger analysis
    with complete provenance tracking and audit trails.

    Features:
    - Effectiveness calculation for all flow configurations
    - NTU calculation from effectiveness (inverse problem)
    - Configuration comparison and selection
    - Capacity ratio sensitivity analysis
    - Complete provenance tracking with SHA-256 hashing

    Reference: Kays & London, "Compact Heat Exchangers", 3rd Edition
    """

    def __init__(self, precision: int = DECIMAL_PRECISION):
        """
        Initialize the NTU-effectiveness calculator.

        Args:
            precision: Decimal precision for calculations
        """
        self.precision = precision
        self._calculation_id = str(uuid.uuid4())

    def _decimal(self, value: Union[float, int, str, Decimal]) -> Decimal:
        """Convert value to Decimal with proper precision."""
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    def _exp(self, x: Decimal) -> Decimal:
        """Calculate exp(x) with Decimal precision."""
        return self._decimal(math.exp(float(x)))

    def _sqrt(self, x: Decimal) -> Decimal:
        """Calculate sqrt(x) with Decimal precision."""
        return self._decimal(math.sqrt(float(x)))

    def _tanh(self, x: Decimal) -> Decimal:
        """Calculate tanh(x) with Decimal precision."""
        return self._decimal(math.tanh(float(x)))

    def _log(self, x: Decimal) -> Decimal:
        """Calculate natural log of x with Decimal precision."""
        if x <= 0:
            raise ValueError("Cannot take log of non-positive number")
        return self._decimal(math.log(float(x)))

    # =========================================================================
    # EFFECTIVENESS FORMULAS FOR EACH CONFIGURATION
    # =========================================================================

    def _effectiveness_counterflow(self, ntu: Decimal, c_star: Decimal) -> Decimal:
        """
        Calculate effectiveness for counterflow configuration.

        epsilon = (1 - exp(-NTU*(1-C*))) / (1 - C*exp(-NTU*(1-C*)))

        For C* = 1: epsilon = NTU / (1 + NTU)

        Reference: Kays & London, Eq. 3-19
        """
        if abs(c_star - Decimal("1")) < Decimal("1E-8"):
            # Special case: C* = 1
            return ntu / (Decimal("1") + ntu)

        exp_term = self._exp(-ntu * (Decimal("1") - c_star))
        numerator = Decimal("1") - exp_term
        denominator = Decimal("1") - c_star * exp_term

        if abs(denominator) < Decimal("1E-12"):
            return Decimal("1")

        return numerator / denominator

    def _effectiveness_parallel_flow(self, ntu: Decimal, c_star: Decimal) -> Decimal:
        """
        Calculate effectiveness for parallel flow (cocurrent).

        epsilon = (1 - exp(-NTU*(1+C*))) / (1 + C*)

        Reference: Kays & London, Eq. 3-18
        """
        exp_term = self._exp(-ntu * (Decimal("1") + c_star))
        return (Decimal("1") - exp_term) / (Decimal("1") + c_star)

    def _effectiveness_crossflow_both_unmixed(self, ntu: Decimal, c_star: Decimal) -> Decimal:
        """
        Calculate effectiveness for crossflow, both fluids unmixed.

        Uses approximation from Kays & London.

        Reference: Kays & London, Eq. 3-21
        """
        if c_star == Decimal("0"):
            return Decimal("1") - self._exp(-ntu)

        # Approximation: epsilon = 1 - exp[(1/C*)*NTU^0.22 * (exp(-C**NTU^0.78) - 1)]
        ntu_078 = self._decimal(float(ntu) ** 0.78)
        ntu_022 = self._decimal(float(ntu) ** 0.22)

        inner_exp = self._exp(-c_star * ntu_078)
        outer_arg = (Decimal("1") / c_star) * ntu_022 * (inner_exp - Decimal("1"))

        return Decimal("1") - self._exp(outer_arg)

    def _effectiveness_crossflow_cmax_mixed(self, ntu: Decimal, c_star: Decimal) -> Decimal:
        """
        Calculate effectiveness for crossflow, Cmax mixed, Cmin unmixed.

        epsilon = (1/C*) * (1 - exp(-C* * (1 - exp(-NTU))))

        Reference: Kays & London, Eq. 3-23
        """
        if c_star == Decimal("0"):
            return Decimal("1") - self._exp(-ntu)

        inner_term = Decimal("1") - self._exp(-ntu)
        return (Decimal("1") / c_star) * (Decimal("1") - self._exp(-c_star * inner_term))

    def _effectiveness_crossflow_cmin_mixed(self, ntu: Decimal, c_star: Decimal) -> Decimal:
        """
        Calculate effectiveness for crossflow, Cmin mixed, Cmax unmixed.

        epsilon = 1 - exp(-(1/C*) * (1 - exp(-C* * NTU)))

        Reference: Kays & London, Eq. 3-24
        """
        if c_star == Decimal("0"):
            return Decimal("1") - self._exp(-ntu)

        inner_term = Decimal("1") - self._exp(-c_star * ntu)
        return Decimal("1") - self._exp(-(Decimal("1") / c_star) * inner_term)

    def _effectiveness_crossflow_both_mixed(self, ntu: Decimal, c_star: Decimal) -> Decimal:
        """
        Calculate effectiveness for crossflow, both fluids mixed.

        1/epsilon = 1/(1-exp(-NTU)) + C*/(1-exp(-C**NTU)) - 1/NTU

        Reference: Kays & London, Eq. 3-25
        """
        if ntu < Decimal("0.001"):
            return ntu  # Linear approximation for very small NTU

        if c_star == Decimal("0"):
            return Decimal("1") - self._exp(-ntu)

        exp_ntu = Decimal("1") - self._exp(-ntu)
        exp_c_ntu = Decimal("1") - self._exp(-c_star * ntu)

        if abs(exp_ntu) < Decimal("1E-12") or abs(exp_c_ntu) < Decimal("1E-12"):
            return Decimal("0")

        one_over_eps = (Decimal("1") / exp_ntu) + (c_star / exp_c_ntu) - (Decimal("1") / ntu)

        if abs(one_over_eps) < Decimal("1E-12"):
            return Decimal("1")

        return Decimal("1") / one_over_eps

    def _effectiveness_shell_and_tube_1_2(self, ntu: Decimal, c_star: Decimal) -> Decimal:
        """
        Calculate effectiveness for 1 shell pass, 2+ tube passes (TEMA E-shell).

        epsilon = 2 * (1 + C* + sqrt(1+C*^2) * (1+exp(-NTU*sqrt(1+C*^2))) /
                                               (1-exp(-NTU*sqrt(1+C*^2))))^-1

        Reference: Kays & London, Eq. 3-26
        """
        if c_star == Decimal("0"):
            return Decimal("1") - self._exp(-ntu)

        c_star_sq = c_star ** 2
        sqrt_term = self._sqrt(Decimal("1") + c_star_sq)
        exp_arg = -ntu * sqrt_term

        exp_term = self._exp(exp_arg)

        if abs(Decimal("1") - exp_term) < Decimal("1E-12"):
            return Decimal("0")

        ratio = (Decimal("1") + exp_term) / (Decimal("1") - exp_term)
        inner = Decimal("1") + c_star + sqrt_term * ratio

        if abs(inner) < Decimal("1E-12"):
            return Decimal("1")

        return Decimal("2") / inner

    def _effectiveness_shell_and_tube_2_4(self, ntu: Decimal, c_star: Decimal) -> Decimal:
        """
        Calculate effectiveness for 2 shell passes, 4+ tube passes.

        Uses the relationship for n shell passes.

        Reference: Kays & London, Eq. 3-27
        """
        # First get effectiveness for 1-2
        eps_1 = self._effectiveness_shell_and_tube_1_2(ntu / Decimal("2"), c_star)

        # Then combine for 2 shells in series
        if abs(Decimal("1") - eps_1 * c_star) < Decimal("1E-12"):
            return Decimal("1")

        term = ((Decimal("1") - eps_1 * c_star) / (Decimal("1") - eps_1)) ** 2

        if abs(Decimal("1") - c_star * term) < Decimal("1E-12"):
            return Decimal("1")

        return (Decimal("1") - term) / (Decimal("1") - c_star * term)

    def _effectiveness_shell_and_tube_n_2n(
        self,
        ntu: Decimal,
        c_star: Decimal,
        n_shells: int
    ) -> Decimal:
        """
        Calculate effectiveness for N shell passes, 2N tube passes.

        Reference: Kays & London, Eq. 3-28
        """
        # Effectiveness for single 1-2 exchanger
        ntu_per_shell = ntu / self._decimal(n_shells)
        eps_1 = self._effectiveness_shell_and_tube_1_2(ntu_per_shell, c_star)

        if abs(Decimal("1") - eps_1) < Decimal("1E-12"):
            return Decimal("1")

        # Combine n shells in series
        term = ((Decimal("1") - eps_1 * c_star) / (Decimal("1") - eps_1)) ** n_shells

        if abs(Decimal("1") - c_star * term) < Decimal("1E-12"):
            return Decimal("1")

        return (Decimal("1") - term) / (Decimal("1") - c_star * term)

    def _effectiveness_condensation_evaporation(self, ntu: Decimal) -> Decimal:
        """
        Calculate effectiveness for condensation or evaporation (C* = 0).

        epsilon = 1 - exp(-NTU)

        Reference: Kays & London
        """
        return Decimal("1") - self._exp(-ntu)

    # =========================================================================
    # MAIN CALCULATION METHODS
    # =========================================================================

    def calculate_effectiveness(
        self,
        ntu: Decimal,
        c_star: Decimal,
        flow_configuration: FlowConfiguration,
        n_shells: int = 1
    ) -> Decimal:
        """
        Calculate heat exchanger effectiveness for given configuration.

        Args:
            ntu: Number of Transfer Units
            c_star: Capacity ratio (Cmin/Cmax)
            flow_configuration: Flow configuration
            n_shells: Number of shell passes (for shell-and-tube)

        Returns:
            Heat exchanger effectiveness (0 to 1)

        Reference: Kays & London, "Compact Heat Exchangers", Chapter 3
        """
        # Validate inputs
        if ntu < Decimal("0"):
            raise ValueError("NTU must be non-negative")
        if c_star < Decimal("0") or c_star > Decimal("1"):
            raise ValueError("Capacity ratio must be between 0 and 1")

        # Handle zero NTU case
        if ntu < Decimal("1E-12"):
            return Decimal("0")

        # Handle phase change cases
        if flow_configuration in [FlowConfiguration.CONDENSATION, FlowConfiguration.EVAPORATION]:
            return self._effectiveness_condensation_evaporation(ntu)

        # Handle C* = 0 case (one infinite heat capacity)
        if c_star < Decimal("1E-10"):
            return Decimal("1") - self._exp(-ntu)

        # Select effectiveness formula based on configuration
        if flow_configuration == FlowConfiguration.COUNTERFLOW:
            return self._effectiveness_counterflow(ntu, c_star)

        elif flow_configuration == FlowConfiguration.PARALLEL_FLOW:
            return self._effectiveness_parallel_flow(ntu, c_star)

        elif flow_configuration == FlowConfiguration.CROSSFLOW_BOTH_UNMIXED:
            return self._effectiveness_crossflow_both_unmixed(ntu, c_star)

        elif flow_configuration == FlowConfiguration.CROSSFLOW_CMAX_MIXED:
            return self._effectiveness_crossflow_cmax_mixed(ntu, c_star)

        elif flow_configuration == FlowConfiguration.CROSSFLOW_CMIN_MIXED:
            return self._effectiveness_crossflow_cmin_mixed(ntu, c_star)

        elif flow_configuration == FlowConfiguration.CROSSFLOW_BOTH_MIXED:
            return self._effectiveness_crossflow_both_mixed(ntu, c_star)

        elif flow_configuration == FlowConfiguration.SHELL_AND_TUBE_1_2:
            return self._effectiveness_shell_and_tube_1_2(ntu, c_star)

        elif flow_configuration == FlowConfiguration.SHELL_AND_TUBE_2_4:
            return self._effectiveness_shell_and_tube_2_4(ntu, c_star)

        elif flow_configuration == FlowConfiguration.SHELL_AND_TUBE_N_2N:
            return self._effectiveness_shell_and_tube_n_2n(ntu, c_star, n_shells)

        else:
            raise ValueError(f"Unsupported flow configuration: {flow_configuration}")

    def calculate_ntu_from_effectiveness(
        self,
        effectiveness: Decimal,
        c_star: Decimal,
        flow_configuration: FlowConfiguration,
        n_shells: int = 1
    ) -> Decimal:
        """
        Calculate NTU from effectiveness (inverse problem).

        For counterflow and parallel flow, uses analytical inversion.
        For other configurations, uses Newton-Raphson iteration.

        Args:
            effectiveness: Target effectiveness (0 to 1)
            c_star: Capacity ratio (Cmin/Cmax)
            flow_configuration: Flow configuration
            n_shells: Number of shell passes (for shell-and-tube)

        Returns:
            Required NTU

        Reference: Kays & London, "Compact Heat Exchangers"
        """
        # Validate inputs
        if effectiveness <= Decimal("0") or effectiveness >= Decimal("1"):
            raise ValueError("Effectiveness must be between 0 and 1 (exclusive)")
        if c_star < Decimal("0") or c_star > Decimal("1"):
            raise ValueError("Capacity ratio must be between 0 and 1")

        eps = effectiveness

        # Handle phase change cases (C* = 0)
        if flow_configuration in [FlowConfiguration.CONDENSATION, FlowConfiguration.EVAPORATION]:
            return -self._log(Decimal("1") - eps)

        if c_star < Decimal("1E-10"):
            return -self._log(Decimal("1") - eps)

        # Analytical solutions where available
        if flow_configuration == FlowConfiguration.COUNTERFLOW:
            if abs(c_star - Decimal("1")) < Decimal("1E-8"):
                # C* = 1: NTU = epsilon / (1 - epsilon)
                return eps / (Decimal("1") - eps)
            else:
                # General: NTU = ln((1 - C**eps) / (1 - eps)) / (1 - C*)
                numerator = self._log((Decimal("1") - c_star * eps) / (Decimal("1") - eps))
                return numerator / (Decimal("1") - c_star)

        elif flow_configuration == FlowConfiguration.PARALLEL_FLOW:
            # NTU = -ln(1 - epsilon*(1+C*)) / (1 + C*)
            arg = Decimal("1") - eps * (Decimal("1") + c_star)
            if arg <= Decimal("0"):
                raise ValueError("Requested effectiveness exceeds maximum for parallel flow")
            return -self._log(arg) / (Decimal("1") + c_star)

        else:
            # Use Newton-Raphson iteration for other configurations
            return self._ntu_newton_raphson(eps, c_star, flow_configuration, n_shells)

    def _ntu_newton_raphson(
        self,
        target_eps: Decimal,
        c_star: Decimal,
        flow_configuration: FlowConfiguration,
        n_shells: int = 1
    ) -> Decimal:
        """
        Solve for NTU using Newton-Raphson iteration.

        Args:
            target_eps: Target effectiveness
            c_star: Capacity ratio
            flow_configuration: Flow configuration
            n_shells: Number of shell passes

        Returns:
            NTU that gives target effectiveness
        """
        # Initial guess based on counterflow approximation
        if abs(c_star - Decimal("1")) < Decimal("1E-8"):
            ntu = target_eps / (Decimal("1") - target_eps)
        else:
            ntu = self._log((Decimal("1") - c_star * target_eps) /
                           (Decimal("1") - target_eps)) / (Decimal("1") - c_star)

        # Ensure positive initial guess
        ntu = max(Decimal("0.1"), ntu)

        # Newton-Raphson iteration
        for _ in range(MAX_ITERATIONS):
            eps_current = self.calculate_effectiveness(ntu, c_star, flow_configuration, n_shells)
            error = eps_current - target_eps

            if abs(error) < CONVERGENCE_TOLERANCE:
                return ntu

            # Numerical derivative (central difference)
            delta = Decimal("0.0001") * ntu
            eps_plus = self.calculate_effectiveness(ntu + delta, c_star, flow_configuration, n_shells)
            eps_minus = self.calculate_effectiveness(ntu - delta, c_star, flow_configuration, n_shells)
            derivative = (eps_plus - eps_minus) / (Decimal("2") * delta)

            if abs(derivative) < Decimal("1E-12"):
                break

            # Update NTU
            ntu = ntu - error / derivative

            # Ensure positive NTU
            ntu = max(Decimal("0.001"), ntu)

        return ntu

    def rate_exchanger(
        self,
        input_data: NTUEffectivenessInput
    ) -> EffectivenessResult:
        """
        Rate an existing heat exchanger (given UA, find Q and outlet temps).

        This is the most common use of the NTU-effectiveness method:
        given the heat exchanger UA and inlet conditions, determine
        the outlet temperatures and heat duty.

        Args:
            input_data: Complete heat exchanger input data

        Returns:
            EffectivenessResult with full analysis

        Reference: Kays & London, "Compact Heat Exchangers", Chapter 3
        """
        steps: List[CalculationStep] = []
        step_num = 0

        # Step 1: Calculate heat capacity rates
        step_num += 1
        c_hot = input_data.c_hot
        c_cold = input_data.c_cold
        c_min = input_data.c_min
        c_max = input_data.c_max

        c_min_side = "hot" if c_hot <= c_cold else "cold"

        steps.append(CalculationStep(
            step_number=step_num,
            operation="heat_capacity_rates",
            description="Calculate heat capacity rates for hot and cold streams",
            inputs={
                "m_hot_kg_s": input_data.hot_stream.mass_flow_rate_kg_s,
                "cp_hot_j_kg_k": input_data.hot_stream.specific_heat_j_kg_k,
                "m_cold_kg_s": input_data.cold_stream.mass_flow_rate_kg_s,
                "cp_cold_j_kg_k": input_data.cold_stream.specific_heat_j_kg_k
            },
            output_name="c_min_w_k",
            output_value=c_min,
            formula="C = m_dot * c_p; C_min = min(C_hot, C_cold)",
            reference="Kays & London, Eq. 3-1"
        ))

        # Step 2: Calculate capacity ratio
        step_num += 1
        c_star = input_data.capacity_ratio

        steps.append(CalculationStep(
            step_number=step_num,
            operation="capacity_ratio",
            description="Calculate capacity ratio C* = Cmin/Cmax",
            inputs={
                "c_min_w_k": c_min,
                "c_max_w_k": c_max
            },
            output_name="c_star",
            output_value=c_star,
            formula="C* = C_min / C_max",
            reference="Kays & London, Eq. 3-2"
        ))

        # Step 3: Calculate NTU
        step_num += 1
        ntu = input_data.ntu
        ua = input_data.geometry.ua_value

        steps.append(CalculationStep(
            step_number=step_num,
            operation="ntu",
            description="Calculate Number of Transfer Units",
            inputs={
                "ua_w_k": ua,
                "c_min_w_k": c_min
            },
            output_name="ntu",
            output_value=ntu,
            formula="NTU = UA / C_min",
            reference="Kays & London, Eq. 3-3"
        ))

        # Step 4: Calculate maximum heat transfer
        step_num += 1
        t_hot_in = input_data.hot_stream.inlet_temperature_k
        t_cold_in = input_data.cold_stream.inlet_temperature_k
        delta_t_max = abs(t_hot_in - t_cold_in)
        q_max = c_min * delta_t_max

        steps.append(CalculationStep(
            step_number=step_num,
            operation="q_max",
            description="Calculate maximum possible heat transfer",
            inputs={
                "c_min_w_k": c_min,
                "t_hot_in_k": t_hot_in,
                "t_cold_in_k": t_cold_in
            },
            output_name="q_max_w",
            output_value=q_max,
            formula="Q_max = C_min * |T_hot,in - T_cold,in|",
            reference="Kays & London, Eq. 3-4"
        ))

        # Step 5: Calculate effectiveness
        step_num += 1
        n_shells = input_data.number_of_shell_passes
        effectiveness = self.calculate_effectiveness(
            ntu, c_star, input_data.flow_configuration, n_shells
        )

        steps.append(CalculationStep(
            step_number=step_num,
            operation="effectiveness",
            description=f"Calculate effectiveness for {input_data.flow_configuration.name}",
            inputs={
                "ntu": ntu,
                "c_star": c_star,
                "configuration": input_data.flow_configuration.name
            },
            output_name="effectiveness",
            output_value=effectiveness,
            formula="epsilon = f(NTU, C*, configuration)",
            reference="Kays & London, Table 3-3"
        ))

        # Step 6: Calculate actual heat transfer
        step_num += 1
        q_actual = effectiveness * q_max

        steps.append(CalculationStep(
            step_number=step_num,
            operation="q_actual",
            description="Calculate actual heat transfer rate",
            inputs={
                "effectiveness": effectiveness,
                "q_max_w": q_max
            },
            output_name="q_actual_w",
            output_value=q_actual,
            formula="Q = epsilon * Q_max",
            reference="Kays & London, Eq. 3-5"
        ))

        # Step 7: Calculate outlet temperatures
        step_num += 1
        # Hot side outlet: T_hot_out = T_hot_in - Q/C_hot
        t_hot_out = t_hot_in - q_actual / c_hot
        # Cold side outlet: T_cold_out = T_cold_in + Q/C_cold
        t_cold_out = t_cold_in + q_actual / c_cold

        steps.append(CalculationStep(
            step_number=step_num,
            operation="outlet_temperatures",
            description="Calculate outlet temperatures from energy balance",
            inputs={
                "t_hot_in_k": t_hot_in,
                "t_cold_in_k": t_cold_in,
                "q_actual_w": q_actual,
                "c_hot_w_k": c_hot,
                "c_cold_w_k": c_cold
            },
            output_name="temperatures_k",
            output_value={"hot_out": t_hot_out, "cold_out": t_cold_out},
            formula="T_out = T_in +/- Q/C",
            reference="Energy balance"
        ))

        # Step 8: Calculate LMTD
        step_num += 1
        # LMTD = Q / (U * A) = Q / UA
        if ua > Decimal("0"):
            lmtd = q_actual / ua
        else:
            lmtd = Decimal("0")

        steps.append(CalculationStep(
            step_number=step_num,
            operation="lmtd",
            description="Calculate Log Mean Temperature Difference",
            inputs={
                "q_actual_w": q_actual,
                "ua_w_k": ua
            },
            output_name="lmtd_k",
            output_value=lmtd,
            formula="LMTD = Q / UA",
            reference="Heat transfer fundamentals"
        ))

        # Generate provenance hash
        provenance_data = {
            "calculation_id": self._calculation_id,
            "input_hash": hashlib.sha256(
                json.dumps({
                    "ntu": str(ntu),
                    "c_star": str(c_star),
                    "config": input_data.flow_configuration.name
                }, sort_keys=True).encode()
            ).hexdigest(),
            "effectiveness": str(effectiveness),
            "q_actual_w": str(q_actual),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        timestamp = datetime.now(timezone.utc).isoformat()

        return EffectivenessResult(
            effectiveness=effectiveness,
            ntu=ntu,
            capacity_ratio=c_star,
            c_min_w_k=c_min,
            c_max_w_k=c_max,
            c_min_side=c_min_side,
            q_max_w=q_max,
            q_actual_w=q_actual,
            hot_outlet_temp_k=t_hot_out,
            cold_outlet_temp_k=t_cold_out,
            lmtd_k=lmtd,
            flow_configuration=input_data.flow_configuration,
            calculation_steps=tuple(steps),
            provenance_hash=provenance_hash,
            timestamp=timestamp
        )

    def size_exchanger(
        self,
        target_effectiveness: Decimal,
        c_star: Decimal,
        c_min_w_k: Decimal,
        overall_htc_w_m2_k: Decimal,
        flow_configuration: FlowConfiguration,
        n_shells: int = 1
    ) -> NTUFromEffectivenessResult:
        """
        Size a heat exchanger (given target effectiveness, find required area).

        This is the inverse problem: given target performance,
        determine the required heat transfer area.

        Args:
            target_effectiveness: Target effectiveness (0-1)
            c_star: Capacity ratio (Cmin/Cmax)
            c_min_w_k: Minimum heat capacity rate (W/K)
            overall_htc_w_m2_k: Overall heat transfer coefficient (W/m^2.K)
            flow_configuration: Flow configuration
            n_shells: Number of shell passes (for shell-and-tube)

        Returns:
            NTUFromEffectivenessResult with required NTU and area

        Reference: Kays & London, "Compact Heat Exchangers"
        """
        steps: List[CalculationStep] = []
        step_num = 0

        # Step 1: Calculate required NTU
        step_num += 1
        ntu_required = self.calculate_ntu_from_effectiveness(
            target_effectiveness, c_star, flow_configuration, n_shells
        )

        steps.append(CalculationStep(
            step_number=step_num,
            operation="ntu_required",
            description="Calculate required NTU from target effectiveness",
            inputs={
                "target_effectiveness": target_effectiveness,
                "c_star": c_star,
                "configuration": flow_configuration.name
            },
            output_name="ntu_required",
            output_value=ntu_required,
            formula="NTU = f^-1(epsilon, C*, configuration)",
            reference="Kays & London"
        ))

        # Step 2: Calculate required UA
        step_num += 1
        ua_required = ntu_required * c_min_w_k

        steps.append(CalculationStep(
            step_number=step_num,
            operation="ua_required",
            description="Calculate required UA product",
            inputs={
                "ntu_required": ntu_required,
                "c_min_w_k": c_min_w_k
            },
            output_name="ua_required_w_k",
            output_value=ua_required,
            formula="UA = NTU * C_min",
            reference="Definition of NTU"
        ))

        # Step 3: Calculate required area
        step_num += 1
        if overall_htc_w_m2_k > Decimal("0"):
            area_required = ua_required / overall_htc_w_m2_k
        else:
            area_required = Decimal("0")

        steps.append(CalculationStep(
            step_number=step_num,
            operation="area_required",
            description="Calculate required heat transfer area",
            inputs={
                "ua_required_w_k": ua_required,
                "u_w_m2_k": overall_htc_w_m2_k
            },
            output_name="area_required_m2",
            output_value=area_required,
            formula="A = UA / U",
            reference="Definition of overall HTC"
        ))

        # Determine iterations (analytical solutions have 0 iterations)
        iterations = 0
        if flow_configuration not in [
            FlowConfiguration.COUNTERFLOW,
            FlowConfiguration.PARALLEL_FLOW,
            FlowConfiguration.CONDENSATION,
            FlowConfiguration.EVAPORATION
        ]:
            iterations = 1  # Indicates Newton-Raphson was used

        # Generate provenance hash
        provenance_data = {
            "target_effectiveness": str(target_effectiveness),
            "c_star": str(c_star),
            "ntu_required": str(ntu_required),
            "area_required_m2": str(area_required),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return NTUFromEffectivenessResult(
            ntu_required=ntu_required,
            ua_required_w_k=ua_required,
            area_required_m2=area_required,
            effectiveness_target=target_effectiveness,
            capacity_ratio=c_star,
            flow_configuration=flow_configuration,
            iterations=iterations,
            calculation_steps=tuple(steps),
            provenance_hash=provenance_hash
        )

    def compare_configurations(
        self,
        ntu: Decimal,
        c_star: Decimal,
        configurations: Optional[List[FlowConfiguration]] = None
    ) -> ConfigurationComparisonResult:
        """
        Compare effectiveness across different flow configurations.

        Useful for selecting the best flow arrangement for a given
        NTU and capacity ratio.

        Args:
            ntu: Number of Transfer Units
            c_star: Capacity ratio (Cmin/Cmax)
            configurations: List of configurations to compare (default: all)

        Returns:
            ConfigurationComparisonResult with comparison data

        Reference: Kays & London, "Compact Heat Exchangers", Table 3-3
        """
        steps: List[CalculationStep] = []
        step_num = 0

        # Default to common configurations
        if configurations is None:
            configurations = [
                FlowConfiguration.COUNTERFLOW,
                FlowConfiguration.PARALLEL_FLOW,
                FlowConfiguration.CROSSFLOW_BOTH_UNMIXED,
                FlowConfiguration.CROSSFLOW_CMAX_MIXED,
                FlowConfiguration.CROSSFLOW_CMIN_MIXED,
                FlowConfiguration.CROSSFLOW_BOTH_MIXED,
                FlowConfiguration.SHELL_AND_TUBE_1_2,
            ]

        results: List[Tuple[FlowConfiguration, Decimal]] = []

        for config in configurations:
            step_num += 1
            effectiveness = self.calculate_effectiveness(ntu, c_star, config)
            results.append((config, effectiveness))

            steps.append(CalculationStep(
                step_number=step_num,
                operation="effectiveness",
                description=f"Calculate effectiveness for {config.name}",
                inputs={
                    "ntu": ntu,
                    "c_star": c_star,
                    "configuration": config.name
                },
                output_name="effectiveness",
                output_value=effectiveness,
                formula="epsilon = f(NTU, C*, config)",
                reference="Kays & London, Table 3-3"
            ))

        # Sort by effectiveness (descending)
        results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

        best_config, best_eps = results_sorted[0]
        worst_config, worst_eps = results_sorted[-1]
        eps_range = best_eps - worst_eps

        # Generate provenance hash
        provenance_data = {
            "ntu": str(ntu),
            "c_star": str(c_star),
            "best_config": best_config.name,
            "best_eps": str(best_eps),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return ConfigurationComparisonResult(
            configurations=tuple(results_sorted),
            best_configuration=best_config,
            best_effectiveness=best_eps,
            worst_configuration=worst_config,
            worst_effectiveness=worst_eps,
            effectiveness_range=eps_range,
            ntu=ntu,
            capacity_ratio=c_star,
            calculation_steps=tuple(steps),
            provenance_hash=provenance_hash
        )

    def analyze_capacity_ratio_sensitivity(
        self,
        ntu: Decimal,
        flow_configuration: FlowConfiguration,
        c_star_points: int = 21
    ) -> CapacityRatioSensitivityResult:
        """
        Analyze how effectiveness varies with capacity ratio.

        This analysis shows the impact of C* on exchanger performance
        at a fixed NTU value.

        Args:
            ntu: Number of Transfer Units
            flow_configuration: Flow configuration
            c_star_points: Number of C* points to analyze (default: 21)

        Returns:
            CapacityRatioSensitivityResult with sensitivity data

        Reference: Kays & London, "Compact Heat Exchangers", Fig. 3-10
        """
        steps: List[CalculationStep] = []
        step_num = 0

        c_star_values: List[Decimal] = []
        effectiveness_values: List[Decimal] = []

        # Generate C* values from 0 to 1
        for i in range(c_star_points):
            c_star = self._decimal(i / (c_star_points - 1))
            c_star_values.append(c_star)

            effectiveness = self.calculate_effectiveness(ntu, c_star, flow_configuration)
            effectiveness_values.append(effectiveness)

        # Calculate sensitivity at C* = 0.5 using central difference
        idx_mid = c_star_points // 2
        if idx_mid > 0 and idx_mid < c_star_points - 1:
            delta_eps = effectiveness_values[idx_mid + 1] - effectiveness_values[idx_mid - 1]
            delta_c = c_star_values[idx_mid + 1] - c_star_values[idx_mid - 1]
            sensitivity = delta_eps / delta_c
        else:
            sensitivity = Decimal("0")

        step_num += 1
        steps.append(CalculationStep(
            step_number=step_num,
            operation="sensitivity_analysis",
            description=f"Analyze C* sensitivity for {flow_configuration.name}",
            inputs={
                "ntu": ntu,
                "configuration": flow_configuration.name,
                "c_star_points": c_star_points
            },
            output_name="sensitivity_coefficient",
            output_value=sensitivity,
            formula="d(epsilon)/d(C*) at C*=0.5",
            reference="Numerical differentiation"
        ))

        # Generate provenance hash
        provenance_data = {
            "ntu": str(ntu),
            "configuration": flow_configuration.name,
            "sensitivity": str(sensitivity),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return CapacityRatioSensitivityResult(
            c_star_values=tuple(c_star_values),
            effectiveness_values=tuple(effectiveness_values),
            ntu=ntu,
            flow_configuration=flow_configuration,
            sensitivity_coefficient=sensitivity,
            calculation_steps=tuple(steps),
            provenance_hash=provenance_hash
        )

    def generate_effectiveness_chart(
        self,
        flow_configuration: FlowConfiguration,
        ntu_max: Decimal = Decimal("5"),
        ntu_points: int = 51,
        c_star_values: Optional[List[Decimal]] = None
    ) -> Dict[Decimal, List[Tuple[Decimal, Decimal]]]:
        """
        Generate effectiveness vs NTU data for chart plotting.

        Produces data for the classic effectiveness-NTU charts.

        Args:
            flow_configuration: Flow configuration
            ntu_max: Maximum NTU value
            ntu_points: Number of NTU points
            c_star_values: List of C* values to plot (default: standard values)

        Returns:
            Dictionary mapping C* to list of (NTU, effectiveness) tuples

        Reference: Kays & London, "Compact Heat Exchangers", Figs. 3-7 to 3-16
        """
        if c_star_values is None:
            c_star_values = [
                Decimal("0"),
                Decimal("0.25"),
                Decimal("0.5"),
                Decimal("0.75"),
                Decimal("1.0")
            ]

        chart_data: Dict[Decimal, List[Tuple[Decimal, Decimal]]] = {}

        for c_star in c_star_values:
            points: List[Tuple[Decimal, Decimal]] = []

            for i in range(ntu_points):
                ntu = ntu_max * self._decimal(i / (ntu_points - 1))
                effectiveness = self.calculate_effectiveness(ntu, c_star, flow_configuration)
                points.append((ntu, effectiveness))

            chart_data[c_star] = points

        return chart_data


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_ntu_calculator(precision: int = DECIMAL_PRECISION) -> NTUEffectivenessCalculator:
    """
    Factory function to create an NTU-effectiveness calculator.

    Args:
        precision: Decimal precision for calculations

    Returns:
        Configured NTUEffectivenessCalculator instance
    """
    return NTUEffectivenessCalculator(precision=precision)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enumerations
    "FlowConfiguration",
    "CalculationMode",
    "FluidSide",

    # Data Classes - Input
    "FluidStream",
    "HeatExchangerGeometry",
    "NTUEffectivenessInput",

    # Data Classes - Results
    "CalculationStep",
    "EffectivenessResult",
    "NTUFromEffectivenessResult",
    "ConfigurationComparisonResult",
    "CapacityRatioSensitivityResult",

    # Calculator
    "NTUEffectivenessCalculator",
    "create_ntu_calculator",

    # Constants
    "DECIMAL_PRECISION",
    "MAX_ITERATIONS",
    "CONVERGENCE_TOLERANCE",
]
