"""
GL-014 EXCHANGERPRO - Epsilon-NTU Calculator

Deterministic effectiveness-NTU calculations for heat exchanger rating
and design. The epsilon-NTU method is preferred when outlet temperatures
are unknown (design mode) or when rating existing exchangers.

Fundamental Definitions:
    C_h = m_dot_h * Cp_h    (Hot side heat capacity rate [W/K])
    C_c = m_dot_c * Cp_c    (Cold side heat capacity rate [W/K])
    C_min = min(C_h, C_c)
    C_max = max(C_h, C_c)
    C_r = C_min / C_max     (Capacity ratio)
    NTU = UA / C_min        (Number of Transfer Units)
    epsilon = Q / Q_max     (Effectiveness)

Configurations Supported:
    - Counter-current flow
    - Co-current (parallel) flow
    - Crossflow (unmixed, one mixed, both mixed)
    - Shell-and-tube (TEMA E, F, G, H, J configurations)
    - Multiple shell passes

Reference:
    - Kays & London, "Compact Heat Exchangers"
    - Incropera & DeWitt, "Fundamentals of Heat and Mass Transfer"
    - TEMA Standards, 10th Edition
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json
import math


# =============================================================================
# Constants
# =============================================================================

# Maximum NTU for numerical stability (practical limit ~10)
MAX_NTU = 20.0

# Minimum capacity ratio for special case handling
MIN_C_RATIO = 0.0001

# Minimum effectiveness threshold
MIN_EFFECTIVENESS = 0.0001


# =============================================================================
# Enums
# =============================================================================

class ExchangerConfiguration(str, Enum):
    """Heat exchanger configuration types per TEMA and industry standards."""

    # Simple flow patterns
    COUNTER_CURRENT = "counter_current"
    CO_CURRENT = "co_current"

    # Crossflow configurations
    CROSSFLOW_UNMIXED_UNMIXED = "crossflow_unmixed_unmixed"
    CROSSFLOW_MIXED_UNMIXED = "crossflow_mixed_unmixed"  # C_max side mixed
    CROSSFLOW_UNMIXED_MIXED = "crossflow_unmixed_mixed"  # C_min side mixed
    CROSSFLOW_MIXED_MIXED = "crossflow_mixed_mixed"

    # Shell-and-tube (TEMA configurations)
    SHELL_TUBE_1_2 = "shell_tube_1_2"    # 1 shell pass, 2 tube passes (TEMA E)
    SHELL_TUBE_1_4 = "shell_tube_1_4"    # 1 shell pass, 4 tube passes
    SHELL_TUBE_2_4 = "shell_tube_2_4"    # 2 shell passes, 4 tube passes (TEMA F)
    SHELL_TUBE_N_2N = "shell_tube_n_2n"  # N shell passes, 2N tube passes

    # Specialized
    SPLIT_FLOW = "split_flow"            # TEMA G shell (split flow)
    DIVIDED_FLOW = "divided_flow"        # TEMA H shell (double split)
    J_SHELL = "j_shell"                  # TEMA J shell


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HeatCapacityRate:
    """Heat capacity rate for a stream."""
    stream_id: str
    m_dot_kg_s: float   # Mass flow rate [kg/s]
    Cp_J_kgK: float     # Specific heat [J/(kg*K)]

    @property
    def C_W_K(self) -> float:
        """Heat capacity rate [W/K]."""
        return self.m_dot_kg_s * self.Cp_J_kgK

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "stream_id": self.stream_id,
            "m_dot_kg_s": self.m_dot_kg_s,
            "Cp_J_kgK": self.Cp_J_kgK,
            "C_W_K": self.C_W_K,
        }


@dataclass
class EpsilonNTUInputs:
    """
    Inputs for epsilon-NTU calculations.

    Can operate in two modes:
    1. Rating mode: Given UA, calculate effectiveness
    2. Design mode: Given epsilon, calculate required NTU/UA
    """
    hot_stream: HeatCapacityRate
    cold_stream: HeatCapacityRate
    configuration: ExchangerConfiguration = ExchangerConfiguration.COUNTER_CURRENT

    # For rating mode (provide UA)
    UA_W_K: Optional[float] = None

    # For design mode (provide target effectiveness)
    target_effectiveness: Optional[float] = None

    # For shell-and-tube configurations
    n_shell_passes: int = 1
    n_tube_passes: int = 2

    # Operating temperatures (optional, for Q_max calculation)
    T_hot_in_K: Optional[float] = None
    T_cold_in_K: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for hashing."""
        return {
            "hot_stream": self.hot_stream.to_dict(),
            "cold_stream": self.cold_stream.to_dict(),
            "configuration": self.configuration.value,
            "UA_W_K": self.UA_W_K,
            "target_effectiveness": self.target_effectiveness,
            "n_shell_passes": self.n_shell_passes,
            "n_tube_passes": self.n_tube_passes,
            "T_hot_in_K": self.T_hot_in_K,
            "T_cold_in_K": self.T_cold_in_K,
        }


@dataclass
class CapacityRatioResult:
    """Heat capacity rate analysis."""
    C_hot_W_K: float            # Hot side capacity rate
    C_cold_W_K: float           # Cold side capacity rate
    C_min_W_K: float            # Minimum capacity rate
    C_max_W_K: float            # Maximum capacity rate
    C_r: float                  # Capacity ratio (C_min/C_max)
    min_side: str               # Which side has C_min ("hot" or "cold")
    is_balanced: bool           # True if C_h ≈ C_c

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "C_hot_W_K": round(self.C_hot_W_K, 3),
            "C_cold_W_K": round(self.C_cold_W_K, 3),
            "C_min_W_K": round(self.C_min_W_K, 3),
            "C_max_W_K": round(self.C_max_W_K, 3),
            "C_r": round(self.C_r, 6),
            "min_side": self.min_side,
            "is_balanced": self.is_balanced,
        }


@dataclass
class EpsilonNTUResult:
    """
    Complete epsilon-NTU calculation result.
    """
    # Capacity rate analysis
    capacity_ratios: CapacityRatioResult

    # Core epsilon-NTU values
    NTU: float                    # Number of Transfer Units
    effectiveness: float          # Heat exchanger effectiveness (0 to 1)
    UA_W_K: float                 # Overall heat transfer coefficient * Area

    # Heat transfer
    Q_max_W: Optional[float]      # Maximum possible heat transfer [W]
    Q_actual_W: Optional[float]   # Actual heat transfer [W]

    # Derived values
    epsilon_max: float            # Maximum achievable effectiveness for config

    # Configuration
    configuration: str
    n_shell_passes: int
    n_tube_passes: int

    # Calculation mode
    calculation_mode: str         # "rating" or "design"

    # Validation
    is_valid: bool
    warnings: List[str] = field(default_factory=list)

    # Calculation trace
    calculation_steps: List[str] = field(default_factory=list)
    epsilon_formula: str = ""

    # Provenance
    inputs_hash: str = ""
    outputs_hash: str = ""
    computation_hash: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time_ms: float = 0.0
    calculator_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "capacity_ratios": self.capacity_ratios.to_dict(),
            "NTU": round(self.NTU, 6),
            "effectiveness": round(self.effectiveness, 6),
            "UA_W_K": round(self.UA_W_K, 3),
            "Q_max_W": round(self.Q_max_W, 3) if self.Q_max_W else None,
            "Q_actual_W": round(self.Q_actual_W, 3) if self.Q_actual_W else None,
            "epsilon_max": round(self.epsilon_max, 6),
            "configuration": self.configuration,
            "n_shell_passes": self.n_shell_passes,
            "n_tube_passes": self.n_tube_passes,
            "calculation_mode": self.calculation_mode,
            "is_valid": self.is_valid,
            "warnings": self.warnings,
            "calculation_steps": self.calculation_steps,
            "epsilon_formula": self.epsilon_formula,
            "inputs_hash": self.inputs_hash,
            "outputs_hash": self.outputs_hash,
            "computation_hash": self.computation_hash,
            "timestamp": self.timestamp.isoformat(),
            "execution_time_ms": round(self.execution_time_ms, 3),
            "calculator_version": self.calculator_version,
        }


# =============================================================================
# Epsilon-NTU Calculator
# =============================================================================

class EpsilonNTUCalculator:
    """
    Deterministic Epsilon-NTU Calculator.

    Provides epsilon-NTU relationships for all common heat exchanger
    configurations. Operates in two modes:

    1. Rating Mode (given UA):
       - Calculate NTU = UA / C_min
       - Calculate effectiveness epsilon from NTU and C_r
       - Calculate Q = epsilon * Q_max

    2. Design Mode (given target epsilon):
       - Calculate required NTU from epsilon and C_r
       - Calculate required UA = NTU * C_min

    Zero-Hallucination Guarantee:
        All calculations are deterministic with bit-perfect reproducibility.
        Same inputs ALWAYS produce identical outputs.

    Example (Rating Mode):
        >>> calc = EpsilonNTUCalculator()
        >>> hot = HeatCapacityRate("hot", m_dot_kg_s=2.0, Cp_J_kgK=4180)
        >>> cold = HeatCapacityRate("cold", m_dot_kg_s=3.0, Cp_J_kgK=4180)
        >>> inputs = EpsilonNTUInputs(
        ...     hot_stream=hot, cold_stream=cold,
        ...     UA_W_K=50000,
        ...     configuration=ExchangerConfiguration.COUNTER_CURRENT
        ... )
        >>> result = calc.calculate(inputs)
        >>> print(f"NTU = {result.NTU:.3f}, epsilon = {result.effectiveness:.4f}")

    Example (Design Mode):
        >>> inputs = EpsilonNTUInputs(
        ...     hot_stream=hot, cold_stream=cold,
        ...     target_effectiveness=0.85,
        ...     configuration=ExchangerConfiguration.COUNTER_CURRENT
        ... )
        >>> result = calc.calculate(inputs)
        >>> print(f"Required NTU = {result.NTU:.3f}, UA = {result.UA_W_K:.0f} W/K")
    """

    NAME = "EpsilonNTUCalculator"
    VERSION = "1.0.0"
    AGENT_ID = "GL-014"

    def __init__(
        self,
        max_ntu: float = MAX_NTU,
        balanced_threshold: float = 0.01,
    ):
        """
        Initialize Epsilon-NTU Calculator.

        Args:
            max_ntu: Maximum NTU for numerical stability
            balanced_threshold: Threshold for C_r ~ 1.0 detection
        """
        self.max_ntu = max_ntu
        self.balanced_threshold = balanced_threshold

    def calculate(self, inputs: EpsilonNTUInputs) -> EpsilonNTUResult:
        """
        Perform epsilon-NTU calculation.

        Automatically detects rating vs. design mode based on inputs.

        Args:
            inputs: Heat capacity rates, UA or target epsilon, configuration

        Returns:
            EpsilonNTUResult with effectiveness, NTU, and provenance
        """
        start_time = datetime.now(timezone.utc)
        warnings: List[str] = []
        calculation_steps: List[str] = []

        # Step 1: Validate inputs
        validation_errors = self._validate_inputs(inputs)
        if validation_errors:
            warnings.extend(validation_errors)

        # Step 2: Calculate capacity ratios
        capacity = self._calculate_capacity_ratios(inputs.hot_stream, inputs.cold_stream)
        calculation_steps.append(
            f"C_hot = {capacity.C_hot_W_K:.1f} W/K, C_cold = {capacity.C_cold_W_K:.1f} W/K"
        )
        calculation_steps.append(
            f"C_min = {capacity.C_min_W_K:.1f} W/K ({capacity.min_side}), C_r = {capacity.C_r:.4f}"
        )

        # Step 3: Determine calculation mode
        if inputs.UA_W_K is not None:
            mode = "rating"
            # Rating mode: UA given, calculate epsilon
            NTU = inputs.UA_W_K / capacity.C_min_W_K
            calculation_steps.append(f"NTU = UA / C_min = {inputs.UA_W_K:.1f} / {capacity.C_min_W_K:.1f} = {NTU:.4f}")

            epsilon = self._calculate_effectiveness(
                NTU=NTU,
                C_r=capacity.C_r,
                configuration=inputs.configuration,
                n_shell_passes=inputs.n_shell_passes,
            )
            UA_W_K = inputs.UA_W_K

        elif inputs.target_effectiveness is not None:
            mode = "design"
            # Design mode: epsilon given, calculate NTU and UA
            epsilon = inputs.target_effectiveness

            NTU = self._calculate_NTU_from_effectiveness(
                epsilon=epsilon,
                C_r=capacity.C_r,
                configuration=inputs.configuration,
                n_shell_passes=inputs.n_shell_passes,
            )
            calculation_steps.append(f"Required NTU for epsilon = {epsilon:.4f}: NTU = {NTU:.4f}")

            UA_W_K = NTU * capacity.C_min_W_K
            calculation_steps.append(f"UA = NTU * C_min = {NTU:.4f} * {capacity.C_min_W_K:.1f} = {UA_W_K:.1f} W/K")

        else:
            # Error: neither UA nor target_effectiveness provided
            warnings.append("Neither UA_W_K nor target_effectiveness provided")
            NTU = 0.0
            epsilon = 0.0
            UA_W_K = 0.0
            mode = "invalid"

        # Clamp NTU to max
        if NTU > self.max_ntu:
            warnings.append(f"NTU ({NTU:.2f}) exceeds practical maximum ({self.max_ntu}). Clamped.")
            NTU = self.max_ntu

        # Get epsilon formula for documentation
        epsilon_formula = self._get_epsilon_formula(inputs.configuration, capacity.C_r)
        calculation_steps.append(f"Epsilon formula: {epsilon_formula}")
        calculation_steps.append(f"Calculated effectiveness = {epsilon:.6f}")

        # Calculate maximum effectiveness for this configuration
        epsilon_max = self._calculate_max_effectiveness(
            C_r=capacity.C_r,
            configuration=inputs.configuration,
        )

        if epsilon > epsilon_max * 0.99:
            warnings.append(
                f"Effectiveness ({epsilon:.4f}) is at or near maximum ({epsilon_max:.4f}) for this configuration"
            )

        # Step 4: Calculate Q if temperatures provided
        Q_max_W = None
        Q_actual_W = None

        if inputs.T_hot_in_K is not None and inputs.T_cold_in_K is not None:
            Q_max_W = capacity.C_min_W_K * (inputs.T_hot_in_K - inputs.T_cold_in_K)
            Q_actual_W = epsilon * Q_max_W
            calculation_steps.append(
                f"Q_max = C_min * (T_h_in - T_c_in) = {capacity.C_min_W_K:.1f} * "
                f"({inputs.T_hot_in_K:.1f} - {inputs.T_cold_in_K:.1f}) = {Q_max_W:.1f} W"
            )
            calculation_steps.append(f"Q_actual = epsilon * Q_max = {epsilon:.4f} * {Q_max_W:.1f} = {Q_actual_W:.1f} W")

        # Step 5: Compute provenance
        end_time = datetime.now(timezone.utc)
        execution_time_ms = (end_time - start_time).total_seconds() * 1000

        inputs_hash = self._compute_hash(inputs.to_dict())

        result_data = {
            "NTU": NTU,
            "effectiveness": epsilon,
            "UA_W_K": UA_W_K,
            "C_r": capacity.C_r,
        }
        outputs_hash = self._compute_hash(result_data)

        computation_hash = self._compute_hash({
            "inputs_hash": inputs_hash,
            "outputs_hash": outputs_hash,
            "calculator": self.NAME,
            "version": self.VERSION,
        })

        return EpsilonNTUResult(
            capacity_ratios=capacity,
            NTU=NTU,
            effectiveness=epsilon,
            UA_W_K=UA_W_K,
            Q_max_W=Q_max_W,
            Q_actual_W=Q_actual_W,
            epsilon_max=epsilon_max,
            configuration=inputs.configuration.value,
            n_shell_passes=inputs.n_shell_passes,
            n_tube_passes=inputs.n_tube_passes,
            calculation_mode=mode,
            is_valid=len(validation_errors) == 0 and mode != "invalid",
            warnings=warnings,
            calculation_steps=calculation_steps,
            epsilon_formula=epsilon_formula,
            inputs_hash=inputs_hash,
            outputs_hash=outputs_hash,
            computation_hash=computation_hash,
            timestamp=start_time,
            execution_time_ms=execution_time_ms,
            calculator_version=self.VERSION,
        )

    def _validate_inputs(self, inputs: EpsilonNTUInputs) -> List[str]:
        """Validate inputs."""
        errors: List[str] = []

        # Capacity rate validation
        if inputs.hot_stream.m_dot_kg_s <= 0:
            errors.append(f"Hot stream mass flow must be positive: {inputs.hot_stream.m_dot_kg_s}")
        if inputs.hot_stream.Cp_J_kgK <= 0:
            errors.append(f"Hot stream Cp must be positive: {inputs.hot_stream.Cp_J_kgK}")
        if inputs.cold_stream.m_dot_kg_s <= 0:
            errors.append(f"Cold stream mass flow must be positive: {inputs.cold_stream.m_dot_kg_s}")
        if inputs.cold_stream.Cp_J_kgK <= 0:
            errors.append(f"Cold stream Cp must be positive: {inputs.cold_stream.Cp_J_kgK}")

        # UA or effectiveness validation
        if inputs.UA_W_K is not None and inputs.UA_W_K <= 0:
            errors.append(f"UA must be positive: {inputs.UA_W_K}")
        if inputs.target_effectiveness is not None:
            if inputs.target_effectiveness <= 0 or inputs.target_effectiveness >= 1:
                errors.append(f"Target effectiveness must be in (0, 1): {inputs.target_effectiveness}")

        # Temperature validation
        if inputs.T_hot_in_K is not None and inputs.T_hot_in_K <= 0:
            errors.append(f"Hot inlet temperature must be positive: {inputs.T_hot_in_K}")
        if inputs.T_cold_in_K is not None and inputs.T_cold_in_K <= 0:
            errors.append(f"Cold inlet temperature must be positive: {inputs.T_cold_in_K}")
        if (inputs.T_hot_in_K is not None and inputs.T_cold_in_K is not None and
                inputs.T_hot_in_K <= inputs.T_cold_in_K):
            errors.append(
                f"Hot inlet must exceed cold inlet: T_h_in={inputs.T_hot_in_K}K, T_c_in={inputs.T_cold_in_K}K"
            )

        return errors

    def _calculate_capacity_ratios(
        self,
        hot: HeatCapacityRate,
        cold: HeatCapacityRate,
    ) -> CapacityRatioResult:
        """Calculate capacity rate analysis."""
        C_hot = hot.C_W_K
        C_cold = cold.C_W_K

        C_min = min(C_hot, C_cold)
        C_max = max(C_hot, C_cold)

        C_r = C_min / C_max if C_max > 0 else 0.0

        min_side = "hot" if C_hot <= C_cold else "cold"
        is_balanced = abs(C_r - 1.0) < self.balanced_threshold

        return CapacityRatioResult(
            C_hot_W_K=C_hot,
            C_cold_W_K=C_cold,
            C_min_W_K=C_min,
            C_max_W_K=C_max,
            C_r=C_r,
            min_side=min_side,
            is_balanced=is_balanced,
        )

    def _calculate_effectiveness(
        self,
        NTU: float,
        C_r: float,
        configuration: ExchangerConfiguration,
        n_shell_passes: int = 1,
    ) -> float:
        """
        Calculate effectiveness from NTU and capacity ratio.

        This is the core deterministic calculation.
        """
        if NTU <= 0:
            return 0.0

        # Clamp C_r to valid range
        C_r = max(MIN_C_RATIO, min(1.0, C_r))

        # Select appropriate correlation
        if configuration == ExchangerConfiguration.COUNTER_CURRENT:
            return self._epsilon_counter_current(NTU, C_r)

        elif configuration == ExchangerConfiguration.CO_CURRENT:
            return self._epsilon_co_current(NTU, C_r)

        elif configuration == ExchangerConfiguration.CROSSFLOW_UNMIXED_UNMIXED:
            return self._epsilon_crossflow_unmixed_unmixed(NTU, C_r)

        elif configuration == ExchangerConfiguration.CROSSFLOW_MIXED_UNMIXED:
            return self._epsilon_crossflow_mixed_unmixed(NTU, C_r)

        elif configuration == ExchangerConfiguration.CROSSFLOW_UNMIXED_MIXED:
            return self._epsilon_crossflow_unmixed_mixed(NTU, C_r)

        elif configuration == ExchangerConfiguration.CROSSFLOW_MIXED_MIXED:
            return self._epsilon_crossflow_mixed_mixed(NTU, C_r)

        elif configuration in (
            ExchangerConfiguration.SHELL_TUBE_1_2,
            ExchangerConfiguration.SHELL_TUBE_1_4,
        ):
            return self._epsilon_shell_tube_1_2n(NTU, C_r)

        elif configuration in (
            ExchangerConfiguration.SHELL_TUBE_2_4,
            ExchangerConfiguration.SHELL_TUBE_N_2N,
        ):
            return self._epsilon_shell_tube_n_2n(NTU, C_r, n_shell_passes)

        elif configuration == ExchangerConfiguration.SPLIT_FLOW:
            return self._epsilon_split_flow(NTU, C_r)

        elif configuration == ExchangerConfiguration.DIVIDED_FLOW:
            return self._epsilon_divided_flow(NTU, C_r)

        elif configuration == ExchangerConfiguration.J_SHELL:
            return self._epsilon_j_shell(NTU, C_r)

        else:
            # Default to counter-current
            return self._epsilon_counter_current(NTU, C_r)

    def _epsilon_counter_current(self, NTU: float, C_r: float) -> float:
        """
        Counter-current flow effectiveness.

        epsilon = (1 - exp(-NTU*(1-C_r))) / (1 - C_r*exp(-NTU*(1-C_r)))

        Special case for C_r = 1:
        epsilon = NTU / (1 + NTU)
        """
        if abs(C_r - 1.0) < MIN_C_RATIO:
            # Balanced capacities
            return NTU / (1.0 + NTU)

        try:
            exp_term = math.exp(-NTU * (1.0 - C_r))
            epsilon = (1.0 - exp_term) / (1.0 - C_r * exp_term)
            return max(0.0, min(1.0, epsilon))
        except (OverflowError, ValueError):
            # Large NTU approaches maximum effectiveness
            return 1.0 if C_r < 1.0 else NTU / (1.0 + NTU)

    def _epsilon_co_current(self, NTU: float, C_r: float) -> float:
        """
        Co-current (parallel) flow effectiveness.

        epsilon = (1 - exp(-NTU*(1+C_r))) / (1 + C_r)
        """
        try:
            exp_term = math.exp(-NTU * (1.0 + C_r))
            epsilon = (1.0 - exp_term) / (1.0 + C_r)
            return max(0.0, min(1.0, epsilon))
        except (OverflowError, ValueError):
            return 1.0 / (1.0 + C_r)

    def _epsilon_crossflow_unmixed_unmixed(self, NTU: float, C_r: float) -> float:
        """
        Crossflow with both fluids unmixed.

        Uses the approximation by Kays & London.
        """
        try:
            term1 = NTU ** 0.22
            term2 = math.exp(-C_r * NTU ** 0.78) - 1.0
            epsilon = 1.0 - math.exp(term2 / (C_r * term1 + 1e-10))
            return max(0.0, min(1.0, epsilon))
        except:
            return self._epsilon_counter_current(NTU, C_r) * 0.95

    def _epsilon_crossflow_mixed_unmixed(self, NTU: float, C_r: float) -> float:
        """
        Crossflow with C_max side mixed, C_min side unmixed.

        epsilon = (1/C_r) * (1 - exp(-C_r * (1 - exp(-NTU))))
        """
        try:
            inner_term = 1.0 - math.exp(-NTU)
            epsilon = (1.0 / C_r) * (1.0 - math.exp(-C_r * inner_term))
            return max(0.0, min(1.0, epsilon))
        except:
            return self._epsilon_counter_current(NTU, C_r) * 0.90

    def _epsilon_crossflow_unmixed_mixed(self, NTU: float, C_r: float) -> float:
        """
        Crossflow with C_min side mixed, C_max side unmixed.

        epsilon = 1 - exp(-(1/C_r) * (1 - exp(-C_r*NTU)))
        """
        try:
            inner_term = 1.0 - math.exp(-C_r * NTU)
            epsilon = 1.0 - math.exp(-(1.0 / C_r) * inner_term)
            return max(0.0, min(1.0, epsilon))
        except:
            return self._epsilon_counter_current(NTU, C_r) * 0.90

    def _epsilon_crossflow_mixed_mixed(self, NTU: float, C_r: float) -> float:
        """
        Crossflow with both fluids mixed.

        epsilon = 1 / (1/(1-exp(-NTU)) + C_r/(1-exp(-NTU*C_r)) - 1/NTU)
        """
        try:
            term1 = 1.0 / (1.0 - math.exp(-NTU) + 1e-10)
            term2 = C_r / (1.0 - math.exp(-NTU * C_r) + 1e-10)
            term3 = 1.0 / (NTU + 1e-10)

            denom = term1 + term2 - term3
            epsilon = 1.0 / denom if denom > 0 else 0.0
            return max(0.0, min(1.0, epsilon))
        except:
            return self._epsilon_counter_current(NTU, C_r) * 0.85

    def _epsilon_shell_tube_1_2n(self, NTU: float, C_r: float) -> float:
        """
        Shell-and-tube with 1 shell pass, 2n tube passes (TEMA E).

        epsilon = 2 / (1 + C_r + sqrt(1+C_r^2) * coth(NTU*sqrt(1+C_r^2)/2))

        Uses hyperbolic cotangent for numerical stability.
        """
        try:
            E = math.sqrt(1.0 + C_r * C_r)
            ntu_E_half = NTU * E / 2.0

            # coth(x) = (e^x + e^-x) / (e^x - e^-x)
            if ntu_E_half > 20:
                coth_term = 1.0
            else:
                exp_pos = math.exp(ntu_E_half)
                exp_neg = math.exp(-ntu_E_half)
                coth_term = (exp_pos + exp_neg) / (exp_pos - exp_neg + 1e-10)

            denom = 1.0 + C_r + E * coth_term
            epsilon = 2.0 / denom

            return max(0.0, min(1.0, epsilon))

        except:
            return self._epsilon_counter_current(NTU, C_r) * 0.9

    def _epsilon_shell_tube_n_2n(self, NTU: float, C_r: float, n: int) -> float:
        """
        Shell-and-tube with n shell passes, 2n tube passes.

        For n > 1, uses the cascade relationship.
        """
        if n == 1:
            return self._epsilon_shell_tube_1_2n(NTU, C_r)

        try:
            # Effectiveness of single shell pass
            epsilon_1 = self._epsilon_shell_tube_1_2n(NTU / n, C_r)

            # Overall effectiveness for n identical exchangers in series
            if abs(C_r - 1.0) < MIN_C_RATIO:
                epsilon = n * epsilon_1 / (1 + (n - 1) * epsilon_1)
            else:
                term = ((1 - epsilon_1 * C_r) / (1 - epsilon_1)) ** n
                epsilon = (1 - term) / (1 - C_r * term)

            return max(0.0, min(1.0, epsilon))

        except:
            return self._epsilon_counter_current(NTU, C_r) * 0.95

    def _epsilon_split_flow(self, NTU: float, C_r: float) -> float:
        """
        Split flow (TEMA G shell) effectiveness.

        Approximation: between 1-2 and counter-current.
        """
        eps_counter = self._epsilon_counter_current(NTU, C_r)
        eps_1_2 = self._epsilon_shell_tube_1_2n(NTU, C_r)
        return (eps_counter + eps_1_2) / 2.0

    def _epsilon_divided_flow(self, NTU: float, C_r: float) -> float:
        """
        Double split flow (TEMA H shell) effectiveness.

        Approximation: closer to counter-current than G shell.
        """
        eps_counter = self._epsilon_counter_current(NTU, C_r)
        eps_1_2 = self._epsilon_shell_tube_1_2n(NTU, C_r)
        return 0.7 * eps_counter + 0.3 * eps_1_2

    def _epsilon_j_shell(self, NTU: float, C_r: float) -> float:
        """
        TEMA J shell effectiveness.

        Single shell pass with shell-side inlet and outlet on same end.
        """
        # J shell is between E shell and counter-current
        eps_counter = self._epsilon_counter_current(NTU, C_r)
        eps_1_2 = self._epsilon_shell_tube_1_2n(NTU, C_r)
        return 0.6 * eps_counter + 0.4 * eps_1_2

    def _calculate_NTU_from_effectiveness(
        self,
        epsilon: float,
        C_r: float,
        configuration: ExchangerConfiguration,
        n_shell_passes: int = 1,
    ) -> float:
        """
        Calculate NTU from effectiveness (inverse problem).

        Uses analytical formulas where available, otherwise numerical iteration.
        """
        epsilon = max(MIN_EFFECTIVENESS, min(0.9999, epsilon))
        C_r = max(MIN_C_RATIO, min(1.0, C_r))

        if configuration == ExchangerConfiguration.COUNTER_CURRENT:
            return self._NTU_counter_current(epsilon, C_r)

        elif configuration == ExchangerConfiguration.CO_CURRENT:
            return self._NTU_co_current(epsilon, C_r)

        else:
            # For other configurations, use numerical inversion
            return self._NTU_numerical(epsilon, C_r, configuration, n_shell_passes)

    def _NTU_counter_current(self, epsilon: float, C_r: float) -> float:
        """
        Inverse: NTU from epsilon for counter-current.

        NTU = (1/(1-C_r)) * ln((1-epsilon*C_r)/(1-epsilon))

        Special case for C_r = 1:
        NTU = epsilon / (1 - epsilon)
        """
        if abs(C_r - 1.0) < MIN_C_RATIO:
            return epsilon / (1.0 - epsilon)

        try:
            arg = (1.0 - epsilon * C_r) / (1.0 - epsilon)
            if arg <= 0:
                return self.max_ntu
            return math.log(arg) / (1.0 - C_r)
        except:
            return self.max_ntu

    def _NTU_co_current(self, epsilon: float, C_r: float) -> float:
        """
        Inverse: NTU from epsilon for co-current.

        NTU = -ln(1 - epsilon*(1+C_r)) / (1+C_r)
        """
        try:
            arg = 1.0 - epsilon * (1.0 + C_r)
            if arg <= 0:
                return self.max_ntu
            return -math.log(arg) / (1.0 + C_r)
        except:
            return self.max_ntu

    def _NTU_numerical(
        self,
        target_epsilon: float,
        C_r: float,
        configuration: ExchangerConfiguration,
        n_shell_passes: int,
    ) -> float:
        """
        Numerical inversion using bisection method.

        Find NTU such that epsilon(NTU, C_r) = target_epsilon.
        """
        # Bisection bounds
        NTU_low = 0.001
        NTU_high = self.max_ntu

        # Bisection tolerance
        tol = 1e-6
        max_iter = 50

        for _ in range(max_iter):
            NTU_mid = (NTU_low + NTU_high) / 2.0
            epsilon_mid = self._calculate_effectiveness(NTU_mid, C_r, configuration, n_shell_passes)

            if abs(epsilon_mid - target_epsilon) < tol:
                return NTU_mid

            if epsilon_mid < target_epsilon:
                NTU_low = NTU_mid
            else:
                NTU_high = NTU_mid

        return (NTU_low + NTU_high) / 2.0

    def _calculate_max_effectiveness(
        self,
        C_r: float,
        configuration: ExchangerConfiguration,
    ) -> float:
        """Calculate maximum achievable effectiveness for configuration."""
        # Most configurations approach counter-current max at high NTU
        if configuration == ExchangerConfiguration.CO_CURRENT:
            return 1.0 / (1.0 + C_r)
        else:
            # Counter-current and most other configurations
            if abs(C_r - 1.0) < MIN_C_RATIO:
                return 1.0
            else:
                return 1.0  # Theoretical max, approached asymptotically

    def _get_epsilon_formula(
        self,
        configuration: ExchangerConfiguration,
        C_r: float,
    ) -> str:
        """Get human-readable epsilon formula for documentation."""
        if configuration == ExchangerConfiguration.COUNTER_CURRENT:
            if abs(C_r - 1.0) < MIN_C_RATIO:
                return "epsilon = NTU / (1 + NTU) [balanced C_r = 1]"
            return "epsilon = (1 - exp(-NTU*(1-C_r))) / (1 - C_r*exp(-NTU*(1-C_r)))"

        elif configuration == ExchangerConfiguration.CO_CURRENT:
            return "epsilon = (1 - exp(-NTU*(1+C_r))) / (1 + C_r)"

        elif configuration == ExchangerConfiguration.CROSSFLOW_UNMIXED_UNMIXED:
            return "epsilon = 1 - exp((exp(-NTU^0.78 * C_r) - 1) / (C_r * NTU^0.22))"

        elif configuration in (
            ExchangerConfiguration.SHELL_TUBE_1_2,
            ExchangerConfiguration.SHELL_TUBE_1_4,
        ):
            return "epsilon = 2 / (1 + C_r + sqrt(1+C_r^2) * coth(NTU*sqrt(1+C_r^2)/2))"

        else:
            return f"Configuration-specific formula for {configuration.value}"

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash."""
        normalized = self._normalize_for_hash(data)
        json_str = json.dumps(normalized, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def _normalize_for_hash(self, obj: Any) -> Any:
        """Normalize for consistent hashing."""
        if obj is None:
            return None
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, int):
            return obj
        elif isinstance(obj, float):
            return round(obj, 10)
        elif isinstance(obj, str):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._normalize_for_hash(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._normalize_for_hash(v) for k, v in sorted(obj.items())}
        else:
            return str(obj)


# =============================================================================
# Convenience Functions
# =============================================================================

def calculate_effectiveness(
    NTU: float,
    C_r: float,
    counter_current: bool = True,
) -> float:
    """
    Quick effectiveness calculation (no provenance tracking).

    Args:
        NTU: Number of Transfer Units
        C_r: Capacity ratio (C_min/C_max)
        counter_current: If True, counter-current; else co-current

    Returns:
        Effectiveness (0 to 1)
    """
    calc = EpsilonNTUCalculator()
    config = (
        ExchangerConfiguration.COUNTER_CURRENT
        if counter_current
        else ExchangerConfiguration.CO_CURRENT
    )
    return calc._calculate_effectiveness(NTU, C_r, config)


def calculate_NTU(
    UA_W_K: float,
    C_min_W_K: float,
) -> float:
    """
    Calculate NTU from UA and C_min.

    NTU = UA / C_min
    """
    if C_min_W_K <= 0:
        return 0.0
    return UA_W_K / C_min_W_K
