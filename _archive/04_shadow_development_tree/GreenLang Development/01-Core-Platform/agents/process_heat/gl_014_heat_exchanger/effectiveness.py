# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO Agent - Effectiveness-NTU Calculations

This module implements the epsilon-NTU (effectiveness-Number of Transfer Units)
method for heat exchanger thermal performance analysis. All calculations are
deterministic with zero hallucination guarantee.

The e-NTU method is preferred over LMTD for:
- Rating existing exchangers
- Performance monitoring
- Off-design analysis

References:
    - Kays & London, "Compact Heat Exchangers" (3rd Ed.)
    - HEDH Heat Exchanger Design Handbook
    - TEMA Standards 9th Edition
    - Incropera & DeWitt, "Fundamentals of Heat and Mass Transfer"

Supported Exchanger Types:
    - Counter-flow
    - Parallel flow
    - Cross-flow (mixed/unmixed)
    - Shell-and-tube (1-2, 2-4, n-2n passes)
    - Multi-pass configurations

Example:
    >>> from greenlang.agents.process_heat.gl_014_heat_exchanger.effectiveness import (
    ...     EffectivenessNTUCalculator
    ... )
    >>> calculator = EffectivenessNTUCalculator()
    >>> result = calculator.calculate_effectiveness(
    ...     ntu=2.0,
    ...     heat_capacity_ratio=0.5,
    ...     flow_arrangement=FlowArrangement.COUNTER_FLOW
    ... )
    >>> print(f"Effectiveness: {result.effectiveness:.3f}")
"""

import math
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_014_heat_exchanger.config import (
    ExchangerType,
    FlowArrangement,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Tolerance for numerical comparisons
EPSILON = 1e-10

# Maximum iterations for iterative calculations
MAX_ITERATIONS = 100

# LMTD correction factor coefficients (for shell-and-tube)
# Source: TEMA Standards
LMTD_CF_COEFFICIENTS = {
    "1-2": {  # 1 shell pass, 2 tube passes
        "a": [0.9, 0.85, 0.78, 0.65],
        "r": [0.25, 0.50, 0.75, 1.0],
    },
    "2-4": {  # 2 shell passes, 4 tube passes
        "a": [0.95, 0.92, 0.88, 0.80],
        "r": [0.25, 0.50, 0.75, 1.0],
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EffectivenessResult:
    """Result of effectiveness calculation."""

    effectiveness: float  # Thermal effectiveness (0-1)
    ntu: float  # Number of Transfer Units
    heat_capacity_ratio: float  # Cr = Cmin/Cmax
    flow_arrangement: str  # Flow arrangement used
    q_max_kw: float  # Maximum possible heat transfer (kW)
    q_actual_kw: float  # Actual heat transfer (kW)
    lmtd_c: float  # LMTD (Celsius)
    lmtd_correction_factor: float  # F factor for multi-pass
    u_required_w_m2k: float  # Required U for this duty
    calculation_method: str  # Calculation method used


@dataclass
class NTUResult:
    """Result of NTU calculation from effectiveness."""

    ntu: float  # Number of Transfer Units
    effectiveness: float  # Input effectiveness
    heat_capacity_ratio: float  # Cr = Cmin/Cmax
    flow_arrangement: str  # Flow arrangement
    converged: bool  # True if calculation converged
    iterations: int  # Number of iterations (if iterative)


@dataclass
class ThermalAnalysisInput:
    """Input for thermal analysis."""

    # Hot stream
    hot_inlet_temp_c: float
    hot_outlet_temp_c: float
    hot_mass_flow_kg_s: float
    hot_cp_kj_kgk: float

    # Cold stream
    cold_inlet_temp_c: float
    cold_outlet_temp_c: float
    cold_mass_flow_kg_s: float
    cold_cp_kj_kgk: float

    # Exchanger
    heat_transfer_area_m2: float
    flow_arrangement: FlowArrangement


# =============================================================================
# EFFECTIVENESS-NTU CALCULATOR
# =============================================================================

class EffectivenessNTUCalculator:
    """
    Effectiveness-NTU method calculator for heat exchangers.

    This class provides deterministic calculations for thermal effectiveness
    using the e-NTU method. All formulas are from established heat transfer
    references with complete provenance.

    Supported configurations:
        - Counter-flow
        - Parallel flow (co-current)
        - Cross-flow (mixed/unmixed variants)
        - Shell-and-tube (various pass arrangements)
        - Multi-pass configurations

    Zero Hallucination Guarantee:
        All calculations use deterministic formulas from:
        - Kays & London correlations
        - HEDH handbook equations
        - TEMA standards

    Example:
        >>> calc = EffectivenessNTUCalculator()
        >>> eff = calc.calculate_effectiveness(ntu=2.0, Cr=0.5, arrangement="counter")
    """

    def __init__(self) -> None:
        """Initialize the effectiveness calculator."""
        self._calculation_count = 0
        logger.info("EffectivenessNTUCalculator initialized")

    def calculate_effectiveness(
        self,
        ntu: float,
        heat_capacity_ratio: float,
        flow_arrangement: FlowArrangement,
        shell_passes: int = 1,
        tube_passes: int = 2,
    ) -> float:
        """
        Calculate thermal effectiveness from NTU and Cr.

        This is the forward problem: given NTU and Cr, find effectiveness.

        Args:
            ntu: Number of Transfer Units (UA/Cmin)
            heat_capacity_ratio: Cr = Cmin/Cmax (0 to 1)
            flow_arrangement: Flow arrangement type
            shell_passes: Number of shell passes (for shell-tube)
            tube_passes: Number of tube passes (for shell-tube)

        Returns:
            Thermal effectiveness (0 to 1)

        Raises:
            ValueError: If inputs are out of valid range
        """
        self._calculation_count += 1

        # Validate inputs
        if ntu < 0:
            raise ValueError(f"NTU must be non-negative, got {ntu}")
        if not 0 <= heat_capacity_ratio <= 1:
            raise ValueError(
                f"Heat capacity ratio must be 0-1, got {heat_capacity_ratio}"
            )

        # Special cases
        if ntu < EPSILON:
            return 0.0

        if heat_capacity_ratio < EPSILON:
            # Cr = 0: One fluid undergoes phase change (evaporation/condensation)
            # effectiveness = 1 - exp(-NTU)
            return 1.0 - math.exp(-ntu)

        # Select calculation method based on flow arrangement
        cr = heat_capacity_ratio

        if flow_arrangement == FlowArrangement.COUNTER_FLOW:
            return self._effectiveness_counter_flow(ntu, cr)

        elif flow_arrangement == FlowArrangement.PARALLEL_FLOW:
            return self._effectiveness_parallel_flow(ntu, cr)

        elif flow_arrangement == FlowArrangement.CROSS_FLOW_UNMIXED:
            return self._effectiveness_cross_flow_unmixed(ntu, cr)

        elif flow_arrangement == FlowArrangement.CROSS_FLOW_MIXED:
            return self._effectiveness_cross_flow_mixed(ntu, cr)

        elif flow_arrangement == FlowArrangement.MULTI_PASS:
            return self._effectiveness_shell_tube(ntu, cr, shell_passes, tube_passes)

        else:
            # Default to counter-flow
            logger.warning(f"Unknown flow arrangement {flow_arrangement}, using counter-flow")
            return self._effectiveness_counter_flow(ntu, cr)

    def calculate_ntu_from_effectiveness(
        self,
        effectiveness: float,
        heat_capacity_ratio: float,
        flow_arrangement: FlowArrangement,
        shell_passes: int = 1,
        tube_passes: int = 2,
    ) -> NTUResult:
        """
        Calculate NTU from thermal effectiveness.

        This is the inverse problem: given effectiveness and Cr, find NTU.

        Args:
            effectiveness: Thermal effectiveness (0 to 1)
            heat_capacity_ratio: Cr = Cmin/Cmax (0 to 1)
            flow_arrangement: Flow arrangement type
            shell_passes: Number of shell passes
            tube_passes: Number of tube passes

        Returns:
            NTUResult with calculated NTU
        """
        self._calculation_count += 1

        # Validate inputs
        if not 0 < effectiveness < 1:
            raise ValueError(f"Effectiveness must be 0-1, got {effectiveness}")
        if not 0 <= heat_capacity_ratio <= 1:
            raise ValueError(
                f"Heat capacity ratio must be 0-1, got {heat_capacity_ratio}"
            )

        eps = effectiveness
        cr = heat_capacity_ratio

        # Special case: Cr = 0
        if cr < EPSILON:
            ntu = -math.log(1.0 - eps)
            return NTUResult(
                ntu=ntu,
                effectiveness=effectiveness,
                heat_capacity_ratio=heat_capacity_ratio,
                flow_arrangement=flow_arrangement.value,
                converged=True,
                iterations=0,
            )

        # Calculate based on flow arrangement
        if flow_arrangement == FlowArrangement.COUNTER_FLOW:
            ntu = self._ntu_counter_flow(eps, cr)
            converged = True
            iterations = 0

        elif flow_arrangement == FlowArrangement.PARALLEL_FLOW:
            ntu = self._ntu_parallel_flow(eps, cr)
            converged = True
            iterations = 0

        else:
            # Use iterative solution for other configurations
            ntu, converged, iterations = self._ntu_iterative(
                eps, cr, flow_arrangement, shell_passes, tube_passes
            )

        return NTUResult(
            ntu=ntu,
            effectiveness=effectiveness,
            heat_capacity_ratio=heat_capacity_ratio,
            flow_arrangement=flow_arrangement.value,
            converged=converged,
            iterations=iterations,
        )

    def analyze_thermal_performance(
        self,
        input_data: ThermalAnalysisInput,
        design_u_w_m2k: Optional[float] = None,
    ) -> EffectivenessResult:
        """
        Perform complete thermal performance analysis.

        Calculates heat duty, LMTD, effectiveness, NTU, and required U value.

        Args:
            input_data: Thermal analysis input data
            design_u_w_m2k: Design overall heat transfer coefficient

        Returns:
            EffectivenessResult with complete analysis
        """
        self._calculation_count += 1

        # Calculate heat capacity rates (kW/K)
        c_hot = input_data.hot_mass_flow_kg_s * input_data.hot_cp_kj_kgk
        c_cold = input_data.cold_mass_flow_kg_s * input_data.cold_cp_kj_kgk

        c_min = min(c_hot, c_cold)
        c_max = max(c_hot, c_cold)

        if c_max < EPSILON:
            raise ValueError("Heat capacity rates too small")

        cr = c_min / c_max

        # Calculate actual heat duty (kW)
        q_hot = c_hot * abs(
            input_data.hot_inlet_temp_c - input_data.hot_outlet_temp_c
        )
        q_cold = c_cold * abs(
            input_data.cold_outlet_temp_c - input_data.cold_inlet_temp_c
        )

        # Use average (should be equal if energy balance is satisfied)
        q_actual = (q_hot + q_cold) / 2

        # Check energy balance
        energy_imbalance = abs(q_hot - q_cold) / max(q_hot, q_cold)
        if energy_imbalance > 0.05:
            logger.warning(
                f"Energy imbalance detected: {energy_imbalance:.1%}"
            )

        # Calculate maximum heat transfer (kW)
        delta_t_max = abs(
            input_data.hot_inlet_temp_c - input_data.cold_inlet_temp_c
        )
        q_max = c_min * delta_t_max

        # Calculate effectiveness
        if q_max < EPSILON:
            effectiveness = 0.0
        else:
            effectiveness = q_actual / q_max

        # Calculate LMTD
        lmtd, lmtd_cf = self.calculate_lmtd(
            input_data.hot_inlet_temp_c,
            input_data.hot_outlet_temp_c,
            input_data.cold_inlet_temp_c,
            input_data.cold_outlet_temp_c,
            input_data.flow_arrangement,
        )

        # Calculate NTU from effectiveness
        ntu_result = self.calculate_ntu_from_effectiveness(
            effectiveness,
            cr,
            input_data.flow_arrangement,
        )

        # Calculate required U
        area_m2 = input_data.heat_transfer_area_m2
        if area_m2 > 0 and lmtd > EPSILON:
            # Q = U * A * F * LMTD
            u_required = (q_actual * 1000) / (area_m2 * lmtd_cf * lmtd)
        else:
            u_required = 0.0

        return EffectivenessResult(
            effectiveness=effectiveness,
            ntu=ntu_result.ntu,
            heat_capacity_ratio=cr,
            flow_arrangement=input_data.flow_arrangement.value,
            q_max_kw=q_max,
            q_actual_kw=q_actual,
            lmtd_c=lmtd,
            lmtd_correction_factor=lmtd_cf,
            u_required_w_m2k=u_required,
            calculation_method="e-NTU",
        )

    def calculate_lmtd(
        self,
        t_hot_in: float,
        t_hot_out: float,
        t_cold_in: float,
        t_cold_out: float,
        flow_arrangement: FlowArrangement,
        shell_passes: int = 1,
    ) -> Tuple[float, float]:
        """
        Calculate Log Mean Temperature Difference and correction factor.

        The LMTD method calculates the effective temperature driving force
        for heat transfer. For multi-pass exchangers, a correction factor F
        is applied.

        Args:
            t_hot_in: Hot stream inlet temperature (C)
            t_hot_out: Hot stream outlet temperature (C)
            t_cold_in: Cold stream inlet temperature (C)
            t_cold_out: Cold stream outlet temperature (C)
            flow_arrangement: Flow arrangement
            shell_passes: Number of shell passes

        Returns:
            Tuple of (LMTD, F correction factor)
        """
        # Calculate terminal temperature differences
        if flow_arrangement == FlowArrangement.COUNTER_FLOW:
            delta_t1 = t_hot_in - t_cold_out
            delta_t2 = t_hot_out - t_cold_in
        else:
            # Parallel flow or others
            delta_t1 = t_hot_in - t_cold_in
            delta_t2 = t_hot_out - t_cold_out

        # Handle edge cases
        if abs(delta_t1 - delta_t2) < EPSILON:
            lmtd = delta_t1
        elif delta_t1 <= 0 or delta_t2 <= 0:
            # Temperature cross - use arithmetic mean
            lmtd = (abs(delta_t1) + abs(delta_t2)) / 2
            logger.warning("Temperature cross detected in LMTD calculation")
        else:
            # Standard LMTD formula
            lmtd = (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)

        # Calculate F correction factor for non-counter-flow arrangements
        if flow_arrangement == FlowArrangement.COUNTER_FLOW:
            f_factor = 1.0
        elif flow_arrangement == FlowArrangement.PARALLEL_FLOW:
            # Parallel flow has inherently lower LMTD, F = 1.0 by definition
            f_factor = 1.0
        else:
            # Calculate F for shell-and-tube or other multi-pass
            f_factor = self._calculate_lmtd_correction_factor(
                t_hot_in, t_hot_out, t_cold_in, t_cold_out, shell_passes
            )

        return lmtd, f_factor

    def calculate_u_from_components(
        self,
        h_shell_w_m2k: float,
        h_tube_w_m2k: float,
        tube_od_m: float,
        tube_id_m: float,
        tube_k_w_mk: float,
        fouling_shell_m2kw: float = 0.0,
        fouling_tube_m2kw: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate overall heat transfer coefficient from individual resistances.

        Uses the thermal resistance network approach:
        1/UA = 1/(h_o * A_o) + R_fo/A_o + ln(r_o/r_i)/(2*pi*k*L) + R_fi/A_i + 1/(h_i * A_i)

        For thin-walled tubes (based on outer area):
        1/U_o = 1/h_o + R_fo + (r_o * ln(r_o/r_i))/k + R_fi*(r_o/r_i) + (r_o/r_i)/h_i

        Args:
            h_shell_w_m2k: Shell side heat transfer coefficient
            h_tube_w_m2k: Tube side heat transfer coefficient
            tube_od_m: Tube outer diameter (m)
            tube_id_m: Tube inner diameter (m)
            tube_k_w_mk: Tube thermal conductivity (W/m-K)
            fouling_shell_m2kw: Shell side fouling resistance (m2K/W)
            fouling_tube_m2kw: Tube side fouling resistance (m2K/W)

        Returns:
            Dictionary with U values (clean and fouled) and resistances
        """
        r_o = tube_od_m / 2
        r_i = tube_id_m / 2
        ratio = r_o / r_i

        # Individual resistances (m2K/W, based on outer area)
        r_shell = 1.0 / h_shell_w_m2k if h_shell_w_m2k > 0 else float('inf')
        r_tube = ratio / h_tube_w_m2k if h_tube_w_m2k > 0 else float('inf')
        r_wall = (r_o * math.log(ratio)) / tube_k_w_mk
        r_fouling_shell = fouling_shell_m2kw
        r_fouling_tube = fouling_tube_m2kw * ratio

        # Total resistances
        r_total_clean = r_shell + r_wall + r_tube
        r_total_fouled = r_total_clean + r_fouling_shell + r_fouling_tube

        # Overall U values
        u_clean = 1.0 / r_total_clean if r_total_clean > 0 else 0.0
        u_fouled = 1.0 / r_total_fouled if r_total_fouled > 0 else 0.0

        return {
            "u_clean_w_m2k": u_clean,
            "u_fouled_w_m2k": u_fouled,
            "r_shell_m2kw": r_shell,
            "r_tube_m2kw": r_tube,
            "r_wall_m2kw": r_wall,
            "r_fouling_total_m2kw": r_fouling_shell + r_fouling_tube,
            "fouling_factor": (u_clean - u_fouled) / u_clean if u_clean > 0 else 0.0,
        }

    def calculate_fouling_from_u(
        self,
        u_clean_w_m2k: float,
        u_current_w_m2k: float,
    ) -> float:
        """
        Calculate fouling resistance from clean and current U values.

        Rf = 1/U_fouled - 1/U_clean

        Args:
            u_clean_w_m2k: Clean overall U value
            u_current_w_m2k: Current (fouled) U value

        Returns:
            Total fouling resistance (m2K/W)
        """
        if u_clean_w_m2k <= 0 or u_current_w_m2k <= 0:
            return 0.0

        if u_current_w_m2k >= u_clean_w_m2k:
            return 0.0  # No fouling (or measurement error)

        return (1.0 / u_current_w_m2k) - (1.0 / u_clean_w_m2k)

    # =========================================================================
    # PRIVATE METHODS - EFFECTIVENESS CALCULATIONS
    # =========================================================================

    def _effectiveness_counter_flow(self, ntu: float, cr: float) -> float:
        """
        Calculate effectiveness for counter-flow arrangement.

        e = [1 - exp(-NTU*(1-Cr))] / [1 - Cr*exp(-NTU*(1-Cr))]

        For Cr = 1: e = NTU / (1 + NTU)
        """
        if abs(1.0 - cr) < EPSILON:
            # Special case: Cr = 1
            return ntu / (1.0 + ntu)

        exp_term = math.exp(-ntu * (1.0 - cr))
        numerator = 1.0 - exp_term
        denominator = 1.0 - cr * exp_term

        if abs(denominator) < EPSILON:
            return 1.0

        return numerator / denominator

    def _effectiveness_parallel_flow(self, ntu: float, cr: float) -> float:
        """
        Calculate effectiveness for parallel flow arrangement.

        e = [1 - exp(-NTU*(1+Cr))] / (1 + Cr)
        """
        exp_term = math.exp(-ntu * (1.0 + cr))
        return (1.0 - exp_term) / (1.0 + cr)

    def _effectiveness_cross_flow_unmixed(self, ntu: float, cr: float) -> float:
        """
        Calculate effectiveness for cross-flow with both fluids unmixed.

        Uses the approximate formula from Kays & London:
        e = 1 - exp[(1/Cr) * NTU^0.22 * {exp(-Cr*NTU^0.78) - 1}]
        """
        if cr < EPSILON:
            return 1.0 - math.exp(-ntu)

        ntu_078 = ntu ** 0.78
        ntu_022 = ntu ** 0.22

        inner = math.exp(-cr * ntu_078) - 1.0
        return 1.0 - math.exp((1.0 / cr) * ntu_022 * inner)

    def _effectiveness_cross_flow_mixed(self, ntu: float, cr: float) -> float:
        """
        Calculate effectiveness for cross-flow with one fluid mixed.

        Cmax mixed (tube side mixed):
        e = (1/Cr) * [1 - exp(-Cr * {1 - exp(-NTU)})]
        """
        if cr < EPSILON:
            return 1.0 - math.exp(-ntu)

        inner = 1.0 - math.exp(-ntu)
        return (1.0 / cr) * (1.0 - math.exp(-cr * inner))

    def _effectiveness_shell_tube(
        self,
        ntu: float,
        cr: float,
        shell_passes: int,
        tube_passes: int,
    ) -> float:
        """
        Calculate effectiveness for shell-and-tube exchangers.

        For 1-2n (1 shell, 2n tube passes):
        e = 2 / [1 + Cr + sqrt(1+Cr^2) * coth(NTU*sqrt(1+Cr^2)/2)]

        For n shell passes, use iterative approach.
        """
        if shell_passes == 1:
            return self._effectiveness_1_2n(ntu, cr)
        else:
            # Multiple shell passes
            return self._effectiveness_n_shell(ntu, cr, shell_passes)

    def _effectiveness_1_2n(self, ntu: float, cr: float) -> float:
        """
        Effectiveness for 1 shell pass, 2n tube passes.

        e = 2 / [1 + Cr + sqrt(1+Cr^2) * coth(NTU*sqrt(1+Cr^2)/2)]
        """
        if cr < EPSILON:
            return 1.0 - math.exp(-ntu)

        sqrt_term = math.sqrt(1.0 + cr * cr)
        ntu_half = ntu * sqrt_term / 2.0

        # coth(x) = cosh(x)/sinh(x)
        if ntu_half > 50:  # Large NTU limit
            coth_term = 1.0
        elif ntu_half < 0.01:  # Small NTU limit
            coth_term = 1.0 / ntu_half
        else:
            coth_term = (math.exp(ntu_half) + math.exp(-ntu_half)) / (
                math.exp(ntu_half) - math.exp(-ntu_half)
            )

        denominator = 1.0 + cr + sqrt_term * coth_term

        if abs(denominator) < EPSILON:
            return 1.0

        return 2.0 / denominator

    def _effectiveness_n_shell(
        self,
        ntu: float,
        cr: float,
        n_shells: int,
    ) -> float:
        """
        Effectiveness for n shell passes (each with 2 tube passes).

        e_n = [(1-e1*Cr)/(1-e1)]^n - 1} / {[(1-e1*Cr)/(1-e1)]^n - Cr}

        Where e1 is the effectiveness of one shell pass.
        """
        # Effectiveness of single shell pass
        ntu_per_shell = ntu / n_shells
        e1 = self._effectiveness_1_2n(ntu_per_shell, cr)

        if abs(1.0 - e1) < EPSILON:
            return 1.0

        if abs(cr) < EPSILON:
            # Simplified for Cr = 0
            return 1.0 - (1.0 - e1) ** n_shells

        # General formula
        ratio = (1.0 - e1 * cr) / (1.0 - e1)
        ratio_n = ratio ** n_shells

        numerator = ratio_n - 1.0
        denominator = ratio_n - cr

        if abs(denominator) < EPSILON:
            return 1.0

        return numerator / denominator

    # =========================================================================
    # PRIVATE METHODS - NTU CALCULATIONS (INVERSE)
    # =========================================================================

    def _ntu_counter_flow(self, eps: float, cr: float) -> float:
        """
        Calculate NTU from effectiveness for counter-flow.

        NTU = ln[(1-eps*Cr)/(1-eps)] / (1-Cr)

        For Cr = 1: NTU = eps / (1 - eps)
        """
        if abs(1.0 - cr) < EPSILON:
            # Special case: Cr = 1
            if eps >= 1.0 - EPSILON:
                return 100.0  # Large NTU limit
            return eps / (1.0 - eps)

        numerator = 1.0 - eps * cr
        denominator = 1.0 - eps

        if numerator <= 0 or denominator <= 0:
            return 100.0  # Large NTU limit

        return math.log(numerator / denominator) / (1.0 - cr)

    def _ntu_parallel_flow(self, eps: float, cr: float) -> float:
        """
        Calculate NTU from effectiveness for parallel flow.

        NTU = -ln[1 - eps*(1+Cr)] / (1+Cr)
        """
        arg = 1.0 - eps * (1.0 + cr)

        if arg <= 0:
            return 100.0  # Large NTU limit

        return -math.log(arg) / (1.0 + cr)

    def _ntu_iterative(
        self,
        eps: float,
        cr: float,
        flow_arrangement: FlowArrangement,
        shell_passes: int,
        tube_passes: int,
    ) -> Tuple[float, bool, int]:
        """
        Calculate NTU using Newton-Raphson iteration.

        Used for configurations without closed-form inverse.
        """
        # Initial guess using counter-flow as approximation
        ntu = self._ntu_counter_flow(eps, cr)
        ntu = min(max(ntu, 0.1), 50.0)  # Bound initial guess

        for iteration in range(MAX_ITERATIONS):
            # Calculate effectiveness at current NTU
            eps_calc = self.calculate_effectiveness(
                ntu, cr, flow_arrangement, shell_passes, tube_passes
            )

            # Check convergence
            error = eps_calc - eps
            if abs(error) < 1e-6:
                return ntu, True, iteration + 1

            # Calculate derivative numerically
            delta = 0.001 * ntu
            eps_plus = self.calculate_effectiveness(
                ntu + delta, cr, flow_arrangement, shell_passes, tube_passes
            )
            derivative = (eps_plus - eps_calc) / delta

            if abs(derivative) < EPSILON:
                break

            # Newton-Raphson update
            ntu_new = ntu - error / derivative
            ntu = max(0.01, min(ntu_new, 100.0))  # Keep bounded

        logger.warning(
            f"NTU iteration did not converge after {MAX_ITERATIONS} iterations"
        )
        return ntu, False, MAX_ITERATIONS

    # =========================================================================
    # PRIVATE METHODS - LMTD CORRECTION FACTOR
    # =========================================================================

    def _calculate_lmtd_correction_factor(
        self,
        t_hot_in: float,
        t_hot_out: float,
        t_cold_in: float,
        t_cold_out: float,
        shell_passes: int,
    ) -> float:
        """
        Calculate LMTD correction factor (F) for shell-and-tube exchangers.

        Uses the analytical expressions from TEMA.

        For 1 shell pass, 2n tube passes:
        R = (T1-T2)/(t2-t1)
        P = (t2-t1)/(T1-t1)
        F = sqrt(R^2+1)*ln[(1-P)/(1-PR)] / [(R-1)*ln{(2-P(R+1-sqrt(R^2+1)))/(2-P(R+1+sqrt(R^2+1)))}]
        """
        # Calculate P and R
        t1 = t_cold_in
        t2 = t_cold_out
        T1 = t_hot_in
        T2 = t_hot_out

        if abs(T1 - t1) < EPSILON:
            return 1.0

        P = (t2 - t1) / (T1 - t1)
        R = (T1 - T2) / (t2 - t1) if abs(t2 - t1) > EPSILON else 0.0

        # Check validity
        if P < EPSILON or P >= 1.0:
            return 1.0
        if R < EPSILON:
            R = EPSILON

        # Special case: R = 1
        if abs(R - 1.0) < 0.01:
            # Use limiting form
            if P < 0.9:
                f = (P * math.sqrt(2)) / ((1 - P) * math.log((2 - P * (2 - math.sqrt(2))) / (2 - P * (2 + math.sqrt(2)))))
                return max(0.5, min(1.0, f))
            return 0.5

        # General formula for 1 shell pass
        sqrt_term = math.sqrt(R * R + 1)

        numerator1 = 1.0 - P
        numerator2 = 1.0 - P * R
        if numerator1 <= 0 or numerator2 <= 0:
            return 0.5

        ln_num = math.log(numerator2 / numerator1)

        denom_arg1 = 2.0 - P * (R + 1.0 - sqrt_term)
        denom_arg2 = 2.0 - P * (R + 1.0 + sqrt_term)

        if denom_arg1 <= 0 or denom_arg2 <= 0 or denom_arg1 / denom_arg2 <= 0:
            return 0.5

        ln_denom = math.log(denom_arg1 / denom_arg2)

        if abs(ln_denom) < EPSILON:
            return 1.0

        f = (sqrt_term * ln_num) / ((R - 1.0) * ln_denom)

        # For multiple shell passes, apply correction
        if shell_passes > 1:
            # Approximate correction for multiple shells
            f = f ** (1.0 / shell_passes)

        return max(0.5, min(1.0, f))

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return self._calculation_count
