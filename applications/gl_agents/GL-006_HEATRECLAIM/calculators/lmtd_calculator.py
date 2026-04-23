"""
GL-006 HEATRECLAIM - LMTD and NTU Heat Exchanger Calculators

Implements deterministic heat exchanger rating and sizing using:
- Log Mean Temperature Difference (LMTD) method
- Effectiveness-NTU (ε-NTU) method

All calculations are reproducible with provenance tracking.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
import hashlib
import json
import logging
import math

from ..core.config import FlowArrangement, ExchangerType

logger = logging.getLogger(__name__)


@dataclass
class LMTDResult:
    """Result from LMTD calculation."""

    LMTD_C: float
    F_correction: float
    effective_LMTD_C: float
    delta_T_hot_end_C: float
    delta_T_cold_end_C: float
    is_valid: bool
    warnings: list


@dataclass
class NTUResult:
    """Result from NTU-effectiveness calculation."""

    effectiveness: float
    NTU: float
    C_ratio: float
    C_min_kW_K: float
    C_max_kW_K: float
    Q_max_kW: float
    Q_actual_kW: float


@dataclass
class SizingResult:
    """Heat exchanger sizing result."""

    duty_kW: float
    UA_kW_K: float
    area_m2: float
    LMTD_C: float
    F_correction: float
    U_assumed_W_m2K: float
    is_feasible: bool
    warnings: list
    provenance_hash: str


class LMTDCalculator:
    """
    Log Mean Temperature Difference calculator.

    Provides deterministic calculations for heat exchanger
    thermal design using the LMTD method with F-correction
    factors for various flow arrangements.

    Example:
        >>> calc = LMTDCalculator()
        >>> result = calc.calculate(
        ...     T_hot_in=150.0, T_hot_out=90.0,
        ...     T_cold_in=30.0, T_cold_out=80.0,
        ... )
        >>> print(f"LMTD = {result.LMTD_C:.2f}°C")
    """

    VERSION = "1.0.0"

    def __init__(self, tolerance: float = 0.1) -> None:
        """
        Initialize LMTD calculator.

        Args:
            tolerance: Minimum temperature difference for valid LMTD
        """
        self.tolerance = tolerance

    def calculate(
        self,
        T_hot_in: float,
        T_hot_out: float,
        T_cold_in: float,
        T_cold_out: float,
        flow_arrangement: FlowArrangement = FlowArrangement.COUNTER_CURRENT,
    ) -> LMTDResult:
        """
        Calculate LMTD and F-correction factor.

        Args:
            T_hot_in: Hot fluid inlet temperature (°C)
            T_hot_out: Hot fluid outlet temperature (°C)
            T_cold_in: Cold fluid inlet temperature (°C)
            T_cold_out: Cold fluid outlet temperature (°C)
            flow_arrangement: Heat exchanger flow pattern

        Returns:
            LMTDResult with LMTD and correction factor
        """
        warnings = []

        # Temperature differences at each end
        if flow_arrangement == FlowArrangement.CO_CURRENT:
            delta_T1 = T_hot_in - T_cold_in  # Hot end
            delta_T2 = T_hot_out - T_cold_out  # Cold end
        else:
            # Counter-current (default)
            delta_T1 = T_hot_in - T_cold_out  # Hot end
            delta_T2 = T_hot_out - T_cold_in  # Cold end

        # Check for temperature cross
        if delta_T1 <= 0 or delta_T2 <= 0:
            warnings.append("Temperature cross detected - infeasible design")
            return LMTDResult(
                LMTD_C=0.0,
                F_correction=0.0,
                effective_LMTD_C=0.0,
                delta_T_hot_end_C=delta_T1,
                delta_T_cold_end_C=delta_T2,
                is_valid=False,
                warnings=warnings,
            )

        # Calculate LMTD
        if abs(delta_T1 - delta_T2) < self.tolerance:
            # Special case: equal temperature differences
            LMTD = delta_T1
        else:
            LMTD = (delta_T1 - delta_T2) / math.log(delta_T1 / delta_T2)

        # Calculate F-correction factor
        F = self._calculate_F_factor(
            T_hot_in, T_hot_out,
            T_cold_in, T_cold_out,
            flow_arrangement,
        )

        if F < 0.75:
            warnings.append(f"Low F-factor ({F:.3f}) - consider different arrangement")

        effective_LMTD = LMTD * F

        return LMTDResult(
            LMTD_C=round(LMTD, 3),
            F_correction=round(F, 4),
            effective_LMTD_C=round(effective_LMTD, 3),
            delta_T_hot_end_C=round(delta_T1, 2),
            delta_T_cold_end_C=round(delta_T2, 2),
            is_valid=True,
            warnings=warnings,
        )

    def _calculate_F_factor(
        self,
        T_hot_in: float,
        T_hot_out: float,
        T_cold_in: float,
        T_cold_out: float,
        flow_arrangement: FlowArrangement,
    ) -> float:
        """
        Calculate LMTD correction factor F.

        F = 1.0 for pure counter-current and co-current flow.
        F < 1.0 for shell-and-tube and cross-flow arrangements.
        """
        if flow_arrangement in (
            FlowArrangement.COUNTER_CURRENT,
            FlowArrangement.CO_CURRENT,
        ):
            return 1.0

        # Calculate P and R parameters
        # P = (T_cold_out - T_cold_in) / (T_hot_in - T_cold_in)
        # R = (T_hot_in - T_hot_out) / (T_cold_out - T_cold_in)
        denom_P = T_hot_in - T_cold_in
        denom_R = T_cold_out - T_cold_in

        if abs(denom_P) < self.tolerance or abs(denom_R) < self.tolerance:
            return 1.0

        P = (T_cold_out - T_cold_in) / denom_P
        R = (T_hot_in - T_hot_out) / denom_R

        # Limit P to valid range
        P = max(0.001, min(0.999, P))

        if flow_arrangement == FlowArrangement.SHELL_PASS_1_TUBE_2:
            # 1-2 shell and tube exchanger
            return self._F_factor_1_2(P, R)
        elif flow_arrangement == FlowArrangement.SHELL_PASS_2_TUBE_4:
            # 2-4 shell and tube exchanger
            return self._F_factor_2_4(P, R)
        elif flow_arrangement == FlowArrangement.CROSS_FLOW:
            # Cross-flow approximation
            return self._F_factor_crossflow(P, R)
        else:
            return 1.0

    def _F_factor_1_2(self, P: float, R: float) -> float:
        """F-factor for 1-2 shell and tube exchanger."""
        if abs(R - 1.0) < 0.001:
            # Special case R = 1
            return (P * math.sqrt(2)) / ((1 - P) * math.log(
                (2 - P * (2 - math.sqrt(2))) /
                (2 - P * (2 + math.sqrt(2)))
            ))

        try:
            S = math.sqrt(R * R + 1)
            term1 = (1 - P) / (1 - P * R)
            if term1 <= 0:
                return 0.5

            ln_term = math.log(term1)
            numerator = S * ln_term

            A = (2 / P - 1 - R + S) / (2 / P - 1 - R - S)
            if A <= 0:
                return 0.5

            denominator = (R - 1) * math.log(A)

            if abs(denominator) < 1e-10:
                return 1.0

            F = numerator / denominator
            return max(0.5, min(1.0, F))

        except (ValueError, ZeroDivisionError):
            return 0.75

    def _F_factor_2_4(self, P: float, R: float) -> float:
        """F-factor for 2-4 shell and tube exchanger."""
        # Approximate by cascading 1-2 factors
        F_1_2 = self._F_factor_1_2(P, R)
        # 2-4 typically gives higher F for same P, R
        return min(1.0, F_1_2 * 1.05)

    def _F_factor_crossflow(self, P: float, R: float) -> float:
        """F-factor for cross-flow (both fluids unmixed)."""
        # Approximation for cross-flow
        try:
            term = 1 - math.exp(-R * (1 - math.exp(-P)))
            if R > 0:
                eff = term / R
            else:
                eff = P
            # F is approximately effectiveness-based correction
            return max(0.7, min(1.0, 0.95 - 0.05 * (1 - eff)))
        except:
            return 0.85


class NTUCalculator:
    """
    Effectiveness-NTU method calculator.

    Used for rating existing exchangers and for design when
    outlet temperatures are unknown.

    Example:
        >>> calc = NTUCalculator()
        >>> result = calc.calculate_effectiveness(
        ...     NTU=2.0, C_ratio=0.5,
        ...     flow_arrangement=FlowArrangement.COUNTER_CURRENT
        ... )
        >>> print(f"Effectiveness = {result.effectiveness:.3f}")
    """

    VERSION = "1.0.0"

    def calculate_effectiveness(
        self,
        NTU: float,
        C_ratio: float,
        flow_arrangement: FlowArrangement = FlowArrangement.COUNTER_CURRENT,
    ) -> float:
        """
        Calculate heat exchanger effectiveness from NTU.

        Args:
            NTU: Number of transfer units (UA/C_min)
            C_ratio: Capacity ratio (C_min/C_max)
            flow_arrangement: Flow configuration

        Returns:
            Effectiveness (0 to 1)
        """
        C_r = max(0.0, min(1.0, C_ratio))

        if NTU <= 0:
            return 0.0

        if flow_arrangement == FlowArrangement.COUNTER_CURRENT:
            return self._effectiveness_counter_current(NTU, C_r)
        elif flow_arrangement == FlowArrangement.CO_CURRENT:
            return self._effectiveness_co_current(NTU, C_r)
        elif flow_arrangement == FlowArrangement.SHELL_PASS_1_TUBE_2:
            return self._effectiveness_1_2_shell(NTU, C_r)
        elif flow_arrangement == FlowArrangement.CROSS_FLOW:
            return self._effectiveness_crossflow(NTU, C_r)
        else:
            return self._effectiveness_counter_current(NTU, C_r)

    def calculate_NTU(
        self,
        effectiveness: float,
        C_ratio: float,
        flow_arrangement: FlowArrangement = FlowArrangement.COUNTER_CURRENT,
    ) -> float:
        """
        Calculate NTU from effectiveness.

        Args:
            effectiveness: Heat exchanger effectiveness (0 to 1)
            C_ratio: Capacity ratio (C_min/C_max)
            flow_arrangement: Flow configuration

        Returns:
            NTU value
        """
        eps = max(0.001, min(0.999, effectiveness))
        C_r = max(0.0, min(1.0, C_ratio))

        if flow_arrangement == FlowArrangement.COUNTER_CURRENT:
            return self._NTU_counter_current(eps, C_r)
        elif flow_arrangement == FlowArrangement.CO_CURRENT:
            return self._NTU_co_current(eps, C_r)
        else:
            return self._NTU_counter_current(eps, C_r)

    def _effectiveness_counter_current(self, NTU: float, C_r: float) -> float:
        """Effectiveness for counter-current flow."""
        if abs(C_r - 1.0) < 0.001:
            # Special case C_r = 1
            return NTU / (1 + NTU)

        exp_term = math.exp(-NTU * (1 - C_r))
        return (1 - exp_term) / (1 - C_r * exp_term)

    def _effectiveness_co_current(self, NTU: float, C_r: float) -> float:
        """Effectiveness for co-current (parallel) flow."""
        return (1 - math.exp(-NTU * (1 + C_r))) / (1 + C_r)

    def _effectiveness_1_2_shell(self, NTU: float, C_r: float) -> float:
        """Effectiveness for 1-2 shell and tube exchanger."""
        try:
            E = math.sqrt(1 + C_r * C_r)
            term = (1 + math.exp(-NTU * E)) / (1 - math.exp(-NTU * E))
            return 2 / (1 + C_r + E * term)
        except:
            return self._effectiveness_counter_current(NTU, C_r) * 0.9

    def _effectiveness_crossflow(self, NTU: float, C_r: float) -> float:
        """Effectiveness for cross-flow (both fluids unmixed)."""
        try:
            term1 = NTU ** 0.22
            term2 = math.exp(-C_r * NTU ** 0.78) - 1
            return 1 - math.exp(term2 / (C_r * term1))
        except:
            return self._effectiveness_counter_current(NTU, C_r) * 0.85

    def _NTU_counter_current(self, eps: float, C_r: float) -> float:
        """NTU for counter-current flow from effectiveness."""
        if abs(C_r - 1.0) < 0.001:
            return eps / (1 - eps)

        return math.log((1 - C_r * eps) / (1 - eps)) / (1 - C_r)

    def _NTU_co_current(self, eps: float, C_r: float) -> float:
        """NTU for co-current flow from effectiveness."""
        return -math.log(1 - eps * (1 + C_r)) / (1 + C_r)

    def size_exchanger(
        self,
        duty_kW: float,
        T_hot_in: float,
        T_hot_out: float,
        T_cold_in: float,
        T_cold_out: float,
        m_dot_hot_kg_s: float,
        Cp_hot_kJ_kgK: float,
        m_dot_cold_kg_s: float,
        Cp_cold_kJ_kgK: float,
        U_assumed_W_m2K: float = 500.0,
        flow_arrangement: FlowArrangement = FlowArrangement.COUNTER_CURRENT,
    ) -> SizingResult:
        """
        Size heat exchanger given duty and temperatures.

        Args:
            duty_kW: Required heat duty
            T_hot_in, T_hot_out: Hot side temperatures
            T_cold_in, T_cold_out: Cold side temperatures
            m_dot_hot_kg_s, Cp_hot_kJ_kgK: Hot side flow and Cp
            m_dot_cold_kg_s, Cp_cold_kJ_kgK: Cold side flow and Cp
            U_assumed_W_m2K: Assumed overall heat transfer coefficient
            flow_arrangement: Flow pattern

        Returns:
            SizingResult with area and UA
        """
        warnings = []

        # Calculate heat capacity rates
        C_hot = m_dot_hot_kg_s * Cp_hot_kJ_kgK  # kW/K
        C_cold = m_dot_cold_kg_s * Cp_cold_kJ_kgK  # kW/K

        # Calculate LMTD
        lmtd_calc = LMTDCalculator()
        lmtd_result = lmtd_calc.calculate(
            T_hot_in, T_hot_out, T_cold_in, T_cold_out, flow_arrangement
        )

        if not lmtd_result.is_valid:
            return SizingResult(
                duty_kW=duty_kW,
                UA_kW_K=0.0,
                area_m2=0.0,
                LMTD_C=0.0,
                F_correction=0.0,
                U_assumed_W_m2K=U_assumed_W_m2K,
                is_feasible=False,
                warnings=lmtd_result.warnings,
                provenance_hash="",
            )

        warnings.extend(lmtd_result.warnings)

        # Calculate UA
        effective_LMTD = lmtd_result.effective_LMTD_C
        if effective_LMTD > 0:
            UA_kW_K = duty_kW / effective_LMTD
        else:
            UA_kW_K = 0.0
            warnings.append("Cannot calculate UA - zero LMTD")

        # Calculate area
        U_kW_m2K = U_assumed_W_m2K / 1000.0
        if U_kW_m2K > 0:
            area_m2 = UA_kW_K / U_kW_m2K
        else:
            area_m2 = 0.0

        # Compute provenance hash
        input_data = {
            "duty_kW": duty_kW,
            "T_hot_in": T_hot_in,
            "T_hot_out": T_hot_out,
            "T_cold_in": T_cold_in,
            "T_cold_out": T_cold_out,
            "U_assumed": U_assumed_W_m2K,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(input_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        return SizingResult(
            duty_kW=round(duty_kW, 2),
            UA_kW_K=round(UA_kW_K, 4),
            area_m2=round(area_m2, 2),
            LMTD_C=lmtd_result.LMTD_C,
            F_correction=lmtd_result.F_correction,
            U_assumed_W_m2K=U_assumed_W_m2K,
            is_feasible=True,
            warnings=warnings,
            provenance_hash=provenance_hash,
        )
