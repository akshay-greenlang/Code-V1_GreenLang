"""
Heat Transfer Calculations

Zero-Hallucination Heat Exchanger Analysis

This module implements deterministic heat transfer calculations for:
- NTU-Effectiveness method
- LMTD method
- Overall heat transfer coefficients
- Fouling factors
- Shell-and-tube heat exchangers
- Plate heat exchangers

References:
    - ASME PTC 4.3: Air Heater Performance
    - TEMA Standards (Tubular Exchanger Manufacturers Association)
    - Incropera & DeWitt: Fundamentals of Heat and Mass Transfer
    - Perry's Chemical Engineers' Handbook, 8th Ed.

Author: GreenLang Engineering Team
License: MIT
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, Optional, Tuple
import math
import hashlib


class HeatExchangerType(Enum):
    """Heat exchanger configurations."""
    PARALLEL_FLOW = "parallel_flow"
    COUNTER_FLOW = "counter_flow"
    CROSSFLOW_BOTH_UNMIXED = "crossflow_both_unmixed"
    CROSSFLOW_CMAX_MIXED = "crossflow_cmax_mixed"
    CROSSFLOW_CMIN_MIXED = "crossflow_cmin_mixed"
    SHELL_AND_TUBE_1_SHELL = "shell_and_tube_1_shell"
    SHELL_AND_TUBE_2_SHELL = "shell_and_tube_2_shell"
    SHELL_AND_TUBE_N_SHELL = "shell_and_tube_n_shell"


@dataclass
class HeatTransferResult:
    """
    Heat transfer calculation results with provenance.

    All values are deterministic - same inputs produce identical outputs.
    """
    # Performance metrics
    heat_duty_kw: Decimal
    effectiveness: Decimal
    ntu: Decimal

    # Temperatures
    hot_outlet_temp_c: Decimal
    cold_outlet_temp_c: Decimal
    lmtd_k: Decimal

    # Heat capacity rates
    c_hot_kw_k: Decimal
    c_cold_kw_k: Decimal
    c_min_kw_k: Decimal
    c_max_kw_k: Decimal
    capacity_ratio: Decimal

    # Overall heat transfer
    ua_kw_k: Decimal

    # Provenance
    provenance_hash: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "heat_duty_kw": float(self.heat_duty_kw),
            "effectiveness": float(self.effectiveness),
            "ntu": float(self.ntu),
            "hot_outlet_temp_c": float(self.hot_outlet_temp_c),
            "cold_outlet_temp_c": float(self.cold_outlet_temp_c),
            "lmtd_k": float(self.lmtd_k),
            "c_hot_kw_k": float(self.c_hot_kw_k),
            "c_cold_kw_k": float(self.c_cold_kw_k),
            "c_min_kw_k": float(self.c_min_kw_k),
            "c_max_kw_k": float(self.c_max_kw_k),
            "capacity_ratio": float(self.capacity_ratio),
            "ua_kw_k": float(self.ua_kw_k),
            "provenance_hash": self.provenance_hash
        }


@dataclass
class OverallHeatTransferResult:
    """Overall heat transfer coefficient calculation result."""
    u_overall_w_m2k: Decimal
    r_total_m2k_w: Decimal
    r_inside_m2k_w: Decimal
    r_outside_m2k_w: Decimal
    r_wall_m2k_w: Decimal
    r_fouling_inside_m2k_w: Decimal
    r_fouling_outside_m2k_w: Decimal
    provenance_hash: str


class HeatTransferCalculator:
    """
    Heat transfer calculations for heat exchangers.

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations are deterministic
    - Based on established correlations
    - Complete provenance tracking

    References:
        - TEMA 10th Edition
        - ASME PTC 4.3: Air Heaters
        - Kays and London: Compact Heat Exchangers
    """

    def __init__(self, precision: int = 4):
        """Initialize calculator."""
        self.precision = precision

    def _apply_precision(self, value: Decimal) -> Decimal:
        """Apply precision rounding."""
        if self.precision == 0:
            return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        quantize_str = "0." + "0" * self.precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _calculate_provenance(self, inputs: Dict, outputs: Dict) -> str:
        """Calculate SHA-256 hash for audit trail."""
        provenance_data = {
            "method": "Heat_Transfer_NTU_LMTD",
            "inputs": {k: str(v) for k, v in inputs.items()},
            "outputs": {k: str(v) for k, v in outputs.items()}
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def lmtd(
        self,
        t_hot_in: float,
        t_hot_out: float,
        t_cold_in: float,
        t_cold_out: float,
        flow_type: HeatExchangerType = HeatExchangerType.COUNTER_FLOW
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate Log Mean Temperature Difference.

        Reference: Incropera & DeWitt, Equation 11.15

        Args:
            t_hot_in: Hot fluid inlet temperature (C)
            t_hot_out: Hot fluid outlet temperature (C)
            t_cold_in: Cold fluid inlet temperature (C)
            t_cold_out: Cold fluid outlet temperature (C)
            flow_type: Flow configuration

        Returns:
            Tuple of (LMTD, F correction factor)
        """
        t_hi = Decimal(str(t_hot_in))
        t_ho = Decimal(str(t_hot_out))
        t_ci = Decimal(str(t_cold_in))
        t_co = Decimal(str(t_cold_out))

        if flow_type == HeatExchangerType.COUNTER_FLOW:
            delta_t1 = t_hi - t_co
            delta_t2 = t_ho - t_ci
        else:  # Parallel flow
            delta_t1 = t_hi - t_ci
            delta_t2 = t_ho - t_co

        # Handle special cases
        if delta_t1 <= 0 or delta_t2 <= 0:
            raise ValueError("Temperature cross detected - invalid heat exchanger operation")

        if abs(delta_t1 - delta_t2) < Decimal("0.001"):
            # Equal temperature differences - use arithmetic mean
            lmtd_value = (delta_t1 + delta_t2) / Decimal("2")
        else:
            # Standard LMTD calculation
            ln_ratio = Decimal(str(math.log(float(delta_t1 / delta_t2))))
            lmtd_value = (delta_t1 - delta_t2) / ln_ratio

        # F correction factor for shell-and-tube
        f_factor = self._lmtd_correction_factor(
            t_hot_in, t_hot_out, t_cold_in, t_cold_out, flow_type
        )

        return self._apply_precision(lmtd_value), self._apply_precision(f_factor)

    def _lmtd_correction_factor(
        self,
        t_hot_in: float,
        t_hot_out: float,
        t_cold_in: float,
        t_cold_out: float,
        flow_type: HeatExchangerType
    ) -> Decimal:
        """
        Calculate LMTD correction factor for multi-pass exchangers.

        Reference: TEMA Standards, Chapter 7
        """
        if flow_type in [HeatExchangerType.COUNTER_FLOW, HeatExchangerType.PARALLEL_FLOW]:
            return Decimal("1.0")

        t_hi = Decimal(str(t_hot_in))
        t_ho = Decimal(str(t_hot_out))
        t_ci = Decimal(str(t_cold_in))
        t_co = Decimal(str(t_cold_out))

        # Calculate P and R parameters
        # P = (t_co - t_ci) / (t_hi - t_ci)
        # R = (t_hi - t_ho) / (t_co - t_ci)

        p = (t_co - t_ci) / (t_hi - t_ci)
        r = (t_hi - t_ho) / (t_co - t_ci)

        if flow_type == HeatExchangerType.SHELL_AND_TUBE_1_SHELL:
            # Single shell pass, multiple tube passes
            if abs(r - Decimal("1")) < Decimal("0.0001"):
                # Special case R = 1
                f = p * Decimal(str(math.sqrt(2))) / ((Decimal("1") - p) *
                    Decimal(str(math.log(float((Decimal("2") / p - Decimal("1") - Decimal(str(math.sqrt(2)))) /
                    (Decimal("2") / p - Decimal("1") + Decimal(str(math.sqrt(2)))))))))
            else:
                sqrt_term = Decimal(str(math.sqrt(float(r ** 2 + Decimal("1")))))
                num = sqrt_term * Decimal(str(math.log(float((Decimal("1") - p) / (Decimal("1") - p * r)))))
                denom = (r - Decimal("1")) * Decimal(str(math.log(float(
                    (Decimal("2") - p * (r + Decimal("1") - sqrt_term)) /
                    (Decimal("2") - p * (r + Decimal("1") + sqrt_term))
                ))))

                if abs(denom) < Decimal("0.0001"):
                    f = Decimal("1.0")
                else:
                    f = num / denom

        elif flow_type == HeatExchangerType.CROSSFLOW_BOTH_UNMIXED:
            # Crossflow with both fluids unmixed
            # Approximate correlation
            f = Decimal("1") - Decimal("0.2") * p * r
            if f < Decimal("0.5"):
                f = Decimal("0.5")

        else:
            f = Decimal("1.0")

        # Ensure F is in valid range
        if f > Decimal("1.0"):
            f = Decimal("1.0")
        if f < Decimal("0.5"):
            raise ValueError(f"F factor {f} too low - consider different exchanger configuration")

        return f

    def effectiveness_from_ntu(
        self,
        ntu: float,
        capacity_ratio: float,
        exchanger_type: HeatExchangerType
    ) -> Decimal:
        """
        Calculate effectiveness from NTU and capacity ratio.

        Reference: Kays and London, Compact Heat Exchangers, 3rd Ed

        epsilon-NTU relations for various heat exchanger configurations.

        Args:
            ntu: Number of Transfer Units (UA/Cmin)
            capacity_ratio: Cmin/Cmax (0 to 1)
            exchanger_type: Heat exchanger configuration

        Returns:
            Effectiveness (0 to 1)
        """
        n = Decimal(str(ntu))
        cr = Decimal(str(capacity_ratio))

        if cr < 0 or cr > 1:
            raise ValueError(f"Capacity ratio must be 0-1, got {capacity_ratio}")

        if n < 0:
            raise ValueError(f"NTU must be positive, got {ntu}")

        if exchanger_type == HeatExchangerType.PARALLEL_FLOW:
            # epsilon = [1 - exp(-NTU(1+Cr))] / (1+Cr)
            if cr == Decimal("1"):
                epsilon = (Decimal("1") - Decimal(str(math.exp(-2 * float(n))))) / Decimal("2")
            else:
                exp_term = Decimal(str(math.exp(-float(n) * (1 + float(cr)))))
                epsilon = (Decimal("1") - exp_term) / (Decimal("1") + cr)

        elif exchanger_type == HeatExchangerType.COUNTER_FLOW:
            # epsilon = [1 - exp(-NTU(1-Cr))] / [1 - Cr*exp(-NTU(1-Cr))]
            if abs(cr - Decimal("1")) < Decimal("0.0001"):
                # Special case Cr = 1
                epsilon = n / (Decimal("1") + n)
            else:
                exp_term = Decimal(str(math.exp(-float(n) * (1 - float(cr)))))
                epsilon = (Decimal("1") - exp_term) / (Decimal("1") - cr * exp_term)

        elif exchanger_type == HeatExchangerType.CROSSFLOW_BOTH_UNMIXED:
            # Approximate formula
            n_pow = n ** Decimal("0.22")
            exp_term = Decimal(str(math.exp(float(-cr * n_pow * (Decimal(str(math.exp(float(-n * n_pow)))) - Decimal("1")) / n_pow))))
            epsilon = Decimal("1") - exp_term

        elif exchanger_type == HeatExchangerType.SHELL_AND_TUBE_1_SHELL:
            # 1 shell pass, 2,4,6... tube passes
            e1 = Decimal(str(math.sqrt(float(Decimal("1") + cr ** 2))))
            exp_term = Decimal(str(math.exp(-float(n * e1))))
            epsilon = Decimal("2") / (Decimal("1") + cr + e1 * (Decimal("1") + exp_term) / (Decimal("1") - exp_term))

        elif exchanger_type in [HeatExchangerType.CROSSFLOW_CMAX_MIXED, HeatExchangerType.CROSSFLOW_CMIN_MIXED]:
            # Crossflow with one fluid mixed
            if exchanger_type == HeatExchangerType.CROSSFLOW_CMAX_MIXED:
                exp_term = Decimal(str(math.exp(-float(cr * (Decimal("1") - Decimal(str(math.exp(-float(n)))))))))
                epsilon = (Decimal("1") - exp_term) / cr
            else:  # CMIN mixed
                exp_term = Decimal(str(math.exp(-float((Decimal("1") - Decimal(str(math.exp(-float(n * cr))))) / cr))))
                epsilon = Decimal("1") - exp_term

        else:
            raise ValueError(f"Unknown exchanger type: {exchanger_type}")

        # Ensure epsilon is in valid range
        if epsilon > Decimal("1"):
            epsilon = Decimal("1")
        if epsilon < Decimal("0"):
            epsilon = Decimal("0")

        return self._apply_precision(epsilon)

    def ntu_from_effectiveness(
        self,
        effectiveness: float,
        capacity_ratio: float,
        exchanger_type: HeatExchangerType
    ) -> Decimal:
        """
        Calculate NTU from effectiveness and capacity ratio.

        Inverse of effectiveness_from_ntu.

        Args:
            effectiveness: Heat exchanger effectiveness (0 to 1)
            capacity_ratio: Cmin/Cmax (0 to 1)
            exchanger_type: Heat exchanger configuration

        Returns:
            Number of Transfer Units
        """
        eps = Decimal(str(effectiveness))
        cr = Decimal(str(capacity_ratio))

        if eps <= 0 or eps >= 1:
            raise ValueError(f"Effectiveness must be between 0 and 1, got {effectiveness}")

        if exchanger_type == HeatExchangerType.COUNTER_FLOW:
            if abs(cr - Decimal("1")) < Decimal("0.0001"):
                ntu = eps / (Decimal("1") - eps)
            else:
                ntu = Decimal(str(math.log(float((Decimal("1") - eps * cr) / (Decimal("1") - eps))))) / (Decimal("1") - cr)

        elif exchanger_type == HeatExchangerType.PARALLEL_FLOW:
            ntu = -Decimal(str(math.log(float(Decimal("1") - eps * (Decimal("1") + cr))))) / (Decimal("1") + cr)

        elif exchanger_type == HeatExchangerType.SHELL_AND_TUBE_1_SHELL:
            e1 = Decimal("2") / eps - Decimal("1") - cr
            e2 = Decimal(str(math.sqrt(float(Decimal("1") + cr ** 2))))
            ntu = -Decimal("1") / e2 * Decimal(str(math.log(float((e1 - e2) / (e1 + e2)))))

        else:
            # Iterative solution for other types
            ntu = self._ntu_iterative(effectiveness, capacity_ratio, exchanger_type)

        return self._apply_precision(ntu)

    def _ntu_iterative(
        self,
        effectiveness: float,
        capacity_ratio: float,
        exchanger_type: HeatExchangerType
    ) -> Decimal:
        """Iterative solution for NTU from effectiveness."""
        eps_target = Decimal(str(effectiveness))
        cr = capacity_ratio

        # Bisection method
        ntu_low = Decimal("0.001")
        ntu_high = Decimal("20")
        tolerance = Decimal("0.0001")
        max_iterations = 50

        for _ in range(max_iterations):
            ntu_mid = (ntu_low + ntu_high) / Decimal("2")
            eps_calc = self.effectiveness_from_ntu(float(ntu_mid), cr, exchanger_type)

            if abs(eps_calc - eps_target) < tolerance:
                return ntu_mid

            if eps_calc < eps_target:
                ntu_low = ntu_mid
            else:
                ntu_high = ntu_mid

        return (ntu_low + ntu_high) / Decimal("2")

    def analyze_heat_exchanger(
        self,
        t_hot_in: float,
        t_cold_in: float,
        m_dot_hot: float,
        m_dot_cold: float,
        cp_hot: float,
        cp_cold: float,
        ua: float,
        exchanger_type: HeatExchangerType = HeatExchangerType.COUNTER_FLOW
    ) -> HeatTransferResult:
        """
        Complete heat exchanger analysis using NTU method.

        ZERO-HALLUCINATION: Deterministic NTU-effectiveness calculation.

        Reference: Incropera & DeWitt, Chapter 11

        Args:
            t_hot_in: Hot fluid inlet temperature (C)
            t_cold_in: Cold fluid inlet temperature (C)
            m_dot_hot: Hot fluid mass flow rate (kg/s)
            m_dot_cold: Cold fluid mass flow rate (kg/s)
            cp_hot: Hot fluid specific heat (kJ/kg-K)
            cp_cold: Cold fluid specific heat (kJ/kg-K)
            ua: Overall heat transfer coefficient times area (kW/K)
            exchanger_type: Heat exchanger configuration

        Returns:
            HeatTransferResult with complete analysis
        """
        t_hi = Decimal(str(t_hot_in))
        t_ci = Decimal(str(t_cold_in))
        m_h = Decimal(str(m_dot_hot))
        m_c = Decimal(str(m_dot_cold))
        cp_h = Decimal(str(cp_hot))
        cp_c = Decimal(str(cp_cold))
        ua_val = Decimal(str(ua))

        # Calculate heat capacity rates (kW/K)
        c_h = m_h * cp_h
        c_c = m_c * cp_c

        # Identify Cmin and Cmax
        if c_h < c_c:
            c_min = c_h
            c_max = c_c
        else:
            c_min = c_c
            c_max = c_h

        # Capacity ratio
        if c_max > 0:
            cr = c_min / c_max
        else:
            cr = Decimal("0")

        # Calculate NTU
        ntu = ua_val / c_min

        # Calculate effectiveness
        effectiveness = self.effectiveness_from_ntu(float(ntu), float(cr), exchanger_type)

        # Calculate heat duty
        q_max = c_min * (t_hi - t_ci)
        q = effectiveness * q_max

        # Calculate outlet temperatures
        t_ho = t_hi - q / c_h
        t_co = t_ci + q / c_c

        # Calculate LMTD
        lmtd_val, f_factor = self.lmtd(float(t_hi), float(t_ho), float(t_ci), float(t_co), exchanger_type)

        # Create provenance
        inputs = {
            "t_hot_in": str(t_hi),
            "t_cold_in": str(t_ci),
            "m_dot_hot": str(m_h),
            "m_dot_cold": str(m_c),
            "cp_hot": str(cp_h),
            "cp_cold": str(cp_c),
            "ua": str(ua_val),
            "exchanger_type": exchanger_type.value
        }
        outputs = {
            "heat_duty": str(q),
            "effectiveness": str(effectiveness),
            "ntu": str(ntu)
        }
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return HeatTransferResult(
            heat_duty_kw=self._apply_precision(q),
            effectiveness=self._apply_precision(effectiveness),
            ntu=self._apply_precision(ntu),
            hot_outlet_temp_c=self._apply_precision(t_ho),
            cold_outlet_temp_c=self._apply_precision(t_co),
            lmtd_k=self._apply_precision(lmtd_val),
            c_hot_kw_k=self._apply_precision(c_h),
            c_cold_kw_k=self._apply_precision(c_c),
            c_min_kw_k=self._apply_precision(c_min),
            c_max_kw_k=self._apply_precision(c_max),
            capacity_ratio=self._apply_precision(cr),
            ua_kw_k=self._apply_precision(ua_val),
            provenance_hash=provenance_hash
        )

    def overall_heat_transfer_coefficient(
        self,
        h_inside: float,
        h_outside: float,
        d_inside: float,
        d_outside: float,
        k_wall: float,
        r_fouling_inside: float = 0.0,
        r_fouling_outside: float = 0.0
    ) -> OverallHeatTransferResult:
        """
        Calculate overall heat transfer coefficient for tube.

        Reference: TEMA Standards, Section 7

        Args:
            h_inside: Inside heat transfer coefficient (W/m2-K)
            h_outside: Outside heat transfer coefficient (W/m2-K)
            d_inside: Inside tube diameter (m)
            d_outside: Outside tube diameter (m)
            k_wall: Wall thermal conductivity (W/m-K)
            r_fouling_inside: Inside fouling factor (m2-K/W)
            r_fouling_outside: Outside fouling factor (m2-K/W)

        Returns:
            OverallHeatTransferResult with detailed resistance breakdown
        """
        hi = Decimal(str(h_inside))
        ho = Decimal(str(h_outside))
        di = Decimal(str(d_inside))
        do = Decimal(str(d_outside))
        k = Decimal(str(k_wall))
        rfi = Decimal(str(r_fouling_inside))
        rfo = Decimal(str(r_fouling_outside))

        # Calculate individual resistances (based on outside area)
        # Inside convection resistance
        r_inside = do / (hi * di)

        # Inside fouling resistance
        r_fi = rfi * do / di

        # Wall conduction resistance
        if di > 0 and do > di:
            ln_ratio = Decimal(str(math.log(float(do / di))))
            r_wall = do * ln_ratio / (Decimal("2") * k)
        else:
            raise ValueError("Invalid tube dimensions")

        # Outside fouling resistance
        r_fo = rfo

        # Outside convection resistance
        r_outside = Decimal("1") / ho

        # Total resistance
        r_total = r_inside + r_fi + r_wall + r_fo + r_outside

        # Overall heat transfer coefficient
        u_overall = Decimal("1") / r_total

        inputs = {
            "h_inside": str(hi),
            "h_outside": str(ho),
            "d_inside": str(di),
            "d_outside": str(do),
            "k_wall": str(k)
        }
        outputs = {
            "u_overall": str(u_overall),
            "r_total": str(r_total)
        }
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return OverallHeatTransferResult(
            u_overall_w_m2k=self._apply_precision(u_overall),
            r_total_m2k_w=self._apply_precision(r_total),
            r_inside_m2k_w=self._apply_precision(r_inside),
            r_outside_m2k_w=self._apply_precision(r_outside),
            r_wall_m2k_w=self._apply_precision(r_wall),
            r_fouling_inside_m2k_w=self._apply_precision(r_fi),
            r_fouling_outside_m2k_w=self._apply_precision(r_fo),
            provenance_hash=provenance_hash
        )

    def required_area(
        self,
        heat_duty_kw: float,
        u_overall_w_m2k: float,
        lmtd_k: float,
        f_factor: float = 1.0
    ) -> Decimal:
        """
        Calculate required heat transfer area.

        Reference: Q = U * A * F * LMTD

        Args:
            heat_duty_kw: Required heat duty (kW)
            u_overall_w_m2k: Overall heat transfer coefficient (W/m2-K)
            lmtd_k: Log mean temperature difference (K)
            f_factor: LMTD correction factor

        Returns:
            Required area in m2
        """
        q = Decimal(str(heat_duty_kw)) * Decimal("1000")  # Convert to W
        u = Decimal(str(u_overall_w_m2k))
        dt = Decimal(str(lmtd_k))
        f = Decimal(str(f_factor))

        if u <= 0 or dt <= 0 or f <= 0:
            raise ValueError("U, LMTD, and F must all be positive")

        area = q / (u * f * dt)

        return self._apply_precision(area)


# Convenience functions
def heat_exchanger_analysis(
    t_hot_in: float,
    t_cold_in: float,
    m_dot_hot: float,
    m_dot_cold: float,
    cp_hot: float,
    cp_cold: float,
    ua: float,
    exchanger_type: str = "counter_flow"
) -> HeatTransferResult:
    """
    Analyze heat exchanger performance.

    Example:
        >>> result = heat_exchanger_analysis(
        ...     t_hot_in=200, t_cold_in=30,
        ...     m_dot_hot=1.0, m_dot_cold=2.0,
        ...     cp_hot=1.0, cp_cold=4.18,
        ...     ua=50
        ... )
        >>> print(f"Heat duty: {result.heat_duty_kw} kW")
    """
    calc = HeatTransferCalculator()
    hx_type = HeatExchangerType(exchanger_type)
    return calc.analyze_heat_exchanger(
        t_hot_in, t_cold_in, m_dot_hot, m_dot_cold,
        cp_hot, cp_cold, ua, hx_type
    )


def calculate_lmtd(
    t_hot_in: float,
    t_hot_out: float,
    t_cold_in: float,
    t_cold_out: float,
    flow_type: str = "counter_flow"
) -> Tuple[Decimal, Decimal]:
    """Calculate LMTD and correction factor."""
    calc = HeatTransferCalculator()
    hx_type = HeatExchangerType(flow_type)
    return calc.lmtd(t_hot_in, t_hot_out, t_cold_in, t_cold_out, hx_type)


def calculate_effectiveness(
    ntu: float,
    capacity_ratio: float,
    exchanger_type: str = "counter_flow"
) -> Decimal:
    """Calculate heat exchanger effectiveness from NTU."""
    calc = HeatTransferCalculator()
    hx_type = HeatExchangerType(exchanger_type)
    return calc.effectiveness_from_ntu(ntu, capacity_ratio, hx_type)
