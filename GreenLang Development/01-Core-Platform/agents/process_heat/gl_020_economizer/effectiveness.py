"""
GL-020 ECONOPULSE - Heat Transfer Effectiveness Calculator

Calculates economizer heat transfer effectiveness using the ε-NTU method.
Compliant with ASME PTC 4.3 and standard heat transfer references.

Standards Reference:
    - ASME PTC 4.3 Air Heater Test Code
    - ASME PTC 4.1 Steam Generating Units
    - Incropera & DeWitt, Fundamentals of Heat and Mass Transfer

Zero-Hallucination: All calculations use deterministic formulas.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import hashlib
import json


# Specific heat constants (BTU/lb-F)
CP_FLUE_GAS = 0.26  # Average for combustion products
CP_WATER = 1.0  # Liquid water


@dataclass
class EffectivenessInput:
    """Input for effectiveness calculation."""
    # Temperatures
    gas_inlet_temp_f: float
    gas_outlet_temp_f: float
    water_inlet_temp_f: float
    water_outlet_temp_f: float

    # Flow rates
    gas_flow_lb_hr: float
    water_flow_lb_hr: float

    # Design values
    design_effectiveness: float = 0.80
    design_ua_btu_hr_f: float = 100000.0
    clean_ua_btu_hr_f: float = 120000.0
    design_ntu: float = 2.0

    # Heat exchanger configuration
    flow_arrangement: str = "counterflow"  # counterflow, parallel, crossflow


class EffectivenessCalculator:
    """
    Calculates heat transfer effectiveness using ε-NTU method.

    The ε-NTU method relates:
    - ε (effectiveness): actual heat transfer / maximum possible heat transfer
    - NTU (Number of Transfer Units): UA / C_min
    - C_r (capacity ratio): C_min / C_max

    For counterflow heat exchanger:
    ε = (1 - exp(-NTU(1-C_r))) / (1 - C_r*exp(-NTU(1-C_r)))

    For C_r = 1 (balanced flow):
    ε = NTU / (1 + NTU)
    """

    def __init__(
        self,
        cp_gas: float = CP_FLUE_GAS,
        cp_water: float = CP_WATER,
    ):
        """
        Initialize calculator.

        Args:
            cp_gas: Specific heat of flue gas (BTU/lb-F)
            cp_water: Specific heat of water (BTU/lb-F)
        """
        self.cp_gas = cp_gas
        self.cp_water = cp_water

    def calculate_capacity_rates(
        self,
        gas_flow_lb_hr: float,
        water_flow_lb_hr: float,
    ) -> Tuple[float, float, float, float]:
        """
        Calculate heat capacity rates.

        Args:
            gas_flow_lb_hr: Gas mass flow rate (lb/hr)
            water_flow_lb_hr: Water mass flow rate (lb/hr)

        Returns:
            Tuple of (C_gas, C_water, C_min, C_max) in BTU/hr-F
        """
        c_gas = gas_flow_lb_hr * self.cp_gas
        c_water = water_flow_lb_hr * self.cp_water

        c_min = min(c_gas, c_water)
        c_max = max(c_gas, c_water)

        return c_gas, c_water, c_min, c_max

    def calculate_lmtd(
        self,
        gas_inlet_temp_f: float,
        gas_outlet_temp_f: float,
        water_inlet_temp_f: float,
        water_outlet_temp_f: float,
        flow_arrangement: str = "counterflow",
    ) -> float:
        """
        Calculate Log Mean Temperature Difference (LMTD).

        For counterflow:
        LMTD = (ΔT1 - ΔT2) / ln(ΔT1/ΔT2)

        where:
        ΔT1 = T_gas_in - T_water_out
        ΔT2 = T_gas_out - T_water_in

        Args:
            gas_inlet_temp_f: Gas inlet temperature (F)
            gas_outlet_temp_f: Gas outlet temperature (F)
            water_inlet_temp_f: Water inlet temperature (F)
            water_outlet_temp_f: Water outlet temperature (F)
            flow_arrangement: Heat exchanger flow arrangement

        Returns:
            LMTD in degrees F
        """
        if flow_arrangement == "counterflow":
            delta_t1 = gas_inlet_temp_f - water_outlet_temp_f
            delta_t2 = gas_outlet_temp_f - water_inlet_temp_f
        else:  # parallel flow
            delta_t1 = gas_inlet_temp_f - water_inlet_temp_f
            delta_t2 = gas_outlet_temp_f - water_outlet_temp_f

        # Handle edge cases
        if delta_t1 <= 0 or delta_t2 <= 0:
            return 1.0  # Minimum value to avoid errors

        if abs(delta_t1 - delta_t2) < 0.1:
            # When delta_t1 ≈ delta_t2, LMTD = delta_t1
            return delta_t1

        try:
            lmtd = (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)
            return max(lmtd, 1.0)  # Ensure positive value
        except (ValueError, ZeroDivisionError):
            return (delta_t1 + delta_t2) / 2  # Arithmetic mean as fallback

    def calculate_actual_effectiveness(
        self,
        gas_inlet_temp_f: float,
        gas_outlet_temp_f: float,
        water_inlet_temp_f: float,
        water_outlet_temp_f: float,
        c_min: float,
        c_max: float,
    ) -> float:
        """
        Calculate actual heat transfer effectiveness from temperatures.

        ε = Q_actual / Q_max
        Q_actual = C_gas * (T_gas_in - T_gas_out) = C_water * (T_water_out - T_water_in)
        Q_max = C_min * (T_gas_in - T_water_in)

        Args:
            gas_inlet_temp_f: Gas inlet temperature (F)
            gas_outlet_temp_f: Gas outlet temperature (F)
            water_inlet_temp_f: Water inlet temperature (F)
            water_outlet_temp_f: Water outlet temperature (F)
            c_min: Minimum capacity rate (BTU/hr-F)
            c_max: Maximum capacity rate (BTU/hr-F)

        Returns:
            Heat transfer effectiveness (0-1)
        """
        # Maximum possible heat transfer
        q_max = c_min * (gas_inlet_temp_f - water_inlet_temp_f)

        if q_max <= 0:
            return 0.0

        # Actual heat transfer (use gas side)
        q_actual = (c_min if c_min == c_max else c_max) * abs(gas_inlet_temp_f - gas_outlet_temp_f)

        # Can also calculate from water side for verification
        # q_actual_water = c_water * (water_outlet_temp_f - water_inlet_temp_f)

        effectiveness = q_actual / q_max

        # Clamp to valid range
        return max(0.0, min(effectiveness, 1.0))

    def effectiveness_counterflow(self, ntu: float, c_r: float) -> float:
        """
        Calculate effectiveness for counterflow heat exchanger.

        ε = (1 - exp(-NTU(1-C_r))) / (1 - C_r*exp(-NTU(1-C_r)))

        For C_r = 1: ε = NTU / (1 + NTU)

        Args:
            ntu: Number of Transfer Units
            c_r: Capacity ratio (C_min/C_max)

        Returns:
            Effectiveness
        """
        if ntu <= 0:
            return 0.0

        if abs(c_r - 1.0) < 0.001:
            # Balanced flow case
            return ntu / (1 + ntu)

        try:
            exp_term = math.exp(-ntu * (1 - c_r))
            effectiveness = (1 - exp_term) / (1 - c_r * exp_term)
            return max(0.0, min(effectiveness, 1.0))
        except OverflowError:
            return 1.0  # Very high NTU approaches 1.0

    def effectiveness_parallel(self, ntu: float, c_r: float) -> float:
        """
        Calculate effectiveness for parallel flow heat exchanger.

        ε = (1 - exp(-NTU(1+C_r))) / (1 + C_r)

        Args:
            ntu: Number of Transfer Units
            c_r: Capacity ratio (C_min/C_max)

        Returns:
            Effectiveness
        """
        if ntu <= 0:
            return 0.0

        try:
            exp_term = math.exp(-ntu * (1 + c_r))
            effectiveness = (1 - exp_term) / (1 + c_r)
            return max(0.0, min(effectiveness, 1.0))
        except OverflowError:
            return 1.0 / (1 + c_r)  # Limiting value

    def calculate_ntu_from_effectiveness(
        self,
        effectiveness: float,
        c_r: float,
        flow_arrangement: str = "counterflow",
    ) -> float:
        """
        Calculate NTU from effectiveness (inverse relationship).

        For counterflow:
        NTU = (1/(1-C_r)) * ln((1-ε*C_r)/(1-ε))

        For C_r = 1:
        NTU = ε / (1 - ε)

        Args:
            effectiveness: Heat transfer effectiveness
            c_r: Capacity ratio
            flow_arrangement: Flow arrangement

        Returns:
            Number of Transfer Units
        """
        if effectiveness <= 0:
            return 0.0
        if effectiveness >= 1.0:
            return float('inf')

        if abs(c_r - 1.0) < 0.001:
            # Balanced flow case
            return effectiveness / (1 - effectiveness)

        if flow_arrangement == "counterflow":
            try:
                ntu = (1 / (1 - c_r)) * math.log((1 - effectiveness * c_r) / (1 - effectiveness))
                return max(0.0, ntu)
            except (ValueError, ZeroDivisionError):
                return 0.0
        else:
            # Parallel flow - solve numerically or use approximation
            # NTU = -ln(1 - ε(1+C_r)) / (1+C_r)
            try:
                ntu = -math.log(1 - effectiveness * (1 + c_r)) / (1 + c_r)
                return max(0.0, ntu)
            except (ValueError, ZeroDivisionError):
                return 0.0

    def calculate_ua_from_ntu(self, ntu: float, c_min: float) -> float:
        """
        Calculate UA from NTU.

        UA = NTU * C_min

        Args:
            ntu: Number of Transfer Units
            c_min: Minimum capacity rate (BTU/hr-F)

        Returns:
            UA value (BTU/hr-F)
        """
        return ntu * c_min

    def calculate(self, input_data: EffectivenessInput) -> dict:
        """
        Perform complete effectiveness analysis.

        Args:
            input_data: EffectivenessInput data

        Returns:
            Dictionary with calculation results
        """
        # Calculate capacity rates
        c_gas, c_water, c_min, c_max = self.calculate_capacity_rates(
            input_data.gas_flow_lb_hr,
            input_data.water_flow_lb_hr,
        )

        capacity_ratio = c_min / c_max if c_max > 0 else 0.0

        # Calculate LMTD
        lmtd = self.calculate_lmtd(
            input_data.gas_inlet_temp_f,
            input_data.gas_outlet_temp_f,
            input_data.water_inlet_temp_f,
            input_data.water_outlet_temp_f,
            input_data.flow_arrangement,
        )

        # Calculate actual effectiveness from temperatures
        actual_effectiveness = self.calculate_actual_effectiveness(
            input_data.gas_inlet_temp_f,
            input_data.gas_outlet_temp_f,
            input_data.water_inlet_temp_f,
            input_data.water_outlet_temp_f,
            c_min,
            c_max,
        )

        # Calculate actual NTU from effectiveness
        actual_ntu = self.calculate_ntu_from_effectiveness(
            actual_effectiveness,
            capacity_ratio,
            input_data.flow_arrangement,
        )

        # Calculate actual UA
        actual_ua = self.calculate_ua_from_ntu(actual_ntu, c_min)

        # Calculate actual heat duty
        actual_duty = c_gas * abs(input_data.gas_inlet_temp_f - input_data.gas_outlet_temp_f)

        # Calculate expected duty based on design effectiveness
        expected_duty = c_min * input_data.design_effectiveness * (
            input_data.gas_inlet_temp_f - input_data.water_inlet_temp_f
        )

        # Temperature differentials
        gas_temp_drop = input_data.gas_inlet_temp_f - input_data.gas_outlet_temp_f
        water_temp_rise = input_data.water_outlet_temp_f - input_data.water_inlet_temp_f
        approach_temp = input_data.gas_inlet_temp_f - input_data.water_outlet_temp_f

        # Effectiveness ratio and deviation
        effectiveness_ratio = actual_effectiveness / input_data.design_effectiveness if input_data.design_effectiveness > 0 else 0.0
        effectiveness_deviation_pct = (1 - effectiveness_ratio) * 100

        # UA degradation
        ua_degradation_pct = (1 - actual_ua / input_data.clean_ua_btu_hr_f) * 100 if input_data.clean_ua_btu_hr_f > 0 else 0.0

        # Duty deficit
        duty_deficit = expected_duty - actual_duty

        # Performance status
        if effectiveness_ratio >= 0.95:
            performance_status = "normal"
            primary_degradation_source = "none"
        elif effectiveness_ratio >= 0.80:
            performance_status = "degraded"
            primary_degradation_source = "fouling"
        else:
            performance_status = "critical"
            primary_degradation_source = "severe_fouling"

        result = {
            "current_effectiveness": round(actual_effectiveness, 4),
            "design_effectiveness": round(input_data.design_effectiveness, 4),
            "effectiveness_ratio": round(effectiveness_ratio, 4),
            "effectiveness_deviation_pct": round(effectiveness_deviation_pct, 2),
            "current_ntu": round(actual_ntu, 3),
            "design_ntu": round(input_data.design_ntu, 3),
            "current_ua_btu_hr_f": round(actual_ua, 0),
            "design_ua_btu_hr_f": round(input_data.design_ua_btu_hr_f, 0),
            "clean_ua_btu_hr_f": round(input_data.clean_ua_btu_hr_f, 0),
            "ua_degradation_pct": round(ua_degradation_pct, 2),
            "actual_duty_btu_hr": round(actual_duty, 0),
            "expected_duty_btu_hr": round(expected_duty, 0),
            "duty_deficit_btu_hr": round(duty_deficit, 0),
            "lmtd_f": round(lmtd, 1),
            "approach_temp_f": round(approach_temp, 1),
            "gas_temp_drop_f": round(gas_temp_drop, 1),
            "water_temp_rise_f": round(water_temp_rise, 1),
            "c_min_btu_hr_f": round(c_min, 0),
            "c_max_btu_hr_f": round(c_max, 0),
            "capacity_ratio": round(capacity_ratio, 4),
            "performance_status": performance_status,
            "primary_degradation_source": primary_degradation_source,
            "calculation_method": "NTU_EPSILON",
            "formula_reference": "ASME PTC 4.3 / Incropera Heat Transfer",
        }

        # Add provenance hash
        result["provenance_hash"] = hashlib.sha256(
            json.dumps(result, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        return result


def create_effectiveness_calculator(
    cp_gas: float = CP_FLUE_GAS,
    cp_water: float = CP_WATER,
) -> EffectivenessCalculator:
    """Factory function to create EffectivenessCalculator."""
    return EffectivenessCalculator(cp_gas=cp_gas, cp_water=cp_water)
