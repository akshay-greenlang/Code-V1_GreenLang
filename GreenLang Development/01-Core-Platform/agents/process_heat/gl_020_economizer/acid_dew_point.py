"""
GL-020 ECONOPULSE - Acid Dew Point Calculator

Calculates sulfuric acid and water dew points for cold-end corrosion prevention.
Uses the Verhoff-Banchero correlation for H2SO4 dew point.

Standards Reference:
    - Verhoff & Banchero, Chemical Engineering Progress, 1974
    - EPA Method 6C for SO2 measurement
    - ASME PTC 4.1 Steam Generating Units

Zero-Hallucination: All calculations are deterministic formulas.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import hashlib
import json


@dataclass
class AcidDewPointInput:
    """Input for acid dew point calculation."""
    flue_gas_moisture_pct: float  # H2O content in flue gas (%)
    flue_gas_so2_ppm: Optional[float] = None  # SO2 in ppm
    fuel_sulfur_pct: float = 0.0  # Fuel sulfur content (%)
    so2_to_so3_conversion_pct: float = 2.0  # SO2 to SO3 conversion (1-5% typical)
    flue_gas_o2_pct: float = 3.0  # O2 for excess air calculation
    cold_end_metal_temp_f: float = 300.0  # Measured metal temperature
    safety_margin_f: float = 30.0  # Safety margin above dew point


class AcidDewPointCalculator:
    """
    Calculates acid dew point using Verhoff-Banchero correlation.

    The sulfuric acid dew point depends on:
    - SO3 concentration (from fuel sulfur and conversion rate)
    - H2O concentration in flue gas

    Formula (Verhoff-Banchero):
    T_adp (K) = 1000 / (2.276 - 0.0294*ln(pH2O) - 0.0858*ln(pSO3) + 0.0062*ln(pH2O)*ln(pSO3))

    where:
    - pH2O = partial pressure of water (atm)
    - pSO3 = partial pressure of SO3 (atm)
    """

    # Physical constants
    ATM_PRESSURE_MBAR = 1013.25

    # Fuel sulfur content defaults by fuel type
    FUEL_SULFUR_DEFAULTS = {
        "natural_gas": 0.001,
        "no2_fuel_oil": 0.3,
        "no6_fuel_oil": 2.0,
        "coal_bituminous": 2.5,
        "coal_sub_bituminous": 0.8,
        "biomass": 0.1,
        "refinery_gas": 0.05,
    }

    # Typical SO2 to SO3 conversion rates
    SO3_CONVERSION_RATES = {
        "natural_gas": 1.0,  # Lower due to clean combustion
        "no2_fuel_oil": 2.0,
        "no6_fuel_oil": 3.0,  # Higher due to vanadium catalyst
        "coal_bituminous": 2.0,
        "coal_sub_bituminous": 1.5,
        "biomass": 1.0,
        "refinery_gas": 1.5,
    }

    def __init__(self, safety_margin_f: float = 30.0):
        """
        Initialize calculator.

        Args:
            safety_margin_f: Safety margin above acid dew point (F)
        """
        self.safety_margin_f = safety_margin_f

    def calculate_so3_from_fuel(
        self,
        fuel_sulfur_pct: float,
        so2_to_so3_conversion_pct: float,
        excess_air_pct: float,
    ) -> float:
        """
        Calculate SO3 concentration from fuel sulfur content.

        Formula:
        SO3 (ppm) = (S% in fuel * 10000 * conversion%) / (1 + excess_air/100)

        This is a simplified mass balance assuming complete combustion.

        Args:
            fuel_sulfur_pct: Fuel sulfur content (%)
            so2_to_so3_conversion_pct: SO2 to SO3 conversion (%)
            excess_air_pct: Excess air percentage

        Returns:
            SO3 concentration in ppm
        """
        # SO2 produced from fuel sulfur (simplified stoichiometry)
        # S + O2 -> SO2, then small fraction SO2 + 0.5*O2 -> SO3
        so2_ppm = fuel_sulfur_pct * 10000 / (1 + excess_air_pct / 100)
        so3_ppm = so2_ppm * (so2_to_so3_conversion_pct / 100)

        return so3_ppm

    def calculate_excess_air(self, o2_pct: float) -> float:
        """
        Calculate excess air from O2 measurement.

        Formula (dry basis):
        Excess Air (%) = O2 / (21 - O2) * 100

        Args:
            o2_pct: Measured O2 in flue gas (%)

        Returns:
            Excess air percentage
        """
        if o2_pct >= 21.0:
            return 0.0
        return (o2_pct / (21.0 - o2_pct)) * 100

    def calculate_water_dew_point(self, h2o_pct: float) -> float:
        """
        Calculate water dew point from moisture content.

        Uses Magnus-Tetens approximation inverted for partial pressure.

        Args:
            h2o_pct: Water vapor content in flue gas (%)

        Returns:
            Water dew point in Fahrenheit
        """
        # Partial pressure of water (atm)
        p_h2o_atm = h2o_pct / 100.0

        # Convert to mmHg for Antoine equation
        p_h2o_mmhg = p_h2o_atm * 760.0

        if p_h2o_mmhg <= 0:
            return 32.0  # Return freezing point for zero moisture

        # Antoine equation inverted: T = B / (A - log10(P)) - C
        # Constants for water (NIST)
        A = 8.07131
        B = 1730.63
        C = 233.426

        try:
            t_celsius = B / (A - math.log10(p_h2o_mmhg)) - C
            t_fahrenheit = t_celsius * 9/5 + 32
            return t_fahrenheit
        except (ValueError, ZeroDivisionError):
            return 100.0  # Default if calculation fails

    def calculate_acid_dew_point_verhoff_banchero(
        self,
        h2o_pct: float,
        so3_ppm: float,
    ) -> float:
        """
        Calculate sulfuric acid dew point using Verhoff-Banchero correlation.

        Reference: Verhoff & Banchero, Chemical Engineering Progress, Vol. 70, No. 8, 1974

        Formula:
        1000/T_adp(K) = 2.276 - 0.0294*ln(pH2O) - 0.0858*ln(pSO3) + 0.0062*ln(pH2O)*ln(pSO3)

        where:
        - pH2O = partial pressure of H2O (mmHg)
        - pSO3 = partial pressure of SO3 (mmHg)

        Args:
            h2o_pct: Water vapor in flue gas (%)
            so3_ppm: SO3 concentration (ppm)

        Returns:
            Acid dew point in Fahrenheit
        """
        # Convert to partial pressures in mmHg
        # Assuming 1 atm = 760 mmHg total pressure
        p_h2o_mmhg = (h2o_pct / 100.0) * 760.0
        p_so3_mmhg = (so3_ppm / 1_000_000.0) * 760.0

        # Ensure valid inputs for logarithm
        if p_h2o_mmhg <= 0:
            p_h2o_mmhg = 0.001
        if p_so3_mmhg <= 0:
            # Very low SO3 means acid dew point approaches water dew point
            return self.calculate_water_dew_point(h2o_pct)

        # Verhoff-Banchero correlation
        ln_h2o = math.log(p_h2o_mmhg)
        ln_so3 = math.log(p_so3_mmhg)

        denominator = (
            2.276
            - 0.0294 * ln_h2o
            - 0.0858 * ln_so3
            + 0.0062 * ln_h2o * ln_so3
        )

        if denominator <= 0:
            return 400.0  # Return high temperature if correlation fails

        t_kelvin = 1000.0 / denominator
        t_celsius = t_kelvin - 273.15
        t_fahrenheit = t_celsius * 9/5 + 32

        return t_fahrenheit

    def calculate_acid_dew_point_okkes(
        self,
        h2o_pct: float,
        so3_ppm: float,
    ) -> float:
        """
        Alternative acid dew point calculation using Okkes correlation.

        Reference: Okkes, Hydrocarbon Processing, 1987

        T_adp (C) = 203.25 + 27.6*log10(pSO3) + 10.83*log10(pH2O) + 1.06*(log10(pSO3)+8)^2.19

        where pressures are in atmospheres.

        Args:
            h2o_pct: Water vapor in flue gas (%)
            so3_ppm: SO3 concentration (ppm)

        Returns:
            Acid dew point in Fahrenheit
        """
        # Convert to partial pressures in atm
        p_h2o_atm = h2o_pct / 100.0
        p_so3_atm = so3_ppm / 1_000_000.0

        if p_h2o_atm <= 0 or p_so3_atm <= 0:
            return self.calculate_water_dew_point(h2o_pct)

        log_so3 = math.log10(p_so3_atm)
        log_h2o = math.log10(p_h2o_atm)

        t_celsius = (
            203.25
            + 27.6 * log_so3
            + 10.83 * log_h2o
            + 1.06 * (log_so3 + 8) ** 2.19
        )

        t_fahrenheit = t_celsius * 9/5 + 32
        return t_fahrenheit

    def assess_corrosion_risk(
        self,
        metal_temp_f: float,
        acid_dew_point_f: float,
        safety_margin_f: float,
    ) -> Tuple[str, str]:
        """
        Assess cold-end corrosion risk.

        Args:
            metal_temp_f: Metal temperature (F)
            acid_dew_point_f: Acid dew point (F)
            safety_margin_f: Required safety margin (F)

        Returns:
            Tuple of (risk_level, recommended_action)
        """
        margin = metal_temp_f - acid_dew_point_f

        if margin < 0:
            return ("critical", "IMMEDIATE: Metal temperature below acid dew point. Increase feedwater temperature or reduce load.")
        elif margin < safety_margin_f * 0.5:
            return ("high", "Increase feedwater temperature by 20-30F to restore safety margin.")
        elif margin < safety_margin_f:
            return ("moderate", "Monitor closely. Consider increasing feedwater temperature by 10-15F.")
        else:
            return ("low", "Operating with adequate safety margin.")

    def calculate(self, input_data: AcidDewPointInput) -> dict:
        """
        Perform complete acid dew point analysis.

        Args:
            input_data: AcidDewPointInput data

        Returns:
            Dictionary with calculation results
        """
        # Calculate excess air
        excess_air_pct = self.calculate_excess_air(input_data.flue_gas_o2_pct)

        # Calculate SO3 concentration
        if input_data.flue_gas_so2_ppm is not None:
            # Direct SO2 measurement available
            so3_ppm = input_data.flue_gas_so2_ppm * (input_data.so2_to_so3_conversion_pct / 100)
        else:
            # Calculate from fuel sulfur
            so3_ppm = self.calculate_so3_from_fuel(
                input_data.fuel_sulfur_pct,
                input_data.so2_to_so3_conversion_pct,
                excess_air_pct,
            )

        # Calculate dew points
        water_dew_point_f = self.calculate_water_dew_point(input_data.flue_gas_moisture_pct)
        acid_dew_point_f = self.calculate_acid_dew_point_verhoff_banchero(
            input_data.flue_gas_moisture_pct,
            so3_ppm,
        )

        # Effective dew point is the higher of the two
        effective_dew_point_f = max(water_dew_point_f, acid_dew_point_f)

        # Calculate margins
        margin_above_acid_dp_f = input_data.cold_end_metal_temp_f - acid_dew_point_f
        margin_above_water_dp_f = input_data.cold_end_metal_temp_f - water_dew_point_f

        # Assess risk
        risk_level, recommended_action = self.assess_corrosion_risk(
            input_data.cold_end_metal_temp_f,
            acid_dew_point_f,
            input_data.safety_margin_f,
        )

        # Calculate minimum recommended metal temperature
        min_recommended_temp_f = acid_dew_point_f + input_data.safety_margin_f

        # Determine if action is required
        action_required = margin_above_acid_dp_f < input_data.safety_margin_f

        # Feedwater temperature adjustment if needed
        feedwater_temp_adjustment_f = None
        if action_required:
            feedwater_temp_adjustment_f = input_data.safety_margin_f - margin_above_acid_dp_f

        # Build result
        result = {
            "sulfuric_acid_dew_point_f": round(acid_dew_point_f, 1),
            "water_dew_point_f": round(water_dew_point_f, 1),
            "effective_dew_point_f": round(effective_dew_point_f, 1),
            "min_metal_temp_f": round(input_data.cold_end_metal_temp_f, 1),
            "avg_metal_temp_f": round(input_data.cold_end_metal_temp_f, 1),
            "margin_above_dew_point_f": round(margin_above_acid_dp_f, 1),
            "corrosion_risk": risk_level,
            "below_dew_point": margin_above_acid_dp_f < 0,
            "margin_adequate": margin_above_acid_dp_f >= input_data.safety_margin_f,
            "so3_concentration_ppm": round(so3_ppm, 2),
            "h2o_concentration_pct": round(input_data.flue_gas_moisture_pct, 1),
            "excess_air_pct": round(excess_air_pct, 1),
            "min_recommended_metal_temp_f": round(min_recommended_temp_f, 1),
            "feedwater_temp_adjustment_f": round(feedwater_temp_adjustment_f, 1) if feedwater_temp_adjustment_f else None,
            "action_required": action_required,
            "recommended_action": recommended_action if action_required else None,
            "calculation_method": "VERHOFF_BANCHERO",
            "formula_reference": "Verhoff & Banchero, Chemical Engineering Progress, 1974",
        }

        # Add provenance hash
        result["provenance_hash"] = hashlib.sha256(
            json.dumps(result, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        return result


def create_acid_dew_point_calculator(safety_margin_f: float = 30.0) -> AcidDewPointCalculator:
    """Factory function to create AcidDewPointCalculator."""
    return AcidDewPointCalculator(safety_margin_f=safety_margin_f)
