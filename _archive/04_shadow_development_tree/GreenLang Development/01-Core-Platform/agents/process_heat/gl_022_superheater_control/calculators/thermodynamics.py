"""
GL-022 SUPERHEATER CONTROL - Thermodynamics Calculator Module

This module provides steam property calculations using IAPWS-IF97 principles
for superheater control applications including:
- Steam property calculations (enthalpy, entropy, specific heat)
- Superheater heat transfer calculations
- Enthalpy balance for spray desuperheating
- Temperature-enthalpy relationships
- Thermal stress calculations

All calculations are ZERO-HALLUCINATION deterministic with complete provenance tracking.

Standards Reference:
    - IAPWS-IF97: Industrial Formulation 1997 for Water and Steam
    - ASME PTC 4.2: Steam Generating Units
    - ASME Section I: Rules for Construction of Power Boilers

Example:
    >>> from greenlang.agents.process_heat.gl_022_superheater_control.calculators.thermodynamics import (
    ...     SteamThermodynamicsCalculator,
    ...     SuperheaterHeatTransferCalculator,
    ... )
    >>>
    >>> thermo = SteamThermodynamicsCalculator()
    >>> enthalpy = thermo.calculate_superheated_enthalpy(500.0, 750.0)
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - IAPWS-IF97 REFERENCE DATA
# =============================================================================

class IAPWSIF97Constants:
    """
    Constants from IAPWS-IF97 formulation for water and steam properties.

    These are simplified correlations suitable for industrial calculations
    in the superheater operating range (typically 300-2500 psig, 500-1100 F).
    """

    # Critical point properties
    CRITICAL_PRESSURE_PSIA = 3208.2
    CRITICAL_PRESSURE_PSIG = 3193.5
    CRITICAL_TEMPERATURE_F = 705.47
    CRITICAL_TEMPERATURE_R = 1165.14

    # Reference state (IAPWS: triple point of water)
    REFERENCE_TEMP_F = 32.018
    REFERENCE_ENTHALPY_BTU_LB = 0.0
    REFERENCE_ENTROPY_BTU_LB_R = 0.0

    # Universal gas constant
    R_STEAM_BTU_LB_R = 0.1102  # BTU/(lb-R) for steam

    # Molecular weight of water
    MW_WATER = 18.015  # g/mol

    # Saturation table (psig: (temp_f, h_f, h_fg, h_g, s_f, s_fg, s_g, v_g))
    # v_g = specific volume of saturated vapor (ft3/lb)
    SATURATION_DATA = {
        0: (212.0, 180.2, 970.3, 1150.5, 0.3121, 1.4447, 1.7568, 26.80),
        15: (250.3, 218.9, 945.4, 1164.3, 0.3680, 1.3607, 1.7287, 16.30),
        50: (298.0, 267.6, 911.0, 1178.6, 0.4295, 1.2625, 1.6920, 8.52),
        100: (337.9, 309.0, 879.5, 1188.5, 0.4832, 1.1781, 1.6613, 4.43),
        150: (365.9, 339.2, 856.8, 1196.0, 0.5208, 1.1167, 1.6375, 3.01),
        200: (387.9, 362.2, 837.4, 1199.6, 0.5500, 1.0685, 1.6185, 2.29),
        250: (406.1, 381.2, 820.1, 1201.3, 0.5736, 1.0282, 1.6018, 1.84),
        300: (421.7, 397.0, 804.3, 1201.3, 0.5932, 0.9930, 1.5862, 1.54),
        400: (448.0, 424.2, 774.4, 1198.6, 0.6261, 0.9318, 1.5579, 1.16),
        500: (470.0, 447.7, 747.1, 1194.8, 0.6531, 0.8797, 1.5328, 0.93),
        600: (489.0, 468.4, 721.4, 1189.8, 0.6763, 0.8335, 1.5098, 0.77),
        700: (506.2, 486.9, 696.8, 1183.7, 0.6967, 0.7917, 1.4884, 0.66),
        800: (521.8, 503.6, 673.2, 1176.8, 0.7149, 0.7534, 1.4683, 0.57),
        900: (536.0, 519.0, 650.4, 1169.4, 0.7314, 0.7180, 1.4494, 0.50),
        1000: (548.6, 533.1, 628.2, 1161.3, 0.7467, 0.6849, 1.4316, 0.45),
        1200: (571.7, 558.4, 585.4, 1143.8, 0.7737, 0.6239, 1.3976, 0.36),
        1500: (600.3, 593.2, 527.0, 1120.2, 0.8082, 0.5446, 1.3528, 0.28),
        2000: (636.0, 639.1, 444.0, 1083.1, 0.8549, 0.4410, 1.2959, 0.19),
    }

    # Superheated steam specific heat correlations (Cp in BTU/lb-F)
    # Cp varies with temperature and pressure
    # Approximate formula: Cp = Cp0 + a*T + b*P
    CP_SUPERHEATED_BASE = 0.48  # BTU/lb-F at moderate conditions
    CP_TEMPERATURE_COEFF = 0.00002  # BTU/lb-F^2
    CP_PRESSURE_COEFF = 0.00001  # BTU/lb-F-psi


class ThermalStressConstants:
    """Constants for thermal stress calculations in superheater tubes."""

    # Material properties for typical superheater tube materials
    # SA-213 T22 (2-1/4 Cr - 1 Mo)
    YOUNGS_MODULUS_T22_PSI = 27.0e6  # Young's modulus at 900F
    THERMAL_EXPANSION_T22 = 7.9e-6  # in/in-F at 900F
    POISSON_RATIO_T22 = 0.30

    # SA-213 T91 (9 Cr - 1 Mo - V)
    YOUNGS_MODULUS_T91_PSI = 26.0e6  # Young's modulus at 1000F
    THERMAL_EXPANSION_T91 = 7.0e-6  # in/in-F at 1000F
    POISSON_RATIO_T91 = 0.30

    # Allowable thermal stress rates (F/min) for superheater tubes
    MAX_TEMP_RATE_NORMAL = 5.0  # F/min for normal operation
    MAX_TEMP_RATE_STARTUP = 15.0  # F/min during startup
    MAX_TEMP_RATE_EMERGENCY = 50.0  # F/min emergency only


# =============================================================================
# DATA CLASSES FOR CALCULATION RESULTS
# =============================================================================

@dataclass
class SteamPropertiesResult:
    """Result of steam property calculation."""
    pressure_psig: float
    temperature_f: float
    enthalpy_btu_lb: float
    entropy_btu_lb_r: float
    specific_heat_cp_btu_lb_f: float
    specific_volume_ft3_lb: float
    phase: str  # "superheated", "saturated_vapor", "wet_steam"
    superheat_f: Optional[float]
    saturation_temp_f: float
    calculation_method: str
    provenance_hash: str


@dataclass
class HeatTransferResult:
    """Result of heat transfer calculation."""
    heat_duty_btu_hr: float
    heat_duty_kw: float
    lmtd_f: float
    ua_btu_hr_f: float
    overall_htc_btu_hr_ft2_f: float
    effectiveness: float
    gas_temp_drop_f: float
    steam_temp_rise_f: float
    calculation_method: str
    provenance_hash: str


@dataclass
class EnthalpyBalanceResult:
    """Result of enthalpy balance calculation."""
    inlet_enthalpy_btu_lb: float
    outlet_enthalpy_btu_lb: float
    spray_water_enthalpy_btu_lb: float
    heat_absorbed_btu_hr: float
    spray_flow_required_lb_hr: float
    outlet_temperature_f: float
    calculation_method: str
    provenance_hash: str


@dataclass
class ThermalStressResult:
    """Result of thermal stress calculation."""
    thermal_stress_psi: float
    temperature_gradient_f: float
    temperature_rate_f_min: float
    stress_ratio: float  # actual/allowable
    is_within_limits: bool
    material: str
    tube_thickness_in: float
    calculation_method: str
    provenance_hash: str


# =============================================================================
# STEAM THERMODYNAMICS CALCULATOR
# =============================================================================

class SteamThermodynamicsCalculator:
    """
    Calculator for steam thermodynamic properties using IAPWS-IF97 principles.

    This class provides ZERO-HALLUCINATION deterministic calculations for:
    - Saturated steam properties (temperature, enthalpy, entropy)
    - Superheated steam properties
    - Specific heat at constant pressure
    - Temperature-enthalpy relationships

    All calculations include provenance tracking with SHA-256 hashes.

    Example:
        >>> calc = SteamThermodynamicsCalculator()
        >>> result = calc.get_steam_properties(500.0, 800.0)
        >>> print(f"Enthalpy: {result.enthalpy_btu_lb:.1f} BTU/lb")
        >>> print(f"Provenance: {result.provenance_hash}")
    """

    def __init__(self) -> None:
        """Initialize steam thermodynamics calculator."""
        self._calculation_count = 0
        logger.debug("SteamThermodynamicsCalculator initialized")

    def get_saturation_properties(
        self,
        pressure_psig: float,
    ) -> Dict[str, float]:
        """
        Get saturation properties at given pressure - DETERMINISTIC.

        Uses linear interpolation of IAPWS-IF97 tabulated data.

        Args:
            pressure_psig: Gauge pressure (psig)

        Returns:
            Dictionary with saturation temperature, enthalpies, entropies, volume

        Formula Reference:
            Linear interpolation of IAPWS-IF97 saturation tables
        """
        # Clamp to valid range
        pressure_psig = max(0, min(2000, pressure_psig))

        # Get bracketing pressures from table
        pressures = sorted(IAPWSIF97Constants.SATURATION_DATA.keys())
        p_low = pressures[0]
        p_high = pressures[-1]

        for i in range(len(pressures) - 1):
            if pressures[i] <= pressure_psig <= pressures[i + 1]:
                p_low = pressures[i]
                p_high = pressures[i + 1]
                break
        else:
            # Pressure above table range - extrapolate from last two points
            if pressure_psig > pressures[-1]:
                p_low = pressures[-2]
                p_high = pressures[-1]

        # Get data points
        data_low = IAPWSIF97Constants.SATURATION_DATA[p_low]
        data_high = IAPWSIF97Constants.SATURATION_DATA[p_high]

        # Linear interpolation factor
        if p_high > p_low:
            factor = (pressure_psig - p_low) / (p_high - p_low)
        else:
            factor = 0.0

        # Interpolate all properties
        def interp(idx: int) -> float:
            return data_low[idx] + factor * (data_high[idx] - data_low[idx])

        return {
            "saturation_temp_f": interp(0),
            "h_f_btu_lb": interp(1),       # Saturated liquid enthalpy
            "h_fg_btu_lb": interp(2),      # Latent heat of vaporization
            "h_g_btu_lb": interp(3),       # Saturated vapor enthalpy
            "s_f_btu_lb_r": interp(4),     # Saturated liquid entropy
            "s_fg_btu_lb_r": interp(5),    # Entropy of vaporization
            "s_g_btu_lb_r": interp(6),     # Saturated vapor entropy
            "v_g_ft3_lb": interp(7),       # Saturated vapor specific volume
        }

    def calculate_superheated_enthalpy(
        self,
        pressure_psig: float,
        temperature_f: float,
    ) -> float:
        """
        Calculate superheated steam enthalpy - DETERMINISTIC.

        Uses the formula:
            h = h_g + Cp * (T - T_sat)

        where Cp is temperature and pressure dependent.

        Args:
            pressure_psig: Gauge pressure (psig)
            temperature_f: Steam temperature (F)

        Returns:
            Specific enthalpy (BTU/lb)

        Formula Reference:
            h = h_g(P) + integral(Cp(T,P) dT) from T_sat to T
            Simplified: h = h_g + Cp_avg * (T - T_sat)
        """
        sat_props = self.get_saturation_properties(pressure_psig)
        t_sat = sat_props["saturation_temp_f"]
        h_g = sat_props["h_g_btu_lb"]

        if temperature_f <= t_sat:
            # At or below saturation - return saturated vapor enthalpy
            return h_g

        # Calculate superheat
        superheat = temperature_f - t_sat

        # Calculate average Cp over the superheat range
        cp_avg = self.calculate_cp_superheated(
            pressure_psig,
            (temperature_f + t_sat) / 2  # Average temperature
        )

        # Enthalpy of superheated steam
        enthalpy = h_g + cp_avg * superheat

        return round(enthalpy, 2)

    def calculate_superheated_entropy(
        self,
        pressure_psig: float,
        temperature_f: float,
    ) -> float:
        """
        Calculate superheated steam entropy - DETERMINISTIC.

        Uses the formula:
            s = s_g + Cp * ln(T/T_sat)

        where temperatures are in Rankine.

        Args:
            pressure_psig: Gauge pressure (psig)
            temperature_f: Steam temperature (F)

        Returns:
            Specific entropy (BTU/lb-R)

        Formula Reference:
            ds = Cp * dT/T (for ideal gas at constant pressure)
            s = s_g + Cp * ln(T/T_sat) where T in absolute units
        """
        sat_props = self.get_saturation_properties(pressure_psig)
        t_sat = sat_props["saturation_temp_f"]
        s_g = sat_props["s_g_btu_lb_r"]

        if temperature_f <= t_sat:
            return s_g

        # Convert to Rankine
        t_sat_r = t_sat + 459.67
        t_r = temperature_f + 459.67

        # Calculate average Cp
        cp_avg = self.calculate_cp_superheated(
            pressure_psig,
            (temperature_f + t_sat) / 2
        )

        # Entropy of superheated steam
        entropy = s_g + cp_avg * math.log(t_r / t_sat_r)

        return round(entropy, 4)

    def calculate_cp_superheated(
        self,
        pressure_psig: float,
        temperature_f: float,
    ) -> float:
        """
        Calculate specific heat at constant pressure for superheated steam - DETERMINISTIC.

        Cp varies with temperature and pressure. This correlation is based on
        regression of IAPWS-IF97 data in the superheater operating range.

        Args:
            pressure_psig: Gauge pressure (psig)
            temperature_f: Steam temperature (F)

        Returns:
            Specific heat Cp (BTU/lb-F)

        Formula Reference:
            Cp(T,P) = Cp0 + a*(T-Tref) + b*(P-Pref)
            where Tref=500F, Pref=500psig for this correlation
        """
        # Base specific heat at reference conditions
        cp_base = IAPWSIF97Constants.CP_SUPERHEATED_BASE

        # Temperature correction (Cp increases with temperature)
        temp_correction = IAPWSIF97Constants.CP_TEMPERATURE_COEFF * (temperature_f - 500.0)

        # Pressure correction (Cp increases with pressure near saturation)
        pressure_psia = pressure_psig + 14.696
        critical_proximity = 1.0 - (pressure_psia / IAPWSIF97Constants.CRITICAL_PRESSURE_PSIA)
        pressure_correction = IAPWSIF97Constants.CP_PRESSURE_COEFF * pressure_psig * (1.0 / max(critical_proximity, 0.1))

        cp = cp_base + temp_correction + pressure_correction

        # Clamp to reasonable range (0.4 to 1.5 BTU/lb-F for steam)
        cp = max(0.40, min(1.50, cp))

        return round(cp, 4)

    def calculate_specific_volume(
        self,
        pressure_psig: float,
        temperature_f: float,
    ) -> float:
        """
        Calculate specific volume of superheated steam - DETERMINISTIC.

        Uses ideal gas approximation with compressibility correction.

        Args:
            pressure_psig: Gauge pressure (psig)
            temperature_f: Steam temperature (F)

        Returns:
            Specific volume (ft3/lb)

        Formula Reference:
            v = Z * R * T / P  (ideal gas with compressibility)
            where Z ~ 1.0 for superheated steam far from saturation
        """
        sat_props = self.get_saturation_properties(pressure_psig)
        t_sat = sat_props["saturation_temp_f"]
        v_g = sat_props["v_g_ft3_lb"]

        if temperature_f <= t_sat:
            return v_g

        # Convert to absolute units
        pressure_psia = pressure_psig + 14.696
        temp_r = temperature_f + 459.67
        t_sat_r = t_sat + 459.67

        # Ideal gas: v proportional to T/P
        # Use saturated vapor as reference
        v_superheated = v_g * (temp_r / t_sat_r)

        # Apply compressibility correction near critical point
        reduced_pressure = pressure_psia / IAPWSIF97Constants.CRITICAL_PRESSURE_PSIA
        if reduced_pressure > 0.5:
            # Near critical - reduce specific volume
            z_correction = 1.0 - 0.1 * (reduced_pressure - 0.5)
            v_superheated *= z_correction

        return round(v_superheated, 4)

    def calculate_temperature_from_enthalpy(
        self,
        pressure_psig: float,
        enthalpy_btu_lb: float,
    ) -> float:
        """
        Calculate temperature from enthalpy at given pressure - DETERMINISTIC.

        Inverse of enthalpy calculation using Newton-Raphson iteration.

        Args:
            pressure_psig: Gauge pressure (psig)
            enthalpy_btu_lb: Specific enthalpy (BTU/lb)

        Returns:
            Steam temperature (F)

        Formula Reference:
            Solve h(T) = h_target iteratively
            T = T_sat + (h - h_g) / Cp
        """
        sat_props = self.get_saturation_properties(pressure_psig)
        t_sat = sat_props["saturation_temp_f"]
        h_g = sat_props["h_g_btu_lb"]
        h_f = sat_props["h_f_btu_lb"]

        # Check if below saturation
        if enthalpy_btu_lb <= h_f:
            # Subcooled liquid - approximate
            return 32.0 + enthalpy_btu_lb  # Cp ~ 1 for liquid water

        if enthalpy_btu_lb <= h_g:
            # Two-phase region - return saturation temperature
            return t_sat

        # Superheated steam - iterate to find temperature
        # Initial guess using base Cp
        cp_initial = IAPWSIF97Constants.CP_SUPERHEATED_BASE
        temp_guess = t_sat + (enthalpy_btu_lb - h_g) / cp_initial

        # Newton-Raphson iteration (max 10 iterations)
        for _ in range(10):
            h_calc = self.calculate_superheated_enthalpy(pressure_psig, temp_guess)
            error = h_calc - enthalpy_btu_lb

            if abs(error) < 0.1:  # Converged within 0.1 BTU/lb
                break

            cp_current = self.calculate_cp_superheated(pressure_psig, temp_guess)
            temp_guess -= error / cp_current

        return round(temp_guess, 1)

    def get_steam_properties(
        self,
        pressure_psig: float,
        temperature_f: float,
    ) -> SteamPropertiesResult:
        """
        Get complete steam properties at given conditions - DETERMINISTIC.

        Args:
            pressure_psig: Gauge pressure (psig)
            temperature_f: Steam temperature (F)

        Returns:
            SteamPropertiesResult with all thermodynamic properties
        """
        self._calculation_count += 1

        sat_props = self.get_saturation_properties(pressure_psig)
        t_sat = sat_props["saturation_temp_f"]

        # Determine phase
        if temperature_f < t_sat - 1.0:
            phase = "subcooled_liquid"
            superheat = None
        elif abs(temperature_f - t_sat) <= 1.0:
            phase = "saturated_vapor"
            superheat = 0.0
        else:
            phase = "superheated"
            superheat = temperature_f - t_sat

        # Calculate properties
        enthalpy = self.calculate_superheated_enthalpy(pressure_psig, temperature_f)
        entropy = self.calculate_superheated_entropy(pressure_psig, temperature_f)
        cp = self.calculate_cp_superheated(pressure_psig, temperature_f)
        specific_volume = self.calculate_specific_volume(pressure_psig, temperature_f)

        # Calculate provenance hash
        provenance_data = {
            "pressure_psig": pressure_psig,
            "temperature_f": temperature_f,
            "enthalpy_btu_lb": enthalpy,
            "entropy_btu_lb_r": entropy,
            "cp_btu_lb_f": cp,
            "specific_volume_ft3_lb": specific_volume,
            "calculation_method": "IAPWS-IF97_interpolation",
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return SteamPropertiesResult(
            pressure_psig=pressure_psig,
            temperature_f=temperature_f,
            enthalpy_btu_lb=enthalpy,
            entropy_btu_lb_r=entropy,
            specific_heat_cp_btu_lb_f=cp,
            specific_volume_ft3_lb=specific_volume,
            phase=phase,
            superheat_f=superheat,
            saturation_temp_f=t_sat,
            calculation_method="IAPWS-IF97_interpolation",
            provenance_hash=provenance_hash,
        )

    def calculate_water_enthalpy(
        self,
        temperature_f: float,
        pressure_psig: Optional[float] = None,
    ) -> float:
        """
        Calculate liquid water enthalpy - DETERMINISTIC.

        For compressed liquid, enthalpy is approximately the same as
        saturated liquid at the same temperature.

        Args:
            temperature_f: Water temperature (F)
            pressure_psig: Pressure (optional, for subcooling check)

        Returns:
            Specific enthalpy (BTU/lb)

        Formula Reference:
            h_f ~ Cp_water * (T - 32F) where Cp_water ~ 1.0 BTU/lb-F
        """
        # Liquid water enthalpy relative to 32F reference
        # Cp of liquid water is approximately 1.0 BTU/lb-F
        enthalpy = 1.0 * (temperature_f - 32.0)

        # Clamp to reasonable values
        enthalpy = max(0.0, min(700.0, enthalpy))

        return round(enthalpy, 2)


# =============================================================================
# SUPERHEATER HEAT TRANSFER CALCULATOR
# =============================================================================

class SuperheaterHeatTransferCalculator:
    """
    Calculator for superheater heat transfer calculations.

    Implements heat transfer correlations for superheater design and
    performance analysis including:
    - Heat duty calculations
    - LMTD (Log Mean Temperature Difference) method
    - Overall heat transfer coefficient estimation
    - Effectiveness-NTU method

    Formula Reference:
        Q = U * A * LMTD
        Q = m * Cp * delta_T
        LMTD = (delta_T1 - delta_T2) / ln(delta_T1 / delta_T2)

    Example:
        >>> calc = SuperheaterHeatTransferCalculator()
        >>> result = calc.calculate_heat_duty(
        ...     steam_flow_lb_hr=100000,
        ...     inlet_temp_f=700.0,
        ...     outlet_temp_f=950.0,
        ...     pressure_psig=600.0,
        ... )
    """

    def __init__(self) -> None:
        """Initialize heat transfer calculator."""
        self.thermo_calc = SteamThermodynamicsCalculator()
        logger.debug("SuperheaterHeatTransferCalculator initialized")

    def calculate_heat_duty(
        self,
        steam_flow_lb_hr: float,
        inlet_temp_f: float,
        outlet_temp_f: float,
        pressure_psig: float,
    ) -> float:
        """
        Calculate superheater heat duty from temperature rise - DETERMINISTIC.

        Formula:
            Q = m_dot * (h_out - h_in)
            or
            Q = m_dot * Cp * (T_out - T_in)

        Args:
            steam_flow_lb_hr: Steam mass flow rate (lb/hr)
            inlet_temp_f: Steam inlet temperature (F)
            outlet_temp_f: Steam outlet temperature (F)
            pressure_psig: Steam pressure (psig)

        Returns:
            Heat duty (BTU/hr)
        """
        # Get enthalpies at inlet and outlet
        h_in = self.thermo_calc.calculate_superheated_enthalpy(pressure_psig, inlet_temp_f)
        h_out = self.thermo_calc.calculate_superheated_enthalpy(pressure_psig, outlet_temp_f)

        # Heat duty = mass flow * enthalpy change
        heat_duty = steam_flow_lb_hr * (h_out - h_in)

        return round(heat_duty, 0)

    def calculate_lmtd(
        self,
        gas_inlet_temp_f: float,
        gas_outlet_temp_f: float,
        steam_inlet_temp_f: float,
        steam_outlet_temp_f: float,
        flow_arrangement: str = "counterflow",
    ) -> float:
        """
        Calculate Log Mean Temperature Difference (LMTD) - DETERMINISTIC.

        Formula (counterflow):
            delta_T1 = T_gas_in - T_steam_out
            delta_T2 = T_gas_out - T_steam_in
            LMTD = (delta_T1 - delta_T2) / ln(delta_T1 / delta_T2)

        Args:
            gas_inlet_temp_f: Hot gas inlet temperature (F)
            gas_outlet_temp_f: Hot gas outlet temperature (F)
            steam_inlet_temp_f: Steam inlet temperature (F)
            steam_outlet_temp_f: Steam outlet temperature (F)
            flow_arrangement: "counterflow" or "parallel"

        Returns:
            LMTD (F)
        """
        if flow_arrangement == "counterflow":
            delta_t1 = gas_inlet_temp_f - steam_outlet_temp_f
            delta_t2 = gas_outlet_temp_f - steam_inlet_temp_f
        else:  # parallel flow
            delta_t1 = gas_inlet_temp_f - steam_inlet_temp_f
            delta_t2 = gas_outlet_temp_f - steam_outlet_temp_f

        # Handle edge cases
        if delta_t1 <= 0 or delta_t2 <= 0:
            logger.warning("Invalid temperature approach - LMTD calculation error")
            return 1.0  # Minimum value to avoid errors

        if abs(delta_t1 - delta_t2) < 0.1:
            # When delta_t1 approximately equals delta_t2, LMTD = delta_t1
            return round(delta_t1, 1)

        try:
            lmtd = (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)
            return round(max(lmtd, 1.0), 1)
        except (ValueError, ZeroDivisionError):
            # Fallback to arithmetic mean
            return round((delta_t1 + delta_t2) / 2, 1)

    def calculate_ua_from_duty_lmtd(
        self,
        heat_duty_btu_hr: float,
        lmtd_f: float,
    ) -> float:
        """
        Calculate overall heat transfer coefficient-area product - DETERMINISTIC.

        Formula:
            UA = Q / LMTD

        Args:
            heat_duty_btu_hr: Heat duty (BTU/hr)
            lmtd_f: Log mean temperature difference (F)

        Returns:
            UA value (BTU/hr-F)
        """
        if lmtd_f <= 0:
            return 0.0

        ua = heat_duty_btu_hr / lmtd_f
        return round(ua, 0)

    def calculate_overall_htc(
        self,
        ua_btu_hr_f: float,
        heat_transfer_area_ft2: float,
    ) -> float:
        """
        Calculate overall heat transfer coefficient - DETERMINISTIC.

        Formula:
            U = UA / A

        Args:
            ua_btu_hr_f: UA value (BTU/hr-F)
            heat_transfer_area_ft2: Heat transfer surface area (ft2)

        Returns:
            Overall heat transfer coefficient U (BTU/hr-ft2-F)
        """
        if heat_transfer_area_ft2 <= 0:
            return 0.0

        u = ua_btu_hr_f / heat_transfer_area_ft2
        return round(u, 2)

    def calculate_effectiveness(
        self,
        steam_flow_lb_hr: float,
        steam_inlet_temp_f: float,
        steam_outlet_temp_f: float,
        gas_inlet_temp_f: float,
        pressure_psig: float,
    ) -> float:
        """
        Calculate heat exchanger effectiveness - DETERMINISTIC.

        Formula:
            epsilon = Q_actual / Q_max
            Q_max = C_min * (T_gas_in - T_steam_in)

        Args:
            steam_flow_lb_hr: Steam mass flow rate (lb/hr)
            steam_inlet_temp_f: Steam inlet temperature (F)
            steam_outlet_temp_f: Steam outlet temperature (F)
            gas_inlet_temp_f: Hot gas inlet temperature (F)
            pressure_psig: Steam pressure (psig)

        Returns:
            Effectiveness (0-1)
        """
        # Actual temperature rise
        actual_rise = steam_outlet_temp_f - steam_inlet_temp_f

        # Maximum possible temperature rise
        max_rise = gas_inlet_temp_f - steam_inlet_temp_f

        if max_rise <= 0:
            return 0.0

        effectiveness = actual_rise / max_rise
        return round(max(0.0, min(1.0, effectiveness)), 4)

    def calculate_heat_transfer(
        self,
        steam_flow_lb_hr: float,
        steam_inlet_temp_f: float,
        steam_outlet_temp_f: float,
        gas_inlet_temp_f: float,
        gas_outlet_temp_f: float,
        pressure_psig: float,
        heat_transfer_area_ft2: float,
        flow_arrangement: str = "counterflow",
    ) -> HeatTransferResult:
        """
        Perform complete heat transfer analysis - DETERMINISTIC.

        Args:
            steam_flow_lb_hr: Steam mass flow rate (lb/hr)
            steam_inlet_temp_f: Steam inlet temperature (F)
            steam_outlet_temp_f: Steam outlet temperature (F)
            gas_inlet_temp_f: Hot gas inlet temperature (F)
            gas_outlet_temp_f: Hot gas outlet temperature (F)
            pressure_psig: Steam pressure (psig)
            heat_transfer_area_ft2: Heat transfer surface area (ft2)
            flow_arrangement: Flow arrangement

        Returns:
            HeatTransferResult with complete analysis
        """
        # Calculate heat duty
        heat_duty_btu_hr = self.calculate_heat_duty(
            steam_flow_lb_hr,
            steam_inlet_temp_f,
            steam_outlet_temp_f,
            pressure_psig,
        )

        # Convert to kW
        heat_duty_kw = heat_duty_btu_hr * 0.000293071

        # Calculate LMTD
        lmtd = self.calculate_lmtd(
            gas_inlet_temp_f,
            gas_outlet_temp_f,
            steam_inlet_temp_f,
            steam_outlet_temp_f,
            flow_arrangement,
        )

        # Calculate UA
        ua = self.calculate_ua_from_duty_lmtd(heat_duty_btu_hr, lmtd)

        # Calculate overall HTC
        htc = self.calculate_overall_htc(ua, heat_transfer_area_ft2)

        # Calculate effectiveness
        effectiveness = self.calculate_effectiveness(
            steam_flow_lb_hr,
            steam_inlet_temp_f,
            steam_outlet_temp_f,
            gas_inlet_temp_f,
            pressure_psig,
        )

        # Temperature changes
        gas_temp_drop = gas_inlet_temp_f - gas_outlet_temp_f
        steam_temp_rise = steam_outlet_temp_f - steam_inlet_temp_f

        # Provenance hash
        provenance_data = {
            "steam_flow_lb_hr": steam_flow_lb_hr,
            "steam_inlet_temp_f": steam_inlet_temp_f,
            "steam_outlet_temp_f": steam_outlet_temp_f,
            "gas_inlet_temp_f": gas_inlet_temp_f,
            "gas_outlet_temp_f": gas_outlet_temp_f,
            "pressure_psig": pressure_psig,
            "heat_transfer_area_ft2": heat_transfer_area_ft2,
            "heat_duty_btu_hr": heat_duty_btu_hr,
            "lmtd_f": lmtd,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return HeatTransferResult(
            heat_duty_btu_hr=heat_duty_btu_hr,
            heat_duty_kw=round(heat_duty_kw, 1),
            lmtd_f=lmtd,
            ua_btu_hr_f=ua,
            overall_htc_btu_hr_ft2_f=htc,
            effectiveness=effectiveness,
            gas_temp_drop_f=round(gas_temp_drop, 1),
            steam_temp_rise_f=round(steam_temp_rise, 1),
            calculation_method="LMTD_method",
            provenance_hash=provenance_hash,
        )


# =============================================================================
# ENTHALPY BALANCE CALCULATOR
# =============================================================================

class EnthalpyBalanceCalculator:
    """
    Calculator for enthalpy balance in spray desuperheating.

    Implements the fundamental enthalpy balance equation for
    mixing superheated steam with spray water.

    Formula:
        m_steam * h_in + m_spray * h_water = (m_steam + m_spray) * h_out

        Solving for spray flow:
        m_spray = m_steam * (h_in - h_out) / (h_out - h_water)

    Example:
        >>> calc = EnthalpyBalanceCalculator()
        >>> result = calc.calculate_spray_flow_required(
        ...     steam_flow_lb_hr=100000,
        ...     inlet_temp_f=950.0,
        ...     target_temp_f=850.0,
        ...     spray_water_temp_f=250.0,
        ...     pressure_psig=600.0,
        ... )
    """

    def __init__(self) -> None:
        """Initialize enthalpy balance calculator."""
        self.thermo_calc = SteamThermodynamicsCalculator()
        logger.debug("EnthalpyBalanceCalculator initialized")

    def calculate_spray_flow_required(
        self,
        steam_flow_lb_hr: float,
        inlet_temp_f: float,
        target_temp_f: float,
        spray_water_temp_f: float,
        pressure_psig: float,
    ) -> EnthalpyBalanceResult:
        """
        Calculate required spray water flow for temperature control - DETERMINISTIC.

        Formula:
            m_spray = m_steam * (h_in - h_out) / (h_out - h_water)

        This is the fundamental spray desuperheating equation derived from
        steady-state enthalpy balance.

        Args:
            steam_flow_lb_hr: Inlet steam mass flow rate (lb/hr)
            inlet_temp_f: Steam inlet temperature (F)
            target_temp_f: Target outlet temperature (F)
            spray_water_temp_f: Spray water temperature (F)
            pressure_psig: System pressure (psig)

        Returns:
            EnthalpyBalanceResult with spray flow and complete analysis
        """
        # Calculate enthalpies
        h_in = self.thermo_calc.calculate_superheated_enthalpy(pressure_psig, inlet_temp_f)
        h_out = self.thermo_calc.calculate_superheated_enthalpy(pressure_psig, target_temp_f)
        h_water = self.thermo_calc.calculate_water_enthalpy(spray_water_temp_f)

        # Check for valid calculation
        if h_out <= h_water:
            logger.warning(
                f"Invalid conditions: outlet enthalpy ({h_out:.1f}) <= "
                f"water enthalpy ({h_water:.1f})"
            )
            spray_flow = 0.0
        elif inlet_temp_f <= target_temp_f:
            # No cooling needed
            spray_flow = 0.0
        else:
            # Calculate spray flow using enthalpy balance
            spray_flow = steam_flow_lb_hr * (h_in - h_out) / (h_out - h_water)

        # Calculate heat removed by spray
        if spray_flow > 0:
            heat_absorbed = spray_flow * (h_out - h_water)
        else:
            heat_absorbed = 0.0

        # Provenance hash
        provenance_data = {
            "steam_flow_lb_hr": steam_flow_lb_hr,
            "inlet_temp_f": inlet_temp_f,
            "target_temp_f": target_temp_f,
            "spray_water_temp_f": spray_water_temp_f,
            "pressure_psig": pressure_psig,
            "h_in_btu_lb": h_in,
            "h_out_btu_lb": h_out,
            "h_water_btu_lb": h_water,
            "spray_flow_lb_hr": spray_flow,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return EnthalpyBalanceResult(
            inlet_enthalpy_btu_lb=h_in,
            outlet_enthalpy_btu_lb=h_out,
            spray_water_enthalpy_btu_lb=h_water,
            heat_absorbed_btu_hr=round(heat_absorbed, 0),
            spray_flow_required_lb_hr=round(spray_flow, 0),
            outlet_temperature_f=target_temp_f,
            calculation_method="enthalpy_balance",
            provenance_hash=provenance_hash,
        )

    def calculate_outlet_temperature(
        self,
        steam_flow_lb_hr: float,
        inlet_temp_f: float,
        spray_flow_lb_hr: float,
        spray_water_temp_f: float,
        pressure_psig: float,
    ) -> float:
        """
        Calculate outlet temperature for given spray flow - DETERMINISTIC.

        Inverse of spray flow calculation - determines resulting temperature
        when spray flow is known.

        Formula:
            h_out = (m_steam * h_in + m_spray * h_water) / (m_steam + m_spray)
            Then solve for T_out from h_out

        Args:
            steam_flow_lb_hr: Inlet steam mass flow rate (lb/hr)
            inlet_temp_f: Steam inlet temperature (F)
            spray_flow_lb_hr: Spray water flow rate (lb/hr)
            spray_water_temp_f: Spray water temperature (F)
            pressure_psig: System pressure (psig)

        Returns:
            Outlet temperature (F)
        """
        # Calculate enthalpies
        h_in = self.thermo_calc.calculate_superheated_enthalpy(pressure_psig, inlet_temp_f)
        h_water = self.thermo_calc.calculate_water_enthalpy(spray_water_temp_f)

        # Calculate outlet enthalpy from energy balance
        total_flow = steam_flow_lb_hr + spray_flow_lb_hr
        if total_flow <= 0:
            return inlet_temp_f

        h_out = (steam_flow_lb_hr * h_in + spray_flow_lb_hr * h_water) / total_flow

        # Convert outlet enthalpy to temperature
        t_out = self.thermo_calc.calculate_temperature_from_enthalpy(pressure_psig, h_out)

        return round(t_out, 1)


# =============================================================================
# THERMAL STRESS CALCULATOR
# =============================================================================

class ThermalStressCalculator:
    """
    Calculator for thermal stress in superheater tubes.

    Implements thermal stress calculations for monitoring tube integrity
    during temperature transients.

    Formula:
        sigma_thermal = E * alpha * delta_T / (1 - nu)

    where:
        E = Young's modulus
        alpha = Thermal expansion coefficient
        delta_T = Temperature gradient
        nu = Poisson's ratio

    Example:
        >>> calc = ThermalStressCalculator()
        >>> result = calc.calculate_thermal_stress(
        ...     temperature_inside_f=950.0,
        ...     temperature_outside_f=900.0,
        ...     tube_thickness_in=0.25,
        ...     material="T22",
        ... )
    """

    MATERIAL_PROPERTIES = {
        "T22": {
            "E_psi": ThermalStressConstants.YOUNGS_MODULUS_T22_PSI,
            "alpha": ThermalStressConstants.THERMAL_EXPANSION_T22,
            "nu": ThermalStressConstants.POISSON_RATIO_T22,
            "allowable_stress_psi": 11000.0,  # At 1000F
        },
        "T91": {
            "E_psi": ThermalStressConstants.YOUNGS_MODULUS_T91_PSI,
            "alpha": ThermalStressConstants.THERMAL_EXPANSION_T91,
            "nu": ThermalStressConstants.POISSON_RATIO_T91,
            "allowable_stress_psi": 15600.0,  # At 1000F
        },
    }

    def __init__(self) -> None:
        """Initialize thermal stress calculator."""
        logger.debug("ThermalStressCalculator initialized")

    def calculate_thermal_stress(
        self,
        temperature_inside_f: float,
        temperature_outside_f: float,
        tube_thickness_in: float,
        material: str = "T22",
    ) -> ThermalStressResult:
        """
        Calculate thermal stress in superheater tube - DETERMINISTIC.

        Formula:
            sigma = E * alpha * delta_T / (1 - nu)

        This formula gives the maximum thermal stress for a temperature
        gradient across the tube wall.

        Args:
            temperature_inside_f: Inside wall temperature (F)
            temperature_outside_f: Outside wall temperature (F)
            tube_thickness_in: Tube wall thickness (inches)
            material: Tube material ("T22" or "T91")

        Returns:
            ThermalStressResult with stress analysis
        """
        # Get material properties
        props = self.MATERIAL_PROPERTIES.get(material, self.MATERIAL_PROPERTIES["T22"])

        E = props["E_psi"]
        alpha = props["alpha"]
        nu = props["nu"]
        allowable = props["allowable_stress_psi"]

        # Temperature gradient
        delta_t = abs(temperature_inside_f - temperature_outside_f)

        # Calculate thermal stress
        thermal_stress = E * alpha * delta_t / (1 - nu)

        # Stress ratio (actual / allowable)
        stress_ratio = thermal_stress / allowable if allowable > 0 else 0.0

        # Check if within limits
        is_within_limits = stress_ratio <= 1.0

        # Provenance hash
        provenance_data = {
            "temperature_inside_f": temperature_inside_f,
            "temperature_outside_f": temperature_outside_f,
            "tube_thickness_in": tube_thickness_in,
            "material": material,
            "E_psi": E,
            "alpha": alpha,
            "thermal_stress_psi": thermal_stress,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return ThermalStressResult(
            thermal_stress_psi=round(thermal_stress, 0),
            temperature_gradient_f=round(delta_t, 1),
            temperature_rate_f_min=0.0,  # Set by caller if tracking transient
            stress_ratio=round(stress_ratio, 3),
            is_within_limits=is_within_limits,
            material=material,
            tube_thickness_in=tube_thickness_in,
            calculation_method="thermal_gradient",
            provenance_hash=provenance_hash,
        )

    def calculate_temperature_rate_limit(
        self,
        current_temp_f: float,
        target_temp_f: float,
        material: str = "T22",
        operation_mode: str = "normal",
    ) -> float:
        """
        Calculate maximum allowable temperature rate of change - DETERMINISTIC.

        Returns the recommended temperature rate limit to avoid excessive
        thermal stress during transients.

        Args:
            current_temp_f: Current temperature (F)
            target_temp_f: Target temperature (F)
            material: Tube material
            operation_mode: "normal", "startup", or "emergency"

        Returns:
            Maximum temperature rate (F/min)
        """
        if operation_mode == "startup":
            max_rate = ThermalStressConstants.MAX_TEMP_RATE_STARTUP
        elif operation_mode == "emergency":
            max_rate = ThermalStressConstants.MAX_TEMP_RATE_EMERGENCY
        else:
            max_rate = ThermalStressConstants.MAX_TEMP_RATE_NORMAL

        # Adjust for material (T91 can handle faster rates)
        if material == "T91":
            max_rate *= 1.2

        # Adjust for temperature level (slower at higher temps)
        avg_temp = (current_temp_f + target_temp_f) / 2
        if avg_temp > 900:
            max_rate *= 0.8
        elif avg_temp > 1000:
            max_rate *= 0.6

        return round(max_rate, 1)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_thermodynamics_calculator() -> SteamThermodynamicsCalculator:
    """Factory function to create SteamThermodynamicsCalculator."""
    return SteamThermodynamicsCalculator()


def create_heat_transfer_calculator() -> SuperheaterHeatTransferCalculator:
    """Factory function to create SuperheaterHeatTransferCalculator."""
    return SuperheaterHeatTransferCalculator()


def create_enthalpy_balance_calculator() -> EnthalpyBalanceCalculator:
    """Factory function to create EnthalpyBalanceCalculator."""
    return EnthalpyBalanceCalculator()


def create_thermal_stress_calculator() -> ThermalStressCalculator:
    """Factory function to create ThermalStressCalculator."""
    return ThermalStressCalculator()
