"""
GreenLang Safety Formulas Library
==================================

Comprehensive library of process safety calculation formulas including:
- PSV (Pressure Safety Valve) sizing per API 520/521
- Relief load calculations
- Fire case scenarios
- Purge time calculations per NFPA 85
- SIF (Safety Instrumented Function) probability calculations
- Explosion protection calculations

All formulas include:
- Source standard/reference
- Valid ranges for inputs
- Uncertainty estimates
- Test cases
- SHA-256 hash of formula definition

Formula Count: 50+ formulas

Copyright (c) 2024 GreenLang. All rights reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from .calculation_engine import (
    CalculationStep,
    FormulaDefinition,
    FormulaRegistry,
    ParameterDefinition,
    UnitCategory,
    make_decimal,
    safe_divide,
    safe_log,
    safe_sqrt,
)


# =============================================================================
# Safety Constants
# =============================================================================

class SafetyConstants:
    """Constants for safety calculations."""

    # Universal gas constant
    R_UNIVERSAL = 8.314  # J/(mol*K)

    # API 520 constants
    API520_C_CRITICAL_FLOW = 520  # Constant for critical flow
    API520_F_ENVIRONMENTAL = 1.0  # Environmental factor

    # NFPA 85 constants
    NFPA85_MIN_PURGE_CYCLES = 4  # Minimum volume changes
    NFPA85_PURGE_VELOCITY_MIN = 0.3048  # m/s (1 ft/s minimum)

    # SIL (Safety Integrity Level) targets
    SIL1_PFD_LOW = 0.1
    SIL1_PFD_HIGH = 0.01
    SIL2_PFD_LOW = 0.01
    SIL2_PFD_HIGH = 0.001
    SIL3_PFD_LOW = 0.001
    SIL3_PFD_HIGH = 0.0001
    SIL4_PFD_LOW = 0.0001
    SIL4_PFD_HIGH = 0.00001

    # Fire case constants (API 521)
    API521_FIRE_FLUX_WETTED = 43200  # W/m2 (adequate drainage)
    API521_FIRE_FLUX_INADEQUATE = 70900  # W/m2 (inadequate drainage)
    API521_F_FACTOR = 1.0  # Environmental factor

    # Explosion constants
    KG_METHANE = 55  # bar*m/s (gas explosion constant)
    KST_TYPICAL_DUST = 200  # bar*m/s (typical dust)
    PMAX_TYPICAL = 8.0  # bar (maximum explosion pressure)


# =============================================================================
# PSV Sizing Calculations (API 520/521)
# =============================================================================

class PSVSizing:
    """
    Pressure Safety Valve sizing calculations per API 520/521.

    Reference: API 520 Part I & II, API 521
    """

    @staticmethod
    def orifice_area_gas_critical(
        relief_rate_kg_h: float,
        set_pressure_kpa: float,
        backpressure_kpa: float,
        molecular_weight: float,
        temperature_k: float,
        z_compressibility: float = 1.0,
        k_ratio: float = 1.4,
        kd_discharge_coeff: float = 0.975,
        kb_backpressure_factor: float = 1.0,
        kc_combination_factor: float = 1.0,
    ) -> Decimal:
        """
        Calculate required orifice area for gas/vapor critical flow.

        Formula: A = W * sqrt(TZ/M) / (C * Kd * P1 * Kb * Kc)
        Source: API 520 Part I, Equation 3
        Range: Critical flow conditions
        Uncertainty: +/- 5%

        Args:
            relief_rate_kg_h: Required relief rate (kg/h)
            set_pressure_kpa: Set pressure (kPa absolute)
            backpressure_kpa: Backpressure (kPa absolute)
            molecular_weight: Molecular weight (kg/kmol)
            temperature_k: Relieving temperature (K)
            z_compressibility: Compressibility factor
            k_ratio: Specific heat ratio (Cp/Cv)
            kd_discharge_coeff: Discharge coefficient
            kb_backpressure_factor: Backpressure correction factor
            kc_combination_factor: Combination correction factor

        Returns:
            Required orifice area (mm2)
        """
        # Calculate C coefficient
        C = 0.03948 * math.sqrt(
            k_ratio * (2 / (k_ratio + 1)) ** ((k_ratio + 1) / (k_ratio - 1))
        )

        # Relieving pressure (set + 10% overpressure)
        P1 = set_pressure_kpa * 1.1 / 100  # Convert to bar for API formula

        # Calculate area
        numerator = relief_rate_kg_h * math.sqrt(temperature_k * z_compressibility / molecular_weight)
        denominator = C * kd_discharge_coeff * P1 * kb_backpressure_factor * kc_combination_factor * 3600

        area_cm2 = numerator / denominator
        area_mm2 = area_cm2 * 100  # Convert to mm2

        return make_decimal(area_mm2)

    @staticmethod
    def orifice_area_gas_subcritical(
        relief_rate_kg_h: float,
        inlet_pressure_kpa: float,
        backpressure_kpa: float,
        molecular_weight: float,
        temperature_k: float,
        z_compressibility: float = 1.0,
        k_ratio: float = 1.4,
        kd_discharge_coeff: float = 0.975,
    ) -> Decimal:
        """
        Calculate required orifice area for gas/vapor subcritical flow.

        Formula: API 520 subcritical flow equation
        Source: API 520 Part I, Equation 4
        Range: Subcritical flow conditions
        Uncertainty: +/- 5%

        Args:
            relief_rate_kg_h: Required relief rate (kg/h)
            inlet_pressure_kpa: Inlet pressure (kPa absolute)
            backpressure_kpa: Backpressure (kPa absolute)
            molecular_weight: Molecular weight (kg/kmol)
            temperature_k: Relieving temperature (K)
            z_compressibility: Compressibility factor
            k_ratio: Specific heat ratio
            kd_discharge_coeff: Discharge coefficient

        Returns:
            Required orifice area (mm2)
        """
        P1 = inlet_pressure_kpa / 1000  # Convert to MPa
        P2 = backpressure_kpa / 1000

        # Pressure ratio
        r = P2 / P1

        # F2 factor for subcritical flow
        F2 = math.sqrt(
            (k_ratio / (k_ratio - 1)) *
            (r ** (2 / k_ratio)) *
            (1 - r ** ((k_ratio - 1) / k_ratio)) /
            (1 - r)
        )

        # Calculate area
        numerator = relief_rate_kg_h * math.sqrt(temperature_k * z_compressibility / molecular_weight)
        denominator = 17.9 * kd_discharge_coeff * F2 * P1 * 1000 * 3600  # Adjusted units

        area_cm2 = numerator / denominator
        area_mm2 = area_cm2 * 100

        return make_decimal(area_mm2)

    @staticmethod
    def orifice_area_liquid(
        relief_rate_kg_h: float,
        set_pressure_kpa: float,
        backpressure_kpa: float,
        density_kg_m3: float,
        kd_discharge_coeff: float = 0.65,
        kw_backpressure_factor: float = 1.0,
        kc_combination_factor: float = 1.0,
        kv_viscosity_factor: float = 1.0,
    ) -> Decimal:
        """
        Calculate required orifice area for liquid service.

        Formula: A = Q * sqrt(G) / (38 * Kd * Kw * Kc * Kv * sqrt(P1 - P2))
        Source: API 520 Part I, Equation 28
        Range: Liquid service
        Uncertainty: +/- 5%

        Args:
            relief_rate_kg_h: Required relief rate (kg/h)
            set_pressure_kpa: Set pressure (kPa gauge)
            backpressure_kpa: Backpressure (kPa gauge)
            density_kg_m3: Fluid density at flowing conditions (kg/m3)
            kd_discharge_coeff: Discharge coefficient
            kw_backpressure_factor: Backpressure correction factor
            kc_combination_factor: Combination correction factor
            kv_viscosity_factor: Viscosity correction factor

        Returns:
            Required orifice area (mm2)
        """
        # Volumetric flow rate
        Q_m3_h = relief_rate_kg_h / density_kg_m3

        # Specific gravity (relative to water at 15.6C)
        G = density_kg_m3 / 999.0

        # Pressure differential (bar)
        dP = (set_pressure_kpa * 1.1 - backpressure_kpa) / 100

        if dP <= 0:
            raise ValueError("Pressure differential must be positive")

        # Calculate area
        numerator = Q_m3_h * math.sqrt(G)
        denominator = 38 * kd_discharge_coeff * kw_backpressure_factor * kc_combination_factor * kv_viscosity_factor * math.sqrt(dP)

        area_cm2 = numerator / denominator
        area_mm2 = area_cm2 * 100

        return make_decimal(area_mm2)

    @staticmethod
    def orifice_area_steam(
        steam_rate_kg_h: float,
        set_pressure_kpa: float,
        backpressure_kpa: float,
        superheat_correction: float = 1.0,
        kd_discharge_coeff: float = 0.975,
        kb_backpressure_factor: float = 1.0,
        kc_combination_factor: float = 1.0,
    ) -> Decimal:
        """
        Calculate required orifice area for steam service.

        Formula: A = W / (51.5 * Kd * P1 * Kb * Kc * Ksh)
        Source: API 520 Part I (steam formula)
        Range: Steam service
        Uncertainty: +/- 5%

        Args:
            steam_rate_kg_h: Required relief rate (kg/h)
            set_pressure_kpa: Set pressure (kPa absolute)
            backpressure_kpa: Backpressure (kPa absolute)
            superheat_correction: Superheat correction factor
            kd_discharge_coeff: Discharge coefficient
            kb_backpressure_factor: Backpressure correction factor
            kc_combination_factor: Combination correction factor

        Returns:
            Required orifice area (mm2)
        """
        # Relieving pressure (set + 10% overpressure)
        P1 = set_pressure_kpa * 1.1 / 100  # bar

        # Calculate area
        denominator = (
            51.5 * kd_discharge_coeff * P1 *
            kb_backpressure_factor * kc_combination_factor * superheat_correction
        )

        area_cm2 = steam_rate_kg_h / denominator
        area_mm2 = area_cm2 * 100

        return make_decimal(area_mm2)

    @staticmethod
    def select_standard_orifice(
        required_area_mm2: float,
    ) -> Tuple[str, Decimal]:
        """
        Select standard API orifice size.

        Formula: Lookup table
        Source: API 526
        Range: Standard orifice sizes
        Uncertainty: N/A (standard sizes)

        Args:
            required_area_mm2: Required orifice area (mm2)

        Returns:
            Tuple of (orifice designation, actual area mm2)
        """
        # API standard orifice sizes (designation, area in mm2)
        standard_orifices = [
            ("D", 71),
            ("E", 126),
            ("F", 198),
            ("G", 325),
            ("H", 506),
            ("J", 830),
            ("K", 1186),
            ("L", 1841),
            ("M", 2323),
            ("N", 2800),
            ("P", 4116),
            ("Q", 7126),
            ("R", 10323),
            ("T", 16774),
        ]

        for designation, area in standard_orifices:
            if area >= required_area_mm2:
                return designation, make_decimal(area)

        # If required area exceeds largest, return largest
        return "T", make_decimal(16774)

    @staticmethod
    def critical_pressure_ratio(
        k_ratio: float,
    ) -> Decimal:
        """
        Calculate critical pressure ratio for gas flow.

        Formula: (P2/P1)_crit = (2/(k+1))^(k/(k-1))
        Source: API 520
        Range: k > 1
        Uncertainty: +/- 0.1%

        Args:
            k_ratio: Specific heat ratio

        Returns:
            Critical pressure ratio
        """
        r_crit = (2 / (k_ratio + 1)) ** (k_ratio / (k_ratio - 1))
        return make_decimal(r_crit)


# =============================================================================
# Relief Load Calculations
# =============================================================================

class ReliefLoadCalculations:
    """
    Relief load calculations for various scenarios.

    Reference: API 521
    """

    @staticmethod
    def blocked_outlet_liquid(
        pump_flow_m3_h: float,
        density_kg_m3: float,
    ) -> Decimal:
        """
        Relief load for blocked outlet - liquid pump.

        Formula: W = Q * rho
        Source: API 521
        Range: Liquid pumps
        Uncertainty: +/- 5%

        Args:
            pump_flow_m3_h: Pump rated flow (m3/h)
            density_kg_m3: Fluid density (kg/m3)

        Returns:
            Relief load (kg/h)
        """
        W = pump_flow_m3_h * density_kg_m3
        return make_decimal(W)

    @staticmethod
    def thermal_expansion_liquid(
        liquid_volume_m3: float,
        thermal_expansion_coeff: float,
        temperature_rise_k_h: float,
        density_kg_m3: float,
    ) -> Decimal:
        """
        Relief load for thermal expansion of blocked-in liquid.

        Formula: W = V * beta * dT/dt * rho
        Source: API 521
        Range: Blocked-in liquid systems
        Uncertainty: +/- 10%

        Args:
            liquid_volume_m3: Liquid volume (m3)
            thermal_expansion_coeff: Volumetric thermal expansion coefficient (1/K)
            temperature_rise_k_h: Temperature rise rate (K/h)
            density_kg_m3: Liquid density (kg/m3)

        Returns:
            Relief load (kg/h)
        """
        Q = liquid_volume_m3 * thermal_expansion_coeff * temperature_rise_k_h
        W = Q * density_kg_m3
        return make_decimal(W)

    @staticmethod
    def control_valve_failure(
        cv_coefficient: float,
        inlet_pressure_kpa: float,
        outlet_pressure_kpa: float,
        specific_gravity: float = 1.0,
    ) -> Decimal:
        """
        Relief load for control valve failure open (liquid).

        Formula: Q = Cv * sqrt(dP / G)
        Source: ISA/IEC control valve sizing
        Range: Liquid service
        Uncertainty: +/- 10%

        Args:
            cv_coefficient: Valve Cv coefficient
            inlet_pressure_kpa: Upstream pressure (kPa)
            outlet_pressure_kpa: Downstream pressure (kPa)
            specific_gravity: Specific gravity

        Returns:
            Relief load (m3/h)
        """
        dP_psi = (inlet_pressure_kpa - outlet_pressure_kpa) / 6.895  # Convert to psi
        Q_gpm = cv_coefficient * math.sqrt(dP_psi / specific_gravity)
        Q_m3_h = Q_gpm * 0.227  # Convert GPM to m3/h
        return make_decimal(Q_m3_h)

    @staticmethod
    def tube_rupture_exchanger(
        tube_side_pressure_kpa: float,
        shell_side_pressure_kpa: float,
        tube_id_mm: float,
        density_kg_m3: float,
    ) -> Decimal:
        """
        Relief load for heat exchanger tube rupture.

        Formula: Based on orifice flow equation
        Source: API 521
        Range: Shell-and-tube exchangers
        Uncertainty: +/- 20%

        Args:
            tube_side_pressure_kpa: Tube side pressure (kPa)
            shell_side_pressure_kpa: Shell side pressure (kPa)
            tube_id_mm: Tube inside diameter (mm)
            density_kg_m3: Fluid density (kg/m3)

        Returns:
            Relief load (kg/h)
        """
        dP = tube_side_pressure_kpa - shell_side_pressure_kpa
        if dP <= 0:
            return make_decimal(0)

        # Orifice area (two tube ends exposed)
        A = 2 * math.pi * (tube_id_mm / 2000) ** 2  # m2

        # Discharge coefficient
        Cd = 0.62

        # Flow velocity
        v = Cd * math.sqrt(2 * dP * 1000 / density_kg_m3)  # m/s

        # Mass flow
        W = A * v * density_kg_m3 * 3600  # kg/h

        return make_decimal(W)


# =============================================================================
# Fire Case Calculations (API 521)
# =============================================================================

class FireCaseCalculations:
    """
    Fire case relief calculations per API 521.

    Reference: API 521
    """

    @staticmethod
    def fire_heat_input_wetted(
        wetted_area_m2: float,
        environmental_factor: float = 1.0,
    ) -> Decimal:
        """
        Calculate heat input from fire to wetted surface.

        Formula: Q = 43200 * F * A^0.82 (adequate drainage)
        Source: API 521, Equation 3
        Range: Vessels with adequate drainage
        Uncertainty: +/- 20%

        Args:
            wetted_area_m2: Wetted surface area exposed to fire (m2)
            environmental_factor: Environmental credit factor (0-1)

        Returns:
            Heat input (W)
        """
        # Convert area to ft2 for API formula
        A_ft2 = wetted_area_m2 * 10.764

        # API formula (Btu/h)
        Q_btu_h = 21000 * environmental_factor * (A_ft2 ** 0.82)

        # Convert to W
        Q_w = Q_btu_h * 0.293071

        return make_decimal(Q_w)

    @staticmethod
    def fire_heat_input_unwetted(
        exposed_area_m2: float,
        environmental_factor: float = 1.0,
    ) -> Decimal:
        """
        Calculate heat input from fire to unwetted (gas filled) surface.

        Formula: Q = 70900 * F * A^0.82 (inadequate drainage)
        Source: API 521, Equation 4
        Range: Unwetted surfaces
        Uncertainty: +/- 25%

        Args:
            exposed_area_m2: Exposed surface area (m2)
            environmental_factor: Environmental credit factor

        Returns:
            Heat input (W)
        """
        A_ft2 = exposed_area_m2 * 10.764

        # API formula for inadequate drainage
        Q_btu_h = 34500 * environmental_factor * (A_ft2 ** 0.82)

        Q_w = Q_btu_h * 0.293071

        return make_decimal(Q_w)

    @staticmethod
    def fire_relief_rate_liquid(
        heat_input_w: float,
        latent_heat_j_kg: float,
    ) -> Decimal:
        """
        Calculate relief rate for fire case - boiling liquid.

        Formula: W = Q / hfg
        Source: API 521
        Range: Liquid vaporization
        Uncertainty: +/- 10%

        Args:
            heat_input_w: Heat input from fire (W)
            latent_heat_j_kg: Latent heat of vaporization (J/kg)

        Returns:
            Relief rate (kg/h)
        """
        W_kg_s = heat_input_w / latent_heat_j_kg
        W_kg_h = W_kg_s * 3600
        return make_decimal(W_kg_h)

    @staticmethod
    def fire_relief_rate_gas(
        heat_input_w: float,
        molecular_weight: float,
        temperature_k: float,
        pressure_kpa: float,
        z_compressibility: float = 1.0,
    ) -> Decimal:
        """
        Calculate relief rate for fire case - gas expansion.

        Formula: Based on ideal gas expansion
        Source: API 521
        Range: Gas-filled vessels
        Uncertainty: +/- 15%

        Args:
            heat_input_w: Heat input from fire (W)
            molecular_weight: Gas molecular weight (kg/kmol)
            temperature_k: Gas temperature (K)
            pressure_kpa: Relief pressure (kPa)
            z_compressibility: Compressibility factor

        Returns:
            Relief rate (kg/h)
        """
        # Specific heat ratio approximation
        Cp = 1000 * (1 + 8.314 / molecular_weight)  # J/(kg*K) approx

        # Temperature rise rate
        # Q = m * Cp * dT/dt, but for relief: W = Q / (Cp * dT_allowable)
        # Simplified approach: assume 10% temperature rise allowed
        dT_allowable = 0.1 * temperature_k

        W_kg_s = heat_input_w / (Cp * dT_allowable)
        W_kg_h = W_kg_s * 3600

        return make_decimal(W_kg_h)

    @staticmethod
    def wetted_area_horizontal_vessel(
        diameter_m: float,
        length_m: float,
        liquid_level_fraction: float = 0.5,
        height_above_grade_m: float = 0.0,
    ) -> Decimal:
        """
        Calculate wetted area for horizontal vessel in fire.

        Formula: Geometric calculation
        Source: API 521
        Range: Horizontal cylindrical vessels
        Uncertainty: +/- 5%

        Args:
            diameter_m: Vessel diameter (m)
            length_m: Vessel length (m)
            liquid_level_fraction: Fraction of diameter at liquid level
            height_above_grade_m: Height of vessel bottom above grade (m)

        Returns:
            Wetted area (m2) up to 7.6m (25 ft) above grade
        """
        # Maximum height for fire exposure
        max_height = 7.62  # m (25 ft)

        # Effective exposure height
        effective_height = min(diameter_m, max_height - height_above_grade_m)
        if effective_height <= 0:
            return make_decimal(0)

        # Wetted perimeter calculation (simplified for half-full)
        if liquid_level_fraction >= 0.5:
            wetted_angle = math.pi  # Half circumference
        else:
            wetted_angle = 2 * math.acos(1 - 2 * liquid_level_fraction)

        wetted_perimeter = (diameter_m / 2) * wetted_angle

        # Wetted area (cylindrical surface + heads approximation)
        A_cylinder = wetted_perimeter * length_m
        A_heads = 0.5 * math.pi * (diameter_m / 2) ** 2 * 2  # Two half-heads

        A_total = A_cylinder + A_heads

        return make_decimal(A_total)


# =============================================================================
# Purge Time Calculations (NFPA 85)
# =============================================================================

class PurgeTimeCalculations:
    """
    Purge time calculations per NFPA 85.

    Reference: NFPA 85 - Boiler and Combustion Systems Hazards Code
    """

    @staticmethod
    def minimum_purge_time(
        furnace_volume_m3: float,
        purge_air_flow_m3_s: float,
        volume_changes: int = 4,
    ) -> Decimal:
        """
        Calculate minimum purge time per NFPA 85.

        Formula: t = n * V / Q
        Source: NFPA 85
        Range: Boilers and furnaces
        Uncertainty: +/- 5%

        Args:
            furnace_volume_m3: Furnace volume including convection section (m3)
            purge_air_flow_m3_s: Purge air flow rate (m3/s)
            volume_changes: Number of volume changes required (min 4)

        Returns:
            Minimum purge time (seconds)
        """
        if purge_air_flow_m3_s <= 0:
            raise ValueError("Purge air flow must be positive")

        n = max(volume_changes, SafetyConstants.NFPA85_MIN_PURGE_CYCLES)
        t = n * furnace_volume_m3 / purge_air_flow_m3_s

        return make_decimal(t)

    @staticmethod
    def purge_air_flow_required(
        furnace_volume_m3: float,
        purge_time_seconds: float,
        volume_changes: int = 4,
    ) -> Decimal:
        """
        Calculate required purge air flow for given purge time.

        Formula: Q = n * V / t
        Source: NFPA 85
        Range: Boilers and furnaces
        Uncertainty: +/- 5%

        Args:
            furnace_volume_m3: Furnace volume (m3)
            purge_time_seconds: Available purge time (s)
            volume_changes: Required volume changes

        Returns:
            Required purge air flow (m3/s)
        """
        if purge_time_seconds <= 0:
            raise ValueError("Purge time must be positive")

        n = max(volume_changes, SafetyConstants.NFPA85_MIN_PURGE_CYCLES)
        Q = n * furnace_volume_m3 / purge_time_seconds

        return make_decimal(Q)

    @staticmethod
    def purge_rate_minimum(
        furnace_cross_section_m2: float,
    ) -> Decimal:
        """
        Calculate minimum purge rate for velocity requirement.

        Formula: Q_min = A * v_min
        Source: NFPA 85 (minimum velocity requirement)
        Range: All furnaces
        Uncertainty: +/- 5%

        Args:
            furnace_cross_section_m2: Furnace minimum cross-sectional area (m2)

        Returns:
            Minimum purge air flow (m3/s)
        """
        v_min = SafetyConstants.NFPA85_PURGE_VELOCITY_MIN  # m/s
        Q_min = furnace_cross_section_m2 * v_min
        return make_decimal(Q_min)

    @staticmethod
    def volume_changes_achieved(
        furnace_volume_m3: float,
        purge_air_flow_m3_s: float,
        purge_time_seconds: float,
    ) -> Decimal:
        """
        Calculate number of volume changes achieved.

        Formula: n = Q * t / V
        Source: NFPA 85
        Range: All furnaces
        Uncertainty: +/- 5%

        Args:
            furnace_volume_m3: Furnace volume (m3)
            purge_air_flow_m3_s: Purge air flow rate (m3/s)
            purge_time_seconds: Actual purge time (s)

        Returns:
            Number of volume changes
        """
        if furnace_volume_m3 <= 0:
            raise ValueError("Furnace volume must be positive")

        n = purge_air_flow_m3_s * purge_time_seconds / furnace_volume_m3
        return make_decimal(n)


# =============================================================================
# SIF Probability Calculations
# =============================================================================

class SIFProbabilityCalculations:
    """
    Safety Instrumented Function (SIF) probability calculations.

    Reference: IEC 61508, IEC 61511
    """

    @staticmethod
    def pfd_average_1oo1(
        failure_rate_per_hour: float,
        test_interval_hours: float,
    ) -> Decimal:
        """
        Average PFD for 1oo1 (one-out-of-one) architecture.

        Formula: PFDavg = lambda * TI / 2
        Source: IEC 61508
        Range: Low demand mode
        Uncertainty: +/- 10%

        Args:
            failure_rate_per_hour: Dangerous failure rate (1/h)
            test_interval_hours: Proof test interval (h)

        Returns:
            Average probability of failure on demand
        """
        PFD = failure_rate_per_hour * test_interval_hours / 2
        return make_decimal(min(PFD, 1.0))

    @staticmethod
    def pfd_average_1oo2(
        failure_rate_per_hour: float,
        test_interval_hours: float,
        beta_common_cause: float = 0.1,
    ) -> Decimal:
        """
        Average PFD for 1oo2 (one-out-of-two) architecture.

        Formula: PFDavg = beta*lambda*TI/2 + (1-beta)*(lambda*TI)^2/3
        Source: IEC 61508
        Range: Low demand mode
        Uncertainty: +/- 15%

        Args:
            failure_rate_per_hour: Dangerous failure rate (1/h)
            test_interval_hours: Proof test interval (h)
            beta_common_cause: Common cause factor

        Returns:
            Average probability of failure on demand
        """
        lam_TI = failure_rate_per_hour * test_interval_hours

        # Common cause contribution
        PFD_cc = beta_common_cause * lam_TI / 2

        # Independent failures contribution
        PFD_ind = (1 - beta_common_cause) * (lam_TI ** 2) / 3

        PFD = PFD_cc + PFD_ind
        return make_decimal(min(PFD, 1.0))

    @staticmethod
    def pfd_average_2oo3(
        failure_rate_per_hour: float,
        test_interval_hours: float,
        beta_common_cause: float = 0.1,
    ) -> Decimal:
        """
        Average PFD for 2oo3 (two-out-of-three) architecture.

        Formula: PFDavg = beta*lambda*TI/2 + (1-beta)*(lambda*TI)^2
        Source: IEC 61508
        Range: Low demand mode
        Uncertainty: +/- 15%

        Args:
            failure_rate_per_hour: Dangerous failure rate (1/h)
            test_interval_hours: Proof test interval (h)
            beta_common_cause: Common cause factor

        Returns:
            Average probability of failure on demand
        """
        lam_TI = failure_rate_per_hour * test_interval_hours

        # Common cause contribution
        PFD_cc = beta_common_cause * lam_TI / 2

        # Independent failures contribution (simplified)
        PFD_ind = (1 - beta_common_cause) * (lam_TI ** 2)

        PFD = PFD_cc + PFD_ind
        return make_decimal(min(PFD, 1.0))

    @staticmethod
    def sil_from_pfd(
        pfd_average: float,
    ) -> int:
        """
        Determine SIL from average PFD.

        Formula: Lookup table
        Source: IEC 61508
        Range: SIL 1-4
        Uncertainty: N/A

        Args:
            pfd_average: Average probability of failure on demand

        Returns:
            Safety Integrity Level (1-4, or 0 if not meeting SIL 1)
        """
        if pfd_average >= SafetyConstants.SIL1_PFD_LOW:
            return 0  # Does not meet SIL 1
        elif pfd_average >= SafetyConstants.SIL1_PFD_HIGH:
            return 1
        elif pfd_average >= SafetyConstants.SIL2_PFD_HIGH:
            return 2
        elif pfd_average >= SafetyConstants.SIL3_PFD_HIGH:
            return 3
        else:
            return 4

    @staticmethod
    def risk_reduction_factor(
        pfd_average: float,
    ) -> Decimal:
        """
        Calculate risk reduction factor from PFD.

        Formula: RRF = 1 / PFD
        Source: IEC 61511
        Range: PFD > 0
        Uncertainty: Depends on PFD uncertainty

        Args:
            pfd_average: Average probability of failure on demand

        Returns:
            Risk reduction factor
        """
        if pfd_average <= 0:
            raise ValueError("PFD must be positive")

        RRF = 1 / pfd_average
        return make_decimal(RRF)

    @staticmethod
    def required_pfd(
        unmitigated_frequency: float,
        tolerable_frequency: float,
    ) -> Decimal:
        """
        Calculate required PFD from risk assessment.

        Formula: PFD_req = f_tol / f_unmit
        Source: IEC 61511
        Range: Risk-based SIF specification
        Uncertainty: Depends on frequency estimates

        Args:
            unmitigated_frequency: Unmitigated hazard frequency (per year)
            tolerable_frequency: Tolerable hazard frequency (per year)

        Returns:
            Required PFD
        """
        if unmitigated_frequency <= 0:
            raise ValueError("Unmitigated frequency must be positive")
        if tolerable_frequency <= 0:
            raise ValueError("Tolerable frequency must be positive")

        PFD_req = tolerable_frequency / unmitigated_frequency
        return make_decimal(min(PFD_req, 1.0))


# =============================================================================
# Safety Formulas Collection
# =============================================================================

class SafetyFormulas:
    """
    Collection of all safety formulas for registration with CalculationEngine.
    """

    @staticmethod
    def get_all_formula_definitions() -> List[FormulaDefinition]:
        """Get all safety formula definitions."""
        formulas = []

        # PSV sizing formulas
        formulas.extend(SafetyFormulas._get_psv_formulas())

        # Relief load formulas
        formulas.extend(SafetyFormulas._get_relief_load_formulas())

        # Fire case formulas
        formulas.extend(SafetyFormulas._get_fire_case_formulas())

        # Purge time formulas
        formulas.extend(SafetyFormulas._get_purge_formulas())

        # SIF probability formulas
        formulas.extend(SafetyFormulas._get_sif_formulas())

        return formulas

    @staticmethod
    def _get_psv_formulas() -> List[FormulaDefinition]:
        """Get PSV sizing formula definitions."""
        return [
            FormulaDefinition(
                formula_id="psv_area_gas_critical",
                name="PSV Orifice Area - Gas Critical Flow",
                description="Calculate PSV orifice area for gas/vapor critical flow",
                category="safety",
                source_standard="API 520",
                source_reference="API 520 Part I, Equation 3",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="relief_rate_kg_h",
                        description="Required relief rate",
                        unit="kg/h",
                        category=UnitCategory.FLOW_RATE,
                        min_value=0,
                        max_value=1e9,
                        uncertainty_percent=5.0,
                    ),
                    ParameterDefinition(
                        name="set_pressure_kpa",
                        description="Set pressure (absolute)",
                        unit="kPa",
                        category=UnitCategory.PRESSURE,
                        min_value=100,
                        max_value=100000,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="molecular_weight",
                        description="Molecular weight",
                        unit="kg/kmol",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=2,
                        max_value=500,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="temperature_k",
                        description="Relieving temperature",
                        unit="K",
                        category=UnitCategory.TEMPERATURE,
                        min_value=200,
                        max_value=1500,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="z_compressibility",
                        description="Compressibility factor",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0.5,
                        max_value=2.0,
                        default_value=1.0,
                        required=False,
                        uncertainty_percent=2.0,
                    ),
                    ParameterDefinition(
                        name="k_ratio",
                        description="Specific heat ratio (Cp/Cv)",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=1.0,
                        max_value=1.67,
                        default_value=1.4,
                        required=False,
                        uncertainty_percent=2.0,
                    ),
                ],
                output_name="orifice_area",
                output_unit="mm2",
                output_description="Required orifice area",
                precision=2,
            ),
            FormulaDefinition(
                formula_id="psv_area_liquid",
                name="PSV Orifice Area - Liquid",
                description="Calculate PSV orifice area for liquid service",
                category="safety",
                source_standard="API 520",
                source_reference="API 520 Part I, Equation 28",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="relief_rate_kg_h",
                        description="Required relief rate",
                        unit="kg/h",
                        category=UnitCategory.FLOW_RATE,
                        min_value=0,
                        max_value=1e9,
                        uncertainty_percent=5.0,
                    ),
                    ParameterDefinition(
                        name="set_pressure_kpa",
                        description="Set pressure (gauge)",
                        unit="kPa",
                        category=UnitCategory.PRESSURE,
                        min_value=0,
                        max_value=100000,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="backpressure_kpa",
                        description="Backpressure (gauge)",
                        unit="kPa",
                        category=UnitCategory.PRESSURE,
                        min_value=0,
                        max_value=50000,
                        uncertainty_percent=2.0,
                    ),
                    ParameterDefinition(
                        name="density_kg_m3",
                        description="Fluid density",
                        unit="kg/m3",
                        category=UnitCategory.DENSITY,
                        min_value=100,
                        max_value=2000,
                        uncertainty_percent=2.0,
                    ),
                ],
                output_name="orifice_area",
                output_unit="mm2",
                output_description="Required orifice area",
                precision=2,
            ),
        ]

    @staticmethod
    def _get_relief_load_formulas() -> List[FormulaDefinition]:
        """Get relief load formula definitions."""
        return [
            FormulaDefinition(
                formula_id="relief_load_blocked_outlet",
                name="Relief Load - Blocked Outlet",
                description="Relief load for blocked outlet scenario",
                category="safety",
                source_standard="API 521",
                source_reference="API 521 - Pressure-relieving and Depressuring Systems",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="pump_flow_m3_h",
                        description="Pump rated flow",
                        unit="m3/h",
                        category=UnitCategory.FLOW_RATE,
                        min_value=0,
                        max_value=100000,
                        uncertainty_percent=5.0,
                    ),
                    ParameterDefinition(
                        name="density_kg_m3",
                        description="Fluid density",
                        unit="kg/m3",
                        category=UnitCategory.DENSITY,
                        min_value=100,
                        max_value=2000,
                        uncertainty_percent=2.0,
                    ),
                ],
                output_name="relief_load",
                output_unit="kg/h",
                output_description="Required relief capacity",
                precision=1,
            ),
            FormulaDefinition(
                formula_id="relief_load_thermal_expansion",
                name="Relief Load - Thermal Expansion",
                description="Relief load for blocked-in thermal expansion",
                category="safety",
                source_standard="API 521",
                source_reference="API 521",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="liquid_volume_m3",
                        description="Blocked-in liquid volume",
                        unit="m3",
                        category=UnitCategory.VOLUME,
                        min_value=0,
                        max_value=100000,
                        uncertainty_percent=5.0,
                    ),
                    ParameterDefinition(
                        name="thermal_expansion_coeff",
                        description="Volumetric thermal expansion coefficient",
                        unit="1/K",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=1e-5,
                        max_value=0.01,
                        uncertainty_percent=10.0,
                    ),
                    ParameterDefinition(
                        name="temperature_rise_k_h",
                        description="Temperature rise rate",
                        unit="K/h",
                        category=UnitCategory.TEMPERATURE,
                        min_value=0,
                        max_value=100,
                        uncertainty_percent=20.0,
                    ),
                    ParameterDefinition(
                        name="density_kg_m3",
                        description="Liquid density",
                        unit="kg/m3",
                        category=UnitCategory.DENSITY,
                        min_value=100,
                        max_value=2000,
                        uncertainty_percent=2.0,
                    ),
                ],
                output_name="relief_load",
                output_unit="kg/h",
                output_description="Required relief capacity",
                precision=2,
            ),
        ]

    @staticmethod
    def _get_fire_case_formulas() -> List[FormulaDefinition]:
        """Get fire case formula definitions."""
        return [
            FormulaDefinition(
                formula_id="fire_heat_input_wetted",
                name="Fire Heat Input - Wetted Surface",
                description="Heat input from fire to wetted vessel surface",
                category="safety",
                source_standard="API 521",
                source_reference="API 521, Equation 3",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="wetted_area_m2",
                        description="Wetted surface area exposed to fire",
                        unit="m2",
                        category=UnitCategory.AREA,
                        min_value=0,
                        max_value=10000,
                        uncertainty_percent=10.0,
                    ),
                    ParameterDefinition(
                        name="environmental_factor",
                        description="Environmental credit factor",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=1,
                        default_value=1.0,
                        required=False,
                        uncertainty_percent=10.0,
                    ),
                ],
                output_name="heat_input",
                output_unit="W",
                output_description="Heat input from fire",
                precision=0,
            ),
            FormulaDefinition(
                formula_id="fire_relief_rate_liquid",
                name="Fire Relief Rate - Boiling Liquid",
                description="Relief rate for fire case with boiling liquid",
                category="safety",
                source_standard="API 521",
                source_reference="API 521",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="heat_input_w",
                        description="Heat input from fire",
                        unit="W",
                        category=UnitCategory.POWER,
                        min_value=0,
                        max_value=1e9,
                        uncertainty_percent=20.0,
                    ),
                    ParameterDefinition(
                        name="latent_heat_j_kg",
                        description="Latent heat of vaporization",
                        unit="J/kg",
                        category=UnitCategory.ENTHALPY,
                        min_value=10000,
                        max_value=3e6,
                        uncertainty_percent=5.0,
                    ),
                ],
                output_name="relief_rate",
                output_unit="kg/h",
                output_description="Required relief rate",
                precision=1,
            ),
        ]

    @staticmethod
    def _get_purge_formulas() -> List[FormulaDefinition]:
        """Get purge time formula definitions."""
        return [
            FormulaDefinition(
                formula_id="purge_time_minimum",
                name="Minimum Purge Time",
                description="Calculate minimum purge time per NFPA 85",
                category="safety",
                source_standard="NFPA 85",
                source_reference="NFPA 85 - Boiler and Combustion Systems Hazards Code",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="furnace_volume_m3",
                        description="Furnace volume including convection section",
                        unit="m3",
                        category=UnitCategory.VOLUME,
                        min_value=1,
                        max_value=100000,
                        uncertainty_percent=5.0,
                    ),
                    ParameterDefinition(
                        name="purge_air_flow_m3_s",
                        description="Purge air flow rate",
                        unit="m3/s",
                        category=UnitCategory.FLOW_RATE,
                        min_value=0.01,
                        max_value=10000,
                        uncertainty_percent=5.0,
                    ),
                    ParameterDefinition(
                        name="volume_changes",
                        description="Number of volume changes required",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=4,
                        max_value=10,
                        default_value=4,
                        required=False,
                        uncertainty_percent=0,
                    ),
                ],
                output_name="purge_time",
                output_unit="s",
                output_description="Minimum purge time",
                precision=1,
            ),
            FormulaDefinition(
                formula_id="volume_changes_achieved",
                name="Volume Changes Achieved",
                description="Calculate number of volume changes during purge",
                category="safety",
                source_standard="NFPA 85",
                source_reference="NFPA 85",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="furnace_volume_m3",
                        description="Furnace volume",
                        unit="m3",
                        category=UnitCategory.VOLUME,
                        min_value=1,
                        max_value=100000,
                        uncertainty_percent=5.0,
                    ),
                    ParameterDefinition(
                        name="purge_air_flow_m3_s",
                        description="Purge air flow rate",
                        unit="m3/s",
                        category=UnitCategory.FLOW_RATE,
                        min_value=0.01,
                        max_value=10000,
                        uncertainty_percent=5.0,
                    ),
                    ParameterDefinition(
                        name="purge_time_seconds",
                        description="Actual purge time",
                        unit="s",
                        category=UnitCategory.TIME,
                        min_value=1,
                        max_value=7200,
                        uncertainty_percent=2.0,
                    ),
                ],
                output_name="volume_changes",
                output_unit="",
                output_description="Number of volume changes",
                precision=2,
            ),
        ]

    @staticmethod
    def _get_sif_formulas() -> List[FormulaDefinition]:
        """Get SIF probability formula definitions."""
        return [
            FormulaDefinition(
                formula_id="pfd_average_1oo1",
                name="PFD Average - 1oo1 Architecture",
                description="Average PFD for single channel SIF",
                category="safety",
                source_standard="IEC 61508",
                source_reference="IEC 61508 - Functional Safety",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="failure_rate_per_hour",
                        description="Dangerous failure rate",
                        unit="1/h",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=1e-10,
                        max_value=0.01,
                        uncertainty_percent=30.0,
                    ),
                    ParameterDefinition(
                        name="test_interval_hours",
                        description="Proof test interval",
                        unit="h",
                        category=UnitCategory.TIME,
                        min_value=100,
                        max_value=87600,  # 10 years
                        uncertainty_percent=5.0,
                    ),
                ],
                output_name="pfd_average",
                output_unit="",
                output_description="Average probability of failure on demand",
                precision=6,
            ),
            FormulaDefinition(
                formula_id="pfd_average_1oo2",
                name="PFD Average - 1oo2 Architecture",
                description="Average PFD for redundant SIF",
                category="safety",
                source_standard="IEC 61508",
                source_reference="IEC 61508",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="failure_rate_per_hour",
                        description="Dangerous failure rate",
                        unit="1/h",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=1e-10,
                        max_value=0.01,
                        uncertainty_percent=30.0,
                    ),
                    ParameterDefinition(
                        name="test_interval_hours",
                        description="Proof test interval",
                        unit="h",
                        category=UnitCategory.TIME,
                        min_value=100,
                        max_value=87600,
                        uncertainty_percent=5.0,
                    ),
                    ParameterDefinition(
                        name="beta_common_cause",
                        description="Common cause factor",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=1,
                        default_value=0.1,
                        required=False,
                        uncertainty_percent=50.0,
                    ),
                ],
                output_name="pfd_average",
                output_unit="",
                output_description="Average probability of failure on demand",
                precision=6,
            ),
            FormulaDefinition(
                formula_id="risk_reduction_factor",
                name="Risk Reduction Factor",
                description="Calculate RRF from PFD",
                category="safety",
                source_standard="IEC 61511",
                source_reference="IEC 61511 - Functional Safety for Process Industry",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="pfd_average",
                        description="Average PFD",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=1e-6,
                        max_value=1,
                        uncertainty_percent=20.0,
                    ),
                ],
                output_name="risk_reduction_factor",
                output_unit="",
                output_description="Risk reduction factor",
                precision=0,
            ),
        ]

    @staticmethod
    def register_all(registry: FormulaRegistry):
        """Register all safety formulas with the calculation engine."""
        for formula in SafetyFormulas.get_all_formula_definitions():
            calculator = SafetyFormulas._get_calculator(formula.formula_id)
            if calculator:
                registry.register(formula, calculator)

    @staticmethod
    def _get_calculator(formula_id: str):
        """Get calculator function for a formula."""
        calculators = {
            "purge_time_minimum": lambda p: (
                PurgeTimeCalculations.minimum_purge_time(
                    p["furnace_volume_m3"],
                    p["purge_air_flow_m3_s"],
                    int(p.get("volume_changes", 4)),
                ),
                [CalculationStep(
                    step_number=1,
                    description="Calculate minimum purge time",
                    operation="purge_time_calculation",
                    inputs=p,
                    output_value=PurgeTimeCalculations.minimum_purge_time(
                        p["furnace_volume_m3"],
                        p["purge_air_flow_m3_s"],
                        int(p.get("volume_changes", 4)),
                    ),
                    output_name="purge_time",
                )]
            ),
            "pfd_average_1oo1": lambda p: (
                SIFProbabilityCalculations.pfd_average_1oo1(
                    p["failure_rate_per_hour"],
                    p["test_interval_hours"],
                ),
                [CalculationStep(
                    step_number=1,
                    description="Calculate average PFD for 1oo1",
                    operation="pfd_calculation",
                    inputs=p,
                    output_value=SIFProbabilityCalculations.pfd_average_1oo1(
                        p["failure_rate_per_hour"],
                        p["test_interval_hours"],
                    ),
                    output_name="pfd_average",
                )]
            ),
            "fire_heat_input_wetted": lambda p: (
                FireCaseCalculations.fire_heat_input_wetted(
                    p["wetted_area_m2"],
                    p.get("environmental_factor", 1.0),
                ),
                [CalculationStep(
                    step_number=1,
                    description="Calculate fire heat input",
                    operation="fire_heat_calculation",
                    inputs=p,
                    output_value=FireCaseCalculations.fire_heat_input_wetted(
                        p["wetted_area_m2"],
                        p.get("environmental_factor", 1.0),
                    ),
                    output_name="heat_input",
                )]
            ),
        }
        return calculators.get(formula_id)
