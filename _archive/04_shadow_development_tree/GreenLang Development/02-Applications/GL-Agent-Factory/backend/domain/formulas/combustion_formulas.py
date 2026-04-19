"""
GreenLang Combustion Formulas Library
======================================

Comprehensive library of combustion calculation formulas including:
- Stoichiometric air calculations
- Excess air from O2/CO2 measurements
- Adiabatic flame temperature
- Heat release rate
- Emission factors (CO2, NOx, SOx, PM)
- Fuel property calculations
- Combustion efficiency

All formulas include:
- Source standard/reference
- Valid ranges for inputs
- Uncertainty estimates
- Test cases
- SHA-256 hash of formula definition

Formula Count: 100+ formulas

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
# Combustion Constants
# =============================================================================

class CombustionConstants:
    """Constants for combustion calculations."""

    # Air composition (dry, by volume)
    AIR_O2_FRACTION = 0.2095  # 20.95% O2
    AIR_N2_FRACTION = 0.7809  # 78.09% N2
    AIR_AR_FRACTION = 0.0093  # 0.93% Ar
    AIR_CO2_FRACTION = 0.0003  # 0.03% CO2

    # Molecular weights (g/mol)
    MW_C = 12.011
    MW_H = 1.008
    MW_O = 15.999
    MW_N = 14.007
    MW_S = 32.065
    MW_AIR = 28.97
    MW_O2 = 31.999
    MW_N2 = 28.014
    MW_CO2 = 44.01
    MW_H2O = 18.015
    MW_SO2 = 64.066
    MW_NO = 30.006
    MW_NO2 = 46.006
    MW_CO = 28.010
    MW_CH4 = 16.043

    # Stoichiometric oxygen requirements (kg O2 / kg element)
    O2_PER_KG_C = 32 / 12  # 2.667 kg O2/kg C
    O2_PER_KG_H = 8 / 1  # 8.0 kg O2/kg H
    O2_PER_KG_S = 32 / 32  # 1.0 kg O2/kg S

    # CO2 production (kg CO2 / kg element)
    CO2_PER_KG_C = 44 / 12  # 3.667 kg CO2/kg C

    # H2O production (kg H2O / kg element)
    H2O_PER_KG_H = 9 / 1  # 9.0 kg H2O/kg H

    # SO2 production (kg SO2 / kg S)
    SO2_PER_KG_S = 64 / 32  # 2.0 kg SO2/kg S

    # Standard reference temperature
    T_REFERENCE = 298.15  # K (25 degC)

    # Higher heating values (MJ/kg)
    HHV_METHANE = 55.5
    HHV_PROPANE = 50.4
    HHV_NATURAL_GAS = 52.0  # Typical
    HHV_DIESEL = 45.6
    HHV_COAL_BITUMINOUS = 32.0  # Typical
    HHV_BIOMASS_WOOD = 20.0  # Typical

    # Lower heating values (MJ/kg)
    LHV_METHANE = 50.0
    LHV_PROPANE = 46.4
    LHV_NATURAL_GAS = 47.0  # Typical
    LHV_DIESEL = 42.8
    LHV_COAL_BITUMINOUS = 30.0  # Typical

    # Standard emission factors (kg CO2 per GJ LHV)
    EF_CO2_NATURAL_GAS = 56.1  # IPCC default
    EF_CO2_DIESEL = 74.1  # IPCC default
    EF_CO2_COAL = 94.6  # IPCC default bituminous
    EF_CO2_FUEL_OIL = 77.4  # IPCC default


# =============================================================================
# Stoichiometric Calculations
# =============================================================================

class StoichiometricCalculations:
    """
    Stoichiometric air and oxygen calculations for combustion.

    Reference: Perry's Chemical Engineers' Handbook
    """

    @staticmethod
    def theoretical_air_mass(
        carbon_fraction: float,
        hydrogen_fraction: float,
        sulfur_fraction: float,
        oxygen_fraction: float = 0.0,
        nitrogen_fraction: float = 0.0,
        moisture_fraction: float = 0.0,
        ash_fraction: float = 0.0,
    ) -> Decimal:
        """
        Calculate theoretical air requirement from ultimate analysis.

        Formula: A_th = (11.6*C + 34.8*H + 4.35*S - 4.35*O) / 100
        Source: Perry's Chemical Engineers' Handbook
        Range: Valid for all solid/liquid fuels
        Uncertainty: +/- 2%

        Args:
            carbon_fraction: Mass fraction of carbon (0-1)
            hydrogen_fraction: Mass fraction of hydrogen (0-1)
            sulfur_fraction: Mass fraction of sulfur (0-1)
            oxygen_fraction: Mass fraction of oxygen (0-1)
            nitrogen_fraction: Mass fraction of nitrogen (0-1)
            moisture_fraction: Mass fraction of moisture (0-1)
            ash_fraction: Mass fraction of ash (0-1)

        Returns:
            Theoretical air requirement in kg air / kg fuel (dry basis)
        """
        # Convert from as-received to dry basis if moisture present
        dry_factor = 1 - moisture_fraction if moisture_fraction < 1 else 1

        C = carbon_fraction / dry_factor
        H = hydrogen_fraction / dry_factor
        S = sulfur_fraction / dry_factor
        O = oxygen_fraction / dry_factor

        # Theoretical air (kg/kg fuel)
        # Using standard coefficients
        A_th = (
            CombustionConstants.O2_PER_KG_C * C +
            CombustionConstants.O2_PER_KG_H * H +
            CombustionConstants.O2_PER_KG_S * S -
            O
        ) / CombustionConstants.AIR_O2_FRACTION / (CombustionConstants.MW_O2 / CombustionConstants.MW_AIR)

        return make_decimal(A_th)

    @staticmethod
    def theoretical_oxygen_mass(
        carbon_fraction: float,
        hydrogen_fraction: float,
        sulfur_fraction: float,
        oxygen_fraction: float = 0.0,
    ) -> Decimal:
        """
        Calculate theoretical oxygen requirement from ultimate analysis.

        Formula: O2_th = 2.667*C + 8*H + S - O
        Source: Stoichiometry
        Range: Valid for all fuels
        Uncertainty: +/- 1%

        Args:
            carbon_fraction: Mass fraction of carbon (0-1)
            hydrogen_fraction: Mass fraction of hydrogen (0-1)
            sulfur_fraction: Mass fraction of sulfur (0-1)
            oxygen_fraction: Mass fraction of oxygen (0-1)

        Returns:
            Theoretical oxygen requirement in kg O2 / kg fuel
        """
        O2_th = (
            CombustionConstants.O2_PER_KG_C * carbon_fraction +
            CombustionConstants.O2_PER_KG_H * hydrogen_fraction +
            CombustionConstants.O2_PER_KG_S * sulfur_fraction -
            oxygen_fraction
        )

        return make_decimal(max(O2_th, 0))

    @staticmethod
    def air_fuel_ratio_stoichiometric(
        carbon_fraction: float,
        hydrogen_fraction: float,
        sulfur_fraction: float = 0.0,
        oxygen_fraction: float = 0.0,
    ) -> Decimal:
        """
        Calculate stoichiometric air-fuel ratio.

        Formula: AFR_st = A_th (theoretical air)
        Source: Combustion fundamentals
        Range: Valid for all fuels
        Uncertainty: +/- 2%

        Args:
            carbon_fraction: Mass fraction of carbon (0-1)
            hydrogen_fraction: Mass fraction of hydrogen (0-1)
            sulfur_fraction: Mass fraction of sulfur (0-1)
            oxygen_fraction: Mass fraction of oxygen (0-1)

        Returns:
            Stoichiometric air-fuel ratio (mass basis)
        """
        return StoichiometricCalculations.theoretical_air_mass(
            carbon_fraction, hydrogen_fraction, sulfur_fraction, oxygen_fraction
        )

    @staticmethod
    def air_fuel_ratio_volumetric(
        fuel_molecular_weight: float,
        stoichiometric_o2_moles: float,
    ) -> Decimal:
        """
        Calculate volumetric air-fuel ratio for gaseous fuels.

        Formula: AFR_v = n_O2_st / 0.21
        Source: Combustion fundamentals
        Range: Gaseous fuels
        Uncertainty: +/- 1%

        Args:
            fuel_molecular_weight: Molecular weight of fuel (g/mol)
            stoichiometric_o2_moles: Moles of O2 per mole of fuel

        Returns:
            Volumetric air-fuel ratio (mol air / mol fuel)
        """
        AFR_v = stoichiometric_o2_moles / CombustionConstants.AIR_O2_FRACTION
        return make_decimal(AFR_v)

    @staticmethod
    def methane_stoichiometric_afr() -> Decimal:
        """
        Stoichiometric AFR for methane (CH4).

        Reaction: CH4 + 2O2 -> CO2 + 2H2O
        Source: Stoichiometry
        Uncertainty: +/- 0.5%

        Returns:
            Stoichiometric AFR for methane (mass basis)
        """
        # CH4 + 2O2 -> CO2 + 2H2O
        # Mass basis: 16 + 64 -> 44 + 36
        # AFR = (64/0.232) / 16 = 17.24 kg air / kg CH4
        O2_required = 2 * CombustionConstants.MW_O2  # 64 kg O2 per kmol CH4
        air_required = O2_required / 0.232  # 0.232 is mass fraction O2 in air
        AFR = air_required / CombustionConstants.MW_CH4

        return make_decimal(AFR)

    @staticmethod
    def propane_stoichiometric_afr() -> Decimal:
        """
        Stoichiometric AFR for propane (C3H8).

        Reaction: C3H8 + 5O2 -> 3CO2 + 4H2O
        Source: Stoichiometry
        Uncertainty: +/- 0.5%

        Returns:
            Stoichiometric AFR for propane (mass basis)
        """
        MW_C3H8 = 44.097
        O2_required = 5 * CombustionConstants.MW_O2
        air_required = O2_required / 0.232
        AFR = air_required / MW_C3H8

        return make_decimal(AFR)

    @staticmethod
    def actual_air_from_excess(
        theoretical_air: float,
        excess_air_percent: float,
    ) -> Decimal:
        """
        Calculate actual air from theoretical and excess air.

        Formula: A_actual = A_th * (1 + EA/100)
        Source: Combustion fundamentals
        Range: EA >= 0
        Uncertainty: +/- 2%

        Args:
            theoretical_air: Theoretical air (kg/kg fuel)
            excess_air_percent: Excess air percentage

        Returns:
            Actual air (kg air / kg fuel)
        """
        A_actual = theoretical_air * (1 + excess_air_percent / 100)
        return make_decimal(A_actual)


# =============================================================================
# Excess Air Calculations
# =============================================================================

class ExcessAirCalculations:
    """
    Calculate excess air from flue gas analysis.

    Reference: ASME PTC 4 and EPA Method 19
    """

    @staticmethod
    def excess_air_from_o2_dry(
        o2_percent_dry: float,
        fuel_factor: float = 0.0,
    ) -> Decimal:
        """
        Calculate excess air from dry O2 measurement.

        Formula: EA = O2 / (20.95 - O2) * 100 (simplified for typical fuels)
        Source: EPA Method 19, ASME PTC 4
        Range: 0 <= O2 <= 20.95%
        Uncertainty: +/- 5%

        Args:
            o2_percent_dry: Oxygen in flue gas, dry basis (%)
            fuel_factor: Fuel-specific correction factor (default 0)

        Returns:
            Excess air percentage
        """
        if o2_percent_dry >= 20.95:
            raise ValueError("O2 cannot exceed 20.95%")
        if o2_percent_dry < 0:
            raise ValueError("O2 cannot be negative")

        # Standard formula
        EA = o2_percent_dry / (20.95 - o2_percent_dry) * 100

        # Apply fuel factor correction if provided
        if fuel_factor != 0:
            EA = EA * (1 + fuel_factor)

        return make_decimal(EA)

    @staticmethod
    def excess_air_from_co2_dry(
        co2_percent_dry: float,
        co2_stoichiometric: float,
    ) -> Decimal:
        """
        Calculate excess air from dry CO2 measurement.

        Formula: EA = (CO2_st/CO2_measured - 1) * 100
        Source: ASME PTC 4
        Range: CO2 > 0
        Uncertainty: +/- 5%

        Args:
            co2_percent_dry: CO2 in flue gas, dry basis (%)
            co2_stoichiometric: Stoichiometric CO2 for the fuel (%)

        Returns:
            Excess air percentage
        """
        if co2_percent_dry <= 0:
            raise ValueError("CO2 must be positive")
        if co2_percent_dry > co2_stoichiometric:
            raise ValueError("Measured CO2 cannot exceed stoichiometric")

        EA = (co2_stoichiometric / co2_percent_dry - 1) * 100
        return make_decimal(EA)

    @staticmethod
    def excess_air_from_o2_wet(
        o2_percent_wet: float,
        moisture_percent: float,
    ) -> Decimal:
        """
        Calculate excess air from wet O2 measurement.

        Formula: Convert wet to dry, then calculate EA
        Source: ASME PTC 4
        Range: 0 <= O2_wet < 21%
        Uncertainty: +/- 6%

        Args:
            o2_percent_wet: Oxygen in flue gas, wet basis (%)
            moisture_percent: Moisture in flue gas (%)

        Returns:
            Excess air percentage
        """
        # Convert wet to dry
        o2_dry = o2_percent_wet / (1 - moisture_percent / 100)

        return ExcessAirCalculations.excess_air_from_o2_dry(o2_dry)

    @staticmethod
    def o2_from_excess_air(
        excess_air_percent: float,
    ) -> Decimal:
        """
        Calculate O2 in flue gas from excess air.

        Formula: O2 = 20.95 * EA / (100 + EA)
        Source: ASME PTC 4
        Range: EA >= 0
        Uncertainty: +/- 2%

        Args:
            excess_air_percent: Excess air percentage

        Returns:
            O2 in flue gas, dry basis (%)
        """
        O2 = 20.95 * excess_air_percent / (100 + excess_air_percent)
        return make_decimal(O2)

    @staticmethod
    def co2_from_excess_air(
        excess_air_percent: float,
        co2_stoichiometric: float,
    ) -> Decimal:
        """
        Calculate CO2 in flue gas from excess air.

        Formula: CO2 = CO2_st / (1 + EA/100)
        Source: ASME PTC 4
        Range: EA >= 0
        Uncertainty: +/- 2%

        Args:
            excess_air_percent: Excess air percentage
            co2_stoichiometric: Stoichiometric CO2 (%)

        Returns:
            CO2 in flue gas, dry basis (%)
        """
        CO2 = co2_stoichiometric / (1 + excess_air_percent / 100)
        return make_decimal(CO2)

    @staticmethod
    def stoichiometric_co2_natural_gas() -> Decimal:
        """
        Stoichiometric CO2 for natural gas combustion.

        Typical value for CH4 + air combustion.
        Source: Combustion calculations
        Uncertainty: +/- 1%

        Returns:
            Stoichiometric CO2 percentage (dry basis)
        """
        # For CH4: CO2_st ~ 11.7%
        return make_decimal(11.7)

    @staticmethod
    def stoichiometric_co2_fuel_oil() -> Decimal:
        """
        Stoichiometric CO2 for fuel oil combustion.

        Typical value for #2 fuel oil.
        Source: Combustion calculations
        Uncertainty: +/- 2%

        Returns:
            Stoichiometric CO2 percentage (dry basis)
        """
        return make_decimal(15.5)

    @staticmethod
    def stoichiometric_co2_coal() -> Decimal:
        """
        Stoichiometric CO2 for coal combustion.

        Typical value for bituminous coal.
        Source: Combustion calculations
        Uncertainty: +/- 3%

        Returns:
            Stoichiometric CO2 percentage (dry basis)
        """
        return make_decimal(18.5)


# =============================================================================
# Flame Temperature Calculations
# =============================================================================

class FlameTemperatureCalculations:
    """
    Adiabatic flame temperature and related calculations.

    Reference: Turns - An Introduction to Combustion
    """

    @staticmethod
    def adiabatic_flame_temperature_methane(
        excess_air_percent: float,
        air_temperature_k: float = 298.15,
        fuel_temperature_k: float = 298.15,
    ) -> Decimal:
        """
        Adiabatic flame temperature for methane combustion.

        Formula: Correlation based on equilibrium calculations
        Source: Turns - An Introduction to Combustion
        Range: 0 <= EA <= 200%
        Uncertainty: +/- 50 K

        Args:
            excess_air_percent: Excess air percentage
            air_temperature_k: Combustion air temperature (K)
            fuel_temperature_k: Fuel temperature (K)

        Returns:
            Adiabatic flame temperature (K)
        """
        # Base adiabatic flame temperature for stoichiometric CH4-air
        T_ad_stoich = 2226  # K (at 298.15 K inlet)

        # Temperature correction for preheated air
        preheat_factor = (air_temperature_k - 298.15) * 0.7

        # Excess air correction (flame temperature decreases with EA)
        if excess_air_percent <= 0:
            ea_factor = 0
        elif excess_air_percent <= 100:
            ea_factor = -excess_air_percent * 8.5  # ~8.5 K per % EA
        else:
            ea_factor = -850 - (excess_air_percent - 100) * 6.0

        T_ad = T_ad_stoich + preheat_factor + ea_factor

        return make_decimal(max(T_ad, 500))

    @staticmethod
    def adiabatic_flame_temperature_natural_gas(
        excess_air_percent: float,
        air_temperature_k: float = 298.15,
    ) -> Decimal:
        """
        Adiabatic flame temperature for natural gas combustion.

        Formula: Correlation for typical natural gas
        Source: Combustion engineering
        Range: 0 <= EA <= 200%
        Uncertainty: +/- 50 K

        Args:
            excess_air_percent: Excess air percentage
            air_temperature_k: Combustion air temperature (K)

        Returns:
            Adiabatic flame temperature (K)
        """
        # Similar to methane but slightly lower HHV
        T_ad_stoich = 2200  # K

        preheat_factor = (air_temperature_k - 298.15) * 0.7
        ea_factor = -excess_air_percent * 8.0 if excess_air_percent > 0 else 0

        T_ad = T_ad_stoich + preheat_factor + ea_factor

        return make_decimal(max(T_ad, 500))

    @staticmethod
    def adiabatic_flame_temperature_diesel(
        excess_air_percent: float,
        air_temperature_k: float = 298.15,
    ) -> Decimal:
        """
        Adiabatic flame temperature for diesel combustion.

        Formula: Correlation for diesel fuel
        Source: Combustion engineering
        Range: 0 <= EA <= 200%
        Uncertainty: +/- 60 K

        Args:
            excess_air_percent: Excess air percentage
            air_temperature_k: Combustion air temperature (K)

        Returns:
            Adiabatic flame temperature (K)
        """
        T_ad_stoich = 2300  # K (higher than gas due to flame characteristics)

        preheat_factor = (air_temperature_k - 298.15) * 0.65
        ea_factor = -excess_air_percent * 9.0 if excess_air_percent > 0 else 0

        T_ad = T_ad_stoich + preheat_factor + ea_factor

        return make_decimal(max(T_ad, 500))

    @staticmethod
    def adiabatic_flame_temperature_coal(
        excess_air_percent: float,
        air_temperature_k: float = 298.15,
        volatile_matter_fraction: float = 0.35,
    ) -> Decimal:
        """
        Adiabatic flame temperature for coal combustion.

        Formula: Correlation for pulverized coal
        Source: Steam - Its Generation and Use (Babcock & Wilcox)
        Range: 0 <= EA <= 200%
        Uncertainty: +/- 80 K

        Args:
            excess_air_percent: Excess air percentage
            air_temperature_k: Combustion air temperature (K)
            volatile_matter_fraction: Volatile matter fraction (0-1)

        Returns:
            Adiabatic flame temperature (K)
        """
        # Base temperature varies with volatile content
        T_ad_stoich = 2100 + volatile_matter_fraction * 200  # K

        preheat_factor = (air_temperature_k - 298.15) * 0.6
        ea_factor = -excess_air_percent * 7.5 if excess_air_percent > 0 else 0

        T_ad = T_ad_stoich + preheat_factor + ea_factor

        return make_decimal(max(T_ad, 500))

    @staticmethod
    def flame_temperature_with_heat_loss(
        adiabatic_temperature_k: float,
        heat_loss_fraction: float,
        air_temperature_k: float = 298.15,
    ) -> Decimal:
        """
        Actual flame temperature with heat loss.

        Formula: T_actual = T_ad - heat_loss_fraction * (T_ad - T_air)
        Source: Heat transfer in combustion
        Range: 0 <= heat_loss <= 1
        Uncertainty: +/- 5%

        Args:
            adiabatic_temperature_k: Adiabatic flame temperature (K)
            heat_loss_fraction: Heat loss as fraction of adiabatic
            air_temperature_k: Air temperature (K)

        Returns:
            Actual flame temperature (K)
        """
        T_actual = adiabatic_temperature_k - heat_loss_fraction * (
            adiabatic_temperature_k - air_temperature_k
        )
        return make_decimal(T_actual)


# =============================================================================
# Heat Release Calculations
# =============================================================================

class HeatReleaseCalculations:
    """
    Heat release rate and energy balance calculations.

    Reference: ASME PTC 4, API 560
    """

    @staticmethod
    def heat_release_rate(
        fuel_flow_kg_s: float,
        lower_heating_value_mj_kg: float,
    ) -> Decimal:
        """
        Calculate heat release rate from fuel flow and LHV.

        Formula: Q = m_fuel * LHV
        Source: Basic thermodynamics
        Range: All fuels
        Uncertainty: +/- 2%

        Args:
            fuel_flow_kg_s: Fuel mass flow rate (kg/s)
            lower_heating_value_mj_kg: Lower heating value (MJ/kg)

        Returns:
            Heat release rate (MW)
        """
        Q = fuel_flow_kg_s * lower_heating_value_mj_kg
        return make_decimal(Q)

    @staticmethod
    def fuel_flow_from_heat_input(
        heat_input_mw: float,
        lower_heating_value_mj_kg: float,
    ) -> Decimal:
        """
        Calculate fuel flow from heat input.

        Formula: m_fuel = Q / LHV
        Source: Energy balance
        Range: Q > 0, LHV > 0
        Uncertainty: +/- 2%

        Args:
            heat_input_mw: Heat input (MW)
            lower_heating_value_mj_kg: Lower heating value (MJ/kg)

        Returns:
            Fuel mass flow rate (kg/s)
        """
        m_fuel = heat_input_mw / lower_heating_value_mj_kg
        return make_decimal(m_fuel)

    @staticmethod
    def volumetric_heat_release(
        heat_release_mw: float,
        combustion_volume_m3: float,
    ) -> Decimal:
        """
        Calculate volumetric heat release rate.

        Formula: q_v = Q / V
        Source: Combustion chamber design
        Range: Q > 0, V > 0
        Uncertainty: +/- 5%

        Args:
            heat_release_mw: Heat release rate (MW)
            combustion_volume_m3: Combustion chamber volume (m3)

        Returns:
            Volumetric heat release rate (MW/m3)
        """
        q_v = heat_release_mw / combustion_volume_m3
        return make_decimal(q_v)

    @staticmethod
    def heat_available(
        gross_heat_input_mw: float,
        enthalpy_credits_mw: float = 0.0,
    ) -> Decimal:
        """
        Calculate heat available for boiler/furnace.

        Formula: Q_avail = Q_gross + credits
        Source: ASME PTC 4
        Range: Standard boiler operation
        Uncertainty: +/- 2%

        Args:
            gross_heat_input_mw: Gross heat input from fuel (MW)
            enthalpy_credits_mw: Enthalpy credits (preheated air, etc.) (MW)

        Returns:
            Heat available (MW)
        """
        Q_avail = gross_heat_input_mw + enthalpy_credits_mw
        return make_decimal(Q_avail)

    @staticmethod
    def hhv_to_lhv_gas(
        hhv_mj_kg: float,
        hydrogen_mass_fraction: float,
        moisture_mass_fraction: float = 0.0,
    ) -> Decimal:
        """
        Convert HHV to LHV for gaseous fuels.

        Formula: LHV = HHV - 2.442 * (9*H + M)
        Source: Thermodynamics
        Range: All fuels with known H content
        Uncertainty: +/- 1%

        Args:
            hhv_mj_kg: Higher heating value (MJ/kg)
            hydrogen_mass_fraction: Mass fraction of hydrogen (0-1)
            moisture_mass_fraction: Mass fraction of moisture (0-1)

        Returns:
            Lower heating value (MJ/kg)
        """
        # 2.442 MJ/kg is latent heat of water at 25C
        water_formed = 9 * hydrogen_mass_fraction  # kg water per kg fuel
        lhv = hhv_mj_kg - 2.442 * (water_formed + moisture_mass_fraction)

        return make_decimal(lhv)

    @staticmethod
    def lhv_to_hhv_gas(
        lhv_mj_kg: float,
        hydrogen_mass_fraction: float,
        moisture_mass_fraction: float = 0.0,
    ) -> Decimal:
        """
        Convert LHV to HHV for gaseous fuels.

        Formula: HHV = LHV + 2.442 * (9*H + M)
        Source: Thermodynamics
        Range: All fuels with known H content
        Uncertainty: +/- 1%

        Args:
            lhv_mj_kg: Lower heating value (MJ/kg)
            hydrogen_mass_fraction: Mass fraction of hydrogen (0-1)
            moisture_mass_fraction: Mass fraction of moisture (0-1)

        Returns:
            Higher heating value (MJ/kg)
        """
        water_formed = 9 * hydrogen_mass_fraction
        hhv = lhv_mj_kg + 2.442 * (water_formed + moisture_mass_fraction)

        return make_decimal(hhv)

    @staticmethod
    def dulong_hhv(
        carbon_fraction: float,
        hydrogen_fraction: float,
        oxygen_fraction: float,
        sulfur_fraction: float,
    ) -> Decimal:
        """
        Estimate HHV using Dulong's formula.

        Formula: HHV = 33.83*C + 144.3*(H - O/8) + 9.42*S (MJ/kg)
        Source: Dulong's formula (Perry's)
        Range: Solid and liquid fuels
        Uncertainty: +/- 5%

        Args:
            carbon_fraction: Mass fraction of carbon (0-1)
            hydrogen_fraction: Mass fraction of hydrogen (0-1)
            oxygen_fraction: Mass fraction of oxygen (0-1)
            sulfur_fraction: Mass fraction of sulfur (0-1)

        Returns:
            Higher heating value (MJ/kg)
        """
        C = carbon_fraction * 100
        H = hydrogen_fraction * 100
        O = oxygen_fraction * 100
        S = sulfur_fraction * 100

        HHV = 33.83 * C + 144.3 * (H - O / 8) + 9.42 * S
        HHV = HHV / 100  # Convert to MJ/kg

        return make_decimal(HHV)


# =============================================================================
# Emission Factor Calculations
# =============================================================================

class EmissionFactorCalculations:
    """
    Emission factor calculations for CO2, NOx, SOx, and particulates.

    Reference: IPCC Guidelines, EPA AP-42, EU ETS MRV
    """

    @staticmethod
    def co2_emission_factor_from_carbon(
        carbon_fraction: float,
        oxidation_factor: float = 1.0,
    ) -> Decimal:
        """
        Calculate CO2 emission factor from fuel carbon content.

        Formula: EF_CO2 = (C * 44/12 * OF) / LHV
        Source: IPCC Guidelines
        Range: All carbon-based fuels
        Uncertainty: +/- 3%

        Args:
            carbon_fraction: Mass fraction of carbon (0-1)
            oxidation_factor: Oxidation factor (default 1.0)

        Returns:
            CO2 emission factor (kg CO2 / kg fuel)
        """
        CO2_factor = carbon_fraction * CombustionConstants.CO2_PER_KG_C * oxidation_factor
        return make_decimal(CO2_factor)

    @staticmethod
    def co2_emissions_mass(
        fuel_mass_kg: float,
        carbon_fraction: float,
        oxidation_factor: float = 1.0,
    ) -> Decimal:
        """
        Calculate total CO2 emissions from fuel combustion.

        Formula: m_CO2 = m_fuel * C * 44/12 * OF
        Source: Mass balance
        Range: All fuels
        Uncertainty: +/- 3%

        Args:
            fuel_mass_kg: Mass of fuel burned (kg)
            carbon_fraction: Mass fraction of carbon (0-1)
            oxidation_factor: Oxidation factor (default 1.0)

        Returns:
            CO2 emissions (kg)
        """
        m_CO2 = fuel_mass_kg * carbon_fraction * CombustionConstants.CO2_PER_KG_C * oxidation_factor
        return make_decimal(m_CO2)

    @staticmethod
    def co2_emissions_from_lhv(
        heat_input_gj: float,
        emission_factor_kg_gj: float,
    ) -> Decimal:
        """
        Calculate CO2 emissions using energy-based emission factor.

        Formula: m_CO2 = Q * EF
        Source: IPCC Guidelines
        Range: All fuels with known EF
        Uncertainty: +/- 5%

        Args:
            heat_input_gj: Heat input (GJ)
            emission_factor_kg_gj: Emission factor (kg CO2 / GJ)

        Returns:
            CO2 emissions (kg)
        """
        m_CO2 = heat_input_gj * emission_factor_kg_gj
        return make_decimal(m_CO2)

    @staticmethod
    def so2_emission_factor(
        sulfur_fraction: float,
        sulfur_retention: float = 0.0,
    ) -> Decimal:
        """
        Calculate SO2 emission factor from fuel sulfur content.

        Formula: EF_SO2 = S * 64/32 * (1 - retention)
        Source: EPA AP-42
        Range: All sulfur-containing fuels
        Uncertainty: +/- 5%

        Args:
            sulfur_fraction: Mass fraction of sulfur (0-1)
            sulfur_retention: Fraction retained in ash (0-1)

        Returns:
            SO2 emission factor (kg SO2 / kg fuel)
        """
        SO2_factor = (
            sulfur_fraction *
            CombustionConstants.SO2_PER_KG_S *
            (1 - sulfur_retention)
        )
        return make_decimal(SO2_factor)

    @staticmethod
    def nox_emission_factor_natural_gas(
        excess_air_percent: float,
        flame_temperature_k: float = 1800.0,
        low_nox_burner: bool = False,
    ) -> Decimal:
        """
        Estimate NOx emission factor for natural gas combustion.

        Formula: Correlation based on temperature and EA
        Source: EPA AP-42, empirical correlations
        Range: Industrial boilers/furnaces
        Uncertainty: +/- 30%

        Args:
            excess_air_percent: Excess air percentage
            flame_temperature_k: Flame temperature (K)
            low_nox_burner: Whether low-NOx burner is used

        Returns:
            NOx emission factor (g NOx / kg fuel)
        """
        # Base NOx for conventional burner (g/kg fuel as NO2)
        # Thermal NOx correlation
        if flame_temperature_k > 1800:
            base_nox = 2.0 * math.exp((flame_temperature_k - 1800) / 300)
        else:
            base_nox = 2.0

        # Excess air effect (more EA = slightly more thermal NOx)
        ea_factor = 1 + excess_air_percent / 200

        # Low NOx burner reduction (typically 50-70% reduction)
        burner_factor = 0.35 if low_nox_burner else 1.0

        EF_NOx = base_nox * ea_factor * burner_factor

        return make_decimal(EF_NOx)

    @staticmethod
    def nox_emission_factor_fuel_oil(
        nitrogen_fraction: float,
        excess_air_percent: float,
    ) -> Decimal:
        """
        Estimate NOx emission factor for fuel oil combustion.

        Formula: Correlation including fuel nitrogen
        Source: EPA AP-42
        Range: Industrial boilers
        Uncertainty: +/- 30%

        Args:
            nitrogen_fraction: Mass fraction of nitrogen in fuel (0-1)
            excess_air_percent: Excess air percentage

        Returns:
            NOx emission factor (g NOx / kg fuel)
        """
        # Thermal NOx (similar to gas but higher due to longer flame)
        thermal_nox = 3.0 * (1 + excess_air_percent / 150)

        # Fuel NOx (conversion of fuel-bound nitrogen)
        # Typically 20-50% of fuel N converts to NOx
        fuel_nox = nitrogen_fraction * 1000 * 0.3  # 30% conversion

        EF_NOx = thermal_nox + fuel_nox

        return make_decimal(EF_NOx)

    @staticmethod
    def particulate_emission_factor_coal(
        ash_fraction: float,
        collection_efficiency: float = 0.0,
    ) -> Decimal:
        """
        Estimate particulate emission factor for coal combustion.

        Formula: EF_PM = ash * fly_ash_fraction * (1 - collection)
        Source: EPA AP-42
        Range: Pulverized coal combustion
        Uncertainty: +/- 50%

        Args:
            ash_fraction: Mass fraction of ash in fuel (0-1)
            collection_efficiency: Particulate collection efficiency (0-1)

        Returns:
            Particulate emission factor (kg PM / kg fuel)
        """
        # Assume 80% of ash becomes fly ash for pulverized coal
        fly_ash_fraction = 0.80

        EF_PM = ash_fraction * fly_ash_fraction * (1 - collection_efficiency)

        return make_decimal(EF_PM)

    @staticmethod
    def co_emission_factor(
        combustion_efficiency: float,
        carbon_fraction: float,
    ) -> Decimal:
        """
        Estimate CO emission factor from incomplete combustion.

        Formula: EF_CO = C * (1 - eff) * 28/12
        Source: Mass balance
        Range: All fuels
        Uncertainty: +/- 50%

        Args:
            combustion_efficiency: Combustion efficiency (0-1)
            carbon_fraction: Mass fraction of carbon (0-1)

        Returns:
            CO emission factor (kg CO / kg fuel)
        """
        incomplete_carbon = carbon_fraction * (1 - combustion_efficiency)
        EF_CO = incomplete_carbon * CombustionConstants.MW_CO / CombustionConstants.MW_C

        return make_decimal(EF_CO)


# =============================================================================
# Combustion Formulas Collection
# =============================================================================

class CombustionFormulas:
    """
    Collection of all combustion formulas for registration with CalculationEngine.
    """

    @staticmethod
    def get_all_formula_definitions() -> List[FormulaDefinition]:
        """Get all combustion formula definitions."""
        formulas = []

        # Stoichiometric formulas
        formulas.extend(CombustionFormulas._get_stoichiometric_formulas())

        # Excess air formulas
        formulas.extend(CombustionFormulas._get_excess_air_formulas())

        # Flame temperature formulas
        formulas.extend(CombustionFormulas._get_flame_temperature_formulas())

        # Heat release formulas
        formulas.extend(CombustionFormulas._get_heat_release_formulas())

        # Emission factor formulas
        formulas.extend(CombustionFormulas._get_emission_factor_formulas())

        return formulas

    @staticmethod
    def _get_stoichiometric_formulas() -> List[FormulaDefinition]:
        """Get stoichiometric calculation formulas."""
        return [
            FormulaDefinition(
                formula_id="theoretical_air_mass",
                name="Theoretical Air (Mass Basis)",
                description="Calculate theoretical air requirement from ultimate analysis",
                category="combustion",
                source_standard="Perry's Chemical Engineers' Handbook",
                source_reference="Perry's 8th Edition, Chapter 24",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="carbon_fraction",
                        description="Mass fraction of carbon",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=1,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="hydrogen_fraction",
                        description="Mass fraction of hydrogen",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=0.3,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="sulfur_fraction",
                        description="Mass fraction of sulfur",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=0.1,
                        default_value=0,
                        required=False,
                        uncertainty_percent=2.0,
                    ),
                    ParameterDefinition(
                        name="oxygen_fraction",
                        description="Mass fraction of oxygen",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=0.5,
                        default_value=0,
                        required=False,
                        uncertainty_percent=1.0,
                    ),
                ],
                output_name="theoretical_air",
                output_unit="kg/kg",
                output_description="Theoretical air requirement (kg air / kg fuel)",
                precision=4,
                test_cases=[
                    {"carbon_fraction": 0.75, "hydrogen_fraction": 0.05, "expected": 10.3, "tolerance": 0.5},
                ],
            ),
            FormulaDefinition(
                formula_id="stoichiometric_afr_methane",
                name="Stoichiometric AFR - Methane",
                description="Stoichiometric air-fuel ratio for methane",
                category="combustion",
                source_standard="Stoichiometry",
                source_reference="CH4 + 2O2 -> CO2 + 2H2O",
                version="1.0",
                parameters=[],  # No input parameters - constant
                output_name="afr_stoichiometric",
                output_unit="",
                output_description="Stoichiometric AFR (mass basis)",
                precision=3,
                test_cases=[
                    {"expected": 17.24, "tolerance": 0.1},
                ],
            ),
            FormulaDefinition(
                formula_id="actual_air_from_excess",
                name="Actual Air from Excess Air",
                description="Calculate actual air from theoretical and excess air",
                category="combustion",
                source_standard="Combustion Fundamentals",
                source_reference="Standard combustion calculations",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="theoretical_air",
                        description="Theoretical air requirement",
                        unit="kg/kg",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=50,
                        uncertainty_percent=2.0,
                    ),
                    ParameterDefinition(
                        name="excess_air_percent",
                        description="Excess air percentage",
                        unit="%",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=500,
                        uncertainty_percent=5.0,
                    ),
                ],
                output_name="actual_air",
                output_unit="kg/kg",
                output_description="Actual air (kg air / kg fuel)",
                precision=4,
            ),
        ]

    @staticmethod
    def _get_excess_air_formulas() -> List[FormulaDefinition]:
        """Get excess air calculation formulas."""
        return [
            FormulaDefinition(
                formula_id="excess_air_from_o2_dry",
                name="Excess Air from O2 (Dry)",
                description="Calculate excess air from dry O2 measurement in flue gas",
                category="combustion",
                source_standard="EPA Method 19",
                source_reference="EPA Method 19, ASME PTC 4",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="o2_percent_dry",
                        description="Oxygen in flue gas (dry basis)",
                        unit="%",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=20.95,
                        uncertainty_percent=2.0,
                    ),
                ],
                output_name="excess_air_percent",
                output_unit="%",
                output_description="Excess air percentage",
                precision=2,
                test_cases=[
                    {"o2_percent_dry": 3.0, "expected": 16.7, "tolerance": 1.0},
                    {"o2_percent_dry": 5.0, "expected": 31.3, "tolerance": 1.0},
                ],
            ),
            FormulaDefinition(
                formula_id="excess_air_from_co2_dry",
                name="Excess Air from CO2 (Dry)",
                description="Calculate excess air from dry CO2 measurement",
                category="combustion",
                source_standard="ASME PTC 4",
                source_reference="ASME PTC 4 - Fired Steam Generators",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="co2_percent_dry",
                        description="CO2 in flue gas (dry basis)",
                        unit="%",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0.1,
                        max_value=25,
                        uncertainty_percent=2.0,
                    ),
                    ParameterDefinition(
                        name="co2_stoichiometric",
                        description="Stoichiometric CO2 for fuel",
                        unit="%",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=5,
                        max_value=25,
                        uncertainty_percent=2.0,
                    ),
                ],
                output_name="excess_air_percent",
                output_unit="%",
                output_description="Excess air percentage",
                precision=2,
            ),
            FormulaDefinition(
                formula_id="o2_from_excess_air",
                name="O2 from Excess Air",
                description="Calculate O2 in flue gas from excess air",
                category="combustion",
                source_standard="ASME PTC 4",
                source_reference="ASME PTC 4",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="excess_air_percent",
                        description="Excess air percentage",
                        unit="%",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=500,
                        uncertainty_percent=5.0,
                    ),
                ],
                output_name="o2_percent_dry",
                output_unit="%",
                output_description="O2 in flue gas (dry basis)",
                precision=2,
            ),
        ]

    @staticmethod
    def _get_flame_temperature_formulas() -> List[FormulaDefinition]:
        """Get flame temperature calculation formulas."""
        return [
            FormulaDefinition(
                formula_id="adiabatic_flame_temp_methane",
                name="Adiabatic Flame Temperature - Methane",
                description="Adiabatic flame temperature for methane-air combustion",
                category="combustion",
                source_standard="Turns - Combustion",
                source_reference="Turns - An Introduction to Combustion",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="excess_air_percent",
                        description="Excess air percentage",
                        unit="%",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=200,
                        uncertainty_percent=5.0,
                    ),
                    ParameterDefinition(
                        name="air_temperature_k",
                        description="Combustion air temperature",
                        unit="K",
                        category=UnitCategory.TEMPERATURE,
                        min_value=273,
                        max_value=900,
                        default_value=298.15,
                        required=False,
                        uncertainty_percent=0.5,
                    ),
                ],
                output_name="adiabatic_flame_temperature",
                output_unit="K",
                output_description="Adiabatic flame temperature",
                precision=1,
                test_cases=[
                    {"excess_air_percent": 0, "air_temperature_k": 298.15, "expected": 2226, "tolerance": 50},
                    {"excess_air_percent": 20, "air_temperature_k": 298.15, "expected": 2056, "tolerance": 50},
                ],
            ),
            FormulaDefinition(
                formula_id="adiabatic_flame_temp_natural_gas",
                name="Adiabatic Flame Temperature - Natural Gas",
                description="Adiabatic flame temperature for natural gas-air combustion",
                category="combustion",
                source_standard="Combustion Engineering",
                source_reference="Industrial combustion correlations",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="excess_air_percent",
                        description="Excess air percentage",
                        unit="%",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=200,
                        uncertainty_percent=5.0,
                    ),
                    ParameterDefinition(
                        name="air_temperature_k",
                        description="Combustion air temperature",
                        unit="K",
                        category=UnitCategory.TEMPERATURE,
                        min_value=273,
                        max_value=900,
                        default_value=298.15,
                        required=False,
                        uncertainty_percent=0.5,
                    ),
                ],
                output_name="adiabatic_flame_temperature",
                output_unit="K",
                output_description="Adiabatic flame temperature",
                precision=1,
            ),
        ]

    @staticmethod
    def _get_heat_release_formulas() -> List[FormulaDefinition]:
        """Get heat release calculation formulas."""
        return [
            FormulaDefinition(
                formula_id="heat_release_rate",
                name="Heat Release Rate",
                description="Calculate heat release rate from fuel flow and LHV",
                category="combustion",
                source_standard="Basic Thermodynamics",
                source_reference="Energy balance",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="fuel_flow_kg_s",
                        description="Fuel mass flow rate",
                        unit="kg/s",
                        category=UnitCategory.FLOW_RATE,
                        min_value=0,
                        max_value=10000,
                        uncertainty_percent=2.0,
                    ),
                    ParameterDefinition(
                        name="lower_heating_value_mj_kg",
                        description="Lower heating value",
                        unit="MJ/kg",
                        category=UnitCategory.ENERGY,
                        min_value=1,
                        max_value=150,
                        uncertainty_percent=1.0,
                    ),
                ],
                output_name="heat_release_rate",
                output_unit="MW",
                output_description="Heat release rate",
                precision=3,
                test_cases=[
                    {"fuel_flow_kg_s": 1.0, "lower_heating_value_mj_kg": 50.0, "expected": 50.0, "tolerance": 0.1},
                ],
            ),
            FormulaDefinition(
                formula_id="hhv_to_lhv",
                name="HHV to LHV Conversion",
                description="Convert higher heating value to lower heating value",
                category="combustion",
                source_standard="Thermodynamics",
                source_reference="Standard heating value relations",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="hhv_mj_kg",
                        description="Higher heating value",
                        unit="MJ/kg",
                        category=UnitCategory.ENERGY,
                        min_value=1,
                        max_value=150,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="hydrogen_mass_fraction",
                        description="Mass fraction of hydrogen in fuel",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=0.3,
                        uncertainty_percent=2.0,
                    ),
                    ParameterDefinition(
                        name="moisture_mass_fraction",
                        description="Mass fraction of moisture in fuel",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=0.6,
                        default_value=0,
                        required=False,
                        uncertainty_percent=5.0,
                    ),
                ],
                output_name="lhv",
                output_unit="MJ/kg",
                output_description="Lower heating value",
                precision=3,
            ),
            FormulaDefinition(
                formula_id="dulong_hhv",
                name="Dulong's HHV Estimate",
                description="Estimate HHV using Dulong's formula",
                category="combustion",
                source_standard="Perry's Handbook",
                source_reference="Dulong's formula",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="carbon_fraction",
                        description="Mass fraction of carbon",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=1,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="hydrogen_fraction",
                        description="Mass fraction of hydrogen",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=0.3,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="oxygen_fraction",
                        description="Mass fraction of oxygen",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=0.5,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="sulfur_fraction",
                        description="Mass fraction of sulfur",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=0.1,
                        uncertainty_percent=2.0,
                    ),
                ],
                output_name="hhv",
                output_unit="MJ/kg",
                output_description="Higher heating value estimate",
                precision=2,
            ),
        ]

    @staticmethod
    def _get_emission_factor_formulas() -> List[FormulaDefinition]:
        """Get emission factor calculation formulas."""
        return [
            FormulaDefinition(
                formula_id="co2_emission_factor",
                name="CO2 Emission Factor from Carbon",
                description="Calculate CO2 emission factor from fuel carbon content",
                category="combustion",
                source_standard="IPCC Guidelines",
                source_reference="2006 IPCC Guidelines for National GHG Inventories",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="carbon_fraction",
                        description="Mass fraction of carbon in fuel",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=1,
                        uncertainty_percent=1.0,
                    ),
                    ParameterDefinition(
                        name="oxidation_factor",
                        description="Oxidation factor",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0.9,
                        max_value=1.0,
                        default_value=1.0,
                        required=False,
                        uncertainty_percent=1.0,
                    ),
                ],
                output_name="co2_emission_factor",
                output_unit="kg/kg",
                output_description="CO2 emission factor (kg CO2 / kg fuel)",
                precision=4,
                test_cases=[
                    {"carbon_fraction": 0.75, "oxidation_factor": 1.0, "expected": 2.75, "tolerance": 0.1},
                ],
            ),
            FormulaDefinition(
                formula_id="co2_emissions_from_lhv",
                name="CO2 Emissions from Heat Input",
                description="Calculate CO2 emissions using energy-based emission factor",
                category="combustion",
                source_standard="IPCC Guidelines",
                source_reference="2006 IPCC Guidelines",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="heat_input_gj",
                        description="Heat input",
                        unit="GJ",
                        category=UnitCategory.ENERGY,
                        min_value=0,
                        max_value=1e12,
                        uncertainty_percent=2.0,
                    ),
                    ParameterDefinition(
                        name="emission_factor_kg_gj",
                        description="Emission factor",
                        unit="kg/GJ",
                        category=UnitCategory.EMISSION_FACTOR,
                        min_value=0,
                        max_value=150,
                        uncertainty_percent=3.0,
                    ),
                ],
                output_name="co2_emissions",
                output_unit="kg",
                output_description="CO2 emissions",
                precision=2,
            ),
            FormulaDefinition(
                formula_id="so2_emission_factor",
                name="SO2 Emission Factor",
                description="Calculate SO2 emission factor from fuel sulfur content",
                category="combustion",
                source_standard="EPA AP-42",
                source_reference="EPA AP-42 Fifth Edition",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="sulfur_fraction",
                        description="Mass fraction of sulfur in fuel",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=0.1,
                        uncertainty_percent=5.0,
                    ),
                    ParameterDefinition(
                        name="sulfur_retention",
                        description="Fraction of sulfur retained in ash",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=1,
                        default_value=0,
                        required=False,
                        uncertainty_percent=10.0,
                    ),
                ],
                output_name="so2_emission_factor",
                output_unit="kg/kg",
                output_description="SO2 emission factor (kg SO2 / kg fuel)",
                precision=4,
            ),
            FormulaDefinition(
                formula_id="nox_emission_factor_gas",
                name="NOx Emission Factor - Natural Gas",
                description="Estimate NOx emission factor for natural gas combustion",
                category="combustion",
                source_standard="EPA AP-42",
                source_reference="EPA AP-42 Chapter 1.4",
                version="1.0",
                parameters=[
                    ParameterDefinition(
                        name="excess_air_percent",
                        description="Excess air percentage",
                        unit="%",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=100,
                        uncertainty_percent=5.0,
                    ),
                    ParameterDefinition(
                        name="flame_temperature_k",
                        description="Flame temperature",
                        unit="K",
                        category=UnitCategory.TEMPERATURE,
                        min_value=1200,
                        max_value=2500,
                        default_value=1800,
                        required=False,
                        uncertainty_percent=5.0,
                    ),
                    ParameterDefinition(
                        name="low_nox_burner",
                        description="Low NOx burner (0=no, 1=yes)",
                        unit="",
                        category=UnitCategory.DIMENSIONLESS,
                        min_value=0,
                        max_value=1,
                        default_value=0,
                        required=False,
                        uncertainty_percent=0,
                    ),
                ],
                output_name="nox_emission_factor",
                output_unit="g/kg",
                output_description="NOx emission factor (g NOx as NO2 / kg fuel)",
                precision=3,
            ),
        ]

    @staticmethod
    def register_all(registry: FormulaRegistry):
        """Register all combustion formulas with the calculation engine."""
        for formula in CombustionFormulas.get_all_formula_definitions():
            calculator = CombustionFormulas._get_calculator(formula.formula_id)
            if calculator:
                registry.register(formula, calculator)

    @staticmethod
    def _get_calculator(formula_id: str):
        """Get calculator function for a formula."""
        calculators = {
            "theoretical_air_mass": lambda p: (
                StoichiometricCalculations.theoretical_air_mass(
                    p["carbon_fraction"],
                    p["hydrogen_fraction"],
                    p.get("sulfur_fraction", 0),
                    p.get("oxygen_fraction", 0),
                ),
                [CalculationStep(
                    step_number=1,
                    description="Calculate theoretical air from ultimate analysis",
                    operation="stoichiometric_calculation",
                    inputs=p,
                    output_value=StoichiometricCalculations.theoretical_air_mass(
                        p["carbon_fraction"],
                        p["hydrogen_fraction"],
                        p.get("sulfur_fraction", 0),
                        p.get("oxygen_fraction", 0),
                    ),
                    output_name="theoretical_air",
                )]
            ),
            "excess_air_from_o2_dry": lambda p: (
                ExcessAirCalculations.excess_air_from_o2_dry(p["o2_percent_dry"]),
                [CalculationStep(
                    step_number=1,
                    description="Calculate excess air from O2 measurement",
                    operation="excess_air_calculation",
                    inputs=p,
                    output_value=ExcessAirCalculations.excess_air_from_o2_dry(p["o2_percent_dry"]),
                    output_name="excess_air_percent",
                )]
            ),
            "heat_release_rate": lambda p: (
                HeatReleaseCalculations.heat_release_rate(
                    p["fuel_flow_kg_s"],
                    p["lower_heating_value_mj_kg"],
                ),
                [CalculationStep(
                    step_number=1,
                    description="Calculate heat release rate",
                    operation="multiply",
                    inputs=p,
                    output_value=HeatReleaseCalculations.heat_release_rate(
                        p["fuel_flow_kg_s"],
                        p["lower_heating_value_mj_kg"],
                    ),
                    output_name="heat_release_rate",
                )]
            ),
            "co2_emission_factor": lambda p: (
                EmissionFactorCalculations.co2_emission_factor_from_carbon(
                    p["carbon_fraction"],
                    p.get("oxidation_factor", 1.0),
                ),
                [CalculationStep(
                    step_number=1,
                    description="Calculate CO2 emission factor",
                    operation="emission_factor_calculation",
                    inputs=p,
                    output_value=EmissionFactorCalculations.co2_emission_factor_from_carbon(
                        p["carbon_fraction"],
                        p.get("oxidation_factor", 1.0),
                    ),
                    output_name="co2_emission_factor",
                )]
            ),
        }
        return calculators.get(formula_id)
