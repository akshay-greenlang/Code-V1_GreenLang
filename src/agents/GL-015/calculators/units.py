"""
GL-015 INSULSCAN Unit Conversion Library

This module provides comprehensive unit conversion functions for
industrial insulation thermal analysis calculations. All conversions
are bidirectional and maintain high numerical precision.

Key Features:
- Temperature conversions (Celsius, Fahrenheit, Kelvin, Rankine)
- Thermal conductivity conversions (SI and Imperial units)
- R-value conversions (thermal resistance)
- Heat loss conversions (W/m, BTU/hr/ft)
- Energy conversions (kWh, MMBtu, therms, GJ)
- Length and area conversions for insulation sizing

Usage:
    >>> from units import TemperatureConverter, ThermalConductivityConverter
    >>> celsius = TemperatureConverter.fahrenheit_to_celsius(212.0)
    >>> k_imperial = ThermalConductivityConverter.si_to_imperial(0.035)

All conversion functions are:
- Deterministic (same input always produces same output)
- Type-safe with clear parameter names
- Documented with units and precision notes
- Cached where appropriate for performance

References:
    - ASTM E380: Standard Practice for Use of SI Units
    - ASHRAE Handbook - Fundamentals (2021)
    - ISO 80000-5: Thermodynamics

Author: GreenLang Engineering Team
Version: 1.0.0
License: Apache 2.0
"""

from dataclasses import dataclass
from typing import Union, Tuple, Optional
from functools import lru_cache
import math


# =============================================================================
# TYPE ALIASES
# =============================================================================

Number = Union[int, float]


# =============================================================================
# CONVERSION CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class ConversionFactors:
    """
    Precise conversion factors for unit conversions.

    All values are exact or to maximum available precision.
    Reference: NIST Guide to SI Units, ASTM E380.
    """

    # Temperature offsets
    CELSIUS_FAHRENHEIT_OFFSET: float = 32.0
    CELSIUS_FAHRENHEIT_RATIO: float = 9.0 / 5.0  # 1.8 exactly
    KELVIN_CELSIUS_OFFSET: float = 273.15
    RANKINE_FAHRENHEIT_OFFSET: float = 459.67

    # Length conversions
    METER_TO_FOOT: float = 3.280839895  # exact: 1/0.3048
    FOOT_TO_METER: float = 0.3048  # exact by definition
    INCH_TO_METER: float = 0.0254  # exact by definition
    METER_TO_INCH: float = 39.37007874  # exact: 1/0.0254
    MILLIMETER_TO_INCH: float = 0.03937007874

    # Area conversions
    SQ_METER_TO_SQ_FOOT: float = 10.76391042  # exact: (1/0.3048)^2
    SQ_FOOT_TO_SQ_METER: float = 0.09290304  # exact: 0.3048^2

    # Energy conversions
    JOULE_TO_BTU: float = 0.000947817  # 1 J = 0.000947817 BTU
    BTU_TO_JOULE: float = 1055.06  # 1 BTU = 1055.06 J (IT)
    KWH_TO_MMBTU: float = 0.003412141633  # 1 kWh = 0.003412 MMBtu
    MMBTU_TO_KWH: float = 293.07107017  # 1 MMBtu = 293.07 kWh
    THERM_TO_MMBTU: float = 0.1  # 1 therm = 100,000 BTU = 0.1 MMBtu
    GJ_TO_MMBTU: float = 0.94781712  # 1 GJ = 0.9478 MMBtu
    MMBTU_TO_GJ: float = 1.055056  # 1 MMBtu = 1.055 GJ

    # Power conversions
    WATT_TO_BTU_HR: float = 3.412141633  # 1 W = 3.412 BTU/hr
    BTU_HR_TO_WATT: float = 0.29307107  # 1 BTU/hr = 0.2931 W
    KW_TO_BTU_HR: float = 3412.141633
    BTU_HR_TO_KW: float = 0.00029307107

    # Thermal conductivity (k-value)
    # SI: W/(m*K)
    # Imperial: BTU*in/(hr*ft^2*F) or BTU/(hr*ft*F)
    # 1 W/(m*K) = 6.9334713 BTU*in/(hr*ft^2*F)
    K_SI_TO_IMPERIAL_INCH: float = 6.933471799  # W/mK to BTU*in/hr*ft^2*F
    K_IMPERIAL_INCH_TO_SI: float = 0.1442279107  # BTU*in/hr*ft^2*F to W/mK

    # 1 W/(m*K) = 0.5778263 BTU/(hr*ft*F)
    K_SI_TO_IMPERIAL_FOOT: float = 0.5778263  # W/mK to BTU/hr*ft*F
    K_IMPERIAL_FOOT_TO_SI: float = 1.7307346  # BTU/hr*ft*F to W/mK

    # R-value (thermal resistance)
    # SI: m^2*K/W
    # Imperial: hr*ft^2*F/BTU
    # 1 m^2*K/W = 5.678263 hr*ft^2*F/BTU
    R_SI_TO_IMPERIAL: float = 5.678263337  # m^2*K/W to hr*ft^2*F/BTU
    R_IMPERIAL_TO_SI: float = 0.1761102  # hr*ft^2*F/BTU to m^2*K/W

    # Heat flux
    # 1 W/m^2 = 0.3170 BTU/(hr*ft^2)
    HEAT_FLUX_SI_TO_IMPERIAL: float = 0.31699832  # W/m^2 to BTU/hr*ft^2
    HEAT_FLUX_IMPERIAL_TO_SI: float = 3.15459075  # BTU/hr*ft^2 to W/m^2

    # Linear heat loss
    # 1 W/m = 1.04 BTU/(hr*ft)
    HEAT_LOSS_SI_TO_IMPERIAL: float = 1.04036585  # W/m to BTU/hr*ft
    HEAT_LOSS_IMPERIAL_TO_SI: float = 0.96121635  # BTU/hr*ft to W/m


# =============================================================================
# TEMPERATURE CONVERSIONS
# =============================================================================

class TemperatureConverter:
    """
    Temperature unit conversions.

    Supports Celsius (C), Fahrenheit (F), Kelvin (K), and Rankine (R).
    All conversions maintain full floating-point precision.

    Usage:
        >>> TemperatureConverter.celsius_to_fahrenheit(100.0)
        212.0
        >>> TemperatureConverter.fahrenheit_to_celsius(32.0)
        0.0
    """

    # Constants for conversion
    _CF = ConversionFactors()

    @staticmethod
    @lru_cache(maxsize=1000)
    def celsius_to_fahrenheit(celsius: Number) -> float:
        """
        Convert Celsius to Fahrenheit.

        Formula: F = C * 9/5 + 32

        Args:
            celsius: Temperature in Celsius

        Returns:
            Temperature in Fahrenheit
        """
        return float(celsius) * (9.0 / 5.0) + 32.0

    @staticmethod
    @lru_cache(maxsize=1000)
    def fahrenheit_to_celsius(fahrenheit: Number) -> float:
        """
        Convert Fahrenheit to Celsius.

        Formula: C = (F - 32) * 5/9

        Args:
            fahrenheit: Temperature in Fahrenheit

        Returns:
            Temperature in Celsius
        """
        return (float(fahrenheit) - 32.0) * (5.0 / 9.0)

    @staticmethod
    @lru_cache(maxsize=1000)
    def celsius_to_kelvin(celsius: Number) -> float:
        """
        Convert Celsius to Kelvin.

        Formula: K = C + 273.15

        Args:
            celsius: Temperature in Celsius

        Returns:
            Temperature in Kelvin

        Raises:
            ValueError: If result would be negative (below absolute zero)
        """
        kelvin = float(celsius) + 273.15
        if kelvin < 0:
            raise ValueError(
                f"Temperature {celsius}C is below absolute zero"
            )
        return kelvin

    @staticmethod
    @lru_cache(maxsize=1000)
    def kelvin_to_celsius(kelvin: Number) -> float:
        """
        Convert Kelvin to Celsius.

        Formula: C = K - 273.15

        Args:
            kelvin: Temperature in Kelvin

        Returns:
            Temperature in Celsius

        Raises:
            ValueError: If kelvin is negative
        """
        if kelvin < 0:
            raise ValueError(
                f"Kelvin temperature cannot be negative: {kelvin}"
            )
        return float(kelvin) - 273.15

    @staticmethod
    @lru_cache(maxsize=1000)
    def fahrenheit_to_kelvin(fahrenheit: Number) -> float:
        """
        Convert Fahrenheit to Kelvin.

        Formula: K = (F - 32) * 5/9 + 273.15

        Args:
            fahrenheit: Temperature in Fahrenheit

        Returns:
            Temperature in Kelvin
        """
        celsius = (float(fahrenheit) - 32.0) * (5.0 / 9.0)
        return celsius + 273.15

    @staticmethod
    @lru_cache(maxsize=1000)
    def kelvin_to_fahrenheit(kelvin: Number) -> float:
        """
        Convert Kelvin to Fahrenheit.

        Formula: F = (K - 273.15) * 9/5 + 32

        Args:
            kelvin: Temperature in Kelvin

        Returns:
            Temperature in Fahrenheit
        """
        celsius = float(kelvin) - 273.15
        return celsius * (9.0 / 5.0) + 32.0

    @staticmethod
    @lru_cache(maxsize=1000)
    def celsius_to_rankine(celsius: Number) -> float:
        """
        Convert Celsius to Rankine.

        Formula: R = (C + 273.15) * 9/5

        Args:
            celsius: Temperature in Celsius

        Returns:
            Temperature in Rankine
        """
        kelvin = float(celsius) + 273.15
        return kelvin * (9.0 / 5.0)

    @staticmethod
    @lru_cache(maxsize=1000)
    def rankine_to_celsius(rankine: Number) -> float:
        """
        Convert Rankine to Celsius.

        Formula: C = R * 5/9 - 273.15

        Args:
            rankine: Temperature in Rankine

        Returns:
            Temperature in Celsius
        """
        kelvin = float(rankine) * (5.0 / 9.0)
        return kelvin - 273.15

    @staticmethod
    @lru_cache(maxsize=1000)
    def fahrenheit_to_rankine(fahrenheit: Number) -> float:
        """
        Convert Fahrenheit to Rankine.

        Formula: R = F + 459.67

        Args:
            fahrenheit: Temperature in Fahrenheit

        Returns:
            Temperature in Rankine
        """
        return float(fahrenheit) + 459.67

    @staticmethod
    @lru_cache(maxsize=1000)
    def rankine_to_fahrenheit(rankine: Number) -> float:
        """
        Convert Rankine to Fahrenheit.

        Formula: F = R - 459.67

        Args:
            rankine: Temperature in Rankine

        Returns:
            Temperature in Fahrenheit
        """
        return float(rankine) - 459.67

    @staticmethod
    @lru_cache(maxsize=1000)
    def kelvin_to_rankine(kelvin: Number) -> float:
        """
        Convert Kelvin to Rankine.

        Formula: R = K * 9/5

        Args:
            kelvin: Temperature in Kelvin

        Returns:
            Temperature in Rankine
        """
        return float(kelvin) * (9.0 / 5.0)

    @staticmethod
    @lru_cache(maxsize=1000)
    def rankine_to_kelvin(rankine: Number) -> float:
        """
        Convert Rankine to Kelvin.

        Formula: K = R * 5/9

        Args:
            rankine: Temperature in Rankine

        Returns:
            Temperature in Kelvin
        """
        return float(rankine) * (5.0 / 9.0)

    @classmethod
    def convert(
        cls,
        value: Number,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        General temperature conversion.

        Args:
            value: Temperature value
            from_unit: Source unit ('C', 'F', 'K', 'R')
            to_unit: Target unit ('C', 'F', 'K', 'R')

        Returns:
            Converted temperature

        Raises:
            ValueError: If unknown unit specified
        """
        # Normalize unit names
        from_unit = from_unit.upper().strip()
        to_unit = to_unit.upper().strip()

        if from_unit == to_unit:
            return float(value)

        # Convert to Celsius first
        if from_unit == 'C':
            celsius = float(value)
        elif from_unit == 'F':
            celsius = cls.fahrenheit_to_celsius(value)
        elif from_unit == 'K':
            celsius = cls.kelvin_to_celsius(value)
        elif from_unit == 'R':
            celsius = cls.rankine_to_celsius(value)
        else:
            raise ValueError(f"Unknown temperature unit: {from_unit}")

        # Convert from Celsius to target
        if to_unit == 'C':
            return celsius
        elif to_unit == 'F':
            return cls.celsius_to_fahrenheit(celsius)
        elif to_unit == 'K':
            return cls.celsius_to_kelvin(celsius)
        elif to_unit == 'R':
            return cls.celsius_to_rankine(celsius)
        else:
            raise ValueError(f"Unknown temperature unit: {to_unit}")


# =============================================================================
# THERMAL CONDUCTIVITY CONVERSIONS
# =============================================================================

class ThermalConductivityConverter:
    """
    Thermal conductivity (k-value) unit conversions.

    SI Unit: W/(m*K) [Watts per meter-Kelvin]
    Imperial Units:
        - BTU*in/(hr*ft^2*F) [BTU-inches per hour-square foot-Fahrenheit]
        - BTU/(hr*ft*F) [BTU per hour-foot-Fahrenheit]

    Usage:
        >>> ThermalConductivityConverter.si_to_imperial(0.035)
        0.2427  # BTU*in/(hr*ft^2*F)
    """

    _CF = ConversionFactors()

    @staticmethod
    @lru_cache(maxsize=1000)
    def si_to_imperial_inch(k_si: Number) -> float:
        """
        Convert W/(m*K) to BTU*in/(hr*ft^2*F).

        This is the common imperial unit for insulation k-values.

        Args:
            k_si: Thermal conductivity in W/(m*K)

        Returns:
            Thermal conductivity in BTU*in/(hr*ft^2*F)
        """
        return float(k_si) * ConversionFactors.K_SI_TO_IMPERIAL_INCH

    @staticmethod
    @lru_cache(maxsize=1000)
    def imperial_inch_to_si(k_imperial: Number) -> float:
        """
        Convert BTU*in/(hr*ft^2*F) to W/(m*K).

        Args:
            k_imperial: Thermal conductivity in BTU*in/(hr*ft^2*F)

        Returns:
            Thermal conductivity in W/(m*K)
        """
        return float(k_imperial) * ConversionFactors.K_IMPERIAL_INCH_TO_SI

    @staticmethod
    @lru_cache(maxsize=1000)
    def si_to_imperial_foot(k_si: Number) -> float:
        """
        Convert W/(m*K) to BTU/(hr*ft*F).

        Args:
            k_si: Thermal conductivity in W/(m*K)

        Returns:
            Thermal conductivity in BTU/(hr*ft*F)
        """
        return float(k_si) * ConversionFactors.K_SI_TO_IMPERIAL_FOOT

    @staticmethod
    @lru_cache(maxsize=1000)
    def imperial_foot_to_si(k_imperial: Number) -> float:
        """
        Convert BTU/(hr*ft*F) to W/(m*K).

        Args:
            k_imperial: Thermal conductivity in BTU/(hr*ft*F)

        Returns:
            Thermal conductivity in W/(m*K)
        """
        return float(k_imperial) * ConversionFactors.K_IMPERIAL_FOOT_TO_SI

    @classmethod
    def si_to_imperial(cls, k_si: Number, inch_basis: bool = True) -> float:
        """
        Convert SI to Imperial units.

        Args:
            k_si: Thermal conductivity in W/(m*K)
            inch_basis: If True, return BTU*in/(hr*ft^2*F)
                       If False, return BTU/(hr*ft*F)

        Returns:
            Thermal conductivity in imperial units
        """
        if inch_basis:
            return cls.si_to_imperial_inch(k_si)
        else:
            return cls.si_to_imperial_foot(k_si)

    @classmethod
    def convert(
        cls,
        value: Number,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        General thermal conductivity conversion.

        Supported units:
            - 'W/mK', 'W/(m*K)', 'SI' - Watts per meter-Kelvin
            - 'BTU*in/hr*ft2*F', 'IP_inch' - BTU-inch per hour-sqft-F
            - 'BTU/hr*ft*F', 'IP_foot' - BTU per hour-foot-F

        Args:
            value: Thermal conductivity value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted thermal conductivity
        """
        # Normalize unit names
        si_units = {'w/mk', 'w/(m*k)', 'si', 'w/m.k', 'w/m*k'}
        inch_units = {'btu*in/hr*ft2*f', 'ip_inch', 'btu-in/hr-ft2-f',
                      'btu.in/hr.ft2.f', 'btu*in/(hr*ft^2*f)'}
        foot_units = {'btu/hr*ft*f', 'ip_foot', 'btu/hr-ft-f',
                      'btu/(hr*ft*f)', 'btu.hr.ft.f'}

        from_lower = from_unit.lower().strip()
        to_lower = to_unit.lower().strip()

        # Convert to SI first
        if from_lower in si_units:
            k_si = float(value)
        elif from_lower in inch_units:
            k_si = cls.imperial_inch_to_si(value)
        elif from_lower in foot_units:
            k_si = cls.imperial_foot_to_si(value)
        else:
            raise ValueError(f"Unknown thermal conductivity unit: {from_unit}")

        # Convert from SI to target
        if to_lower in si_units:
            return k_si
        elif to_lower in inch_units:
            return cls.si_to_imperial_inch(k_si)
        elif to_lower in foot_units:
            return cls.si_to_imperial_foot(k_si)
        else:
            raise ValueError(f"Unknown thermal conductivity unit: {to_unit}")


# =============================================================================
# R-VALUE (THERMAL RESISTANCE) CONVERSIONS
# =============================================================================

class RValueConverter:
    """
    Thermal resistance (R-value) unit conversions.

    SI Unit: m^2*K/W [square meter-Kelvin per Watt]
    Imperial Unit: hr*ft^2*F/BTU [hour-square foot-Fahrenheit per BTU]

    Note: R-value is the inverse of U-value (thermal transmittance).

    Usage:
        >>> RValueConverter.si_to_imperial(1.0)
        5.678  # hr*ft^2*F/BTU
    """

    _CF = ConversionFactors()

    @staticmethod
    @lru_cache(maxsize=1000)
    def si_to_imperial(r_si: Number) -> float:
        """
        Convert m^2*K/W to hr*ft^2*F/BTU.

        Args:
            r_si: R-value in m^2*K/W

        Returns:
            R-value in hr*ft^2*F/BTU
        """
        return float(r_si) * ConversionFactors.R_SI_TO_IMPERIAL

    @staticmethod
    @lru_cache(maxsize=1000)
    def imperial_to_si(r_imperial: Number) -> float:
        """
        Convert hr*ft^2*F/BTU to m^2*K/W.

        Args:
            r_imperial: R-value in hr*ft^2*F/BTU

        Returns:
            R-value in m^2*K/W
        """
        return float(r_imperial) * ConversionFactors.R_IMPERIAL_TO_SI

    @staticmethod
    def from_k_value(
        k_value: Number,
        thickness_m: Number,
        k_unit: str = "W/mK"
    ) -> float:
        """
        Calculate R-value from thermal conductivity and thickness.

        R = thickness / k

        Args:
            k_value: Thermal conductivity
            thickness_m: Insulation thickness in meters
            k_unit: Unit of k_value (default "W/mK")

        Returns:
            R-value in m^2*K/W
        """
        # Convert k to SI if needed
        if k_unit.lower() not in ['w/mk', 'si', 'w/(m*k)']:
            k_si = ThermalConductivityConverter.convert(k_value, k_unit, "W/mK")
        else:
            k_si = float(k_value)

        if k_si <= 0:
            raise ValueError(f"k-value must be positive: {k_si}")
        if thickness_m <= 0:
            raise ValueError(f"Thickness must be positive: {thickness_m}")

        return float(thickness_m) / k_si

    @staticmethod
    def to_u_value(r_value: Number, unit: str = "SI") -> float:
        """
        Convert R-value to U-value (thermal transmittance).

        U = 1/R

        Args:
            r_value: Thermal resistance
            unit: "SI" for W/(m^2*K) or "IP" for BTU/(hr*ft^2*F)

        Returns:
            U-value in specified units
        """
        if r_value <= 0:
            raise ValueError(f"R-value must be positive: {r_value}")

        return 1.0 / float(r_value)

    @classmethod
    def convert(
        cls,
        value: Number,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        General R-value conversion.

        Args:
            value: R-value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted R-value
        """
        si_units = {'m2k/w', 'm^2*k/w', 'si', 'm2.k/w', 'rsi'}
        ip_units = {'hr*ft2*f/btu', 'ip', 'ft2.f.hr/btu', 'r-value'}

        from_lower = from_unit.lower().strip()
        to_lower = to_unit.lower().strip()

        # Convert to SI first
        if from_lower in si_units:
            r_si = float(value)
        elif from_lower in ip_units:
            r_si = cls.imperial_to_si(value)
        else:
            raise ValueError(f"Unknown R-value unit: {from_unit}")

        # Convert from SI to target
        if to_lower in si_units:
            return r_si
        elif to_lower in ip_units:
            return cls.si_to_imperial(r_si)
        else:
            raise ValueError(f"Unknown R-value unit: {to_unit}")


# =============================================================================
# HEAT LOSS CONVERSIONS
# =============================================================================

class HeatLossConverter:
    """
    Heat loss rate unit conversions.

    For linear heat loss (pipes):
        SI: W/m [Watts per meter of pipe length]
        Imperial: BTU/(hr*ft) [BTU per hour per foot of pipe length]

    For surface heat flux:
        SI: W/m^2 [Watts per square meter]
        Imperial: BTU/(hr*ft^2) [BTU per hour per square foot]

    Usage:
        >>> HeatLossConverter.linear_si_to_imperial(100.0)
        104.04  # BTU/hr*ft
    """

    _CF = ConversionFactors()

    # =========================================================================
    # Linear Heat Loss (W/m <-> BTU/hr/ft)
    # =========================================================================

    @staticmethod
    @lru_cache(maxsize=1000)
    def linear_si_to_imperial(q_wm: Number) -> float:
        """
        Convert W/m to BTU/(hr*ft).

        Args:
            q_wm: Heat loss in W/m

        Returns:
            Heat loss in BTU/(hr*ft)
        """
        return float(q_wm) * ConversionFactors.HEAT_LOSS_SI_TO_IMPERIAL

    @staticmethod
    @lru_cache(maxsize=1000)
    def linear_imperial_to_si(q_btu: Number) -> float:
        """
        Convert BTU/(hr*ft) to W/m.

        Args:
            q_btu: Heat loss in BTU/(hr*ft)

        Returns:
            Heat loss in W/m
        """
        return float(q_btu) * ConversionFactors.HEAT_LOSS_IMPERIAL_TO_SI

    # =========================================================================
    # Surface Heat Flux (W/m^2 <-> BTU/hr/ft^2)
    # =========================================================================

    @staticmethod
    @lru_cache(maxsize=1000)
    def flux_si_to_imperial(q_wm2: Number) -> float:
        """
        Convert W/m^2 to BTU/(hr*ft^2).

        Args:
            q_wm2: Heat flux in W/m^2

        Returns:
            Heat flux in BTU/(hr*ft^2)
        """
        return float(q_wm2) * ConversionFactors.HEAT_FLUX_SI_TO_IMPERIAL

    @staticmethod
    @lru_cache(maxsize=1000)
    def flux_imperial_to_si(q_btu: Number) -> float:
        """
        Convert BTU/(hr*ft^2) to W/m^2.

        Args:
            q_btu: Heat flux in BTU/(hr*ft^2)

        Returns:
            Heat flux in W/m^2
        """
        return float(q_btu) * ConversionFactors.HEAT_FLUX_IMPERIAL_TO_SI

    @classmethod
    def convert_linear(
        cls,
        value: Number,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert linear heat loss between units.

        Args:
            value: Heat loss value
            from_unit: Source unit ('W/m', 'BTU/hr/ft')
            to_unit: Target unit

        Returns:
            Converted heat loss
        """
        si_units = {'w/m', 'si', 'watt/m', 'w/meter'}
        ip_units = {'btu/hr/ft', 'ip', 'btu/hr*ft', 'btu/(hr*ft)'}

        from_lower = from_unit.lower().strip()
        to_lower = to_unit.lower().strip()

        if from_lower in si_units:
            q_si = float(value)
        elif from_lower in ip_units:
            q_si = cls.linear_imperial_to_si(value)
        else:
            raise ValueError(f"Unknown linear heat loss unit: {from_unit}")

        if to_lower in si_units:
            return q_si
        elif to_lower in ip_units:
            return cls.linear_si_to_imperial(q_si)
        else:
            raise ValueError(f"Unknown linear heat loss unit: {to_unit}")


# =============================================================================
# ENERGY CONVERSIONS
# =============================================================================

class EnergyConverter:
    """
    Energy unit conversions.

    Supports:
        - kWh (kilowatt-hours)
        - MMBtu (million BTU)
        - therms (100,000 BTU)
        - GJ (gigajoules)
        - MJ (megajoules)
        - BTU (British Thermal Units)

    Usage:
        >>> EnergyConverter.kwh_to_mmbtu(1000.0)
        3.412  # MMBtu
    """

    _CF = ConversionFactors()

    # =========================================================================
    # kWh Conversions
    # =========================================================================

    @staticmethod
    @lru_cache(maxsize=1000)
    def kwh_to_mmbtu(kwh: Number) -> float:
        """
        Convert kWh to MMBtu.

        Args:
            kwh: Energy in kilowatt-hours

        Returns:
            Energy in million BTU
        """
        return float(kwh) * ConversionFactors.KWH_TO_MMBTU

    @staticmethod
    @lru_cache(maxsize=1000)
    def mmbtu_to_kwh(mmbtu: Number) -> float:
        """
        Convert MMBtu to kWh.

        Args:
            mmbtu: Energy in million BTU

        Returns:
            Energy in kilowatt-hours
        """
        return float(mmbtu) * ConversionFactors.MMBTU_TO_KWH

    @staticmethod
    @lru_cache(maxsize=1000)
    def kwh_to_gj(kwh: Number) -> float:
        """
        Convert kWh to GJ.

        Args:
            kwh: Energy in kilowatt-hours

        Returns:
            Energy in gigajoules
        """
        return float(kwh) * 0.0036  # 1 kWh = 3.6 MJ = 0.0036 GJ

    @staticmethod
    @lru_cache(maxsize=1000)
    def gj_to_kwh(gj: Number) -> float:
        """
        Convert GJ to kWh.

        Args:
            gj: Energy in gigajoules

        Returns:
            Energy in kilowatt-hours
        """
        return float(gj) * 277.778  # 1 GJ = 277.778 kWh

    @staticmethod
    @lru_cache(maxsize=1000)
    def kwh_to_therms(kwh: Number) -> float:
        """
        Convert kWh to therms.

        Args:
            kwh: Energy in kilowatt-hours

        Returns:
            Energy in therms
        """
        mmbtu = float(kwh) * ConversionFactors.KWH_TO_MMBTU
        return mmbtu * 10.0  # 1 MMBtu = 10 therms

    @staticmethod
    @lru_cache(maxsize=1000)
    def therms_to_kwh(therms: Number) -> float:
        """
        Convert therms to kWh.

        Args:
            therms: Energy in therms

        Returns:
            Energy in kilowatt-hours
        """
        mmbtu = float(therms) * 0.1  # 1 therm = 0.1 MMBtu
        return mmbtu * ConversionFactors.MMBTU_TO_KWH

    # =========================================================================
    # MMBtu Conversions
    # =========================================================================

    @staticmethod
    @lru_cache(maxsize=1000)
    def mmbtu_to_gj(mmbtu: Number) -> float:
        """
        Convert MMBtu to GJ.

        Args:
            mmbtu: Energy in million BTU

        Returns:
            Energy in gigajoules
        """
        return float(mmbtu) * ConversionFactors.MMBTU_TO_GJ

    @staticmethod
    @lru_cache(maxsize=1000)
    def gj_to_mmbtu(gj: Number) -> float:
        """
        Convert GJ to MMBtu.

        Args:
            gj: Energy in gigajoules

        Returns:
            Energy in million BTU
        """
        return float(gj) * ConversionFactors.GJ_TO_MMBTU

    @staticmethod
    @lru_cache(maxsize=1000)
    def mmbtu_to_therms(mmbtu: Number) -> float:
        """
        Convert MMBtu to therms.

        Args:
            mmbtu: Energy in million BTU

        Returns:
            Energy in therms
        """
        return float(mmbtu) * 10.0  # 1 MMBtu = 10 therms

    @staticmethod
    @lru_cache(maxsize=1000)
    def therms_to_mmbtu(therms: Number) -> float:
        """
        Convert therms to MMBtu.

        Args:
            therms: Energy in therms

        Returns:
            Energy in million BTU
        """
        return float(therms) * 0.1  # 1 therm = 0.1 MMBtu

    # =========================================================================
    # BTU Conversions
    # =========================================================================

    @staticmethod
    @lru_cache(maxsize=1000)
    def btu_to_kj(btu: Number) -> float:
        """
        Convert BTU to kJ.

        Args:
            btu: Energy in BTU

        Returns:
            Energy in kilojoules
        """
        return float(btu) * 1.055056  # 1 BTU = 1.055 kJ

    @staticmethod
    @lru_cache(maxsize=1000)
    def kj_to_btu(kj: Number) -> float:
        """
        Convert kJ to BTU.

        Args:
            kj: Energy in kilojoules

        Returns:
            Energy in BTU
        """
        return float(kj) * 0.947817  # 1 kJ = 0.9478 BTU

    @staticmethod
    @lru_cache(maxsize=1000)
    def btu_to_mmbtu(btu: Number) -> float:
        """
        Convert BTU to MMBtu.

        Args:
            btu: Energy in BTU

        Returns:
            Energy in million BTU
        """
        return float(btu) / 1_000_000.0

    @staticmethod
    @lru_cache(maxsize=1000)
    def mmbtu_to_btu(mmbtu: Number) -> float:
        """
        Convert MMBtu to BTU.

        Args:
            mmbtu: Energy in million BTU

        Returns:
            Energy in BTU
        """
        return float(mmbtu) * 1_000_000.0

    @classmethod
    def convert(
        cls,
        value: Number,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        General energy conversion.

        Supported units: kWh, MWh, GWh, BTU, MMBtu, therm, GJ, MJ, kJ

        Args:
            value: Energy value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted energy value
        """
        # Normalize units and convert to kWh as base
        unit_to_kwh = {
            'kwh': 1.0,
            'mwh': 1000.0,
            'gwh': 1_000_000.0,
            'btu': 0.000293071,
            'mmbtu': 293.071,
            'therm': 29.3071,
            'gj': 277.778,
            'mj': 0.277778,
            'kj': 0.000277778,
            'j': 0.000000277778,
            'wh': 0.001
        }

        from_lower = from_unit.lower().strip()
        to_lower = to_unit.lower().strip()

        if from_lower not in unit_to_kwh:
            raise ValueError(f"Unknown energy unit: {from_unit}")
        if to_lower not in unit_to_kwh:
            raise ValueError(f"Unknown energy unit: {to_unit}")

        # Convert to kWh, then to target
        kwh = float(value) * unit_to_kwh[from_lower]
        return kwh / unit_to_kwh[to_lower]


# =============================================================================
# POWER CONVERSIONS
# =============================================================================

class PowerConverter:
    """
    Power unit conversions.

    Supports:
        - W (Watts)
        - kW (kilowatts)
        - MW (megawatts)
        - BTU/hr (BTU per hour)
        - hp (horsepower)

    Usage:
        >>> PowerConverter.kw_to_btu_hr(10.0)
        34121.4  # BTU/hr
    """

    @staticmethod
    @lru_cache(maxsize=1000)
    def w_to_btu_hr(watts: Number) -> float:
        """
        Convert Watts to BTU/hr.

        Args:
            watts: Power in Watts

        Returns:
            Power in BTU/hr
        """
        return float(watts) * ConversionFactors.WATT_TO_BTU_HR

    @staticmethod
    @lru_cache(maxsize=1000)
    def btu_hr_to_w(btu_hr: Number) -> float:
        """
        Convert BTU/hr to Watts.

        Args:
            btu_hr: Power in BTU/hr

        Returns:
            Power in Watts
        """
        return float(btu_hr) * ConversionFactors.BTU_HR_TO_WATT

    @staticmethod
    @lru_cache(maxsize=1000)
    def kw_to_btu_hr(kw: Number) -> float:
        """
        Convert kW to BTU/hr.

        Args:
            kw: Power in kilowatts

        Returns:
            Power in BTU/hr
        """
        return float(kw) * ConversionFactors.KW_TO_BTU_HR

    @staticmethod
    @lru_cache(maxsize=1000)
    def btu_hr_to_kw(btu_hr: Number) -> float:
        """
        Convert BTU/hr to kW.

        Args:
            btu_hr: Power in BTU/hr

        Returns:
            Power in kilowatts
        """
        return float(btu_hr) * ConversionFactors.BTU_HR_TO_KW

    @staticmethod
    @lru_cache(maxsize=1000)
    def hp_to_kw(hp: Number) -> float:
        """
        Convert horsepower to kW.

        Args:
            hp: Power in horsepower

        Returns:
            Power in kilowatts
        """
        return float(hp) * 0.7457  # 1 hp = 0.7457 kW

    @staticmethod
    @lru_cache(maxsize=1000)
    def kw_to_hp(kw: Number) -> float:
        """
        Convert kW to horsepower.

        Args:
            kw: Power in kilowatts

        Returns:
            Power in horsepower
        """
        return float(kw) / 0.7457

    @classmethod
    def convert(
        cls,
        value: Number,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        General power conversion.

        Args:
            value: Power value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted power value
        """
        unit_to_w = {
            'w': 1.0,
            'kw': 1000.0,
            'mw': 1_000_000.0,
            'btu/hr': ConversionFactors.BTU_HR_TO_WATT,
            'btu_hr': ConversionFactors.BTU_HR_TO_WATT,
            'hp': 745.7,
            'ton': 3516.85,  # refrigeration ton
        }

        from_lower = from_unit.lower().strip()
        to_lower = to_unit.lower().strip()

        if from_lower not in unit_to_w:
            raise ValueError(f"Unknown power unit: {from_unit}")
        if to_lower not in unit_to_w:
            raise ValueError(f"Unknown power unit: {to_unit}")

        watts = float(value) * unit_to_w[from_lower]
        return watts / unit_to_w[to_lower]


# =============================================================================
# LENGTH AND AREA CONVERSIONS
# =============================================================================

class LengthConverter:
    """
    Length unit conversions.

    Supports m, mm, cm, ft, in.
    """

    @staticmethod
    @lru_cache(maxsize=1000)
    def meter_to_foot(m: Number) -> float:
        """Convert meters to feet."""
        return float(m) * ConversionFactors.METER_TO_FOOT

    @staticmethod
    @lru_cache(maxsize=1000)
    def foot_to_meter(ft: Number) -> float:
        """Convert feet to meters."""
        return float(ft) * ConversionFactors.FOOT_TO_METER

    @staticmethod
    @lru_cache(maxsize=1000)
    def meter_to_inch(m: Number) -> float:
        """Convert meters to inches."""
        return float(m) * ConversionFactors.METER_TO_INCH

    @staticmethod
    @lru_cache(maxsize=1000)
    def inch_to_meter(inch: Number) -> float:
        """Convert inches to meters."""
        return float(inch) * ConversionFactors.INCH_TO_METER

    @staticmethod
    @lru_cache(maxsize=1000)
    def mm_to_inch(mm: Number) -> float:
        """Convert millimeters to inches."""
        return float(mm) * ConversionFactors.MILLIMETER_TO_INCH

    @staticmethod
    @lru_cache(maxsize=1000)
    def inch_to_mm(inch: Number) -> float:
        """Convert inches to millimeters."""
        return float(inch) * 25.4  # Exact

    @classmethod
    def convert(
        cls,
        value: Number,
        from_unit: str,
        to_unit: str
    ) -> float:
        """General length conversion."""
        unit_to_m = {
            'm': 1.0,
            'meter': 1.0,
            'mm': 0.001,
            'cm': 0.01,
            'km': 1000.0,
            'ft': ConversionFactors.FOOT_TO_METER,
            'foot': ConversionFactors.FOOT_TO_METER,
            'feet': ConversionFactors.FOOT_TO_METER,
            'in': ConversionFactors.INCH_TO_METER,
            'inch': ConversionFactors.INCH_TO_METER,
        }

        from_lower = from_unit.lower().strip()
        to_lower = to_unit.lower().strip()

        if from_lower not in unit_to_m:
            raise ValueError(f"Unknown length unit: {from_unit}")
        if to_lower not in unit_to_m:
            raise ValueError(f"Unknown length unit: {to_unit}")

        meters = float(value) * unit_to_m[from_lower]
        return meters / unit_to_m[to_lower]


class AreaConverter:
    """
    Area unit conversions.

    Supports m^2, ft^2, in^2.
    """

    @staticmethod
    @lru_cache(maxsize=1000)
    def sq_meter_to_sq_foot(m2: Number) -> float:
        """Convert square meters to square feet."""
        return float(m2) * ConversionFactors.SQ_METER_TO_SQ_FOOT

    @staticmethod
    @lru_cache(maxsize=1000)
    def sq_foot_to_sq_meter(ft2: Number) -> float:
        """Convert square feet to square meters."""
        return float(ft2) * ConversionFactors.SQ_FOOT_TO_SQ_METER

    @classmethod
    def convert(
        cls,
        value: Number,
        from_unit: str,
        to_unit: str
    ) -> float:
        """General area conversion."""
        unit_to_m2 = {
            'm2': 1.0,
            'm^2': 1.0,
            'sq_m': 1.0,
            'mm2': 1e-6,
            'cm2': 1e-4,
            'ft2': ConversionFactors.SQ_FOOT_TO_SQ_METER,
            'ft^2': ConversionFactors.SQ_FOOT_TO_SQ_METER,
            'sq_ft': ConversionFactors.SQ_FOOT_TO_SQ_METER,
            'in2': 0.00064516,
            'in^2': 0.00064516,
        }

        from_lower = from_unit.lower().strip()
        to_lower = to_unit.lower().strip()

        if from_lower not in unit_to_m2:
            raise ValueError(f"Unknown area unit: {from_unit}")
        if to_lower not in unit_to_m2:
            raise ValueError(f"Unknown area unit: {to_unit}")

        sq_meters = float(value) * unit_to_m2[from_lower]
        return sq_meters / unit_to_m2[to_lower]


# =============================================================================
# UNIVERSAL CONVERTER CLASS
# =============================================================================

class UniversalConverter:
    """
    Universal unit converter dispatching to appropriate converter.

    Usage:
        >>> UniversalConverter.convert(100, 'C', 'F', 'temperature')
        212.0
    """

    _CONVERTERS = {
        'temperature': TemperatureConverter.convert,
        'thermal_conductivity': ThermalConductivityConverter.convert,
        'r_value': RValueConverter.convert,
        'heat_loss': HeatLossConverter.convert_linear,
        'energy': EnergyConverter.convert,
        'power': PowerConverter.convert,
        'length': LengthConverter.convert,
        'area': AreaConverter.convert,
    }

    @classmethod
    def convert(
        cls,
        value: Number,
        from_unit: str,
        to_unit: str,
        quantity_type: str
    ) -> float:
        """
        Convert between units for a given quantity type.

        Args:
            value: Numeric value to convert
            from_unit: Source unit
            to_unit: Target unit
            quantity_type: Type of quantity (temperature, energy, etc.)

        Returns:
            Converted value

        Raises:
            ValueError: If quantity type or units unknown
        """
        quantity_lower = quantity_type.lower().strip()

        if quantity_lower not in cls._CONVERTERS:
            available = ", ".join(cls._CONVERTERS.keys())
            raise ValueError(
                f"Unknown quantity type: {quantity_type}. "
                f"Available: {available}"
            )

        return cls._CONVERTERS[quantity_lower](value, from_unit, to_unit)

    @classmethod
    def list_quantity_types(cls) -> list:
        """Return list of supported quantity types."""
        return list(cls._CONVERTERS.keys())


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def celsius_to_fahrenheit(c: Number) -> float:
    """Shorthand for temperature conversion."""
    return TemperatureConverter.celsius_to_fahrenheit(c)


def fahrenheit_to_celsius(f: Number) -> float:
    """Shorthand for temperature conversion."""
    return TemperatureConverter.fahrenheit_to_celsius(f)


def k_si_to_imperial(k: Number) -> float:
    """Shorthand for thermal conductivity conversion."""
    return ThermalConductivityConverter.si_to_imperial_inch(k)


def k_imperial_to_si(k: Number) -> float:
    """Shorthand for thermal conductivity conversion."""
    return ThermalConductivityConverter.imperial_inch_to_si(k)


def r_si_to_imperial(r: Number) -> float:
    """Shorthand for R-value conversion."""
    return RValueConverter.si_to_imperial(r)


def r_imperial_to_si(r: Number) -> float:
    """Shorthand for R-value conversion."""
    return RValueConverter.imperial_to_si(r)


# =============================================================================
# VERSION AND METADATA
# =============================================================================

__version__ = "1.0.0"
__author__ = "GreenLang Engineering Team"
__license__ = "Apache 2.0"

__all__ = [
    # Constants
    "ConversionFactors",

    # Temperature
    "TemperatureConverter",

    # Thermal conductivity
    "ThermalConductivityConverter",

    # R-value
    "RValueConverter",

    # Heat loss
    "HeatLossConverter",

    # Energy
    "EnergyConverter",

    # Power
    "PowerConverter",

    # Length and Area
    "LengthConverter",
    "AreaConverter",

    # Universal
    "UniversalConverter",

    # Convenience functions
    "celsius_to_fahrenheit",
    "fahrenheit_to_celsius",
    "k_si_to_imperial",
    "k_imperial_to_si",
    "r_si_to_imperial",
    "r_imperial_to_si",
]
