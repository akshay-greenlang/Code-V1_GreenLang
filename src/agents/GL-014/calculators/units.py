"""
Unit Conversion Library for GL-014 EXCHANGER-PRO

This module provides comprehensive unit conversion utilities for all
heat exchanger calculations. Supports SI and Imperial units with
bidirectional conversion methods.

Example:
    >>> from units import UnitConverter
    >>> converter = UnitConverter()
    >>> temp_c = converter.temperature_to_celsius(212, "F")
    >>> print(temp_c)  # 100.0
"""

from typing import Dict, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


# =============================================================================
# UNIT CATEGORIES AND ENUMERATIONS
# =============================================================================

class TemperatureUnit(Enum):
    """Temperature unit enumeration."""
    CELSIUS = "C"
    FAHRENHEIT = "F"
    KELVIN = "K"
    RANKINE = "R"


class PressureUnit(Enum):
    """Pressure unit enumeration."""
    PASCAL = "Pa"
    KILOPASCAL = "kPa"
    MEGAPASCAL = "MPa"
    BAR = "bar"
    MILLIBAR = "mbar"
    PSI = "psi"
    PSIA = "psia"
    PSIG = "psig"
    ATM = "atm"
    MMHG = "mmHg"
    TORR = "torr"
    INH2O = "inH2O"
    KGF_CM2 = "kgf/cm2"


class FlowRateUnit(Enum):
    """Flow rate unit enumeration."""
    KG_S = "kg/s"
    KG_HR = "kg/hr"
    KG_MIN = "kg/min"
    LB_HR = "lb/hr"
    LB_S = "lb/s"
    M3_S = "m3/s"
    M3_HR = "m3/hr"
    L_S = "L/s"
    L_MIN = "L/min"
    GPM = "GPM"
    GPH = "gph"
    CFM = "CFM"
    SCFM = "SCFM"


class HeatTransferCoefficientUnit(Enum):
    """Heat transfer coefficient unit enumeration."""
    W_M2K = "W/m2K"
    W_M2C = "W/m2C"
    KW_M2K = "kW/m2K"
    BTU_HRFT2F = "BTU/hr.ft2.F"
    KCAL_HRM2C = "kcal/hr.m2.C"


class ThermalConductivityUnit(Enum):
    """Thermal conductivity unit enumeration."""
    W_MK = "W/mK"
    W_MC = "W/mC"
    BTU_HRFTF = "BTU/hr.ft.F"
    KCAL_HRMC = "kcal/hr.m.C"
    CAL_SCMC = "cal/s.cm.C"


class FoulingResistanceUnit(Enum):
    """Fouling resistance unit enumeration."""
    M2K_W = "m2K/W"
    M2C_W = "m2C/W"
    HRFT2F_BTU = "hr.ft2.F/BTU"


class HeatDutyUnit(Enum):
    """Heat duty unit enumeration."""
    W = "W"
    KW = "kW"
    MW = "MW"
    BTU_HR = "BTU/hr"
    MMBTU_HR = "MMBTU/hr"
    KCAL_HR = "kcal/hr"
    HP = "hp"
    TON_REF = "ton_ref"


class AreaUnit(Enum):
    """Area unit enumeration."""
    M2 = "m2"
    CM2 = "cm2"
    MM2 = "mm2"
    FT2 = "ft2"
    IN2 = "in2"


class LengthUnit(Enum):
    """Length unit enumeration."""
    M = "m"
    CM = "cm"
    MM = "mm"
    FT = "ft"
    IN = "in"
    MICRON = "micron"


class ViscosityUnit(Enum):
    """Dynamic viscosity unit enumeration."""
    PA_S = "Pa.s"
    MPAS = "mPa.s"
    CP = "cP"
    POISE = "P"
    LB_FT_HR = "lb/ft.hr"
    LB_FT_S = "lb/ft.s"


class DensityUnit(Enum):
    """Density unit enumeration."""
    KG_M3 = "kg/m3"
    G_CM3 = "g/cm3"
    G_ML = "g/mL"
    LB_FT3 = "lb/ft3"
    LB_GAL = "lb/gal"


class SpecificHeatUnit(Enum):
    """Specific heat unit enumeration."""
    J_KGK = "J/kgK"
    KJ_KGK = "kJ/kgK"
    BTU_LBF = "BTU/lb.F"
    KCAL_KGC = "kcal/kg.C"
    CAL_GC = "cal/g.C"


class VelocityUnit(Enum):
    """Velocity unit enumeration."""
    M_S = "m/s"
    FT_S = "ft/s"
    KM_HR = "km/hr"
    MPH = "mph"


# =============================================================================
# CONVERSION FACTORS (TO SI BASE UNITS)
# =============================================================================

@dataclass
class ConversionFactor:
    """Stores conversion factor and offset for unit conversion."""
    factor: float  # Multiply by this to convert to SI
    offset: float = 0.0  # Add this after multiplication (for temperature)
    description: str = ""


# =============================================================================
# UNIT CONVERTER CLASS
# =============================================================================

class UnitConverter:
    """
    Comprehensive unit converter for heat exchanger calculations.

    Supports bidirectional conversion between SI and Imperial units
    for all relevant quantities in heat exchanger design.

    Example:
        >>> converter = UnitConverter()
        >>> h_si = converter.convert_heat_transfer_coefficient(100, "BTU/hr.ft2.F", "W/m2K")
        >>> print(f"{h_si:.2f}")  # 567.83
    """

    # Temperature conversion constants
    TEMP_CONVERSIONS = {
        "C": {"to_K": (1.0, 273.15), "to_F": (1.8, 32.0), "to_R": (1.8, 491.67)},
        "F": {"to_K": (5/9, 255.372), "to_C": (5/9, -17.778), "to_R": (1.0, 459.67)},
        "K": {"to_C": (1.0, -273.15), "to_F": (1.8, -459.67), "to_R": (1.8, 0.0)},
        "R": {"to_K": (5/9, 0.0), "to_C": (5/9, -273.15), "to_F": (1.0, -459.67)},
    }

    # Pressure conversion to Pascal
    PRESSURE_TO_PA: Dict[str, float] = {
        "Pa": 1.0,
        "kPa": 1000.0,
        "MPa": 1e6,
        "bar": 1e5,
        "mbar": 100.0,
        "psi": 6894.757,
        "psia": 6894.757,
        "psig": 6894.757,  # Note: psig needs atmospheric correction
        "atm": 101325.0,
        "mmHg": 133.322,
        "torr": 133.322,
        "inH2O": 249.089,
        "kgf/cm2": 98066.5,
    }

    # Mass flow rate conversion to kg/s
    MASS_FLOW_TO_KGS: Dict[str, float] = {
        "kg/s": 1.0,
        "kg/hr": 1/3600,
        "kg/min": 1/60,
        "lb/hr": 0.000125998,
        "lb/s": 0.453592,
    }

    # Volumetric flow rate conversion to m3/s
    VOL_FLOW_TO_M3S: Dict[str, float] = {
        "m3/s": 1.0,
        "m3/hr": 1/3600,
        "L/s": 0.001,
        "L/min": 1/60000,
        "GPM": 6.309e-5,
        "gph": 1.0515e-6,
        "CFM": 4.719e-4,
    }

    # Heat transfer coefficient conversion to W/m2K
    HTC_TO_WM2K: Dict[str, float] = {
        "W/m2K": 1.0,
        "W/m2C": 1.0,
        "kW/m2K": 1000.0,
        "BTU/hr.ft2.F": 5.678263,
        "kcal/hr.m2.C": 1.163,
    }

    # Thermal conductivity conversion to W/mK
    THERMAL_COND_TO_WMK: Dict[str, float] = {
        "W/mK": 1.0,
        "W/mC": 1.0,
        "BTU/hr.ft.F": 1.730735,
        "kcal/hr.m.C": 1.163,
        "cal/s.cm.C": 418.68,
    }

    # Fouling resistance conversion to m2K/W
    FOULING_TO_M2KW: Dict[str, float] = {
        "m2K/W": 1.0,
        "m2C/W": 1.0,
        "hr.ft2.F/BTU": 0.176110,
    }

    # Heat duty conversion to Watts
    HEAT_DUTY_TO_W: Dict[str, float] = {
        "W": 1.0,
        "kW": 1000.0,
        "MW": 1e6,
        "BTU/hr": 0.293071,
        "MMBTU/hr": 293071.0,
        "kcal/hr": 1.163,
        "hp": 745.7,
        "ton_ref": 3516.85,
    }

    # Area conversion to m2
    AREA_TO_M2: Dict[str, float] = {
        "m2": 1.0,
        "cm2": 1e-4,
        "mm2": 1e-6,
        "ft2": 0.092903,
        "in2": 6.4516e-4,
    }

    # Length conversion to meters
    LENGTH_TO_M: Dict[str, float] = {
        "m": 1.0,
        "cm": 0.01,
        "mm": 0.001,
        "ft": 0.3048,
        "in": 0.0254,
        "micron": 1e-6,
    }

    # Dynamic viscosity conversion to Pa.s
    VISCOSITY_TO_PAS: Dict[str, float] = {
        "Pa.s": 1.0,
        "mPa.s": 0.001,
        "cP": 0.001,
        "P": 0.1,
        "lb/ft.hr": 4.134e-4,
        "lb/ft.s": 1.4882,
    }

    # Density conversion to kg/m3
    DENSITY_TO_KGM3: Dict[str, float] = {
        "kg/m3": 1.0,
        "g/cm3": 1000.0,
        "g/mL": 1000.0,
        "lb/ft3": 16.0185,
        "lb/gal": 119.826,
    }

    # Specific heat conversion to J/kgK
    SPECIFIC_HEAT_TO_JKGK: Dict[str, float] = {
        "J/kgK": 1.0,
        "kJ/kgK": 1000.0,
        "BTU/lb.F": 4186.8,
        "kcal/kg.C": 4186.8,
        "cal/g.C": 4186.8,
    }

    # Velocity conversion to m/s
    VELOCITY_TO_MS: Dict[str, float] = {
        "m/s": 1.0,
        "ft/s": 0.3048,
        "km/hr": 1/3.6,
        "mph": 0.44704,
    }

    def __init__(self):
        """Initialize the unit converter."""
        self._cache: Dict[str, float] = {}
        logger.debug("UnitConverter initialized")

    # =========================================================================
    # TEMPERATURE CONVERSIONS (Non-linear)
    # =========================================================================

    def temperature_to_celsius(self, value: float, from_unit: str) -> float:
        """
        Convert temperature to Celsius.

        Args:
            value: Temperature value
            from_unit: Source unit (C, F, K, R)

        Returns:
            Temperature in Celsius
        """
        from_unit = from_unit.upper().replace("DEG", "").strip()

        if from_unit == "C":
            return value
        elif from_unit == "F":
            return (value - 32) * 5 / 9
        elif from_unit == "K":
            return value - 273.15
        elif from_unit == "R":
            return (value - 491.67) * 5 / 9
        else:
            raise ValueError(f"Unknown temperature unit: {from_unit}")

    def temperature_to_fahrenheit(self, value: float, from_unit: str) -> float:
        """
        Convert temperature to Fahrenheit.

        Args:
            value: Temperature value
            from_unit: Source unit (C, F, K, R)

        Returns:
            Temperature in Fahrenheit
        """
        celsius = self.temperature_to_celsius(value, from_unit)
        return celsius * 9 / 5 + 32

    def temperature_to_kelvin(self, value: float, from_unit: str) -> float:
        """
        Convert temperature to Kelvin.

        Args:
            value: Temperature value
            from_unit: Source unit (C, F, K, R)

        Returns:
            Temperature in Kelvin
        """
        celsius = self.temperature_to_celsius(value, from_unit)
        return celsius + 273.15

    def temperature_to_rankine(self, value: float, from_unit: str) -> float:
        """
        Convert temperature to Rankine.

        Args:
            value: Temperature value
            from_unit: Source unit (C, F, K, R)

        Returns:
            Temperature in Rankine
        """
        kelvin = self.temperature_to_kelvin(value, from_unit)
        return kelvin * 9 / 5

    def convert_temperature(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        General temperature conversion.

        Args:
            value: Temperature value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted temperature value
        """
        to_unit = to_unit.upper().replace("DEG", "").strip()

        if to_unit == "C":
            return self.temperature_to_celsius(value, from_unit)
        elif to_unit == "F":
            return self.temperature_to_fahrenheit(value, from_unit)
        elif to_unit == "K":
            return self.temperature_to_kelvin(value, from_unit)
        elif to_unit == "R":
            return self.temperature_to_rankine(value, from_unit)
        else:
            raise ValueError(f"Unknown temperature unit: {to_unit}")

    def temperature_difference_to_celsius(
        self,
        delta: float,
        from_unit: str
    ) -> float:
        """
        Convert temperature difference to Celsius scale.

        Note: Temperature differences have different conversion
        than absolute temperatures (no offset).

        Args:
            delta: Temperature difference
            from_unit: Source unit

        Returns:
            Temperature difference in Celsius
        """
        from_unit = from_unit.upper().replace("DEG", "").strip()

        if from_unit in ["C", "K"]:
            return delta
        elif from_unit in ["F", "R"]:
            return delta * 5 / 9
        else:
            raise ValueError(f"Unknown temperature unit: {from_unit}")

    # =========================================================================
    # LINEAR UNIT CONVERSIONS
    # =========================================================================

    def _linear_convert(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        conversion_table: Dict[str, float]
    ) -> float:
        """
        Generic linear unit conversion.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit
            conversion_table: Dict mapping unit names to SI factors

        Returns:
            Converted value
        """
        if from_unit not in conversion_table:
            raise ValueError(f"Unknown unit: {from_unit}")
        if to_unit not in conversion_table:
            raise ValueError(f"Unknown unit: {to_unit}")

        # Convert to SI, then to target
        si_value = value * conversion_table[from_unit]
        return si_value / conversion_table[to_unit]

    def convert_pressure(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        atmospheric_pressure_pa: float = 101325.0
    ) -> float:
        """
        Convert pressure between units.

        Args:
            value: Pressure value
            from_unit: Source unit
            to_unit: Target unit
            atmospheric_pressure_pa: Atmospheric pressure for psig conversion

        Returns:
            Converted pressure value
        """
        # Handle gauge pressure
        if from_unit.lower() == "psig":
            value = value + atmospheric_pressure_pa / 6894.757
            from_unit = "psia"

        result = self._linear_convert(value, from_unit, to_unit, self.PRESSURE_TO_PA)

        if to_unit.lower() == "psig":
            result = result - atmospheric_pressure_pa / 6894.757

        return result

    def convert_mass_flow_rate(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert mass flow rate between units.

        Args:
            value: Flow rate value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted flow rate value
        """
        return self._linear_convert(value, from_unit, to_unit, self.MASS_FLOW_TO_KGS)

    def convert_volumetric_flow_rate(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert volumetric flow rate between units.

        Args:
            value: Flow rate value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted flow rate value
        """
        return self._linear_convert(value, from_unit, to_unit, self.VOL_FLOW_TO_M3S)

    def convert_heat_transfer_coefficient(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert heat transfer coefficient between units.

        Args:
            value: Heat transfer coefficient value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted heat transfer coefficient value
        """
        return self._linear_convert(value, from_unit, to_unit, self.HTC_TO_WM2K)

    def convert_thermal_conductivity(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert thermal conductivity between units.

        Args:
            value: Thermal conductivity value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted thermal conductivity value
        """
        return self._linear_convert(
            value, from_unit, to_unit, self.THERMAL_COND_TO_WMK
        )

    def convert_fouling_resistance(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert fouling resistance between units.

        Args:
            value: Fouling resistance value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted fouling resistance value
        """
        return self._linear_convert(value, from_unit, to_unit, self.FOULING_TO_M2KW)

    def convert_heat_duty(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert heat duty between units.

        Args:
            value: Heat duty value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted heat duty value
        """
        return self._linear_convert(value, from_unit, to_unit, self.HEAT_DUTY_TO_W)

    def convert_area(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert area between units.

        Args:
            value: Area value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted area value
        """
        return self._linear_convert(value, from_unit, to_unit, self.AREA_TO_M2)

    def convert_length(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert length between units.

        Args:
            value: Length value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted length value
        """
        return self._linear_convert(value, from_unit, to_unit, self.LENGTH_TO_M)

    def convert_viscosity(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert dynamic viscosity between units.

        Args:
            value: Viscosity value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted viscosity value
        """
        return self._linear_convert(value, from_unit, to_unit, self.VISCOSITY_TO_PAS)

    def convert_density(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert density between units.

        Args:
            value: Density value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted density value
        """
        return self._linear_convert(value, from_unit, to_unit, self.DENSITY_TO_KGM3)

    def convert_specific_heat(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert specific heat between units.

        Args:
            value: Specific heat value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted specific heat value
        """
        return self._linear_convert(
            value, from_unit, to_unit, self.SPECIFIC_HEAT_TO_JKGK
        )

    def convert_velocity(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert velocity between units.

        Args:
            value: Velocity value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted velocity value
        """
        return self._linear_convert(value, from_unit, to_unit, self.VELOCITY_TO_MS)

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def to_si(self, value: float, unit: str, quantity_type: str) -> float:
        """
        Convert any value to SI units.

        Args:
            value: Value to convert
            unit: Current unit
            quantity_type: Type of quantity (temperature, pressure, etc.)

        Returns:
            Value in SI units
        """
        quantity_map = {
            "temperature": ("C", self.convert_temperature),
            "pressure": ("Pa", self.convert_pressure),
            "mass_flow": ("kg/s", self.convert_mass_flow_rate),
            "vol_flow": ("m3/s", self.convert_volumetric_flow_rate),
            "htc": ("W/m2K", self.convert_heat_transfer_coefficient),
            "thermal_cond": ("W/mK", self.convert_thermal_conductivity),
            "fouling": ("m2K/W", self.convert_fouling_resistance),
            "heat_duty": ("W", self.convert_heat_duty),
            "area": ("m2", self.convert_area),
            "length": ("m", self.convert_length),
            "viscosity": ("Pa.s", self.convert_viscosity),
            "density": ("kg/m3", self.convert_density),
            "specific_heat": ("J/kgK", self.convert_specific_heat),
            "velocity": ("m/s", self.convert_velocity),
        }

        if quantity_type not in quantity_map:
            raise ValueError(f"Unknown quantity type: {quantity_type}")

        si_unit, converter = quantity_map[quantity_type]
        return converter(value, unit, si_unit)

    def from_si(self, value: float, to_unit: str, quantity_type: str) -> float:
        """
        Convert from SI units to specified units.

        Args:
            value: Value in SI units
            to_unit: Target unit
            quantity_type: Type of quantity

        Returns:
            Converted value
        """
        quantity_map = {
            "temperature": ("C", self.convert_temperature),
            "pressure": ("Pa", self.convert_pressure),
            "mass_flow": ("kg/s", self.convert_mass_flow_rate),
            "vol_flow": ("m3/s", self.convert_volumetric_flow_rate),
            "htc": ("W/m2K", self.convert_heat_transfer_coefficient),
            "thermal_cond": ("W/mK", self.convert_thermal_conductivity),
            "fouling": ("m2K/W", self.convert_fouling_resistance),
            "heat_duty": ("W", self.convert_heat_duty),
            "area": ("m2", self.convert_area),
            "length": ("m", self.convert_length),
            "viscosity": ("Pa.s", self.convert_viscosity),
            "density": ("kg/m3", self.convert_density),
            "specific_heat": ("J/kgK", self.convert_specific_heat),
            "velocity": ("m/s", self.convert_velocity),
        }

        if quantity_type not in quantity_map:
            raise ValueError(f"Unknown quantity type: {quantity_type}")

        si_unit, converter = quantity_map[quantity_type]
        return converter(value, si_unit, to_unit)

    # =========================================================================
    # DERIVED QUANTITY CONVERSIONS
    # =========================================================================

    def convert_overall_htc(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert overall heat transfer coefficient (U).

        Alias for convert_heat_transfer_coefficient.

        Args:
            value: U value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted U value
        """
        return self.convert_heat_transfer_coefficient(value, from_unit, to_unit)

    def convert_film_coefficient(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """
        Convert film heat transfer coefficient (h).

        Alias for convert_heat_transfer_coefficient.

        Args:
            value: h value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted h value
        """
        return self.convert_heat_transfer_coefficient(value, from_unit, to_unit)

    def mass_flow_to_volumetric(
        self,
        mass_flow: float,
        mass_flow_unit: str,
        density: float,
        density_unit: str,
        vol_flow_unit: str
    ) -> float:
        """
        Convert mass flow rate to volumetric flow rate.

        Args:
            mass_flow: Mass flow rate
            mass_flow_unit: Mass flow unit
            density: Fluid density
            density_unit: Density unit
            vol_flow_unit: Target volumetric flow unit

        Returns:
            Volumetric flow rate in specified units
        """
        # Convert to SI
        m_dot_si = self.convert_mass_flow_rate(mass_flow, mass_flow_unit, "kg/s")
        rho_si = self.convert_density(density, density_unit, "kg/m3")

        # Calculate volumetric flow in m3/s
        v_dot_si = m_dot_si / rho_si

        # Convert to target unit
        return self.convert_volumetric_flow_rate(v_dot_si, "m3/s", vol_flow_unit)

    def volumetric_to_mass_flow(
        self,
        vol_flow: float,
        vol_flow_unit: str,
        density: float,
        density_unit: str,
        mass_flow_unit: str
    ) -> float:
        """
        Convert volumetric flow rate to mass flow rate.

        Args:
            vol_flow: Volumetric flow rate
            vol_flow_unit: Volume flow unit
            density: Fluid density
            density_unit: Density unit
            mass_flow_unit: Target mass flow unit

        Returns:
            Mass flow rate in specified units
        """
        # Convert to SI
        v_dot_si = self.convert_volumetric_flow_rate(vol_flow, vol_flow_unit, "m3/s")
        rho_si = self.convert_density(density, density_unit, "kg/m3")

        # Calculate mass flow in kg/s
        m_dot_si = v_dot_si * rho_si

        # Convert to target unit
        return self.convert_mass_flow_rate(m_dot_si, "kg/s", mass_flow_unit)

    # =========================================================================
    # VALIDATION AND UTILITY METHODS
    # =========================================================================

    def is_valid_unit(self, unit: str, quantity_type: str) -> bool:
        """
        Check if a unit is valid for a given quantity type.

        Args:
            unit: Unit to check
            quantity_type: Type of quantity

        Returns:
            True if valid, False otherwise
        """
        table_map = {
            "pressure": self.PRESSURE_TO_PA,
            "mass_flow": self.MASS_FLOW_TO_KGS,
            "vol_flow": self.VOL_FLOW_TO_M3S,
            "htc": self.HTC_TO_WM2K,
            "thermal_cond": self.THERMAL_COND_TO_WMK,
            "fouling": self.FOULING_TO_M2KW,
            "heat_duty": self.HEAT_DUTY_TO_W,
            "area": self.AREA_TO_M2,
            "length": self.LENGTH_TO_M,
            "viscosity": self.VISCOSITY_TO_PAS,
            "density": self.DENSITY_TO_KGM3,
            "specific_heat": self.SPECIFIC_HEAT_TO_JKGK,
            "velocity": self.VELOCITY_TO_MS,
        }

        if quantity_type == "temperature":
            return unit.upper().replace("DEG", "").strip() in ["C", "F", "K", "R"]

        if quantity_type not in table_map:
            return False

        return unit in table_map[quantity_type]

    def get_available_units(self, quantity_type: str) -> list:
        """
        Get list of available units for a quantity type.

        Args:
            quantity_type: Type of quantity

        Returns:
            List of available unit strings
        """
        table_map = {
            "temperature": ["C", "F", "K", "R"],
            "pressure": list(self.PRESSURE_TO_PA.keys()),
            "mass_flow": list(self.MASS_FLOW_TO_KGS.keys()),
            "vol_flow": list(self.VOL_FLOW_TO_M3S.keys()),
            "htc": list(self.HTC_TO_WM2K.keys()),
            "thermal_cond": list(self.THERMAL_COND_TO_WMK.keys()),
            "fouling": list(self.FOULING_TO_M2KW.keys()),
            "heat_duty": list(self.HEAT_DUTY_TO_W.keys()),
            "area": list(self.AREA_TO_M2.keys()),
            "length": list(self.LENGTH_TO_M.keys()),
            "viscosity": list(self.VISCOSITY_TO_PAS.keys()),
            "density": list(self.DENSITY_TO_KGM3.keys()),
            "specific_heat": list(self.SPECIFIC_HEAT_TO_JKGK.keys()),
            "velocity": list(self.VELOCITY_TO_MS.keys()),
        }

        return table_map.get(quantity_type, [])

    def get_si_unit(self, quantity_type: str) -> str:
        """
        Get the SI unit for a quantity type.

        Args:
            quantity_type: Type of quantity

        Returns:
            SI unit string
        """
        si_units = {
            "temperature": "C",
            "pressure": "Pa",
            "mass_flow": "kg/s",
            "vol_flow": "m3/s",
            "htc": "W/m2K",
            "thermal_cond": "W/mK",
            "fouling": "m2K/W",
            "heat_duty": "W",
            "area": "m2",
            "length": "m",
            "viscosity": "Pa.s",
            "density": "kg/m3",
            "specific_heat": "J/kgK",
            "velocity": "m/s",
        }

        return si_units.get(quantity_type, "")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def convert(value: float, from_unit: str, to_unit: str, quantity_type: str) -> float:
    """
    Convenience function for unit conversion.

    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit
        quantity_type: Type of quantity

    Returns:
        Converted value
    """
    converter = UnitConverter()

    converters = {
        "temperature": converter.convert_temperature,
        "pressure": converter.convert_pressure,
        "mass_flow": converter.convert_mass_flow_rate,
        "vol_flow": converter.convert_volumetric_flow_rate,
        "htc": converter.convert_heat_transfer_coefficient,
        "thermal_cond": converter.convert_thermal_conductivity,
        "fouling": converter.convert_fouling_resistance,
        "heat_duty": converter.convert_heat_duty,
        "area": converter.convert_area,
        "length": converter.convert_length,
        "viscosity": converter.convert_viscosity,
        "density": converter.convert_density,
        "specific_heat": converter.convert_specific_heat,
        "velocity": converter.convert_velocity,
    }

    if quantity_type not in converters:
        raise ValueError(f"Unknown quantity type: {quantity_type}")

    return converters[quantity_type](value, from_unit, to_unit)


# Export all classes and functions
__all__ = [
    "TemperatureUnit",
    "PressureUnit",
    "FlowRateUnit",
    "HeatTransferCoefficientUnit",
    "ThermalConductivityUnit",
    "FoulingResistanceUnit",
    "HeatDutyUnit",
    "AreaUnit",
    "LengthUnit",
    "ViscosityUnit",
    "DensityUnit",
    "SpecificHeatUnit",
    "VelocityUnit",
    "ConversionFactor",
    "UnitConverter",
    "convert",
]
