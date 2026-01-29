"""
Conversion context definitions for GL-FOUND-X-003 Unit Conversion Engine.

This module defines the ConversionContext dataclass that captures all contextual
parameters required for unit conversions in sustainability reporting, including:

- GWP (Global Warming Potential) version for CO2e conversions
- Basis (LHV/HHV for energy, wet/dry for mass)
- Reference conditions for gas volume conversions (temperature, pressure)
- Atmospheric pressure for gauge-to-absolute pressure conversions

All functions in this module are pure and deterministic with no side effects.

Example:
    >>> from gl_normalizer_core.conversion.contexts import ConversionContext
    >>> context = ConversionContext(
    ...     gwp_version="AR6",
    ...     basis="LHV",
    ...     temperature_ref=15.0,
    ...     pressure_ref=101.325,
    ... )
    >>> print(context.is_stp)
    False
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class GWPVersion(str, Enum):
    """
    IPCC Global Warming Potential assessment report versions.

    The GWP values differ between IPCC assessment reports. Regulatory
    frameworks typically specify which version to use for compliance.

    Attributes:
        AR4: Fourth Assessment Report (2007) - 100-year GWP values.
        AR5: Fifth Assessment Report (2014) - 100-year GWP values.
        AR6: Sixth Assessment Report (2021) - 100-year GWP values.
    """

    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"


class EnergyBasis(str, Enum):
    """
    Energy content basis for fuel conversions.

    Attributes:
        LHV: Lower Heating Value (net calorific value) - excludes latent heat.
        HHV: Higher Heating Value (gross calorific value) - includes latent heat.
    """

    LHV = "LHV"
    HHV = "HHV"


class MassBasis(str, Enum):
    """
    Mass measurement basis for materials with moisture content.

    Attributes:
        WET: Wet basis - includes moisture content.
        DRY: Dry basis - excludes moisture content.
    """

    WET = "wet"
    DRY = "dry"


class PressureMode(str, Enum):
    """
    Pressure measurement mode.

    Attributes:
        ABSOLUTE: Absolute pressure referenced to perfect vacuum.
        GAUGE: Gauge pressure referenced to atmospheric pressure.
    """

    ABSOLUTE = "absolute"
    GAUGE = "gauge"


# Standard reference conditions
STP_TEMPERATURE_C: float = 0.0  # Standard Temperature: 0 degrees Celsius
STP_PRESSURE_KPA: float = 101.325  # Standard Pressure: 101.325 kPa

NTP_TEMPERATURE_C: float = 20.0  # Normal Temperature: 20 degrees Celsius
NTP_PRESSURE_KPA: float = 101.325  # Normal Pressure: 101.325 kPa

ISO_TEMPERATURE_C: float = 15.0  # ISO reference temperature: 15 degrees Celsius
ISO_PRESSURE_KPA: float = 101.325  # ISO reference pressure: 101.325 kPa

DEFAULT_ATMOSPHERIC_PRESSURE_KPA: float = 101.325  # Sea level atmospheric pressure


@dataclass(frozen=True)
class ConversionContext:
    """
    Immutable context for unit conversions containing all required parameters.

    This dataclass captures the contextual information needed for accurate
    unit conversions in sustainability reporting. It is immutable (frozen=True)
    to ensure deterministic conversions and support for hash-based caching.

    Attributes:
        gwp_version: IPCC assessment report version for GWP values (AR4, AR5, AR6).
        basis: Energy basis for fuel conversions (LHV or HHV).
        mass_basis: Mass basis for materials with moisture (wet or dry).
        temperature_ref: Reference temperature in degrees Celsius for gas volumes.
        pressure_ref: Reference pressure in kPa (absolute) for gas volumes.
        atmospheric_pressure: Local atmospheric pressure in kPa for gauge conversions.
        pressure_mode: Input pressure mode (absolute or gauge).

    Example:
        >>> context = ConversionContext(
        ...     gwp_version="AR6",
        ...     basis="LHV",
        ...     temperature_ref=15.0,
        ...     pressure_ref=101.325,
        ... )
        >>> print(context.gwp_version)
        AR6
    """

    gwp_version: Optional[str] = field(default=None)
    basis: Optional[str] = field(default=None)
    mass_basis: Optional[str] = field(default=None)
    temperature_ref: Optional[float] = field(default=None)
    pressure_ref: Optional[float] = field(default=None)
    atmospheric_pressure: float = field(default=DEFAULT_ATMOSPHERIC_PRESSURE_KPA)
    pressure_mode: str = field(default=PressureMode.ABSOLUTE.value)

    def __post_init__(self) -> None:
        """Validate context parameters after initialization."""
        # Validate gwp_version if provided
        if self.gwp_version is not None:
            valid_gwp = {v.value for v in GWPVersion}
            if self.gwp_version not in valid_gwp:
                raise ValueError(
                    f"Invalid gwp_version '{self.gwp_version}'. "
                    f"Must be one of: {', '.join(sorted(valid_gwp))}"
                )

        # Validate basis if provided
        if self.basis is not None:
            valid_basis = {v.value for v in EnergyBasis}
            if self.basis not in valid_basis:
                raise ValueError(
                    f"Invalid basis '{self.basis}'. "
                    f"Must be one of: {', '.join(sorted(valid_basis))}"
                )

        # Validate mass_basis if provided
        if self.mass_basis is not None:
            valid_mass_basis = {v.value for v in MassBasis}
            if self.mass_basis not in valid_mass_basis:
                raise ValueError(
                    f"Invalid mass_basis '{self.mass_basis}'. "
                    f"Must be one of: {', '.join(sorted(valid_mass_basis))}"
                )

        # Validate temperature_ref if provided
        if self.temperature_ref is not None:
            if self.temperature_ref < -273.15:
                raise ValueError(
                    f"Invalid temperature_ref {self.temperature_ref}. "
                    "Temperature cannot be below absolute zero (-273.15 C)."
                )

        # Validate pressure_ref if provided
        if self.pressure_ref is not None:
            if self.pressure_ref <= 0:
                raise ValueError(
                    f"Invalid pressure_ref {self.pressure_ref}. "
                    "Pressure must be positive."
                )

        # Validate atmospheric_pressure
        if self.atmospheric_pressure <= 0:
            raise ValueError(
                f"Invalid atmospheric_pressure {self.atmospheric_pressure}. "
                "Atmospheric pressure must be positive."
            )

        # Validate pressure_mode
        valid_modes = {v.value for v in PressureMode}
        if self.pressure_mode not in valid_modes:
            raise ValueError(
                f"Invalid pressure_mode '{self.pressure_mode}'. "
                f"Must be one of: {', '.join(sorted(valid_modes))}"
            )

    @property
    def is_stp(self) -> bool:
        """
        Check if reference conditions match Standard Temperature and Pressure.

        STP is defined as 0 degrees Celsius and 101.325 kPa.

        Returns:
            True if reference conditions match STP, False otherwise.
        """
        if self.temperature_ref is None or self.pressure_ref is None:
            return False
        return (
            abs(self.temperature_ref - STP_TEMPERATURE_C) < 0.01
            and abs(self.pressure_ref - STP_PRESSURE_KPA) < 0.01
        )

    @property
    def is_ntp(self) -> bool:
        """
        Check if reference conditions match Normal Temperature and Pressure.

        NTP is defined as 20 degrees Celsius and 101.325 kPa.

        Returns:
            True if reference conditions match NTP, False otherwise.
        """
        if self.temperature_ref is None or self.pressure_ref is None:
            return False
        return (
            abs(self.temperature_ref - NTP_TEMPERATURE_C) < 0.01
            and abs(self.pressure_ref - NTP_PRESSURE_KPA) < 0.01
        )

    @property
    def is_iso(self) -> bool:
        """
        Check if reference conditions match ISO reference conditions.

        ISO reference is defined as 15 degrees Celsius and 101.325 kPa.

        Returns:
            True if reference conditions match ISO, False otherwise.
        """
        if self.temperature_ref is None or self.pressure_ref is None:
            return False
        return (
            abs(self.temperature_ref - ISO_TEMPERATURE_C) < 0.01
            and abs(self.pressure_ref - ISO_PRESSURE_KPA) < 0.01
        )

    @property
    def has_reference_conditions(self) -> bool:
        """
        Check if reference conditions are fully specified.

        Returns:
            True if both temperature and pressure references are set.
        """
        return self.temperature_ref is not None and self.pressure_ref is not None

    @property
    def has_gwp_version(self) -> bool:
        """
        Check if GWP version is specified.

        Returns:
            True if gwp_version is set.
        """
        return self.gwp_version is not None

    @property
    def has_basis(self) -> bool:
        """
        Check if energy basis is specified.

        Returns:
            True if basis is set.
        """
        return self.basis is not None

    def with_gwp_version(self, gwp_version: str) -> "ConversionContext":
        """
        Create a new context with updated GWP version.

        Args:
            gwp_version: New GWP version (AR4, AR5, AR6).

        Returns:
            New ConversionContext with updated gwp_version.
        """
        return ConversionContext(
            gwp_version=gwp_version,
            basis=self.basis,
            mass_basis=self.mass_basis,
            temperature_ref=self.temperature_ref,
            pressure_ref=self.pressure_ref,
            atmospheric_pressure=self.atmospheric_pressure,
            pressure_mode=self.pressure_mode,
        )

    def with_basis(self, basis: str) -> "ConversionContext":
        """
        Create a new context with updated energy basis.

        Args:
            basis: New energy basis (LHV or HHV).

        Returns:
            New ConversionContext with updated basis.
        """
        return ConversionContext(
            gwp_version=self.gwp_version,
            basis=basis,
            mass_basis=self.mass_basis,
            temperature_ref=self.temperature_ref,
            pressure_ref=self.pressure_ref,
            atmospheric_pressure=self.atmospheric_pressure,
            pressure_mode=self.pressure_mode,
        )

    def with_reference_conditions(
        self,
        temperature_c: float,
        pressure_kpa: float,
    ) -> "ConversionContext":
        """
        Create a new context with updated reference conditions.

        Args:
            temperature_c: Reference temperature in degrees Celsius.
            pressure_kpa: Reference pressure in kPa (absolute).

        Returns:
            New ConversionContext with updated reference conditions.
        """
        return ConversionContext(
            gwp_version=self.gwp_version,
            basis=self.basis,
            mass_basis=self.mass_basis,
            temperature_ref=temperature_c,
            pressure_ref=pressure_kpa,
            atmospheric_pressure=self.atmospheric_pressure,
            pressure_mode=self.pressure_mode,
        )

    def with_atmospheric_pressure(
        self,
        atmospheric_pressure: float,
    ) -> "ConversionContext":
        """
        Create a new context with updated atmospheric pressure.

        Args:
            atmospheric_pressure: Local atmospheric pressure in kPa.

        Returns:
            New ConversionContext with updated atmospheric pressure.
        """
        return ConversionContext(
            gwp_version=self.gwp_version,
            basis=self.basis,
            mass_basis=self.mass_basis,
            temperature_ref=self.temperature_ref,
            pressure_ref=self.pressure_ref,
            atmospheric_pressure=atmospheric_pressure,
            pressure_mode=self.pressure_mode,
        )

    @classmethod
    def create_stp(cls) -> "ConversionContext":
        """
        Create a context with Standard Temperature and Pressure conditions.

        Returns:
            ConversionContext with STP reference conditions (0 C, 101.325 kPa).
        """
        return cls(
            temperature_ref=STP_TEMPERATURE_C,
            pressure_ref=STP_PRESSURE_KPA,
        )

    @classmethod
    def create_ntp(cls) -> "ConversionContext":
        """
        Create a context with Normal Temperature and Pressure conditions.

        Returns:
            ConversionContext with NTP reference conditions (20 C, 101.325 kPa).
        """
        return cls(
            temperature_ref=NTP_TEMPERATURE_C,
            pressure_ref=NTP_PRESSURE_KPA,
        )

    @classmethod
    def create_iso(cls) -> "ConversionContext":
        """
        Create a context with ISO reference conditions.

        Returns:
            ConversionContext with ISO reference conditions (15 C, 101.325 kPa).
        """
        return cls(
            temperature_ref=ISO_TEMPERATURE_C,
            pressure_ref=ISO_PRESSURE_KPA,
        )

    @classmethod
    def create_default(cls) -> "ConversionContext":
        """
        Create a context with default values for all parameters.

        Default context uses:
        - AR6 GWP version (most recent IPCC report)
        - LHV energy basis (common in European regulations)
        - ISO reference conditions (15 C, 101.325 kPa)

        Returns:
            ConversionContext with sensible defaults.
        """
        return cls(
            gwp_version=GWPVersion.AR6.value,
            basis=EnergyBasis.LHV.value,
            temperature_ref=ISO_TEMPERATURE_C,
            pressure_ref=ISO_PRESSURE_KPA,
        )


def gauge_to_absolute(
    gauge_pressure_kpa: float,
    atmospheric_pressure_kpa: float = DEFAULT_ATMOSPHERIC_PRESSURE_KPA,
) -> float:
    """
    Convert gauge pressure to absolute pressure.

    Gauge pressure is measured relative to atmospheric pressure.
    Absolute pressure is measured relative to perfect vacuum.

    Args:
        gauge_pressure_kpa: Gauge pressure in kPa.
        atmospheric_pressure_kpa: Local atmospheric pressure in kPa.

    Returns:
        Absolute pressure in kPa.

    Example:
        >>> gauge_to_absolute(100.0, 101.325)
        201.325
    """
    return gauge_pressure_kpa + atmospheric_pressure_kpa


def absolute_to_gauge(
    absolute_pressure_kpa: float,
    atmospheric_pressure_kpa: float = DEFAULT_ATMOSPHERIC_PRESSURE_KPA,
) -> float:
    """
    Convert absolute pressure to gauge pressure.

    Gauge pressure is measured relative to atmospheric pressure.
    Absolute pressure is measured relative to perfect vacuum.

    Args:
        absolute_pressure_kpa: Absolute pressure in kPa.
        atmospheric_pressure_kpa: Local atmospheric pressure in kPa.

    Returns:
        Gauge pressure in kPa.

    Raises:
        ValueError: If absolute pressure is less than atmospheric.

    Example:
        >>> absolute_to_gauge(201.325, 101.325)
        100.0
    """
    gauge = absolute_pressure_kpa - atmospheric_pressure_kpa
    if gauge < 0:
        raise ValueError(
            f"Absolute pressure ({absolute_pressure_kpa} kPa) cannot be less than "
            f"atmospheric pressure ({atmospheric_pressure_kpa} kPa) for gauge conversion."
        )
    return gauge


def celsius_to_kelvin(temperature_c: float) -> float:
    """
    Convert temperature from Celsius to Kelvin.

    Args:
        temperature_c: Temperature in degrees Celsius.

    Returns:
        Temperature in Kelvin.

    Raises:
        ValueError: If temperature is below absolute zero.

    Example:
        >>> celsius_to_kelvin(0.0)
        273.15
    """
    if temperature_c < -273.15:
        raise ValueError(
            f"Temperature {temperature_c} C is below absolute zero (-273.15 C)."
        )
    return temperature_c + 273.15


def kelvin_to_celsius(temperature_k: float) -> float:
    """
    Convert temperature from Kelvin to Celsius.

    Args:
        temperature_k: Temperature in Kelvin.

    Returns:
        Temperature in degrees Celsius.

    Raises:
        ValueError: If temperature is below absolute zero.

    Example:
        >>> kelvin_to_celsius(273.15)
        0.0
    """
    if temperature_k < 0:
        raise ValueError(
            f"Temperature {temperature_k} K is below absolute zero (0 K)."
        )
    return temperature_k - 273.15


__all__ = [
    "ConversionContext",
    "GWPVersion",
    "EnergyBasis",
    "MassBasis",
    "PressureMode",
    "STP_TEMPERATURE_C",
    "STP_PRESSURE_KPA",
    "NTP_TEMPERATURE_C",
    "NTP_PRESSURE_KPA",
    "ISO_TEMPERATURE_C",
    "ISO_PRESSURE_KPA",
    "DEFAULT_ATMOSPHERIC_PRESSURE_KPA",
    "gauge_to_absolute",
    "absolute_to_gauge",
    "celsius_to_kelvin",
    "kelvin_to_celsius",
]
