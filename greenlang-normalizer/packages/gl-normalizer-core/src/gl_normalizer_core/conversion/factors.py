"""
Conversion factor registry for GL-FOUND-X-003 Unit Conversion Engine.

This module provides a registry of conversion factors organized by dimension,
with support for versioning, deprecation tracking, and YAML-based configuration.
All functions are pure and deterministic with no I/O operations at runtime.

The registry includes:
- Base SI unit conversions
- Energy conversions (with LHV/HHV basis)
- Mass conversions
- Volume conversions
- Temperature conversions (affine transformations)
- Pressure conversions (with gauge/absolute handling)
- GHG emissions (GWP-versioned CO2e conversions)

Example:
    >>> from gl_normalizer_core.conversion.factors import ConversionFactorRegistry
    >>> registry = ConversionFactorRegistry()
    >>> factor = registry.get_factor("kWh", "MJ")
    >>> print(factor.value)
    3.6
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

from gl_normalizer_core.conversion.contexts import GWPVersion, EnergyBasis


class ConversionType(str, Enum):
    """
    Type of conversion operation.

    Attributes:
        MULTIPLY: Simple multiplication by factor.
        DIVIDE: Division by factor.
        AFFINE: Affine transformation (offset + scale).
        INVERSE: Inverse of another conversion.
        COMPOUND: Multi-step compound conversion.
    """

    MULTIPLY = "multiply"
    DIVIDE = "divide"
    AFFINE = "affine"
    INVERSE = "inverse"
    COMPOUND = "compound"


class FactorStatus(str, Enum):
    """
    Status of a conversion factor.

    Attributes:
        ACTIVE: Factor is current and recommended for use.
        DEPRECATED: Factor is outdated; newer version available.
        RETIRED: Factor should not be used; may produce warnings.
    """

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


@dataclass(frozen=True)
class ConversionFactor:
    """
    Immutable conversion factor with metadata.

    Attributes:
        from_unit: Source unit.
        to_unit: Target unit.
        value: Conversion factor value.
        conversion_type: Type of conversion operation.
        offset: Offset for affine conversions (temperature).
        dimension: Physical dimension of the units.
        version: Version of this factor.
        status: Factor status (active, deprecated, retired).
        source: Source of the factor (standard, regulation, etc.).
        notes: Additional notes about the factor.
        superseded_by: Version that supersedes this factor (if deprecated).
    """

    from_unit: str
    to_unit: str
    value: float
    conversion_type: ConversionType = ConversionType.MULTIPLY
    offset: float = 0.0
    dimension: Optional[str] = None
    version: str = "1.0.0"
    status: FactorStatus = FactorStatus.ACTIVE
    source: Optional[str] = None
    notes: Optional[str] = None
    superseded_by: Optional[str] = None

    def apply(self, input_value: float) -> float:
        """
        Apply this conversion factor to a value.

        Args:
            input_value: Value to convert.

        Returns:
            Converted value.

        Example:
            >>> factor = ConversionFactor("kWh", "MJ", 3.6)
            >>> factor.apply(100)
            360.0
        """
        if self.conversion_type == ConversionType.MULTIPLY:
            return input_value * self.value
        elif self.conversion_type == ConversionType.DIVIDE:
            return input_value / self.value
        elif self.conversion_type == ConversionType.AFFINE:
            return input_value * self.value + self.offset
        else:
            return input_value * self.value

    def apply_inverse(self, output_value: float) -> float:
        """
        Apply the inverse of this conversion factor.

        Args:
            output_value: Value to convert back.

        Returns:
            Original value.

        Example:
            >>> factor = ConversionFactor("kWh", "MJ", 3.6)
            >>> factor.apply_inverse(360.0)
            100.0
        """
        if self.conversion_type == ConversionType.MULTIPLY:
            return output_value / self.value
        elif self.conversion_type == ConversionType.DIVIDE:
            return output_value * self.value
        elif self.conversion_type == ConversionType.AFFINE:
            return (output_value - self.offset) / self.value
        else:
            return output_value / self.value

    @property
    def is_deprecated(self) -> bool:
        """Check if this factor is deprecated."""
        return self.status == FactorStatus.DEPRECATED

    @property
    def is_active(self) -> bool:
        """Check if this factor is active."""
        return self.status == FactorStatus.ACTIVE


@dataclass(frozen=True)
class GWPFactor:
    """
    Global Warming Potential factor for a greenhouse gas.

    GWP values express the warming effect of a gas relative to CO2
    over a specified time horizon (typically 100 years).

    Attributes:
        gas: Gas identifier (e.g., CH4, N2O, SF6).
        gwp_version: IPCC assessment report version.
        value: GWP value relative to CO2.
        time_horizon_years: Time horizon for the GWP (default 100).
        source: Source reference.
    """

    gas: str
    gwp_version: str
    value: float
    time_horizon_years: int = 100
    source: Optional[str] = None


# =============================================================================
# Conversion Factor Tables
# =============================================================================

# Energy conversions (all relative to MJ as base)
ENERGY_FACTORS: Dict[Tuple[str, str], ConversionFactor] = {
    # kWh conversions
    ("kWh", "MJ"): ConversionFactor("kWh", "MJ", 3.6, dimension="energy"),
    ("MJ", "kWh"): ConversionFactor("MJ", "kWh", 1.0 / 3.6, dimension="energy"),
    ("kWh", "GJ"): ConversionFactor("kWh", "GJ", 0.0036, dimension="energy"),
    ("GJ", "kWh"): ConversionFactor("GJ", "kWh", 277.777778, dimension="energy"),

    # MWh conversions
    ("MWh", "MJ"): ConversionFactor("MWh", "MJ", 3600.0, dimension="energy"),
    ("MJ", "MWh"): ConversionFactor("MJ", "MWh", 1.0 / 3600.0, dimension="energy"),
    ("MWh", "kWh"): ConversionFactor("MWh", "kWh", 1000.0, dimension="energy"),
    ("kWh", "MWh"): ConversionFactor("kWh", "MWh", 0.001, dimension="energy"),
    ("MWh", "GJ"): ConversionFactor("MWh", "GJ", 3.6, dimension="energy"),
    ("GJ", "MWh"): ConversionFactor("GJ", "MWh", 1.0 / 3.6, dimension="energy"),

    # GJ conversions
    ("GJ", "MJ"): ConversionFactor("GJ", "MJ", 1000.0, dimension="energy"),
    ("MJ", "GJ"): ConversionFactor("MJ", "GJ", 0.001, dimension="energy"),
    ("TJ", "GJ"): ConversionFactor("TJ", "GJ", 1000.0, dimension="energy"),
    ("GJ", "TJ"): ConversionFactor("GJ", "TJ", 0.001, dimension="energy"),
    ("TJ", "MJ"): ConversionFactor("TJ", "MJ", 1000000.0, dimension="energy"),
    ("MJ", "TJ"): ConversionFactor("MJ", "TJ", 0.000001, dimension="energy"),

    # BTU conversions
    ("BTU", "MJ"): ConversionFactor("BTU", "MJ", 0.001055056, dimension="energy"),
    ("MJ", "BTU"): ConversionFactor("MJ", "BTU", 947.817, dimension="energy"),
    ("MMBTU", "MJ"): ConversionFactor("MMBTU", "MJ", 1055.056, dimension="energy"),
    ("MJ", "MMBTU"): ConversionFactor("MJ", "MMBTU", 0.000947817, dimension="energy"),
    ("MMBTU", "GJ"): ConversionFactor("MMBTU", "GJ", 1.055056, dimension="energy"),
    ("GJ", "MMBTU"): ConversionFactor("GJ", "MMBTU", 0.947817, dimension="energy"),

    # Therm conversions
    ("therm", "MJ"): ConversionFactor("therm", "MJ", 105.4804, dimension="energy"),
    ("MJ", "therm"): ConversionFactor("MJ", "therm", 0.009478171, dimension="energy"),

    # Joule conversions
    ("J", "MJ"): ConversionFactor("J", "MJ", 0.000001, dimension="energy"),
    ("MJ", "J"): ConversionFactor("MJ", "J", 1000000.0, dimension="energy"),
    ("kJ", "MJ"): ConversionFactor("kJ", "MJ", 0.001, dimension="energy"),
    ("MJ", "kJ"): ConversionFactor("MJ", "kJ", 1000.0, dimension="energy"),
}


# Mass conversions (all relative to kg as base)
MASS_FACTORS: Dict[Tuple[str, str], ConversionFactor] = {
    # Metric mass
    ("g", "kg"): ConversionFactor("g", "kg", 0.001, dimension="mass"),
    ("kg", "g"): ConversionFactor("kg", "g", 1000.0, dimension="mass"),
    ("mg", "kg"): ConversionFactor("mg", "kg", 0.000001, dimension="mass"),
    ("kg", "mg"): ConversionFactor("kg", "mg", 1000000.0, dimension="mass"),
    ("ug", "kg"): ConversionFactor("ug", "kg", 1e-9, dimension="mass"),
    ("kg", "ug"): ConversionFactor("kg", "ug", 1e9, dimension="mass"),
    ("t", "kg"): ConversionFactor("t", "kg", 1000.0, dimension="mass"),
    ("kg", "t"): ConversionFactor("kg", "t", 0.001, dimension="mass"),
    ("tonne", "kg"): ConversionFactor("tonne", "kg", 1000.0, dimension="mass"),
    ("kg", "tonne"): ConversionFactor("kg", "tonne", 0.001, dimension="mass"),
    ("metric_ton", "kg"): ConversionFactor("metric_ton", "kg", 1000.0, dimension="mass"),
    ("kg", "metric_ton"): ConversionFactor("kg", "metric_ton", 0.001, dimension="mass"),
    ("Mt", "kg"): ConversionFactor("Mt", "kg", 1000000000.0, dimension="mass"),
    ("kg", "Mt"): ConversionFactor("kg", "Mt", 0.000000001, dimension="mass"),
    ("Gt", "kg"): ConversionFactor("Gt", "kg", 1e12, dimension="mass"),
    ("kg", "Gt"): ConversionFactor("kg", "Gt", 1e-12, dimension="mass"),
    ("kt", "kg"): ConversionFactor("kt", "kg", 1000000.0, dimension="mass"),
    ("kg", "kt"): ConversionFactor("kg", "kt", 0.000001, dimension="mass"),

    # Imperial mass
    ("lb", "kg"): ConversionFactor("lb", "kg", 0.45359237, dimension="mass"),
    ("kg", "lb"): ConversionFactor("kg", "lb", 2.20462262, dimension="mass"),
    ("oz", "kg"): ConversionFactor("oz", "kg", 0.0283495231, dimension="mass"),
    ("kg", "oz"): ConversionFactor("kg", "oz", 35.2739619, dimension="mass"),
    ("short_ton", "kg"): ConversionFactor("short_ton", "kg", 907.18474, dimension="mass"),
    ("kg", "short_ton"): ConversionFactor("kg", "short_ton", 0.00110231131, dimension="mass"),
    ("long_ton", "kg"): ConversionFactor("long_ton", "kg", 1016.0469088, dimension="mass"),
    ("kg", "long_ton"): ConversionFactor("kg", "long_ton", 0.000984206528, dimension="mass"),

    # Troy mass (precious metals)
    ("oz_troy", "kg"): ConversionFactor("oz_troy", "kg", 0.0311034768, dimension="mass"),
    ("kg", "oz_troy"): ConversionFactor("kg", "oz_troy", 32.1507466, dimension="mass"),
    ("lb_troy", "kg"): ConversionFactor("lb_troy", "kg", 0.3732417216, dimension="mass"),
    ("kg", "lb_troy"): ConversionFactor("kg", "lb_troy", 2.67922888, dimension="mass"),
}


# Volume conversions (all relative to m3 as base)
VOLUME_FACTORS: Dict[Tuple[str, str], ConversionFactor] = {
    # Metric volume
    ("L", "m3"): ConversionFactor("L", "m3", 0.001, dimension="volume"),
    ("m3", "L"): ConversionFactor("m3", "L", 1000.0, dimension="volume"),
    ("mL", "m3"): ConversionFactor("mL", "m3", 0.000001, dimension="volume"),
    ("m3", "mL"): ConversionFactor("m3", "mL", 1000000.0, dimension="volume"),
    ("cL", "m3"): ConversionFactor("cL", "m3", 0.00001, dimension="volume"),
    ("m3", "cL"): ConversionFactor("m3", "cL", 100000.0, dimension="volume"),
    ("dL", "m3"): ConversionFactor("dL", "m3", 0.0001, dimension="volume"),
    ("m3", "dL"): ConversionFactor("m3", "dL", 10000.0, dimension="volume"),
    ("hL", "m3"): ConversionFactor("hL", "m3", 0.1, dimension="volume"),
    ("m3", "hL"): ConversionFactor("m3", "hL", 10.0, dimension="volume"),
    ("kL", "m3"): ConversionFactor("kL", "m3", 1.0, dimension="volume"),
    ("m3", "kL"): ConversionFactor("m3", "kL", 1.0, dimension="volume"),
    ("ML", "m3"): ConversionFactor("ML", "m3", 1000.0, dimension="volume"),
    ("m3", "ML"): ConversionFactor("m3", "ML", 0.001, dimension="volume"),

    # US customary volume
    ("fl_oz_US", "m3"): ConversionFactor("fl_oz_US", "m3", 2.95735296e-5, dimension="volume"),
    ("m3", "fl_oz_US"): ConversionFactor("m3", "fl_oz_US", 33814.0227, dimension="volume"),
    ("cup_US", "m3"): ConversionFactor("cup_US", "m3", 2.365882365e-4, dimension="volume"),
    ("m3", "cup_US"): ConversionFactor("m3", "cup_US", 4226.75284, dimension="volume"),
    ("pt_US", "m3"): ConversionFactor("pt_US", "m3", 4.73176473e-4, dimension="volume"),
    ("m3", "pt_US"): ConversionFactor("m3", "pt_US", 2113.37642, dimension="volume"),
    ("qt_US", "m3"): ConversionFactor("qt_US", "m3", 9.46352946e-4, dimension="volume"),
    ("m3", "qt_US"): ConversionFactor("m3", "qt_US", 1056.68821, dimension="volume"),
    ("gal", "m3"): ConversionFactor("gal", "m3", 0.003785411784, dimension="volume"),
    ("m3", "gal"): ConversionFactor("m3", "gal", 264.172052, dimension="volume"),
    ("gal_US", "m3"): ConversionFactor("gal_US", "m3", 0.003785411784, dimension="volume"),
    ("m3", "gal_US"): ConversionFactor("m3", "gal_US", 264.172052, dimension="volume"),
    ("bbl", "m3"): ConversionFactor("bbl", "m3", 0.158987294928, dimension="volume"),
    ("m3", "bbl"): ConversionFactor("m3", "bbl", 6.28981077, dimension="volume"),
    ("Mbbl", "m3"): ConversionFactor("Mbbl", "m3", 158.987294928, dimension="volume"),
    ("m3", "Mbbl"): ConversionFactor("m3", "Mbbl", 0.00628981077, dimension="volume"),

    # UK Imperial volume
    ("fl_oz_UK", "m3"): ConversionFactor("fl_oz_UK", "m3", 2.84130625e-5, dimension="volume"),
    ("m3", "fl_oz_UK"): ConversionFactor("m3", "fl_oz_UK", 35195.0797, dimension="volume"),
    ("pt_UK", "m3"): ConversionFactor("pt_UK", "m3", 5.6826125e-4, dimension="volume"),
    ("m3", "pt_UK"): ConversionFactor("m3", "pt_UK", 1759.75399, dimension="volume"),
    ("gal_UK", "m3"): ConversionFactor("gal_UK", "m3", 0.00454609, dimension="volume"),
    ("m3", "gal_UK"): ConversionFactor("m3", "gal_UK", 219.969157, dimension="volume"),

    # Cubic units
    ("in3", "m3"): ConversionFactor("in3", "m3", 1.6387064e-5, dimension="volume"),
    ("m3", "in3"): ConversionFactor("m3", "in3", 61023.7441, dimension="volume"),
    ("ft3", "m3"): ConversionFactor("ft3", "m3", 0.0283168466, dimension="volume"),
    ("m3", "ft3"): ConversionFactor("m3", "ft3", 35.3146667, dimension="volume"),
    ("yd3", "m3"): ConversionFactor("yd3", "m3", 0.764554858, dimension="volume"),
    ("m3", "yd3"): ConversionFactor("m3", "yd3", 1.30795062, dimension="volume"),
    ("acre_ft", "m3"): ConversionFactor("acre_ft", "m3", 1233.48184, dimension="volume"),
    ("m3", "acre_ft"): ConversionFactor("m3", "acre_ft", 0.000810713194, dimension="volume"),

    # Liter aliases
    ("liter", "m3"): ConversionFactor("liter", "m3", 0.001, dimension="volume"),
    ("m3", "liter"): ConversionFactor("m3", "liter", 1000.0, dimension="volume"),
    ("litre", "m3"): ConversionFactor("litre", "m3", 0.001, dimension="volume"),
    ("m3", "litre"): ConversionFactor("m3", "litre", 1000.0, dimension="volume"),
}


# Temperature conversions (using affine transformations)
# Formula: output = input * value + offset
TEMPERATURE_FACTORS: Dict[Tuple[str, str], ConversionFactor] = {
    # Celsius identity
    ("degC", "degC"): ConversionFactor(
        "degC", "degC", 1.0, ConversionType.MULTIPLY, offset=0.0, dimension="temperature"
    ),

    # Celsius to Kelvin: K = C + 273.15
    ("degC", "K"): ConversionFactor(
        "degC", "K", 1.0, ConversionType.AFFINE, offset=273.15, dimension="temperature"
    ),
    # Kelvin to Celsius: C = K - 273.15
    ("K", "degC"): ConversionFactor(
        "K", "degC", 1.0, ConversionType.AFFINE, offset=-273.15, dimension="temperature"
    ),

    # Fahrenheit to Celsius: C = (F - 32) * 5/9
    ("degF", "degC"): ConversionFactor(
        "degF", "degC", 5.0 / 9.0, ConversionType.AFFINE, offset=-32.0 * 5.0 / 9.0, dimension="temperature"
    ),
    # Celsius to Fahrenheit: F = C * 9/5 + 32
    ("degC", "degF"): ConversionFactor(
        "degC", "degF", 9.0 / 5.0, ConversionType.AFFINE, offset=32.0, dimension="temperature"
    ),

    # Fahrenheit to Kelvin: K = (F - 32) * 5/9 + 273.15 = F * 5/9 + 255.3722...
    ("degF", "K"): ConversionFactor(
        "degF", "K", 5.0 / 9.0, ConversionType.AFFINE, offset=(-32.0 * 5.0 / 9.0) + 273.15, dimension="temperature"
    ),
    # Kelvin to Fahrenheit: F = (K - 273.15) * 9/5 + 32 = K * 9/5 - 459.67
    ("K", "degF"): ConversionFactor(
        "K", "degF", 9.0 / 5.0, ConversionType.AFFINE, offset=-459.67, dimension="temperature"
    ),

    # Rankine to Celsius: C = (R - 491.67) * 5/9 = R * 5/9 - 273.15
    ("degR", "degC"): ConversionFactor(
        "degR", "degC", 5.0 / 9.0, ConversionType.AFFINE, offset=-273.15, dimension="temperature"
    ),
    # Celsius to Rankine: R = (C + 273.15) * 9/5 = C * 9/5 + 491.67
    ("degC", "degR"): ConversionFactor(
        "degC", "degR", 9.0 / 5.0, ConversionType.AFFINE, offset=491.67, dimension="temperature"
    ),

    # Rankine to Kelvin: K = R * 5/9
    ("degR", "K"): ConversionFactor(
        "degR", "K", 5.0 / 9.0, ConversionType.MULTIPLY, offset=0.0, dimension="temperature"
    ),
    # Kelvin to Rankine: R = K * 9/5
    ("K", "degR"): ConversionFactor(
        "K", "degR", 9.0 / 5.0, ConversionType.MULTIPLY, offset=0.0, dimension="temperature"
    ),
}


# Pressure conversions (all relative to kPa_abs as base)
# Note: Gauge pressures (kPag, psig) use affine transformation with atmospheric offset
PRESSURE_FACTORS: Dict[Tuple[str, str], ConversionFactor] = {
    # Bar conversions
    ("bar", "kPa_abs"): ConversionFactor("bar", "kPa_abs", 100.0, dimension="pressure"),
    ("kPa_abs", "bar"): ConversionFactor("kPa_abs", "bar", 0.01, dimension="pressure"),
    ("mbar", "kPa_abs"): ConversionFactor("mbar", "kPa_abs", 0.1, dimension="pressure"),
    ("kPa_abs", "mbar"): ConversionFactor("kPa_abs", "mbar", 10.0, dimension="pressure"),

    # Atmosphere conversions
    ("atm", "kPa_abs"): ConversionFactor("atm", "kPa_abs", 101.325, dimension="pressure"),
    ("kPa_abs", "atm"): ConversionFactor("kPa_abs", "atm", 0.00986923267, dimension="pressure"),

    # PSI conversions (absolute)
    ("psi", "kPa_abs"): ConversionFactor("psi", "kPa_abs", 6.89475729, dimension="pressure"),
    ("kPa_abs", "psi"): ConversionFactor("kPa_abs", "psi", 0.145037738, dimension="pressure"),
    ("psia", "kPa_abs"): ConversionFactor("psia", "kPa_abs", 6.89475729, dimension="pressure"),
    ("kPa_abs", "psia"): ConversionFactor("kPa_abs", "psia", 0.145037738, dimension="pressure"),

    # Gauge pressure conversions (add atmospheric pressure: 101.325 kPa)
    # kPag to kPa_abs: kPa_abs = kPag + 101.325
    ("kPag", "kPa_abs"): ConversionFactor(
        "kPag", "kPa_abs", 1.0, ConversionType.AFFINE, offset=101.325, dimension="pressure"
    ),
    ("kPa_abs", "kPag"): ConversionFactor(
        "kPa_abs", "kPag", 1.0, ConversionType.AFFINE, offset=-101.325, dimension="pressure"
    ),
    # psig to kPa_abs: kPa_abs = psig * 6.89476 + 101.325
    ("psig", "kPa_abs"): ConversionFactor(
        "psig", "kPa_abs", 6.89475729, ConversionType.AFFINE, offset=101.325, dimension="pressure"
    ),
    ("kPa_abs", "psig"): ConversionFactor(
        "kPa_abs", "psig", 0.145037738, ConversionType.AFFINE, offset=-14.6959, dimension="pressure"
    ),

    # Pascal conversions
    ("Pa", "kPa_abs"): ConversionFactor("Pa", "kPa_abs", 0.001, dimension="pressure"),
    ("kPa_abs", "Pa"): ConversionFactor("kPa_abs", "Pa", 1000.0, dimension="pressure"),
    ("hPa", "kPa_abs"): ConversionFactor("hPa", "kPa_abs", 0.1, dimension="pressure"),
    ("kPa_abs", "hPa"): ConversionFactor("kPa_abs", "hPa", 10.0, dimension="pressure"),
    ("MPa", "kPa_abs"): ConversionFactor("MPa", "kPa_abs", 1000.0, dimension="pressure"),
    ("kPa_abs", "MPa"): ConversionFactor("kPa_abs", "MPa", 0.001, dimension="pressure"),

    # mmHg (Torr) conversions
    ("mmHg", "kPa_abs"): ConversionFactor("mmHg", "kPa_abs", 0.133322387, dimension="pressure"),
    ("kPa_abs", "mmHg"): ConversionFactor("kPa_abs", "mmHg", 7.50061683, dimension="pressure"),
    ("Torr", "kPa_abs"): ConversionFactor("Torr", "kPa_abs", 0.133322387, dimension="pressure"),
    ("kPa_abs", "Torr"): ConversionFactor("kPa_abs", "Torr", 7.50061683, dimension="pressure"),

    # inHg conversions
    ("inHg", "kPa_abs"): ConversionFactor("inHg", "kPa_abs", 3.38638816, dimension="pressure"),
    ("kPa_abs", "inHg"): ConversionFactor("kPa_abs", "inHg", 0.295299830, dimension="pressure"),
}


# GWP factors by IPCC assessment report version (100-year horizon)
GWP_FACTORS: Dict[Tuple[str, str], GWPFactor] = {
    # AR4 GWP values (IPCC Fourth Assessment Report, 2007)
    ("CH4", "AR4"): GWPFactor("CH4", "AR4", 25.0, source="IPCC AR4 WG1 Table 2.14"),
    ("N2O", "AR4"): GWPFactor("N2O", "AR4", 298.0, source="IPCC AR4 WG1 Table 2.14"),
    ("SF6", "AR4"): GWPFactor("SF6", "AR4", 22800.0, source="IPCC AR4 WG1 Table 2.14"),
    ("HFC-134a", "AR4"): GWPFactor("HFC-134a", "AR4", 1430.0, source="IPCC AR4 WG1 Table 2.14"),
    ("HFC-32", "AR4"): GWPFactor("HFC-32", "AR4", 675.0, source="IPCC AR4 WG1 Table 2.14"),
    ("NF3", "AR4"): GWPFactor("NF3", "AR4", 17200.0, source="IPCC AR4 WG1 Table 2.14"),
    ("PFC-14", "AR4"): GWPFactor("PFC-14", "AR4", 7390.0, source="IPCC AR4 WG1 Table 2.14"),
    ("PFC-116", "AR4"): GWPFactor("PFC-116", "AR4", 12200.0, source="IPCC AR4 WG1 Table 2.14"),

    # AR5 GWP values (IPCC Fifth Assessment Report, 2014)
    ("CH4", "AR5"): GWPFactor("CH4", "AR5", 28.0, source="IPCC AR5 WG1 Table 8.7"),
    ("N2O", "AR5"): GWPFactor("N2O", "AR5", 265.0, source="IPCC AR5 WG1 Table 8.7"),
    ("SF6", "AR5"): GWPFactor("SF6", "AR5", 23500.0, source="IPCC AR5 WG1 Table 8.7"),
    ("HFC-134a", "AR5"): GWPFactor("HFC-134a", "AR5", 1300.0, source="IPCC AR5 WG1 Table 8.7"),
    ("HFC-32", "AR5"): GWPFactor("HFC-32", "AR5", 677.0, source="IPCC AR5 WG1 Table 8.7"),
    ("NF3", "AR5"): GWPFactor("NF3", "AR5", 16100.0, source="IPCC AR5 WG1 Table 8.7"),
    ("PFC-14", "AR5"): GWPFactor("PFC-14", "AR5", 6630.0, source="IPCC AR5 WG1 Table 8.7"),
    ("PFC-116", "AR5"): GWPFactor("PFC-116", "AR5", 11100.0, source="IPCC AR5 WG1 Table 8.7"),

    # AR6 GWP values (IPCC Sixth Assessment Report, 2021)
    ("CH4", "AR6"): GWPFactor("CH4", "AR6", 27.9, source="IPCC AR6 WG1 Table 7.15"),
    ("N2O", "AR6"): GWPFactor("N2O", "AR6", 273.0, source="IPCC AR6 WG1 Table 7.15"),
    ("SF6", "AR6"): GWPFactor("SF6", "AR6", 25200.0, source="IPCC AR6 WG1 Table 7.15"),
    ("HFC-134a", "AR6"): GWPFactor("HFC-134a", "AR6", 1530.0, source="IPCC AR6 WG1 Table 7.15"),
    ("HFC-32", "AR6"): GWPFactor("HFC-32", "AR6", 771.0, source="IPCC AR6 WG1 Table 7.15"),
    ("NF3", "AR6"): GWPFactor("NF3", "AR6", 17400.0, source="IPCC AR6 WG1 Table 7.15"),
    ("PFC-14", "AR6"): GWPFactor("PFC-14", "AR6", 7380.0, source="IPCC AR6 WG1 Table 7.15"),
    ("PFC-116", "AR6"): GWPFactor("PFC-116", "AR6", 12400.0, source="IPCC AR6 WG1 Table 7.15"),
}


# LHV/HHV conversion ratios for common fuels
# These are the ratios: value * LHV = HHV (or LHV = HHV / value)
LHV_HHV_RATIOS: Dict[str, float] = {
    "natural_gas": 1.109,
    "diesel": 1.065,
    "gasoline": 1.069,
    "coal_bituminous": 1.050,
    "coal_anthracite": 1.030,
    "fuel_oil": 1.055,
    "lpg": 1.080,
    "propane": 1.082,
    "butane": 1.076,
    "hydrogen": 1.183,
    "biogas": 1.109,
    "biomass": 1.070,
    "wood": 1.070,
}


class ConversionFactorRegistry:
    """
    Registry for conversion factors with versioning and deprecation support.

    This class provides a centralized registry for all unit conversion factors,
    with support for:
    - Multiple factor versions
    - Deprecation tracking and warnings
    - Factor lookup by unit pair
    - Custom factor registration

    All methods are pure functions with no I/O operations.

    Example:
        >>> registry = ConversionFactorRegistry()
        >>> factor = registry.get_factor("kWh", "MJ")
        >>> print(factor.value)
        3.6
    """

    def __init__(self) -> None:
        """Initialize the registry with default conversion factors."""
        self._factors: Dict[Tuple[str, str], ConversionFactor] = {}
        self._gwp_factors: Dict[Tuple[str, str], GWPFactor] = {}
        self._deprecated_factors: Dict[Tuple[str, str], ConversionFactor] = {}
        self._version: str = "2026.01.0"

        # Load default factors
        self._load_default_factors()

    def _load_default_factors(self) -> None:
        """Load all default conversion factors into the registry."""
        # Load energy factors
        self._factors.update(ENERGY_FACTORS)

        # Load mass factors
        self._factors.update(MASS_FACTORS)

        # Load volume factors
        self._factors.update(VOLUME_FACTORS)

        # Load temperature factors
        self._factors.update(TEMPERATURE_FACTORS)

        # Load pressure factors
        self._factors.update(PRESSURE_FACTORS)

        # Load GWP factors
        self._gwp_factors.update(GWP_FACTORS)

    @property
    def version(self) -> str:
        """Get the registry version."""
        return self._version

    def get_factor(
        self,
        from_unit: str,
        to_unit: str,
    ) -> Optional[ConversionFactor]:
        """
        Get the conversion factor between two units.

        Args:
            from_unit: Source unit.
            to_unit: Target unit.

        Returns:
            ConversionFactor if found, None otherwise.

        Example:
            >>> registry = ConversionFactorRegistry()
            >>> factor = registry.get_factor("kWh", "MJ")
            >>> print(factor.value)
            3.6
        """
        key = (from_unit, to_unit)
        return self._factors.get(key)

    def get_factor_with_aliases(
        self,
        from_unit: str,
        to_unit: str,
    ) -> Optional[ConversionFactor]:
        """
        Get conversion factor, trying unit aliases if direct lookup fails.

        Args:
            from_unit: Source unit (or alias).
            to_unit: Target unit (or alias).

        Returns:
            ConversionFactor if found, None otherwise.
        """
        # Try direct lookup first
        factor = self.get_factor(from_unit, to_unit)
        if factor is not None:
            return factor

        # Try common aliases
        from_aliases = _get_unit_aliases(from_unit)
        to_aliases = _get_unit_aliases(to_unit)

        for from_alias in from_aliases:
            for to_alias in to_aliases:
                factor = self.get_factor(from_alias, to_alias)
                if factor is not None:
                    return factor

        return None

    def get_gwp_factor(
        self,
        gas: str,
        gwp_version: str,
    ) -> Optional[GWPFactor]:
        """
        Get the GWP factor for a greenhouse gas.

        Args:
            gas: Gas identifier (e.g., CH4, N2O).
            gwp_version: IPCC assessment report version (AR4, AR5, AR6).

        Returns:
            GWPFactor if found, None otherwise.

        Example:
            >>> registry = ConversionFactorRegistry()
            >>> gwp = registry.get_gwp_factor("CH4", "AR6")
            >>> print(gwp.value)
            27.9
        """
        key = (gas, gwp_version)
        return self._gwp_factors.get(key)

    def get_lhv_hhv_ratio(self, fuel: str) -> Optional[float]:
        """
        Get the LHV/HHV conversion ratio for a fuel.

        Args:
            fuel: Fuel identifier (e.g., natural_gas, diesel).

        Returns:
            Ratio value if found, None otherwise.

        Example:
            >>> registry = ConversionFactorRegistry()
            >>> ratio = registry.get_lhv_hhv_ratio("natural_gas")
            >>> print(ratio)
            1.109
        """
        # Normalize fuel name
        fuel_normalized = fuel.lower().replace(" ", "_").replace("-", "_")
        return LHV_HHV_RATIOS.get(fuel_normalized)

    def has_factor(self, from_unit: str, to_unit: str) -> bool:
        """
        Check if a conversion factor exists.

        Args:
            from_unit: Source unit.
            to_unit: Target unit.

        Returns:
            True if factor exists, False otherwise.
        """
        return (from_unit, to_unit) in self._factors

    def get_all_factors_for_dimension(
        self,
        dimension: str,
    ) -> List[ConversionFactor]:
        """
        Get all conversion factors for a specific dimension.

        Args:
            dimension: Physical dimension (e.g., energy, mass).

        Returns:
            List of ConversionFactor objects for that dimension.
        """
        return [
            factor
            for factor in self._factors.values()
            if factor.dimension == dimension
        ]

    def get_supported_units(self, dimension: str) -> FrozenSet[str]:
        """
        Get all supported units for a dimension.

        Args:
            dimension: Physical dimension.

        Returns:
            Frozenset of unit strings.
        """
        units = set()
        for factor in self._factors.values():
            if factor.dimension == dimension:
                units.add(factor.from_unit)
                units.add(factor.to_unit)
        return frozenset(units)

    def get_supported_gases(self) -> FrozenSet[str]:
        """
        Get all supported greenhouse gases for GWP conversion.

        Returns:
            Frozenset of gas identifiers.
        """
        return frozenset(gas for gas, _ in self._gwp_factors.keys())

    def register_factor(self, factor: ConversionFactor) -> None:
        """
        Register a custom conversion factor.

        Args:
            factor: ConversionFactor to register.

        Note:
            This will overwrite any existing factor for the same unit pair.
        """
        key = (factor.from_unit, factor.to_unit)
        self._factors[key] = factor

    def register_gwp_factor(self, gwp_factor: GWPFactor) -> None:
        """
        Register a custom GWP factor.

        Args:
            gwp_factor: GWPFactor to register.
        """
        key = (gwp_factor.gas, gwp_factor.gwp_version)
        self._gwp_factors[key] = gwp_factor

    def is_deprecated(self, from_unit: str, to_unit: str) -> bool:
        """
        Check if a conversion factor is deprecated.

        Args:
            from_unit: Source unit.
            to_unit: Target unit.

        Returns:
            True if the factor is deprecated, False otherwise.
        """
        factor = self.get_factor(from_unit, to_unit)
        return factor is not None and factor.is_deprecated

    def get_deprecation_warning(
        self,
        from_unit: str,
        to_unit: str,
    ) -> Optional[str]:
        """
        Get the deprecation warning for a factor if it is deprecated.

        Args:
            from_unit: Source unit.
            to_unit: Target unit.

        Returns:
            Warning message if deprecated, None otherwise.
        """
        factor = self.get_factor(from_unit, to_unit)
        if factor is None or not factor.is_deprecated:
            return None

        msg = f"Conversion factor {from_unit} -> {to_unit} is deprecated."
        if factor.superseded_by:
            msg += f" Superseded by version {factor.superseded_by}."
        return msg


def _get_unit_aliases(unit: str) -> List[str]:
    """
    Get common aliases for a unit string.

    Args:
        unit: Unit string.

    Returns:
        List of alias strings including the original.
    """
    aliases = [unit]

    # Common normalizations
    unit_lower = unit.lower()
    unit_upper = unit.upper()

    if unit_lower not in aliases:
        aliases.append(unit_lower)

    # Handle common variations
    alias_map = {
        "kwh": ["kWh", "KWH", "kilowatt_hour", "kilowatthour"],
        "mwh": ["MWh", "MWH", "megawatt_hour", "megawatthour"],
        "mj": ["MJ", "megajoule", "megajoules"],
        "gj": ["GJ", "gigajoule", "gigajoules"],
        "kg": ["kilogram", "kilograms"],
        "t": ["tonne", "tonnes", "metric_ton", "metric_tons"],
        "m3": ["m^3", "cubic_meter", "cubic_meters", "cbm"],
        "l": ["L", "liter", "litre", "liters", "litres"],
        "degc": ["C", "celsius", "degrees_celsius"],
        "k": ["K", "kelvin"],
        "kpa_abs": ["kPa", "kilopascal", "kilopascals"],
    }

    for key, values in alias_map.items():
        if unit_lower == key or unit in values:
            for v in values:
                if v not in aliases:
                    aliases.append(v)

    return aliases


def create_inverse_factor(factor: ConversionFactor) -> ConversionFactor:
    """
    Create the inverse of a conversion factor.

    Args:
        factor: Original conversion factor.

    Returns:
        New ConversionFactor for the inverse conversion.

    Example:
        >>> factor = ConversionFactor("kWh", "MJ", 3.6)
        >>> inverse = create_inverse_factor(factor)
        >>> print(inverse.from_unit, inverse.to_unit, inverse.value)
        MJ kWh 0.2777...
    """
    if factor.conversion_type == ConversionType.AFFINE:
        # For affine: y = mx + b, inverse is x = (y - b) / m
        return ConversionFactor(
            from_unit=factor.to_unit,
            to_unit=factor.from_unit,
            value=1.0 / factor.value,
            conversion_type=ConversionType.AFFINE,
            offset=-factor.offset / factor.value,
            dimension=factor.dimension,
            version=factor.version,
            status=factor.status,
            source=factor.source,
        )
    else:
        return ConversionFactor(
            from_unit=factor.to_unit,
            to_unit=factor.from_unit,
            value=1.0 / factor.value,
            conversion_type=factor.conversion_type,
            dimension=factor.dimension,
            version=factor.version,
            status=factor.status,
            source=factor.source,
        )


__all__ = [
    "ConversionFactor",
    "ConversionType",
    "FactorStatus",
    "GWPFactor",
    "ConversionFactorRegistry",
    "create_inverse_factor",
    "ENERGY_FACTORS",
    "MASS_FACTORS",
    "VOLUME_FACTORS",
    "TEMPERATURE_FACTORS",
    "PRESSURE_FACTORS",
    "GWP_FACTORS",
    "LHV_HHV_RATIOS",
]
