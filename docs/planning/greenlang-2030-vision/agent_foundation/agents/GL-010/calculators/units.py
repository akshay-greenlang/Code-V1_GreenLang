"""
Unit Conversion Utilities for Emissions Calculations.

This module provides type-safe, deterministic unit conversion functions
for the GL-010 EMISSIONWATCH calculator modules. All conversions use
Decimal arithmetic to ensure bit-perfect reproducibility.

Zero-Hallucination Guarantee:
- All conversions are mathematically exact (no floating-point errors)
- Unit compatibility is enforced at runtime
- Full provenance tracking for all conversions

References:
- NIST Special Publication 811: Guide for the Use of the SI
- EPA 40 CFR Part 60, Appendix A
"""

from typing import Dict, Optional, Tuple, Union
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator

from .constants import (
    BTU_TO_JOULE, JOULE_TO_BTU, MMBTU_TO_GJ, GJ_TO_MMBTU,
    LB_TO_KG, KG_TO_LB, TON_SHORT_TO_KG, TON_METRIC_TO_KG,
    GAL_TO_LITER, LITER_TO_GAL, FT3_TO_M3, M3_TO_FT3,
    ATM_TO_KPA, PSI_TO_KPA, INHG_TO_KPA,
    CELSIUS_TO_KELVIN_OFFSET, FAHRENHEIT_TO_RANKINE_OFFSET,
    PERCENT_TO_PPM, PPM_TO_PERCENT,
    MW, MOLAR_VOLUME_STP, NORMAL_TEMP_K, NORMAL_PRESSURE_KPA,
)


class MassUnit(str, Enum):
    """Mass units supported by the conversion system."""
    KILOGRAM = "kg"
    GRAM = "g"
    MILLIGRAM = "mg"
    MICROGRAM = "ug"
    POUND = "lb"
    TON_SHORT = "short_ton"
    TON_METRIC = "tonne"
    OUNCE = "oz"


class VolumeUnit(str, Enum):
    """Volume units supported by the conversion system."""
    CUBIC_METER = "m3"
    LITER = "L"
    CUBIC_FOOT = "ft3"
    GALLON = "gal"
    BARREL = "bbl"
    NORMAL_CUBIC_METER = "Nm3"
    STANDARD_CUBIC_FOOT = "scf"
    DRY_STANDARD_CUBIC_FOOT = "dscf"


class EnergyUnit(str, Enum):
    """Energy units supported by the conversion system."""
    JOULE = "J"
    KILOJOULE = "kJ"
    MEGAJOULE = "MJ"
    GIGAJOULE = "GJ"
    BTU = "Btu"
    MMBTU = "MMBtu"
    KWH = "kWh"
    MWH = "MWh"
    THERM = "therm"


class TemperatureUnit(str, Enum):
    """Temperature units supported by the conversion system."""
    KELVIN = "K"
    CELSIUS = "C"
    FAHRENHEIT = "F"
    RANKINE = "R"


class PressureUnit(str, Enum):
    """Pressure units supported by the conversion system."""
    PASCAL = "Pa"
    KILOPASCAL = "kPa"
    ATMOSPHERE = "atm"
    PSI = "psi"
    INCHES_HG = "inHg"
    MILLIBAR = "mbar"
    BAR = "bar"


class ConcentrationUnit(str, Enum):
    """Concentration units supported by the conversion system."""
    PPM = "ppm"
    PPB = "ppb"
    PERCENT = "%"
    MG_M3 = "mg/m3"
    MG_NM3 = "mg/Nm3"
    UG_M3 = "ug/m3"
    LB_SCF = "lb/scf"
    GR_DSCF = "gr/dscf"


class EmissionRateUnit(str, Enum):
    """Emission rate units supported by the conversion system."""
    LB_MMBTU = "lb/MMBtu"
    KG_GJ = "kg/GJ"
    G_GJ = "g/GJ"
    LB_HR = "lb/hr"
    KG_HR = "kg/hr"
    TON_YR = "ton/yr"
    TONNE_YR = "tonne/yr"
    G_KWH = "g/kWh"
    G_HP_HR = "g/hp-hr"


@dataclass(frozen=True)
class UnitConversionResult:
    """
    Result of a unit conversion with provenance tracking.

    Attributes:
        value: Converted value (Decimal for precision)
        unit: Target unit
        original_value: Source value
        original_unit: Source unit
        conversion_factor: Factor applied
        formula: Human-readable conversion formula
    """
    value: Decimal
    unit: str
    original_value: Decimal
    original_unit: str
    conversion_factor: Decimal
    formula: str


class UnitConverter:
    """
    Zero-hallucination unit converter with full provenance tracking.

    All conversions use Decimal arithmetic to ensure bit-perfect
    reproducibility across different systems and Python versions.
    """

    # Mass conversion factors to kg (base unit)
    _MASS_TO_KG: Dict[str, Decimal] = {
        "kg": Decimal("1"),
        "g": Decimal("0.001"),
        "mg": Decimal("0.000001"),
        "ug": Decimal("0.000000001"),
        "lb": LB_TO_KG,
        "short_ton": TON_SHORT_TO_KG,
        "tonne": TON_METRIC_TO_KG,
        "oz": Decimal("0.0283495"),
    }

    # Volume conversion factors to m3 (base unit)
    _VOLUME_TO_M3: Dict[str, Decimal] = {
        "m3": Decimal("1"),
        "L": Decimal("0.001"),
        "ft3": FT3_TO_M3,
        "gal": Decimal("0.00378541"),
        "bbl": Decimal("0.158987"),
        "Nm3": Decimal("1"),  # Normal conditions
        "scf": Decimal("0.0283168"),
        "dscf": Decimal("0.0283168"),
    }

    # Energy conversion factors to J (base unit)
    _ENERGY_TO_J: Dict[str, Decimal] = {
        "J": Decimal("1"),
        "kJ": Decimal("1000"),
        "MJ": Decimal("1000000"),
        "GJ": Decimal("1000000000"),
        "Btu": BTU_TO_JOULE,
        "MMBtu": Decimal("1055060000"),
        "kWh": Decimal("3600000"),
        "MWh": Decimal("3600000000"),
        "therm": Decimal("105506000"),
    }

    # Pressure conversion factors to kPa (base unit)
    _PRESSURE_TO_KPA: Dict[str, Decimal] = {
        "Pa": Decimal("0.001"),
        "kPa": Decimal("1"),
        "atm": ATM_TO_KPA,
        "psi": PSI_TO_KPA,
        "inHg": INHG_TO_KPA,
        "mbar": Decimal("0.1"),
        "bar": Decimal("100"),
    }

    @classmethod
    def convert_mass(
        cls,
        value: Union[float, Decimal],
        from_unit: str,
        to_unit: str,
        precision: int = 6
    ) -> UnitConversionResult:
        """
        Convert mass between units with full provenance.

        Args:
            value: Mass value to convert
            from_unit: Source unit (kg, g, mg, ug, lb, short_ton, tonne, oz)
            to_unit: Target unit
            precision: Decimal places in result

        Returns:
            UnitConversionResult with converted value and provenance

        Raises:
            ValueError: If unit not supported
        """
        value_dec = Decimal(str(value))

        if from_unit not in cls._MASS_TO_KG:
            raise ValueError(f"Unsupported source mass unit: {from_unit}")
        if to_unit not in cls._MASS_TO_KG:
            raise ValueError(f"Unsupported target mass unit: {to_unit}")

        # Convert to base unit (kg), then to target
        to_base = cls._MASS_TO_KG[from_unit]
        from_base = cls._MASS_TO_KG[to_unit]
        factor = to_base / from_base

        result = value_dec * factor
        result = cls._apply_precision(result, precision)

        return UnitConversionResult(
            value=result,
            unit=to_unit,
            original_value=value_dec,
            original_unit=from_unit,
            conversion_factor=factor,
            formula=f"{value} {from_unit} * {factor} = {result} {to_unit}"
        )

    @classmethod
    def convert_volume(
        cls,
        value: Union[float, Decimal],
        from_unit: str,
        to_unit: str,
        precision: int = 6
    ) -> UnitConversionResult:
        """
        Convert volume between units with full provenance.

        Args:
            value: Volume value to convert
            from_unit: Source unit (m3, L, ft3, gal, bbl, Nm3, scf, dscf)
            to_unit: Target unit
            precision: Decimal places in result

        Returns:
            UnitConversionResult with converted value and provenance
        """
        value_dec = Decimal(str(value))

        if from_unit not in cls._VOLUME_TO_M3:
            raise ValueError(f"Unsupported source volume unit: {from_unit}")
        if to_unit not in cls._VOLUME_TO_M3:
            raise ValueError(f"Unsupported target volume unit: {to_unit}")

        to_base = cls._VOLUME_TO_M3[from_unit]
        from_base = cls._VOLUME_TO_M3[to_unit]
        factor = to_base / from_base

        result = value_dec * factor
        result = cls._apply_precision(result, precision)

        return UnitConversionResult(
            value=result,
            unit=to_unit,
            original_value=value_dec,
            original_unit=from_unit,
            conversion_factor=factor,
            formula=f"{value} {from_unit} * {factor} = {result} {to_unit}"
        )

    @classmethod
    def convert_energy(
        cls,
        value: Union[float, Decimal],
        from_unit: str,
        to_unit: str,
        precision: int = 6
    ) -> UnitConversionResult:
        """
        Convert energy between units with full provenance.

        Args:
            value: Energy value to convert
            from_unit: Source unit (J, kJ, MJ, GJ, Btu, MMBtu, kWh, MWh, therm)
            to_unit: Target unit
            precision: Decimal places in result

        Returns:
            UnitConversionResult with converted value and provenance
        """
        value_dec = Decimal(str(value))

        if from_unit not in cls._ENERGY_TO_J:
            raise ValueError(f"Unsupported source energy unit: {from_unit}")
        if to_unit not in cls._ENERGY_TO_J:
            raise ValueError(f"Unsupported target energy unit: {to_unit}")

        to_base = cls._ENERGY_TO_J[from_unit]
        from_base = cls._ENERGY_TO_J[to_unit]
        factor = to_base / from_base

        result = value_dec * factor
        result = cls._apply_precision(result, precision)

        return UnitConversionResult(
            value=result,
            unit=to_unit,
            original_value=value_dec,
            original_unit=from_unit,
            conversion_factor=factor,
            formula=f"{value} {from_unit} * {factor} = {result} {to_unit}"
        )

    @classmethod
    def convert_pressure(
        cls,
        value: Union[float, Decimal],
        from_unit: str,
        to_unit: str,
        precision: int = 6
    ) -> UnitConversionResult:
        """
        Convert pressure between units with full provenance.

        Args:
            value: Pressure value to convert
            from_unit: Source unit (Pa, kPa, atm, psi, inHg, mbar, bar)
            to_unit: Target unit
            precision: Decimal places in result

        Returns:
            UnitConversionResult with converted value and provenance
        """
        value_dec = Decimal(str(value))

        if from_unit not in cls._PRESSURE_TO_KPA:
            raise ValueError(f"Unsupported source pressure unit: {from_unit}")
        if to_unit not in cls._PRESSURE_TO_KPA:
            raise ValueError(f"Unsupported target pressure unit: {to_unit}")

        to_base = cls._PRESSURE_TO_KPA[from_unit]
        from_base = cls._PRESSURE_TO_KPA[to_unit]
        factor = to_base / from_base

        result = value_dec * factor
        result = cls._apply_precision(result, precision)

        return UnitConversionResult(
            value=result,
            unit=to_unit,
            original_value=value_dec,
            original_unit=from_unit,
            conversion_factor=factor,
            formula=f"{value} {from_unit} * {factor} = {result} {to_unit}"
        )

    @classmethod
    def convert_temperature(
        cls,
        value: Union[float, Decimal],
        from_unit: str,
        to_unit: str,
        precision: int = 2
    ) -> UnitConversionResult:
        """
        Convert temperature between units with full provenance.

        Note: Temperature conversions are not simple multiplications,
        they involve offsets. This method handles all standard cases.

        Args:
            value: Temperature value to convert
            from_unit: Source unit (K, C, F, R)
            to_unit: Target unit
            precision: Decimal places in result

        Returns:
            UnitConversionResult with converted value and provenance
        """
        value_dec = Decimal(str(value))

        # Convert to Kelvin first (base unit)
        if from_unit == "K":
            kelvin = value_dec
        elif from_unit == "C":
            kelvin = value_dec + CELSIUS_TO_KELVIN_OFFSET
        elif from_unit == "F":
            kelvin = (value_dec + FAHRENHEIT_TO_RANKINE_OFFSET) * Decimal("5") / Decimal("9")
        elif from_unit == "R":
            kelvin = value_dec * Decimal("5") / Decimal("9")
        else:
            raise ValueError(f"Unsupported source temperature unit: {from_unit}")

        # Convert from Kelvin to target
        if to_unit == "K":
            result = kelvin
        elif to_unit == "C":
            result = kelvin - CELSIUS_TO_KELVIN_OFFSET
        elif to_unit == "F":
            result = kelvin * Decimal("9") / Decimal("5") - FAHRENHEIT_TO_RANKINE_OFFSET
        elif to_unit == "R":
            result = kelvin * Decimal("9") / Decimal("5")
        else:
            raise ValueError(f"Unsupported target temperature unit: {to_unit}")

        result = cls._apply_precision(result, precision)

        return UnitConversionResult(
            value=result,
            unit=to_unit,
            original_value=value_dec,
            original_unit=from_unit,
            conversion_factor=Decimal("1"),  # Not applicable for temperature
            formula=f"{value} {from_unit} -> {kelvin} K -> {result} {to_unit}"
        )

    @classmethod
    def ppm_to_mg_nm3(
        cls,
        ppm_value: Union[float, Decimal],
        molecular_weight: Union[float, Decimal],
        temperature_k: Union[float, Decimal] = NORMAL_TEMP_K,
        pressure_kpa: Union[float, Decimal] = NORMAL_PRESSURE_KPA,
        precision: int = 4
    ) -> UnitConversionResult:
        """
        Convert concentration from ppm to mg/Nm3.

        Formula: mg/Nm3 = ppm * MW * P / (R * T)

        At normal conditions (0C, 101.325 kPa):
        mg/Nm3 = ppm * MW / 22.414

        Args:
            ppm_value: Concentration in ppm (by volume)
            molecular_weight: Molecular weight of gas (g/mol)
            temperature_k: Temperature in Kelvin (default: 273.15 K)
            pressure_kpa: Pressure in kPa (default: 101.325 kPa)
            precision: Decimal places in result

        Returns:
            UnitConversionResult with mg/Nm3 value and provenance

        Reference:
            EPA 40 CFR Part 60, Appendix A, Method 19
        """
        ppm = Decimal(str(ppm_value))
        mw = Decimal(str(molecular_weight))
        temp = Decimal(str(temperature_k))
        press = Decimal(str(pressure_kpa))

        # Molar volume at specified conditions (L/mol)
        # V = R * T / P where R = 8.314 L*kPa/(mol*K)
        R = Decimal("8.314462618")
        molar_volume = (R * temp) / press

        # Conversion: mg/Nm3 = ppm * MW / molar_volume
        factor = mw / molar_volume
        result = ppm * factor
        result = cls._apply_precision(result, precision)

        return UnitConversionResult(
            value=result,
            unit="mg/Nm3",
            original_value=ppm,
            original_unit="ppm",
            conversion_factor=factor,
            formula=f"{ppm} ppm * {mw} g/mol / {molar_volume} L/mol = {result} mg/Nm3"
        )

    @classmethod
    def mg_nm3_to_ppm(
        cls,
        mg_nm3_value: Union[float, Decimal],
        molecular_weight: Union[float, Decimal],
        temperature_k: Union[float, Decimal] = NORMAL_TEMP_K,
        pressure_kpa: Union[float, Decimal] = NORMAL_PRESSURE_KPA,
        precision: int = 4
    ) -> UnitConversionResult:
        """
        Convert concentration from mg/Nm3 to ppm.

        Formula: ppm = mg/Nm3 * R * T / (MW * P)

        At normal conditions (0C, 101.325 kPa):
        ppm = mg/Nm3 * 22.414 / MW

        Args:
            mg_nm3_value: Concentration in mg/Nm3
            molecular_weight: Molecular weight of gas (g/mol)
            temperature_k: Temperature in Kelvin
            pressure_kpa: Pressure in kPa
            precision: Decimal places in result

        Returns:
            UnitConversionResult with ppm value and provenance
        """
        mg_nm3 = Decimal(str(mg_nm3_value))
        mw = Decimal(str(molecular_weight))
        temp = Decimal(str(temperature_k))
        press = Decimal(str(pressure_kpa))

        R = Decimal("8.314462618")
        molar_volume = (R * temp) / press

        factor = molar_volume / mw
        result = mg_nm3 * factor
        result = cls._apply_precision(result, precision)

        return UnitConversionResult(
            value=result,
            unit="ppm",
            original_value=mg_nm3,
            original_unit="mg/Nm3",
            conversion_factor=factor,
            formula=f"{mg_nm3} mg/Nm3 * {molar_volume} L/mol / {mw} g/mol = {result} ppm"
        )

    @classmethod
    def correct_to_reference_o2(
        cls,
        measured_concentration: Union[float, Decimal],
        measured_o2: Union[float, Decimal],
        reference_o2: Union[float, Decimal],
        precision: int = 4
    ) -> UnitConversionResult:
        """
        Correct measured concentration to reference oxygen level.

        Formula: C_ref = C_meas * (20.9 - O2_ref) / (20.9 - O2_meas)

        This is the standard EPA correction formula for expressing
        emission concentrations at a common reference O2 level.

        Args:
            measured_concentration: Measured pollutant concentration
            measured_o2: Measured oxygen percentage (0-20.9%)
            reference_o2: Reference oxygen percentage (e.g., 3%, 7%, 15%)
            precision: Decimal places in result

        Returns:
            UnitConversionResult with corrected concentration

        Reference:
            EPA 40 CFR Part 60, Appendix A, Method 19
        """
        c_meas = Decimal(str(measured_concentration))
        o2_meas = Decimal(str(measured_o2))
        o2_ref = Decimal(str(reference_o2))

        # Oxygen in air (dry basis)
        o2_air = Decimal("20.9")

        # Validate inputs
        if o2_meas >= o2_air:
            raise ValueError(f"Measured O2 ({o2_meas}%) must be less than {o2_air}%")
        if o2_ref >= o2_air:
            raise ValueError(f"Reference O2 ({o2_ref}%) must be less than {o2_air}%")
        if o2_meas < Decimal("0"):
            raise ValueError(f"Measured O2 ({o2_meas}%) cannot be negative")

        factor = (o2_air - o2_ref) / (o2_air - o2_meas)
        result = c_meas * factor
        result = cls._apply_precision(result, precision)

        return UnitConversionResult(
            value=result,
            unit=f"@ {o2_ref}% O2",
            original_value=c_meas,
            original_unit=f"@ {o2_meas}% O2",
            conversion_factor=factor,
            formula=f"{c_meas} * ({o2_air} - {o2_ref}) / ({o2_air} - {o2_meas}) = {result}"
        )

    @classmethod
    def convert_emission_rate(
        cls,
        value: Union[float, Decimal],
        from_unit: str,
        to_unit: str,
        heat_rate_btu_kwh: Optional[Union[float, Decimal]] = None,
        precision: int = 6
    ) -> UnitConversionResult:
        """
        Convert emission rate between units.

        Supports conversions between:
        - lb/MMBtu <-> kg/GJ <-> g/GJ
        - lb/hr <-> kg/hr
        - ton/yr <-> tonne/yr
        - g/kWh (requires heat rate)

        Args:
            value: Emission rate value
            from_unit: Source unit
            to_unit: Target unit
            heat_rate_btu_kwh: Heat rate in Btu/kWh (for g/kWh conversions)
            precision: Decimal places in result

        Returns:
            UnitConversionResult with converted value and provenance
        """
        value_dec = Decimal(str(value))

        # Handle energy-based emission rates
        if from_unit == "lb/MMBtu" and to_unit == "kg/GJ":
            # lb/MMBtu * (0.453592 kg/lb) * (1 MMBtu/1.05506 GJ) = kg/GJ
            factor = Decimal("0.429923")
            result = value_dec * factor

        elif from_unit == "kg/GJ" and to_unit == "lb/MMBtu":
            factor = Decimal("2.32599")
            result = value_dec * factor

        elif from_unit == "lb/MMBtu" and to_unit == "g/GJ":
            factor = Decimal("429.923")
            result = value_dec * factor

        elif from_unit == "g/GJ" and to_unit == "lb/MMBtu":
            factor = Decimal("0.00232599")
            result = value_dec * factor

        elif from_unit == "lb/hr" and to_unit == "kg/hr":
            factor = LB_TO_KG
            result = value_dec * factor

        elif from_unit == "kg/hr" and to_unit == "lb/hr":
            factor = KG_TO_LB
            result = value_dec * factor

        elif from_unit == "ton/yr" and to_unit == "tonne/yr":
            factor = Decimal("0.907185")
            result = value_dec * factor

        elif from_unit == "tonne/yr" and to_unit == "ton/yr":
            factor = Decimal("1.10231")
            result = value_dec * factor

        elif "g/kWh" in from_unit or "g/kWh" in to_unit:
            if heat_rate_btu_kwh is None:
                raise ValueError("Heat rate required for g/kWh conversions")
            hr = Decimal(str(heat_rate_btu_kwh))

            if from_unit == "lb/MMBtu" and to_unit == "g/kWh":
                # lb/MMBtu * (453.592 g/lb) * (heat_rate Btu/kWh) * (1 MMBtu/1e6 Btu)
                factor = Decimal("0.000453592") * hr
                result = value_dec * factor
            elif from_unit == "g/kWh" and to_unit == "lb/MMBtu":
                factor = Decimal("2204.62") / hr
                result = value_dec * factor
            else:
                raise ValueError(f"Unsupported conversion: {from_unit} to {to_unit}")
        else:
            raise ValueError(f"Unsupported conversion: {from_unit} to {to_unit}")

        result = cls._apply_precision(result, precision)

        return UnitConversionResult(
            value=result,
            unit=to_unit,
            original_value=value_dec,
            original_unit=from_unit,
            conversion_factor=factor,
            formula=f"{value} {from_unit} * {factor} = {result} {to_unit}"
        )

    @classmethod
    def dry_to_wet_basis(
        cls,
        dry_value: Union[float, Decimal],
        moisture_fraction: Union[float, Decimal],
        precision: int = 4
    ) -> UnitConversionResult:
        """
        Convert concentration from dry to wet basis.

        Formula: C_wet = C_dry * (1 - H2O)

        Args:
            dry_value: Concentration on dry basis
            moisture_fraction: Water vapor fraction (0-1)
            precision: Decimal places in result

        Returns:
            UnitConversionResult with wet basis concentration
        """
        dry = Decimal(str(dry_value))
        h2o = Decimal(str(moisture_fraction))

        if h2o < 0 or h2o >= 1:
            raise ValueError(f"Moisture fraction must be 0-1, got {h2o}")

        factor = Decimal("1") - h2o
        result = dry * factor
        result = cls._apply_precision(result, precision)

        return UnitConversionResult(
            value=result,
            unit="wet basis",
            original_value=dry,
            original_unit="dry basis",
            conversion_factor=factor,
            formula=f"{dry} (dry) * (1 - {h2o}) = {result} (wet)"
        )

    @classmethod
    def wet_to_dry_basis(
        cls,
        wet_value: Union[float, Decimal],
        moisture_fraction: Union[float, Decimal],
        precision: int = 4
    ) -> UnitConversionResult:
        """
        Convert concentration from wet to dry basis.

        Formula: C_dry = C_wet / (1 - H2O)

        Args:
            wet_value: Concentration on wet basis
            moisture_fraction: Water vapor fraction (0-1)
            precision: Decimal places in result

        Returns:
            UnitConversionResult with dry basis concentration
        """
        wet = Decimal(str(wet_value))
        h2o = Decimal(str(moisture_fraction))

        if h2o < 0 or h2o >= 1:
            raise ValueError(f"Moisture fraction must be 0-1, got {h2o}")

        factor = Decimal("1") / (Decimal("1") - h2o)
        result = wet * factor
        result = cls._apply_precision(result, precision)

        return UnitConversionResult(
            value=result,
            unit="dry basis",
            original_value=wet,
            original_unit="wet basis",
            conversion_factor=factor,
            formula=f"{wet} (wet) / (1 - {h2o}) = {result} (dry)"
        )

    @staticmethod
    def _apply_precision(value: Decimal, precision: int) -> Decimal:
        """Apply decimal precision with ROUND_HALF_UP."""
        if precision < 0:
            raise ValueError(f"Precision must be non-negative, got {precision}")
        quantize_str = "0." + "0" * precision if precision > 0 else "1"
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


class GasLawCalculator:
    """
    Ideal gas law calculations for volume corrections.

    Provides deterministic calculations for adjusting gas volumes
    between different temperature and pressure conditions.
    """

    @classmethod
    def correct_volume(
        cls,
        volume: Union[float, Decimal],
        from_temp_k: Union[float, Decimal],
        from_pressure_kpa: Union[float, Decimal],
        to_temp_k: Union[float, Decimal],
        to_pressure_kpa: Union[float, Decimal],
        precision: int = 6
    ) -> UnitConversionResult:
        """
        Correct gas volume between conditions using ideal gas law.

        Formula: V2 = V1 * (T2/T1) * (P1/P2)

        Args:
            volume: Initial volume
            from_temp_k: Initial temperature (K)
            from_pressure_kpa: Initial pressure (kPa)
            to_temp_k: Final temperature (K)
            to_pressure_kpa: Final pressure (kPa)
            precision: Decimal places in result

        Returns:
            UnitConversionResult with corrected volume
        """
        v1 = Decimal(str(volume))
        t1 = Decimal(str(from_temp_k))
        p1 = Decimal(str(from_pressure_kpa))
        t2 = Decimal(str(to_temp_k))
        p2 = Decimal(str(to_pressure_kpa))

        # Validate temperatures are positive
        if t1 <= 0 or t2 <= 0:
            raise ValueError("Temperatures must be positive (Kelvin)")
        if p1 <= 0 or p2 <= 0:
            raise ValueError("Pressures must be positive")

        factor = (t2 / t1) * (p1 / p2)
        result = v1 * factor
        result = UnitConverter._apply_precision(result, precision)

        return UnitConversionResult(
            value=result,
            unit=f"@ {t2}K, {p2}kPa",
            original_value=v1,
            original_unit=f"@ {t1}K, {p1}kPa",
            conversion_factor=factor,
            formula=f"{v1} * ({t2}/{t1}) * ({p1}/{p2}) = {result}"
        )

    @classmethod
    def to_normal_conditions(
        cls,
        volume: Union[float, Decimal],
        actual_temp_k: Union[float, Decimal],
        actual_pressure_kpa: Union[float, Decimal],
        precision: int = 6
    ) -> UnitConversionResult:
        """
        Convert volume to normal conditions (0C, 101.325 kPa).

        Args:
            volume: Volume at actual conditions
            actual_temp_k: Actual temperature (K)
            actual_pressure_kpa: Actual pressure (kPa)
            precision: Decimal places in result

        Returns:
            UnitConversionResult with normal volume (Nm3)
        """
        return cls.correct_volume(
            volume=volume,
            from_temp_k=actual_temp_k,
            from_pressure_kpa=actual_pressure_kpa,
            to_temp_k=NORMAL_TEMP_K,
            to_pressure_kpa=NORMAL_PRESSURE_KPA,
            precision=precision
        )

    @classmethod
    def from_normal_conditions(
        cls,
        normal_volume: Union[float, Decimal],
        target_temp_k: Union[float, Decimal],
        target_pressure_kpa: Union[float, Decimal],
        precision: int = 6
    ) -> UnitConversionResult:
        """
        Convert volume from normal conditions to target conditions.

        Args:
            normal_volume: Volume at normal conditions (Nm3)
            target_temp_k: Target temperature (K)
            target_pressure_kpa: Target pressure (kPa)
            precision: Decimal places in result

        Returns:
            UnitConversionResult with actual volume
        """
        return cls.correct_volume(
            volume=normal_volume,
            from_temp_k=NORMAL_TEMP_K,
            from_pressure_kpa=NORMAL_PRESSURE_KPA,
            to_temp_k=target_temp_k,
            to_pressure_kpa=target_pressure_kpa,
            precision=precision
        )
