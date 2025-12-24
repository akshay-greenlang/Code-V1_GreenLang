"""
GL-011 FuelCraft - Deterministic Unit Converter

Zero-hallucination unit conversion library for fuel procurement optimization.
All conversions are DETERMINISTIC with complete provenance tracking.

Supports:
- Energy: MJ <-> MMBtu <-> kWh <-> MWh
- Mass: kg <-> lb <-> ton <-> tonne
- Volume: m3 <-> gallon <-> barrel (with temperature correction)

Standards:
- NIST SP 811 (Guide for SI Units)
- ASTM D1250 (Temperature Volume Correction)
- ISO 80000-1 (Quantities and Units)
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json


class EnergyUnit(Enum):
    """Energy units for fuel calculations."""
    MJ = "MJ"           # Megajoule (SI base for energy content)
    GJ = "GJ"           # Gigajoule
    KWH = "kWh"         # Kilowatt-hour
    MWH = "MWh"         # Megawatt-hour
    MMBTU = "MMBtu"     # Million BTU (US standard)
    BTU = "BTU"         # British Thermal Unit
    THERM = "therm"     # 100,000 BTU
    KCAL = "kcal"       # Kilocalorie


class MassUnit(Enum):
    """Mass units for fuel quantities."""
    KG = "kg"           # Kilogram (SI base)
    G = "g"             # Gram
    TONNE = "tonne"     # Metric tonne (1000 kg)
    LB = "lb"           # Pound
    TON = "ton"         # Short ton (2000 lb)
    LONG_TON = "long_ton"  # Long ton (2240 lb)


class VolumeUnit(Enum):
    """Volume units for liquid fuels."""
    M3 = "m3"           # Cubic meter (SI base)
    L = "L"             # Liter
    GAL = "gal"         # US Gallon
    BBL = "bbl"         # Barrel (42 US gallons)
    IMP_GAL = "imp_gal" # Imperial Gallon


@dataclass(frozen=True)
class ConversionFactor:
    """
    Immutable conversion factor with full provenance.

    All factors are defined relative to SI base units
    to ensure consistency and traceability.
    """
    factor: Decimal          # Multiplication factor to SI base
    offset: Decimal          # Additive offset (for temperature only)
    source_standard: str     # Reference standard (NIST, ASTM, ISO)
    effective_date: date     # Date factor became effective
    uncertainty: Decimal     # Relative uncertainty (dimensionless)
    notes: str = ""          # Additional notes

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for hashing."""
        return {
            "factor": str(self.factor),
            "offset": str(self.offset),
            "source_standard": self.source_standard,
            "effective_date": self.effective_date.isoformat(),
            "uncertainty": str(self.uncertainty),
            "notes": self.notes
        }


@dataclass
class ConversionResult:
    """
    Result of unit conversion with complete provenance.

    Includes SHA-256 hash for audit trail.
    """
    input_value: Decimal
    input_unit: str
    output_value: Decimal
    output_unit: str
    conversion_factor: ConversionFactor
    temperature_c: Optional[Decimal] = None  # For volume correction
    provenance_hash: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    calculation_steps: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Compute provenance hash after initialization."""
        if not self.provenance_hash:
            self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 provenance hash."""
        data = {
            "input_value": str(self.input_value),
            "input_unit": self.input_unit,
            "output_value": str(self.output_value),
            "output_unit": self.output_unit,
            "factor": self.conversion_factor.to_dict(),
            "temperature_c": str(self.temperature_c) if self.temperature_c else None,
            "timestamp": self.timestamp.isoformat()
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "input_value": str(self.input_value),
            "input_unit": self.input_unit,
            "output_value": str(self.output_value),
            "output_unit": self.output_unit,
            "conversion_factor": self.conversion_factor.to_dict(),
            "temperature_c": str(self.temperature_c) if self.temperature_c else None,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp.isoformat(),
            "calculation_steps": self.calculation_steps
        }


class EnergyConverter:
    """
    Deterministic energy unit converter.

    All conversions use Decimal arithmetic with explicit rounding
    to ensure bit-perfect reproducibility.

    Reference: NIST SP 811 Appendix B.8
    """

    # Energy conversion factors to MJ (SI base for energy content)
    # Source: NIST SP 811 (2008) with updates
    _FACTORS: Dict[str, ConversionFactor] = {
        "MJ": ConversionFactor(
            factor=Decimal("1.0"),
            offset=Decimal("0"),
            source_standard="NIST SP 811",
            effective_date=date(2008, 1, 1),
            uncertainty=Decimal("0")
        ),
        "GJ": ConversionFactor(
            factor=Decimal("1000.0"),
            offset=Decimal("0"),
            source_standard="NIST SP 811",
            effective_date=date(2008, 1, 1),
            uncertainty=Decimal("0")
        ),
        "kWh": ConversionFactor(
            factor=Decimal("3.6"),
            offset=Decimal("0"),
            source_standard="NIST SP 811",
            effective_date=date(2008, 1, 1),
            uncertainty=Decimal("0"),
            notes="1 kWh = 3.6 MJ exactly"
        ),
        "MWh": ConversionFactor(
            factor=Decimal("3600.0"),
            offset=Decimal("0"),
            source_standard="NIST SP 811",
            effective_date=date(2008, 1, 1),
            uncertainty=Decimal("0"),
            notes="1 MWh = 3600 MJ exactly"
        ),
        "MMBtu": ConversionFactor(
            factor=Decimal("1055.05585262"),
            offset=Decimal("0"),
            source_standard="NIST SP 811",
            effective_date=date(2008, 1, 1),
            uncertainty=Decimal("0.000001"),
            notes="1 MMBtu = 1,055,055.85262 kJ (thermochemical)"
        ),
        "BTU": ConversionFactor(
            factor=Decimal("0.00105505585262"),
            offset=Decimal("0"),
            source_standard="NIST SP 811",
            effective_date=date(2008, 1, 1),
            uncertainty=Decimal("0.000001"),
            notes="1 BTU = 1.05505585262 kJ (thermochemical)"
        ),
        "therm": ConversionFactor(
            factor=Decimal("105.505585262"),
            offset=Decimal("0"),
            source_standard="NIST SP 811",
            effective_date=date(2008, 1, 1),
            uncertainty=Decimal("0.000001"),
            notes="1 therm = 100,000 BTU"
        ),
        "kcal": ConversionFactor(
            factor=Decimal("0.004184"),
            offset=Decimal("0"),
            source_standard="NIST SP 811",
            effective_date=date(2008, 1, 1),
            uncertainty=Decimal("0"),
            notes="1 kcal = 4.184 kJ exactly (thermochemical)"
        ),
    }

    PRECISION: int = 10  # Decimal places for calculations

    @classmethod
    def convert(
        cls,
        value: Union[float, Decimal, str],
        from_unit: str,
        to_unit: str,
        precision: int = 6
    ) -> ConversionResult:
        """
        Convert energy between units - DETERMINISTIC.

        Args:
            value: Input value
            from_unit: Source unit (MJ, kWh, MMBtu, etc.)
            to_unit: Target unit
            precision: Output decimal places (default 6)

        Returns:
            ConversionResult with full provenance

        Raises:
            ValueError: If unit not supported
        """
        # Convert input to Decimal for precision
        input_value = Decimal(str(value))

        # Validate units
        if from_unit not in cls._FACTORS:
            raise ValueError(f"Unsupported source unit: {from_unit}")
        if to_unit not in cls._FACTORS:
            raise ValueError(f"Unsupported target unit: {to_unit}")

        from_factor = cls._FACTORS[from_unit]
        to_factor = cls._FACTORS[to_unit]

        # Step 1: Convert to MJ (base unit)
        mj_value = input_value * from_factor.factor

        # Step 2: Convert from MJ to target
        output_value = mj_value / to_factor.factor

        # Apply precision with ROUND_HALF_UP (regulatory standard)
        quantize_str = "0." + "0" * precision
        output_value = output_value.quantize(
            Decimal(quantize_str),
            rounding=ROUND_HALF_UP
        )

        # Create combined factor for provenance
        combined_factor = ConversionFactor(
            factor=from_factor.factor / to_factor.factor,
            offset=Decimal("0"),
            source_standard=f"{from_factor.source_standard}, {to_factor.source_standard}",
            effective_date=max(from_factor.effective_date, to_factor.effective_date),
            uncertainty=from_factor.uncertainty + to_factor.uncertainty,
            notes=f"{from_unit} -> MJ -> {to_unit}"
        )

        # Build calculation steps for audit
        steps = [
            {"step": 1, "operation": "input", "value": str(input_value), "unit": from_unit},
            {"step": 2, "operation": "to_base", "value": str(mj_value), "unit": "MJ",
             "factor": str(from_factor.factor)},
            {"step": 3, "operation": "from_base", "value": str(output_value), "unit": to_unit,
             "factor": str(to_factor.factor)},
        ]

        return ConversionResult(
            input_value=input_value,
            input_unit=from_unit,
            output_value=output_value,
            output_unit=to_unit,
            conversion_factor=combined_factor,
            calculation_steps=steps
        )

    @classmethod
    def get_factor(cls, unit: str) -> ConversionFactor:
        """Get conversion factor for unit."""
        if unit not in cls._FACTORS:
            raise ValueError(f"Unsupported unit: {unit}")
        return cls._FACTORS[unit]


class MassConverter:
    """
    Deterministic mass unit converter.

    Reference: NIST Handbook 44 (2023)
    """

    # Mass conversion factors to kg (SI base)
    _FACTORS: Dict[str, ConversionFactor] = {
        "kg": ConversionFactor(
            factor=Decimal("1.0"),
            offset=Decimal("0"),
            source_standard="SI Definition",
            effective_date=date(2019, 5, 20),  # SI redefinition date
            uncertainty=Decimal("0")
        ),
        "g": ConversionFactor(
            factor=Decimal("0.001"),
            offset=Decimal("0"),
            source_standard="SI Definition",
            effective_date=date(2019, 5, 20),
            uncertainty=Decimal("0")
        ),
        "tonne": ConversionFactor(
            factor=Decimal("1000.0"),
            offset=Decimal("0"),
            source_standard="SI Definition",
            effective_date=date(2019, 5, 20),
            uncertainty=Decimal("0"),
            notes="1 tonne = 1000 kg exactly"
        ),
        "lb": ConversionFactor(
            factor=Decimal("0.45359237"),
            offset=Decimal("0"),
            source_standard="NIST Handbook 44",
            effective_date=date(1959, 7, 1),
            uncertainty=Decimal("0"),
            notes="1 lb = 0.45359237 kg exactly (International pound)"
        ),
        "ton": ConversionFactor(
            factor=Decimal("907.18474"),
            offset=Decimal("0"),
            source_standard="NIST Handbook 44",
            effective_date=date(1959, 7, 1),
            uncertainty=Decimal("0"),
            notes="1 short ton = 2000 lb = 907.18474 kg"
        ),
        "long_ton": ConversionFactor(
            factor=Decimal("1016.0469088"),
            offset=Decimal("0"),
            source_standard="NIST Handbook 44",
            effective_date=date(1959, 7, 1),
            uncertainty=Decimal("0"),
            notes="1 long ton = 2240 lb = 1016.0469088 kg"
        ),
    }

    PRECISION: int = 10

    @classmethod
    def convert(
        cls,
        value: Union[float, Decimal, str],
        from_unit: str,
        to_unit: str,
        precision: int = 6
    ) -> ConversionResult:
        """
        Convert mass between units - DETERMINISTIC.

        Args:
            value: Input value
            from_unit: Source unit (kg, lb, tonne, etc.)
            to_unit: Target unit
            precision: Output decimal places

        Returns:
            ConversionResult with full provenance
        """
        input_value = Decimal(str(value))

        if from_unit not in cls._FACTORS:
            raise ValueError(f"Unsupported source unit: {from_unit}")
        if to_unit not in cls._FACTORS:
            raise ValueError(f"Unsupported target unit: {to_unit}")

        from_factor = cls._FACTORS[from_unit]
        to_factor = cls._FACTORS[to_unit]

        # Convert via kg base
        kg_value = input_value * from_factor.factor
        output_value = kg_value / to_factor.factor

        # Apply precision
        quantize_str = "0." + "0" * precision
        output_value = output_value.quantize(
            Decimal(quantize_str),
            rounding=ROUND_HALF_UP
        )

        combined_factor = ConversionFactor(
            factor=from_factor.factor / to_factor.factor,
            offset=Decimal("0"),
            source_standard=f"{from_factor.source_standard}, {to_factor.source_standard}",
            effective_date=max(from_factor.effective_date, to_factor.effective_date),
            uncertainty=from_factor.uncertainty + to_factor.uncertainty,
            notes=f"{from_unit} -> kg -> {to_unit}"
        )

        steps = [
            {"step": 1, "operation": "input", "value": str(input_value), "unit": from_unit},
            {"step": 2, "operation": "to_base", "value": str(kg_value), "unit": "kg",
             "factor": str(from_factor.factor)},
            {"step": 3, "operation": "from_base", "value": str(output_value), "unit": to_unit,
             "factor": str(to_factor.factor)},
        ]

        return ConversionResult(
            input_value=input_value,
            input_unit=from_unit,
            output_value=output_value,
            output_unit=to_unit,
            conversion_factor=combined_factor,
            calculation_steps=steps
        )

    @classmethod
    def get_factor(cls, unit: str) -> ConversionFactor:
        """Get conversion factor for unit."""
        if unit not in cls._FACTORS:
            raise ValueError(f"Unsupported unit: {unit}")
        return cls._FACTORS[unit]


class VolumeConverter:
    """
    Deterministic volume unit converter with temperature correction.

    Implements ASTM D1250 (API MPMS Chapter 11.1) for petroleum
    volume correction at standard reference temperatures.

    Reference temperatures:
    - API/ASTM: 60F (15.56C)
    - ISO: 15C
    """

    # Volume conversion factors to m3 (SI base)
    _FACTORS: Dict[str, ConversionFactor] = {
        "m3": ConversionFactor(
            factor=Decimal("1.0"),
            offset=Decimal("0"),
            source_standard="SI Definition",
            effective_date=date(2019, 5, 20),
            uncertainty=Decimal("0")
        ),
        "L": ConversionFactor(
            factor=Decimal("0.001"),
            offset=Decimal("0"),
            source_standard="SI Definition",
            effective_date=date(2019, 5, 20),
            uncertainty=Decimal("0"),
            notes="1 L = 0.001 m3 exactly"
        ),
        "gal": ConversionFactor(
            factor=Decimal("0.003785411784"),
            offset=Decimal("0"),
            source_standard="NIST Handbook 44",
            effective_date=date(1959, 7, 1),
            uncertainty=Decimal("0"),
            notes="1 US gallon = 231 in3 = 3.785411784 L exactly"
        ),
        "bbl": ConversionFactor(
            factor=Decimal("0.158987294928"),
            offset=Decimal("0"),
            source_standard="API MPMS",
            effective_date=date(2004, 1, 1),
            uncertainty=Decimal("0"),
            notes="1 barrel = 42 US gallons = 158.987294928 L"
        ),
        "imp_gal": ConversionFactor(
            factor=Decimal("0.00454609"),
            offset=Decimal("0"),
            source_standard="NIST Handbook 44",
            effective_date=date(1985, 1, 1),
            uncertainty=Decimal("0"),
            notes="1 imperial gallon = 4.54609 L exactly"
        ),
    }

    # API gravity coefficients for volume correction (ASTM D1250)
    # These are simplified - production systems use full tables
    REFERENCE_TEMP_C: Decimal = Decimal("15.0")  # ISO reference
    REFERENCE_TEMP_F: Decimal = Decimal("60.0")  # API reference

    PRECISION: int = 10

    @classmethod
    def convert(
        cls,
        value: Union[float, Decimal, str],
        from_unit: str,
        to_unit: str,
        temperature_c: Optional[Union[float, Decimal]] = None,
        api_gravity: Optional[Union[float, Decimal]] = None,
        precision: int = 6
    ) -> ConversionResult:
        """
        Convert volume between units with optional temperature correction.

        Args:
            value: Input value
            from_unit: Source unit (m3, gal, bbl, etc.)
            to_unit: Target unit
            temperature_c: Observed temperature in Celsius (for correction)
            api_gravity: API gravity of product (for correction)
            precision: Output decimal places

        Returns:
            ConversionResult with full provenance

        Notes:
            If temperature_c and api_gravity are provided, volume is corrected
            to standard reference temperature (15C) per ASTM D1250.
        """
        input_value = Decimal(str(value))

        if from_unit not in cls._FACTORS:
            raise ValueError(f"Unsupported source unit: {from_unit}")
        if to_unit not in cls._FACTORS:
            raise ValueError(f"Unsupported target unit: {to_unit}")

        from_factor = cls._FACTORS[from_unit]
        to_factor = cls._FACTORS[to_unit]

        # Convert to m3 base
        m3_value = input_value * from_factor.factor

        steps = [
            {"step": 1, "operation": "input", "value": str(input_value), "unit": from_unit},
            {"step": 2, "operation": "to_base", "value": str(m3_value), "unit": "m3",
             "factor": str(from_factor.factor)},
        ]

        # Apply temperature correction if parameters provided
        temp_decimal: Optional[Decimal] = None
        if temperature_c is not None and api_gravity is not None:
            temp_decimal = Decimal(str(temperature_c))
            api_decimal = Decimal(str(api_gravity))

            # Calculate volume correction factor (VCF)
            vcf = cls._calculate_vcf(temp_decimal, api_decimal)
            m3_value = m3_value * vcf

            steps.append({
                "step": 3,
                "operation": "temp_correction",
                "value": str(m3_value),
                "unit": "m3@15C",
                "vcf": str(vcf),
                "observed_temp_c": str(temp_decimal),
                "api_gravity": str(api_decimal)
            })

        # Convert from m3 to target
        output_value = m3_value / to_factor.factor

        # Apply precision
        quantize_str = "0." + "0" * precision
        output_value = output_value.quantize(
            Decimal(quantize_str),
            rounding=ROUND_HALF_UP
        )

        steps.append({
            "step": len(steps) + 1,
            "operation": "from_base",
            "value": str(output_value),
            "unit": to_unit,
            "factor": str(to_factor.factor)
        })

        combined_factor = ConversionFactor(
            factor=from_factor.factor / to_factor.factor,
            offset=Decimal("0"),
            source_standard=f"{from_factor.source_standard}, {to_factor.source_standard}",
            effective_date=max(from_factor.effective_date, to_factor.effective_date),
            uncertainty=from_factor.uncertainty + to_factor.uncertainty,
            notes=f"{from_unit} -> m3 -> {to_unit}"
        )

        return ConversionResult(
            input_value=input_value,
            input_unit=from_unit,
            output_value=output_value,
            output_unit=to_unit,
            conversion_factor=combined_factor,
            temperature_c=temp_decimal,
            calculation_steps=steps
        )

    @classmethod
    def _calculate_vcf(
        cls,
        temperature_c: Decimal,
        api_gravity: Decimal
    ) -> Decimal:
        """
        Calculate Volume Correction Factor per ASTM D1250.

        This is a simplified calculation. Production systems should
        use the full ASTM D1250 tables.

        Args:
            temperature_c: Observed temperature in Celsius
            api_gravity: API gravity of product

        Returns:
            Volume correction factor (dimensionless)
        """
        # Convert API gravity to relative density (API equation)
        # SG = 141.5 / (API + 131.5)
        sg = Decimal("141.5") / (api_gravity + Decimal("131.5"))

        # Simplified thermal expansion coefficient
        # alpha = K0 + K1*SG (typical crude oil approximation)
        K0 = Decimal("613.9723")
        K1 = Decimal("0.0")
        K2 = Decimal("0.0")

        # For crude oils and products, use API Table 6A/6B coefficients
        # This is a simplified linear approximation
        alpha = K0 / (sg * sg) / Decimal("1000000")

        # Temperature difference from reference
        delta_t = temperature_c - cls.REFERENCE_TEMP_C

        # VCF = exp(-alpha * delta_T * (1 + 0.8 * alpha * delta_T))
        # Simplified: VCF ~ 1 - alpha * delta_T for small delta_T
        vcf = Decimal("1.0") - alpha * delta_t

        # Clamp to reasonable range
        vcf = max(Decimal("0.9"), min(Decimal("1.1"), vcf))

        return vcf.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    @classmethod
    def get_factor(cls, unit: str) -> ConversionFactor:
        """Get conversion factor for unit."""
        if unit not in cls._FACTORS:
            raise ValueError(f"Unsupported unit: {unit}")
        return cls._FACTORS[unit]


class UnitConverter:
    """
    Unified deterministic unit converter.

    Combines energy, mass, and volume conversions with
    complete provenance tracking for regulatory compliance.
    """

    NAME: str = "UnitConverter"
    VERSION: str = "1.0.0"

    def __init__(self):
        """Initialize converter."""
        self._energy = EnergyConverter
        self._mass = MassConverter
        self._volume = VolumeConverter

    def convert_energy(
        self,
        value: Union[float, Decimal, str],
        from_unit: str,
        to_unit: str,
        precision: int = 6
    ) -> ConversionResult:
        """Convert energy units."""
        return self._energy.convert(value, from_unit, to_unit, precision)

    def convert_mass(
        self,
        value: Union[float, Decimal, str],
        from_unit: str,
        to_unit: str,
        precision: int = 6
    ) -> ConversionResult:
        """Convert mass units."""
        return self._mass.convert(value, from_unit, to_unit, precision)

    def convert_volume(
        self,
        value: Union[float, Decimal, str],
        from_unit: str,
        to_unit: str,
        temperature_c: Optional[Union[float, Decimal]] = None,
        api_gravity: Optional[Union[float, Decimal]] = None,
        precision: int = 6
    ) -> ConversionResult:
        """Convert volume units with optional temperature correction."""
        return self._volume.convert(
            value, from_unit, to_unit, temperature_c, api_gravity, precision
        )

    @staticmethod
    def get_supported_energy_units() -> List[str]:
        """Get list of supported energy units."""
        return list(EnergyConverter._FACTORS.keys())

    @staticmethod
    def get_supported_mass_units() -> List[str]:
        """Get list of supported mass units."""
        return list(MassConverter._FACTORS.keys())

    @staticmethod
    def get_supported_volume_units() -> List[str]:
        """Get list of supported volume units."""
        return list(VolumeConverter._FACTORS.keys())
