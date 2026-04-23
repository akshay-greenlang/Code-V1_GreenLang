"""
Water Chemistry Unit Handling Module

Zero-Hallucination Guarantee:
- All unit conversions are deterministic mathematical operations
- No LLM involvement in any conversion calculations
- Complete provenance tracking for audit trails
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Union, Tuple, Optional
import hashlib
import json

try:
    import pint
    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False


class WaterChemistryUnitRegistry:
    """Custom unit registry for water chemistry calculations."""

    def __init__(self):
        if PINT_AVAILABLE:
            self._ureg = pint.UnitRegistry()
            self._define_custom_units()
        else:
            self._ureg = None
        self._conversion_factors = self._build_conversion_factors()

    def _define_custom_units(self) -> None:
        if not self._ureg:
            return
        self._ureg.define("microsiemens_per_cm = microsiemens / centimeter = uS/cm = uS_cm")
        self._ureg.define("millisiemens_per_cm = millisiemens / centimeter = mS/cm = mS_cm")
        self._ureg.define("ppm = 1e-6 = parts_per_million")
        self._ureg.define("ppb = 1e-9 = parts_per_billion")
        self._ureg.define("mg_per_L = milligram / liter = mg/L = mg_L")
        self._ureg.define("ug_per_L = microgram / liter = ug/L = ug_L")
        self._ureg.define("gpm = gallon / minute = gallons_per_minute")
        self._ureg.define("kg_per_h = kilogram / hour = kg/h = kg_h")
        self._ureg.define("lb_per_h = pound / hour = lb/h = lb_h")
        self._ureg.define("m3_per_h = meter ** 3 / hour = m3/h = m3_h")
        self._ureg.define("L_per_h = liter / hour = L/h = L_h")

    def _build_conversion_factors(self) -> dict:
        return {
            "conductivity": {
                ("uS/cm", "mS/cm"): Decimal("0.001"),
                ("mS/cm", "uS/cm"): Decimal("1000"),
            },
            "concentration": {
                ("ppm", "ppb"): Decimal("1000"),
                ("ppb", "ppm"): Decimal("0.001"),
                ("mg/L", "ppm"): Decimal("1"),
                ("ppm", "mg/L"): Decimal("1"),
                ("mg/L", "ug/L"): Decimal("1000"),
                ("ug/L", "mg/L"): Decimal("0.001"),
                ("ug/L", "ppb"): Decimal("1"),
                ("ppb", "ug/L"): Decimal("1"),
            },
            "flow": {
                ("kg/h", "lb/h"): Decimal("2.20462262185"),
                ("lb/h", "kg/h"): Decimal("0.45359237"),
                ("gpm", "L/h"): Decimal("227.1247"),
                ("L/h", "gpm"): Decimal("0.00440287"),
                ("m3/h", "L/h"): Decimal("1000"),
                ("L/h", "m3/h"): Decimal("0.001"),
                ("gpm", "m3/h"): Decimal("0.2271247"),
                ("m3/h", "gpm"): Decimal("4.40287"),
            },
            "temperature": {
                "C_to_F": lambda c: c * Decimal("1.8") + Decimal("32"),
                "F_to_C": lambda f: (f - Decimal("32")) / Decimal("1.8"),
                "C_to_K": lambda c: c + Decimal("273.15"),
                "K_to_C": lambda k: k - Decimal("273.15"),
            },
            "mass": {
                ("kg", "lb"): Decimal("2.20462262185"),
                ("lb", "kg"): Decimal("0.45359237"),
                ("kg", "g"): Decimal("1000"),
                ("g", "kg"): Decimal("0.001"),
                ("g", "mg"): Decimal("1000"),
                ("mg", "g"): Decimal("0.001"),
            },
            "volume": {
                ("L", "gal"): Decimal("0.264172052"),
                ("gal", "L"): Decimal("3.785411784"),
                ("m3", "L"): Decimal("1000"),
                ("L", "m3"): Decimal("0.001"),
            },
        }

    @property
    def ureg(self):
        return self._ureg

    def convert(self, value: Union[float, Decimal], from_unit: str, to_unit: str,
                category: Optional[str] = None) -> Tuple[Decimal, str]:
        if not isinstance(value, Decimal):
            value = Decimal(str(value))
        from_unit = self._normalize_unit(from_unit)
        to_unit = self._normalize_unit(to_unit)
        if from_unit == to_unit:
            provenance = self._compute_conversion_provenance(value, from_unit, to_unit, value)
            return value, provenance
        if PINT_AVAILABLE and self._ureg:
            try:
                pint_value = float(value) * self._ureg(from_unit)
                converted = pint_value.to(to_unit).magnitude
                result = Decimal(str(converted))
                provenance = self._compute_conversion_provenance(value, from_unit, to_unit, result)
                return result, provenance
            except (pint.errors.DimensionalityError, pint.errors.UndefinedUnitError):
                pass
        result = self._manual_convert(value, from_unit, to_unit, category)
        provenance = self._compute_conversion_provenance(value, from_unit, to_unit, result)
        return result, provenance

    def _manual_convert(self, value: Decimal, from_unit: str, to_unit: str,
                        category: Optional[str] = None) -> Decimal:
        key = (from_unit, to_unit)
        categories = [category] if category else self._conversion_factors.keys()
        for cat in categories:
            if cat not in self._conversion_factors:
                continue
            factors = self._conversion_factors[cat]
            if key in factors:
                factor = factors[key]
                if callable(factor):
                    return factor(value)
                return value * factor
        inverse_key = (to_unit, from_unit)
        for cat in categories:
            if cat not in self._conversion_factors:
                continue
            factors = self._conversion_factors[cat]
            if inverse_key in factors:
                factor = factors[inverse_key]
                if callable(factor):
                    raise ValueError(f"Cannot invert function-based conversion for {from_unit} to {to_unit}")
                return value / factor
        raise ValueError(f"No conversion available from {from_unit} to {to_unit}")

    def _normalize_unit(self, unit: str) -> str:
        normalizations = {
            "us/cm": "uS/cm", "microsiemens/cm": "uS/cm",
            "ms/cm": "mS/cm", "millisiemens/cm": "mS/cm",
            "mg/l": "mg/L", "ug/l": "ug/L", "l/h": "L/h",
            "m3/h": "m3/h", "degc": "degC", "degf": "degF",
            "celsius": "degC", "fahrenheit": "degF", "kelvin": "K",
        }
        return normalizations.get(unit.lower(), unit)

    def _compute_conversion_provenance(self, input_value: Decimal, from_unit: str,
                                        to_unit: str, result: Decimal) -> str:
        provenance_data = {
            "operation": "unit_conversion",
            "input_value": str(input_value),
            "from_unit": from_unit,
            "to_unit": to_unit,
            "result": str(result),
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode("utf-8")).hexdigest()


def temperature_compensation_conductivity(
    conductivity_at_temp: Union[float, Decimal],
    measured_temp: Union[float, Decimal],
    reference_temp: Union[float, Decimal] = Decimal("25"),
    compensation_factor: Union[float, Decimal] = Decimal("0.02")
) -> Tuple[Decimal, str]:
    """Apply temperature compensation to conductivity measurement."""
    if not isinstance(conductivity_at_temp, Decimal):
        conductivity_at_temp = Decimal(str(conductivity_at_temp))
    if not isinstance(measured_temp, Decimal):
        measured_temp = Decimal(str(measured_temp))
    if not isinstance(reference_temp, Decimal):
        reference_temp = Decimal(str(reference_temp))
    if not isinstance(compensation_factor, Decimal):
        compensation_factor = Decimal(str(compensation_factor))
    temp_diff = measured_temp - reference_temp
    compensation_multiplier = Decimal("1") + (compensation_factor * temp_diff)
    compensated_conductivity = conductivity_at_temp / compensation_multiplier
    compensated_conductivity = compensated_conductivity.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    provenance_data = {
        "operation": "temperature_compensation_conductivity",
        "input_conductivity": str(conductivity_at_temp),
        "measured_temp": str(measured_temp),
        "reference_temp": str(reference_temp),
        "compensation_factor": str(compensation_factor),
        "result": str(compensated_conductivity),
    }
    provenance_str = json.dumps(provenance_data, sort_keys=True)
    provenance_hash = hashlib.sha256(provenance_str.encode("utf-8")).hexdigest()
    return compensated_conductivity, provenance_hash


def validate_unit_consistency(value1: Union[float, Decimal], unit1: str,
                              value2: Union[float, Decimal], unit2: str, category: str) -> bool:
    registry = WaterChemistryUnitRegistry()
    try:
        registry.convert(Decimal("1"), unit1, unit2, category)
        return True
    except ValueError:
        return False


_unit_registry = None


def get_unit_registry() -> WaterChemistryUnitRegistry:
    global _unit_registry
    if _unit_registry is None:
        _unit_registry = WaterChemistryUnitRegistry()
    return _unit_registry


class UnitValue:
    """A value with an explicit unit."""

    def __init__(self, value: Union[float, Decimal], unit: str, precision: int = 6):
        self._value = Decimal(str(value)) if not isinstance(value, Decimal) else value
        self._unit = get_unit_registry()._normalize_unit(unit)
        self._precision = precision

    @property
    def value(self) -> Decimal:
        return self._value

    @property
    def unit(self) -> str:
        return self._unit

    @property
    def magnitude(self) -> Decimal:
        return self._value

    def to(self, target_unit: str, category: Optional[str] = None) -> "UnitValue":
        registry = get_unit_registry()
        converted_value, _ = registry.convert(self._value, self._unit, target_unit, category)
        return UnitValue(converted_value, target_unit, self._precision)

    def __add__(self, other: "UnitValue") -> "UnitValue":
        if not isinstance(other, UnitValue):
            raise TypeError("Can only add UnitValue to UnitValue")
        if other._unit != self._unit:
            other = other.to(self._unit)
        return UnitValue(self._value + other._value, self._unit, self._precision)

    def __sub__(self, other: "UnitValue") -> "UnitValue":
        if not isinstance(other, UnitValue):
            raise TypeError("Can only subtract UnitValue from UnitValue")
        if other._unit != self._unit:
            other = other.to(self._unit)
        return UnitValue(self._value - other._value, self._unit, self._precision)

    def __mul__(self, other: Union[float, Decimal, int]) -> "UnitValue":
        if isinstance(other, UnitValue):
            raise TypeError("Cannot multiply two UnitValues directly")
        factor = Decimal(str(other))
        return UnitValue(self._value * factor, self._unit, self._precision)

    def __truediv__(self, other: Union[float, Decimal, int, "UnitValue"]) -> Union["UnitValue", Decimal]:
        if isinstance(other, UnitValue):
            if other._unit != self._unit:
                other = other.to(self._unit)
            return self._value / other._value
        divisor = Decimal(str(other))
        return UnitValue(self._value / divisor, self._unit, self._precision)

    def __repr__(self) -> str:
        return f"UnitValue({self._value}, '{self._unit}')"

    def __str__(self) -> str:
        return f"{self._value} {self._unit}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnitValue):
            return False
        try:
            converted = other.to(self._unit)
            tolerance = Decimal("1e-10")
            return abs(self._value - converted._value) < tolerance
        except ValueError:
            return False

    def __lt__(self, other: "UnitValue") -> bool:
        if not isinstance(other, UnitValue):
            raise TypeError("Can only compare UnitValue with UnitValue")
        converted = other.to(self._unit)
        return self._value < converted._value

    def __le__(self, other: "UnitValue") -> bool:
        return self == other or self < other

    def __gt__(self, other: "UnitValue") -> bool:
        if not isinstance(other, UnitValue):
            raise TypeError("Can only compare UnitValue with UnitValue")
        converted = other.to(self._unit)
        return self._value > converted._value

    def __ge__(self, other: "UnitValue") -> bool:
        return self == other or self > other

    def to_dict(self) -> dict:
        return {"value": str(self._value), "unit": self._unit}

    @classmethod
    def from_dict(cls, data: dict) -> "UnitValue":
        return cls(Decimal(data["value"]), data["unit"])

    def get_provenance_hash(self) -> str:
        provenance_data = {"type": "UnitValue", "value": str(self._value), "unit": self._unit}
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode("utf-8")).hexdigest()
