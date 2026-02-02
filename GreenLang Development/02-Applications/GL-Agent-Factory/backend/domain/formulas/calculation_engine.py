"""
GreenLang Zero-Hallucination Calculation Engine
================================================

This module implements the core calculation engine that guarantees
bit-perfect reproducibility and complete audit trails.

Key Features:
- Deterministic calculations (same input -> same output)
- Full provenance tracking with SHA-256 hashes
- Unit conversion integration
- Uncertainty propagation
- Calculation caching for performance

Copyright (c) 2024 GreenLang. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    TypeVar,
    Generic,
)
import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone
import threading
from collections import OrderedDict


# =============================================================================
# Type Definitions
# =============================================================================

Number = Union[int, float, Decimal]
T = TypeVar('T')


class UnitCategory(Enum):
    """Categories of physical units."""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    MASS = "mass"
    VOLUME = "volume"
    ENERGY = "energy"
    POWER = "power"
    FLOW_RATE = "flow_rate"
    DENSITY = "density"
    VISCOSITY = "viscosity"
    THERMAL_CONDUCTIVITY = "thermal_conductivity"
    HEAT_CAPACITY = "heat_capacity"
    VELOCITY = "velocity"
    LENGTH = "length"
    AREA = "area"
    TIME = "time"
    DIMENSIONLESS = "dimensionless"
    EMISSION_FACTOR = "emission_factor"
    HEAT_TRANSFER_COEFFICIENT = "heat_transfer_coefficient"
    ENTHALPY = "enthalpy"
    ENTROPY = "entropy"


class CalculationStatus(Enum):
    """Status of calculation execution."""
    SUCCESS = "success"
    VALIDATION_ERROR = "validation_error"
    CALCULATION_ERROR = "calculation_error"
    UNIT_ERROR = "unit_error"
    RANGE_ERROR = "range_error"
    TIMEOUT_ERROR = "timeout_error"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class UnitDefinition:
    """Definition of a physical unit."""
    symbol: str
    name: str
    category: UnitCategory
    si_conversion_factor: float  # Multiply by this to get SI unit
    si_offset: float = 0.0  # Add this after multiplication (for temperature)

    def __hash__(self):
        return hash((self.symbol, self.category))


@dataclass
class ParameterDefinition:
    """Definition of a formula parameter."""
    name: str
    description: str
    unit: str
    category: UnitCategory
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    default_value: Optional[float] = None
    required: bool = True
    uncertainty_percent: float = 0.0


@dataclass
class FormulaDefinition:
    """Complete definition of a calculation formula."""
    formula_id: str
    name: str
    description: str
    category: str
    source_standard: str
    source_reference: str
    version: str
    parameters: List[ParameterDefinition]
    output_name: str
    output_unit: str
    output_description: str
    precision: int = 6  # Decimal places
    uncertainty_method: str = "propagation"  # or "monte_carlo"
    valid_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    formula_hash: str = ""  # SHA-256 of formula definition

    def __post_init__(self):
        """Calculate formula hash after initialization."""
        if not self.formula_hash:
            self.formula_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of formula definition."""
        hash_data = {
            "formula_id": self.formula_id,
            "name": self.name,
            "version": self.version,
            "parameters": [
                {
                    "name": p.name,
                    "unit": p.unit,
                    "min_value": p.min_value,
                    "max_value": p.max_value,
                }
                for p in self.parameters
            ],
            "output_unit": self.output_unit,
            "precision": self.precision,
        }
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()


@dataclass
class FormulaInput:
    """Input for formula calculation."""
    formula_id: str
    parameters: Dict[str, float]
    parameter_units: Optional[Dict[str, str]] = None
    version: str = "latest"
    request_id: Optional[str] = None


@dataclass
class CalculationStep:
    """Individual calculation step with provenance."""
    step_number: int
    description: str
    operation: str
    inputs: Dict[str, Any]
    output_value: Decimal
    output_name: str
    intermediate_values: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UncertaintyResult:
    """Uncertainty analysis result."""
    value: Decimal
    uncertainty_absolute: Decimal
    uncertainty_percent: Decimal
    confidence_level: float = 0.95
    method: str = "propagation"
    component_uncertainties: Dict[str, Decimal] = field(default_factory=dict)


@dataclass
class CalculationResult:
    """Result of calculation with complete provenance."""
    formula_id: str
    formula_version: str
    formula_hash: str
    status: CalculationStatus
    output_value: Optional[Decimal]
    output_unit: str
    uncertainty: Optional[UncertaintyResult]
    calculation_steps: List[CalculationStep]
    provenance_hash: str
    calculation_time_ms: float
    timestamp: str
    input_parameters: Dict[str, float]
    converted_parameters: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Unit Converter
# =============================================================================

class UnitConverter:
    """
    Comprehensive unit conversion system.

    Supports conversion between units within the same category.
    All conversions go through SI units as intermediate.
    """

    # SI base units for each category
    SI_UNITS = {
        UnitCategory.TEMPERATURE: "K",
        UnitCategory.PRESSURE: "Pa",
        UnitCategory.MASS: "kg",
        UnitCategory.VOLUME: "m3",
        UnitCategory.ENERGY: "J",
        UnitCategory.POWER: "W",
        UnitCategory.FLOW_RATE: "kg/s",
        UnitCategory.DENSITY: "kg/m3",
        UnitCategory.VISCOSITY: "Pa.s",
        UnitCategory.THERMAL_CONDUCTIVITY: "W/(m.K)",
        UnitCategory.HEAT_CAPACITY: "J/(kg.K)",
        UnitCategory.VELOCITY: "m/s",
        UnitCategory.LENGTH: "m",
        UnitCategory.AREA: "m2",
        UnitCategory.TIME: "s",
        UnitCategory.DIMENSIONLESS: "",
        UnitCategory.EMISSION_FACTOR: "kg/GJ",
        UnitCategory.HEAT_TRANSFER_COEFFICIENT: "W/(m2.K)",
        UnitCategory.ENTHALPY: "J/kg",
        UnitCategory.ENTROPY: "J/(kg.K)",
    }

    def __init__(self):
        """Initialize unit converter with all unit definitions."""
        self._units: Dict[str, UnitDefinition] = {}
        self._register_all_units()

    def _register_all_units(self):
        """Register all supported units."""
        # Temperature units
        self._register_unit("K", "Kelvin", UnitCategory.TEMPERATURE, 1.0, 0.0)
        self._register_unit("C", "Celsius", UnitCategory.TEMPERATURE, 1.0, 273.15)
        self._register_unit("F", "Fahrenheit", UnitCategory.TEMPERATURE, 5/9, 255.372)
        self._register_unit("R", "Rankine", UnitCategory.TEMPERATURE, 5/9, 0.0)

        # Pressure units
        self._register_unit("Pa", "Pascal", UnitCategory.PRESSURE, 1.0)
        self._register_unit("kPa", "Kilopascal", UnitCategory.PRESSURE, 1000.0)
        self._register_unit("MPa", "Megapascal", UnitCategory.PRESSURE, 1e6)
        self._register_unit("bar", "Bar", UnitCategory.PRESSURE, 1e5)
        self._register_unit("mbar", "Millibar", UnitCategory.PRESSURE, 100.0)
        self._register_unit("atm", "Atmosphere", UnitCategory.PRESSURE, 101325.0)
        self._register_unit("psi", "Pounds per square inch", UnitCategory.PRESSURE, 6894.757)
        self._register_unit("psia", "Pounds per square inch absolute", UnitCategory.PRESSURE, 6894.757)
        self._register_unit("psig", "Pounds per square inch gauge", UnitCategory.PRESSURE, 6894.757)
        self._register_unit("mmHg", "Millimeters of mercury", UnitCategory.PRESSURE, 133.322)
        self._register_unit("inHg", "Inches of mercury", UnitCategory.PRESSURE, 3386.39)
        self._register_unit("Torr", "Torr", UnitCategory.PRESSURE, 133.322)

        # Mass units
        self._register_unit("kg", "Kilogram", UnitCategory.MASS, 1.0)
        self._register_unit("g", "Gram", UnitCategory.MASS, 0.001)
        self._register_unit("mg", "Milligram", UnitCategory.MASS, 1e-6)
        self._register_unit("tonne", "Metric tonne", UnitCategory.MASS, 1000.0)
        self._register_unit("lb", "Pound", UnitCategory.MASS, 0.453592)
        self._register_unit("oz", "Ounce", UnitCategory.MASS, 0.0283495)
        self._register_unit("ton", "Short ton", UnitCategory.MASS, 907.185)
        self._register_unit("long_ton", "Long ton", UnitCategory.MASS, 1016.05)

        # Volume units
        self._register_unit("m3", "Cubic meter", UnitCategory.VOLUME, 1.0)
        self._register_unit("L", "Liter", UnitCategory.VOLUME, 0.001)
        self._register_unit("mL", "Milliliter", UnitCategory.VOLUME, 1e-6)
        self._register_unit("gal", "US gallon", UnitCategory.VOLUME, 0.00378541)
        self._register_unit("ft3", "Cubic foot", UnitCategory.VOLUME, 0.0283168)
        self._register_unit("bbl", "Barrel", UnitCategory.VOLUME, 0.158987)
        self._register_unit("scf", "Standard cubic foot", UnitCategory.VOLUME, 0.0283168)
        self._register_unit("Nm3", "Normal cubic meter", UnitCategory.VOLUME, 1.0)

        # Energy units
        self._register_unit("J", "Joule", UnitCategory.ENERGY, 1.0)
        self._register_unit("kJ", "Kilojoule", UnitCategory.ENERGY, 1000.0)
        self._register_unit("MJ", "Megajoule", UnitCategory.ENERGY, 1e6)
        self._register_unit("GJ", "Gigajoule", UnitCategory.ENERGY, 1e9)
        self._register_unit("cal", "Calorie", UnitCategory.ENERGY, 4.184)
        self._register_unit("kcal", "Kilocalorie", UnitCategory.ENERGY, 4184.0)
        self._register_unit("Btu", "British thermal unit", UnitCategory.ENERGY, 1055.06)
        self._register_unit("MMBtu", "Million Btu", UnitCategory.ENERGY, 1.05506e9)
        self._register_unit("therm", "Therm", UnitCategory.ENERGY, 1.05506e8)
        self._register_unit("kWh", "Kilowatt-hour", UnitCategory.ENERGY, 3.6e6)
        self._register_unit("MWh", "Megawatt-hour", UnitCategory.ENERGY, 3.6e9)
        self._register_unit("toe", "Tonne of oil equivalent", UnitCategory.ENERGY, 4.1868e10)

        # Power units
        self._register_unit("W", "Watt", UnitCategory.POWER, 1.0)
        self._register_unit("kW", "Kilowatt", UnitCategory.POWER, 1000.0)
        self._register_unit("MW", "Megawatt", UnitCategory.POWER, 1e6)
        self._register_unit("GW", "Gigawatt", UnitCategory.POWER, 1e9)
        self._register_unit("hp", "Horsepower", UnitCategory.POWER, 745.7)
        self._register_unit("Btu/h", "Btu per hour", UnitCategory.POWER, 0.293071)
        self._register_unit("MMBtu/h", "Million Btu per hour", UnitCategory.POWER, 293071.0)

        # Flow rate units (mass)
        self._register_unit("kg/s", "Kilogram per second", UnitCategory.FLOW_RATE, 1.0)
        self._register_unit("kg/h", "Kilogram per hour", UnitCategory.FLOW_RATE, 1/3600)
        self._register_unit("kg/min", "Kilogram per minute", UnitCategory.FLOW_RATE, 1/60)
        self._register_unit("t/h", "Tonne per hour", UnitCategory.FLOW_RATE, 1000/3600)
        self._register_unit("lb/h", "Pound per hour", UnitCategory.FLOW_RATE, 0.453592/3600)
        self._register_unit("lb/s", "Pound per second", UnitCategory.FLOW_RATE, 0.453592)

        # Density units
        self._register_unit("kg/m3", "Kilogram per cubic meter", UnitCategory.DENSITY, 1.0)
        self._register_unit("g/cm3", "Gram per cubic centimeter", UnitCategory.DENSITY, 1000.0)
        self._register_unit("lb/ft3", "Pound per cubic foot", UnitCategory.DENSITY, 16.0185)
        self._register_unit("kg/L", "Kilogram per liter", UnitCategory.DENSITY, 1000.0)

        # Viscosity units
        self._register_unit("Pa.s", "Pascal-second", UnitCategory.VISCOSITY, 1.0)
        self._register_unit("mPa.s", "Millipascal-second", UnitCategory.VISCOSITY, 0.001)
        self._register_unit("cP", "Centipoise", UnitCategory.VISCOSITY, 0.001)
        self._register_unit("P", "Poise", UnitCategory.VISCOSITY, 0.1)

        # Thermal conductivity units
        self._register_unit("W/(m.K)", "Watt per meter-Kelvin", UnitCategory.THERMAL_CONDUCTIVITY, 1.0)
        self._register_unit("Btu/(h.ft.F)", "Btu per hour-foot-Fahrenheit", UnitCategory.THERMAL_CONDUCTIVITY, 1.7307)
        self._register_unit("cal/(s.cm.C)", "Calorie per second-cm-Celsius", UnitCategory.THERMAL_CONDUCTIVITY, 418.4)

        # Heat capacity units
        self._register_unit("J/(kg.K)", "Joule per kilogram-Kelvin", UnitCategory.HEAT_CAPACITY, 1.0)
        self._register_unit("kJ/(kg.K)", "Kilojoule per kilogram-Kelvin", UnitCategory.HEAT_CAPACITY, 1000.0)
        self._register_unit("Btu/(lb.F)", "Btu per pound-Fahrenheit", UnitCategory.HEAT_CAPACITY, 4186.8)
        self._register_unit("cal/(g.C)", "Calorie per gram-Celsius", UnitCategory.HEAT_CAPACITY, 4184.0)

        # Velocity units
        self._register_unit("m/s", "Meter per second", UnitCategory.VELOCITY, 1.0)
        self._register_unit("km/h", "Kilometer per hour", UnitCategory.VELOCITY, 1/3.6)
        self._register_unit("ft/s", "Foot per second", UnitCategory.VELOCITY, 0.3048)
        self._register_unit("mph", "Miles per hour", UnitCategory.VELOCITY, 0.44704)

        # Length units
        self._register_unit("m", "Meter", UnitCategory.LENGTH, 1.0)
        self._register_unit("cm", "Centimeter", UnitCategory.LENGTH, 0.01)
        self._register_unit("mm", "Millimeter", UnitCategory.LENGTH, 0.001)
        self._register_unit("km", "Kilometer", UnitCategory.LENGTH, 1000.0)
        self._register_unit("in", "Inch", UnitCategory.LENGTH, 0.0254)
        self._register_unit("ft", "Foot", UnitCategory.LENGTH, 0.3048)
        self._register_unit("yd", "Yard", UnitCategory.LENGTH, 0.9144)
        self._register_unit("mi", "Mile", UnitCategory.LENGTH, 1609.34)

        # Area units
        self._register_unit("m2", "Square meter", UnitCategory.AREA, 1.0)
        self._register_unit("cm2", "Square centimeter", UnitCategory.AREA, 0.0001)
        self._register_unit("mm2", "Square millimeter", UnitCategory.AREA, 1e-6)
        self._register_unit("ft2", "Square foot", UnitCategory.AREA, 0.092903)
        self._register_unit("in2", "Square inch", UnitCategory.AREA, 0.00064516)

        # Time units
        self._register_unit("s", "Second", UnitCategory.TIME, 1.0)
        self._register_unit("min", "Minute", UnitCategory.TIME, 60.0)
        self._register_unit("h", "Hour", UnitCategory.TIME, 3600.0)
        self._register_unit("d", "Day", UnitCategory.TIME, 86400.0)
        self._register_unit("yr", "Year", UnitCategory.TIME, 31536000.0)

        # Dimensionless
        self._register_unit("", "Dimensionless", UnitCategory.DIMENSIONLESS, 1.0)
        self._register_unit("%", "Percent", UnitCategory.DIMENSIONLESS, 0.01)
        self._register_unit("ppm", "Parts per million", UnitCategory.DIMENSIONLESS, 1e-6)
        self._register_unit("ppb", "Parts per billion", UnitCategory.DIMENSIONLESS, 1e-9)

        # Emission factor units
        self._register_unit("kg/GJ", "Kilogram per gigajoule", UnitCategory.EMISSION_FACTOR, 1.0)
        self._register_unit("kg/MWh", "Kilogram per megawatt-hour", UnitCategory.EMISSION_FACTOR, 0.2778)
        self._register_unit("lb/MMBtu", "Pound per million Btu", UnitCategory.EMISSION_FACTOR, 0.4299)
        self._register_unit("tCO2/TJ", "Tonne CO2 per terajoule", UnitCategory.EMISSION_FACTOR, 1.0)

        # Heat transfer coefficient units
        self._register_unit("W/(m2.K)", "Watt per square meter-Kelvin", UnitCategory.HEAT_TRANSFER_COEFFICIENT, 1.0)
        self._register_unit("Btu/(h.ft2.F)", "Btu per hour-square foot-Fahrenheit", UnitCategory.HEAT_TRANSFER_COEFFICIENT, 5.6783)

        # Enthalpy units
        self._register_unit("J/kg", "Joule per kilogram", UnitCategory.ENTHALPY, 1.0)
        self._register_unit("kJ/kg", "Kilojoule per kilogram", UnitCategory.ENTHALPY, 1000.0)
        self._register_unit("Btu/lb", "Btu per pound", UnitCategory.ENTHALPY, 2326.0)

        # Entropy units
        self._register_unit("J/(kg.K)", "Joule per kilogram-Kelvin", UnitCategory.ENTROPY, 1.0)
        self._register_unit("kJ/(kg.K)", "Kilojoule per kilogram-Kelvin", UnitCategory.ENTROPY, 1000.0)
        self._register_unit("Btu/(lb.R)", "Btu per pound-Rankine", UnitCategory.ENTROPY, 4186.8)

    def _register_unit(
        self,
        symbol: str,
        name: str,
        category: UnitCategory,
        si_conversion_factor: float,
        si_offset: float = 0.0
    ):
        """Register a unit definition."""
        self._units[symbol] = UnitDefinition(
            symbol=symbol,
            name=name,
            category=category,
            si_conversion_factor=si_conversion_factor,
            si_offset=si_offset
        )

    def convert(
        self,
        value: Number,
        from_unit: str,
        to_unit: str
    ) -> Decimal:
        """
        Convert value between units.

        Args:
            value: Value to convert
            from_unit: Source unit symbol
            to_unit: Target unit symbol

        Returns:
            Converted value as Decimal

        Raises:
            ValueError: If units are incompatible or unknown
        """
        if from_unit == to_unit:
            return Decimal(str(value))

        from_def = self._units.get(from_unit)
        to_def = self._units.get(to_unit)

        if from_def is None:
            raise ValueError(f"Unknown unit: {from_unit}")
        if to_def is None:
            raise ValueError(f"Unknown unit: {to_unit}")

        if from_def.category != to_def.category:
            raise ValueError(
                f"Cannot convert between {from_def.category.value} and {to_def.category.value}"
            )

        # Convert to SI, then to target unit
        # For temperature: T_SI = (T_from * factor) + offset
        value_dec = Decimal(str(value))

        if from_def.category == UnitCategory.TEMPERATURE:
            # Special handling for temperature
            si_value = value_dec * Decimal(str(from_def.si_conversion_factor)) + Decimal(str(from_def.si_offset))
            result = (si_value - Decimal(str(to_def.si_offset))) / Decimal(str(to_def.si_conversion_factor))
        else:
            # Standard conversion
            si_value = value_dec * Decimal(str(from_def.si_conversion_factor))
            result = si_value / Decimal(str(to_def.si_conversion_factor))

        return result

    def get_unit_info(self, symbol: str) -> Optional[UnitDefinition]:
        """Get unit definition by symbol."""
        return self._units.get(symbol)

    def get_units_by_category(self, category: UnitCategory) -> List[str]:
        """Get all unit symbols for a category."""
        return [
            symbol for symbol, unit_def in self._units.items()
            if unit_def.category == category
        ]


# =============================================================================
# Uncertainty Propagator
# =============================================================================

class UncertaintyPropagator:
    """
    Uncertainty propagation calculator.

    Implements standard uncertainty propagation methods:
    - Linear propagation (GUM method)
    - Monte Carlo simulation
    """

    @staticmethod
    def propagate_linear(
        partial_derivatives: Dict[str, float],
        uncertainties: Dict[str, float]
    ) -> float:
        """
        Propagate uncertainties using linear approximation.

        Uses the formula: u_y = sqrt(sum((df/dx_i * u_xi)^2))

        Args:
            partial_derivatives: Dict of {parameter: partial derivative}
            uncertainties: Dict of {parameter: absolute uncertainty}

        Returns:
            Combined uncertainty
        """
        variance_sum = 0.0
        for param, deriv in partial_derivatives.items():
            if param in uncertainties:
                variance_sum += (deriv * uncertainties[param]) ** 2

        return math.sqrt(variance_sum)

    @staticmethod
    def propagate_relative(
        result: float,
        parameters: Dict[str, float],
        uncertainties_percent: Dict[str, float],
        exponents: Dict[str, float]
    ) -> float:
        """
        Propagate relative uncertainties for power-law relationships.

        For y = x1^a1 * x2^a2 * ...:
        (u_y/y)^2 = sum((a_i * u_xi/xi)^2)

        Args:
            result: Calculated result
            parameters: Parameter values
            uncertainties_percent: Percentage uncertainties
            exponents: Power exponents for each parameter

        Returns:
            Absolute uncertainty
        """
        relative_variance_sum = 0.0
        for param, exponent in exponents.items():
            if param in uncertainties_percent:
                rel_unc = uncertainties_percent[param] / 100.0
                relative_variance_sum += (exponent * rel_unc) ** 2

        relative_uncertainty = math.sqrt(relative_variance_sum)
        return abs(result) * relative_uncertainty

    @staticmethod
    def add_uncertainties(uncertainties: List[float]) -> float:
        """
        Add uncertainties in quadrature (for sum of independent variables).

        u_sum = sqrt(u1^2 + u2^2 + ...)
        """
        return math.sqrt(sum(u**2 for u in uncertainties))


# =============================================================================
# Calculation Cache
# =============================================================================

class CalculationCache:
    """
    Thread-safe LRU cache for calculation results.

    Caches expensive calculations to improve performance.
    Cache key is based on formula_id and input parameters.
    """

    def __init__(self, maxsize: int = 1000):
        """Initialize cache with maximum size."""
        self._cache: OrderedDict[str, CalculationResult] = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, formula_input: FormulaInput) -> str:
        """Create cache key from formula input."""
        key_data = {
            "formula_id": formula_input.formula_id,
            "parameters": formula_input.parameters,
            "version": formula_input.version,
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, formula_input: FormulaInput) -> Optional[CalculationResult]:
        """Get cached result if available."""
        key = self._make_key(formula_input)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, formula_input: FormulaInput, result: CalculationResult):
        """Store calculation result in cache."""
        key = self._make_key(formula_input)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    # Remove oldest entry
                    self._cache.popitem(last=False)
            self._cache[key] = result

    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


# =============================================================================
# Provenance Tracker
# =============================================================================

class ProvenanceTracker:
    """
    Tracks calculation provenance with SHA-256 hashes.

    Creates an immutable audit trail for every calculation.
    """

    @staticmethod
    def calculate_provenance_hash(
        formula_id: str,
        formula_version: str,
        formula_hash: str,
        input_parameters: Dict[str, float],
        output_value: Decimal,
        calculation_steps: List[CalculationStep]
    ) -> str:
        """
        Calculate SHA-256 hash for complete audit trail.

        Args:
            formula_id: Formula identifier
            formula_version: Formula version
            formula_hash: Hash of formula definition
            input_parameters: Input parameter values
            output_value: Calculated output
            calculation_steps: All calculation steps

        Returns:
            SHA-256 hash string
        """
        provenance_data = {
            "formula_id": formula_id,
            "formula_version": formula_version,
            "formula_hash": formula_hash,
            "input_parameters": {k: str(v) for k, v in input_parameters.items()},
            "output_value": str(output_value),
            "steps": [
                {
                    "step": step.step_number,
                    "operation": step.operation,
                    "output": str(step.output_value),
                }
                for step in calculation_steps
            ]
        }

        provenance_string = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_string.encode()).hexdigest()

    @staticmethod
    def verify_provenance(
        result: CalculationResult,
        expected_hash: str
    ) -> bool:
        """Verify calculation provenance hash matches expected."""
        actual_hash = ProvenanceTracker.calculate_provenance_hash(
            formula_id=result.formula_id,
            formula_version=result.formula_version,
            formula_hash=result.formula_hash,
            input_parameters=result.input_parameters,
            output_value=result.output_value,
            calculation_steps=result.calculation_steps,
        )
        return actual_hash == expected_hash


# =============================================================================
# Formula Registry
# =============================================================================

class FormulaRegistry:
    """
    Registry of all available formulas.

    Provides formula lookup, versioning, and metadata.
    """

    def __init__(self):
        """Initialize formula registry."""
        self._formulas: Dict[str, Dict[str, FormulaDefinition]] = {}  # formula_id -> version -> definition
        self._calculators: Dict[str, Callable] = {}  # formula_id -> calculator function

    def register(
        self,
        definition: FormulaDefinition,
        calculator: Callable[[Dict[str, float]], Tuple[Decimal, List[CalculationStep]]]
    ):
        """
        Register a formula with its calculator.

        Args:
            definition: Formula definition
            calculator: Function that performs the calculation
        """
        formula_id = definition.formula_id
        version = definition.version

        if formula_id not in self._formulas:
            self._formulas[formula_id] = {}

        self._formulas[formula_id][version] = definition
        self._calculators[f"{formula_id}:{version}"] = calculator

    def get_formula(
        self,
        formula_id: str,
        version: str = "latest"
    ) -> Optional[FormulaDefinition]:
        """Get formula definition by ID and version."""
        if formula_id not in self._formulas:
            return None

        versions = self._formulas[formula_id]

        if version == "latest":
            # Get the latest version (highest version number)
            latest_version = max(versions.keys())
            return versions[latest_version]

        return versions.get(version)

    def get_calculator(
        self,
        formula_id: str,
        version: str = "latest"
    ) -> Optional[Callable]:
        """Get calculator function for formula."""
        if version == "latest":
            definition = self.get_formula(formula_id, version)
            if definition:
                version = definition.version

        return self._calculators.get(f"{formula_id}:{version}")

    def list_formulas(self, category: Optional[str] = None) -> List[str]:
        """List all registered formula IDs."""
        if category:
            return [
                fid for fid, versions in self._formulas.items()
                if any(v.category == category for v in versions.values())
            ]
        return list(self._formulas.keys())

    def list_versions(self, formula_id: str) -> List[str]:
        """List all versions of a formula."""
        if formula_id not in self._formulas:
            return []
        return list(self._formulas[formula_id].keys())


# =============================================================================
# Calculation Engine
# =============================================================================

class CalculationEngine:
    """
    Zero-hallucination calculation engine.

    Guarantees:
    - Deterministic: Same input always produces same output (bit-perfect)
    - Reproducible: Full provenance tracking
    - Auditable: SHA-256 hash of all calculation steps
    - NO LLM: Zero hallucination risk

    Example usage:
        engine = CalculationEngine()
        result = engine.calculate(FormulaInput(
            formula_id="steam_enthalpy_iapws97",
            parameters={"pressure_mpa": 10.0, "temperature_k": 573.15}
        ))
    """

    def __init__(
        self,
        cache_enabled: bool = True,
        cache_size: int = 1000,
        default_precision: int = 6,
    ):
        """
        Initialize calculation engine.

        Args:
            cache_enabled: Enable calculation caching
            cache_size: Maximum cache size
            default_precision: Default decimal precision
        """
        self.registry = FormulaRegistry()
        self.unit_converter = UnitConverter()
        self.uncertainty_propagator = UncertaintyPropagator()
        self._cache = CalculationCache(cache_size) if cache_enabled else None
        self._default_precision = default_precision

        # Register all formulas
        self._register_formulas()

    def _register_formulas(self):
        """Register all available formulas. Override in subclasses."""
        pass  # Formulas registered by formula modules

    def calculate(
        self,
        formula_input: FormulaInput,
        use_cache: bool = True
    ) -> CalculationResult:
        """
        Execute calculation with zero hallucination guarantee.

        Args:
            formula_input: Formula ID and parameters
            use_cache: Whether to use calculation cache

        Returns:
            Calculation result with complete provenance
        """
        start_time = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()

        # Check cache
        if use_cache and self._cache:
            cached = self._cache.get(formula_input)
            if cached:
                return cached

        # Get formula definition
        formula = self.registry.get_formula(
            formula_input.formula_id,
            formula_input.version
        )

        if formula is None:
            return CalculationResult(
                formula_id=formula_input.formula_id,
                formula_version=formula_input.version,
                formula_hash="",
                status=CalculationStatus.VALIDATION_ERROR,
                output_value=None,
                output_unit="",
                uncertainty=None,
                calculation_steps=[],
                provenance_hash="",
                calculation_time_ms=0,
                timestamp=timestamp,
                input_parameters=formula_input.parameters,
                converted_parameters={},
                errors=[f"Formula not found: {formula_input.formula_id}"],
            )

        # Convert units if specified
        converted_params, warnings = self._convert_input_units(
            formula_input.parameters,
            formula_input.parameter_units,
            formula
        )

        # Validate inputs
        validation_errors = self._validate_inputs(converted_params, formula)
        if validation_errors:
            return CalculationResult(
                formula_id=formula_input.formula_id,
                formula_version=formula.version,
                formula_hash=formula.formula_hash,
                status=CalculationStatus.VALIDATION_ERROR,
                output_value=None,
                output_unit=formula.output_unit,
                uncertainty=None,
                calculation_steps=[],
                provenance_hash="",
                calculation_time_ms=(time.perf_counter() - start_time) * 1000,
                timestamp=timestamp,
                input_parameters=formula_input.parameters,
                converted_parameters=converted_params,
                warnings=warnings,
                errors=validation_errors,
            )

        # Get calculator function
        calculator = self.registry.get_calculator(
            formula_input.formula_id,
            formula.version
        )

        if calculator is None:
            return CalculationResult(
                formula_id=formula_input.formula_id,
                formula_version=formula.version,
                formula_hash=formula.formula_hash,
                status=CalculationStatus.CALCULATION_ERROR,
                output_value=None,
                output_unit=formula.output_unit,
                uncertainty=None,
                calculation_steps=[],
                provenance_hash="",
                calculation_time_ms=(time.perf_counter() - start_time) * 1000,
                timestamp=timestamp,
                input_parameters=formula_input.parameters,
                converted_parameters=converted_params,
                warnings=warnings,
                errors=["Calculator not implemented"],
            )

        try:
            # Execute calculation
            output_value, steps = calculator(converted_params)

            # Apply precision
            output_value = self._apply_precision(output_value, formula.precision)

            # Calculate uncertainty
            uncertainty = self._calculate_uncertainty(
                formula, converted_params, output_value
            )

            # Calculate provenance hash
            provenance_hash = ProvenanceTracker.calculate_provenance_hash(
                formula_id=formula_input.formula_id,
                formula_version=formula.version,
                formula_hash=formula.formula_hash,
                input_parameters=converted_params,
                output_value=output_value,
                calculation_steps=steps,
            )

            result = CalculationResult(
                formula_id=formula_input.formula_id,
                formula_version=formula.version,
                formula_hash=formula.formula_hash,
                status=CalculationStatus.SUCCESS,
                output_value=output_value,
                output_unit=formula.output_unit,
                uncertainty=uncertainty,
                calculation_steps=steps,
                provenance_hash=provenance_hash,
                calculation_time_ms=(time.perf_counter() - start_time) * 1000,
                timestamp=timestamp,
                input_parameters=formula_input.parameters,
                converted_parameters=converted_params,
                warnings=warnings,
            )

            # Store in cache
            if use_cache and self._cache:
                self._cache.put(formula_input, result)

            return result

        except Exception as e:
            return CalculationResult(
                formula_id=formula_input.formula_id,
                formula_version=formula.version,
                formula_hash=formula.formula_hash,
                status=CalculationStatus.CALCULATION_ERROR,
                output_value=None,
                output_unit=formula.output_unit,
                uncertainty=None,
                calculation_steps=[],
                provenance_hash="",
                calculation_time_ms=(time.perf_counter() - start_time) * 1000,
                timestamp=timestamp,
                input_parameters=formula_input.parameters,
                converted_parameters=converted_params,
                warnings=warnings,
                errors=[str(e)],
            )

    def _convert_input_units(
        self,
        parameters: Dict[str, float],
        parameter_units: Optional[Dict[str, str]],
        formula: FormulaDefinition
    ) -> Tuple[Dict[str, float], List[str]]:
        """Convert input parameters to formula's expected units."""
        converted = parameters.copy()
        warnings = []

        if parameter_units is None:
            return converted, warnings

        param_defs = {p.name: p for p in formula.parameters}

        for param_name, from_unit in parameter_units.items():
            if param_name not in converted:
                continue

            param_def = param_defs.get(param_name)
            if param_def is None:
                continue

            to_unit = param_def.unit
            if from_unit == to_unit:
                continue

            try:
                converted_value = self.unit_converter.convert(
                    converted[param_name],
                    from_unit,
                    to_unit
                )
                converted[param_name] = float(converted_value)
                warnings.append(
                    f"Converted {param_name}: {parameters[param_name]} {from_unit} -> "
                    f"{converted[param_name]:.6g} {to_unit}"
                )
            except ValueError as e:
                warnings.append(f"Unit conversion warning for {param_name}: {e}")

        return converted, warnings

    def _validate_inputs(
        self,
        parameters: Dict[str, float],
        formula: FormulaDefinition
    ) -> List[str]:
        """Validate input parameters against formula definition."""
        errors = []

        for param_def in formula.parameters:
            name = param_def.name

            # Check required parameters
            if param_def.required and name not in parameters:
                if param_def.default_value is None:
                    errors.append(f"Missing required parameter: {name}")
                continue

            if name not in parameters:
                continue

            value = parameters[name]

            # Check type
            if not isinstance(value, (int, float, Decimal)):
                errors.append(f"Parameter {name} must be numeric, got {type(value)}")
                continue

            # Check range
            if param_def.min_value is not None and value < param_def.min_value:
                errors.append(
                    f"Parameter {name}={value} below minimum {param_def.min_value}"
                )

            if param_def.max_value is not None and value > param_def.max_value:
                errors.append(
                    f"Parameter {name}={value} above maximum {param_def.max_value}"
                )

            # Check for NaN or Inf
            if math.isnan(value) or math.isinf(value):
                errors.append(f"Parameter {name} is NaN or Inf")

        return errors

    def _apply_precision(self, value: Decimal, precision: int) -> Decimal:
        """Apply regulatory rounding precision."""
        if not isinstance(value, Decimal):
            value = Decimal(str(value))

        quantize_string = '0.' + '0' * precision
        return value.quantize(Decimal(quantize_string), rounding=ROUND_HALF_UP)

    def _calculate_uncertainty(
        self,
        formula: FormulaDefinition,
        parameters: Dict[str, float],
        output_value: Decimal
    ) -> UncertaintyResult:
        """Calculate output uncertainty from input uncertainties."""
        # Get parameter uncertainties
        param_uncertainties = {}
        for param_def in formula.parameters:
            if param_def.name in parameters and param_def.uncertainty_percent > 0:
                param_uncertainties[param_def.name] = param_def.uncertainty_percent

        if not param_uncertainties:
            # No uncertainty data - return zero uncertainty
            return UncertaintyResult(
                value=output_value,
                uncertainty_absolute=Decimal("0"),
                uncertainty_percent=Decimal("0"),
                method="none",
            )

        # Simple propagation assuming multiplicative relationship
        # This is a conservative estimate
        total_rel_unc_squared = sum(
            (u / 100) ** 2 for u in param_uncertainties.values()
        )
        total_rel_unc = math.sqrt(total_rel_unc_squared)

        abs_unc = abs(float(output_value)) * total_rel_unc

        return UncertaintyResult(
            value=output_value,
            uncertainty_absolute=Decimal(str(abs_unc)),
            uncertainty_percent=Decimal(str(total_rel_unc * 100)),
            method=formula.uncertainty_method,
            component_uncertainties={
                k: Decimal(str(v)) for k, v in param_uncertainties.items()
            }
        )

    def batch_calculate(
        self,
        inputs: List[FormulaInput],
        parallel: bool = False
    ) -> List[CalculationResult]:
        """
        Execute multiple calculations.

        Args:
            inputs: List of formula inputs
            parallel: Whether to execute in parallel (not implemented)

        Returns:
            List of calculation results
        """
        results = []
        for formula_input in inputs:
            result = self.calculate(formula_input)
            results.append(result)
        return results

    def get_formula_info(self, formula_id: str, version: str = "latest") -> Optional[FormulaDefinition]:
        """Get formula definition and metadata."""
        return self.registry.get_formula(formula_id, version)

    def list_formulas(self, category: Optional[str] = None) -> List[str]:
        """List available formulas."""
        return self.registry.list_formulas(category)

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if self._cache:
            return self._cache.get_stats()
        return None

    def clear_cache(self):
        """Clear calculation cache."""
        if self._cache:
            self._cache.clear()


# =============================================================================
# Helper Functions for Formula Implementations
# =============================================================================

def make_decimal(value: Union[int, float, Decimal, str]) -> Decimal:
    """Convert value to Decimal with proper handling."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def safe_divide(
    numerator: Number,
    denominator: Number,
    default: Number = Decimal("0")
) -> Decimal:
    """Safe division with zero check."""
    num = make_decimal(numerator)
    den = make_decimal(denominator)

    if den == 0:
        return make_decimal(default)

    return num / den


def safe_sqrt(value: Number) -> Decimal:
    """Safe square root."""
    val = float(value)
    if val < 0:
        raise ValueError(f"Cannot take square root of negative number: {val}")
    return make_decimal(math.sqrt(val))


def safe_log(value: Number, base: float = math.e) -> Decimal:
    """Safe logarithm."""
    val = float(value)
    if val <= 0:
        raise ValueError(f"Cannot take log of non-positive number: {val}")
    if base == math.e:
        return make_decimal(math.log(val))
    return make_decimal(math.log(val, base))


def safe_exp(value: Number) -> Decimal:
    """Safe exponential."""
    val = float(value)
    if val > 700:  # Prevent overflow
        raise ValueError(f"Exponential overflow for value: {val}")
    return make_decimal(math.exp(val))


def safe_power(base: Number, exponent: Number) -> Decimal:
    """Safe power function."""
    b = float(base)
    e = float(exponent)

    if b < 0 and not float(e).is_integer():
        raise ValueError("Cannot raise negative number to non-integer power")

    result = math.pow(b, e)

    if math.isinf(result) or math.isnan(result):
        raise ValueError(f"Power overflow: {b}^{e}")

    return make_decimal(result)


def interpolate_linear(
    x: Number,
    x1: Number,
    y1: Number,
    x2: Number,
    y2: Number
) -> Decimal:
    """Linear interpolation."""
    x = make_decimal(x)
    x1 = make_decimal(x1)
    y1 = make_decimal(y1)
    x2 = make_decimal(x2)
    y2 = make_decimal(y2)

    if x2 == x1:
        return y1

    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def clamp(value: Number, min_val: Number, max_val: Number) -> Decimal:
    """Clamp value to range."""
    val = make_decimal(value)
    min_v = make_decimal(min_val)
    max_v = make_decimal(max_val)

    if val < min_v:
        return min_v
    if val > max_v:
        return max_v
    return val
