"""
Data Transformers Module for GL-014 EXCHANGER-PRO (Heat Exchanger Optimization Agent).

Provides ETL utilities for data ingestion including:
- Unit conversion (temperature, pressure, flow, etc.)
- Schema transformation (vendor to canonical format)
- Data quality assessment
- Outlier detection
- Missing data handling
- Time alignment

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)
import logging
import math
import re
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict

from pydantic import BaseModel, Field, ConfigDict

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class UnitCategory(str, Enum):
    """Categories of measurement units for heat exchangers."""

    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW_RATE_MASS = "flow_rate_mass"
    FLOW_RATE_VOLUME = "flow_rate_volume"
    POWER = "power"
    ENERGY = "energy"
    HEAT_TRANSFER_COEFFICIENT = "heat_transfer_coefficient"  # UA
    FOULING_RESISTANCE = "fouling_resistance"  # m2K/W
    AREA = "area"
    LENGTH = "length"
    VELOCITY = "velocity"
    DENSITY = "density"
    VISCOSITY = "viscosity"
    THERMAL_CONDUCTIVITY = "thermal_conductivity"
    SPECIFIC_HEAT = "specific_heat"


class TimeZoneHandling(str, Enum):
    """Timezone handling strategies."""

    PRESERVE = "preserve"
    CONVERT_TO_UTC = "convert_to_utc"
    CONVERT_TO_LOCAL = "convert_to_local"
    STRIP_TIMEZONE = "strip_timezone"


class MissingDataStrategy(str, Enum):
    """Strategies for handling missing data."""

    DROP = "drop"
    FILL_ZERO = "fill_zero"
    FILL_NULL = "fill_null"
    FILL_MEAN = "fill_mean"
    FILL_MEDIAN = "fill_median"
    FILL_FORWARD = "fill_forward"
    FILL_BACKWARD = "fill_backward"
    INTERPOLATE_LINEAR = "interpolate_linear"
    FILL_DEFAULT = "fill_default"


class OutlierMethod(str, Enum):
    """Methods for outlier detection."""

    Z_SCORE = "z_score"
    MODIFIED_Z_SCORE = "modified_z_score"
    IQR = "iqr"
    BOUNDS = "bounds"
    RATE_OF_CHANGE = "rate_of_change"


class DataQualityDimension(str, Enum):
    """Data quality dimensions."""

    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    RANGE_CONFORMITY = "range_conformity"


# =============================================================================
# Unit Conversion
# =============================================================================


@dataclass
class UnitDefinition:
    """Definition of a measurement unit."""

    symbol: str
    name: str
    category: UnitCategory
    base_unit: str
    to_base_factor: float
    to_base_offset: float = 0.0


# Comprehensive unit definitions for heat exchanger systems
UNIT_DEFINITIONS: Dict[str, UnitDefinition] = {
    # Temperature
    "K": UnitDefinition("K", "kelvin", UnitCategory.TEMPERATURE, "K", 1.0),
    "C": UnitDefinition("C", "celsius", UnitCategory.TEMPERATURE, "K", 1.0, 273.15),
    "F": UnitDefinition("F", "fahrenheit", UnitCategory.TEMPERATURE, "K", 5/9, 255.372),
    "R": UnitDefinition("R", "rankine", UnitCategory.TEMPERATURE, "K", 5/9),

    # Pressure
    "Pa": UnitDefinition("Pa", "pascal", UnitCategory.PRESSURE, "Pa", 1.0),
    "kPa": UnitDefinition("kPa", "kilopascal", UnitCategory.PRESSURE, "Pa", 1000.0),
    "MPa": UnitDefinition("MPa", "megapascal", UnitCategory.PRESSURE, "Pa", 1000000.0),
    "bar": UnitDefinition("bar", "bar", UnitCategory.PRESSURE, "Pa", 100000.0),
    "mbar": UnitDefinition("mbar", "millibar", UnitCategory.PRESSURE, "Pa", 100.0),
    "psi": UnitDefinition("psi", "pounds per square inch", UnitCategory.PRESSURE, "Pa", 6894.757),
    "psig": UnitDefinition("psig", "psi gauge", UnitCategory.PRESSURE, "Pa", 6894.757),
    "atm": UnitDefinition("atm", "atmosphere", UnitCategory.PRESSURE, "Pa", 101325.0),
    "mmHg": UnitDefinition("mmHg", "millimeters of mercury", UnitCategory.PRESSURE, "Pa", 133.322),
    "inHg": UnitDefinition("inHg", "inches of mercury", UnitCategory.PRESSURE, "Pa", 3386.39),
    "inH2O": UnitDefinition("inH2O", "inches of water", UnitCategory.PRESSURE, "Pa", 249.089),

    # Mass flow rate
    "kg/s": UnitDefinition("kg/s", "kilograms per second", UnitCategory.FLOW_RATE_MASS, "kg/s", 1.0),
    "kg/h": UnitDefinition("kg/h", "kilograms per hour", UnitCategory.FLOW_RATE_MASS, "kg/s", 1/3600),
    "kg/min": UnitDefinition("kg/min", "kilograms per minute", UnitCategory.FLOW_RATE_MASS, "kg/s", 1/60),
    "t/h": UnitDefinition("t/h", "tonnes per hour", UnitCategory.FLOW_RATE_MASS, "kg/s", 1000/3600),
    "lb/s": UnitDefinition("lb/s", "pounds per second", UnitCategory.FLOW_RATE_MASS, "kg/s", 0.453592),
    "lb/h": UnitDefinition("lb/h", "pounds per hour", UnitCategory.FLOW_RATE_MASS, "kg/s", 0.453592/3600),
    "lb/min": UnitDefinition("lb/min", "pounds per minute", UnitCategory.FLOW_RATE_MASS, "kg/s", 0.453592/60),

    # Volumetric flow rate
    "m3/s": UnitDefinition("m3/s", "cubic meters per second", UnitCategory.FLOW_RATE_VOLUME, "m3/s", 1.0),
    "m3/h": UnitDefinition("m3/h", "cubic meters per hour", UnitCategory.FLOW_RATE_VOLUME, "m3/s", 1/3600),
    "L/s": UnitDefinition("L/s", "liters per second", UnitCategory.FLOW_RATE_VOLUME, "m3/s", 0.001),
    "L/min": UnitDefinition("L/min", "liters per minute", UnitCategory.FLOW_RATE_VOLUME, "m3/s", 0.001/60),
    "L/h": UnitDefinition("L/h", "liters per hour", UnitCategory.FLOW_RATE_VOLUME, "m3/s", 0.001/3600),
    "gpm": UnitDefinition("gpm", "gallons per minute", UnitCategory.FLOW_RATE_VOLUME, "m3/s", 0.0000630902),
    "gph": UnitDefinition("gph", "gallons per hour", UnitCategory.FLOW_RATE_VOLUME, "m3/s", 0.0000630902/60),
    "ft3/s": UnitDefinition("ft3/s", "cubic feet per second", UnitCategory.FLOW_RATE_VOLUME, "m3/s", 0.0283168),
    "ft3/min": UnitDefinition("ft3/min", "cubic feet per minute", UnitCategory.FLOW_RATE_VOLUME, "m3/s", 0.0283168/60),
    "bbl/d": UnitDefinition("bbl/d", "barrels per day", UnitCategory.FLOW_RATE_VOLUME, "m3/s", 0.158987/86400),

    # Power/Heat duty
    "W": UnitDefinition("W", "watt", UnitCategory.POWER, "W", 1.0),
    "kW": UnitDefinition("kW", "kilowatt", UnitCategory.POWER, "W", 1000.0),
    "MW": UnitDefinition("MW", "megawatt", UnitCategory.POWER, "W", 1000000.0),
    "BTU/h": UnitDefinition("BTU/h", "BTU per hour", UnitCategory.POWER, "W", 0.293071),
    "BTU/s": UnitDefinition("BTU/s", "BTU per second", UnitCategory.POWER, "W", 1055.06),
    "hp": UnitDefinition("hp", "horsepower", UnitCategory.POWER, "W", 745.7),
    "kcal/h": UnitDefinition("kcal/h", "kilocalories per hour", UnitCategory.POWER, "W", 1.163),
    "MJ/h": UnitDefinition("MJ/h", "megajoules per hour", UnitCategory.POWER, "W", 277.778),

    # Energy
    "J": UnitDefinition("J", "joule", UnitCategory.ENERGY, "J", 1.0),
    "kJ": UnitDefinition("kJ", "kilojoule", UnitCategory.ENERGY, "J", 1000.0),
    "MJ": UnitDefinition("MJ", "megajoule", UnitCategory.ENERGY, "J", 1000000.0),
    "Wh": UnitDefinition("Wh", "watt-hour", UnitCategory.ENERGY, "J", 3600.0),
    "kWh": UnitDefinition("kWh", "kilowatt-hour", UnitCategory.ENERGY, "J", 3600000.0),
    "MWh": UnitDefinition("MWh", "megawatt-hour", UnitCategory.ENERGY, "J", 3600000000.0),
    "BTU": UnitDefinition("BTU", "British thermal unit", UnitCategory.ENERGY, "J", 1055.06),
    "cal": UnitDefinition("cal", "calorie", UnitCategory.ENERGY, "J", 4.184),
    "kcal": UnitDefinition("kcal", "kilocalorie", UnitCategory.ENERGY, "J", 4184.0),

    # Heat transfer coefficient (UA)
    "W/K": UnitDefinition("W/K", "watts per kelvin", UnitCategory.HEAT_TRANSFER_COEFFICIENT, "W/K", 1.0),
    "kW/K": UnitDefinition("kW/K", "kilowatts per kelvin", UnitCategory.HEAT_TRANSFER_COEFFICIENT, "W/K", 1000.0),
    "W/C": UnitDefinition("W/C", "watts per celsius", UnitCategory.HEAT_TRANSFER_COEFFICIENT, "W/K", 1.0),
    "BTU/h.F": UnitDefinition("BTU/h.F", "BTU per hour fahrenheit", UnitCategory.HEAT_TRANSFER_COEFFICIENT, "W/K", 0.5275),

    # Fouling resistance
    "m2K/W": UnitDefinition("m2K/W", "square meters kelvin per watt", UnitCategory.FOULING_RESISTANCE, "m2K/W", 1.0),
    "m2C/W": UnitDefinition("m2C/W", "square meters celsius per watt", UnitCategory.FOULING_RESISTANCE, "m2K/W", 1.0),
    "ft2hF/BTU": UnitDefinition("ft2hF/BTU", "sq ft hour F per BTU", UnitCategory.FOULING_RESISTANCE, "m2K/W", 0.1761),

    # Area
    "m2": UnitDefinition("m2", "square meters", UnitCategory.AREA, "m2", 1.0),
    "cm2": UnitDefinition("cm2", "square centimeters", UnitCategory.AREA, "m2", 0.0001),
    "ft2": UnitDefinition("ft2", "square feet", UnitCategory.AREA, "m2", 0.092903),
    "in2": UnitDefinition("in2", "square inches", UnitCategory.AREA, "m2", 0.00064516),

    # Length
    "m": UnitDefinition("m", "meter", UnitCategory.LENGTH, "m", 1.0),
    "mm": UnitDefinition("mm", "millimeter", UnitCategory.LENGTH, "m", 0.001),
    "cm": UnitDefinition("cm", "centimeter", UnitCategory.LENGTH, "m", 0.01),
    "ft": UnitDefinition("ft", "foot", UnitCategory.LENGTH, "m", 0.3048),
    "in": UnitDefinition("in", "inch", UnitCategory.LENGTH, "m", 0.0254),

    # Velocity
    "m/s": UnitDefinition("m/s", "meters per second", UnitCategory.VELOCITY, "m/s", 1.0),
    "ft/s": UnitDefinition("ft/s", "feet per second", UnitCategory.VELOCITY, "m/s", 0.3048),
    "km/h": UnitDefinition("km/h", "kilometers per hour", UnitCategory.VELOCITY, "m/s", 0.277778),

    # Density
    "kg/m3": UnitDefinition("kg/m3", "kilograms per cubic meter", UnitCategory.DENSITY, "kg/m3", 1.0),
    "g/cm3": UnitDefinition("g/cm3", "grams per cubic centimeter", UnitCategory.DENSITY, "kg/m3", 1000.0),
    "lb/ft3": UnitDefinition("lb/ft3", "pounds per cubic foot", UnitCategory.DENSITY, "kg/m3", 16.0185),

    # Viscosity (dynamic)
    "Pa.s": UnitDefinition("Pa.s", "pascal seconds", UnitCategory.VISCOSITY, "Pa.s", 1.0),
    "mPa.s": UnitDefinition("mPa.s", "millipascal seconds", UnitCategory.VISCOSITY, "Pa.s", 0.001),
    "cP": UnitDefinition("cP", "centipoise", UnitCategory.VISCOSITY, "Pa.s", 0.001),

    # Thermal conductivity
    "W/mK": UnitDefinition("W/mK", "watts per meter kelvin", UnitCategory.THERMAL_CONDUCTIVITY, "W/mK", 1.0),
    "BTU/hftF": UnitDefinition("BTU/hftF", "BTU per hour foot F", UnitCategory.THERMAL_CONDUCTIVITY, "W/mK", 1.7307),

    # Specific heat
    "J/kgK": UnitDefinition("J/kgK", "joules per kg kelvin", UnitCategory.SPECIFIC_HEAT, "J/kgK", 1.0),
    "kJ/kgK": UnitDefinition("kJ/kgK", "kilojoules per kg kelvin", UnitCategory.SPECIFIC_HEAT, "J/kgK", 1000.0),
    "BTU/lbF": UnitDefinition("BTU/lbF", "BTU per pound F", UnitCategory.SPECIFIC_HEAT, "J/kgK", 4186.8),
}


class UnitConverter:
    """Handles unit conversions for heat exchanger data."""

    def __init__(self) -> None:
        """Initialize unit converter."""
        self._definitions = UNIT_DEFINITIONS.copy()
        self._aliases: Dict[str, str] = {
            # Temperature aliases
            "celsius": "C",
            "fahrenheit": "F",
            "kelvin": "K",
            "degC": "C",
            "degF": "F",
            "deg_c": "C",
            "deg_f": "F",

            # Pressure aliases
            "pascal": "Pa",
            "kilopascal": "kPa",
            "megapascal": "MPa",
            "barg": "bar",
            "psia": "psi",

            # Flow aliases
            "kg/hr": "kg/h",
            "m3/hr": "m3/h",
            "l/s": "L/s",
            "l/min": "L/min",
            "gal/min": "gpm",

            # Power aliases
            "watt": "W",
            "kilowatt": "kW",
            "megawatt": "MW",
            "btu/hr": "BTU/h",
            "btu/h": "BTU/h",

            # Heat transfer aliases
            "w/k": "W/K",
            "kw/k": "kW/K",
        }

    def add_unit(self, definition: UnitDefinition) -> None:
        """Add a custom unit definition."""
        self._definitions[definition.symbol] = definition

    def add_alias(self, alias: str, unit: str) -> None:
        """Add a unit alias."""
        self._aliases[alias.lower()] = unit

    def normalize_unit(self, unit: str) -> str:
        """Normalize unit symbol to standard form."""
        # Check aliases first
        lower_unit = unit.lower().strip()
        if lower_unit in self._aliases:
            return self._aliases[lower_unit]

        # Check for exact match
        if unit in self._definitions:
            return unit

        # Try case-insensitive match
        for symbol in self._definitions:
            if symbol.lower() == lower_unit:
                return symbol

        return unit

    def get_category(self, unit: str) -> Optional[UnitCategory]:
        """Get category for a unit."""
        normalized = self.normalize_unit(unit)
        if normalized in self._definitions:
            return self._definitions[normalized].category
        return None

    def convert(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
    ) -> float:
        """
        Convert value between units.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value

        Raises:
            ValueError: If units are incompatible or unknown
        """
        from_normalized = self.normalize_unit(from_unit)
        to_normalized = self.normalize_unit(to_unit)

        if from_normalized == to_normalized:
            return value

        if from_normalized not in self._definitions:
            raise ValueError(f"Unknown unit: {from_unit}")
        if to_normalized not in self._definitions:
            raise ValueError(f"Unknown unit: {to_unit}")

        from_def = self._definitions[from_normalized]
        to_def = self._definitions[to_normalized]

        if from_def.category != to_def.category:
            raise ValueError(
                f"Cannot convert between {from_def.category.value} and {to_def.category.value}"
            )

        # Convert to base unit
        base_value = value * from_def.to_base_factor + from_def.to_base_offset

        # Convert from base to target
        result = (base_value - to_def.to_base_offset) / to_def.to_base_factor

        return result

    def convert_batch(
        self,
        values: List[float],
        from_unit: str,
        to_unit: str,
    ) -> List[float]:
        """Convert a batch of values."""
        return [self.convert(v, from_unit, to_unit) for v in values]


# =============================================================================
# Timestamp Normalization
# =============================================================================


class TimestampNormalizer:
    """Normalizes timestamps from various formats and timezones."""

    COMMON_FORMATS = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%Y%m%d%H%M%S",
    ]

    def __init__(
        self,
        handling: TimeZoneHandling = TimeZoneHandling.CONVERT_TO_UTC,
        default_timezone: Optional[str] = None
    ) -> None:
        """Initialize timestamp normalizer."""
        self._handling = handling
        self._default_tz = default_timezone

    def normalize(
        self,
        timestamp: Union[str, datetime, int, float],
        source_timezone: Optional[str] = None
    ) -> datetime:
        """
        Normalize timestamp to datetime.

        Args:
            timestamp: Input timestamp (string, datetime, or unix epoch)
            source_timezone: Source timezone for naive timestamps

        Returns:
            Normalized datetime
        """
        # Handle different input types
        if isinstance(timestamp, datetime):
            dt = timestamp
        elif isinstance(timestamp, (int, float)):
            # Unix epoch (seconds or milliseconds)
            if timestamp > 1e11:  # Milliseconds
                timestamp = timestamp / 1000
            dt = datetime.utcfromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            dt = self._parse_string(timestamp)
        else:
            raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")

        # Handle timezone
        return self._apply_timezone_handling(dt, source_timezone)

    def _parse_string(self, timestamp_str: str) -> datetime:
        """Parse timestamp string."""
        # Try ISO format first
        try:
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except ValueError:
            pass

        # Try common formats
        for fmt in self.COMMON_FORMATS:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue

        raise ValueError(f"Could not parse timestamp: {timestamp_str}")

    def _apply_timezone_handling(
        self,
        dt: datetime,
        source_timezone: Optional[str]
    ) -> datetime:
        """Apply timezone handling strategy."""
        if self._handling == TimeZoneHandling.PRESERVE:
            return dt

        elif self._handling == TimeZoneHandling.CONVERT_TO_UTC:
            if dt.tzinfo is None:
                # Assume UTC for naive timestamps
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        elif self._handling == TimeZoneHandling.STRIP_TIMEZONE:
            return dt.replace(tzinfo=None)

        return dt

    def align_timestamps(
        self,
        timestamps: List[datetime],
        interval_seconds: int,
        method: str = "floor"
    ) -> List[datetime]:
        """
        Align timestamps to regular intervals.

        Args:
            timestamps: List of timestamps
            interval_seconds: Alignment interval
            method: Alignment method (floor, ceil, round)

        Returns:
            Aligned timestamps
        """
        aligned = []
        for ts in timestamps:
            epoch = ts.timestamp()

            if method == "floor":
                aligned_epoch = (epoch // interval_seconds) * interval_seconds
            elif method == "ceil":
                aligned_epoch = math.ceil(epoch / interval_seconds) * interval_seconds
            else:  # round
                aligned_epoch = round(epoch / interval_seconds) * interval_seconds

            aligned.append(datetime.fromtimestamp(aligned_epoch, tz=ts.tzinfo))

        return aligned


# =============================================================================
# Schema Transformation
# =============================================================================


@dataclass
class FieldMapping:
    """Mapping between source and target fields."""

    source_field: str
    target_field: str
    transform: Optional[Callable[[Any], Any]] = None
    default_value: Any = None
    required: bool = False
    unit_from: Optional[str] = None
    unit_to: Optional[str] = None


class SchemaMapper:
    """Maps data between different schemas (vendor to canonical)."""

    def __init__(
        self,
        mappings: List[FieldMapping],
        unit_converter: Optional[UnitConverter] = None,
        strict: bool = False
    ) -> None:
        """
        Initialize schema mapper.

        Args:
            mappings: List of field mappings
            unit_converter: Unit converter for unit transformations
            strict: Raise errors on missing required fields
        """
        self._mappings = {m.source_field: m for m in mappings}
        self._unit_converter = unit_converter or UnitConverter()
        self._strict = strict

    def map_record(
        self,
        source_data: Dict[str, Any],
        include_unmapped: bool = False
    ) -> Dict[str, Any]:
        """
        Map a single record from source to target schema.

        Args:
            source_data: Source record
            include_unmapped: Include unmapped source fields

        Returns:
            Mapped record
        """
        result = {}
        used_source_fields = set()

        for source_field, mapping in self._mappings.items():
            value = self._get_nested_value(source_data, source_field)
            used_source_fields.add(source_field.split(".")[0])

            if value is None:
                if mapping.required and self._strict:
                    raise ValueError(f"Required field '{source_field}' is missing")
                value = mapping.default_value

            if value is not None:
                # Apply transform
                if mapping.transform:
                    try:
                        value = mapping.transform(value)
                    except Exception as e:
                        logger.warning(
                            f"Transform failed for {source_field}: {e}"
                        )
                        value = mapping.default_value

                # Apply unit conversion
                if mapping.unit_from and mapping.unit_to and isinstance(value, (int, float)):
                    try:
                        value = self._unit_converter.convert(
                            value,
                            mapping.unit_from,
                            mapping.unit_to
                        )
                    except Exception as e:
                        logger.warning(
                            f"Unit conversion failed for {source_field}: {e}"
                        )

            self._set_nested_value(result, mapping.target_field, value)

        # Include unmapped fields if requested
        if include_unmapped:
            for key, value in source_data.items():
                if key not in used_source_fields:
                    result[f"_unmapped_{key}"] = value

        return result

    def map_records(
        self,
        source_data: List[Dict[str, Any]],
        include_unmapped: bool = False
    ) -> List[Dict[str, Any]]:
        """Map multiple records."""
        return [self.map_record(record, include_unmapped) for record in source_data]

    def _get_nested_value(
        self,
        data: Dict[str, Any],
        path: str
    ) -> Any:
        """Get value from nested dict using dot notation."""
        parts = path.split(".")
        value = data

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None

        return value

    def _set_nested_value(
        self,
        data: Dict[str, Any],
        path: str,
        value: Any
    ) -> None:
        """Set value in nested dict using dot notation."""
        parts = path.split(".")
        current = data

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value


# =============================================================================
# Data Quality Assessment
# =============================================================================


@dataclass
class QualityMetric:
    """Quality metric result."""

    dimension: DataQualityDimension
    score: float  # 0.0 to 1.0
    issues: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class DataQualityScorer:
    """Assesses data quality for heat exchanger data."""

    def __init__(
        self,
        required_fields: Optional[List[str]] = None,
        numeric_fields: Optional[List[str]] = None,
        range_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        max_age_seconds: Optional[int] = None
    ) -> None:
        """
        Initialize data quality scorer.

        Args:
            required_fields: Fields that must be present and non-null
            numeric_fields: Fields that must be numeric
            range_constraints: Valid ranges for fields (field: (min, max))
            max_age_seconds: Maximum age for timestamps
        """
        self._required_fields = required_fields or []
        self._numeric_fields = numeric_fields or []
        self._range_constraints = range_constraints or {}
        self._max_age = max_age_seconds

        # Heat exchanger specific ranges
        self._default_ranges = {
            "temperature": (-50.0, 500.0),  # Celsius
            "pressure": (0.0, 50000.0),  # kPa
            "flow_rate": (0.0, 10000.0),  # kg/s
            "duty": (0.0, 100000.0),  # kW
            "ua": (0.0, 10000000.0),  # W/K
            "fouling_factor": (0.0, 0.01),  # m2K/W
            "effectiveness": (0.0, 1.0),
        }

    def score_record(
        self,
        record: Dict[str, Any],
        timestamp_field: Optional[str] = None
    ) -> Tuple[float, List[QualityMetric]]:
        """
        Score a single record's quality.

        Args:
            record: Data record to score
            timestamp_field: Field containing timestamp

        Returns:
            Tuple of (overall_score, list of quality metrics)
        """
        metrics = []

        # Completeness
        completeness = self._score_completeness(record)
        metrics.append(completeness)

        # Validity (numeric fields are numeric)
        validity = self._score_validity(record)
        metrics.append(validity)

        # Range conformity
        range_score = self._score_range_conformity(record)
        metrics.append(range_score)

        # Timeliness
        if timestamp_field:
            timeliness = self._score_timeliness(record, timestamp_field)
            metrics.append(timeliness)

        # Calculate overall score (weighted average)
        weights = {
            DataQualityDimension.COMPLETENESS: 0.3,
            DataQualityDimension.VALIDITY: 0.3,
            DataQualityDimension.RANGE_CONFORMITY: 0.25,
            DataQualityDimension.TIMELINESS: 0.15,
        }

        total_weight = 0.0
        weighted_score = 0.0

        for metric in metrics:
            weight = weights.get(metric.dimension, 0.1)
            weighted_score += metric.score * weight
            total_weight += weight

        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0

        return overall_score, metrics

    def _score_completeness(self, record: Dict[str, Any]) -> QualityMetric:
        """Score completeness of required fields."""
        issues = []
        if not self._required_fields:
            return QualityMetric(
                dimension=DataQualityDimension.COMPLETENESS,
                score=1.0
            )

        present = 0
        for field in self._required_fields:
            value = record.get(field)
            if value is not None and value != "":
                present += 1
            else:
                issues.append(f"Missing required field: {field}")

        score = present / len(self._required_fields) if self._required_fields else 1.0

        return QualityMetric(
            dimension=DataQualityDimension.COMPLETENESS,
            score=score,
            issues=issues,
            details={
                "present": present,
                "required": len(self._required_fields)
            }
        )

    def _score_validity(self, record: Dict[str, Any]) -> QualityMetric:
        """Score validity of numeric fields."""
        issues = []
        if not self._numeric_fields:
            return QualityMetric(
                dimension=DataQualityDimension.VALIDITY,
                score=1.0
            )

        valid = 0
        checked = 0

        for field in self._numeric_fields:
            value = record.get(field)
            if value is None:
                continue

            checked += 1
            if isinstance(value, (int, float)) and not math.isnan(value) and not math.isinf(value):
                valid += 1
            else:
                issues.append(f"Invalid numeric value for {field}: {value}")

        score = valid / checked if checked > 0 else 1.0

        return QualityMetric(
            dimension=DataQualityDimension.VALIDITY,
            score=score,
            issues=issues,
            details={"valid": valid, "checked": checked}
        )

    def _score_range_conformity(self, record: Dict[str, Any]) -> QualityMetric:
        """Score conformity to expected ranges."""
        issues = []
        in_range = 0
        checked = 0

        for field, (min_val, max_val) in self._range_constraints.items():
            value = record.get(field)
            if value is None or not isinstance(value, (int, float)):
                continue

            checked += 1
            if min_val <= value <= max_val:
                in_range += 1
            else:
                issues.append(
                    f"Value out of range for {field}: {value} "
                    f"(expected {min_val} to {max_val})"
                )

        score = in_range / checked if checked > 0 else 1.0

        return QualityMetric(
            dimension=DataQualityDimension.RANGE_CONFORMITY,
            score=score,
            issues=issues,
            details={"in_range": in_range, "checked": checked}
        )

    def _score_timeliness(
        self,
        record: Dict[str, Any],
        timestamp_field: str
    ) -> QualityMetric:
        """Score timeliness of data."""
        issues = []

        timestamp = record.get(timestamp_field)
        if timestamp is None:
            return QualityMetric(
                dimension=DataQualityDimension.TIMELINESS,
                score=0.0,
                issues=["Missing timestamp"]
            )

        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                return QualityMetric(
                    dimension=DataQualityDimension.TIMELINESS,
                    score=0.5,
                    issues=["Could not parse timestamp"]
                )

        if not isinstance(timestamp, datetime):
            return QualityMetric(
                dimension=DataQualityDimension.TIMELINESS,
                score=0.5,
                issues=["Invalid timestamp type"]
            )

        # Check age
        now = datetime.utcnow()
        if timestamp.tzinfo:
            now = now.replace(tzinfo=timezone.utc)

        try:
            age_seconds = (now - timestamp).total_seconds()
        except TypeError:
            age_seconds = 0

        if self._max_age and age_seconds > self._max_age:
            score = max(0.0, 1.0 - (age_seconds - self._max_age) / self._max_age)
            issues.append(f"Data is {age_seconds/3600:.1f} hours old")
        else:
            score = 1.0

        return QualityMetric(
            dimension=DataQualityDimension.TIMELINESS,
            score=score,
            issues=issues,
            details={"age_seconds": age_seconds}
        )


# =============================================================================
# Missing Data Handler
# =============================================================================


class MissingDataHandler:
    """Handles missing data in time series."""

    def __init__(
        self,
        strategy: MissingDataStrategy = MissingDataStrategy.INTERPOLATE_LINEAR,
        default_values: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize missing data handler.

        Args:
            strategy: Default handling strategy
            default_values: Default values for specific fields
        """
        self._strategy = strategy
        self._default_values = default_values or {}

    def handle(
        self,
        values: List[Optional[float]],
        timestamps: Optional[List[datetime]] = None,
        field_name: Optional[str] = None,
        strategy: Optional[MissingDataStrategy] = None
    ) -> List[Optional[float]]:
        """
        Handle missing values in a series.

        Args:
            values: List of values (None for missing)
            timestamps: Optional timestamps for interpolation
            field_name: Field name for default lookup
            strategy: Override default strategy

        Returns:
            Values with missing data handled
        """
        strategy = strategy or self._strategy

        if strategy == MissingDataStrategy.DROP:
            return [v for v in values if v is not None]

        elif strategy == MissingDataStrategy.FILL_NULL:
            return values

        elif strategy == MissingDataStrategy.FILL_ZERO:
            return [v if v is not None else 0.0 for v in values]

        elif strategy == MissingDataStrategy.FILL_DEFAULT:
            default = self._default_values.get(field_name, 0.0)
            return [v if v is not None else default for v in values]

        elif strategy == MissingDataStrategy.FILL_MEAN:
            valid_values = [v for v in values if v is not None]
            if not valid_values:
                return values
            mean = statistics.mean(valid_values)
            return [v if v is not None else mean for v in values]

        elif strategy == MissingDataStrategy.FILL_MEDIAN:
            valid_values = [v for v in values if v is not None]
            if not valid_values:
                return values
            median = statistics.median(valid_values)
            return [v if v is not None else median for v in values]

        elif strategy == MissingDataStrategy.FILL_FORWARD:
            result = []
            last_valid = None
            for v in values:
                if v is not None:
                    last_valid = v
                result.append(last_valid)
            return result

        elif strategy == MissingDataStrategy.FILL_BACKWARD:
            result = []
            next_valid = None
            for v in reversed(values):
                if v is not None:
                    next_valid = v
                result.append(next_valid)
            return list(reversed(result))

        elif strategy == MissingDataStrategy.INTERPOLATE_LINEAR:
            return self._interpolate_linear(values, timestamps)

        return values

    def _interpolate_linear(
        self,
        values: List[Optional[float]],
        timestamps: Optional[List[datetime]] = None
    ) -> List[Optional[float]]:
        """Linear interpolation for missing values."""
        result = list(values)
        n = len(values)

        # Find and interpolate gaps
        i = 0
        while i < n:
            if values[i] is None:
                # Find start of gap
                gap_start = i

                # Find end of gap
                gap_end = i
                while gap_end < n and values[gap_end] is None:
                    gap_end += 1

                # Interpolate if we have bounds
                if gap_start > 0 and gap_end < n:
                    v_start = values[gap_start - 1]
                    v_end = values[gap_end]

                    if timestamps:
                        t_start = timestamps[gap_start - 1].timestamp()
                        t_end = timestamps[gap_end].timestamp()

                        for j in range(gap_start, gap_end):
                            t = timestamps[j].timestamp()
                            ratio = (t - t_start) / (t_end - t_start)
                            result[j] = v_start + ratio * (v_end - v_start)
                    else:
                        # Index-based interpolation
                        for j in range(gap_start, gap_end):
                            ratio = (j - gap_start + 1) / (gap_end - gap_start + 1)
                            result[j] = v_start + ratio * (v_end - v_start)

                i = gap_end
            else:
                i += 1

        return result


# =============================================================================
# Outlier Detection
# =============================================================================


class OutlierDetector:
    """Detects outliers in time series data."""

    def __init__(
        self,
        method: OutlierMethod = OutlierMethod.IQR,
        threshold: float = 3.0,
        bounds: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Initialize outlier detector.

        Args:
            method: Detection method
            threshold: Threshold for z-score methods
            bounds: (min, max) bounds for BOUNDS method
        """
        self._method = method
        self._threshold = threshold
        self._bounds = bounds

    def detect(
        self,
        values: List[float],
        timestamps: Optional[List[datetime]] = None
    ) -> List[bool]:
        """
        Detect outliers in a series.

        Args:
            values: List of values
            timestamps: Optional timestamps for rate-of-change

        Returns:
            List of booleans (True = outlier)
        """
        if len(values) < 3:
            return [False] * len(values)

        if self._method == OutlierMethod.Z_SCORE:
            return self._z_score_detect(values)

        elif self._method == OutlierMethod.MODIFIED_Z_SCORE:
            return self._modified_z_score_detect(values)

        elif self._method == OutlierMethod.IQR:
            return self._iqr_detect(values)

        elif self._method == OutlierMethod.BOUNDS:
            return self._bounds_detect(values)

        elif self._method == OutlierMethod.RATE_OF_CHANGE:
            return self._rate_of_change_detect(values, timestamps)

        return [False] * len(values)

    def _z_score_detect(self, values: List[float]) -> List[bool]:
        """Detect outliers using z-score."""
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 1.0

        if stdev == 0:
            return [False] * len(values)

        return [abs((v - mean) / stdev) > self._threshold for v in values]

    def _modified_z_score_detect(self, values: List[float]) -> List[bool]:
        """Detect outliers using modified z-score (MAD-based)."""
        median = statistics.median(values)
        deviations = [abs(v - median) for v in values]
        mad = statistics.median(deviations)

        if mad == 0:
            return [False] * len(values)

        modified_z = [0.6745 * (v - median) / mad for v in values]
        return [abs(z) > self._threshold for z in modified_z]

    def _iqr_detect(self, values: List[float]) -> List[bool]:
        """Detect outliers using IQR method."""
        sorted_values = sorted(values)
        n = len(sorted_values)

        q1_idx = n // 4
        q3_idx = 3 * n // 4

        q1 = sorted_values[q1_idx]
        q3 = sorted_values[q3_idx]
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return [v < lower_bound or v > upper_bound for v in values]

    def _bounds_detect(self, values: List[float]) -> List[bool]:
        """Detect outliers using fixed bounds."""
        if not self._bounds:
            return [False] * len(values)

        min_val, max_val = self._bounds
        return [v < min_val or v > max_val for v in values]

    def _rate_of_change_detect(
        self,
        values: List[float],
        timestamps: Optional[List[datetime]] = None
    ) -> List[bool]:
        """Detect outliers based on rate of change."""
        if len(values) < 2:
            return [False] * len(values)

        # Calculate rates of change
        rates = []
        for i in range(1, len(values)):
            if timestamps:
                dt = (timestamps[i] - timestamps[i-1]).total_seconds()
                if dt > 0:
                    rates.append(abs(values[i] - values[i-1]) / dt)
                else:
                    rates.append(0)
            else:
                rates.append(abs(values[i] - values[i-1]))

        if not rates:
            return [False] * len(values)

        # Use z-score on rates
        mean_rate = statistics.mean(rates)
        stdev_rate = statistics.stdev(rates) if len(rates) > 1 else 1.0

        if stdev_rate == 0:
            return [False] * len(values)

        outliers = [False]  # First point can't be rate outlier
        for rate in rates:
            z = abs((rate - mean_rate) / stdev_rate)
            outliers.append(z > self._threshold)

        return outliers


# =============================================================================
# Comprehensive Data Transformer
# =============================================================================


class DataTransformer:
    """
    Comprehensive data transformer for heat exchanger data.

    Combines all transformation utilities into a single interface.
    """

    def __init__(
        self,
        schema_mappings: Optional[List[FieldMapping]] = None,
        required_fields: Optional[List[str]] = None,
        numeric_fields: Optional[List[str]] = None,
        range_constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        target_units: Optional[Dict[str, str]] = None,
        missing_data_strategy: MissingDataStrategy = MissingDataStrategy.INTERPOLATE_LINEAR,
        outlier_method: OutlierMethod = OutlierMethod.IQR,
        quality_threshold: float = 0.6
    ) -> None:
        """
        Initialize data transformer.

        Args:
            schema_mappings: Field mappings for schema transformation
            required_fields: Required fields for quality assessment
            numeric_fields: Numeric fields for validation
            range_constraints: Valid ranges for fields
            target_units: Target units for conversion (field: unit)
            missing_data_strategy: Strategy for missing data
            outlier_method: Method for outlier detection
            quality_threshold: Minimum quality score threshold
        """
        self._unit_converter = UnitConverter()
        self._timestamp_normalizer = TimestampNormalizer()

        self._schema_mapper = SchemaMapper(
            mappings=schema_mappings or [],
            unit_converter=self._unit_converter
        ) if schema_mappings else None

        self._quality_scorer = DataQualityScorer(
            required_fields=required_fields,
            numeric_fields=numeric_fields,
            range_constraints=range_constraints
        )

        self._missing_handler = MissingDataHandler(strategy=missing_data_strategy)
        self._outlier_detector = OutlierDetector(method=outlier_method)

        self._target_units = target_units or {}
        self._quality_threshold = quality_threshold

    def transform(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        validate: bool = True,
        handle_outliers: bool = True
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Transform data through the complete pipeline.

        Args:
            data: Input data (single record or list)
            validate: Perform quality validation
            handle_outliers: Detect and flag outliers

        Returns:
            Transformed data with quality metadata
        """
        if isinstance(data, list):
            return [self._transform_record(r, validate, handle_outliers) for r in data]
        return self._transform_record(data, validate, handle_outliers)

    def _transform_record(
        self,
        record: Dict[str, Any],
        validate: bool,
        handle_outliers: bool
    ) -> Dict[str, Any]:
        """Transform a single record."""
        result = record.copy()

        # Apply schema mapping
        if self._schema_mapper:
            result = self._schema_mapper.map_record(result)

        # Apply unit conversions
        for field, target_unit in self._target_units.items():
            if field in result and isinstance(result[field], (int, float)):
                # Would need source unit from metadata
                pass

        # Validate quality
        if validate:
            quality_score, metrics = self._quality_scorer.score_record(result)
            result["_quality_score"] = quality_score
            result["_quality_passed"] = quality_score >= self._quality_threshold
            result["_quality_issues"] = [
                issue
                for metric in metrics
                for issue in metric.issues
            ]

        return result

    @property
    def unit_converter(self) -> UnitConverter:
        """Get unit converter."""
        return self._unit_converter

    @property
    def timestamp_normalizer(self) -> TimestampNormalizer:
        """Get timestamp normalizer."""
        return self._timestamp_normalizer

    @property
    def quality_scorer(self) -> DataQualityScorer:
        """Get quality scorer."""
        return self._quality_scorer
