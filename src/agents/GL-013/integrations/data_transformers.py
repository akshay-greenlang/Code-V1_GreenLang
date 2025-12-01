"""
Data Transformers Module for GL-013 PREDICTMAINT (Predictive Maintenance Agent).

Provides data transformation utilities including unit conversions, timestamp
normalization, schema mapping between systems, data quality scoring,
missing data handling, and outlier detection.

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
    """Categories of measurement units."""

    LENGTH = "length"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"
    FREQUENCY = "frequency"
    FORCE = "force"
    POWER = "power"
    ENERGY = "energy"
    FLOW_RATE = "flow_rate"
    VOLUME = "volume"
    MASS = "mass"
    TIME = "time"
    ELECTRIC_CURRENT = "electric_current"
    VOLTAGE = "voltage"
    RESISTANCE = "resistance"
    VIBRATION = "vibration"
    ANGLE = "angle"
    ANGULAR_VELOCITY = "angular_velocity"


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
    FILL_MODE = "fill_mode"
    FILL_FORWARD = "fill_forward"
    FILL_BACKWARD = "fill_backward"
    INTERPOLATE_LINEAR = "interpolate_linear"
    INTERPOLATE_SPLINE = "interpolate_spline"


class OutlierMethod(str, Enum):
    """Methods for outlier detection."""

    Z_SCORE = "z_score"
    MODIFIED_Z_SCORE = "modified_z_score"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    LOF = "local_outlier_factor"
    DBSCAN = "dbscan"


class DataQualityDimension(str, Enum):
    """Data quality dimensions."""

    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"


# =============================================================================
# Unit Conversion
# =============================================================================


@dataclass
class UnitDefinition:
    """Definition of a measurement unit."""

    symbol: str
    name: str
    category: UnitCategory
    base_unit: str  # The SI base unit for this category
    to_base_factor: float  # Multiply by this to convert to base
    to_base_offset: float = 0.0  # Add this after multiplying (for temperature)


# Unit conversion tables
UNIT_DEFINITIONS: Dict[str, UnitDefinition] = {
    # Length
    "m": UnitDefinition("m", "meter", UnitCategory.LENGTH, "m", 1.0),
    "mm": UnitDefinition("mm", "millimeter", UnitCategory.LENGTH, "m", 0.001),
    "cm": UnitDefinition("cm", "centimeter", UnitCategory.LENGTH, "m", 0.01),
    "km": UnitDefinition("km", "kilometer", UnitCategory.LENGTH, "m", 1000.0),
    "in": UnitDefinition("in", "inch", UnitCategory.LENGTH, "m", 0.0254),
    "ft": UnitDefinition("ft", "foot", UnitCategory.LENGTH, "m", 0.3048),
    "yd": UnitDefinition("yd", "yard", UnitCategory.LENGTH, "m", 0.9144),
    "mi": UnitDefinition("mi", "mile", UnitCategory.LENGTH, "m", 1609.344),
    "mil": UnitDefinition("mil", "mil", UnitCategory.LENGTH, "m", 0.0000254),
    "um": UnitDefinition("um", "micrometer", UnitCategory.LENGTH, "m", 0.000001),

    # Temperature
    "K": UnitDefinition("K", "kelvin", UnitCategory.TEMPERATURE, "K", 1.0),
    "C": UnitDefinition("C", "celsius", UnitCategory.TEMPERATURE, "K", 1.0, 273.15),
    "F": UnitDefinition("F", "fahrenheit", UnitCategory.TEMPERATURE, "K", 5/9, 255.372),

    # Pressure
    "Pa": UnitDefinition("Pa", "pascal", UnitCategory.PRESSURE, "Pa", 1.0),
    "kPa": UnitDefinition("kPa", "kilopascal", UnitCategory.PRESSURE, "Pa", 1000.0),
    "MPa": UnitDefinition("MPa", "megapascal", UnitCategory.PRESSURE, "Pa", 1000000.0),
    "bar": UnitDefinition("bar", "bar", UnitCategory.PRESSURE, "Pa", 100000.0),
    "mbar": UnitDefinition("mbar", "millibar", UnitCategory.PRESSURE, "Pa", 100.0),
    "psi": UnitDefinition("psi", "pounds per square inch", UnitCategory.PRESSURE, "Pa", 6894.757),
    "atm": UnitDefinition("atm", "atmosphere", UnitCategory.PRESSURE, "Pa", 101325.0),

    # Velocity (vibration)
    "m/s": UnitDefinition("m/s", "meters per second", UnitCategory.VELOCITY, "m/s", 1.0),
    "mm/s": UnitDefinition("mm/s", "millimeters per second", UnitCategory.VELOCITY, "m/s", 0.001),
    "in/s": UnitDefinition("in/s", "inches per second", UnitCategory.VELOCITY, "m/s", 0.0254),
    "ft/s": UnitDefinition("ft/s", "feet per second", UnitCategory.VELOCITY, "m/s", 0.3048),
    "km/h": UnitDefinition("km/h", "kilometers per hour", UnitCategory.VELOCITY, "m/s", 0.277778),
    "mph": UnitDefinition("mph", "miles per hour", UnitCategory.VELOCITY, "m/s", 0.44704),

    # Acceleration
    "m/s2": UnitDefinition("m/s2", "meters per second squared", UnitCategory.ACCELERATION, "m/s2", 1.0),
    "g": UnitDefinition("g", "gravitational acceleration", UnitCategory.ACCELERATION, "m/s2", 9.80665),
    "mg": UnitDefinition("mg", "milli-g", UnitCategory.ACCELERATION, "m/s2", 0.00980665),
    "ft/s2": UnitDefinition("ft/s2", "feet per second squared", UnitCategory.ACCELERATION, "m/s2", 0.3048),

    # Frequency
    "Hz": UnitDefinition("Hz", "hertz", UnitCategory.FREQUENCY, "Hz", 1.0),
    "kHz": UnitDefinition("kHz", "kilohertz", UnitCategory.FREQUENCY, "Hz", 1000.0),
    "MHz": UnitDefinition("MHz", "megahertz", UnitCategory.FREQUENCY, "Hz", 1000000.0),
    "rpm": UnitDefinition("rpm", "revolutions per minute", UnitCategory.FREQUENCY, "Hz", 1/60),
    "cpm": UnitDefinition("cpm", "cycles per minute", UnitCategory.FREQUENCY, "Hz", 1/60),

    # Force
    "N": UnitDefinition("N", "newton", UnitCategory.FORCE, "N", 1.0),
    "kN": UnitDefinition("kN", "kilonewton", UnitCategory.FORCE, "N", 1000.0),
    "lbf": UnitDefinition("lbf", "pound-force", UnitCategory.FORCE, "N", 4.44822),
    "kgf": UnitDefinition("kgf", "kilogram-force", UnitCategory.FORCE, "N", 9.80665),

    # Power
    "W": UnitDefinition("W", "watt", UnitCategory.POWER, "W", 1.0),
    "kW": UnitDefinition("kW", "kilowatt", UnitCategory.POWER, "W", 1000.0),
    "MW": UnitDefinition("MW", "megawatt", UnitCategory.POWER, "W", 1000000.0),
    "hp": UnitDefinition("hp", "horsepower", UnitCategory.POWER, "W", 745.7),

    # Energy
    "J": UnitDefinition("J", "joule", UnitCategory.ENERGY, "J", 1.0),
    "kJ": UnitDefinition("kJ", "kilojoule", UnitCategory.ENERGY, "J", 1000.0),
    "MJ": UnitDefinition("MJ", "megajoule", UnitCategory.ENERGY, "J", 1000000.0),
    "Wh": UnitDefinition("Wh", "watt-hour", UnitCategory.ENERGY, "J", 3600.0),
    "kWh": UnitDefinition("kWh", "kilowatt-hour", UnitCategory.ENERGY, "J", 3600000.0),
    "cal": UnitDefinition("cal", "calorie", UnitCategory.ENERGY, "J", 4.184),
    "BTU": UnitDefinition("BTU", "British thermal unit", UnitCategory.ENERGY, "J", 1055.06),

    # Flow rate (volume)
    "m3/s": UnitDefinition("m3/s", "cubic meters per second", UnitCategory.FLOW_RATE, "m3/s", 1.0),
    "L/s": UnitDefinition("L/s", "liters per second", UnitCategory.FLOW_RATE, "m3/s", 0.001),
    "L/min": UnitDefinition("L/min", "liters per minute", UnitCategory.FLOW_RATE, "m3/s", 0.001/60),
    "m3/h": UnitDefinition("m3/h", "cubic meters per hour", UnitCategory.FLOW_RATE, "m3/s", 1/3600),
    "gpm": UnitDefinition("gpm", "gallons per minute", UnitCategory.FLOW_RATE, "m3/s", 0.0000630902),

    # Volume
    "m3": UnitDefinition("m3", "cubic meter", UnitCategory.VOLUME, "m3", 1.0),
    "L": UnitDefinition("L", "liter", UnitCategory.VOLUME, "m3", 0.001),
    "mL": UnitDefinition("mL", "milliliter", UnitCategory.VOLUME, "m3", 0.000001),
    "gal": UnitDefinition("gal", "gallon", UnitCategory.VOLUME, "m3", 0.00378541),

    # Mass
    "kg": UnitDefinition("kg", "kilogram", UnitCategory.MASS, "kg", 1.0),
    "g": UnitDefinition("g", "gram", UnitCategory.MASS, "kg", 0.001),
    "mg": UnitDefinition("mg", "milligram", UnitCategory.MASS, "kg", 0.000001),
    "t": UnitDefinition("t", "metric ton", UnitCategory.MASS, "kg", 1000.0),
    "lb": UnitDefinition("lb", "pound", UnitCategory.MASS, "kg", 0.453592),
    "oz": UnitDefinition("oz", "ounce", UnitCategory.MASS, "kg", 0.0283495),

    # Time
    "s": UnitDefinition("s", "second", UnitCategory.TIME, "s", 1.0),
    "ms": UnitDefinition("ms", "millisecond", UnitCategory.TIME, "s", 0.001),
    "us": UnitDefinition("us", "microsecond", UnitCategory.TIME, "s", 0.000001),
    "min": UnitDefinition("min", "minute", UnitCategory.TIME, "s", 60.0),
    "h": UnitDefinition("h", "hour", UnitCategory.TIME, "s", 3600.0),
    "d": UnitDefinition("d", "day", UnitCategory.TIME, "s", 86400.0),

    # Electric current
    "A": UnitDefinition("A", "ampere", UnitCategory.ELECTRIC_CURRENT, "A", 1.0),
    "mA": UnitDefinition("mA", "milliampere", UnitCategory.ELECTRIC_CURRENT, "A", 0.001),
    "uA": UnitDefinition("uA", "microampere", UnitCategory.ELECTRIC_CURRENT, "A", 0.000001),
    "kA": UnitDefinition("kA", "kiloampere", UnitCategory.ELECTRIC_CURRENT, "A", 1000.0),

    # Voltage
    "V": UnitDefinition("V", "volt", UnitCategory.VOLTAGE, "V", 1.0),
    "mV": UnitDefinition("mV", "millivolt", UnitCategory.VOLTAGE, "V", 0.001),
    "kV": UnitDefinition("kV", "kilovolt", UnitCategory.VOLTAGE, "V", 1000.0),

    # Angle
    "rad": UnitDefinition("rad", "radian", UnitCategory.ANGLE, "rad", 1.0),
    "deg": UnitDefinition("deg", "degree", UnitCategory.ANGLE, "rad", math.pi/180),
    "mrad": UnitDefinition("mrad", "milliradian", UnitCategory.ANGLE, "rad", 0.001),

    # Angular velocity
    "rad/s": UnitDefinition("rad/s", "radians per second", UnitCategory.ANGULAR_VELOCITY, "rad/s", 1.0),
    "deg/s": UnitDefinition("deg/s", "degrees per second", UnitCategory.ANGULAR_VELOCITY, "rad/s", math.pi/180),
}


class UnitConverter:
    """Handles unit conversions between different measurement systems."""

    def __init__(self) -> None:
        """Initialize unit converter."""
        self._definitions = UNIT_DEFINITIONS.copy()
        self._aliases: Dict[str, str] = {
            # Common aliases
            "celsius": "C",
            "fahrenheit": "F",
            "kelvin": "K",
            "meters": "m",
            "millimeters": "mm",
            "inches": "in",
            "feet": "ft",
            "seconds": "s",
            "minutes": "min",
            "hours": "h",
            "hertz": "Hz",
            "pascal": "Pa",
            "bar": "bar",
            "newton": "N",
            "watt": "W",
            "joule": "J",
            "ampere": "A",
            "volt": "V",
            "mm/sec": "mm/s",
            "in/sec": "in/s",
            "ips": "in/s",
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
        lower_unit = unit.lower()
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

    def get_conversion_factor(self, from_unit: str, to_unit: str) -> float:
        """
        Get conversion factor between units.

        Note: This only works for units without offset (not temperature).
        """
        from_normalized = self.normalize_unit(from_unit)
        to_normalized = self.normalize_unit(to_unit)

        if from_normalized not in self._definitions:
            raise ValueError(f"Unknown unit: {from_unit}")
        if to_normalized not in self._definitions:
            raise ValueError(f"Unknown unit: {to_unit}")

        from_def = self._definitions[from_normalized]
        to_def = self._definitions[to_normalized]

        if from_def.to_base_offset != 0 or to_def.to_base_offset != 0:
            raise ValueError("Cannot get simple factor for units with offset")

        return from_def.to_base_factor / to_def.to_base_factor


# =============================================================================
# Timestamp Normalization
# =============================================================================


class TimestampNormalizer:
    """Normalizes timestamps to a consistent format."""

    # Common timestamp patterns
    PATTERNS = [
        # ISO 8601
        r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?(?:Z|([+-]\d{2}):?(\d{2}))?",
        # Date with time
        r"(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?",
        # US format
        r"(\d{1,2})/(\d{1,2})/(\d{4}) (\d{1,2}):(\d{2}):?(\d{2})?(?:\s*(AM|PM))?",
        # European format
        r"(\d{1,2})\.(\d{1,2})\.(\d{4}) (\d{2}):(\d{2}):?(\d{2})?",
        # Unix timestamp (seconds)
        r"^(\d{10})$",
        # Unix timestamp (milliseconds)
        r"^(\d{13})$",
    ]

    def __init__(
        self,
        default_timezone: str = "UTC",
        handling: TimeZoneHandling = TimeZoneHandling.CONVERT_TO_UTC,
    ) -> None:
        """
        Initialize timestamp normalizer.

        Args:
            default_timezone: Default timezone for naive timestamps
            handling: Timezone handling strategy
        """
        self._default_tz_str = default_timezone
        self._handling = handling
        self._default_tz = timezone.utc  # Default to UTC

    def normalize(
        self,
        timestamp: Union[str, int, float, datetime],
        source_timezone: Optional[str] = None,
    ) -> datetime:
        """
        Normalize timestamp to standard format.

        Args:
            timestamp: Input timestamp in various formats
            source_timezone: Source timezone if known

        Returns:
            Normalized datetime object
        """
        if isinstance(timestamp, datetime):
            dt = timestamp
        elif isinstance(timestamp, (int, float)):
            dt = self._from_unix(timestamp)
        elif isinstance(timestamp, str):
            dt = self._parse_string(timestamp)
        else:
            raise ValueError(f"Cannot parse timestamp of type {type(timestamp)}")

        # Handle timezone
        if dt.tzinfo is None:
            # Naive datetime - apply default timezone
            dt = dt.replace(tzinfo=self._default_tz)

        if self._handling == TimeZoneHandling.CONVERT_TO_UTC:
            dt = dt.astimezone(timezone.utc)
        elif self._handling == TimeZoneHandling.STRIP_TIMEZONE:
            dt = dt.replace(tzinfo=None)

        return dt

    def _from_unix(self, timestamp: Union[int, float]) -> datetime:
        """Convert Unix timestamp to datetime."""
        # Detect if milliseconds
        if timestamp > 1e11:
            timestamp = timestamp / 1000

        return datetime.fromtimestamp(timestamp, tz=timezone.utc)

    def _parse_string(self, timestamp: str) -> datetime:
        """Parse string timestamp."""
        timestamp = timestamp.strip()

        # Try Unix timestamp
        if timestamp.isdigit():
            return self._from_unix(int(timestamp))

        # Try ISO format
        try:
            # Handle Z suffix
            if timestamp.endswith('Z'):
                timestamp = timestamp[:-1] + '+00:00'
            return datetime.fromisoformat(timestamp)
        except ValueError:
            pass

        # Try common formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y/%m/%d %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%d-%b-%Y %H:%M:%S",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(timestamp, fmt)
            except ValueError:
                continue

        raise ValueError(f"Cannot parse timestamp: {timestamp}")

    def format_iso(self, dt: datetime) -> str:
        """Format datetime as ISO 8601 string."""
        return dt.isoformat()

    def format_unix(self, dt: datetime) -> float:
        """Format datetime as Unix timestamp."""
        return dt.timestamp()

    def format_unix_ms(self, dt: datetime) -> int:
        """Format datetime as Unix timestamp in milliseconds."""
        return int(dt.timestamp() * 1000)


# =============================================================================
# Schema Mapping
# =============================================================================


@dataclass
class FieldMapping:
    """Mapping between source and target field."""

    source_field: str
    target_field: str
    transform: Optional[Callable[[Any], Any]] = None
    default_value: Any = None
    required: bool = False


class SchemaMapper:
    """Maps data between different schemas/systems."""

    def __init__(self) -> None:
        """Initialize schema mapper."""
        self._mappings: Dict[str, Dict[str, FieldMapping]] = {}
        self._transformers: Dict[str, Callable[[Any], Any]] = {}

        # Register default transformers
        self._register_default_transformers()

    def _register_default_transformers(self) -> None:
        """Register default field transformers."""
        self._transformers["string"] = str
        self._transformers["int"] = int
        self._transformers["float"] = float
        self._transformers["bool"] = bool
        self._transformers["lower"] = lambda x: str(x).lower()
        self._transformers["upper"] = lambda x: str(x).upper()
        self._transformers["strip"] = lambda x: str(x).strip()
        self._transformers["iso_date"] = lambda x: datetime.fromisoformat(str(x)).isoformat()

    def register_transformer(self, name: str, func: Callable[[Any], Any]) -> None:
        """Register a custom transformer function."""
        self._transformers[name] = func

    def register_mapping(
        self,
        mapping_name: str,
        source_field: str,
        target_field: str,
        transform: Optional[str] = None,
        default_value: Any = None,
        required: bool = False,
    ) -> None:
        """
        Register a field mapping.

        Args:
            mapping_name: Name of the mapping configuration
            source_field: Source field name (supports dot notation)
            target_field: Target field name (supports dot notation)
            transform: Optional transformer name
            default_value: Default value if source is missing
            required: Whether field is required
        """
        if mapping_name not in self._mappings:
            self._mappings[mapping_name] = {}

        transform_func = None
        if transform:
            if transform in self._transformers:
                transform_func = self._transformers[transform]
            else:
                logger.warning(f"Unknown transformer: {transform}")

        self._mappings[mapping_name][source_field] = FieldMapping(
            source_field=source_field,
            target_field=target_field,
            transform=transform_func,
            default_value=default_value,
            required=required,
        )

    def map(
        self,
        mapping_name: str,
        source_data: Dict[str, Any],
        strict: bool = False,
    ) -> Dict[str, Any]:
        """
        Map source data to target schema.

        Args:
            mapping_name: Name of mapping configuration
            source_data: Source data dictionary
            strict: Raise error on missing required fields

        Returns:
            Mapped data dictionary
        """
        if mapping_name not in self._mappings:
            raise ValueError(f"Unknown mapping: {mapping_name}")

        result = {}
        mappings = self._mappings[mapping_name]

        for source_field, mapping in mappings.items():
            value = self._get_nested_value(source_data, source_field)

            if value is None:
                if mapping.required and strict:
                    raise ValueError(f"Missing required field: {source_field}")
                value = mapping.default_value

            if value is not None and mapping.transform:
                try:
                    value = mapping.transform(value)
                except Exception as e:
                    logger.warning(f"Transform error for {source_field}: {e}")
                    value = mapping.default_value

            if value is not None:
                self._set_nested_value(result, mapping.target_field, value)

        return result

    def map_batch(
        self,
        mapping_name: str,
        source_data_list: List[Dict[str, Any]],
        strict: bool = False,
    ) -> List[Dict[str, Any]]:
        """Map a batch of source data."""
        return [self.map(mapping_name, data, strict) for data in source_data_list]

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get value from nested path (dot notation)."""
        keys = path.split('.')
        value = data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any) -> None:
        """Set value at nested path (dot notation)."""
        keys = path.split('.')
        current = data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value


# =============================================================================
# Data Quality Scoring
# =============================================================================


@dataclass
class QualityMetric:
    """Single quality metric."""

    dimension: DataQualityDimension
    score: float  # 0.0 to 1.0
    weight: float = 1.0
    issues: List[str] = field(default_factory=list)


class DataQualityScorer:
    """Scores data quality across multiple dimensions."""

    def __init__(
        self,
        required_fields: Optional[List[str]] = None,
        field_validators: Optional[Dict[str, Callable[[Any], bool]]] = None,
        field_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        """
        Initialize data quality scorer.

        Args:
            required_fields: List of required field names
            field_validators: Dict of field -> validation function
            field_ranges: Dict of field -> (min, max) valid range
        """
        self._required_fields = required_fields or []
        self._field_validators = field_validators or {}
        self._field_ranges = field_ranges or {}

        # Dimension weights
        self._weights = {
            DataQualityDimension.COMPLETENESS: 0.25,
            DataQualityDimension.VALIDITY: 0.30,
            DataQualityDimension.ACCURACY: 0.20,
            DataQualityDimension.CONSISTENCY: 0.15,
            DataQualityDimension.TIMELINESS: 0.10,
        }

    def set_weight(self, dimension: DataQualityDimension, weight: float) -> None:
        """Set weight for a quality dimension."""
        self._weights[dimension] = weight

    def score(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score data quality.

        Returns:
            Quality assessment with overall score and dimension breakdowns
        """
        metrics = []

        # Completeness
        completeness = self._score_completeness(data)
        metrics.append(completeness)

        # Validity
        validity = self._score_validity(data)
        metrics.append(validity)

        # Accuracy (range checks)
        accuracy = self._score_accuracy(data)
        metrics.append(accuracy)

        # Consistency
        consistency = self._score_consistency(data)
        metrics.append(consistency)

        # Timeliness
        timeliness = self._score_timeliness(data)
        metrics.append(timeliness)

        # Calculate weighted overall score
        total_weight = sum(self._weights.get(m.dimension, 1.0) for m in metrics)
        overall_score = sum(
            m.score * self._weights.get(m.dimension, 1.0)
            for m in metrics
        ) / total_weight if total_weight > 0 else 0.0

        all_issues = []
        for m in metrics:
            all_issues.extend(m.issues)

        return {
            "overall_score": overall_score,
            "dimensions": {
                m.dimension.value: {
                    "score": m.score,
                    "weight": self._weights.get(m.dimension, 1.0),
                    "issues": m.issues,
                }
                for m in metrics
            },
            "issues": all_issues,
            "recommendations": self._generate_recommendations(metrics),
        }

    def _score_completeness(self, data: Dict[str, Any]) -> QualityMetric:
        """Score data completeness."""
        issues = []

        if not self._required_fields:
            return QualityMetric(
                dimension=DataQualityDimension.COMPLETENESS,
                score=1.0,
            )

        present = 0
        for field in self._required_fields:
            value = self._get_nested(data, field)
            if value is not None and value != "":
                present += 1
            else:
                issues.append(f"Missing required field: {field}")

        score = present / len(self._required_fields)

        return QualityMetric(
            dimension=DataQualityDimension.COMPLETENESS,
            score=score,
            issues=issues,
        )

    def _score_validity(self, data: Dict[str, Any]) -> QualityMetric:
        """Score data validity."""
        issues = []

        if not self._field_validators:
            return QualityMetric(
                dimension=DataQualityDimension.VALIDITY,
                score=1.0,
            )

        valid = 0
        total = 0

        for field, validator in self._field_validators.items():
            value = self._get_nested(data, field)
            if value is not None:
                total += 1
                try:
                    if validator(value):
                        valid += 1
                    else:
                        issues.append(f"Invalid value for field: {field}")
                except Exception:
                    issues.append(f"Validation error for field: {field}")

        score = valid / total if total > 0 else 1.0

        return QualityMetric(
            dimension=DataQualityDimension.VALIDITY,
            score=score,
            issues=issues,
        )

    def _score_accuracy(self, data: Dict[str, Any]) -> QualityMetric:
        """Score data accuracy (range checks)."""
        issues = []

        if not self._field_ranges:
            return QualityMetric(
                dimension=DataQualityDimension.ACCURACY,
                score=1.0,
            )

        in_range = 0
        total = 0

        for field, (min_val, max_val) in self._field_ranges.items():
            value = self._get_nested(data, field)
            if value is not None:
                total += 1
                try:
                    num_val = float(value)
                    if min_val <= num_val <= max_val:
                        in_range += 1
                    else:
                        issues.append(f"Value out of range for field: {field}")
                except (ValueError, TypeError):
                    issues.append(f"Non-numeric value for range field: {field}")

        score = in_range / total if total > 0 else 1.0

        return QualityMetric(
            dimension=DataQualityDimension.ACCURACY,
            score=score,
            issues=issues,
        )

    def _score_consistency(self, data: Dict[str, Any]) -> QualityMetric:
        """Score data consistency."""
        issues = []
        score = 1.0

        # Check for type consistency in lists
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 1:
                types = set(type(v).__name__ for v in value if v is not None)
                if len(types) > 1:
                    issues.append(f"Inconsistent types in list: {key}")
                    score *= 0.9

        return QualityMetric(
            dimension=DataQualityDimension.CONSISTENCY,
            score=score,
            issues=issues,
        )

    def _score_timeliness(self, data: Dict[str, Any]) -> QualityMetric:
        """Score data timeliness."""
        issues = []
        score = 1.0

        # Look for timestamp fields
        timestamp_fields = ["timestamp", "created_at", "updated_at", "time", "date"]

        for field in timestamp_fields:
            value = self._get_nested(data, field)
            if value is not None:
                try:
                    if isinstance(value, datetime):
                        dt = value
                    else:
                        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))

                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)

                    age = datetime.now(timezone.utc) - dt

                    if age > timedelta(days=1):
                        score *= 0.8
                        issues.append(f"Data is more than 1 day old")
                    elif age > timedelta(hours=1):
                        score *= 0.95

                    break
                except Exception:
                    pass

        return QualityMetric(
            dimension=DataQualityDimension.TIMELINESS,
            score=score,
            issues=issues,
        )

    def _generate_recommendations(self, metrics: List[QualityMetric]) -> List[str]:
        """Generate recommendations based on quality issues."""
        recommendations = []

        for metric in metrics:
            if metric.score < 0.8:
                if metric.dimension == DataQualityDimension.COMPLETENESS:
                    recommendations.append("Ensure all required fields are populated")
                elif metric.dimension == DataQualityDimension.VALIDITY:
                    recommendations.append("Review and correct invalid field values")
                elif metric.dimension == DataQualityDimension.ACCURACY:
                    recommendations.append("Check values against expected ranges")
                elif metric.dimension == DataQualityDimension.CONSISTENCY:
                    recommendations.append("Standardize data formats and types")
                elif metric.dimension == DataQualityDimension.TIMELINESS:
                    recommendations.append("Update data more frequently")

        return recommendations

    def _get_nested(self, data: Dict[str, Any], path: str) -> Any:
        """Get value from nested path."""
        keys = path.split('.')
        value = data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value


# =============================================================================
# Missing Data Handler
# =============================================================================


class MissingDataHandler:
    """Handles missing data in datasets."""

    def __init__(
        self,
        default_strategy: MissingDataStrategy = MissingDataStrategy.FILL_NULL,
        field_strategies: Optional[Dict[str, MissingDataStrategy]] = None,
    ) -> None:
        """
        Initialize missing data handler.

        Args:
            default_strategy: Default handling strategy
            field_strategies: Per-field strategies
        """
        self._default_strategy = default_strategy
        self._field_strategies = field_strategies or {}

    def handle(
        self,
        data: List[Dict[str, Any]],
        fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Handle missing data in dataset.

        Args:
            data: List of data records
            fields: Fields to process (all if not specified)

        Returns:
            Processed data
        """
        if not data:
            return data

        # Determine fields to process
        if fields is None:
            fields = list(set().union(*[set(d.keys()) for d in data]))

        # Process each field
        result = [d.copy() for d in data]

        for field in fields:
            strategy = self._field_strategies.get(field, self._default_strategy)
            values = [d.get(field) for d in result]

            if strategy == MissingDataStrategy.DROP:
                result = [d for d in result if d.get(field) is not None]
            elif strategy == MissingDataStrategy.FILL_ZERO:
                for d in result:
                    if d.get(field) is None:
                        d[field] = 0
            elif strategy == MissingDataStrategy.FILL_MEAN:
                mean_val = self._calculate_mean(values)
                for d in result:
                    if d.get(field) is None:
                        d[field] = mean_val
            elif strategy == MissingDataStrategy.FILL_MEDIAN:
                median_val = self._calculate_median(values)
                for d in result:
                    if d.get(field) is None:
                        d[field] = median_val
            elif strategy == MissingDataStrategy.FILL_MODE:
                mode_val = self._calculate_mode(values)
                for d in result:
                    if d.get(field) is None:
                        d[field] = mode_val
            elif strategy == MissingDataStrategy.FILL_FORWARD:
                self._fill_forward(result, field)
            elif strategy == MissingDataStrategy.FILL_BACKWARD:
                self._fill_backward(result, field)
            elif strategy == MissingDataStrategy.INTERPOLATE_LINEAR:
                self._interpolate_linear(result, field)

        return result

    def _calculate_mean(self, values: List[Any]) -> Optional[float]:
        """Calculate mean of numeric values."""
        numeric = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
        return statistics.mean(numeric) if numeric else None

    def _calculate_median(self, values: List[Any]) -> Optional[float]:
        """Calculate median of numeric values."""
        numeric = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
        return statistics.median(numeric) if numeric else None

    def _calculate_mode(self, values: List[Any]) -> Any:
        """Calculate mode of values."""
        non_null = [v for v in values if v is not None]
        if not non_null:
            return None
        try:
            return statistics.mode(non_null)
        except statistics.StatisticsError:
            return non_null[0]

    def _fill_forward(self, data: List[Dict[str, Any]], field: str) -> None:
        """Fill missing values with previous value."""
        last_value = None
        for d in data:
            if d.get(field) is not None:
                last_value = d[field]
            elif last_value is not None:
                d[field] = last_value

    def _fill_backward(self, data: List[Dict[str, Any]], field: str) -> None:
        """Fill missing values with next value."""
        last_value = None
        for d in reversed(data):
            if d.get(field) is not None:
                last_value = d[field]
            elif last_value is not None:
                d[field] = last_value

    def _interpolate_linear(self, data: List[Dict[str, Any]], field: str) -> None:
        """Linear interpolation for missing values."""
        # Find missing indices
        for i, d in enumerate(data):
            if d.get(field) is None:
                # Find previous and next non-null values
                prev_idx = None
                prev_val = None
                for j in range(i - 1, -1, -1):
                    if data[j].get(field) is not None:
                        prev_idx = j
                        prev_val = data[j][field]
                        break

                next_idx = None
                next_val = None
                for j in range(i + 1, len(data)):
                    if data[j].get(field) is not None:
                        next_idx = j
                        next_val = data[j][field]
                        break

                if prev_val is not None and next_val is not None:
                    # Interpolate
                    try:
                        ratio = (i - prev_idx) / (next_idx - prev_idx)
                        d[field] = prev_val + (next_val - prev_val) * ratio
                    except Exception:
                        pass
                elif prev_val is not None:
                    d[field] = prev_val
                elif next_val is not None:
                    d[field] = next_val


# =============================================================================
# Outlier Detection
# =============================================================================


class OutlierDetector:
    """Detects outliers in data using various methods."""

    def __init__(
        self,
        method: OutlierMethod = OutlierMethod.Z_SCORE,
        threshold: float = 3.0,
    ) -> None:
        """
        Initialize outlier detector.

        Args:
            method: Detection method
            threshold: Detection threshold
        """
        self._method = method
        self._threshold = threshold

    def detect(
        self,
        values: List[float],
    ) -> List[Tuple[int, float, bool]]:
        """
        Detect outliers in a list of values.

        Returns:
            List of (index, value, is_outlier) tuples
        """
        if self._method == OutlierMethod.Z_SCORE:
            return self._detect_zscore(values)
        elif self._method == OutlierMethod.MODIFIED_Z_SCORE:
            return self._detect_modified_zscore(values)
        elif self._method == OutlierMethod.IQR:
            return self._detect_iqr(values)
        else:
            raise ValueError(f"Unsupported method: {self._method}")

    def _detect_zscore(self, values: List[float]) -> List[Tuple[int, float, bool]]:
        """Z-score based outlier detection."""
        if len(values) < 2:
            return [(i, v, False) for i, v in enumerate(values)]

        mean = statistics.mean(values)
        std = statistics.stdev(values)

        if std == 0:
            return [(i, v, False) for i, v in enumerate(values)]

        results = []
        for i, v in enumerate(values):
            z_score = abs(v - mean) / std
            is_outlier = z_score > self._threshold
            results.append((i, v, is_outlier))

        return results

    def _detect_modified_zscore(self, values: List[float]) -> List[Tuple[int, float, bool]]:
        """Modified Z-score (MAD-based) outlier detection."""
        if len(values) < 2:
            return [(i, v, False) for i, v in enumerate(values)]

        median = statistics.median(values)
        mad = statistics.median([abs(v - median) for v in values])

        if mad == 0:
            return [(i, v, False) for i, v in enumerate(values)]

        results = []
        for i, v in enumerate(values):
            modified_z = 0.6745 * (v - median) / mad
            is_outlier = abs(modified_z) > self._threshold
            results.append((i, v, is_outlier))

        return results

    def _detect_iqr(self, values: List[float]) -> List[Tuple[int, float, bool]]:
        """IQR-based outlier detection."""
        if len(values) < 4:
            return [(i, v, False) for i, v in enumerate(values)]

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        q1 = sorted_vals[n // 4]
        q3 = sorted_vals[3 * n // 4]
        iqr = q3 - q1

        lower_bound = q1 - self._threshold * iqr
        upper_bound = q3 + self._threshold * iqr

        results = []
        for i, v in enumerate(values):
            is_outlier = v < lower_bound or v > upper_bound
            results.append((i, v, is_outlier))

        return results

    def remove_outliers(
        self,
        data: List[Dict[str, Any]],
        field: str,
    ) -> List[Dict[str, Any]]:
        """
        Remove records with outlier values in specified field.

        Returns:
            Data with outliers removed
        """
        values = [d.get(field) for d in data if isinstance(d.get(field), (int, float))]

        if not values:
            return data

        detection = self.detect(values)
        outlier_values = set(v for _, v, is_outlier in detection if is_outlier)

        return [d for d in data if d.get(field) not in outlier_values]

    def flag_outliers(
        self,
        data: List[Dict[str, Any]],
        field: str,
        flag_field: str = "is_outlier",
    ) -> List[Dict[str, Any]]:
        """
        Flag records with outlier values.

        Returns:
            Data with outlier flag added
        """
        values = []
        indices = []
        for i, d in enumerate(data):
            val = d.get(field)
            if isinstance(val, (int, float)):
                values.append(val)
                indices.append(i)

        if not values:
            return data

        detection = self.detect(values)

        result = [d.copy() for d in data]
        for idx, (_, _, is_outlier) in zip(indices, detection):
            result[idx][flag_field] = is_outlier

        return result


# =============================================================================
# Composite Transformer
# =============================================================================


class DataTransformer:
    """
    Composite data transformer combining all transformation utilities.

    Provides a unified interface for data transformation operations.
    """

    def __init__(self) -> None:
        """Initialize data transformer."""
        self.unit_converter = UnitConverter()
        self.timestamp_normalizer = TimestampNormalizer()
        self.schema_mapper = SchemaMapper()
        self.quality_scorer = DataQualityScorer()
        self.missing_handler = MissingDataHandler()
        self.outlier_detector = OutlierDetector()

    def transform_record(
        self,
        data: Dict[str, Any],
        mapping_name: Optional[str] = None,
        unit_conversions: Optional[Dict[str, Tuple[str, str]]] = None,
        timestamp_fields: Optional[List[str]] = None,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Transform a single data record.

        Args:
            data: Input data record
            mapping_name: Optional schema mapping to apply
            unit_conversions: Dict of field -> (from_unit, to_unit)
            timestamp_fields: Fields to normalize as timestamps
            validate: Whether to validate and score quality

        Returns:
            Transformed data with optional quality score
        """
        result = data.copy()

        # Apply schema mapping
        if mapping_name:
            result = self.schema_mapper.map(mapping_name, result)

        # Convert units
        if unit_conversions:
            for field, (from_unit, to_unit) in unit_conversions.items():
                if field in result and result[field] is not None:
                    try:
                        result[field] = self.unit_converter.convert(
                            result[field], from_unit, to_unit
                        )
                    except Exception as e:
                        logger.warning(f"Unit conversion failed for {field}: {e}")

        # Normalize timestamps
        if timestamp_fields:
            for field in timestamp_fields:
                if field in result and result[field] is not None:
                    try:
                        result[field] = self.timestamp_normalizer.normalize(result[field])
                    except Exception as e:
                        logger.warning(f"Timestamp normalization failed for {field}: {e}")

        # Add quality score
        if validate:
            quality = self.quality_scorer.score(result)
            result["_quality_score"] = quality["overall_score"]

        return result

    def transform_batch(
        self,
        data: List[Dict[str, Any]],
        mapping_name: Optional[str] = None,
        unit_conversions: Optional[Dict[str, Tuple[str, str]]] = None,
        timestamp_fields: Optional[List[str]] = None,
        handle_missing: bool = True,
        detect_outliers: Optional[str] = None,
        min_quality_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Transform a batch of data records.

        Args:
            data: List of data records
            mapping_name: Optional schema mapping
            unit_conversions: Unit conversion specifications
            timestamp_fields: Timestamp fields to normalize
            handle_missing: Whether to handle missing values
            detect_outliers: Field for outlier detection
            min_quality_score: Minimum quality score to keep

        Returns:
            Transformed data batch
        """
        # Transform each record
        result = [
            self.transform_record(
                d, mapping_name, unit_conversions, timestamp_fields, validate=True
            )
            for d in data
        ]

        # Handle missing data
        if handle_missing:
            result = self.missing_handler.handle(result)

        # Detect and flag outliers
        if detect_outliers:
            result = self.outlier_detector.flag_outliers(result, detect_outliers)

        # Filter by quality score
        if min_quality_score > 0:
            result = [
                d for d in result
                if d.get("_quality_score", 1.0) >= min_quality_score
            ]

        return result
