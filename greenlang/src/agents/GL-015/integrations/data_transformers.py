"""
Data Transformers Module for GL-015 INSULSCAN (Insulation Inspection Agent).

Provides ETL utilities for data ingestion including:
- IR image format conversion and processing
- Temperature matrix normalization
- Unit conversion on ingestion
- Schema validation
- Data quality assessment for thermal imaging data

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
    """Categories of measurement units for insulation inspection."""

    TEMPERATURE = "temperature"
    LENGTH = "length"
    AREA = "area"
    THICKNESS = "thickness"
    THERMAL_CONDUCTIVITY = "thermal_conductivity"
    THERMAL_RESISTANCE = "thermal_resistance"  # R-value
    HEAT_TRANSFER_COEFFICIENT = "heat_transfer_coefficient"  # U-value
    HEAT_FLUX = "heat_flux"  # W/m2
    POWER = "power"
    ENERGY = "energy"
    VELOCITY = "velocity"  # Wind speed
    HUMIDITY = "humidity"
    EMISSIVITY = "emissivity"
    DENSITY = "density"


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
    PHYSICAL_LIMITS = "physical_limits"  # Domain-specific limits


class DataQualityDimension(str, Enum):
    """Data quality dimensions."""

    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    RANGE_CONFORMITY = "range_conformity"
    SPATIAL_CONSISTENCY = "spatial_consistency"  # For thermal images


class ImageProcessingStep(str, Enum):
    """Steps in thermal image processing pipeline."""

    RAW_EXTRACTION = "raw_extraction"
    RADIOMETRIC_CONVERSION = "radiometric_conversion"
    NOISE_REDUCTION = "noise_reduction"
    EDGE_ENHANCEMENT = "edge_enhancement"
    TEMPERATURE_CALIBRATION = "temperature_calibration"
    NORMALIZATION = "normalization"


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


# Comprehensive unit definitions for insulation inspection
UNIT_DEFINITIONS: Dict[str, UnitDefinition] = {
    # Temperature
    "K": UnitDefinition("K", "kelvin", UnitCategory.TEMPERATURE, "K", 1.0),
    "C": UnitDefinition("C", "celsius", UnitCategory.TEMPERATURE, "K", 1.0, 273.15),
    "F": UnitDefinition("F", "fahrenheit", UnitCategory.TEMPERATURE, "K", 5/9, 255.372),
    "R": UnitDefinition("R", "rankine", UnitCategory.TEMPERATURE, "K", 5/9),

    # Length
    "m": UnitDefinition("m", "meter", UnitCategory.LENGTH, "m", 1.0),
    "mm": UnitDefinition("mm", "millimeter", UnitCategory.LENGTH, "m", 0.001),
    "cm": UnitDefinition("cm", "centimeter", UnitCategory.LENGTH, "m", 0.01),
    "ft": UnitDefinition("ft", "foot", UnitCategory.LENGTH, "m", 0.3048),
    "in": UnitDefinition("in", "inch", UnitCategory.LENGTH, "m", 0.0254),

    # Area
    "m2": UnitDefinition("m2", "square meters", UnitCategory.AREA, "m2", 1.0),
    "cm2": UnitDefinition("cm2", "square centimeters", UnitCategory.AREA, "m2", 0.0001),
    "ft2": UnitDefinition("ft2", "square feet", UnitCategory.AREA, "m2", 0.092903),
    "in2": UnitDefinition("in2", "square inches", UnitCategory.AREA, "m2", 0.00064516),

    # Thermal conductivity (W/mK)
    "W/mK": UnitDefinition("W/mK", "watts per meter kelvin", UnitCategory.THERMAL_CONDUCTIVITY, "W/mK", 1.0),
    "W/mC": UnitDefinition("W/mC", "watts per meter celsius", UnitCategory.THERMAL_CONDUCTIVITY, "W/mK", 1.0),
    "BTU.in/h.ft2.F": UnitDefinition("BTU.in/h.ft2.F", "BTU inch per hour sq ft F", UnitCategory.THERMAL_CONDUCTIVITY, "W/mK", 0.1442),

    # Thermal resistance (R-value) - m2K/W
    "m2K/W": UnitDefinition("m2K/W", "m2 kelvin per watt", UnitCategory.THERMAL_RESISTANCE, "m2K/W", 1.0),
    "m2C/W": UnitDefinition("m2C/W", "m2 celsius per watt", UnitCategory.THERMAL_RESISTANCE, "m2K/W", 1.0),
    "h.ft2.F/BTU": UnitDefinition("h.ft2.F/BTU", "hour sq ft F per BTU", UnitCategory.THERMAL_RESISTANCE, "m2K/W", 0.1761),
    "RSI": UnitDefinition("RSI", "RSI", UnitCategory.THERMAL_RESISTANCE, "m2K/W", 1.0),
    "R_IP": UnitDefinition("R_IP", "R-value imperial", UnitCategory.THERMAL_RESISTANCE, "m2K/W", 0.1761),

    # Heat transfer coefficient (U-value) - W/m2K
    "W/m2K": UnitDefinition("W/m2K", "watts per m2 kelvin", UnitCategory.HEAT_TRANSFER_COEFFICIENT, "W/m2K", 1.0),
    "W/m2C": UnitDefinition("W/m2C", "watts per m2 celsius", UnitCategory.HEAT_TRANSFER_COEFFICIENT, "W/m2K", 1.0),
    "BTU/h.ft2.F": UnitDefinition("BTU/h.ft2.F", "BTU per hour sq ft F", UnitCategory.HEAT_TRANSFER_COEFFICIENT, "W/m2K", 5.678),

    # Heat flux - W/m2
    "W/m2": UnitDefinition("W/m2", "watts per square meter", UnitCategory.HEAT_FLUX, "W/m2", 1.0),
    "kW/m2": UnitDefinition("kW/m2", "kilowatts per square meter", UnitCategory.HEAT_FLUX, "W/m2", 1000.0),
    "BTU/h.ft2": UnitDefinition("BTU/h.ft2", "BTU per hour sq ft", UnitCategory.HEAT_FLUX, "W/m2", 3.1546),

    # Power
    "W": UnitDefinition("W", "watt", UnitCategory.POWER, "W", 1.0),
    "kW": UnitDefinition("kW", "kilowatt", UnitCategory.POWER, "W", 1000.0),
    "MW": UnitDefinition("MW", "megawatt", UnitCategory.POWER, "W", 1000000.0),
    "BTU/h": UnitDefinition("BTU/h", "BTU per hour", UnitCategory.POWER, "W", 0.293071),
    "hp": UnitDefinition("hp", "horsepower", UnitCategory.POWER, "W", 745.7),

    # Energy
    "J": UnitDefinition("J", "joule", UnitCategory.ENERGY, "J", 1.0),
    "kJ": UnitDefinition("kJ", "kilojoule", UnitCategory.ENERGY, "J", 1000.0),
    "MJ": UnitDefinition("MJ", "megajoule", UnitCategory.ENERGY, "J", 1000000.0),
    "Wh": UnitDefinition("Wh", "watt-hour", UnitCategory.ENERGY, "J", 3600.0),
    "kWh": UnitDefinition("kWh", "kilowatt-hour", UnitCategory.ENERGY, "J", 3600000.0),
    "MWh": UnitDefinition("MWh", "megawatt-hour", UnitCategory.ENERGY, "J", 3600000000.0),
    "BTU": UnitDefinition("BTU", "British thermal unit", UnitCategory.ENERGY, "J", 1055.06),
    "therm": UnitDefinition("therm", "therm", UnitCategory.ENERGY, "J", 105505600.0),

    # Velocity (wind speed)
    "m/s": UnitDefinition("m/s", "meters per second", UnitCategory.VELOCITY, "m/s", 1.0),
    "km/h": UnitDefinition("km/h", "kilometers per hour", UnitCategory.VELOCITY, "m/s", 0.277778),
    "mph": UnitDefinition("mph", "miles per hour", UnitCategory.VELOCITY, "m/s", 0.44704),
    "ft/s": UnitDefinition("ft/s", "feet per second", UnitCategory.VELOCITY, "m/s", 0.3048),
    "knot": UnitDefinition("knot", "knot", UnitCategory.VELOCITY, "m/s", 0.514444),

    # Humidity
    "%RH": UnitDefinition("%RH", "percent relative humidity", UnitCategory.HUMIDITY, "%RH", 1.0),
    "g/m3": UnitDefinition("g/m3", "grams per cubic meter", UnitCategory.HUMIDITY, "g/m3", 1.0),

    # Density
    "kg/m3": UnitDefinition("kg/m3", "kilograms per cubic meter", UnitCategory.DENSITY, "kg/m3", 1.0),
    "lb/ft3": UnitDefinition("lb/ft3", "pounds per cubic foot", UnitCategory.DENSITY, "kg/m3", 16.0185),
}


class UnitConverter:
    """Handles unit conversions for insulation inspection data."""

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
            "degrees_c": "C",
            "degrees_f": "F",

            # Length aliases
            "meter": "m",
            "meters": "m",
            "millimeter": "mm",
            "millimeters": "mm",
            "inch": "in",
            "inches": "in",
            "foot": "ft",
            "feet": "ft",

            # Area aliases
            "sqm": "m2",
            "sqft": "ft2",
            "square_meters": "m2",
            "square_feet": "ft2",

            # Power aliases
            "watt": "W",
            "watts": "W",
            "kilowatt": "kW",
            "kilowatts": "kW",

            # Energy aliases
            "kwh": "kWh",
            "mwh": "MWh",

            # Velocity aliases
            "m_per_s": "m/s",
            "km_per_h": "km/h",
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

    def convert_temperature_matrix(
        self,
        matrix: List[List[float]],
        from_unit: str,
        to_unit: str,
    ) -> List[List[float]]:
        """Convert a temperature matrix."""
        return [
            [self.convert(t, from_unit, to_unit) for t in row]
            for row in matrix
        ]


# =============================================================================
# Temperature Matrix Processing
# =============================================================================


class TemperatureMatrixNormalizer:
    """Normalizes temperature matrices from thermal cameras."""

    def __init__(
        self,
        target_unit: str = "C",
        remove_outliers: bool = True,
        outlier_method: OutlierMethod = OutlierMethod.IQR,
        normalize_range: bool = False,
        target_range: Tuple[float, float] = (0.0, 1.0)
    ) -> None:
        """
        Initialize temperature matrix normalizer.

        Args:
            target_unit: Target temperature unit
            remove_outliers: Whether to remove outliers
            outlier_method: Outlier detection method
            normalize_range: Normalize to 0-1 range
            target_range: Target range for normalization
        """
        self._target_unit = target_unit
        self._remove_outliers = remove_outliers
        self._outlier_method = outlier_method
        self._normalize_range = normalize_range
        self._target_range = target_range
        self._unit_converter = UnitConverter()

    def normalize(
        self,
        matrix: List[List[float]],
        source_unit: str = "C",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Normalize a temperature matrix.

        Args:
            matrix: Input temperature matrix
            source_unit: Source temperature unit
            metadata: Optional metadata

        Returns:
            Tuple of (normalized matrix, statistics)
        """
        metadata = metadata or {}
        stats = {}

        # Convert units if needed
        if source_unit != self._target_unit:
            matrix = self._unit_converter.convert_temperature_matrix(
                matrix, source_unit, self._target_unit
            )

        # Flatten for statistics
        flat = [t for row in matrix for t in row]
        original_stats = self._calculate_stats(flat)
        stats["original"] = original_stats

        # Remove outliers
        if self._remove_outliers:
            matrix, outlier_mask = self._remove_outliers_from_matrix(matrix, flat)
            stats["outliers_removed"] = sum(sum(row) for row in outlier_mask)

        # Recalculate stats after outlier removal
        flat = [t for row in matrix for t in row if t is not None]
        clean_stats = self._calculate_stats(flat)
        stats["clean"] = clean_stats

        # Normalize range if requested
        if self._normalize_range:
            matrix = self._normalize_to_range(matrix, clean_stats)
            stats["normalized_range"] = self._target_range

        stats["unit"] = self._target_unit

        return matrix, stats

    def _calculate_stats(
        self,
        values: List[float]
    ) -> Dict[str, float]:
        """Calculate statistics for a list of values."""
        valid_values = [v for v in values if v is not None and not math.isnan(v)]

        if not valid_values:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "count": 0,
            }

        return {
            "min": min(valid_values),
            "max": max(valid_values),
            "mean": statistics.mean(valid_values),
            "median": statistics.median(valid_values),
            "std": statistics.stdev(valid_values) if len(valid_values) > 1 else 0.0,
            "count": len(valid_values),
        }

    def _remove_outliers_from_matrix(
        self,
        matrix: List[List[float]],
        flat: List[float]
    ) -> Tuple[List[List[float]], List[List[bool]]]:
        """Remove outliers from matrix."""
        # Calculate thresholds
        if self._outlier_method == OutlierMethod.IQR:
            lower, upper = self._iqr_bounds(flat)
        elif self._outlier_method == OutlierMethod.Z_SCORE:
            lower, upper = self._z_score_bounds(flat, 3.0)
        elif self._outlier_method == OutlierMethod.PHYSICAL_LIMITS:
            # Physical temperature limits for insulation inspection
            lower, upper = -50.0, 500.0  # Celsius
        else:
            lower, upper = float('-inf'), float('inf')

        result = []
        mask = []

        for row in matrix:
            new_row = []
            mask_row = []
            for t in row:
                is_outlier = t < lower or t > upper
                if is_outlier:
                    # Replace with None or interpolate
                    new_row.append(None)
                else:
                    new_row.append(t)
                mask_row.append(is_outlier)
            result.append(new_row)
            mask.append(mask_row)

        # Fill None values with interpolation
        result = self._interpolate_missing(result)

        return result, mask

    def _iqr_bounds(
        self,
        values: List[float],
        factor: float = 1.5
    ) -> Tuple[float, float]:
        """Calculate IQR-based outlier bounds."""
        sorted_values = sorted(v for v in values if v is not None)
        n = len(sorted_values)
        q1 = sorted_values[n // 4]
        q3 = sorted_values[3 * n // 4]
        iqr = q3 - q1
        return q1 - factor * iqr, q3 + factor * iqr

    def _z_score_bounds(
        self,
        values: List[float],
        threshold: float = 3.0
    ) -> Tuple[float, float]:
        """Calculate z-score-based outlier bounds."""
        valid_values = [v for v in values if v is not None]
        mean = statistics.mean(valid_values)
        std = statistics.stdev(valid_values) if len(valid_values) > 1 else 1.0
        return mean - threshold * std, mean + threshold * std

    def _interpolate_missing(
        self,
        matrix: List[List[float]]
    ) -> List[List[float]]:
        """Interpolate missing values in matrix."""
        height = len(matrix)
        width = len(matrix[0]) if matrix else 0

        result = [row[:] for row in matrix]

        for i in range(height):
            for j in range(width):
                if result[i][j] is None:
                    # Find neighbors
                    neighbors = []
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            if result[ni][nj] is not None:
                                neighbors.append(result[ni][nj])

                    if neighbors:
                        result[i][j] = sum(neighbors) / len(neighbors)
                    else:
                        # Use overall mean
                        all_valid = [
                            t for row in result
                            for t in row if t is not None
                        ]
                        result[i][j] = sum(all_valid) / len(all_valid) if all_valid else 0.0

        return result

    def _normalize_to_range(
        self,
        matrix: List[List[float]],
        stats: Dict[str, float]
    ) -> List[List[float]]:
        """Normalize matrix to target range."""
        min_val = stats["min"]
        max_val = stats["max"]
        range_val = max_val - min_val

        if range_val == 0:
            return matrix

        target_min, target_max = self._target_range
        target_range = target_max - target_min

        return [
            [
                target_min + (t - min_val) / range_val * target_range
                for t in row
            ]
            for row in matrix
        ]


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
    """Maps data between different schemas."""

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


class ThermalDataQualityScorer:
    """Assesses data quality for thermal imaging data."""

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

        # Thermal imaging specific ranges
        self._thermal_ranges = {
            "temperature_c": (-273.15, 1000.0),  # Physical limits
            "emissivity": (0.01, 1.0),
            "humidity_percent": (0.0, 100.0),
            "wind_speed_m_s": (0.0, 100.0),
            "heat_loss_kw": (0.0, 100000.0),
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

    def score_temperature_matrix(
        self,
        matrix: List[List[float]],
        expected_range: Optional[Tuple[float, float]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score quality of a temperature matrix.

        Args:
            matrix: Temperature matrix
            expected_range: Expected temperature range

        Returns:
            Tuple of (score, details)
        """
        if not matrix or not matrix[0]:
            return 0.0, {"error": "Empty matrix"}

        flat = [t for row in matrix for t in row if t is not None]

        if not flat:
            return 0.0, {"error": "No valid temperatures"}

        # Calculate statistics
        stats = {
            "min": min(flat),
            "max": max(flat),
            "mean": statistics.mean(flat),
            "std": statistics.stdev(flat) if len(flat) > 1 else 0.0,
            "count": len(flat),
            "null_count": sum(1 for row in matrix for t in row if t is None),
        }

        issues = []
        score = 100.0

        # Check for null values
        null_ratio = stats["null_count"] / (len(matrix) * len(matrix[0]))
        if null_ratio > 0.1:
            score -= 20
            issues.append(f"High null ratio: {null_ratio:.1%}")
        elif null_ratio > 0:
            score -= 5

        # Check range conformity
        if expected_range:
            min_exp, max_exp = expected_range
            if stats["min"] < min_exp:
                score -= 15
                issues.append(f"Temperatures below expected: {stats['min']:.1f}C")
            if stats["max"] > max_exp:
                score -= 15
                issues.append(f"Temperatures above expected: {stats['max']:.1f}C")

        # Check for suspicious uniformity
        if stats["std"] < 0.1:
            score -= 10
            issues.append("Suspiciously uniform temperatures")

        # Check for physical limits
        if stats["min"] < -50 or stats["max"] > 500:
            score -= 20
            issues.append("Values outside typical insulation inspection range")

        score = max(0, min(100, score))

        return score / 100.0, {
            "statistics": stats,
            "issues": issues,
            "expected_range": expected_range,
        }

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

        # Combine custom and thermal-specific ranges
        all_ranges = {**self._thermal_ranges, **self._range_constraints}

        for field, (min_val, max_val) in all_ranges.items():
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
# IR Image Format Processing
# =============================================================================


class ThermalImageProcessor:
    """Processes thermal images from various camera formats."""

    def __init__(
        self,
        target_temperature_unit: str = "C",
        apply_noise_reduction: bool = True,
        normalize_output: bool = False
    ) -> None:
        """
        Initialize thermal image processor.

        Args:
            target_temperature_unit: Target temperature unit
            apply_noise_reduction: Apply noise reduction filter
            normalize_output: Normalize output to 0-1 range
        """
        self._target_unit = target_temperature_unit
        self._noise_reduction = apply_noise_reduction
        self._normalize = normalize_output
        self._unit_converter = UnitConverter()
        self._matrix_normalizer = TemperatureMatrixNormalizer(
            target_unit=target_temperature_unit,
            normalize_range=normalize_output
        )

    def process_raw_data(
        self,
        raw_data: bytes,
        width: int,
        height: int,
        bits_per_pixel: int = 16,
        byte_order: str = "little",
        calibration: Optional[Dict[str, float]] = None
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Process raw radiometric data to temperature matrix.

        Args:
            raw_data: Raw byte data
            width: Image width
            height: Image height
            bits_per_pixel: Bits per pixel (typically 14 or 16)
            byte_order: Byte order (little or big)
            calibration: Calibration coefficients

        Returns:
            Tuple of (temperature matrix, processing metadata)
        """
        import struct

        metadata = {
            "width": width,
            "height": height,
            "bits_per_pixel": bits_per_pixel,
            "processing_steps": []
        }

        # Unpack raw values
        format_char = "<" if byte_order == "little" else ">"
        if bits_per_pixel <= 8:
            format_char += "B"
            pixel_size = 1
        elif bits_per_pixel <= 16:
            format_char += "H"
            pixel_size = 2
        else:
            format_char += "I"
            pixel_size = 4

        expected_size = width * height * pixel_size
        if len(raw_data) < expected_size:
            raise ValueError(
                f"Insufficient data: expected {expected_size}, got {len(raw_data)}"
            )

        # Extract raw values
        raw_values = []
        for i in range(0, expected_size, pixel_size):
            value = struct.unpack(format_char, raw_data[i:i + pixel_size])[0]
            raw_values.append(value)

        metadata["processing_steps"].append(ImageProcessingStep.RAW_EXTRACTION.value)

        # Convert to temperature using calibration
        calibration = calibration or {
            "planck_r1": 21106.77,
            "planck_b": 1501.0,
            "planck_f": 1.0,
            "planck_o": -7340.0,
            "planck_r2": 0.012545258,
            "emissivity": 0.95,
            "reflected_temp_k": 295.15,
        }

        temperatures = self._apply_radiometric_conversion(
            raw_values,
            calibration
        )
        metadata["processing_steps"].append(
            ImageProcessingStep.RADIOMETRIC_CONVERSION.value
        )

        # Reshape to matrix
        matrix = [
            temperatures[i * width:(i + 1) * width]
            for i in range(height)
        ]

        # Apply noise reduction
        if self._noise_reduction:
            matrix = self._apply_noise_reduction(matrix)
            metadata["processing_steps"].append(
                ImageProcessingStep.NOISE_REDUCTION.value
            )

        # Normalize
        matrix, norm_stats = self._matrix_normalizer.normalize(
            matrix,
            source_unit="K",
            metadata=metadata
        )
        metadata["statistics"] = norm_stats

        return matrix, metadata

    def _apply_radiometric_conversion(
        self,
        raw_values: List[int],
        calibration: Dict[str, float]
    ) -> List[float]:
        """
        Convert raw sensor values to temperature using Planck's law.

        Uses the standard radiometric equation:
        T = B / ln(R1 / (R2 * (S + O)) + F)

        Where:
        - T is object temperature in Kelvin
        - S is raw sensor value
        - R1, R2, B, F, O are calibration constants
        """
        R1 = calibration.get("planck_r1", 21106.77)
        R2 = calibration.get("planck_r2", 0.012545258)
        B = calibration.get("planck_b", 1501.0)
        F = calibration.get("planck_f", 1.0)
        O = calibration.get("planck_o", -7340.0)
        emissivity = calibration.get("emissivity", 0.95)
        reflected_temp = calibration.get("reflected_temp_k", 295.15)

        temperatures = []
        for S in raw_values:
            try:
                # Apply emissivity correction
                S_corrected = S - (1 - emissivity) * (
                    R1 / (R2 * math.exp(B / reflected_temp) - F) + O
                )

                # Calculate temperature
                T = B / math.log(R1 / (R2 * (S_corrected + O)) + F)

                temperatures.append(T)
            except (ValueError, ZeroDivisionError, OverflowError):
                temperatures.append(float('nan'))

        return temperatures

    def _apply_noise_reduction(
        self,
        matrix: List[List[float]],
        kernel_size: int = 3
    ) -> List[List[float]]:
        """Apply median filter for noise reduction."""
        height = len(matrix)
        width = len(matrix[0])
        half_k = kernel_size // 2

        result = [[0.0] * width for _ in range(height)]

        for i in range(height):
            for j in range(width):
                neighbors = []
                for di in range(-half_k, half_k + 1):
                    for dj in range(-half_k, half_k + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            val = matrix[ni][nj]
                            if not math.isnan(val):
                                neighbors.append(val)

                if neighbors:
                    result[i][j] = statistics.median(neighbors)
                else:
                    result[i][j] = matrix[i][j]

        return result


# =============================================================================
# Comprehensive Data Transformer
# =============================================================================


class InsulationDataTransformer:
    """
    Comprehensive data transformer for insulation inspection data.

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
            quality_threshold: Minimum quality score threshold
        """
        self._unit_converter = UnitConverter()

        self._schema_mapper = SchemaMapper(
            mappings=schema_mappings or [],
            unit_converter=self._unit_converter
        ) if schema_mappings else None

        self._quality_scorer = ThermalDataQualityScorer(
            required_fields=required_fields,
            numeric_fields=numeric_fields,
            range_constraints=range_constraints
        )

        self._matrix_normalizer = TemperatureMatrixNormalizer()
        self._image_processor = ThermalImageProcessor()

        self._target_units = target_units or {}
        self._quality_threshold = quality_threshold

    def transform(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        validate: bool = True
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Transform data through the complete pipeline.

        Args:
            data: Input data (single record or list)
            validate: Perform quality validation

        Returns:
            Transformed data with quality metadata
        """
        if isinstance(data, list):
            return [self._transform_record(r, validate) for r in data]
        return self._transform_record(data, validate)

    def _transform_record(
        self,
        record: Dict[str, Any],
        validate: bool
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

    def transform_temperature_matrix(
        self,
        matrix: List[List[float]],
        source_unit: str = "C",
        target_unit: str = "C"
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Transform a temperature matrix.

        Args:
            matrix: Input temperature matrix
            source_unit: Source temperature unit
            target_unit: Target temperature unit

        Returns:
            Tuple of (transformed matrix, metadata)
        """
        # Normalize matrix
        normalized, stats = self._matrix_normalizer.normalize(
            matrix,
            source_unit=source_unit
        )

        # Convert units if needed
        if source_unit != target_unit:
            normalized = self._unit_converter.convert_temperature_matrix(
                normalized, source_unit, target_unit
            )
            stats["converted_to"] = target_unit

        # Score quality
        quality_score, quality_details = self._quality_scorer.score_temperature_matrix(
            normalized
        )
        stats["quality_score"] = quality_score
        stats["quality_details"] = quality_details

        return normalized, stats

    @property
    def unit_converter(self) -> UnitConverter:
        """Get unit converter."""
        return self._unit_converter

    @property
    def quality_scorer(self) -> ThermalDataQualityScorer:
        """Get quality scorer."""
        return self._quality_scorer

    @property
    def image_processor(self) -> ThermalImageProcessor:
        """Get image processor."""
        return self._image_processor
