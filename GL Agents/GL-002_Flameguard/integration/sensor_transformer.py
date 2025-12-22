"""
GL-002 FLAMEGUARD - Sensor Data Transformer

Transforms raw sensor data to engineering units with:
- Unit conversion
- Signal conditioning
- Outlier detection
- Data quality assessment
- Derived calculations
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import logging
import math
import statistics

logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """Data quality status."""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    STALE = "stale"
    OUT_OF_RANGE = "out_of_range"
    FROZEN = "frozen"
    RATE_OF_CHANGE = "rate_of_change"
    SENSOR_FAULT = "sensor_fault"


class TransformationType(Enum):
    """Transformation types."""
    LINEAR = "linear"  # y = mx + b
    POLYNOMIAL = "polynomial"  # y = sum(c_i * x^i)
    SQUARE_ROOT = "square_root"  # For flow calculations
    LOGARITHMIC = "logarithmic"
    EXPONENTIAL = "exponential"
    LOOKUP_TABLE = "lookup_table"
    CUSTOM = "custom"


class FilterType(Enum):
    """Signal filter types."""
    NONE = "none"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL = "exponential"
    MEDIAN = "median"
    LOW_PASS = "low_pass"
    RATE_LIMIT = "rate_limit"


@dataclass
class SensorReading:
    """Processed sensor reading."""
    tag: str
    raw_value: float
    value: float
    unit: str
    quality: DataQuality
    timestamp: datetime

    # Metadata
    transformation_applied: str = ""
    filter_applied: str = ""
    quality_checks: List[str] = field(default_factory=list)

    # Provenance
    source_hash: str = ""

    def to_dict(self) -> Dict:
        return {
            "tag": self.tag,
            "raw_value": self.raw_value,
            "value": self.value,
            "unit": self.unit,
            "quality": self.quality.value,
            "timestamp": self.timestamp.isoformat(),
            "transformation": self.transformation_applied,
            "filter": self.filter_applied,
            "quality_checks": self.quality_checks,
            "source_hash": self.source_hash,
        }


@dataclass
class TransformationRule:
    """Sensor transformation configuration."""
    tag: str

    # Transformation
    transform_type: TransformationType = TransformationType.LINEAR
    coefficients: List[float] = field(default_factory=lambda: [1.0, 0.0])  # [slope, intercept]
    lookup_table: Optional[Dict[float, float]] = None
    custom_function: Optional[Callable[[float], float]] = None

    # Unit conversion
    input_unit: str = ""
    output_unit: str = ""

    # Filtering
    filter_type: FilterType = FilterType.NONE
    filter_window: int = 5
    filter_alpha: float = 0.1  # For exponential filter

    # Quality checks
    valid_range: Tuple[float, float] = (-float('inf'), float('inf'))
    max_rate_of_change: Optional[float] = None  # Per second
    stale_timeout_s: float = 60.0
    frozen_tolerance: float = 0.001
    frozen_count: int = 10

    # Derived calculation
    is_derived: bool = False
    source_tags: List[str] = field(default_factory=list)
    derived_formula: Optional[str] = None


class SignalFilter:
    """Signal filtering implementation."""

    def __init__(
        self,
        filter_type: FilterType,
        window_size: int = 5,
        alpha: float = 0.1,
    ) -> None:
        self.filter_type = filter_type
        self.window_size = window_size
        self.alpha = alpha

        self._buffer: List[float] = []
        self._last_value: Optional[float] = None

    def filter(self, value: float) -> float:
        """Apply filter to value."""
        if self.filter_type == FilterType.NONE:
            return value

        elif self.filter_type == FilterType.MOVING_AVERAGE:
            self._buffer.append(value)
            if len(self._buffer) > self.window_size:
                self._buffer.pop(0)
            return statistics.mean(self._buffer)

        elif self.filter_type == FilterType.EXPONENTIAL:
            if self._last_value is None:
                self._last_value = value
            else:
                self._last_value = self.alpha * value + (1 - self.alpha) * self._last_value
            return self._last_value

        elif self.filter_type == FilterType.MEDIAN:
            self._buffer.append(value)
            if len(self._buffer) > self.window_size:
                self._buffer.pop(0)
            return statistics.median(self._buffer)

        elif self.filter_type == FilterType.RATE_LIMIT:
            if self._last_value is None:
                self._last_value = value
            else:
                max_change = self.alpha  # Use alpha as max change per sample
                change = value - self._last_value
                if abs(change) > max_change:
                    change = max_change if change > 0 else -max_change
                self._last_value = self._last_value + change
            return self._last_value

        return value

    def reset(self) -> None:
        """Reset filter state."""
        self._buffer.clear()
        self._last_value = None


class QualityChecker:
    """Data quality checking."""

    def __init__(
        self,
        valid_range: Tuple[float, float],
        max_rate_of_change: Optional[float] = None,
        stale_timeout_s: float = 60.0,
        frozen_tolerance: float = 0.001,
        frozen_count: int = 10,
    ) -> None:
        self.valid_range = valid_range
        self.max_rate_of_change = max_rate_of_change
        self.stale_timeout_s = stale_timeout_s
        self.frozen_tolerance = frozen_tolerance
        self.frozen_count = frozen_count

        self._last_value: Optional[float] = None
        self._last_timestamp: Optional[datetime] = None
        self._frozen_counter = 0

    def check(
        self,
        value: float,
        timestamp: datetime,
    ) -> Tuple[DataQuality, List[str]]:
        """Check data quality."""
        quality = DataQuality.GOOD
        issues: List[str] = []

        # Range check
        if value < self.valid_range[0] or value > self.valid_range[1]:
            quality = DataQuality.OUT_OF_RANGE
            issues.append(f"Out of range [{self.valid_range[0]}, {self.valid_range[1]}]")

        # Rate of change check
        if (self._last_value is not None and
            self._last_timestamp is not None and
            self.max_rate_of_change is not None):

            dt = (timestamp - self._last_timestamp).total_seconds()
            if dt > 0:
                rate = abs(value - self._last_value) / dt
                if rate > self.max_rate_of_change:
                    if quality == DataQuality.GOOD:
                        quality = DataQuality.RATE_OF_CHANGE
                    issues.append(f"Rate {rate:.2f}/s exceeds max {self.max_rate_of_change}")

        # Stale check
        if self._last_timestamp is not None:
            age = (timestamp - self._last_timestamp).total_seconds()
            if age > self.stale_timeout_s:
                quality = DataQuality.STALE
                issues.append(f"Data stale: {age:.1f}s since last update")

        # Frozen check
        if self._last_value is not None:
            if abs(value - self._last_value) < self.frozen_tolerance:
                self._frozen_counter += 1
                if self._frozen_counter >= self.frozen_count:
                    quality = DataQuality.FROZEN
                    issues.append(f"Value frozen for {self._frozen_counter} samples")
            else:
                self._frozen_counter = 0

        self._last_value = value
        self._last_timestamp = timestamp

        return quality, issues

    def reset(self) -> None:
        """Reset checker state."""
        self._last_value = None
        self._last_timestamp = None
        self._frozen_counter = 0


class SensorDataTransformer:
    """
    Transforms raw sensor data to engineering units.

    Features:
    - Linear, polynomial, and custom transformations
    - Signal filtering (moving average, exponential, median)
    - Data quality assessment
    - Derived calculations from multiple sensors
    - Provenance tracking with hashing
    """

    def __init__(self) -> None:
        self._rules: Dict[str, TransformationRule] = {}
        self._filters: Dict[str, SignalFilter] = {}
        self._quality_checkers: Dict[str, QualityChecker] = {}
        self._last_readings: Dict[str, SensorReading] = {}

        # Unit conversion factors
        self._unit_conversions = {
            ("degF", "degC"): lambda x: (x - 32) * 5/9,
            ("degC", "degF"): lambda x: x * 9/5 + 32,
            ("psi", "bar"): lambda x: x * 0.0689476,
            ("bar", "psi"): lambda x: x * 14.5038,
            ("psig", "kPa"): lambda x: (x + 14.696) * 6.89476,
            ("kPa", "psig"): lambda x: x / 6.89476 - 14.696,
            ("scfh", "m3/hr"): lambda x: x * 0.0283168,
            ("m3/hr", "scfh"): lambda x: x * 35.3147,
            ("klb/hr", "kg/hr"): lambda x: x * 453.592,
            ("kg/hr", "klb/hr"): lambda x: x / 453.592,
            ("BTU/scf", "MJ/m3"): lambda x: x * 0.0373,
            ("in WC", "Pa"): lambda x: x * 248.84,
            ("Pa", "in WC"): lambda x: x / 248.84,
        }

        logger.info("SensorDataTransformer initialized")

    def add_rule(self, rule: TransformationRule) -> None:
        """Add transformation rule."""
        self._rules[rule.tag] = rule

        # Create filter
        self._filters[rule.tag] = SignalFilter(
            rule.filter_type,
            rule.filter_window,
            rule.filter_alpha,
        )

        # Create quality checker
        self._quality_checkers[rule.tag] = QualityChecker(
            rule.valid_range,
            rule.max_rate_of_change,
            rule.stale_timeout_s,
            rule.frozen_tolerance,
            rule.frozen_count,
        )

        logger.debug(f"Added transformation rule: {rule.tag}")

    def transform(
        self,
        tag: str,
        raw_value: float,
        timestamp: Optional[datetime] = None,
    ) -> SensorReading:
        """Transform raw sensor value."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        if tag not in self._rules:
            # No rule - pass through
            return SensorReading(
                tag=tag,
                raw_value=raw_value,
                value=raw_value,
                unit="",
                quality=DataQuality.GOOD,
                timestamp=timestamp,
                source_hash=self._compute_hash(tag, raw_value, timestamp),
            )

        rule = self._rules[tag]
        quality_checks: List[str] = []

        # Apply transformation
        transformed_value = self._apply_transformation(raw_value, rule)
        transformation_name = rule.transform_type.value

        # Apply unit conversion
        if rule.input_unit and rule.output_unit:
            key = (rule.input_unit, rule.output_unit)
            if key in self._unit_conversions:
                transformed_value = self._unit_conversions[key](transformed_value)

        # Apply filter
        filter_name = rule.filter_type.value
        if rule.filter_type != FilterType.NONE:
            filtered_value = self._filters[tag].filter(transformed_value)
            if abs(filtered_value - transformed_value) > 0.01 * abs(transformed_value):
                quality_checks.append(f"Filtered: {transformed_value:.2f} -> {filtered_value:.2f}")
            transformed_value = filtered_value

        # Check quality
        quality, issues = self._quality_checkers[tag].check(transformed_value, timestamp)
        quality_checks.extend(issues)

        # Create reading
        reading = SensorReading(
            tag=tag,
            raw_value=raw_value,
            value=round(transformed_value, 4),
            unit=rule.output_unit or rule.input_unit,
            quality=quality,
            timestamp=timestamp,
            transformation_applied=transformation_name,
            filter_applied=filter_name,
            quality_checks=quality_checks,
            source_hash=self._compute_hash(tag, raw_value, timestamp),
        )

        self._last_readings[tag] = reading
        return reading

    def transform_batch(
        self,
        values: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, SensorReading]:
        """Transform batch of sensor values."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        readings: Dict[str, SensorReading] = {}

        # Transform individual sensors
        for tag, raw_value in values.items():
            readings[tag] = self.transform(tag, raw_value, timestamp)

        # Calculate derived values
        for tag, rule in self._rules.items():
            if rule.is_derived and tag not in readings:
                derived_reading = self._calculate_derived(rule, readings, timestamp)
                if derived_reading:
                    readings[tag] = derived_reading

        return readings

    def _apply_transformation(
        self,
        value: float,
        rule: TransformationRule,
    ) -> float:
        """Apply transformation to value."""
        if rule.transform_type == TransformationType.LINEAR:
            # y = mx + b
            slope = rule.coefficients[0] if len(rule.coefficients) > 0 else 1.0
            intercept = rule.coefficients[1] if len(rule.coefficients) > 1 else 0.0
            return slope * value + intercept

        elif rule.transform_type == TransformationType.POLYNOMIAL:
            # y = c0 + c1*x + c2*x^2 + ...
            result = 0.0
            for i, coef in enumerate(rule.coefficients):
                result += coef * (value ** i)
            return result

        elif rule.transform_type == TransformationType.SQUARE_ROOT:
            # For differential pressure flow: flow = k * sqrt(dp)
            k = rule.coefficients[0] if rule.coefficients else 1.0
            return k * math.sqrt(max(0, value))

        elif rule.transform_type == TransformationType.LOGARITHMIC:
            # y = a * ln(x) + b
            a = rule.coefficients[0] if len(rule.coefficients) > 0 else 1.0
            b = rule.coefficients[1] if len(rule.coefficients) > 1 else 0.0
            return a * math.log(max(0.001, value)) + b

        elif rule.transform_type == TransformationType.EXPONENTIAL:
            # y = a * e^(bx)
            a = rule.coefficients[0] if len(rule.coefficients) > 0 else 1.0
            b = rule.coefficients[1] if len(rule.coefficients) > 1 else 1.0
            return a * math.exp(b * value)

        elif rule.transform_type == TransformationType.LOOKUP_TABLE:
            if rule.lookup_table:
                return self._interpolate_lookup(value, rule.lookup_table)
            return value

        elif rule.transform_type == TransformationType.CUSTOM:
            if rule.custom_function:
                return rule.custom_function(value)
            return value

        return value

    def _interpolate_lookup(
        self,
        value: float,
        table: Dict[float, float],
    ) -> float:
        """Interpolate value from lookup table."""
        keys = sorted(table.keys())

        if value <= keys[0]:
            return table[keys[0]]
        if value >= keys[-1]:
            return table[keys[-1]]

        # Find surrounding points
        for i in range(len(keys) - 1):
            if keys[i] <= value <= keys[i + 1]:
                x0, x1 = keys[i], keys[i + 1]
                y0, y1 = table[x0], table[x1]

                # Linear interpolation
                return y0 + (y1 - y0) * (value - x0) / (x1 - x0)

        return value

    def _calculate_derived(
        self,
        rule: TransformationRule,
        readings: Dict[str, SensorReading],
        timestamp: datetime,
    ) -> Optional[SensorReading]:
        """Calculate derived value from source tags."""
        # Check all source tags available
        source_values = {}
        for source_tag in rule.source_tags:
            if source_tag not in readings:
                return None
            if readings[source_tag].quality == DataQuality.BAD:
                return None
            source_values[source_tag] = readings[source_tag].value

        # Evaluate formula
        if rule.derived_formula:
            try:
                # Safe evaluation with limited scope
                result = eval(rule.derived_formula, {"__builtins__": {}}, {
                    **source_values,
                    "sqrt": math.sqrt,
                    "log": math.log,
                    "exp": math.exp,
                    "abs": abs,
                    "min": min,
                    "max": max,
                })

                # Determine quality from sources
                worst_quality = DataQuality.GOOD
                for source_tag in rule.source_tags:
                    if readings[source_tag].quality.value > worst_quality.value:
                        worst_quality = readings[source_tag].quality

                return SensorReading(
                    tag=rule.tag,
                    raw_value=result,
                    value=round(result, 4),
                    unit=rule.output_unit,
                    quality=worst_quality,
                    timestamp=timestamp,
                    transformation_applied="derived",
                    quality_checks=[f"Derived from: {rule.source_tags}"],
                    source_hash=self._compute_hash(rule.tag, result, timestamp),
                )
            except Exception as e:
                logger.error(f"Error calculating derived value {rule.tag}: {e}")
                return None

        return None

    def _compute_hash(
        self,
        tag: str,
        value: float,
        timestamp: datetime,
    ) -> str:
        """Compute provenance hash."""
        data = f"{tag}:{value}:{timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def get_last_reading(self, tag: str) -> Optional[SensorReading]:
        """Get last reading for tag."""
        return self._last_readings.get(tag)

    def get_all_readings(self) -> Dict[str, SensorReading]:
        """Get all last readings."""
        return dict(self._last_readings)

    def reset(self, tag: Optional[str] = None) -> None:
        """Reset filter and quality checker state."""
        if tag:
            if tag in self._filters:
                self._filters[tag].reset()
            if tag in self._quality_checkers:
                self._quality_checkers[tag].reset()
        else:
            for f in self._filters.values():
                f.reset()
            for q in self._quality_checkers.values():
                q.reset()


def create_boiler_transformation_rules() -> List[TransformationRule]:
    """Create standard boiler transformation rules."""
    return [
        # Pressure sensors
        TransformationRule(
            tag="drum_pressure",
            transform_type=TransformationType.LINEAR,
            coefficients=[1.0, 0.0],
            input_unit="psig",
            output_unit="psig",
            filter_type=FilterType.EXPONENTIAL,
            filter_alpha=0.2,
            valid_range=(0.0, 200.0),
            max_rate_of_change=10.0,  # psig/sec
            stale_timeout_s=30.0,
        ),

        # Level sensor
        TransformationRule(
            tag="drum_level",
            transform_type=TransformationType.LINEAR,
            coefficients=[1.0, 0.0],
            input_unit="inches",
            output_unit="inches",
            filter_type=FilterType.MOVING_AVERAGE,
            filter_window=5,
            valid_range=(-10.0, 10.0),
            max_rate_of_change=2.0,  # inches/sec
            frozen_tolerance=0.01,
        ),

        # Flow sensors (differential pressure)
        TransformationRule(
            tag="steam_flow",
            transform_type=TransformationType.SQUARE_ROOT,
            coefficients=[100.0],  # K factor
            input_unit="in WC",
            output_unit="klb/hr",
            filter_type=FilterType.EXPONENTIAL,
            filter_alpha=0.15,
            valid_range=(0.0, 500.0),
        ),

        # Temperature sensors
        TransformationRule(
            tag="steam_temperature",
            transform_type=TransformationType.LINEAR,
            coefficients=[1.0, 0.0],
            input_unit="degF",
            output_unit="degF",
            filter_type=FilterType.EXPONENTIAL,
            filter_alpha=0.1,
            valid_range=(32.0, 1000.0),
            max_rate_of_change=5.0,  # degF/sec
        ),

        TransformationRule(
            tag="flue_gas_temp",
            transform_type=TransformationType.LINEAR,
            coefficients=[1.0, 0.0],
            input_unit="degF",
            output_unit="degF",
            filter_type=FilterType.MOVING_AVERAGE,
            filter_window=3,
            valid_range=(100.0, 800.0),
        ),

        # Combustion analysis
        TransformationRule(
            tag="o2_percent",
            transform_type=TransformationType.LINEAR,
            coefficients=[1.0, 0.0],
            input_unit="%",
            output_unit="%",
            filter_type=FilterType.EXPONENTIAL,
            filter_alpha=0.2,
            valid_range=(0.0, 21.0),
            max_rate_of_change=1.0,  # %/sec
        ),

        TransformationRule(
            tag="co_ppm",
            transform_type=TransformationType.LINEAR,
            coefficients=[1.0, 0.0],
            input_unit="ppm",
            output_unit="ppm",
            filter_type=FilterType.MEDIAN,
            filter_window=5,
            valid_range=(0.0, 1000.0),
        ),

        # Derived calculations
        TransformationRule(
            tag="excess_air_percent",
            transform_type=TransformationType.CUSTOM,
            is_derived=True,
            source_tags=["o2_percent"],
            derived_formula="o2_percent / (21.0 - o2_percent) * 100.0",
            output_unit="%",
        ),

        TransformationRule(
            tag="combustion_efficiency",
            transform_type=TransformationType.CUSTOM,
            is_derived=True,
            source_tags=["flue_gas_temp", "o2_percent"],
            derived_formula="100 - (0.5 * (flue_gas_temp - 70) / 100) - (0.1 * o2_percent)",
            output_unit="%",
        ),
    ]
