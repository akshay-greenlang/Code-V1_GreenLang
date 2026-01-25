"""
GL-003 UNIFIEDSTEAM - Sensor Data Transformer

Transforms raw sensor data to engineering units with:
- Unit conversion (kPa/bar/psi, C/F/K, kg/s/lb/hr, etc.)
- Calibration application (offset, gain, polynomial)
- Range validation with plausibility checks
- Rate-of-change validation
- Cross-sensor consistency checks
- Data quality flagging with OPC-style quality codes
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging
import math
import statistics
import hashlib

logger = logging.getLogger(__name__)


class QualityCode(Enum):
    """OPC-UA style quality codes for sensor data."""
    # Good quality (0x00)
    GOOD = 0x00000000
    GOOD_LOCAL_OVERRIDE = 0x00D80000
    GOOD_NO_DATA = 0x00A30000

    # Uncertain quality (0x40)
    UNCERTAIN = 0x40000000
    UNCERTAIN_LAST_USABLE_VALUE = 0x40480000
    UNCERTAIN_SENSOR_NOT_ACCURATE = 0x40890000
    UNCERTAIN_EU_EXCEEDED = 0x408A0000
    UNCERTAIN_SUB_NORMAL = 0x408B0000

    # Bad quality (0x80)
    BAD = 0x80000000
    BAD_CONFIG_ERROR = 0x80040000
    BAD_NOT_CONNECTED = 0x80100000
    BAD_DEVICE_FAILURE = 0x80110000
    BAD_SENSOR_FAILURE = 0x80120000
    BAD_COMM_FAILURE = 0x80130000
    BAD_OUT_OF_RANGE = 0x80350000
    BAD_OUT_OF_SERVICE = 0x808D0000

    def is_good(self) -> bool:
        """Check if quality is good."""
        return (self.value & 0xC0000000) == 0x00000000

    def is_uncertain(self) -> bool:
        """Check if quality is uncertain."""
        return (self.value & 0xC0000000) == 0x40000000

    def is_bad(self) -> bool:
        """Check if quality is bad."""
        return (self.value & 0xC0000000) == 0x80000000


class UnitCategory(Enum):
    """Categories of engineering units."""
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    MASS_FLOW = "mass_flow"
    VOLUMETRIC_FLOW = "volumetric_flow"
    ENERGY = "energy"
    POWER = "power"
    LENGTH = "length"
    VELOCITY = "velocity"
    DENSITY = "density"
    PERCENTAGE = "percentage"
    DIMENSIONLESS = "dimensionless"


@dataclass
class CalibrationParams:
    """Calibration parameters for sensor adjustment."""
    # Linear calibration: y = gain * x + offset
    offset: float = 0.0
    gain: float = 1.0

    # Polynomial calibration: y = c0 + c1*x + c2*x^2 + ...
    polynomial_coefficients: Optional[List[float]] = None

    # Lookup table calibration
    lookup_table: Optional[Dict[float, float]] = None

    # Calibration metadata
    calibration_date: Optional[datetime] = None
    next_calibration_date: Optional[datetime] = None
    calibrator_id: str = ""
    certificate_number: str = ""

    # Temperature compensation
    reference_temperature: float = 25.0  # Celsius
    temperature_coefficient: float = 0.0  # %/degC

    def is_due_for_calibration(self) -> bool:
        """Check if calibration is overdue."""
        if self.next_calibration_date:
            return datetime.now(timezone.utc) > self.next_calibration_date
        return False


@dataclass
class ValidationResult:
    """Result of value validation."""
    is_valid: bool
    quality_code: QualityCode
    issues: List[str] = field(default_factory=list)
    corrected_value: Optional[float] = None

    # Validation details
    range_check_passed: bool = True
    rate_check_passed: bool = True
    consistency_check_passed: bool = True
    plausibility_check_passed: bool = True

    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "quality_code": self.quality_code.name,
            "issues": self.issues,
            "corrected_value": self.corrected_value,
            "range_check_passed": self.range_check_passed,
            "rate_check_passed": self.rate_check_passed,
            "consistency_check_passed": self.consistency_check_passed,
            "plausibility_check_passed": self.plausibility_check_passed,
        }


@dataclass
class QualifiedValue:
    """Sensor value with quality metadata."""
    tag: str
    raw_value: float
    value: float
    unit: str
    quality_code: QualityCode
    timestamp: datetime

    # Quality details
    validation_result: Optional[ValidationResult] = None

    # Transformation tracking
    calibration_applied: bool = False
    unit_converted: bool = False
    filter_applied: bool = False

    # Provenance
    source_hash: str = ""

    def to_dict(self) -> Dict:
        return {
            "tag": self.tag,
            "raw_value": self.raw_value,
            "value": self.value,
            "unit": self.unit,
            "quality_code": self.quality_code.name,
            "timestamp": self.timestamp.isoformat(),
            "is_good": self.quality_code.is_good(),
            "source_hash": self.source_hash,
        }


@dataclass
class TransformedData:
    """Batch of transformed sensor data."""
    values: Dict[str, QualifiedValue]
    timestamp: datetime
    total_count: int
    good_count: int
    uncertain_count: int
    bad_count: int

    # Overall quality score (0-100)
    quality_score: float = 100.0

    # Processing metadata
    processing_time_ms: float = 0.0
    transformations_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "values": {k: v.to_dict() for k, v in self.values.items()},
            "timestamp": self.timestamp.isoformat(),
            "total_count": self.total_count,
            "good_count": self.good_count,
            "uncertain_count": self.uncertain_count,
            "bad_count": self.bad_count,
            "quality_score": self.quality_score,
            "processing_time_ms": self.processing_time_ms,
        }


class UnitConverter:
    """
    Engineering unit converter for steam system measurements.

    Supports conversions between common industrial units:
    - Pressure: kPa, bar, psi, psig, psia, inHg, mmHg, atm
    - Temperature: C, F, K, R
    - Mass flow: kg/s, kg/hr, lb/hr, ton/hr, klb/hr
    - Volumetric flow: m3/s, m3/hr, gpm, scfm, acfm
    - Energy: J, kJ, MJ, BTU, kWh, therm
    - Power: W, kW, MW, hp, BTU/hr
    """

    # Unit definitions: (category, base_unit, conversion_to_base)
    UNITS = {
        # Pressure (base: Pa)
        "Pa": (UnitCategory.PRESSURE, "Pa", 1.0),
        "kPa": (UnitCategory.PRESSURE, "Pa", 1000.0),
        "MPa": (UnitCategory.PRESSURE, "Pa", 1e6),
        "bar": (UnitCategory.PRESSURE, "Pa", 1e5),
        "mbar": (UnitCategory.PRESSURE, "Pa", 100.0),
        "psi": (UnitCategory.PRESSURE, "Pa", 6894.76),
        "psig": (UnitCategory.PRESSURE, "Pa", 6894.76),  # Gauge, handled specially
        "psia": (UnitCategory.PRESSURE, "Pa", 6894.76),  # Absolute
        "inHg": (UnitCategory.PRESSURE, "Pa", 3386.39),
        "mmHg": (UnitCategory.PRESSURE, "Pa", 133.322),
        "atm": (UnitCategory.PRESSURE, "Pa", 101325.0),
        "inH2O": (UnitCategory.PRESSURE, "Pa", 249.089),

        # Temperature (base: K)
        "K": (UnitCategory.TEMPERATURE, "K", 1.0),
        "degC": (UnitCategory.TEMPERATURE, "K", 1.0),  # Offset: +273.15
        "C": (UnitCategory.TEMPERATURE, "K", 1.0),
        "degF": (UnitCategory.TEMPERATURE, "K", 5/9),  # Offset: (F-32)*5/9+273.15
        "F": (UnitCategory.TEMPERATURE, "K", 5/9),
        "R": (UnitCategory.TEMPERATURE, "K", 5/9),  # Rankine

        # Mass flow (base: kg/s)
        "kg/s": (UnitCategory.MASS_FLOW, "kg/s", 1.0),
        "kg/hr": (UnitCategory.MASS_FLOW, "kg/s", 1/3600),
        "kg/h": (UnitCategory.MASS_FLOW, "kg/s", 1/3600),
        "t/hr": (UnitCategory.MASS_FLOW, "kg/s", 1000/3600),
        "ton/hr": (UnitCategory.MASS_FLOW, "kg/s", 1000/3600),
        "lb/hr": (UnitCategory.MASS_FLOW, "kg/s", 0.453592/3600),
        "lb/h": (UnitCategory.MASS_FLOW, "kg/s", 0.453592/3600),
        "klb/hr": (UnitCategory.MASS_FLOW, "kg/s", 453.592/3600),
        "lb/s": (UnitCategory.MASS_FLOW, "kg/s", 0.453592),

        # Volumetric flow (base: m3/s)
        "m3/s": (UnitCategory.VOLUMETRIC_FLOW, "m3/s", 1.0),
        "m3/hr": (UnitCategory.VOLUMETRIC_FLOW, "m3/s", 1/3600),
        "m3/h": (UnitCategory.VOLUMETRIC_FLOW, "m3/s", 1/3600),
        "L/s": (UnitCategory.VOLUMETRIC_FLOW, "m3/s", 0.001),
        "L/min": (UnitCategory.VOLUMETRIC_FLOW, "m3/s", 0.001/60),
        "gpm": (UnitCategory.VOLUMETRIC_FLOW, "m3/s", 6.309e-5),
        "scfm": (UnitCategory.VOLUMETRIC_FLOW, "m3/s", 0.000472),
        "acfm": (UnitCategory.VOLUMETRIC_FLOW, "m3/s", 0.000472),

        # Energy (base: J)
        "J": (UnitCategory.ENERGY, "J", 1.0),
        "kJ": (UnitCategory.ENERGY, "J", 1000.0),
        "MJ": (UnitCategory.ENERGY, "J", 1e6),
        "GJ": (UnitCategory.ENERGY, "J", 1e9),
        "BTU": (UnitCategory.ENERGY, "J", 1055.06),
        "MMBTU": (UnitCategory.ENERGY, "J", 1055.06e6),
        "kWh": (UnitCategory.ENERGY, "J", 3.6e6),
        "MWh": (UnitCategory.ENERGY, "J", 3.6e9),
        "therm": (UnitCategory.ENERGY, "J", 1.055e8),

        # Power (base: W)
        "W": (UnitCategory.POWER, "W", 1.0),
        "kW": (UnitCategory.POWER, "W", 1000.0),
        "MW": (UnitCategory.POWER, "W", 1e6),
        "hp": (UnitCategory.POWER, "W", 745.7),
        "BTU/hr": (UnitCategory.POWER, "W", 0.293071),

        # Percentage
        "%": (UnitCategory.PERCENTAGE, "%", 1.0),
        "ppm": (UnitCategory.PERCENTAGE, "%", 0.0001),
        "ppb": (UnitCategory.PERCENTAGE, "%", 0.0000001),
    }

    # Atmospheric pressure for gauge/absolute conversions
    ATM_PA = 101325.0
    ATM_PSI = 14.696

    @classmethod
    def convert(
        cls,
        value: float,
        from_unit: str,
        to_unit: str,
    ) -> float:
        """
        Convert value between units.

        Args:
            value: Value in source units
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value

        Raises:
            ValueError: If units are incompatible or unknown
        """
        if from_unit == to_unit:
            return value

        # Handle temperature specially due to offsets
        if cls._is_temperature(from_unit) and cls._is_temperature(to_unit):
            return cls._convert_temperature(value, from_unit, to_unit)

        # Handle pressure gauge/absolute conversions
        if cls._is_pressure(from_unit) and cls._is_pressure(to_unit):
            return cls._convert_pressure(value, from_unit, to_unit)

        # Standard conversion via base unit
        if from_unit not in cls.UNITS or to_unit not in cls.UNITS:
            raise ValueError(f"Unknown unit: {from_unit} or {to_unit}")

        from_info = cls.UNITS[from_unit]
        to_info = cls.UNITS[to_unit]

        if from_info[0] != to_info[0]:
            raise ValueError(
                f"Cannot convert between {from_info[0].value} and {to_info[0].value}"
            )

        # Convert to base, then to target
        base_value = value * from_info[2]
        target_value = base_value / to_info[2]

        return target_value

    @classmethod
    def _is_temperature(cls, unit: str) -> bool:
        """Check if unit is temperature."""
        return unit in ["K", "C", "degC", "F", "degF", "R"]

    @classmethod
    def _is_pressure(cls, unit: str) -> bool:
        """Check if unit is pressure."""
        return unit in cls.UNITS and cls.UNITS[unit][0] == UnitCategory.PRESSURE

    @classmethod
    def _convert_temperature(cls, value: float, from_unit: str, to_unit: str) -> float:
        """Convert temperature with proper offset handling."""
        # Convert to Kelvin first
        kelvin: float

        if from_unit in ["C", "degC"]:
            kelvin = value + 273.15
        elif from_unit in ["F", "degF"]:
            kelvin = (value - 32) * 5/9 + 273.15
        elif from_unit == "R":
            kelvin = value * 5/9
        else:  # K
            kelvin = value

        # Convert from Kelvin to target
        if to_unit in ["C", "degC"]:
            return kelvin - 273.15
        elif to_unit in ["F", "degF"]:
            return (kelvin - 273.15) * 9/5 + 32
        elif to_unit == "R":
            return kelvin * 9/5
        else:  # K
            return kelvin

    @classmethod
    def _convert_pressure(cls, value: float, from_unit: str, to_unit: str) -> float:
        """Convert pressure with gauge/absolute handling."""
        # Convert to Pa (absolute)
        pa_abs: float

        from_factor = cls.UNITS[from_unit][2]

        if from_unit == "psig":
            # Gauge to absolute: add atmospheric
            pa_abs = (value + cls.ATM_PSI) * from_factor
        elif from_unit == "psia":
            pa_abs = value * from_factor
        else:
            pa_abs = value * from_factor

        # Convert to target unit
        to_factor = cls.UNITS[to_unit][2]

        if to_unit == "psig":
            # Absolute to gauge: subtract atmospheric
            return pa_abs / to_factor - cls.ATM_PSI
        elif to_unit == "psia":
            return pa_abs / to_factor
        else:
            return pa_abs / to_factor

    @classmethod
    def get_category(cls, unit: str) -> Optional[UnitCategory]:
        """Get unit category."""
        if unit in cls.UNITS:
            return cls.UNITS[unit][0]
        return None

    @classmethod
    def get_compatible_units(cls, unit: str) -> List[str]:
        """Get list of units compatible with given unit."""
        if unit not in cls.UNITS:
            return []

        category = cls.UNITS[unit][0]
        return [u for u, info in cls.UNITS.items() if info[0] == category]


class SensorTransformer:
    """
    Transforms raw sensor data for steam system optimization.

    Features:
    - Unit conversion between engineering units
    - Calibration application (linear, polynomial, lookup)
    - Range validation with configurable limits
    - Rate-of-change validation
    - Cross-sensor consistency checks
    - Quality code assignment (OPC-UA style)
    - Batch processing with quality scoring

    Example:
        transformer = SensorTransformer()

        # Configure sensor
        transformer.configure_sensor(
            tag="Header.Pressure",
            from_unit="psia",
            to_unit="bar",
            calibration=CalibrationParams(offset=-0.5, gain=1.02),
            valid_range=(0.0, 20.0),
            max_rate_of_change=5.0
        )

        # Transform value
        result = transformer.apply_calibration(raw_value=150.0, calibration_params)
        validated = transformer.validate_range(result, min_val=0, max_val=200)
        qualified = transformer.apply_quality_flags(result, quality_code)
    """

    def __init__(self) -> None:
        """Initialize sensor transformer."""
        self._sensor_configs: Dict[str, Dict] = {}
        self._last_values: Dict[str, Tuple[float, datetime]] = {}
        self._value_history: Dict[str, List[Tuple[float, datetime]]] = {}
        self._history_size = 100  # Keep last 100 values for analysis

        # Cross-sensor consistency rules
        self._consistency_rules: List[Dict] = []

        # Statistics
        self._stats = {
            "transformations": 0,
            "calibrations_applied": 0,
            "unit_conversions": 0,
            "range_violations": 0,
            "rate_violations": 0,
            "consistency_violations": 0,
        }

        logger.info("SensorTransformer initialized")

    def configure_sensor(
        self,
        tag: str,
        from_unit: str,
        to_unit: str,
        calibration: Optional[CalibrationParams] = None,
        valid_range: Optional[Tuple[float, float]] = None,
        max_rate_of_change: Optional[float] = None,
        stale_timeout_s: float = 60.0,
        description: str = "",
    ) -> None:
        """
        Configure transformation for a sensor tag.

        Args:
            tag: Sensor tag name
            from_unit: Raw value unit
            to_unit: Target engineering unit
            calibration: Calibration parameters
            valid_range: (min, max) valid range in target units
            max_rate_of_change: Maximum change per second
            stale_timeout_s: Timeout for stale data detection
            description: Sensor description
        """
        self._sensor_configs[tag] = {
            "from_unit": from_unit,
            "to_unit": to_unit,
            "calibration": calibration,
            "valid_range": valid_range,
            "max_rate_of_change": max_rate_of_change,
            "stale_timeout_s": stale_timeout_s,
            "description": description,
        }

        logger.debug(f"Configured sensor: {tag} ({from_unit} -> {to_unit})")

    def add_consistency_rule(
        self,
        name: str,
        tags: List[str],
        check_function: Callable[[Dict[str, float]], bool],
        description: str = "",
    ) -> None:
        """
        Add cross-sensor consistency rule.

        Args:
            name: Rule name
            tags: Tags involved in the check
            check_function: Function that returns True if consistent
            description: Rule description

        Example:
            # Outlet temp must be less than inlet temp (energy loss)
            transformer.add_consistency_rule(
                "heat_exchanger_temp",
                ["inlet_temp", "outlet_temp"],
                lambda v: v["outlet_temp"] < v["inlet_temp"],
                "Outlet temperature must be less than inlet"
            )
        """
        self._consistency_rules.append({
            "name": name,
            "tags": tags,
            "check": check_function,
            "description": description,
        })

    def normalize_units(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
    ) -> float:
        """
        Convert value between engineering units.

        Args:
            value: Value in source units
            from_unit: Source unit (e.g., "psia", "degF", "lb/hr")
            to_unit: Target unit (e.g., "bar", "degC", "kg/s")

        Returns:
            Converted value

        Raises:
            ValueError: If conversion not possible
        """
        self._stats["unit_conversions"] += 1
        return UnitConverter.convert(value, from_unit, to_unit)

    def apply_calibration(
        self,
        raw_value: float,
        calibration_params: CalibrationParams,
    ) -> float:
        """
        Apply calibration to raw sensor value.

        Args:
            raw_value: Uncalibrated sensor reading
            calibration_params: Calibration parameters

        Returns:
            Calibrated value
        """
        self._stats["calibrations_applied"] += 1

        # Polynomial calibration takes precedence
        if calibration_params.polynomial_coefficients:
            result = 0.0
            for i, coef in enumerate(calibration_params.polynomial_coefficients):
                result += coef * (raw_value ** i)
            return result

        # Lookup table interpolation
        if calibration_params.lookup_table:
            return self._interpolate_lookup(
                raw_value,
                calibration_params.lookup_table
            )

        # Linear calibration: y = gain * x + offset
        result = calibration_params.gain * raw_value + calibration_params.offset

        # Temperature compensation
        # (Would need ambient temperature input in production)

        return result

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

        # Find surrounding points for interpolation
        for i in range(len(keys) - 1):
            if keys[i] <= value <= keys[i + 1]:
                x0, x1 = keys[i], keys[i + 1]
                y0, y1 = table[x0], table[x1]

                # Linear interpolation
                return y0 + (y1 - y0) * (value - x0) / (x1 - x0)

        return value

    def validate_range(
        self,
        value: float,
        min_val: float,
        max_val: float,
    ) -> ValidationResult:
        """
        Validate value is within acceptable range.

        Args:
            value: Value to validate
            min_val: Minimum acceptable value
            max_val: Maximum acceptable value

        Returns:
            ValidationResult with quality code and issues
        """
        issues: List[str] = []
        quality_code = QualityCode.GOOD

        if value < min_val:
            self._stats["range_violations"] += 1
            issues.append(f"Value {value:.4f} below minimum {min_val:.4f}")
            quality_code = QualityCode.BAD_OUT_OF_RANGE

        elif value > max_val:
            self._stats["range_violations"] += 1
            issues.append(f"Value {value:.4f} above maximum {max_val:.4f}")
            quality_code = QualityCode.BAD_OUT_OF_RANGE

        # Check for warning threshold (within 10% of limits)
        elif value < min_val + 0.1 * (max_val - min_val):
            issues.append(f"Value {value:.4f} near lower limit {min_val:.4f}")
            quality_code = QualityCode.UNCERTAIN_EU_EXCEEDED

        elif value > max_val - 0.1 * (max_val - min_val):
            issues.append(f"Value {value:.4f} near upper limit {max_val:.4f}")
            quality_code = QualityCode.UNCERTAIN_EU_EXCEEDED

        return ValidationResult(
            is_valid=quality_code.is_good() or quality_code.is_uncertain(),
            quality_code=quality_code,
            issues=issues,
            range_check_passed=quality_code.is_good(),
        )

    def validate_rate_of_change(
        self,
        tag: str,
        value: float,
        timestamp: datetime,
        max_rate: float,
    ) -> ValidationResult:
        """
        Validate rate of change is within acceptable limits.

        Args:
            tag: Sensor tag for history lookup
            value: Current value
            timestamp: Current timestamp
            max_rate: Maximum acceptable change per second

        Returns:
            ValidationResult
        """
        issues: List[str] = []
        quality_code = QualityCode.GOOD

        if tag in self._last_values:
            last_value, last_time = self._last_values[tag]
            dt = (timestamp - last_time).total_seconds()

            if dt > 0:
                rate = abs(value - last_value) / dt

                if rate > max_rate:
                    self._stats["rate_violations"] += 1
                    issues.append(
                        f"Rate of change {rate:.4f}/s exceeds limit {max_rate:.4f}/s"
                    )
                    quality_code = QualityCode.UNCERTAIN_SENSOR_NOT_ACCURATE

        # Update history
        self._last_values[tag] = (value, timestamp)

        # Update detailed history
        if tag not in self._value_history:
            self._value_history[tag] = []
        self._value_history[tag].append((value, timestamp))
        if len(self._value_history[tag]) > self._history_size:
            self._value_history[tag].pop(0)

        return ValidationResult(
            is_valid=quality_code.is_good() or quality_code.is_uncertain(),
            quality_code=quality_code,
            issues=issues,
            rate_check_passed=quality_code.is_good(),
        )

    def check_consistency(
        self,
        values: Dict[str, float],
    ) -> ValidationResult:
        """
        Check cross-sensor consistency.

        Args:
            values: Dict of tag -> value

        Returns:
            ValidationResult
        """
        issues: List[str] = []
        quality_code = QualityCode.GOOD

        for rule in self._consistency_rules:
            # Check if all required tags present
            if not all(tag in values for tag in rule["tags"]):
                continue

            # Extract relevant values
            rule_values = {tag: values[tag] for tag in rule["tags"]}

            try:
                if not rule["check"](rule_values):
                    self._stats["consistency_violations"] += 1
                    issues.append(f"Consistency rule '{rule['name']}' failed: {rule['description']}")
                    quality_code = QualityCode.UNCERTAIN_SENSOR_NOT_ACCURATE
            except Exception as e:
                logger.warning(f"Error in consistency check '{rule['name']}': {e}")

        return ValidationResult(
            is_valid=quality_code.is_good() or quality_code.is_uncertain(),
            quality_code=quality_code,
            issues=issues,
            consistency_check_passed=quality_code.is_good(),
        )

    def apply_quality_flags(
        self,
        value: float,
        quality_code: QualityCode,
        tag: str = "",
        unit: str = "",
        raw_value: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> QualifiedValue:
        """
        Create qualified value with quality flags.

        Args:
            value: Transformed value
            quality_code: Quality code to apply
            tag: Sensor tag name
            unit: Engineering unit
            raw_value: Original raw value
            timestamp: Value timestamp

        Returns:
            QualifiedValue with quality metadata
        """
        ts = timestamp or datetime.now(timezone.utc)

        # Compute provenance hash
        hash_data = f"{tag}:{value}:{ts.isoformat()}"
        source_hash = hashlib.sha256(hash_data.encode()).hexdigest()[:16]

        return QualifiedValue(
            tag=tag,
            raw_value=raw_value if raw_value is not None else value,
            value=round(value, 6),
            unit=unit,
            quality_code=quality_code,
            timestamp=ts,
            source_hash=source_hash,
        )

    def transform_single(
        self,
        tag: str,
        raw_value: float,
        timestamp: Optional[datetime] = None,
    ) -> QualifiedValue:
        """
        Transform single sensor value using configured rules.

        Args:
            tag: Sensor tag
            raw_value: Raw sensor value
            timestamp: Value timestamp

        Returns:
            QualifiedValue with all transformations applied
        """
        ts = timestamp or datetime.now(timezone.utc)
        self._stats["transformations"] += 1

        # Get configuration
        config = self._sensor_configs.get(tag, {})

        # Apply calibration
        value = raw_value
        calibration_applied = False
        if config.get("calibration"):
            value = self.apply_calibration(value, config["calibration"])
            calibration_applied = True

        # Apply unit conversion
        unit_converted = False
        from_unit = config.get("from_unit", "")
        to_unit = config.get("to_unit", from_unit)
        if from_unit and to_unit and from_unit != to_unit:
            value = self.normalize_units(value, from_unit, to_unit)
            unit_converted = True

        # Validate range
        quality_code = QualityCode.GOOD
        validation_result = None
        issues: List[str] = []

        if config.get("valid_range"):
            min_val, max_val = config["valid_range"]
            range_result = self.validate_range(value, min_val, max_val)
            issues.extend(range_result.issues)
            if not range_result.is_valid:
                quality_code = range_result.quality_code
            validation_result = range_result

        # Validate rate of change
        if config.get("max_rate_of_change"):
            rate_result = self.validate_rate_of_change(
                tag, value, ts, config["max_rate_of_change"]
            )
            issues.extend(rate_result.issues)
            if not rate_result.is_valid and quality_code.is_good():
                quality_code = rate_result.quality_code

        # Create qualified value
        result = QualifiedValue(
            tag=tag,
            raw_value=raw_value,
            value=round(value, 6),
            unit=to_unit,
            quality_code=quality_code,
            timestamp=ts,
            validation_result=validation_result,
            calibration_applied=calibration_applied,
            unit_converted=unit_converted,
            source_hash=hashlib.sha256(
                f"{tag}:{value}:{ts.isoformat()}".encode()
            ).hexdigest()[:16],
        )

        return result

    def transform_batch(
        self,
        raw_data: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> TransformedData:
        """
        Transform batch of sensor values.

        Args:
            raw_data: Dict of tag -> raw_value
            timestamp: Common timestamp for batch

        Returns:
            TransformedData with all values and quality statistics
        """
        import time
        start_time = time.perf_counter()

        ts = timestamp or datetime.now(timezone.utc)
        values: Dict[str, QualifiedValue] = {}

        # Transform each value
        for tag, raw_value in raw_data.items():
            values[tag] = self.transform_single(tag, raw_value, ts)

        # Cross-sensor consistency check
        transformed_values = {tag: v.value for tag, v in values.items()}
        consistency_result = self.check_consistency(transformed_values)

        if not consistency_result.is_valid:
            # Downgrade quality for affected sensors
            for tag in values:
                if values[tag].quality_code.is_good():
                    values[tag].quality_code = QualityCode.UNCERTAIN_SENSOR_NOT_ACCURATE

        # Calculate statistics
        good_count = sum(1 for v in values.values() if v.quality_code.is_good())
        uncertain_count = sum(1 for v in values.values() if v.quality_code.is_uncertain())
        bad_count = sum(1 for v in values.values() if v.quality_code.is_bad())
        total_count = len(values)

        # Quality score (0-100)
        quality_score = 100.0
        if total_count > 0:
            quality_score = (good_count * 100 + uncertain_count * 50) / total_count

        processing_time = (time.perf_counter() - start_time) * 1000

        return TransformedData(
            values=values,
            timestamp=ts,
            total_count=total_count,
            good_count=good_count,
            uncertain_count=uncertain_count,
            bad_count=bad_count,
            quality_score=round(quality_score, 2),
            processing_time_ms=round(processing_time, 3),
            transformations_applied=["calibration", "unit_conversion", "validation"],
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get transformation statistics."""
        return {
            **self._stats,
            "configured_sensors": len(self._sensor_configs),
            "consistency_rules": len(self._consistency_rules),
        }

    def get_value_history(
        self,
        tag: str,
        limit: int = 100,
    ) -> List[Tuple[float, datetime]]:
        """Get historical values for a tag."""
        return self._value_history.get(tag, [])[-limit:]


def create_steam_system_transformer() -> SensorTransformer:
    """Create transformer configured for steam system monitoring."""
    transformer = SensorTransformer()

    # Configure common steam system sensors

    # Pressure sensors
    transformer.configure_sensor(
        tag="header.pressure",
        from_unit="psig",
        to_unit="bar",
        calibration=CalibrationParams(offset=0.0, gain=1.0),
        valid_range=(0.0, 20.0),
        max_rate_of_change=2.0,
        description="Main steam header pressure",
    )

    transformer.configure_sensor(
        tag="header.pressure_psig",
        from_unit="psig",
        to_unit="psig",
        valid_range=(0.0, 300.0),
        max_rate_of_change=30.0,
        description="Main steam header pressure (psig)",
    )

    # Temperature sensors
    transformer.configure_sensor(
        tag="header.temperature",
        from_unit="degF",
        to_unit="degC",
        valid_range=(100.0, 400.0),
        max_rate_of_change=10.0,
        description="Main steam header temperature",
    )

    # Flow sensors
    transformer.configure_sensor(
        tag="header.flow",
        from_unit="klb/hr",
        to_unit="kg/s",
        valid_range=(0.0, 50.0),
        max_rate_of_change=5.0,
        description="Main steam header flow",
    )

    # Add consistency rules
    transformer.add_consistency_rule(
        name="superheat_check",
        tags=["header.temperature", "header.saturation_temp"],
        check_function=lambda v: v.get("header.temperature", 0) >= v.get("header.saturation_temp", 0),
        description="Steam temperature must be at or above saturation",
    )

    return transformer
