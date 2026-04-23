# -*- coding: utf-8 -*-
"""
GL-014 ExchangerPro - Tag Mapping Module

Canonical tag schema mapping for heat exchangers:
- Exchanger-specific tag schema (shell/tube, inlet/outlet)
- Site-specific tag translation
- Unit normalization (temperature, pressure, flow)
- Tag validation

Canonical Naming Convention for Heat Exchangers:
    {site}.{area}.{exchanger}.{side}.{location}.{measurement}

Examples:
    - plant1.area2.hx001.shell.inlet.temperature
    - plant1.area2.hx001.tube.outlet.pressure
    - plant1.area2.hx001.shell.flow

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class MeasurementType(str, Enum):
    """Heat exchanger measurement types."""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW = "flow"
    DIFFERENTIAL_PRESSURE = "differential_pressure"
    LEVEL = "level"
    DENSITY = "density"
    VISCOSITY = "viscosity"
    FOULING_FACTOR = "fouling_factor"
    HEAT_DUTY = "heat_duty"
    HTC = "htc"


class ExchangerSide(str, Enum):
    """Heat exchanger sides."""
    SHELL = "shell"
    TUBE = "tube"
    BOTH = "both"


class ExchangerLocation(str, Enum):
    """Measurement location on exchanger."""
    INLET = "inlet"
    OUTLET = "outlet"
    MID = "mid"
    OVERALL = "overall"


class TemperatureUnit(str, Enum):
    """Temperature units."""
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"
    KELVIN = "kelvin"
    RANKINE = "rankine"


class PressureUnit(str, Enum):
    """Pressure units."""
    PASCAL = "Pa"
    KILOPASCAL = "kPa"
    MEGAPASCAL = "MPa"
    BAR = "bar"
    MBAR = "mbar"
    PSI = "psi"
    PSIG = "psig"
    ATM = "atm"
    MMHG = "mmHg"
    INHG = "inHg"


class FlowUnit(str, Enum):
    """Flow units."""
    KG_PER_S = "kg/s"
    KG_PER_H = "kg/h"
    T_PER_H = "t/h"
    LB_PER_H = "lb/h"
    LB_PER_S = "lb/s"
    M3_PER_H = "m3/h"
    GAL_PER_MIN = "gpm"
    L_PER_MIN = "L/min"


class TagQuality(str, Enum):
    """Tag data quality."""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    SUBSTITUTED = "substituted"


# =============================================================================
# ENGINEERING UNITS
# =============================================================================

class EngineeringUnit(BaseModel):
    """Engineering unit definition."""
    unit_id: str = Field(..., description="Unit identifier")
    display_name: str = Field(..., description="Display name")
    symbol: str = Field(..., description="Unit symbol")
    measurement_type: MeasurementType = Field(..., description="Measurement type")
    is_si: bool = Field(default=False, description="Is SI unit")

    # Conversion to SI base
    conversion_factor: float = Field(
        default=1.0,
        description="Multiply to convert to SI"
    )
    conversion_offset: float = Field(
        default=0.0,
        description="Add after multiply for conversion"
    )


# Standard engineering units for heat exchangers
ENGINEERING_UNITS: Dict[str, EngineeringUnit] = {
    # Temperature
    "celsius": EngineeringUnit(
        unit_id="celsius",
        display_name="Celsius",
        symbol="C",
        measurement_type=MeasurementType.TEMPERATURE,
        is_si=True,
    ),
    "fahrenheit": EngineeringUnit(
        unit_id="fahrenheit",
        display_name="Fahrenheit",
        symbol="F",
        measurement_type=MeasurementType.TEMPERATURE,
        conversion_factor=5/9,
        conversion_offset=-32 * 5/9,
    ),
    "kelvin": EngineeringUnit(
        unit_id="kelvin",
        display_name="Kelvin",
        symbol="K",
        measurement_type=MeasurementType.TEMPERATURE,
        conversion_factor=1,
        conversion_offset=-273.15,
    ),

    # Pressure
    "kPa": EngineeringUnit(
        unit_id="kPa",
        display_name="Kilopascal",
        symbol="kPa",
        measurement_type=MeasurementType.PRESSURE,
        is_si=True,
    ),
    "bar": EngineeringUnit(
        unit_id="bar",
        display_name="Bar",
        symbol="bar",
        measurement_type=MeasurementType.PRESSURE,
        conversion_factor=100,
    ),
    "psi": EngineeringUnit(
        unit_id="psi",
        display_name="Pounds per square inch",
        symbol="psi",
        measurement_type=MeasurementType.PRESSURE,
        conversion_factor=6.89476,
    ),

    # Flow
    "kg/s": EngineeringUnit(
        unit_id="kg/s",
        display_name="Kilograms per second",
        symbol="kg/s",
        measurement_type=MeasurementType.FLOW,
        is_si=True,
    ),
    "kg/h": EngineeringUnit(
        unit_id="kg/h",
        display_name="Kilograms per hour",
        symbol="kg/h",
        measurement_type=MeasurementType.FLOW,
        conversion_factor=1/3600,
    ),
    "t/h": EngineeringUnit(
        unit_id="t/h",
        display_name="Tonnes per hour",
        symbol="t/h",
        measurement_type=MeasurementType.FLOW,
        conversion_factor=1000/3600,
    ),
}


# =============================================================================
# CANONICAL TAG NAMING
# =============================================================================

class CanonicalTagName(BaseModel):
    """
    Parsed canonical tag name for heat exchangers.

    Format: {site}.{area}.{exchanger}.{side}.{location}.{measurement}

    Examples:
        plant1.area2.hx001.shell.inlet.temperature
        plant1.area2.hx001.tube.outlet.flow
    """
    site: str = Field(..., description="Site identifier")
    area: str = Field(..., description="Plant area")
    exchanger: str = Field(..., description="Exchanger identifier")
    side: ExchangerSide = Field(..., description="Shell or tube side")
    location: ExchangerLocation = Field(..., description="Inlet/outlet/mid")
    measurement: MeasurementType = Field(..., description="Measurement type")
    qualifier: Optional[str] = Field(None, description="Optional qualifier")

    @classmethod
    def parse(cls, canonical_name: str) -> "CanonicalTagName":
        """Parse canonical name string."""
        parts = canonical_name.lower().split(".")

        if len(parts) < 6:
            raise ValueError(
                f"Canonical name must have at least 6 parts: {canonical_name}"
            )

        return cls(
            site=parts[0],
            area=parts[1],
            exchanger=parts[2],
            side=ExchangerSide(parts[3]),
            location=ExchangerLocation(parts[4]),
            measurement=MeasurementType(parts[5]),
            qualifier=parts[6] if len(parts) > 6 else None,
        )

    def to_string(self) -> str:
        """Convert to canonical name string."""
        base = f"{self.site}.{self.area}.{self.exchanger}.{self.side.value}.{self.location.value}.{self.measurement.value}"
        if self.qualifier:
            return f"{base}.{self.qualifier}"
        return base

    @validator("site", "area", "exchanger")
    def validate_identifier(cls, v, field):
        """Validate identifier format."""
        if not v:
            raise ValueError(f"{field.name} cannot be empty")
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", v):
            raise ValueError(
                f"{field.name} must start with letter and contain only "
                f"alphanumeric characters and underscores: {v}"
            )
        return v.lower()

    def matches_pattern(self, pattern: str) -> bool:
        """
        Check if name matches pattern with wildcards.

        Wildcards:
            * - matches single part
            ** - matches multiple parts
        """
        name_parts = self.to_string().split(".")
        pattern_parts = pattern.lower().split(".")

        i, j = 0, 0
        while i < len(name_parts) and j < len(pattern_parts):
            if pattern_parts[j] == "**":
                if j == len(pattern_parts) - 1:
                    return True
                j += 1
                while i < len(name_parts):
                    if name_parts[i] == pattern_parts[j]:
                        break
                    i += 1
            elif pattern_parts[j] == "*" or pattern_parts[j] == name_parts[i]:
                i += 1
                j += 1
            else:
                return False

        return i == len(name_parts) and j == len(pattern_parts)


# =============================================================================
# TAG MAPPING ENTRY
# =============================================================================

class TagMappingEntry(BaseModel):
    """Single tag mapping entry."""
    # Identifiers
    canonical_name: str = Field(..., description="Canonical tag name")
    vendor_tag: str = Field(..., description="Vendor/PLC tag address")
    node_id: str = Field(..., description="OPC-UA node ID")

    # Display
    display_name: str = Field(..., description="Human-readable name")
    description: Optional[str] = Field(None, description="Description")

    # Type
    measurement_type: MeasurementType = Field(..., description="Measurement type")
    data_type: str = Field(default="Double", description="Data type")

    # Exchanger context
    exchanger_id: str = Field(..., description="Exchanger ID")
    side: ExchangerSide = Field(..., description="Shell/Tube")
    location: ExchangerLocation = Field(..., description="Inlet/Outlet")

    # Engineering units
    source_unit: str = Field(..., description="Source unit")
    target_unit: str = Field(..., description="Target (canonical) unit")

    # Scaling
    raw_low: Optional[float] = Field(None, description="Raw value low")
    raw_high: Optional[float] = Field(None, description="Raw value high")
    eng_low: Optional[float] = Field(None, description="Engineering low")
    eng_high: Optional[float] = Field(None, description="Engineering high")

    # Validation
    valid_range_low: Optional[float] = Field(None, description="Valid minimum")
    valid_range_high: Optional[float] = Field(None, description="Valid maximum")
    rate_of_change_limit: Optional[float] = Field(
        None,
        description="Max rate of change per second"
    )

    # Versioning
    version: str = Field(default="1.0.0", description="Mapping version")
    effective_date: Optional[datetime] = Field(None, description="Effective date")

    def apply_scaling(self, raw_value: float) -> float:
        """Apply scaling to convert raw to engineering value."""
        if None in (self.raw_low, self.raw_high, self.eng_low, self.eng_high):
            return raw_value

        raw_range = self.raw_high - self.raw_low
        if raw_range == 0:
            return self.eng_low

        eng_range = self.eng_high - self.eng_low
        return self.eng_low + ((raw_value - self.raw_low) / raw_range * eng_range)

    def is_in_valid_range(self, value: float) -> bool:
        """Check if value is in valid range."""
        if self.valid_range_low is not None and value < self.valid_range_low:
            return False
        if self.valid_range_high is not None and value > self.valid_range_high:
            return False
        return True


# =============================================================================
# EXCHANGER TAG SCHEMA
# =============================================================================

class ExchangerTagSchema(BaseModel):
    """
    Complete tag schema for a heat exchanger.

    Defines all expected tags for an exchanger type.
    """
    schema_id: str = Field(
        default_factory=lambda: str(__import__("uuid").uuid4()),
        description="Schema ID"
    )
    schema_name: str = Field(..., description="Schema name")
    exchanger_type: str = Field(..., description="Exchanger type")

    # Required tags
    required_tags: List[str] = Field(
        default_factory=lambda: [
            "shell.inlet.temperature",
            "shell.outlet.temperature",
            "tube.inlet.temperature",
            "tube.outlet.temperature",
            "shell.flow",
            "tube.flow",
        ],
        description="Required tag patterns"
    )

    # Optional tags
    optional_tags: List[str] = Field(
        default_factory=lambda: [
            "shell.inlet.pressure",
            "shell.outlet.pressure",
            "tube.inlet.pressure",
            "tube.outlet.pressure",
            "shell.differential_pressure",
            "tube.differential_pressure",
        ],
        description="Optional tag patterns"
    )

    # Default units
    default_temperature_unit: TemperatureUnit = Field(
        default=TemperatureUnit.CELSIUS,
        description="Default temperature unit"
    )
    default_pressure_unit: PressureUnit = Field(
        default=PressureUnit.KILOPASCAL,
        description="Default pressure unit"
    )
    default_flow_unit: FlowUnit = Field(
        default=FlowUnit.KG_PER_S,
        description="Default flow unit"
    )

    # Version
    version: str = Field(default="1.0.0", description="Schema version")

    def validate_tags(self, tags: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate if provided tags satisfy schema requirements.

        Returns:
            Tuple of (is_valid, missing_tags)
        """
        missing = []

        for required in self.required_tags:
            pattern = f"*.*.*.{required}"
            found = False
            for tag in tags:
                try:
                    canonical = CanonicalTagName.parse(tag)
                    if canonical.matches_pattern(pattern):
                        found = True
                        break
                except:
                    continue

            if not found:
                missing.append(required)

        return len(missing) == 0, missing


# =============================================================================
# SITE TAG TRANSLATION
# =============================================================================

class SiteTagTranslation(BaseModel):
    """
    Site-specific tag translation rules.

    Maps vendor-specific tag naming to canonical names.
    """
    site_id: str = Field(..., description="Site identifier")
    site_name: str = Field(..., description="Site name")

    # Vendor information
    vendor_system: str = Field(..., description="DCS/PLC vendor")
    tag_naming_convention: str = Field(..., description="Vendor naming convention")

    # Translation rules
    tag_prefix_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Prefix translations"
    )
    tag_suffix_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Suffix translations"
    )

    # Regex patterns
    pattern_rules: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Regex pattern rules"
    )

    # Direct mappings
    direct_mappings: Dict[str, str] = Field(
        default_factory=dict,
        description="Direct vendor to canonical mappings"
    )

    def translate(self, vendor_tag: str) -> Optional[str]:
        """
        Translate vendor tag to canonical name.

        Args:
            vendor_tag: Vendor-specific tag

        Returns:
            Canonical name or None if no translation
        """
        # Check direct mappings first
        if vendor_tag in self.direct_mappings:
            return self.direct_mappings[vendor_tag]

        # Apply prefix translations
        translated = vendor_tag
        for prefix, replacement in self.tag_prefix_map.items():
            if translated.startswith(prefix):
                translated = replacement + translated[len(prefix):]
                break

        # Apply suffix translations
        for suffix, replacement in self.tag_suffix_map.items():
            if translated.endswith(suffix):
                translated = translated[:-len(suffix)] + replacement
                break

        # Apply regex patterns
        for rule in self.pattern_rules:
            pattern = rule.get("pattern", "")
            replacement = rule.get("replacement", "")
            if pattern:
                translated = re.sub(pattern, replacement, translated)

        return translated if translated != vendor_tag else None


# =============================================================================
# TAG MAPPING CONFIGURATION
# =============================================================================

class TagMappingConfig(BaseModel):
    """Complete tag mapping configuration."""
    config_id: str = Field(..., description="Configuration ID")
    config_name: str = Field(..., description="Configuration name")
    version: str = Field(..., description="Configuration version")

    # Site
    site_id: str = Field(..., description="Site identifier")
    site_name: str = Field(..., description="Site name")

    # Mappings
    mappings: List[TagMappingEntry] = Field(
        default_factory=list,
        description="Tag mappings"
    )

    # Site translation
    site_translation: Optional[SiteTagTranslation] = Field(
        None,
        description="Site translation rules"
    )

    # Schemas
    exchanger_schemas: List[ExchangerTagSchema] = Field(
        default_factory=list,
        description="Exchanger schemas"
    )

    # Audit
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    author: Optional[str] = Field(None, description="Author")
    checksum: Optional[str] = Field(None, description="SHA-256 checksum")

    @classmethod
    def load_from_yaml(cls, path: Path) -> "TagMappingConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        config = cls(**data)
        config.checksum = cls._calculate_checksum(data)
        return config

    @classmethod
    def load_from_json(cls, path: Path) -> "TagMappingConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        config = cls(**data)
        config.checksum = cls._calculate_checksum(data)
        return config

    @staticmethod
    def _calculate_checksum(data: Dict) -> str:
        """Calculate SHA-256 checksum."""
        data_copy = {k: v for k, v in data.items() if k != "checksum"}
        canonical = json.dumps(data_copy, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def get_mapping(self, canonical_name: str) -> Optional[TagMappingEntry]:
        """Get mapping by canonical name."""
        for mapping in self.mappings:
            if mapping.canonical_name.lower() == canonical_name.lower():
                return mapping
        return None

    def get_mappings_by_exchanger(self, exchanger_id: str) -> List[TagMappingEntry]:
        """Get all mappings for an exchanger."""
        return [
            m for m in self.mappings
            if m.exchanger_id.lower() == exchanger_id.lower()
        ]


# =============================================================================
# UNIT NORMALIZER
# =============================================================================

class UnitNormalizer:
    """
    Engineering unit normalizer for heat exchanger measurements.

    Converts between various units for temperature, pressure, and flow.
    """

    # Temperature conversions (to Celsius)
    @staticmethod
    def temperature_to_celsius(value: float, from_unit: str) -> float:
        """Convert temperature to Celsius."""
        from_unit = from_unit.lower()

        if from_unit in ("celsius", "c", "degc"):
            return value
        elif from_unit in ("fahrenheit", "f", "degf"):
            return (value - 32) * 5 / 9
        elif from_unit in ("kelvin", "k"):
            return value - 273.15
        elif from_unit in ("rankine", "r", "degr"):
            return (value - 491.67) * 5 / 9
        else:
            logger.warning(f"Unknown temperature unit: {from_unit}")
            return value

    @staticmethod
    def celsius_to_unit(value: float, to_unit: str) -> float:
        """Convert Celsius to target unit."""
        to_unit = to_unit.lower()

        if to_unit in ("celsius", "c", "degc"):
            return value
        elif to_unit in ("fahrenheit", "f", "degf"):
            return value * 9 / 5 + 32
        elif to_unit in ("kelvin", "k"):
            return value + 273.15
        elif to_unit in ("rankine", "r", "degr"):
            return (value + 273.15) * 9 / 5
        else:
            logger.warning(f"Unknown temperature unit: {to_unit}")
            return value

    # Pressure conversions (to kPa)
    PRESSURE_TO_KPA: Dict[str, float] = {
        "pa": 0.001,
        "kpa": 1.0,
        "mpa": 1000.0,
        "bar": 100.0,
        "mbar": 0.1,
        "psi": 6.89476,
        "psig": 6.89476,
        "atm": 101.325,
        "mmhg": 0.133322,
        "inhg": 3.38639,
    }

    @classmethod
    def pressure_to_kpa(cls, value: float, from_unit: str) -> float:
        """Convert pressure to kPa."""
        factor = cls.PRESSURE_TO_KPA.get(from_unit.lower(), 1.0)
        return value * factor

    @classmethod
    def kpa_to_unit(cls, value: float, to_unit: str) -> float:
        """Convert kPa to target unit."""
        factor = cls.PRESSURE_TO_KPA.get(to_unit.lower(), 1.0)
        return value / factor

    # Flow conversions (to kg/s)
    FLOW_TO_KG_S: Dict[str, float] = {
        "kg/s": 1.0,
        "kg/h": 1/3600,
        "t/h": 1000/3600,
        "lb/h": 0.453592/3600,
        "lb/s": 0.453592,
        "m3/h": 1000/3600,  # Assumes water density
        "gpm": 3.78541 * 1000 / 60 / 3600,  # Gallons per minute
        "l/min": 1000 / 60 / 3600,
    }

    @classmethod
    def flow_to_kg_s(cls, value: float, from_unit: str) -> float:
        """Convert flow to kg/s."""
        factor = cls.FLOW_TO_KG_S.get(from_unit.lower().replace(" ", ""), 1.0)
        return value * factor

    @classmethod
    def kg_s_to_unit(cls, value: float, to_unit: str) -> float:
        """Convert kg/s to target unit."""
        factor = cls.FLOW_TO_KG_S.get(to_unit.lower().replace(" ", ""), 1.0)
        return value / factor

    @classmethod
    def convert(
        cls,
        value: float,
        from_unit: str,
        to_unit: str,
        measurement_type: MeasurementType,
    ) -> float:
        """
        Convert value between units.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit
            measurement_type: Type of measurement

        Returns:
            Converted value
        """
        if from_unit.lower() == to_unit.lower():
            return value

        if measurement_type == MeasurementType.TEMPERATURE:
            celsius = cls.temperature_to_celsius(value, from_unit)
            return cls.celsius_to_unit(celsius, to_unit)

        elif measurement_type == MeasurementType.PRESSURE:
            kpa = cls.pressure_to_kpa(value, from_unit)
            return cls.kpa_to_unit(kpa, to_unit)

        elif measurement_type == MeasurementType.FLOW:
            kg_s = cls.flow_to_kg_s(value, from_unit)
            return cls.kg_s_to_unit(kg_s, to_unit)

        else:
            logger.warning(
                f"No conversion available for {measurement_type}"
            )
            return value


# =============================================================================
# TAG VALIDATOR
# =============================================================================

class TagValidationResult(BaseModel):
    """Result of tag validation."""
    tag_id: str = Field(..., description="Tag ID")
    is_valid: bool = Field(..., description="Validation passed")
    value: Any = Field(..., description="Validated value")
    quality: TagQuality = Field(default=TagQuality.GOOD, description="Quality")
    errors: List[str] = Field(default_factory=list, description="Errors")
    warnings: List[str] = Field(default_factory=list, description="Warnings")

    # Validation details
    range_check_passed: bool = Field(default=True, description="Range check")
    rate_check_passed: bool = Field(default=True, description="Rate check")
    type_check_passed: bool = Field(default=True, description="Type check")


class TagValidator:
    """
    Tag value validator for heat exchanger measurements.

    Validates:
    - Value range
    - Rate of change
    - Data type
    - Quality
    """

    def __init__(self, config: Optional[TagMappingConfig] = None):
        """Initialize validator."""
        self.config = config
        self._last_values: Dict[str, Tuple[float, datetime]] = {}

    def validate(
        self,
        tag_id: str,
        value: Any,
        timestamp: datetime,
        mapping: Optional[TagMappingEntry] = None,
    ) -> TagValidationResult:
        """
        Validate a tag value.

        Args:
            tag_id: Tag identifier
            value: Value to validate
            timestamp: Value timestamp
            mapping: Optional tag mapping

        Returns:
            Validation result
        """
        result = TagValidationResult(
            tag_id=tag_id,
            is_valid=True,
            value=value,
        )

        # Get mapping from config if not provided
        if mapping is None and self.config:
            for m in self.config.mappings:
                if m.canonical_name == tag_id or m.vendor_tag == tag_id:
                    mapping = m
                    break

        # Type check
        if mapping:
            if not self._validate_type(value, mapping.data_type):
                result.type_check_passed = False
                result.errors.append(
                    f"Type mismatch: expected {mapping.data_type}"
                )

        # Range check
        if mapping and isinstance(value, (int, float)):
            if not mapping.is_in_valid_range(value):
                result.range_check_passed = False
                result.errors.append(
                    f"Value {value} outside valid range "
                    f"[{mapping.valid_range_low}, {mapping.valid_range_high}]"
                )

        # Rate of change check
        if mapping and mapping.rate_of_change_limit and isinstance(value, (int, float)):
            last = self._last_values.get(tag_id)
            if last:
                last_value, last_time = last
                dt = (timestamp - last_time).total_seconds()
                if dt > 0:
                    rate = abs(value - last_value) / dt
                    if rate > mapping.rate_of_change_limit:
                        result.rate_check_passed = False
                        result.warnings.append(
                            f"Rate of change {rate:.2f}/s exceeds limit "
                            f"{mapping.rate_of_change_limit}/s"
                        )

        # Update last value
        if isinstance(value, (int, float)):
            self._last_values[tag_id] = (value, timestamp)

        # Set overall validity
        result.is_valid = (
            result.range_check_passed and
            result.type_check_passed
        )

        # Set quality based on validation
        if not result.is_valid:
            result.quality = TagQuality.BAD
        elif not result.rate_check_passed:
            result.quality = TagQuality.UNCERTAIN

        return result

    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type."""
        expected = expected_type.lower()

        if expected in ("double", "float"):
            return isinstance(value, (int, float))
        elif expected in ("int", "integer", "int32", "int64"):
            return isinstance(value, int)
        elif expected in ("bool", "boolean"):
            return isinstance(value, bool)
        elif expected in ("string", "str"):
            return isinstance(value, str)
        else:
            return True


# =============================================================================
# TAG MAPPER
# =============================================================================

class TagMapper:
    """
    Main tag mapper for GL-014 ExchangerPro.

    Provides:
    - Canonical name mapping
    - Unit normalization
    - Tag validation
    - Site-specific translation

    Example:
        >>> config = TagMappingConfig.load_from_yaml(Path("tags.yaml"))
        >>> mapper = TagMapper(config)
        >>> canonical = mapper.get_canonical_name("PLANT1.HX001.TI001")
        >>> normalized = mapper.normalize_value(100, "fahrenheit", "celsius", MeasurementType.TEMPERATURE)
    """

    def __init__(self, config: Optional[TagMappingConfig] = None):
        """
        Initialize tag mapper.

        Args:
            config: Optional tag mapping configuration
        """
        self.config = config
        self.unit_normalizer = UnitNormalizer()
        self.validator = TagValidator(config)

        # Build indices
        self._canonical_index: Dict[str, TagMappingEntry] = {}
        self._vendor_index: Dict[str, TagMappingEntry] = {}

        if config:
            self._build_indices()

    def _build_indices(self) -> None:
        """Build lookup indices."""
        if not self.config:
            return

        for mapping in self.config.mappings:
            self._canonical_index[mapping.canonical_name.lower()] = mapping
            self._vendor_index[mapping.vendor_tag.lower()] = mapping

    def set_config(self, config: TagMappingConfig) -> None:
        """Set configuration."""
        self.config = config
        self.validator = TagValidator(config)
        self._build_indices()

    def load_config(self, path: Path) -> None:
        """Load configuration from file."""
        if path.suffix in (".yml", ".yaml"):
            self.config = TagMappingConfig.load_from_yaml(path)
        elif path.suffix == ".json":
            self.config = TagMappingConfig.load_from_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        self.validator = TagValidator(self.config)
        self._build_indices()

    # =========================================================================
    # MAPPING OPERATIONS
    # =========================================================================

    def get_canonical_name(self, vendor_tag: str) -> Optional[str]:
        """Get canonical name for vendor tag."""
        mapping = self._vendor_index.get(vendor_tag.lower())
        if mapping:
            return mapping.canonical_name

        # Try site translation
        if self.config and self.config.site_translation:
            return self.config.site_translation.translate(vendor_tag)

        return None

    def get_vendor_tag(self, canonical_name: str) -> Optional[str]:
        """Get vendor tag for canonical name."""
        mapping = self._canonical_index.get(canonical_name.lower())
        return mapping.vendor_tag if mapping else None

    def get_mapping(self, tag_id: str) -> Optional[TagMappingEntry]:
        """Get mapping by canonical or vendor tag."""
        mapping = self._canonical_index.get(tag_id.lower())
        if mapping:
            return mapping
        return self._vendor_index.get(tag_id.lower())

    def get_mappings_by_exchanger(self, exchanger_id: str) -> List[TagMappingEntry]:
        """Get all mappings for an exchanger."""
        if not self.config:
            return []
        return self.config.get_mappings_by_exchanger(exchanger_id)

    def get_mappings_by_side(
        self,
        exchanger_id: str,
        side: ExchangerSide,
    ) -> List[TagMappingEntry]:
        """Get mappings for specific exchanger side."""
        mappings = self.get_mappings_by_exchanger(exchanger_id)
        return [m for m in mappings if m.side == side]

    def get_mappings_by_measurement(
        self,
        exchanger_id: str,
        measurement: MeasurementType,
    ) -> List[TagMappingEntry]:
        """Get mappings for specific measurement type."""
        mappings = self.get_mappings_by_exchanger(exchanger_id)
        return [m for m in mappings if m.measurement_type == measurement]

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    def normalize_value(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        measurement_type: MeasurementType,
    ) -> float:
        """
        Normalize value to target unit.

        Args:
            value: Value to normalize
            from_unit: Source unit
            to_unit: Target unit
            measurement_type: Measurement type

        Returns:
            Normalized value
        """
        return self.unit_normalizer.convert(
            value, from_unit, to_unit, measurement_type
        )

    def normalize_data_point(
        self,
        tag_id: str,
        value: float,
        source_unit: Optional[str] = None,
    ) -> Tuple[float, str]:
        """
        Normalize a data point to canonical units.

        Args:
            tag_id: Tag identifier
            value: Raw value
            source_unit: Optional source unit override

        Returns:
            Tuple of (normalized_value, target_unit)
        """
        mapping = self.get_mapping(tag_id)
        if not mapping:
            return value, source_unit or "unknown"

        from_unit = source_unit or mapping.source_unit
        to_unit = mapping.target_unit

        normalized = self.normalize_value(
            value, from_unit, to_unit, mapping.measurement_type
        )

        return normalized, to_unit

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_value(
        self,
        tag_id: str,
        value: Any,
        timestamp: Optional[datetime] = None,
    ) -> TagValidationResult:
        """
        Validate a tag value.

        Args:
            tag_id: Tag identifier
            value: Value to validate
            timestamp: Optional timestamp

        Returns:
            Validation result
        """
        ts = timestamp or datetime.now(timezone.utc)
        return self.validator.validate(tag_id, value, ts)

    # =========================================================================
    # SCHEMA OPERATIONS
    # =========================================================================

    def get_exchanger_schema(
        self,
        exchanger_type: str,
    ) -> Optional[ExchangerTagSchema]:
        """Get schema for exchanger type."""
        if not self.config:
            return None

        for schema in self.config.exchanger_schemas:
            if schema.exchanger_type.lower() == exchanger_type.lower():
                return schema

        return None

    def validate_exchanger_tags(
        self,
        exchanger_id: str,
        exchanger_type: str,
    ) -> Tuple[bool, List[str]]:
        """
        Validate that exchanger has all required tags.

        Args:
            exchanger_id: Exchanger identifier
            exchanger_type: Exchanger type

        Returns:
            Tuple of (is_valid, missing_tags)
        """
        schema = self.get_exchanger_schema(exchanger_type)
        if not schema:
            return True, []

        mappings = self.get_mappings_by_exchanger(exchanger_id)
        tag_names = [m.canonical_name for m in mappings]

        return schema.validate_tags(tag_names)

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get mapper statistics."""
        return {
            "config_loaded": self.config is not None,
            "config_version": self.config.version if self.config else None,
            "total_mappings": len(self._canonical_index),
            "exchangers": len(set(
                m.exchanger_id for m in self._canonical_index.values()
            )),
            "schemas": len(
                self.config.exchanger_schemas if self.config else []
            ),
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "MeasurementType",
    "ExchangerSide",
    "ExchangerLocation",
    "TemperatureUnit",
    "PressureUnit",
    "FlowUnit",
    "TagQuality",

    # Engineering Units
    "EngineeringUnit",
    "ENGINEERING_UNITS",

    # Canonical Naming
    "CanonicalTagName",

    # Tag Mapping
    "TagMappingEntry",
    "ExchangerTagSchema",
    "SiteTagTranslation",
    "TagMappingConfig",

    # Unit Normalizer
    "UnitNormalizer",

    # Validator
    "TagValidationResult",
    "TagValidator",

    # Main Mapper
    "TagMapper",
]
