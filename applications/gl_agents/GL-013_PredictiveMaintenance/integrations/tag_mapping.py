# -*- coding: utf-8 -*-
"""
Tag Mapping and Governance for GL-001 ThermalCommand OPC-UA Integration

This module provides comprehensive tag governance including:
- Canonical name mapping (vendor-agnostic tag names)
- Unit conversion and normalization
- Bad value handling strategies
- Timestamp alignment and drift detection
- Versioned tag mapping file support

Canonical Naming Convention:
    {system}.{equipment}.{measurement}

Examples:
    - steam.headerA.pressure
    - steam.headerA.temperature
    - boiler.B1.fuel_flow
    - turbine.T1.inlet_pressure
    - condenser.C1.vacuum

Author: GL-BackendDeveloper
Version: 1.0.0
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator, root_validator
import yaml

from integrations.opcua_schemas import (
    OPCUADataPoint,
    OPCUAQualityCode,
    OPCUATagConfig,
    TagMetadata,
    TagDataType,
    EngineeringUnit,
    ENGINEERING_UNITS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class BadValueStrategy(str, Enum):
    """Strategy for handling bad quality values."""
    REJECT = "reject"  # Reject bad values entirely
    SUBSTITUTE_LAST_GOOD = "substitute_last_good"  # Use last known good value
    SUBSTITUTE_DEFAULT = "substitute_default"  # Use configured default value
    INTERPOLATE = "interpolate"  # Interpolate from surrounding good values
    MARK_AND_PASS = "mark_and_pass"  # Pass through with quality marker


class TimestampSource(str, Enum):
    """Source of timestamp for data alignment."""
    SOURCE = "source"  # Use device/source timestamp
    SERVER = "server"  # Use OPC-UA server timestamp
    CLIENT = "client"  # Use client receive timestamp
    ALIGNED = "aligned"  # Align to nearest interval


# =============================================================================
# CANONICAL TAG NAMING
# =============================================================================

class CanonicalTagName(BaseModel):
    """
    Parsed canonical tag name with validation.

    Format: {system}.{equipment}.{measurement}[.{qualifier}]

    Examples:
        steam.headerA.pressure
        steam.headerA.temperature.setpoint
        boiler.B1.fuel_flow.actual
    """
    system: str = Field(..., description="System identifier (e.g., steam, boiler)")
    equipment: str = Field(..., description="Equipment identifier (e.g., headerA, B1)")
    measurement: str = Field(..., description="Measurement type (e.g., pressure, temperature)")
    qualifier: Optional[str] = Field(None, description="Optional qualifier (e.g., setpoint, actual)")

    @classmethod
    def parse(cls, canonical_name: str) -> "CanonicalTagName":
        """Parse a canonical name string into components."""
        parts = canonical_name.split(".")
        if len(parts) < 3:
            raise ValueError(
                f"Canonical name must have at least 3 parts: {canonical_name}"
            )

        return cls(
            system=parts[0],
            equipment=parts[1],
            measurement=parts[2],
            qualifier=parts[3] if len(parts) > 3 else None,
        )

    def to_string(self) -> str:
        """Convert back to canonical name string."""
        base = f"{self.system}.{self.equipment}.{self.measurement}"
        if self.qualifier:
            return f"{base}.{self.qualifier}"
        return base

    @validator("system", "equipment", "measurement")
    def validate_identifier(cls, v, field):
        """Validate identifier format."""
        if not v:
            raise ValueError(f"{field.name} cannot be empty")
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", v):
            raise ValueError(
                f"{field.name} must start with letter and contain only "
                f"alphanumeric characters and underscores: {v}"
            )
        return v

    def matches_pattern(self, pattern: str) -> bool:
        """
        Check if canonical name matches a pattern.

        Patterns support wildcards:
            * - matches any single part
            ** - matches any number of parts

        Examples:
            steam.*.pressure - matches steam.headerA.pressure
            boiler.** - matches boiler.B1.fuel_flow
        """
        name_parts = self.to_string().split(".")
        pattern_parts = pattern.split(".")

        i, j = 0, 0
        while i < len(name_parts) and j < len(pattern_parts):
            if pattern_parts[j] == "**":
                # ** matches rest of name
                if j == len(pattern_parts) - 1:
                    return True
                # Try matching remaining pattern at each position
                for k in range(i, len(name_parts)):
                    temp_name = ".".join(name_parts[k:])
                    temp_pattern = ".".join(pattern_parts[j + 1:])
                    if CanonicalTagName.parse(
                        f"x.x.{temp_name}" if len(temp_name.split(".")) < 3
                        else temp_name
                    ).matches_pattern(temp_pattern):
                        return True
                return False
            elif pattern_parts[j] == "*":
                i += 1
                j += 1
            elif pattern_parts[j] == name_parts[i]:
                i += 1
                j += 1
            else:
                return False

        return i == len(name_parts) and j == len(pattern_parts)


# =============================================================================
# TAG MAPPING ENTRY
# =============================================================================

class TagMappingEntry(BaseModel):
    """
    Single tag mapping entry linking vendor tag to canonical name.

    Supports scaling, unit conversion, and quality handling.
    """
    # Identification
    canonical_name: str = Field(..., description="Canonical tag name")
    vendor_tag: str = Field(..., description="Vendor/PLC tag address")
    node_id: str = Field(..., description="OPC-UA node ID")

    # Display
    display_name: str = Field(..., description="Human-readable name")
    description: Optional[str] = Field(None, description="Tag description")

    # Data type
    data_type: TagDataType = Field(..., description="Expected data type")
    source_data_type: Optional[TagDataType] = Field(
        None,
        description="Source data type if different"
    )

    # Engineering units
    engineering_unit: str = Field(..., description="Engineering unit key")
    source_unit: Optional[str] = Field(
        None,
        description="Source unit if different from engineering unit"
    )

    # Scaling (source to engineering)
    raw_low: Optional[float] = Field(None, description="Raw value low")
    raw_high: Optional[float] = Field(None, description="Raw value high")
    eng_low: Optional[float] = Field(None, description="Engineering value low")
    eng_high: Optional[float] = Field(None, description="Engineering value high")

    # Range validation
    valid_range_low: Optional[float] = Field(None, description="Valid range minimum")
    valid_range_high: Optional[float] = Field(None, description="Valid range maximum")

    # Bad value handling
    bad_value_strategy: BadValueStrategy = Field(
        default=BadValueStrategy.SUBSTITUTE_LAST_GOOD,
        description="Strategy for bad values"
    )
    default_value: Optional[float] = Field(None, description="Default value for substitution")
    max_bad_duration_s: int = Field(
        default=60,
        ge=1,
        description="Max duration to substitute bad values"
    )

    # Timestamp
    timestamp_source: TimestampSource = Field(
        default=TimestampSource.SOURCE,
        description="Timestamp source preference"
    )
    max_timestamp_drift_s: float = Field(
        default=5.0,
        ge=0.0,
        description="Max allowed timestamp drift in seconds"
    )

    # Metadata
    equipment_id: Optional[str] = Field(None, description="Parent equipment")
    system_id: Optional[str] = Field(None, description="Parent system")
    area_id: Optional[str] = Field(None, description="Plant area")

    # Versioning
    version: str = Field(default="1.0.0", description="Mapping version")
    effective_date: Optional[datetime] = Field(None, description="Effective date")
    deprecated_date: Optional[datetime] = Field(None, description="Deprecation date")

    @validator("engineering_unit")
    def validate_engineering_unit(cls, v):
        """Validate engineering unit is known."""
        if v not in ENGINEERING_UNITS and v not in ["custom", "none"]:
            logger.warning(f"Unknown engineering unit: {v}")
        return v

    def get_engineering_unit(self) -> Optional[EngineeringUnit]:
        """Get engineering unit object."""
        return ENGINEERING_UNITS.get(self.engineering_unit)

    def apply_scaling(self, raw_value: float) -> float:
        """Apply scaling to convert raw value to engineering units."""
        if None in (self.raw_low, self.raw_high, self.eng_low, self.eng_high):
            return raw_value

        raw_range = self.raw_high - self.raw_low
        if raw_range == 0:
            return self.eng_low

        eng_range = self.eng_high - self.eng_low
        scaled = self.eng_low + ((raw_value - self.raw_low) / raw_range * eng_range)

        return scaled

    def reverse_scaling(self, eng_value: float) -> float:
        """Reverse scaling to convert engineering units to raw value."""
        if None in (self.raw_low, self.raw_high, self.eng_low, self.eng_high):
            return eng_value

        eng_range = self.eng_high - self.eng_low
        if eng_range == 0:
            return self.raw_low

        raw_range = self.raw_high - self.raw_low
        raw = self.raw_low + ((eng_value - self.eng_low) / eng_range * raw_range)

        return raw

    def is_in_valid_range(self, value: float) -> bool:
        """Check if value is within valid range."""
        if self.valid_range_low is not None and value < self.valid_range_low:
            return False
        if self.valid_range_high is not None and value > self.valid_range_high:
            return False
        return True

    def is_deprecated(self) -> bool:
        """Check if mapping is deprecated."""
        if self.deprecated_date is None:
            return False
        return datetime.now(timezone.utc) > self.deprecated_date


# =============================================================================
# TAG MAPPING CONFIGURATION
# =============================================================================

class TagMappingConfig(BaseModel):
    """
    Complete tag mapping configuration with versioning support.

    Can be loaded from YAML/JSON files for version control.
    """
    # Metadata
    config_id: str = Field(..., description="Configuration identifier")
    config_name: str = Field(..., description="Configuration name")
    description: Optional[str] = Field(None, description="Configuration description")

    # Version control
    version: str = Field(..., description="Configuration version (semver)")
    effective_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Effective date"
    )
    author: Optional[str] = Field(None, description="Configuration author")
    checksum: Optional[str] = Field(None, description="SHA-256 checksum")

    # Site information
    site_id: str = Field(..., description="Site identifier")
    site_name: str = Field(..., description="Site name")

    # Mappings
    mappings: List[TagMappingEntry] = Field(
        default_factory=list,
        description="Tag mappings"
    )

    # Global settings
    default_bad_value_strategy: BadValueStrategy = Field(
        default=BadValueStrategy.SUBSTITUTE_LAST_GOOD,
        description="Default bad value strategy"
    )
    default_timestamp_source: TimestampSource = Field(
        default=TimestampSource.SOURCE,
        description="Default timestamp source"
    )
    max_timestamp_drift_s: float = Field(
        default=5.0,
        ge=0.0,
        description="Global max timestamp drift"
    )

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
        """Calculate SHA-256 checksum of configuration."""
        # Remove checksum field if present
        data_copy = {k: v for k, v in data.items() if k != "checksum"}
        canonical = json.dumps(data_copy, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save_to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        data = self.dict(exclude={"checksum"})
        data["checksum"] = self._calculate_checksum(data)

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def save_to_json(self, path: Path) -> None:
        """Save configuration to JSON file."""
        data = self.dict(exclude={"checksum"})
        data["checksum"] = self._calculate_checksum(data)

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def verify_checksum(self) -> bool:
        """Verify configuration checksum."""
        if not self.checksum:
            return True
        data = self.dict(exclude={"checksum"})
        expected = self._calculate_checksum(data)
        return expected == self.checksum

    def get_mapping(self, canonical_name: str) -> Optional[TagMappingEntry]:
        """Get mapping by canonical name."""
        for mapping in self.mappings:
            if mapping.canonical_name == canonical_name:
                return mapping
        return None

    def get_mapping_by_vendor_tag(self, vendor_tag: str) -> Optional[TagMappingEntry]:
        """Get mapping by vendor tag."""
        for mapping in self.mappings:
            if mapping.vendor_tag == vendor_tag:
                return mapping
        return None

    def get_mappings_by_pattern(self, pattern: str) -> List[TagMappingEntry]:
        """Get all mappings matching a pattern."""
        results = []
        for mapping in self.mappings:
            try:
                canonical = CanonicalTagName.parse(mapping.canonical_name)
                if canonical.matches_pattern(pattern):
                    results.append(mapping)
            except ValueError:
                continue
        return results


# =============================================================================
# UNIT CONVERTER
# =============================================================================

class UnitConverter:
    """
    Engineering unit converter for process heat measurements.

    Provides accurate conversions between common industrial units.
    All conversions use Decimal for precision.
    """

    # Conversion factors to SI base units
    # Temperature: Kelvin, Pressure: Pascal, Flow: kg/s, Power: Watt

    TEMPERATURE_TO_KELVIN: Dict[str, Callable[[Decimal], Decimal]] = {
        "celsius": lambda x: x + Decimal("273.15"),
        "fahrenheit": lambda x: (x - Decimal("32")) * Decimal("5") / Decimal("9") + Decimal("273.15"),
        "kelvin": lambda x: x,
        "rankine": lambda x: x * Decimal("5") / Decimal("9"),
    }

    TEMPERATURE_FROM_KELVIN: Dict[str, Callable[[Decimal], Decimal]] = {
        "celsius": lambda x: x - Decimal("273.15"),
        "fahrenheit": lambda x: (x - Decimal("273.15")) * Decimal("9") / Decimal("5") + Decimal("32"),
        "kelvin": lambda x: x,
        "rankine": lambda x: x * Decimal("9") / Decimal("5"),
    }

    PRESSURE_TO_PASCAL: Dict[str, Decimal] = {
        "pascal": Decimal("1"),
        "pa": Decimal("1"),
        "kilopascal": Decimal("1000"),
        "kpa": Decimal("1000"),
        "megapascal": Decimal("1000000"),
        "mpa": Decimal("1000000"),
        "bar": Decimal("100000"),
        "mbar": Decimal("100"),
        "psi": Decimal("6894.757"),
        "atm": Decimal("101325"),
        "mmhg": Decimal("133.322"),
        "inhg": Decimal("3386.39"),
    }

    FLOW_TO_KG_PER_S: Dict[str, Decimal] = {
        "kg_per_s": Decimal("1"),
        "kg/s": Decimal("1"),
        "kg_per_h": Decimal("1") / Decimal("3600"),
        "kg/h": Decimal("1") / Decimal("3600"),
        "t_per_h": Decimal("1000") / Decimal("3600"),
        "t/h": Decimal("1000") / Decimal("3600"),
        "lb_per_h": Decimal("0.453592") / Decimal("3600"),
        "lb/h": Decimal("0.453592") / Decimal("3600"),
        "lb_per_s": Decimal("0.453592"),
        "lb/s": Decimal("0.453592"),
    }

    POWER_TO_WATT: Dict[str, Decimal] = {
        "watt": Decimal("1"),
        "w": Decimal("1"),
        "kilowatt": Decimal("1000"),
        "kw": Decimal("1000"),
        "megawatt": Decimal("1000000"),
        "mw": Decimal("1000000"),
        "btu_per_h": Decimal("0.293071"),
        "btu/h": Decimal("0.293071"),
        "hp": Decimal("745.7"),
        "kcal_per_h": Decimal("1.163"),
        "mmbtu_per_h": Decimal("293071"),
    }

    ENERGY_TO_JOULE: Dict[str, Decimal] = {
        "joule": Decimal("1"),
        "j": Decimal("1"),
        "kilojoule": Decimal("1000"),
        "kj": Decimal("1000"),
        "megajoule": Decimal("1000000"),
        "mj": Decimal("1000000"),
        "kwh": Decimal("3600000"),
        "mwh": Decimal("3600000000"),
        "btu": Decimal("1055.06"),
        "mmbtu": Decimal("1055060000"),
        "therm": Decimal("105506000"),
        "kcal": Decimal("4184"),
    }

    @classmethod
    def convert_temperature(
        cls,
        value: Union[float, Decimal],
        from_unit: str,
        to_unit: str,
    ) -> Decimal:
        """
        Convert temperature between units.

        Args:
            value: Temperature value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted temperature value
        """
        value = Decimal(str(value))
        from_unit = from_unit.lower().replace(" ", "_")
        to_unit = to_unit.lower().replace(" ", "_")

        if from_unit == to_unit:
            return value

        # Convert to Kelvin first
        if from_unit not in cls.TEMPERATURE_TO_KELVIN:
            raise ValueError(f"Unknown temperature unit: {from_unit}")
        kelvin = cls.TEMPERATURE_TO_KELVIN[from_unit](value)

        # Convert from Kelvin to target
        if to_unit not in cls.TEMPERATURE_FROM_KELVIN:
            raise ValueError(f"Unknown temperature unit: {to_unit}")
        return cls.TEMPERATURE_FROM_KELVIN[to_unit](kelvin)

    @classmethod
    def convert_pressure(
        cls,
        value: Union[float, Decimal],
        from_unit: str,
        to_unit: str,
    ) -> Decimal:
        """
        Convert pressure between units.

        Args:
            value: Pressure value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted pressure value
        """
        value = Decimal(str(value))
        from_unit = from_unit.lower().replace(" ", "_")
        to_unit = to_unit.lower().replace(" ", "_")

        if from_unit == to_unit:
            return value

        # Convert to Pascal first
        if from_unit not in cls.PRESSURE_TO_PASCAL:
            raise ValueError(f"Unknown pressure unit: {from_unit}")
        pascal = value * cls.PRESSURE_TO_PASCAL[from_unit]

        # Convert from Pascal to target
        if to_unit not in cls.PRESSURE_TO_PASCAL:
            raise ValueError(f"Unknown pressure unit: {to_unit}")
        return pascal / cls.PRESSURE_TO_PASCAL[to_unit]

    @classmethod
    def convert_flow(
        cls,
        value: Union[float, Decimal],
        from_unit: str,
        to_unit: str,
    ) -> Decimal:
        """
        Convert mass flow rate between units.

        Args:
            value: Flow rate value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted flow rate value
        """
        value = Decimal(str(value))
        from_unit = from_unit.lower().replace(" ", "_")
        to_unit = to_unit.lower().replace(" ", "_")

        if from_unit == to_unit:
            return value

        # Convert to kg/s first
        if from_unit not in cls.FLOW_TO_KG_PER_S:
            raise ValueError(f"Unknown flow unit: {from_unit}")
        kg_per_s = value * cls.FLOW_TO_KG_PER_S[from_unit]

        # Convert from kg/s to target
        if to_unit not in cls.FLOW_TO_KG_PER_S:
            raise ValueError(f"Unknown flow unit: {to_unit}")
        return kg_per_s / cls.FLOW_TO_KG_PER_S[to_unit]

    @classmethod
    def convert_power(
        cls,
        value: Union[float, Decimal],
        from_unit: str,
        to_unit: str,
    ) -> Decimal:
        """
        Convert power between units.

        Args:
            value: Power value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted power value
        """
        value = Decimal(str(value))
        from_unit = from_unit.lower().replace(" ", "_")
        to_unit = to_unit.lower().replace(" ", "_")

        if from_unit == to_unit:
            return value

        # Convert to Watt first
        if from_unit not in cls.POWER_TO_WATT:
            raise ValueError(f"Unknown power unit: {from_unit}")
        watt = value * cls.POWER_TO_WATT[from_unit]

        # Convert from Watt to target
        if to_unit not in cls.POWER_TO_WATT:
            raise ValueError(f"Unknown power unit: {to_unit}")
        return watt / cls.POWER_TO_WATT[to_unit]

    @classmethod
    def convert_energy(
        cls,
        value: Union[float, Decimal],
        from_unit: str,
        to_unit: str,
    ) -> Decimal:
        """
        Convert energy between units.

        Args:
            value: Energy value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted energy value
        """
        value = Decimal(str(value))
        from_unit = from_unit.lower().replace(" ", "_")
        to_unit = to_unit.lower().replace(" ", "_")

        if from_unit == to_unit:
            return value

        # Convert to Joule first
        if from_unit not in cls.ENERGY_TO_JOULE:
            raise ValueError(f"Unknown energy unit: {from_unit}")
        joule = value * cls.ENERGY_TO_JOULE[from_unit]

        # Convert from Joule to target
        if to_unit not in cls.ENERGY_TO_JOULE:
            raise ValueError(f"Unknown energy unit: {to_unit}")
        return joule / cls.ENERGY_TO_JOULE[to_unit]


# =============================================================================
# TAG GOVERNANCE
# =============================================================================

class TagGovernance:
    """
    Tag governance enforcement for data quality and compliance.

    Ensures:
    - Canonical naming compliance
    - Unit normalization
    - Quality code handling
    - Timestamp alignment
    - Value range validation
    """

    def __init__(self, config: TagMappingConfig):
        """
        Initialize tag governance.

        Args:
            config: Tag mapping configuration
        """
        self.config = config
        self._mapping_index: Dict[str, TagMappingEntry] = {}
        self._vendor_index: Dict[str, TagMappingEntry] = {}
        self._last_good_values: Dict[str, Tuple[float, datetime]] = {}
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        """Rebuild lookup indices for efficient access."""
        self._mapping_index.clear()
        self._vendor_index.clear()

        for mapping in self.config.mappings:
            self._mapping_index[mapping.canonical_name] = mapping
            self._vendor_index[mapping.vendor_tag] = mapping

    def validate_canonical_name(self, name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a canonical tag name.

        Args:
            name: Canonical name to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            canonical = CanonicalTagName.parse(name)
            return True, None
        except ValueError as e:
            return False, str(e)

    def normalize_data_point(
        self,
        data_point: OPCUADataPoint,
        mapping: Optional[TagMappingEntry] = None,
    ) -> OPCUADataPoint:
        """
        Normalize a data point according to governance rules.

        Applies:
        - Unit conversion
        - Scaling
        - Bad value handling
        - Timestamp alignment

        Args:
            data_point: Raw data point
            mapping: Optional explicit mapping (auto-lookup if not provided)

        Returns:
            Normalized data point
        """
        # Get mapping
        if mapping is None:
            mapping = self._mapping_index.get(data_point.canonical_name)

        if mapping is None:
            logger.warning(f"No mapping found for {data_point.canonical_name}")
            return data_point

        # Create copy for modification
        normalized = data_point.copy(deep=True)

        # Handle bad values
        if not data_point.quality_code.is_good():
            normalized = self._handle_bad_value(normalized, mapping)

        # Apply scaling if value is numeric
        if isinstance(data_point.value, (int, float)):
            scaled_value = mapping.apply_scaling(float(data_point.value))
            normalized.scaled_value = scaled_value

            # Store as last good value if quality is good
            if data_point.quality_code.is_good():
                self._last_good_values[data_point.canonical_name] = (
                    scaled_value,
                    data_point.source_timestamp,
                )

        # Set engineering unit
        eu = mapping.get_engineering_unit()
        if eu:
            normalized.engineering_unit = eu.display_name

        # Align timestamp
        normalized = self._align_timestamp(normalized, mapping)

        # Recalculate provenance hash
        normalized.provenance_hash = normalized.calculate_provenance_hash()

        return normalized

    def _handle_bad_value(
        self,
        data_point: OPCUADataPoint,
        mapping: TagMappingEntry,
    ) -> OPCUADataPoint:
        """
        Handle bad quality value according to strategy.

        Args:
            data_point: Data point with bad quality
            mapping: Tag mapping configuration

        Returns:
            Handled data point
        """
        strategy = mapping.bad_value_strategy

        if strategy == BadValueStrategy.REJECT:
            # Keep as-is, downstream will handle rejection
            return data_point

        elif strategy == BadValueStrategy.SUBSTITUTE_LAST_GOOD:
            last_good = self._last_good_values.get(data_point.canonical_name)
            if last_good:
                value, timestamp = last_good
                age = (datetime.now(timezone.utc) - timestamp).total_seconds()
                if age <= mapping.max_bad_duration_s:
                    data_point.value = value
                    data_point.scaled_value = value
                    # Mark quality as substituted
                    data_point.quality_code = OPCUAQualityCode.UNCERTAIN_LAST_USABLE

        elif strategy == BadValueStrategy.SUBSTITUTE_DEFAULT:
            if mapping.default_value is not None:
                data_point.value = mapping.default_value
                data_point.scaled_value = mapping.default_value
                data_point.quality_code = OPCUAQualityCode.UNCERTAIN_SUB_NORMAL

        elif strategy == BadValueStrategy.MARK_AND_PASS:
            # Keep value but ensure quality is marked
            pass

        return data_point

    def _align_timestamp(
        self,
        data_point: OPCUADataPoint,
        mapping: TagMappingEntry,
    ) -> OPCUADataPoint:
        """
        Align data point timestamp according to configuration.

        Args:
            data_point: Data point
            mapping: Tag mapping configuration

        Returns:
            Data point with aligned timestamp
        """
        # Check timestamp drift
        now = datetime.now(timezone.utc)

        if mapping.timestamp_source == TimestampSource.SOURCE:
            ref_time = data_point.source_timestamp
        elif mapping.timestamp_source == TimestampSource.SERVER:
            ref_time = data_point.server_timestamp
        elif mapping.timestamp_source == TimestampSource.CLIENT:
            ref_time = data_point.received_timestamp
        else:
            ref_time = data_point.source_timestamp

        drift = abs((now - ref_time).total_seconds())
        if drift > mapping.max_timestamp_drift_s:
            logger.warning(
                f"Timestamp drift {drift:.1f}s exceeds max {mapping.max_timestamp_drift_s}s "
                f"for {data_point.canonical_name}"
            )
            # Mark quality as uncertain due to timestamp
            if data_point.quality_code.is_good():
                data_point.quality_code = OPCUAQualityCode.UNCERTAIN_SUB_NORMAL

        return data_point

    def validate_value_range(
        self,
        canonical_name: str,
        value: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate value is within configured range.

        Args:
            canonical_name: Tag canonical name
            value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        mapping = self._mapping_index.get(canonical_name)
        if mapping is None:
            return True, None  # No mapping = no validation

        if not mapping.is_in_valid_range(value):
            return False, (
                f"Value {value} outside valid range "
                f"[{mapping.valid_range_low}, {mapping.valid_range_high}]"
            )

        return True, None


# =============================================================================
# TAG MAPPER
# =============================================================================

class TagMapper:
    """
    Main tag mapper coordinating all mapping and governance operations.

    Provides high-level interface for:
    - Loading versioned mapping configurations
    - Mapping vendor tags to canonical names
    - Normalizing and validating data points
    - Unit conversion
    """

    def __init__(self, config: Optional[TagMappingConfig] = None):
        """
        Initialize tag mapper.

        Args:
            config: Optional initial configuration
        """
        self.config: Optional[TagMappingConfig] = config
        self.governance: Optional[TagGovernance] = None
        self.unit_converter = UnitConverter()
        self._version_history: List[Tuple[str, datetime]] = []

        if config:
            self._initialize_governance()

    def _initialize_governance(self) -> None:
        """Initialize governance from current configuration."""
        if self.config:
            self.governance = TagGovernance(self.config)

    def load_config(self, path: Path) -> None:
        """
        Load configuration from file.

        Supports YAML and JSON formats.

        Args:
            path: Path to configuration file
        """
        if path.suffix in (".yml", ".yaml"):
            self.config = TagMappingConfig.load_from_yaml(path)
        elif path.suffix == ".json":
            self.config = TagMappingConfig.load_from_json(path)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

        # Verify checksum
        if not self.config.verify_checksum():
            logger.warning(f"Configuration checksum mismatch for {path}")

        # Track version
        self._version_history.append((self.config.version, datetime.now(timezone.utc)))

        # Initialize governance
        self._initialize_governance()

        logger.info(
            f"Loaded tag mapping config v{self.config.version} "
            f"with {len(self.config.mappings)} mappings"
        )

    def set_config(self, config: TagMappingConfig) -> None:
        """
        Set configuration programmatically.

        Args:
            config: Tag mapping configuration
        """
        self.config = config
        self._version_history.append((config.version, datetime.now(timezone.utc)))
        self._initialize_governance()

    def get_canonical_name(self, vendor_tag: str) -> Optional[str]:
        """
        Get canonical name for a vendor tag.

        Args:
            vendor_tag: Vendor/PLC tag address

        Returns:
            Canonical name or None if not mapped
        """
        if not self.config:
            return None

        mapping = self.config.get_mapping_by_vendor_tag(vendor_tag)
        return mapping.canonical_name if mapping else None

    def get_vendor_tag(self, canonical_name: str) -> Optional[str]:
        """
        Get vendor tag for a canonical name.

        Args:
            canonical_name: Canonical tag name

        Returns:
            Vendor tag or None if not mapped
        """
        if not self.config:
            return None

        mapping = self.config.get_mapping(canonical_name)
        return mapping.vendor_tag if mapping else None

    def get_mapping(self, canonical_name: str) -> Optional[TagMappingEntry]:
        """
        Get complete mapping for a canonical name.

        Args:
            canonical_name: Canonical tag name

        Returns:
            Mapping entry or None
        """
        if not self.config:
            return None
        return self.config.get_mapping(canonical_name)

    def normalize_data_point(self, data_point: OPCUADataPoint) -> OPCUADataPoint:
        """
        Normalize a data point using governance rules.

        Args:
            data_point: Raw data point

        Returns:
            Normalized data point
        """
        if not self.governance:
            return data_point
        return self.governance.normalize_data_point(data_point)

    def convert_units(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        measurement_type: str,
    ) -> float:
        """
        Convert value between engineering units.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit
            measurement_type: Type of measurement (temperature, pressure, flow, power, energy)

        Returns:
            Converted value
        """
        if measurement_type == "temperature":
            return float(self.unit_converter.convert_temperature(value, from_unit, to_unit))
        elif measurement_type == "pressure":
            return float(self.unit_converter.convert_pressure(value, from_unit, to_unit))
        elif measurement_type == "flow":
            return float(self.unit_converter.convert_flow(value, from_unit, to_unit))
        elif measurement_type == "power":
            return float(self.unit_converter.convert_power(value, from_unit, to_unit))
        elif measurement_type == "energy":
            return float(self.unit_converter.convert_energy(value, from_unit, to_unit))
        else:
            raise ValueError(f"Unknown measurement type: {measurement_type}")

    def validate_tag_value(
        self,
        canonical_name: str,
        value: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a tag value against configured ranges.

        Args:
            canonical_name: Tag canonical name
            value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.governance:
            return True, None
        return self.governance.validate_value_range(canonical_name, value)

    def create_tag_config(self, canonical_name: str) -> Optional[OPCUATagConfig]:
        """
        Create OPCUATagConfig from mapping configuration.

        Args:
            canonical_name: Canonical tag name

        Returns:
            OPCUATagConfig or None if not mapped
        """
        mapping = self.get_mapping(canonical_name)
        if not mapping:
            return None

        eu = mapping.get_engineering_unit()

        metadata = TagMetadata(
            tag_id=canonical_name.replace(".", "_"),
            node_id=mapping.node_id,
            canonical_name=mapping.canonical_name,
            display_name=mapping.display_name,
            description=mapping.description,
            data_type=mapping.data_type,
            engineering_unit=eu,
            eu_range_low=mapping.eng_low,
            eu_range_high=mapping.eng_high,
            raw_low=mapping.raw_low,
            raw_high=mapping.raw_high,
            scaled_low=mapping.eng_low,
            scaled_high=mapping.eng_high,
            equipment_id=mapping.equipment_id,
            system_id=mapping.system_id,
            area_id=mapping.area_id,
            version=mapping.version,
        )

        return OPCUATagConfig(metadata=metadata)

    def get_tags_by_system(self, system: str) -> List[str]:
        """
        Get all canonical names for a system.

        Args:
            system: System identifier (e.g., "steam", "boiler")

        Returns:
            List of canonical names
        """
        if not self.config:
            return []

        return [
            m.canonical_name
            for m in self.config.mappings
            if m.canonical_name.startswith(f"{system}.")
        ]

    def get_tags_by_equipment(self, equipment: str) -> List[str]:
        """
        Get all canonical names for an equipment.

        Args:
            equipment: Equipment identifier

        Returns:
            List of canonical names
        """
        if not self.config:
            return []

        return [
            m.canonical_name
            for m in self.config.mappings
            if m.equipment_id == equipment
        ]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "BadValueStrategy",
    "TimestampSource",
    # Canonical Naming
    "CanonicalTagName",
    # Mapping
    "TagMappingEntry",
    "TagMappingConfig",
    # Unit Conversion
    "UnitConverter",
    # Governance
    "TagGovernance",
    # Main Interface
    "TagMapper",
]
