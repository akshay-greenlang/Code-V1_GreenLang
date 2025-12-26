"""
Tag Mapping Configuration Module for GL-015 INSULSCAN (Insulation Inspection Agent).

Provides comprehensive tag mapping functionality:
- Map OPC-UA tags to asset IDs
- Map sensor channels to measurement types
- Configurable transformations and scaling
- Validation of tag mappings
- Tag group management
- Import/export capabilities

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
    TypeVar,
    Union,
)
import asyncio
import json
import logging
import re
import uuid
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class TagSource(str, Enum):
    """Tag source systems."""

    OPCUA = "opcua"
    HISTORIAN_PI = "historian_pi"
    HISTORIAN_AVEVA = "historian_aveva"
    THERMAL_CAMERA = "thermal_camera"
    MANUAL_ENTRY = "manual_entry"
    CALCULATED = "calculated"
    EXTERNAL_API = "external_api"


class MeasurementType(str, Enum):
    """Types of measurements."""

    # Temperature measurements
    SURFACE_TEMPERATURE = "surface_temperature"
    OPERATING_TEMPERATURE = "operating_temperature"
    AMBIENT_TEMPERATURE = "ambient_temperature"
    PROCESS_TEMPERATURE = "process_temperature"
    JACKET_TEMPERATURE = "jacket_temperature"
    SKIN_TEMPERATURE = "skin_temperature"

    # Thermal camera measurements
    HOT_SPOT_TEMPERATURE = "hot_spot_temperature"
    COLD_SPOT_TEMPERATURE = "cold_spot_temperature"
    AVERAGE_TEMPERATURE = "average_temperature"
    RADIOMETRIC_MAX = "radiometric_max"
    RADIOMETRIC_MIN = "radiometric_min"

    # Heat loss related
    HEAT_LOSS = "heat_loss"
    THERMAL_EFFICIENCY = "thermal_efficiency"
    ENERGY_CONSUMPTION = "energy_consumption"

    # Process measurements
    FLOW_RATE = "flow_rate"
    PRESSURE = "pressure"
    STEAM_QUALITY = "steam_quality"

    # Environmental
    WIND_SPEED = "wind_speed"
    HUMIDITY = "humidity"
    SOLAR_RADIATION = "solar_radiation"

    # Status
    EQUIPMENT_STATUS = "equipment_status"
    ALARM_STATUS = "alarm_status"

    # Other
    CUSTOM = "custom"


class EngineeringUnit(str, Enum):
    """Engineering units for tag values."""

    # Temperature
    CELSIUS = "degC"
    FAHRENHEIT = "degF"
    KELVIN = "K"

    # Energy
    KILOWATT = "kW"
    MEGAWATT = "MW"
    BTU_PER_HOUR = "BTU/hr"
    WATTS = "W"

    # Flow
    KG_PER_SECOND = "kg/s"
    KG_PER_HOUR = "kg/hr"
    M3_PER_HOUR = "m3/hr"
    GPM = "gpm"

    # Pressure
    BAR = "bar"
    PSI = "psi"
    KPA = "kPa"
    MPA = "MPa"

    # Length
    METERS = "m"
    MILLIMETERS = "mm"
    FEET = "ft"
    INCHES = "in"

    # Area
    SQUARE_METERS = "m2"
    SQUARE_FEET = "ft2"

    # Velocity
    METERS_PER_SECOND = "m/s"
    KM_PER_HOUR = "km/hr"

    # Percentage
    PERCENT = "%"

    # Dimensionless
    UNITLESS = "unitless"

    # Custom
    CUSTOM = "custom"


class TransformationType(str, Enum):
    """Types of value transformations."""

    NONE = "none"
    SCALE = "scale"  # y = mx + b
    POLYNOMIAL = "polynomial"  # y = a0 + a1*x + a2*x^2 + ...
    UNIT_CONVERSION = "unit_conversion"  # Convert between units
    LOOKUP_TABLE = "lookup_table"  # Lookup table interpolation
    CUSTOM_FUNCTION = "custom_function"  # Custom Python function
    CLAMP = "clamp"  # Clamp to range
    DEADBAND = "deadband"  # Apply deadband


class ValidationRule(str, Enum):
    """Tag validation rules."""

    RANGE = "range"  # Value within range
    NOT_NULL = "not_null"  # Value not null
    NUMERIC = "numeric"  # Value is numeric
    POSITIVE = "positive"  # Value > 0
    NON_NEGATIVE = "non_negative"  # Value >= 0
    RATE_OF_CHANGE = "rate_of_change"  # Rate of change limit
    STALE_DATA = "stale_data"  # Data freshness check
    PATTERN = "pattern"  # Regex pattern match
    CUSTOM = "custom"  # Custom validation


class MappingStatus(str, Enum):
    """Tag mapping status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING_VALIDATION = "pending_validation"
    VALIDATION_FAILED = "validation_failed"
    DEPRECATED = "deprecated"


# =============================================================================
# Pydantic Models - Transformations
# =============================================================================


class ScaleTransformation(BaseModel):
    """Linear scale transformation: y = scale * x + offset."""

    model_config = ConfigDict(frozen=True)

    scale: float = Field(default=1.0, description="Scale factor")
    offset: float = Field(default=0.0, description="Offset")

    def apply(self, value: float) -> float:
        """Apply transformation."""
        return self.scale * value + self.offset


class PolynomialTransformation(BaseModel):
    """Polynomial transformation: y = sum(coefficients[i] * x^i)."""

    model_config = ConfigDict(frozen=True)

    coefficients: List[float] = Field(
        ...,
        min_length=1,
        description="Polynomial coefficients [a0, a1, a2, ...]"
    )

    def apply(self, value: float) -> float:
        """Apply transformation."""
        result = 0.0
        for i, coeff in enumerate(self.coefficients):
            result += coeff * (value ** i)
        return result


class UnitConversion(BaseModel):
    """Unit conversion specification."""

    model_config = ConfigDict(frozen=True)

    from_unit: EngineeringUnit = Field(..., description="Source unit")
    to_unit: EngineeringUnit = Field(..., description="Target unit")

    def apply(self, value: float) -> float:
        """Apply unit conversion."""
        # Temperature conversions
        if self.from_unit == EngineeringUnit.CELSIUS:
            if self.to_unit == EngineeringUnit.FAHRENHEIT:
                return value * 9/5 + 32
            elif self.to_unit == EngineeringUnit.KELVIN:
                return value + 273.15

        if self.from_unit == EngineeringUnit.FAHRENHEIT:
            if self.to_unit == EngineeringUnit.CELSIUS:
                return (value - 32) * 5/9
            elif self.to_unit == EngineeringUnit.KELVIN:
                return (value - 32) * 5/9 + 273.15

        if self.from_unit == EngineeringUnit.KELVIN:
            if self.to_unit == EngineeringUnit.CELSIUS:
                return value - 273.15
            elif self.to_unit == EngineeringUnit.FAHRENHEIT:
                return (value - 273.15) * 9/5 + 32

        # Pressure conversions
        if self.from_unit == EngineeringUnit.BAR:
            if self.to_unit == EngineeringUnit.PSI:
                return value * 14.5038
            elif self.to_unit == EngineeringUnit.KPA:
                return value * 100

        if self.from_unit == EngineeringUnit.PSI:
            if self.to_unit == EngineeringUnit.BAR:
                return value / 14.5038
            elif self.to_unit == EngineeringUnit.KPA:
                return value * 6.89476

        # Default: return unchanged
        return value


class LookupTableEntry(BaseModel):
    """Entry in a lookup table."""

    model_config = ConfigDict(frozen=True)

    input_value: float = Field(..., description="Input value")
    output_value: float = Field(..., description="Output value")


class LookupTableTransformation(BaseModel):
    """Lookup table with linear interpolation."""

    model_config = ConfigDict(frozen=False)

    entries: List[LookupTableEntry] = Field(
        ...,
        min_length=2,
        description="Lookup table entries (must be sorted by input_value)"
    )
    extrapolate: bool = Field(
        default=False,
        description="Allow extrapolation outside table range"
    )

    def apply(self, value: float) -> float:
        """Apply lookup table with interpolation."""
        sorted_entries = sorted(self.entries, key=lambda e: e.input_value)

        # Handle values outside range
        if value <= sorted_entries[0].input_value:
            if self.extrapolate:
                # Linear extrapolation from first two points
                x1, y1 = sorted_entries[0].input_value, sorted_entries[0].output_value
                x2, y2 = sorted_entries[1].input_value, sorted_entries[1].output_value
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                return y1 + slope * (value - x1)
            return sorted_entries[0].output_value

        if value >= sorted_entries[-1].input_value:
            if self.extrapolate:
                # Linear extrapolation from last two points
                x1, y1 = sorted_entries[-2].input_value, sorted_entries[-2].output_value
                x2, y2 = sorted_entries[-1].input_value, sorted_entries[-1].output_value
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                return y2 + slope * (value - x2)
            return sorted_entries[-1].output_value

        # Linear interpolation between points
        for i in range(len(sorted_entries) - 1):
            x1, y1 = sorted_entries[i].input_value, sorted_entries[i].output_value
            x2, y2 = sorted_entries[i + 1].input_value, sorted_entries[i + 1].output_value

            if x1 <= value <= x2:
                # Linear interpolation
                t = (value - x1) / (x2 - x1) if x2 != x1 else 0
                return y1 + t * (y2 - y1)

        return value


class ClampTransformation(BaseModel):
    """Clamp values to a range."""

    model_config = ConfigDict(frozen=True)

    min_value: Optional[float] = Field(default=None, description="Minimum value")
    max_value: Optional[float] = Field(default=None, description="Maximum value")

    def apply(self, value: float) -> float:
        """Apply clamping."""
        result = value
        if self.min_value is not None:
            result = max(result, self.min_value)
        if self.max_value is not None:
            result = min(result, self.max_value)
        return result


class DeadbandTransformation(BaseModel):
    """Apply deadband to filter small changes."""

    model_config = ConfigDict(frozen=False)

    deadband: float = Field(..., ge=0, description="Deadband value")
    deadband_type: str = Field(
        default="absolute",
        description="Deadband type (absolute, percent)"
    )
    last_value: Optional[float] = Field(
        default=None,
        description="Last reported value"
    )

    def apply(self, value: float) -> Optional[float]:
        """Apply deadband. Returns None if within deadband."""
        if self.last_value is None:
            self.last_value = value
            return value

        if self.deadband_type == "percent":
            threshold = abs(self.last_value * self.deadband / 100)
        else:
            threshold = self.deadband

        if abs(value - self.last_value) > threshold:
            self.last_value = value
            return value

        return None  # Within deadband


class TransformationChain(BaseModel):
    """Chain of transformations applied in sequence."""

    model_config = ConfigDict(frozen=False)

    transformations: List[
        Union[
            ScaleTransformation,
            PolynomialTransformation,
            UnitConversion,
            LookupTableTransformation,
            ClampTransformation,
        ]
    ] = Field(
        default_factory=list,
        description="Transformations to apply in order"
    )

    def apply(self, value: float) -> float:
        """Apply all transformations in sequence."""
        result = value
        for transform in self.transformations:
            result = transform.apply(result)
        return result


# =============================================================================
# Pydantic Models - Validation
# =============================================================================


class RangeValidation(BaseModel):
    """Validate value is within range."""

    model_config = ConfigDict(frozen=True)

    min_value: Optional[float] = Field(default=None, description="Minimum value")
    max_value: Optional[float] = Field(default=None, description="Maximum value")
    include_min: bool = Field(default=True, description="Include minimum")
    include_max: bool = Field(default=True, description="Include maximum")

    def validate_value(self, value: float) -> Tuple[bool, Optional[str]]:
        """Validate value."""
        if self.min_value is not None:
            if self.include_min:
                if value < self.min_value:
                    return False, f"Value {value} below minimum {self.min_value}"
            else:
                if value <= self.min_value:
                    return False, f"Value {value} at or below minimum {self.min_value}"

        if self.max_value is not None:
            if self.include_max:
                if value > self.max_value:
                    return False, f"Value {value} above maximum {self.max_value}"
            else:
                if value >= self.max_value:
                    return False, f"Value {value} at or above maximum {self.max_value}"

        return True, None


class RateOfChangeValidation(BaseModel):
    """Validate rate of change."""

    model_config = ConfigDict(frozen=False)

    max_rate_per_second: float = Field(
        ...,
        gt=0,
        description="Maximum rate of change per second"
    )
    last_value: Optional[float] = Field(default=None, description="Last value")
    last_timestamp: Optional[datetime] = Field(default=None, description="Last timestamp")

    def validate_value(
        self,
        value: float,
        timestamp: datetime
    ) -> Tuple[bool, Optional[str]]:
        """Validate rate of change."""
        if self.last_value is None or self.last_timestamp is None:
            self.last_value = value
            self.last_timestamp = timestamp
            return True, None

        dt = (timestamp - self.last_timestamp).total_seconds()
        if dt <= 0:
            return True, None

        rate = abs(value - self.last_value) / dt

        if rate > self.max_rate_per_second:
            return False, f"Rate of change {rate:.2f}/s exceeds maximum {self.max_rate_per_second}/s"

        self.last_value = value
        self.last_timestamp = timestamp
        return True, None


class StaleDataValidation(BaseModel):
    """Validate data freshness."""

    model_config = ConfigDict(frozen=True)

    max_age_seconds: float = Field(
        ...,
        gt=0,
        description="Maximum age of data in seconds"
    )

    def validate_timestamp(self, timestamp: datetime) -> Tuple[bool, Optional[str]]:
        """Validate data freshness."""
        age = (datetime.utcnow() - timestamp).total_seconds()
        if age > self.max_age_seconds:
            return False, f"Data age {age:.0f}s exceeds maximum {self.max_age_seconds}s"
        return True, None


class ValidationConfig(BaseModel):
    """Complete validation configuration for a tag."""

    model_config = ConfigDict(frozen=False)

    range_validation: Optional[RangeValidation] = Field(
        default=None,
        description="Range validation"
    )
    rate_of_change_validation: Optional[RateOfChangeValidation] = Field(
        default=None,
        description="Rate of change validation"
    )
    stale_data_validation: Optional[StaleDataValidation] = Field(
        default=None,
        description="Stale data validation"
    )
    not_null: bool = Field(
        default=True,
        description="Value must not be null"
    )
    numeric: bool = Field(
        default=True,
        description="Value must be numeric"
    )

    def validate(
        self,
        value: Optional[float],
        timestamp: Optional[datetime] = None
    ) -> Tuple[bool, List[str]]:
        """Run all validations."""
        errors = []

        # Null check
        if self.not_null and value is None:
            errors.append("Value is null")
            return False, errors

        if value is None:
            return True, []

        # Numeric check
        if self.numeric:
            try:
                float(value)
            except (TypeError, ValueError):
                errors.append(f"Value {value} is not numeric")
                return False, errors

        # Range validation
        if self.range_validation:
            valid, error = self.range_validation.validate_value(value)
            if not valid:
                errors.append(error)

        # Rate of change validation
        if self.rate_of_change_validation and timestamp:
            valid, error = self.rate_of_change_validation.validate_value(
                value, timestamp
            )
            if not valid:
                errors.append(error)

        # Stale data validation
        if self.stale_data_validation and timestamp:
            valid, error = self.stale_data_validation.validate_timestamp(timestamp)
            if not valid:
                errors.append(error)

        return len(errors) == 0, errors


# =============================================================================
# Pydantic Models - Tag Mapping
# =============================================================================


class TagMapping(BaseModel):
    """Complete mapping for a single tag."""

    model_config = ConfigDict(frozen=False)

    mapping_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique mapping identifier"
    )
    mapping_name: str = Field(
        ...,
        description="Human-readable mapping name"
    )

    # Source tag information
    source_tag_name: str = Field(
        ...,
        description="Tag name in source system"
    )
    source_system: TagSource = Field(
        ...,
        description="Source system"
    )
    source_node_id: Optional[str] = Field(
        default=None,
        description="OPC-UA node ID or historian tag ID"
    )
    source_path: Optional[str] = Field(
        default=None,
        description="Full path in source system"
    )

    # Target mapping
    equipment_id: str = Field(
        ...,
        description="Target equipment ID"
    )
    asset_id: Optional[str] = Field(
        default=None,
        description="Target asset ID"
    )
    measurement_type: MeasurementType = Field(
        ...,
        description="Type of measurement"
    )
    measurement_point: Optional[str] = Field(
        default=None,
        description="Specific measurement point on equipment"
    )

    # Engineering
    source_unit: EngineeringUnit = Field(
        default=EngineeringUnit.CELSIUS,
        description="Engineering unit in source"
    )
    target_unit: EngineeringUnit = Field(
        default=EngineeringUnit.CELSIUS,
        description="Desired engineering unit"
    )
    data_type: str = Field(
        default="Float64",
        description="Data type"
    )

    # Transformations
    transformations: TransformationChain = Field(
        default_factory=TransformationChain,
        description="Value transformations"
    )

    # Validation
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        description="Validation configuration"
    )

    # Metadata
    description: Optional[str] = Field(
        default=None,
        description="Mapping description"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Metadata tags for filtering"
    )

    # Status
    status: MappingStatus = Field(
        default=MappingStatus.ACTIVE,
        description="Mapping status"
    )
    enabled: bool = Field(
        default=True,
        description="Is mapping enabled"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp"
    )
    validated_at: Optional[datetime] = Field(
        default=None,
        description="Last validation timestamp"
    )

    def apply_transformations(self, value: float) -> float:
        """Apply all transformations to value."""
        # Apply unit conversion first if needed
        if self.source_unit != self.target_unit:
            converter = UnitConversion(
                from_unit=self.source_unit,
                to_unit=self.target_unit
            )
            value = converter.apply(value)

        # Apply transformation chain
        return self.transformations.apply(value)

    def validate_value(
        self,
        value: Optional[float],
        timestamp: Optional[datetime] = None
    ) -> Tuple[bool, List[str]]:
        """Validate a value against this mapping's rules."""
        return self.validation.validate(value, timestamp)


class TagMappingGroup(BaseModel):
    """Group of related tag mappings."""

    model_config = ConfigDict(frozen=False)

    group_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Group identifier"
    )
    group_name: str = Field(
        ...,
        description="Group name"
    )
    description: Optional[str] = Field(
        default=None,
        description="Group description"
    )

    # Group scope
    equipment_id: Optional[str] = Field(
        default=None,
        description="Equipment this group belongs to"
    )
    functional_location: Optional[str] = Field(
        default=None,
        description="Functional location"
    )
    source_system: Optional[TagSource] = Field(
        default=None,
        description="Source system for all mappings"
    )

    # Mappings
    mappings: List[TagMapping] = Field(
        default_factory=list,
        description="Mappings in this group"
    )

    # Status
    enabled: bool = Field(
        default=True,
        description="Group enabled"
    )

    def get_mapping_by_name(self, name: str) -> Optional[TagMapping]:
        """Get mapping by name."""
        for mapping in self.mappings:
            if mapping.mapping_name == name:
                return mapping
        return None

    def get_mapping_by_source_tag(self, source_tag: str) -> Optional[TagMapping]:
        """Get mapping by source tag name."""
        for mapping in self.mappings:
            if mapping.source_tag_name == source_tag:
                return mapping
        return None

    def get_mappings_by_type(
        self,
        measurement_type: MeasurementType
    ) -> List[TagMapping]:
        """Get all mappings of a specific type."""
        return [
            m for m in self.mappings
            if m.measurement_type == measurement_type
        ]


# =============================================================================
# Tag Mapping Manager
# =============================================================================


class TagMappingManager:
    """
    Manages tag mappings for the INSULSCAN agent.

    Provides:
    - CRUD operations for mappings
    - Lookup by various criteria
    - Validation of mappings
    - Import/export capabilities
    - Caching and performance optimization
    """

    def __init__(self) -> None:
        """Initialize tag mapping manager."""
        self._logger = logging.getLogger(f"{__name__}.Manager")

        # Storage
        self._mappings: Dict[str, TagMapping] = {}
        self._groups: Dict[str, TagMappingGroup] = {}

        # Indices for fast lookup
        self._by_source_tag: Dict[str, str] = {}  # source_tag -> mapping_id
        self._by_equipment: Dict[str, Set[str]] = {}  # equipment_id -> set(mapping_ids)
        self._by_measurement_type: Dict[MeasurementType, Set[str]] = {}

        # Statistics
        self._mappings_count = 0
        self._groups_count = 0

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def add_mapping(self, mapping: TagMapping) -> str:
        """
        Add a tag mapping.

        Args:
            mapping: Tag mapping to add

        Returns:
            Mapping ID
        """
        mapping_id = mapping.mapping_id

        # Store mapping
        self._mappings[mapping_id] = mapping

        # Update indices
        self._by_source_tag[mapping.source_tag_name] = mapping_id

        if mapping.equipment_id not in self._by_equipment:
            self._by_equipment[mapping.equipment_id] = set()
        self._by_equipment[mapping.equipment_id].add(mapping_id)

        if mapping.measurement_type not in self._by_measurement_type:
            self._by_measurement_type[mapping.measurement_type] = set()
        self._by_measurement_type[mapping.measurement_type].add(mapping_id)

        self._mappings_count += 1

        self._logger.info(
            f"Added mapping: {mapping.mapping_name} ({mapping_id})"
        )

        return mapping_id

    def update_mapping(self, mapping: TagMapping) -> bool:
        """
        Update an existing mapping.

        Args:
            mapping: Updated mapping

        Returns:
            True if updated
        """
        mapping_id = mapping.mapping_id

        if mapping_id not in self._mappings:
            self._logger.warning(f"Mapping not found: {mapping_id}")
            return False

        old_mapping = self._mappings[mapping_id]

        # Update indices if source tag changed
        if old_mapping.source_tag_name != mapping.source_tag_name:
            del self._by_source_tag[old_mapping.source_tag_name]
            self._by_source_tag[mapping.source_tag_name] = mapping_id

        # Update indices if equipment changed
        if old_mapping.equipment_id != mapping.equipment_id:
            self._by_equipment[old_mapping.equipment_id].discard(mapping_id)
            if mapping.equipment_id not in self._by_equipment:
                self._by_equipment[mapping.equipment_id] = set()
            self._by_equipment[mapping.equipment_id].add(mapping_id)

        # Update indices if measurement type changed
        if old_mapping.measurement_type != mapping.measurement_type:
            self._by_measurement_type[old_mapping.measurement_type].discard(mapping_id)
            if mapping.measurement_type not in self._by_measurement_type:
                self._by_measurement_type[mapping.measurement_type] = set()
            self._by_measurement_type[mapping.measurement_type].add(mapping_id)

        # Update timestamp
        mapping.updated_at = datetime.utcnow()

        # Store updated mapping
        self._mappings[mapping_id] = mapping

        self._logger.info(f"Updated mapping: {mapping.mapping_name}")

        return True

    def remove_mapping(self, mapping_id: str) -> bool:
        """
        Remove a mapping.

        Args:
            mapping_id: Mapping ID to remove

        Returns:
            True if removed
        """
        if mapping_id not in self._mappings:
            return False

        mapping = self._mappings[mapping_id]

        # Remove from indices
        del self._by_source_tag[mapping.source_tag_name]
        self._by_equipment[mapping.equipment_id].discard(mapping_id)
        self._by_measurement_type[mapping.measurement_type].discard(mapping_id)

        # Remove mapping
        del self._mappings[mapping_id]
        self._mappings_count -= 1

        self._logger.info(f"Removed mapping: {mapping.mapping_name}")

        return True

    def get_mapping(self, mapping_id: str) -> Optional[TagMapping]:
        """Get mapping by ID."""
        return self._mappings.get(mapping_id)

    def get_all_mappings(self) -> List[TagMapping]:
        """Get all mappings."""
        return list(self._mappings.values())

    # =========================================================================
    # Group Operations
    # =========================================================================

    def add_group(self, group: TagMappingGroup) -> str:
        """
        Add a mapping group.

        Args:
            group: Group to add

        Returns:
            Group ID
        """
        group_id = group.group_id
        self._groups[group_id] = group
        self._groups_count += 1

        # Add all mappings in the group
        for mapping in group.mappings:
            self.add_mapping(mapping)

        self._logger.info(f"Added group: {group.group_name} with {len(group.mappings)} mappings")

        return group_id

    def get_group(self, group_id: str) -> Optional[TagMappingGroup]:
        """Get group by ID."""
        return self._groups.get(group_id)

    def get_all_groups(self) -> List[TagMappingGroup]:
        """Get all groups."""
        return list(self._groups.values())

    # =========================================================================
    # Lookup Operations
    # =========================================================================

    def get_by_source_tag(self, source_tag: str) -> Optional[TagMapping]:
        """
        Get mapping by source tag name.

        Args:
            source_tag: Source tag name

        Returns:
            Tag mapping or None
        """
        mapping_id = self._by_source_tag.get(source_tag)
        if mapping_id:
            return self._mappings.get(mapping_id)
        return None

    def get_by_equipment(self, equipment_id: str) -> List[TagMapping]:
        """
        Get all mappings for an equipment.

        Args:
            equipment_id: Equipment ID

        Returns:
            List of mappings
        """
        mapping_ids = self._by_equipment.get(equipment_id, set())
        return [self._mappings[mid] for mid in mapping_ids if mid in self._mappings]

    def get_by_measurement_type(
        self,
        measurement_type: MeasurementType
    ) -> List[TagMapping]:
        """
        Get all mappings of a measurement type.

        Args:
            measurement_type: Measurement type

        Returns:
            List of mappings
        """
        mapping_ids = self._by_measurement_type.get(measurement_type, set())
        return [self._mappings[mid] for mid in mapping_ids if mid in self._mappings]

    def get_temperature_mappings(
        self,
        equipment_id: Optional[str] = None
    ) -> List[TagMapping]:
        """
        Get all temperature-related mappings.

        Args:
            equipment_id: Optional equipment filter

        Returns:
            List of temperature mappings
        """
        temp_types = [
            MeasurementType.SURFACE_TEMPERATURE,
            MeasurementType.OPERATING_TEMPERATURE,
            MeasurementType.AMBIENT_TEMPERATURE,
            MeasurementType.PROCESS_TEMPERATURE,
            MeasurementType.JACKET_TEMPERATURE,
            MeasurementType.SKIN_TEMPERATURE,
            MeasurementType.HOT_SPOT_TEMPERATURE,
            MeasurementType.COLD_SPOT_TEMPERATURE,
            MeasurementType.AVERAGE_TEMPERATURE,
        ]

        mappings = []
        for temp_type in temp_types:
            mappings.extend(self.get_by_measurement_type(temp_type))

        if equipment_id:
            mappings = [m for m in mappings if m.equipment_id == equipment_id]

        return mappings

    def search_mappings(
        self,
        query: str,
        source_system: Optional[TagSource] = None,
        status: Optional[MappingStatus] = None,
        enabled_only: bool = True
    ) -> List[TagMapping]:
        """
        Search mappings by various criteria.

        Args:
            query: Search query (searches name, description, tags)
            source_system: Filter by source system
            status: Filter by status
            enabled_only: Only return enabled mappings

        Returns:
            Matching mappings
        """
        results = []
        query_lower = query.lower()

        for mapping in self._mappings.values():
            # Apply filters
            if enabled_only and not mapping.enabled:
                continue
            if source_system and mapping.source_system != source_system:
                continue
            if status and mapping.status != status:
                continue

            # Search in fields
            if (query_lower in mapping.mapping_name.lower() or
                query_lower in mapping.source_tag_name.lower() or
                (mapping.description and query_lower in mapping.description.lower()) or
                any(query_lower in tag.lower() for tag in mapping.tags)):
                results.append(mapping)

        return results

    # =========================================================================
    # Validation
    # =========================================================================

    def validate_mapping(self, mapping: TagMapping) -> Tuple[bool, List[str]]:
        """
        Validate a tag mapping configuration.

        Args:
            mapping: Mapping to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Required fields
        if not mapping.source_tag_name:
            errors.append("Source tag name is required")
        if not mapping.equipment_id:
            errors.append("Equipment ID is required")

        # Check for duplicate source tag
        existing = self.get_by_source_tag(mapping.source_tag_name)
        if existing and existing.mapping_id != mapping.mapping_id:
            errors.append(f"Duplicate source tag: {mapping.source_tag_name}")

        # Validate range if specified
        if mapping.validation.range_validation:
            rv = mapping.validation.range_validation
            if rv.min_value is not None and rv.max_value is not None:
                if rv.min_value > rv.max_value:
                    errors.append("Minimum value cannot be greater than maximum")

        # Validate transformations
        if mapping.transformations.transformations:
            for i, transform in enumerate(mapping.transformations.transformations):
                if isinstance(transform, PolynomialTransformation):
                    if not transform.coefficients:
                        errors.append(f"Polynomial transformation {i} has no coefficients")

        is_valid = len(errors) == 0

        if is_valid:
            mapping.status = MappingStatus.ACTIVE
            mapping.validated_at = datetime.utcnow()
        else:
            mapping.status = MappingStatus.VALIDATION_FAILED

        return is_valid, errors

    def validate_all_mappings(self) -> Dict[str, Tuple[bool, List[str]]]:
        """
        Validate all mappings.

        Returns:
            Dictionary of mapping_id -> (is_valid, errors)
        """
        results = {}
        for mapping_id, mapping in self._mappings.items():
            results[mapping_id] = self.validate_mapping(mapping)
        return results

    # =========================================================================
    # Import/Export
    # =========================================================================

    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export all mappings to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "version": "1.0.0",
            "exported_at": datetime.utcnow().isoformat(),
            "mappings": [
                m.model_dump(mode='json') for m in self._mappings.values()
            ],
            "groups": [
                g.model_dump(mode='json') for g in self._groups.values()
            ],
        }

    def export_to_json(self, file_path: Path) -> None:
        """
        Export mappings to JSON file.

        Args:
            file_path: Output file path
        """
        data = self.export_to_dict()
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        self._logger.info(f"Exported {len(self._mappings)} mappings to {file_path}")

    def import_from_dict(self, data: Dict[str, Any]) -> int:
        """
        Import mappings from dictionary.

        Args:
            data: Dictionary with mappings

        Returns:
            Number of mappings imported
        """
        imported = 0

        # Import mappings
        for mapping_data in data.get("mappings", []):
            try:
                mapping = TagMapping(**mapping_data)
                self.add_mapping(mapping)
                imported += 1
            except Exception as e:
                self._logger.warning(f"Failed to import mapping: {e}")

        # Import groups
        for group_data in data.get("groups", []):
            try:
                group = TagMappingGroup(**group_data)
                self._groups[group.group_id] = group
                self._groups_count += 1
            except Exception as e:
                self._logger.warning(f"Failed to import group: {e}")

        self._logger.info(f"Imported {imported} mappings")

        return imported

    def import_from_json(self, file_path: Path) -> int:
        """
        Import mappings from JSON file.

        Args:
            file_path: Input file path

        Returns:
            Number of mappings imported
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        return self.import_from_dict(data)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get mapping statistics."""
        # Count by source system
        by_source = {}
        for mapping in self._mappings.values():
            source = mapping.source_system.value
            by_source[source] = by_source.get(source, 0) + 1

        # Count by measurement type
        by_type = {}
        for mapping in self._mappings.values():
            mtype = mapping.measurement_type.value
            by_type[mtype] = by_type.get(mtype, 0) + 1

        # Count by status
        by_status = {}
        for mapping in self._mappings.values():
            status = mapping.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_mappings": self._mappings_count,
            "total_groups": self._groups_count,
            "by_source_system": by_source,
            "by_measurement_type": by_type,
            "by_status": by_status,
            "equipment_count": len(self._by_equipment),
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_temperature_mapping(
    source_tag: str,
    equipment_id: str,
    measurement_type: MeasurementType = MeasurementType.SURFACE_TEMPERATURE,
    source_system: TagSource = TagSource.OPCUA,
    source_unit: EngineeringUnit = EngineeringUnit.CELSIUS,
    low_limit: Optional[float] = None,
    high_limit: Optional[float] = None,
    **kwargs
) -> TagMapping:
    """
    Create a temperature tag mapping.

    Args:
        source_tag: Source tag name
        equipment_id: Equipment ID
        measurement_type: Type of temperature measurement
        source_system: Source system
        source_unit: Source unit
        low_limit: Low limit for validation
        high_limit: High limit for validation
        **kwargs: Additional mapping properties

    Returns:
        Configured TagMapping
    """
    # Setup validation
    validation = ValidationConfig(
        range_validation=RangeValidation(
            min_value=low_limit or -273.15,  # Absolute zero
            max_value=high_limit or 1500.0,  # Reasonable max
        )
    )

    return TagMapping(
        mapping_name=f"{equipment_id}_{measurement_type.value}",
        source_tag_name=source_tag,
        source_system=source_system,
        equipment_id=equipment_id,
        measurement_type=measurement_type,
        source_unit=source_unit,
        target_unit=EngineeringUnit.CELSIUS,
        validation=validation,
        **kwargs
    )


def create_equipment_mapping_group(
    equipment_id: str,
    equipment_name: str,
    tags: Dict[MeasurementType, str],
    source_system: TagSource = TagSource.OPCUA,
    functional_location: Optional[str] = None,
) -> TagMappingGroup:
    """
    Create a mapping group for equipment with multiple sensors.

    Args:
        equipment_id: Equipment ID
        equipment_name: Equipment name
        tags: Dictionary of measurement type to source tag name
        source_system: Source system
        functional_location: Functional location

    Returns:
        Configured TagMappingGroup
    """
    mappings = []

    for measurement_type, source_tag in tags.items():
        mapping = create_temperature_mapping(
            source_tag=source_tag,
            equipment_id=equipment_id,
            measurement_type=measurement_type,
            source_system=source_system,
        )
        mappings.append(mapping)

    return TagMappingGroup(
        group_name=f"{equipment_name} Sensors",
        description=f"Temperature sensors for {equipment_name}",
        equipment_id=equipment_id,
        functional_location=functional_location,
        source_system=source_system,
        mappings=mappings,
    )


def create_tag_mapping_manager() -> TagMappingManager:
    """Create a new tag mapping manager instance."""
    return TagMappingManager()
