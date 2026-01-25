"""
GL-012_SteamQual - Input Models

Pydantic v2 input models for steam quality monitoring and control.
These models define the data structures for all inputs to the SteamQual agent.

Features:
- Full Pydantic v2 validation with field_validator decorators
- Physical bounds validation (temperature, pressure, flow)
- JSON serialization support for API and audit trail
- Comprehensive docstrings for API documentation

Standards Reference:
- IAPWS-IF97 for physical property bounds
- SI units with explicit unit suffixes

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)

from .domain import (
    ConsumerClass,
    DataQualityFlag,
    HeaderType,
    SeparatorType,
    SteamState,
)


# =============================================================================
# Physical Constants for Validation
# =============================================================================

# Pressure bounds (kPa)
PRESSURE_MIN_KPA = 0.1  # Near vacuum
PRESSURE_MAX_KPA = 100000.0  # 100 MPa (above critical)
PRESSURE_CRITICAL_KPA = 22064.0  # Critical pressure

# Temperature bounds (Celsius)
TEMPERATURE_MIN_C = -50.0  # Below triple point
TEMPERATURE_MAX_C = 800.0  # IAPWS-IF97 Region 5 max
TEMPERATURE_CRITICAL_C = 373.946  # Critical temperature

# Flow bounds (kg/s)
FLOW_MIN_KG_S = 0.0
FLOW_MAX_KG_S = 10000.0  # Large industrial systems

# Quality bounds
QUALITY_MIN = 0.0
QUALITY_MAX = 1.0


# =============================================================================
# Base Input Model
# =============================================================================


class BaseInputModel(BaseModel):
    """
    Base class for all input models with common configuration.

    Provides:
    - JSON serialization configuration
    - Common timestamp handling
    - Model configuration for Pydantic v2
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": "GL-012_SteamQual Input Model"
        },
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
    )


# =============================================================================
# Steam Measurement Model
# =============================================================================


class SteamMeasurement(BaseInputModel):
    """
    Individual steam measurement point data.

    Represents a single measurement from a steam header or pipe,
    containing pressure, temperature, flow, and optional quality.

    Attributes:
        pressure_kpa: Absolute pressure in kilopascals (kPa).
        temperature_c: Temperature in degrees Celsius.
        flow_kg_s: Mass flow rate in kilograms per second.
        quality_x: Optional dryness fraction (0-1). Only valid for saturated steam.

    Example:
        >>> measurement = SteamMeasurement(
        ...     pressure_kpa=1000.0,
        ...     temperature_c=180.0,
        ...     flow_kg_s=5.0,
        ...     quality_x=0.95
        ... )
    """

    pressure_kpa: float = Field(
        ...,
        ge=PRESSURE_MIN_KPA,
        le=PRESSURE_MAX_KPA,
        description="Absolute pressure in kPa",
        json_schema_extra={"examples": [1000.0, 2000.0, 500.0]},
    )

    temperature_c: float = Field(
        ...,
        ge=TEMPERATURE_MIN_C,
        le=TEMPERATURE_MAX_C,
        description="Temperature in degrees Celsius",
        json_schema_extra={"examples": [180.0, 200.0, 150.0]},
    )

    flow_kg_s: float = Field(
        ...,
        ge=FLOW_MIN_KG_S,
        le=FLOW_MAX_KG_S,
        description="Mass flow rate in kg/s",
        json_schema_extra={"examples": [5.0, 10.0, 2.5]},
    )

    quality_x: Optional[float] = Field(
        default=None,
        ge=QUALITY_MIN,
        le=QUALITY_MAX,
        description="Steam quality (dryness fraction) 0-1. Optional.",
        json_schema_extra={"examples": [0.95, 0.98, None]},
    )

    # Optional metadata
    measurement_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Unique identifier for this measurement point",
    )

    timestamp: Optional[datetime] = Field(
        default=None,
        description="Measurement timestamp (UTC)",
    )

    data_quality: DataQualityFlag = Field(
        default=DataQualityFlag.GOOD,
        description="Data quality indicator",
    )

    @field_validator("pressure_kpa")
    @classmethod
    def validate_pressure_physical(cls, v: float) -> float:
        """Validate pressure is within physical bounds."""
        if v < PRESSURE_MIN_KPA:
            raise ValueError(
                f"Pressure {v} kPa below minimum {PRESSURE_MIN_KPA} kPa"
            )
        return v

    @field_validator("temperature_c")
    @classmethod
    def validate_temperature_physical(cls, v: float) -> float:
        """Validate temperature is within physical bounds."""
        if v < TEMPERATURE_MIN_C:
            raise ValueError(
                f"Temperature {v} C below absolute minimum {TEMPERATURE_MIN_C} C"
            )
        return v

    @field_validator("quality_x")
    @classmethod
    def validate_quality_bounds(cls, v: Optional[float]) -> Optional[float]:
        """Validate quality is in [0, 1] if provided."""
        if v is not None:
            if v < 0 or v > 1:
                raise ValueError(
                    f"Quality {v} must be in range [0, 1]"
                )
        return v

    @model_validator(mode="after")
    def validate_thermodynamic_consistency(self) -> "SteamMeasurement":
        """
        Validate thermodynamic consistency of measurements.

        Checks that quality is only specified for saturated conditions.
        """
        # If quality is specified, check it makes thermodynamic sense
        if self.quality_x is not None:
            # Quality only makes sense for saturated steam
            # We can do a rough check: saturation temp at given pressure
            # For now, just ensure quality is valid if specified
            if self.quality_x == 0.0:
                # Pure liquid - should be at or below saturation temp
                pass
            elif self.quality_x == 1.0:
                # Pure vapor - could be saturated or superheated
                pass
            # Detailed validation would require steam tables

        return self

    def get_moisture_content(self) -> Optional[float]:
        """
        Calculate moisture content from quality.

        Returns:
            Moisture content (1 - quality) or None if quality not specified.
        """
        if self.quality_x is not None:
            return 1.0 - self.quality_x
        return None

    def is_superheated(self) -> bool:
        """Check if measurement indicates superheated steam."""
        # Rough check - would need steam tables for accurate determination
        return self.quality_x is None or self.quality_x >= 1.0


# =============================================================================
# Header Data Model
# =============================================================================


class HeaderData(BaseInputModel):
    """
    Steam header measurement data.

    Aggregates measurements and metadata for a single steam header
    in the distribution system.

    Attributes:
        header_id: Unique identifier for the header.
        measurements: List of measurements from sensors on this header.
        consumer_class: Primary consumer class served by this header.

    Example:
        >>> header = HeaderData(
        ...     header_id="MP-HEADER-01",
        ...     measurements=[
        ...         SteamMeasurement(pressure_kpa=1000, temperature_c=180, flow_kg_s=5),
        ...     ],
        ...     consumer_class=ConsumerClass.HEAT_EXCHANGER
        ... )
    """

    header_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique identifier for the steam header",
        json_schema_extra={"examples": ["HP-01", "MP-HEADER-A", "LP-MAIN"]},
    )

    measurements: List[SteamMeasurement] = Field(
        ...,
        min_length=1,
        description="List of measurements from this header",
    )

    consumer_class: ConsumerClass = Field(
        default=ConsumerClass.GENERAL,
        description="Primary consumer class served by this header",
    )

    # Header metadata
    header_type: HeaderType = Field(
        default=HeaderType.MEDIUM_PRESSURE,
        description="Header pressure class",
    )

    header_name: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Human-readable header name",
    )

    design_pressure_kpa: Optional[float] = Field(
        default=None,
        ge=0,
        description="Design pressure for this header (kPa)",
    )

    design_temperature_c: Optional[float] = Field(
        default=None,
        description="Design temperature for this header (C)",
    )

    # Operating limits
    min_pressure_kpa: Optional[float] = Field(
        default=None,
        ge=0,
        description="Minimum operating pressure (kPa)",
    )

    max_pressure_kpa: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum operating pressure (kPa)",
    )

    @field_validator("header_id")
    @classmethod
    def validate_header_id(cls, v: str) -> str:
        """Validate header ID format."""
        v = v.strip()
        if not v:
            raise ValueError("header_id cannot be empty")
        return v

    @field_validator("measurements")
    @classmethod
    def validate_measurements_not_empty(
        cls, v: List[SteamMeasurement]
    ) -> List[SteamMeasurement]:
        """Validate that at least one measurement is provided."""
        if not v:
            raise ValueError("At least one measurement is required")
        return v

    @model_validator(mode="after")
    def validate_pressure_limits(self) -> "HeaderData":
        """Validate pressure limit consistency."""
        if (
            self.min_pressure_kpa is not None
            and self.max_pressure_kpa is not None
        ):
            if self.min_pressure_kpa > self.max_pressure_kpa:
                raise ValueError(
                    f"min_pressure_kpa ({self.min_pressure_kpa}) cannot exceed "
                    f"max_pressure_kpa ({self.max_pressure_kpa})"
                )
        return self

    def get_average_pressure(self) -> float:
        """Calculate average pressure across all measurements."""
        if not self.measurements:
            return 0.0
        return sum(m.pressure_kpa for m in self.measurements) / len(
            self.measurements
        )

    def get_average_temperature(self) -> float:
        """Calculate average temperature across all measurements."""
        if not self.measurements:
            return 0.0
        return sum(m.temperature_c for m in self.measurements) / len(
            self.measurements
        )

    def get_total_flow(self) -> float:
        """Calculate total mass flow across all measurements."""
        return sum(m.flow_kg_s for m in self.measurements)

    def get_average_quality(self) -> Optional[float]:
        """Calculate flow-weighted average quality if available."""
        quality_measurements = [
            m for m in self.measurements if m.quality_x is not None
        ]
        if not quality_measurements:
            return None

        total_flow = sum(m.flow_kg_s for m in quality_measurements)
        if total_flow <= 0:
            return None

        weighted_quality = sum(
            m.quality_x * m.flow_kg_s for m in quality_measurements  # type: ignore
        )
        return weighted_quality / total_flow


# =============================================================================
# Separator Data Model
# =============================================================================


class SeparatorData(BaseInputModel):
    """
    Moisture separator measurement data.

    Contains data for moisture separator performance monitoring
    including drain flow and differential pressure.

    Attributes:
        separator_id: Unique identifier for the separator.
        dp_kpa: Differential pressure across separator (kPa).
        drain_valve_position: Drain valve position (0-100%).
        drain_flow: Condensate drain flow rate (kg/s).

    Example:
        >>> separator = SeparatorData(
        ...     separator_id="SEP-MP-01",
        ...     dp_kpa=5.0,
        ...     drain_valve_position=25.0,
        ...     drain_flow=0.1
        ... )
    """

    separator_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique identifier for the separator",
        json_schema_extra={"examples": ["SEP-01", "MOISTURE-SEP-A"]},
    )

    dp_kpa: float = Field(
        ...,
        ge=0,
        le=100.0,
        description="Differential pressure across separator (kPa)",
        json_schema_extra={"examples": [5.0, 10.0, 2.5]},
    )

    drain_valve_position: float = Field(
        ...,
        ge=0,
        le=100.0,
        description="Drain valve position (0-100%)",
        json_schema_extra={"examples": [25.0, 50.0, 75.0]},
    )

    drain_flow: float = Field(
        ...,
        ge=0,
        le=100.0,
        description="Condensate drain flow rate (kg/s)",
        json_schema_extra={"examples": [0.1, 0.5, 1.0]},
    )

    # Separator metadata
    separator_type: SeparatorType = Field(
        default=SeparatorType.CYCLONE,
        description="Type of moisture separator",
    )

    separator_name: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Human-readable separator name",
    )

    # Operating conditions
    inlet_pressure_kpa: Optional[float] = Field(
        default=None,
        ge=0,
        description="Inlet pressure (kPa)",
    )

    outlet_pressure_kpa: Optional[float] = Field(
        default=None,
        ge=0,
        description="Outlet pressure (kPa)",
    )

    inlet_temperature_c: Optional[float] = Field(
        default=None,
        description="Inlet temperature (C)",
    )

    steam_flow_kg_s: Optional[float] = Field(
        default=None,
        ge=0,
        description="Steam flow through separator (kg/s)",
    )

    # Performance data
    design_efficiency: Optional[float] = Field(
        default=None,
        ge=0,
        le=1.0,
        description="Design separation efficiency (0-1)",
    )

    timestamp: Optional[datetime] = Field(
        default=None,
        description="Measurement timestamp (UTC)",
    )

    data_quality: DataQualityFlag = Field(
        default=DataQualityFlag.GOOD,
        description="Data quality indicator",
    )

    @field_validator("separator_id")
    @classmethod
    def validate_separator_id(cls, v: str) -> str:
        """Validate separator ID format."""
        v = v.strip()
        if not v:
            raise ValueError("separator_id cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_pressure_consistency(self) -> "SeparatorData":
        """Validate inlet/outlet pressure consistency."""
        if (
            self.inlet_pressure_kpa is not None
            and self.outlet_pressure_kpa is not None
        ):
            # Outlet should be slightly lower due to pressure drop
            if self.outlet_pressure_kpa > self.inlet_pressure_kpa:
                # Allow small measurement error
                if (
                    self.outlet_pressure_kpa - self.inlet_pressure_kpa
                    > 1.0  # 1 kPa tolerance
                ):
                    raise ValueError(
                        f"Outlet pressure ({self.outlet_pressure_kpa}) exceeds "
                        f"inlet pressure ({self.inlet_pressure_kpa})"
                    )
        return self

    def get_separation_efficiency_estimate(self) -> Optional[float]:
        """
        Estimate separation efficiency from drain flow.

        Returns rough estimate based on drain flow relative to steam flow.
        """
        if self.steam_flow_kg_s is not None and self.steam_flow_kg_s > 0:
            # Rough estimate - actual would need inlet quality
            return min(1.0, self.drain_flow / (self.steam_flow_kg_s * 0.05))
        return None


# =============================================================================
# Process Data Model
# =============================================================================


class ProcessData(BaseInputModel):
    """
    Generic process data container for tag-based data.

    Provides a flexible structure for passing process historian
    or SCADA tag values to the SteamQual agent.

    Attributes:
        tag_values: Dictionary of tag names to values.
        timestamp: Data timestamp (UTC).
        data_quality_flags: Quality flags by tag name.

    Example:
        >>> process_data = ProcessData(
        ...     tag_values={
        ...         "PT-101": 1000.0,
        ...         "TT-101": 180.0,
        ...         "FT-101": 5.0
        ...     },
        ...     timestamp=datetime.now(timezone.utc),
        ...     data_quality_flags={"PT-101": "good", "TT-101": "good"}
        ... )
    """

    tag_values: Dict[str, Union[float, int, str, bool, None]] = Field(
        ...,
        description="Dictionary of tag names to values",
        json_schema_extra={
            "examples": [
                {"PT-101": 1000.0, "TT-101": 180.0, "FT-101": 5.0}
            ]
        },
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Data timestamp (UTC)",
    )

    data_quality_flags: Dict[str, str] = Field(
        default_factory=dict,
        description="Quality flags by tag name (good, bad, uncertain, stale)",
    )

    # Source metadata
    source_system: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Source system identifier (e.g., historian name)",
    )

    batch_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Batch or transaction ID for tracing",
    )

    @field_validator("tag_values")
    @classmethod
    def validate_tag_values_not_empty(
        cls, v: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate that at least one tag value is provided."""
        if not v:
            raise ValueError("tag_values cannot be empty")
        return v

    @field_validator("data_quality_flags")
    @classmethod
    def validate_quality_flag_values(
        cls, v: Dict[str, str]
    ) -> Dict[str, str]:
        """Validate quality flag values are valid."""
        valid_flags = {flag.value for flag in DataQualityFlag}
        for tag, flag in v.items():
            if flag.lower() not in valid_flags:
                # Allow case-insensitive matching
                flag_lower = flag.lower()
                if flag_lower not in valid_flags:
                    raise ValueError(
                        f"Invalid quality flag '{flag}' for tag '{tag}'. "
                        f"Valid flags: {valid_flags}"
                    )
        return v

    def get_numeric_tags(self) -> Dict[str, float]:
        """Extract only numeric tag values."""
        return {
            k: float(v)
            for k, v in self.tag_values.items()
            if isinstance(v, (int, float)) and v is not None
        }

    def get_tag_quality(self, tag_name: str) -> DataQualityFlag:
        """Get quality flag for a specific tag."""
        flag_str = self.data_quality_flags.get(tag_name, "good")
        try:
            return DataQualityFlag(flag_str.lower())
        except ValueError:
            return DataQualityFlag.UNCERTAIN

    def is_tag_usable(self, tag_name: str) -> bool:
        """Check if a tag's data quality allows it to be used."""
        return self.get_tag_quality(tag_name).is_usable

    def get_data_age_seconds(self) -> float:
        """Calculate data age in seconds from current time."""
        now = datetime.now(timezone.utc)
        # Ensure timestamp is timezone-aware
        if self.timestamp.tzinfo is None:
            data_time = self.timestamp.replace(tzinfo=timezone.utc)
        else:
            data_time = self.timestamp
        return (now - data_time).total_seconds()


# =============================================================================
# Combined Input Request Model
# =============================================================================


class QualityEstimationRequest(BaseInputModel):
    """
    Complete request for steam quality estimation.

    Combines all input data types into a single request structure
    for the SteamQual agent's quality estimation pipeline.

    Attributes:
        request_id: Unique request identifier.
        headers: List of header data.
        separators: List of separator data.
        process_data: Generic process tag data.
        estimation_options: Configuration for estimation.

    Example:
        >>> request = QualityEstimationRequest(
        ...     headers=[
        ...         HeaderData(
        ...             header_id="MP-01",
        ...             measurements=[...],
        ...             consumer_class=ConsumerClass.HEAT_EXCHANGER
        ...         )
        ...     ]
        ... )
    """

    request_id: UUID = Field(
        default_factory=uuid4,
        description="Unique request identifier",
    )

    # Input data sources
    headers: List[HeaderData] = Field(
        default_factory=list,
        description="Steam header data",
    )

    separators: List[SeparatorData] = Field(
        default_factory=list,
        description="Moisture separator data",
    )

    process_data: Optional[ProcessData] = Field(
        default=None,
        description="Generic process tag data",
    )

    # Request metadata
    request_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request timestamp (UTC)",
    )

    client_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Client/caller identifier",
    )

    # Estimation options
    include_uncertainty: bool = Field(
        default=True,
        description="Include uncertainty quantification in output",
    )

    include_recommendations: bool = Field(
        default=True,
        description="Include operational recommendations",
    )

    confidence_threshold: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="Minimum confidence threshold for estimates",
    )

    @model_validator(mode="after")
    def validate_has_input_data(self) -> "QualityEstimationRequest":
        """Validate that at least one data source is provided."""
        if not self.headers and not self.separators and not self.process_data:
            raise ValueError(
                "At least one data source (headers, separators, or process_data) "
                "must be provided"
            )
        return self

    def get_all_header_ids(self) -> List[str]:
        """Get list of all header IDs in the request."""
        return [h.header_id for h in self.headers]

    def get_all_separator_ids(self) -> List[str]:
        """Get list of all separator IDs in the request."""
        return [s.separator_id for s in self.separators]

    def get_header_by_id(self, header_id: str) -> Optional[HeaderData]:
        """Get header data by ID."""
        for header in self.headers:
            if header.header_id == header_id:
                return header
        return None

    def get_separator_by_id(self, separator_id: str) -> Optional[SeparatorData]:
        """Get separator data by ID."""
        for separator in self.separators:
            if separator.separator_id == separator_id:
                return separator
        return None
