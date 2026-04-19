"""
Industrial Data Models for GreenLang.

This module provides Pydantic models for industrial protocol data structures
including tag values, metadata, alarms, and historical queries.

Models:
    - TagValue: Real-time tag value with timestamp and quality
    - TagMetadata: Engineering units, ranges, and tag configuration
    - AlarmEvent: Alarm and event data
    - HistoricalQuery: Time-series query specification
    - BatchReadRequest/Response: Batch operations

Example:
    >>> from integrations.industrial.data_models import TagValue, DataQuality
    >>>
    >>> tag = TagValue(
    ...     tag_id="TI-101",
    ...     value=75.5,
    ...     timestamp=datetime.utcnow(),
    ...     quality=DataQuality.GOOD
    ... )
"""

import logging
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Quality Codes
# =============================================================================


class DataQuality(IntEnum):
    """
    OPC-UA style data quality codes.

    Quality codes indicate the reliability and validity of a tag value.
    Based on OPC-UA StatusCode patterns.
    """

    # Good quality (0x00)
    GOOD = 0
    GOOD_LOCAL_OVERRIDE = 1
    GOOD_CASCADE_OVERRIDE = 2

    # Uncertain quality (0x40)
    UNCERTAIN = 64
    UNCERTAIN_LAST_USABLE = 65
    UNCERTAIN_SENSOR_NOT_ACCURATE = 66
    UNCERTAIN_EU_EXCEEDED = 67
    UNCERTAIN_SUBSTITUTED = 68

    # Bad quality (0x80)
    BAD = 128
    BAD_CONFIG_ERROR = 129
    BAD_NOT_CONNECTED = 130
    BAD_DEVICE_FAILURE = 131
    BAD_SENSOR_FAILURE = 132
    BAD_COMM_FAILURE = 133
    BAD_OUT_OF_SERVICE = 134
    BAD_WAITING_FOR_INIT = 135

    @property
    def is_good(self) -> bool:
        """Check if quality is good."""
        return self.value < 64

    @property
    def is_uncertain(self) -> bool:
        """Check if quality is uncertain."""
        return 64 <= self.value < 128

    @property
    def is_bad(self) -> bool:
        """Check if quality is bad."""
        return self.value >= 128


class AlarmSeverity(IntEnum):
    """Alarm severity levels following ISA-18.2."""

    DIAGNOSTIC = 0
    LOW = 250
    MEDIUM = 500
    HIGH = 750
    CRITICAL = 1000


class AlarmState(str, Enum):
    """Alarm state transitions."""

    NORMAL = "normal"
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"
    SHELVED = "shelved"
    OUT_OF_SERVICE = "out_of_service"


class DataType(str, Enum):
    """Supported data types for industrial tags."""

    BOOLEAN = "boolean"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    STRING = "string"
    DATETIME = "datetime"
    BYTE_ARRAY = "byte_array"


class AggregationType(str, Enum):
    """Time-series aggregation types."""

    RAW = "raw"
    INTERPOLATED = "interpolated"
    AVERAGE = "average"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    TOTAL = "total"
    COUNT = "count"
    STANDARD_DEVIATION = "stdev"
    VARIANCE = "variance"
    RANGE = "range"
    DELTA = "delta"
    TIME_WEIGHTED_AVERAGE = "time_weighted_average"
    FIRST = "first"
    LAST = "last"


# =============================================================================
# Core Data Models
# =============================================================================


class TagValue(BaseModel):
    """
    Real-time tag value with timestamp and quality.

    Represents a single point value from an industrial data source,
    including quality indicators and optional metadata.

    Attributes:
        tag_id: Unique tag identifier
        value: The tag value (any supported type)
        timestamp: When the value was sampled
        quality: Data quality code
        source_timestamp: Original source timestamp
        server_timestamp: Server receive timestamp
        status_code: Raw status code from source

    Example:
        >>> tag = TagValue(
        ...     tag_id="FIC-101.PV",
        ...     value=150.75,
        ...     timestamp=datetime.utcnow(),
        ...     quality=DataQuality.GOOD
        ... )
    """

    tag_id: str = Field(..., description="Unique tag identifier")
    value: Any = Field(..., description="Tag value")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Sample timestamp"
    )
    quality: DataQuality = Field(
        default=DataQuality.GOOD,
        description="Data quality code"
    )
    source_timestamp: Optional[datetime] = Field(
        None,
        description="Original source timestamp"
    )
    server_timestamp: Optional[datetime] = Field(
        None,
        description="Server receive timestamp"
    )
    status_code: Optional[int] = Field(
        None,
        description="Raw status code"
    )
    unit: Optional[str] = Field(
        None,
        description="Engineering unit"
    )

    @property
    def is_good_quality(self) -> bool:
        """Check if value has good quality."""
        return self.quality.is_good

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tag_id": self.tag_id,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "quality": self.quality.value,
            "quality_name": self.quality.name,
            "unit": self.unit,
        }


class TagMetadata(BaseModel):
    """
    Tag metadata including engineering units and configuration.

    Contains static configuration data for a tag including
    engineering unit conversion, alarm limits, and description.

    Attributes:
        tag_id: Unique tag identifier
        description: Human-readable description
        engineering_unit: Unit of measure (e.g., "degC", "kg/h")
        data_type: Native data type
        eu_low: Engineering unit low range
        eu_high: Engineering unit high range
        raw_low: Raw value low range
        raw_high: Raw value high range
        alarm_limits: Alarm setpoints

    Example:
        >>> metadata = TagMetadata(
        ...     tag_id="TI-101",
        ...     description="Reactor Temperature",
        ...     engineering_unit="degC",
        ...     eu_low=0.0,
        ...     eu_high=500.0
        ... )
    """

    tag_id: str = Field(..., description="Unique tag identifier")
    description: str = Field("", description="Tag description")
    engineering_unit: str = Field("", description="Engineering unit")
    data_type: DataType = Field(
        DataType.FLOAT64,
        description="Data type"
    )

    # Engineering unit range
    eu_low: Optional[float] = Field(None, description="EU low range")
    eu_high: Optional[float] = Field(None, description="EU high range")

    # Raw value range (for scaling)
    raw_low: Optional[float] = Field(None, description="Raw low range")
    raw_high: Optional[float] = Field(None, description="Raw high range")

    # Alarm limits
    alarm_high_high: Optional[float] = Field(None, description="High-high alarm")
    alarm_high: Optional[float] = Field(None, description="High alarm")
    alarm_low: Optional[float] = Field(None, description="Low alarm")
    alarm_low_low: Optional[float] = Field(None, description="Low-low alarm")

    # Additional metadata
    source_system: Optional[str] = Field(None, description="Source system")
    source_address: Optional[str] = Field(None, description="Source address")
    scan_rate_ms: Optional[int] = Field(None, description="Scan rate in ms")

    # Hierarchical tags
    parent_tag: Optional[str] = Field(None, description="Parent tag")
    area: Optional[str] = Field(None, description="Plant area")
    unit: Optional[str] = Field(None, description="Process unit")

    # Custom attributes
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom attributes"
    )

    def scale_raw_to_eu(self, raw_value: float) -> float:
        """
        Scale raw value to engineering units.

        Args:
            raw_value: Raw value from device

        Returns:
            Scaled engineering unit value
        """
        if None in (self.raw_low, self.raw_high, self.eu_low, self.eu_high):
            return raw_value

        raw_range = self.raw_high - self.raw_low
        eu_range = self.eu_high - self.eu_low

        if raw_range == 0:
            return self.eu_low

        return self.eu_low + (raw_value - self.raw_low) * eu_range / raw_range

    def scale_eu_to_raw(self, eu_value: float) -> float:
        """
        Scale engineering units to raw value.

        Args:
            eu_value: Engineering unit value

        Returns:
            Raw value for device
        """
        if None in (self.raw_low, self.raw_high, self.eu_low, self.eu_high):
            return eu_value

        raw_range = self.raw_high - self.raw_low
        eu_range = self.eu_high - self.eu_low

        if eu_range == 0:
            return self.raw_low

        return self.raw_low + (eu_value - self.eu_low) * raw_range / eu_range

    def check_alarm_status(self, value: float) -> Optional[str]:
        """
        Check value against alarm limits.

        Args:
            value: Current value

        Returns:
            Alarm level name or None if normal
        """
        if self.alarm_high_high is not None and value >= self.alarm_high_high:
            return "HIGH_HIGH"
        if self.alarm_high is not None and value >= self.alarm_high:
            return "HIGH"
        if self.alarm_low_low is not None and value <= self.alarm_low_low:
            return "LOW_LOW"
        if self.alarm_low is not None and value <= self.alarm_low:
            return "LOW"
        return None


class AlarmEvent(BaseModel):
    """
    Alarm and event data following ISA-18.2.

    Represents an alarm occurrence with state transitions,
    timestamps, and acknowledgment tracking.

    Attributes:
        alarm_id: Unique alarm identifier
        tag_id: Associated tag
        message: Alarm message text
        severity: Alarm severity level
        state: Current alarm state
        active_time: When alarm became active
        ack_time: When alarm was acknowledged
        clear_time: When alarm cleared

    Example:
        >>> alarm = AlarmEvent(
        ...     alarm_id="ALM-001",
        ...     tag_id="TI-101",
        ...     message="High temperature alarm",
        ...     severity=AlarmSeverity.HIGH,
        ...     state=AlarmState.ACTIVE
        ... )
    """

    alarm_id: str = Field(..., description="Unique alarm identifier")
    tag_id: str = Field(..., description="Associated tag")
    message: str = Field(..., description="Alarm message")
    severity: AlarmSeverity = Field(
        AlarmSeverity.MEDIUM,
        description="Severity level"
    )
    state: AlarmState = Field(
        AlarmState.NORMAL,
        description="Current state"
    )
    priority: int = Field(0, ge=0, le=999, description="Alarm priority")

    # Timestamps
    active_time: Optional[datetime] = Field(None, description="Activation time")
    ack_time: Optional[datetime] = Field(None, description="Acknowledgment time")
    clear_time: Optional[datetime] = Field(None, description="Clear time")
    return_time: Optional[datetime] = Field(None, description="Return to normal time")

    # Value information
    trigger_value: Optional[float] = Field(None, description="Value that triggered alarm")
    limit_value: Optional[float] = Field(None, description="Alarm limit value")

    # Acknowledgment
    ack_user: Optional[str] = Field(None, description="User who acknowledged")
    ack_comment: Optional[str] = Field(None, description="Acknowledgment comment")

    # Source information
    source_system: Optional[str] = Field(None, description="Source system")
    area: Optional[str] = Field(None, description="Plant area")

    @property
    def is_active(self) -> bool:
        """Check if alarm is currently active."""
        return self.state == AlarmState.ACTIVE

    @property
    def needs_acknowledgment(self) -> bool:
        """Check if alarm needs acknowledgment."""
        return self.state == AlarmState.ACTIVE and self.ack_time is None

    @property
    def duration(self) -> Optional[timedelta]:
        """Get alarm duration."""
        if self.active_time is None:
            return None
        end_time = self.clear_time or datetime.utcnow()
        return end_time - self.active_time


# =============================================================================
# Query Models
# =============================================================================


class HistoricalQuery(BaseModel):
    """
    Time-series historical query specification.

    Defines parameters for querying historical tag data
    from process historians.

    Attributes:
        tag_ids: List of tags to query
        start_time: Query start time
        end_time: Query end time
        aggregation: Aggregation type
        interval_ms: Aggregation interval
        max_points: Maximum points to return
        include_quality: Include quality data

    Example:
        >>> query = HistoricalQuery(
        ...     tag_ids=["TI-101", "FI-101"],
        ...     start_time=datetime(2024, 1, 1),
        ...     end_time=datetime(2024, 1, 2),
        ...     aggregation=AggregationType.AVERAGE,
        ...     interval_ms=60000
        ... )
    """

    tag_ids: List[str] = Field(..., min_length=1, description="Tags to query")
    start_time: datetime = Field(..., description="Query start time")
    end_time: datetime = Field(..., description="Query end time")
    aggregation: AggregationType = Field(
        AggregationType.RAW,
        description="Aggregation type"
    )
    interval_ms: Optional[int] = Field(
        None,
        ge=100,
        description="Aggregation interval in milliseconds"
    )
    max_points: int = Field(
        10000,
        ge=1,
        le=1000000,
        description="Maximum points per tag"
    )
    include_quality: bool = Field(True, description="Include quality data")
    filter_bad_quality: bool = Field(False, description="Filter bad quality")

    @field_validator("end_time")
    @classmethod
    def end_after_start(cls, v: datetime, info) -> datetime:
        """Validate end time is after start time."""
        start_time = info.data.get("start_time")
        if start_time and v <= start_time:
            raise ValueError("end_time must be after start_time")
        return v

    @property
    def time_range(self) -> timedelta:
        """Get query time range."""
        return self.end_time - self.start_time

    @property
    def expected_points(self) -> int:
        """Estimate expected number of points."""
        if self.aggregation == AggregationType.RAW:
            return self.max_points
        if self.interval_ms:
            total_ms = self.time_range.total_seconds() * 1000
            return min(int(total_ms / self.interval_ms), self.max_points)
        return self.max_points


class HistoricalResult(BaseModel):
    """
    Historical query result.

    Contains time-series data returned from a historical query.

    Attributes:
        tag_id: Tag identifier
        values: List of TagValue objects
        query: Original query specification
        start_time: Actual start time of data
        end_time: Actual end time of data
        point_count: Number of points returned
    """

    tag_id: str = Field(..., description="Tag identifier")
    values: List[TagValue] = Field(default_factory=list, description="Values")
    start_time: datetime = Field(..., description="Actual start time")
    end_time: datetime = Field(..., description="Actual end time")
    point_count: int = Field(0, description="Number of points")
    aggregation: AggregationType = Field(AggregationType.RAW)

    @property
    def is_empty(self) -> bool:
        """Check if result is empty."""
        return len(self.values) == 0

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate statistics for the result."""
        if self.is_empty:
            return {}

        numeric_values = [
            v.value for v in self.values
            if isinstance(v.value, (int, float)) and v.is_good_quality
        ]

        if not numeric_values:
            return {}

        return {
            "min": min(numeric_values),
            "max": max(numeric_values),
            "avg": sum(numeric_values) / len(numeric_values),
            "count": len(numeric_values),
            "first": numeric_values[0],
            "last": numeric_values[-1],
        }


# =============================================================================
# Batch Operation Models
# =============================================================================


class BatchReadRequest(BaseModel):
    """
    Batch read request for multiple tags.

    Attributes:
        tag_ids: List of tags to read
        include_metadata: Include tag metadata
        include_quality: Include quality info
    """

    tag_ids: List[str] = Field(..., min_length=1, description="Tags to read")
    include_metadata: bool = Field(False, description="Include metadata")
    include_quality: bool = Field(True, description="Include quality")


class BatchReadResponse(BaseModel):
    """
    Batch read response with multiple tag values.

    Attributes:
        values: Dictionary of tag_id to TagValue
        errors: Dictionary of tag_id to error message
        timestamp: Response timestamp
        success_count: Number of successful reads
        error_count: Number of failed reads
    """

    values: Dict[str, TagValue] = Field(
        default_factory=dict,
        description="Tag values"
    )
    errors: Dict[str, str] = Field(
        default_factory=dict,
        description="Read errors"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )

    @property
    def success_count(self) -> int:
        """Count of successful reads."""
        return len(self.values)

    @property
    def error_count(self) -> int:
        """Count of failed reads."""
        return len(self.errors)

    @property
    def total_count(self) -> int:
        """Total tags requested."""
        return self.success_count + self.error_count


class BatchWriteRequest(BaseModel):
    """
    Batch write request for multiple tags.

    Attributes:
        writes: Dictionary of tag_id to value
        validate_ranges: Validate against EU ranges
        synchronous: Wait for write confirmation
    """

    writes: Dict[str, Any] = Field(..., min_length=1, description="Write values")
    validate_ranges: bool = Field(True, description="Validate ranges")
    synchronous: bool = Field(True, description="Wait for confirmation")


class BatchWriteResponse(BaseModel):
    """
    Batch write response.

    Attributes:
        success: Dictionary of tag_id to success status
        errors: Dictionary of tag_id to error message
        timestamp: Response timestamp
    """

    success: Dict[str, bool] = Field(
        default_factory=dict,
        description="Write success status"
    )
    errors: Dict[str, str] = Field(
        default_factory=dict,
        description="Write errors"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )

    @property
    def success_count(self) -> int:
        """Count of successful writes."""
        return sum(1 for v in self.success.values() if v)

    @property
    def error_count(self) -> int:
        """Count of failed writes."""
        return len(self.errors)


# =============================================================================
# Subscription Models
# =============================================================================


class SubscriptionConfig(BaseModel):
    """
    Subscription configuration for real-time data.

    Attributes:
        tag_ids: Tags to subscribe to
        publishing_interval_ms: Data publishing interval
        sampling_interval_ms: Device sampling interval
        queue_size: Sample queue size
        discard_oldest: Discard oldest on overflow
        deadband_type: Deadband type (absolute/percent)
        deadband_value: Deadband value
    """

    tag_ids: List[str] = Field(..., min_length=1, description="Tags to subscribe")
    publishing_interval_ms: int = Field(
        1000,
        ge=10,
        description="Publishing interval in ms"
    )
    sampling_interval_ms: int = Field(
        100,
        ge=1,
        description="Sampling interval in ms"
    )
    queue_size: int = Field(10, ge=1, le=1000, description="Queue size")
    discard_oldest: bool = Field(True, description="Discard oldest on overflow")
    deadband_type: Optional[str] = Field(None, description="Deadband type")
    deadband_value: Optional[float] = Field(None, description="Deadband value")


class SubscriptionStatus(BaseModel):
    """
    Subscription status information.

    Attributes:
        subscription_id: Unique subscription identifier
        state: Current subscription state
        tag_count: Number of subscribed tags
        created_at: When subscription was created
        last_update: Last data received
        samples_received: Total samples received
    """

    subscription_id: str = Field(..., description="Subscription ID")
    state: str = Field(..., description="Subscription state")
    tag_count: int = Field(0, description="Number of tags")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation time"
    )
    last_update: Optional[datetime] = Field(None, description="Last update")
    samples_received: int = Field(0, description="Total samples")
    errors: int = Field(0, description="Error count")


# =============================================================================
# Connection Models
# =============================================================================


class ConnectionState(str, Enum):
    """Connection state enumeration."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    CLOSING = "closing"


class ConnectionMetrics(BaseModel):
    """
    Connection metrics and statistics.

    Attributes:
        state: Current connection state
        connected_since: When connection was established
        reconnect_count: Number of reconnections
        total_requests: Total requests made
        failed_requests: Failed request count
        avg_response_ms: Average response time
    """

    state: ConnectionState = Field(
        ConnectionState.DISCONNECTED,
        description="Connection state"
    )
    connected_since: Optional[datetime] = Field(None, description="Connected since")
    reconnect_count: int = Field(0, description="Reconnection count")
    total_requests: int = Field(0, description="Total requests")
    failed_requests: int = Field(0, description="Failed requests")
    avg_response_ms: float = Field(0.0, description="Average response time")
    last_error: Optional[str] = Field(None, description="Last error message")
    last_error_time: Optional[datetime] = Field(None, description="Last error time")

    @property
    def success_rate(self) -> float:
        """Calculate request success rate."""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests

    @property
    def uptime(self) -> Optional[timedelta]:
        """Calculate connection uptime."""
        if self.connected_since is None:
            return None
        return datetime.utcnow() - self.connected_since


# =============================================================================
# Unit Conversion
# =============================================================================


class UnitConversion(BaseModel):
    """
    Unit conversion specification.

    Defines conversion between engineering units.

    Attributes:
        from_unit: Source unit
        to_unit: Target unit
        multiplier: Conversion multiplier
        offset: Conversion offset
    """

    from_unit: str = Field(..., description="Source unit")
    to_unit: str = Field(..., description="Target unit")
    multiplier: float = Field(1.0, description="Conversion multiplier")
    offset: float = Field(0.0, description="Conversion offset")

    def convert(self, value: float) -> float:
        """Convert value from source to target unit."""
        return value * self.multiplier + self.offset

    def reverse(self, value: float) -> float:
        """Convert value from target to source unit."""
        return (value - self.offset) / self.multiplier


# Common unit conversions
UNIT_CONVERSIONS: Dict[str, UnitConversion] = {
    "degC_to_degF": UnitConversion(
        from_unit="degC",
        to_unit="degF",
        multiplier=1.8,
        offset=32.0
    ),
    "degF_to_degC": UnitConversion(
        from_unit="degF",
        to_unit="degC",
        multiplier=0.5556,
        offset=-17.7778
    ),
    "bar_to_psi": UnitConversion(
        from_unit="bar",
        to_unit="psi",
        multiplier=14.5038
    ),
    "psi_to_bar": UnitConversion(
        from_unit="psi",
        to_unit="bar",
        multiplier=0.0689476
    ),
    "kg_to_lb": UnitConversion(
        from_unit="kg",
        to_unit="lb",
        multiplier=2.20462
    ),
    "lb_to_kg": UnitConversion(
        from_unit="lb",
        to_unit="kg",
        multiplier=0.453592
    ),
    "m3_to_gal": UnitConversion(
        from_unit="m3",
        to_unit="gal",
        multiplier=264.172
    ),
    "gal_to_m3": UnitConversion(
        from_unit="gal",
        to_unit="m3",
        multiplier=0.00378541
    ),
    "kW_to_hp": UnitConversion(
        from_unit="kW",
        to_unit="hp",
        multiplier=1.34102
    ),
    "hp_to_kW": UnitConversion(
        from_unit="hp",
        to_unit="kW",
        multiplier=0.7457
    ),
}


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert value between units.

    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Converted value

    Raises:
        ValueError: If conversion not supported
    """
    if from_unit == to_unit:
        return value

    key = f"{from_unit}_to_{to_unit}"
    conversion = UNIT_CONVERSIONS.get(key)

    if conversion is None:
        raise ValueError(f"No conversion defined from {from_unit} to {to_unit}")

    return conversion.convert(value)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Quality
    "DataQuality",
    "AlarmSeverity",
    "AlarmState",
    "DataType",
    "AggregationType",
    # Core models
    "TagValue",
    "TagMetadata",
    "AlarmEvent",
    # Query models
    "HistoricalQuery",
    "HistoricalResult",
    # Batch models
    "BatchReadRequest",
    "BatchReadResponse",
    "BatchWriteRequest",
    "BatchWriteResponse",
    # Subscription models
    "SubscriptionConfig",
    "SubscriptionStatus",
    # Connection models
    "ConnectionState",
    "ConnectionMetrics",
    # Unit conversion
    "UnitConversion",
    "UNIT_CONVERSIONS",
    "convert_units",
]
