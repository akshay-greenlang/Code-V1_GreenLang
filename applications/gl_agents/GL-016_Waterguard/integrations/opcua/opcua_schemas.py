"""
GL-016 Waterguard OPC-UA Schemas

Data models for OPC-UA tag values, subscriptions, and quality codes.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum, IntFlag
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# OPC-UA Quality Codes
# =============================================================================

class OPCUAQualityBits(IntFlag):
    """OPC-UA quality bit flags per UA Specification."""
    GOOD = 0x00000000
    UNCERTAIN = 0x40000000
    BAD = 0x80000000

    # Good sub-codes
    GOOD_LOCAL_OVERRIDE = 0x00D80000
    GOOD_CLAMPED = 0x00300000

    # Uncertain sub-codes
    UNCERTAIN_INITIAL_VALUE = 0x40910000
    UNCERTAIN_SENSOR_CAL = 0x40A20000
    UNCERTAIN_LAST_USABLE_VALUE = 0x408F0000
    UNCERTAIN_OUT_OF_RANGE = 0x40A30000

    # Bad sub-codes
    BAD_CONFIG_ERROR = 0x80870000
    BAD_NOT_CONNECTED = 0x808A0000
    BAD_DEVICE_FAILURE = 0x808B0000
    BAD_SENSOR_FAILURE = 0x808C0000
    BAD_LAST_KNOWN_VALUE = 0x808D0000
    BAD_COMM_FAILURE = 0x808E0000
    BAD_OUT_OF_SERVICE = 0x808F0000
    BAD_WAITING_FOR_INITIAL_DATA = 0x80910000


class OPCUAQuality(str, Enum):
    """Simplified OPC-UA quality categories."""
    GOOD = "good"
    GOOD_CLAMPED = "good_clamped"
    GOOD_LOCAL_OVERRIDE = "good_local_override"
    UNCERTAIN = "uncertain"
    UNCERTAIN_INITIAL = "uncertain_initial"
    UNCERTAIN_SENSOR = "uncertain_sensor"
    UNCERTAIN_LAST_USABLE = "uncertain_last_usable"
    BAD = "bad"
    BAD_NOT_CONNECTED = "bad_not_connected"
    BAD_DEVICE_FAILURE = "bad_device_failure"
    BAD_SENSOR_FAILURE = "bad_sensor_failure"
    BAD_COMM_FAILURE = "bad_comm_failure"
    BAD_OUT_OF_SERVICE = "bad_out_of_service"

    @classmethod
    def from_status_code(cls, status_code: int) -> "OPCUAQuality":
        """Convert OPC-UA status code to quality enum."""
        # Check major quality bits
        if status_code & 0xC0000000 == 0:
            # Good
            if status_code & 0x00300000:
                return cls.GOOD_CLAMPED
            elif status_code & 0x00D80000:
                return cls.GOOD_LOCAL_OVERRIDE
            return cls.GOOD

        elif status_code & 0xC0000000 == 0x40000000:
            # Uncertain
            if status_code & 0x00910000:
                return cls.UNCERTAIN_INITIAL
            elif status_code & 0x00A20000:
                return cls.UNCERTAIN_SENSOR
            elif status_code & 0x008F0000:
                return cls.UNCERTAIN_LAST_USABLE
            return cls.UNCERTAIN

        else:
            # Bad
            if status_code & 0x008A0000:
                return cls.BAD_NOT_CONNECTED
            elif status_code & 0x008B0000:
                return cls.BAD_DEVICE_FAILURE
            elif status_code & 0x008C0000:
                return cls.BAD_SENSOR_FAILURE
            elif status_code & 0x008E0000:
                return cls.BAD_COMM_FAILURE
            elif status_code & 0x008F0000:
                return cls.BAD_OUT_OF_SERVICE
            return cls.BAD

    @property
    def is_good(self) -> bool:
        """Check if quality is good."""
        return self.value.startswith("good")

    @property
    def is_uncertain(self) -> bool:
        """Check if quality is uncertain."""
        return self.value.startswith("uncertain")

    @property
    def is_bad(self) -> bool:
        """Check if quality is bad."""
        return self.value.startswith("bad")


# =============================================================================
# Tag Value
# =============================================================================

class TagValue(BaseModel):
    """
    OPC-UA tag value with full quality and timestamp metadata.

    Represents a single value read from an OPC-UA server with
    all associated quality information for proper handling.
    """

    # Identification
    node_id: str = Field(..., description="OPC-UA node ID (e.g., 'ns=2;s=AI_PO4_001')")
    tag_name: str = Field(..., description="Human-readable tag name")

    # Value
    value: Union[float, int, bool, str, None] = Field(
        ...,
        description="Tag value (type depends on tag)"
    )
    data_type: str = Field(
        default="Float",
        description="OPC-UA data type (Float, Int32, Boolean, String)"
    )

    # Timestamps
    source_timestamp: datetime = Field(
        ...,
        description="Timestamp from source device (PLC/DCS)"
    )
    server_timestamp: datetime = Field(
        ...,
        description="Timestamp when server received value"
    )
    read_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when value was read by client"
    )

    # Quality
    quality: OPCUAQuality = Field(
        default=OPCUAQuality.GOOD,
        description="Quality code"
    )
    status_code: int = Field(
        default=0,
        description="Raw OPC-UA status code"
    )

    # Engineering units
    engineering_units: str = Field(
        default="",
        description="Engineering units (e.g., 'ppm', 'degC')"
    )
    eu_range_low: Optional[float] = Field(
        default=None,
        description="Engineering unit range low"
    )
    eu_range_high: Optional[float] = Field(
        default=None,
        description="Engineering unit range high"
    )

    # Metadata
    is_array: bool = Field(default=False, description="Value is an array")
    array_dimensions: Optional[List[int]] = Field(
        default=None,
        description="Array dimensions if applicable"
    )

    @field_validator("value", mode="before")
    @classmethod
    def validate_value(cls, v: Any) -> Any:
        """Validate and convert value types."""
        if v is None:
            return None
        # Handle numpy types if present
        try:
            import numpy as np
            if isinstance(v, (np.integer, np.floating)):
                return float(v)
        except ImportError:
            pass
        return v

    @property
    def is_valid(self) -> bool:
        """Check if value is valid for use."""
        return self.quality.is_good and self.value is not None

    @property
    def age_seconds(self) -> float:
        """Get age of value in seconds."""
        return (datetime.utcnow() - self.source_timestamp).total_seconds()

    def is_stale(self, max_age_seconds: float = 60.0) -> bool:
        """Check if value is stale."""
        return self.age_seconds > max_age_seconds

    def is_in_range(self) -> bool:
        """Check if value is within engineering unit range."""
        if self.value is None or not isinstance(self.value, (int, float)):
            return True
        if self.eu_range_low is not None and self.value < self.eu_range_low:
            return False
        if self.eu_range_high is not None and self.value > self.eu_range_high:
            return False
        return True


# =============================================================================
# Tag Subscription
# =============================================================================

class SubscriptionMode(str, Enum):
    """OPC-UA subscription modes."""
    REPORTING = "reporting"  # Report on value change
    SAMPLING = "sampling"    # Report at fixed interval
    EXCEPTION = "exception"  # Report on deadband violation


class TagSubscription(BaseModel):
    """
    Configuration for an OPC-UA tag subscription.

    Defines how the tag should be monitored and when to
    publish value changes.
    """

    # Tag identification
    node_id: str = Field(..., description="OPC-UA node ID")
    tag_name: str = Field(..., description="Human-readable name")

    # Subscription settings
    subscription_id: UUID = Field(
        default_factory=uuid4,
        description="Unique subscription ID"
    )
    mode: SubscriptionMode = Field(
        default=SubscriptionMode.REPORTING,
        description="Subscription mode"
    )

    # Timing
    sampling_interval_ms: int = Field(
        default=1000,
        description="Sampling interval in milliseconds"
    )
    publishing_interval_ms: int = Field(
        default=1000,
        description="Publishing interval in milliseconds"
    )
    queue_size: int = Field(
        default=10,
        description="Queue size for buffering values"
    )

    # Filtering
    deadband_type: str = Field(
        default="absolute",
        description="Deadband type (absolute, percent, none)"
    )
    deadband_value: float = Field(
        default=0.0,
        description="Deadband value"
    )

    # Behavior
    discard_oldest: bool = Field(
        default=True,
        description="Discard oldest values when queue full"
    )
    enabled: bool = Field(default=True, description="Subscription enabled")

    # Metadata
    group: Optional[str] = Field(
        default=None,
        description="Subscription group name"
    )
    priority: int = Field(
        default=100,
        description="Priority (1-255, lower is higher priority)"
    )


# =============================================================================
# Subscription Group
# =============================================================================

class SubscriptionGroup(BaseModel):
    """
    Group of related tag subscriptions.

    Allows bulk management of subscriptions for a process area.
    """

    group_id: UUID = Field(default_factory=uuid4, description="Group ID")
    group_name: str = Field(..., description="Group name")
    description: str = Field(default="", description="Group description")

    # Subscriptions
    subscriptions: List[TagSubscription] = Field(
        default_factory=list,
        description="Subscriptions in this group"
    )

    # Group settings (override individual settings)
    sampling_interval_ms: Optional[int] = Field(
        default=None,
        description="Override sampling interval"
    )
    publishing_interval_ms: Optional[int] = Field(
        default=None,
        description="Override publishing interval"
    )
    enabled: bool = Field(default=True, description="Group enabled")

    def add_subscription(self, subscription: TagSubscription) -> None:
        """Add subscription to group."""
        subscription.group = self.group_name
        self.subscriptions.append(subscription)

    def get_node_ids(self) -> List[str]:
        """Get all node IDs in group."""
        return [sub.node_id for sub in self.subscriptions]


# =============================================================================
# Batch Read/Write
# =============================================================================

class BatchReadRequest(BaseModel):
    """Request for batch reading multiple tags."""

    request_id: UUID = Field(default_factory=uuid4, description="Request ID")
    node_ids: List[str] = Field(..., description="Node IDs to read")
    max_age_ms: int = Field(
        default=0,
        description="Max age of cached values (0 = always read)"
    )


class BatchReadResponse(BaseModel):
    """Response from batch read operation."""

    request_id: UUID = Field(..., description="Original request ID")
    values: List[TagValue] = Field(..., description="Read values")
    success_count: int = Field(..., description="Number of successful reads")
    failure_count: int = Field(..., description="Number of failed reads")
    duration_ms: int = Field(..., description="Total read duration")


class BatchWriteRequest(BaseModel):
    """Request for batch writing multiple tags."""

    request_id: UUID = Field(default_factory=uuid4, description="Request ID")
    writes: List[Dict[str, Any]] = Field(
        ...,
        description="List of {node_id, value} pairs"
    )


class BatchWriteResponse(BaseModel):
    """Response from batch write operation."""

    request_id: UUID = Field(..., description="Original request ID")
    results: List[Dict[str, Any]] = Field(
        ...,
        description="Write results per node"
    )
    success_count: int = Field(..., description="Number of successful writes")
    failure_count: int = Field(..., description="Number of failed writes")
    duration_ms: int = Field(..., description="Total write duration")
