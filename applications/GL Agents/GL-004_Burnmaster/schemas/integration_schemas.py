"""
Integration Schemas - Data models for external system integration.

This module defines Pydantic models for tag values, tag quality, write requests,
and connection status. These schemas support integration with OPC, DCS, SCADA,
and other industrial control systems.

Example:
    >>> from integration_schemas import TagValue, TagQuality, WriteRequest
    >>> tag = TagValue(
    ...     tag="FC-101.PV",
    ...     value=2.5,
    ...     quality=TagQuality.GOOD,
    ...     timestamp=datetime.utcnow()
    ... )
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator, computed_field
import hashlib


class TagQuality(str, Enum):
    """
    Data quality indicator for tag values.

    GOOD: Data is valid and reliable
    BAD: Data is invalid or unreliable
    UNCERTAIN: Data quality is uncertain
    STALE: Data is older than expected
    CONFIG_ERROR: Configuration error
    COMM_FAILURE: Communication failure
    SENSOR_FAILURE: Sensor failure detected
    OUT_OF_RANGE: Value is out of expected range
    """
    GOOD = "good"
    BAD = "bad"
    UNCERTAIN = "uncertain"
    STALE = "stale"
    CONFIG_ERROR = "config_error"
    COMM_FAILURE = "comm_failure"
    SENSOR_FAILURE = "sensor_failure"
    OUT_OF_RANGE = "out_of_range"


class ConnectionState(str, Enum):
    """Connection state for external system."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    UNKNOWN = "unknown"


class WriteRequestStatus(str, Enum):
    """Status of a write request."""
    PENDING = "pending"
    QUEUED = "queued"
    SENT = "sent"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class DataType(str, Enum):
    """Data type for tag values."""
    FLOAT = "float"
    DOUBLE = "double"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT16 = "uint16"
    UINT32 = "uint32"
    UINT64 = "uint64"
    BOOLEAN = "boolean"
    STRING = "string"
    DATETIME = "datetime"


class ProtocolType(str, Enum):
    """Communication protocol type."""
    OPC_DA = "opc_da"
    OPC_UA = "opc_ua"
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    PROFINET = "profinet"
    ETHERNET_IP = "ethernet_ip"
    REST_API = "rest_api"
    MQTT = "mqtt"
    DATABASE = "database"


class TagValue(BaseModel):
    """
    Single tag/point value from control system.

    Represents a real-time or historical value from a tag/point
    with quality indicator and timestamp.

    Attributes:
        tag: Tag/point name or identifier
        value: Numeric or string value
        quality: Data quality indicator
        timestamp: When value was sampled
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    tag: str = Field(..., min_length=1, max_length=200, description="Tag/point name or identifier")
    value: Union[float, int, bool, str] = Field(..., description="Tag value")
    quality: TagQuality = Field(default=TagQuality.GOOD, description="Data quality indicator")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Sample timestamp")

    # Type information
    data_type: DataType = Field(default=DataType.FLOAT, description="Data type of value")
    unit: str = Field(default="", max_length=50, description="Engineering unit")

    # Scaling
    raw_value: Optional[Union[float, int]] = Field(default=None, description="Raw unscaled value")
    scale_factor: float = Field(default=1.0, description="Scale factor applied")
    offset: float = Field(default=0.0, description="Offset applied")

    # Limits
    low_limit: Optional[float] = Field(default=None, description="Low engineering limit")
    high_limit: Optional[float] = Field(default=None, description="High engineering limit")

    # Source
    source_system: Optional[str] = Field(default=None, max_length=100, description="Source system identifier")
    source_address: Optional[str] = Field(default=None, max_length=200, description="Source address (e.g., OPC item path)")

    @computed_field
    @property
    def is_good_quality(self) -> bool:
        """Check if quality is good."""
        return self.quality == TagQuality.GOOD

    @computed_field
    @property
    def is_valid(self) -> bool:
        """Check if value is valid (good or uncertain quality)."""
        return self.quality in [TagQuality.GOOD, TagQuality.UNCERTAIN]

    @computed_field
    @property
    def age_seconds(self) -> float:
        """Calculate age of value in seconds."""
        return (datetime.utcnow() - self.timestamp).total_seconds()

    @computed_field
    @property
    def is_stale(self) -> bool:
        """Check if value is stale (>60 seconds old)."""
        return self.age_seconds > 60.0

    def is_within_limits(self) -> bool:
        """Check if value is within limits."""
        if isinstance(self.value, (int, float)):
            if self.low_limit is not None and self.value < self.low_limit:
                return False
            if self.high_limit is not None and self.value > self.high_limit:
                return False
        return True


class TagValueBatch(BaseModel):
    """Batch of tag values."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    batch_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"), description="Batch identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Batch timestamp")
    source_system: str = Field(default="", max_length=100, description="Source system")

    # Values
    values: List[TagValue] = Field(default_factory=list, description="List of tag values")

    # Statistics
    total_count: int = Field(default=0, ge=0, description="Total tag count")
    good_count: int = Field(default=0, ge=0, description="Good quality count")
    bad_count: int = Field(default=0, ge=0, description="Bad quality count")

    # Timing
    acquisition_time_ms: Optional[float] = Field(default=None, ge=0.0, description="Time to acquire values")

    @computed_field
    @property
    def quality_percent(self) -> float:
        """Calculate percentage of good quality values."""
        if self.total_count == 0:
            return 100.0
        return (self.good_count / self.total_count) * 100.0

    def get_value(self, tag: str) -> Optional[TagValue]:
        """Get value for a specific tag."""
        for value in self.values:
            if value.tag == tag:
                return value
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to simple dictionary of tag -> value."""
        return {v.tag: v.value for v in self.values if v.is_good_quality}


class WriteRequest(BaseModel):
    """
    Request to write a value to external system.

    Represents a write request with full tracking of status,
    approval, and execution.

    Attributes:
        tag: Tag/point to write
        value: Value to write
        reason: Reason for the write
        requestor: Who/what requested the write
        approved: Whether write is approved
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    request_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"), description="Request identifier")
    tag: str = Field(..., min_length=1, max_length=200, description="Tag/point to write")
    value: Union[float, int, bool, str] = Field(..., description="Value to write")
    data_type: DataType = Field(default=DataType.FLOAT, description="Data type of value")

    # Status
    status: WriteRequestStatus = Field(default=WriteRequestStatus.PENDING, description="Request status")
    reason: str = Field(..., min_length=1, max_length=500, description="Reason for the write")

    # Requestor information
    requestor: str = Field(..., min_length=1, max_length=100, description="Who/what requested the write")
    requestor_type: str = Field(default="system", max_length=50, description="Requestor type (system, operator, etc.)")
    source_system: str = Field(default="BURNMASTER", max_length=100, description="Source system")

    # Approval
    requires_approval: bool = Field(default=True, description="Whether approval is required")
    approved: bool = Field(default=False, description="Whether write is approved")
    approved_by: Optional[str] = Field(default=None, max_length=100, description="Who approved")
    approved_at: Optional[datetime] = Field(default=None, description="Approval timestamp")
    rejection_reason: Optional[str] = Field(default=None, max_length=500, description="Rejection reason")

    # Previous value
    previous_value: Optional[Union[float, int, bool, str]] = Field(default=None, description="Value before write")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When request was created")
    sent_at: Optional[datetime] = Field(default=None, description="When request was sent")
    confirmed_at: Optional[datetime] = Field(default=None, description="When write was confirmed")
    expires_at: Optional[datetime] = Field(default=None, description="When request expires")

    # Error handling
    error_message: Optional[str] = Field(default=None, max_length=500, description="Error message if failed")
    retry_count: int = Field(default=0, ge=0, description="Number of retries")
    max_retries: int = Field(default=3, ge=0, description="Maximum retries allowed")

    # Target system
    target_system: Optional[str] = Field(default=None, max_length=100, description="Target system identifier")
    target_address: Optional[str] = Field(default=None, max_length=200, description="Target address")

    # Audit
    audit_id: Optional[str] = Field(default=None, max_length=100, description="Related audit record ID")

    @computed_field
    @property
    def is_pending(self) -> bool:
        """Check if request is still pending."""
        return self.status in [WriteRequestStatus.PENDING, WriteRequestStatus.QUEUED]

    @computed_field
    @property
    def is_complete(self) -> bool:
        """Check if request is complete (success or failure)."""
        return self.status in [
            WriteRequestStatus.CONFIRMED,
            WriteRequestStatus.FAILED,
            WriteRequestStatus.REJECTED,
            WriteRequestStatus.CANCELLED
        ]

    @computed_field
    @property
    def is_expired(self) -> bool:
        """Check if request has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    @computed_field
    @property
    def can_retry(self) -> bool:
        """Check if request can be retried."""
        return self.retry_count < self.max_retries and self.status == WriteRequestStatus.FAILED

    def approve(self, approver: str) -> None:
        """Approve the write request."""
        self.approved = True
        self.approved_by = approver
        self.approved_at = datetime.utcnow()
        self.status = WriteRequestStatus.QUEUED

    def reject(self, reason: str, rejector: str) -> None:
        """Reject the write request."""
        self.approved = False
        self.approved_by = rejector
        self.approved_at = datetime.utcnow()
        self.rejection_reason = reason
        self.status = WriteRequestStatus.REJECTED

    def mark_sent(self) -> None:
        """Mark request as sent."""
        self.sent_at = datetime.utcnow()
        self.status = WriteRequestStatus.SENT

    def confirm(self) -> None:
        """Confirm write was successful."""
        self.confirmed_at = datetime.utcnow()
        self.status = WriteRequestStatus.CONFIRMED

    def fail(self, error: str) -> None:
        """Mark request as failed."""
        self.error_message = error
        self.status = WriteRequestStatus.FAILED
        self.retry_count += 1


class ConnectionStatus(BaseModel):
    """
    Connection status for external system integration.

    Tracks the connection state, health, and statistics for
    an external system connection.

    Attributes:
        connection_id: Unique connection identifier
        connected: Whether currently connected
        last_heartbeat: Last successful communication
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    connection_id: str = Field(..., min_length=1, max_length=100, description="Unique connection identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Connection name")
    description: str = Field(default="", max_length=500, description="Connection description")

    # Connection state
    state: ConnectionState = Field(default=ConnectionState.UNKNOWN, description="Current connection state")
    connected: bool = Field(default=False, description="Whether currently connected")

    # Protocol and endpoint
    protocol: ProtocolType = Field(..., description="Communication protocol")
    endpoint: str = Field(..., min_length=1, max_length=500, description="Connection endpoint (URL, IP, etc.)")
    port: Optional[int] = Field(default=None, ge=1, le=65535, description="Port number")

    # Timestamps
    last_heartbeat: Optional[datetime] = Field(default=None, description="Last successful heartbeat")
    last_connect: Optional[datetime] = Field(default=None, description="Last successful connection")
    last_disconnect: Optional[datetime] = Field(default=None, description="Last disconnection")
    last_error: Optional[datetime] = Field(default=None, description="Last error timestamp")
    last_error_message: Optional[str] = Field(default=None, max_length=500, description="Last error message")

    # Statistics
    uptime_percent: float = Field(default=100.0, ge=0.0, le=100.0, description="Connection uptime percentage")
    messages_sent: int = Field(default=0, ge=0, description="Total messages sent")
    messages_received: int = Field(default=0, ge=0, description="Total messages received")
    errors_count: int = Field(default=0, ge=0, description="Total error count")
    reconnect_count: int = Field(default=0, ge=0, description="Number of reconnections")

    # Performance
    latency_ms: Optional[float] = Field(default=None, ge=0.0, description="Current latency in ms")
    avg_latency_ms: Optional[float] = Field(default=None, ge=0.0, description="Average latency in ms")
    max_latency_ms: Optional[float] = Field(default=None, ge=0.0, description="Maximum latency in ms")
    throughput_per_second: Optional[float] = Field(default=None, ge=0.0, description="Messages per second")

    # Configuration
    heartbeat_interval_s: float = Field(default=30.0, ge=1.0, le=300.0, description="Heartbeat interval")
    timeout_s: float = Field(default=30.0, ge=1.0, le=300.0, description="Connection timeout")
    retry_interval_s: float = Field(default=5.0, ge=1.0, le=60.0, description="Retry interval")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum connection retries")

    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Status timestamp")

    @computed_field
    @property
    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        if not self.connected:
            return False
        if self.last_heartbeat is None:
            return False
        heartbeat_age = (datetime.utcnow() - self.last_heartbeat).total_seconds()
        return heartbeat_age < self.heartbeat_interval_s * 2

    @computed_field
    @property
    def heartbeat_age_seconds(self) -> Optional[float]:
        """Calculate age of last heartbeat in seconds."""
        if self.last_heartbeat is None:
            return None
        return (datetime.utcnow() - self.last_heartbeat).total_seconds()

    @computed_field
    @property
    def needs_reconnect(self) -> bool:
        """Check if connection needs reconnection."""
        if self.connected and self.is_healthy:
            return False
        return self.state in [ConnectionState.DISCONNECTED, ConnectionState.ERROR]


class TagConfiguration(BaseModel):
    """Configuration for a tag/point."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    tag: str = Field(..., min_length=1, max_length=200, description="Tag name")
    description: str = Field(default="", max_length=500, description="Tag description")
    data_type: DataType = Field(default=DataType.FLOAT, description="Data type")
    unit: str = Field(default="", max_length=50, description="Engineering unit")

    # Source configuration
    source_address: str = Field(..., min_length=1, max_length=500, description="Source address")
    connection_id: str = Field(..., min_length=1, max_length=100, description="Connection to use")

    # Scaling
    scale_factor: float = Field(default=1.0, description="Scale factor")
    offset: float = Field(default=0.0, description="Offset")
    raw_low: Optional[float] = Field(default=None, description="Raw low value")
    raw_high: Optional[float] = Field(default=None, description="Raw high value")
    eng_low: Optional[float] = Field(default=None, description="Engineering low value")
    eng_high: Optional[float] = Field(default=None, description="Engineering high value")

    # Limits
    low_limit: Optional[float] = Field(default=None, description="Low engineering limit")
    high_limit: Optional[float] = Field(default=None, description="High engineering limit")
    low_alarm: Optional[float] = Field(default=None, description="Low alarm threshold")
    high_alarm: Optional[float] = Field(default=None, description="High alarm threshold")

    # Access
    read_enabled: bool = Field(default=True, description="Read enabled")
    write_enabled: bool = Field(default=False, description="Write enabled")
    write_requires_approval: bool = Field(default=True, description="Write requires approval")

    # Scan
    scan_rate_ms: int = Field(default=1000, ge=100, le=60000, description="Scan rate in ms")
    deadband: float = Field(default=0.0, ge=0.0, description="Deadband for change detection")

    # Metadata
    active: bool = Field(default=True, description="Whether tag is active")
    group: Optional[str] = Field(default=None, max_length=100, description="Tag group")
    equipment_id: Optional[str] = Field(default=None, max_length=100, description="Related equipment")


class IntegrationConfig(BaseModel):
    """Configuration for integration module."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    config_id: str = Field(default="default", description="Configuration identifier")
    name: str = Field(default="Default Integration Config", max_length=200, description="Configuration name")

    # Connections
    connections: List[ConnectionStatus] = Field(default_factory=list, description="Connection configurations")

    # Tags
    tags: List[TagConfiguration] = Field(default_factory=list, description="Tag configurations")

    # Write settings
    write_queue_size: int = Field(default=100, ge=10, le=10000, description="Maximum write queue size")
    write_timeout_s: float = Field(default=30.0, ge=1.0, le=300.0, description="Write timeout")
    write_retry_count: int = Field(default=3, ge=0, le=10, description="Write retry count")
    require_write_approval: bool = Field(default=True, description="Require approval for writes")

    # Read settings
    default_scan_rate_ms: int = Field(default=1000, ge=100, le=60000, description="Default scan rate")
    batch_size: int = Field(default=100, ge=1, le=1000, description="Read batch size")
    stale_threshold_s: float = Field(default=60.0, ge=1.0, le=3600.0, description="Stale data threshold")

    # Health monitoring
    health_check_interval_s: float = Field(default=30.0, ge=10.0, le=300.0, description="Health check interval")
    heartbeat_interval_s: float = Field(default=30.0, ge=10.0, le=300.0, description="Heartbeat interval")

    # Logging
    log_all_reads: bool = Field(default=False, description="Log all read operations")
    log_all_writes: bool = Field(default=True, description="Log all write operations")

    # Version
    version: str = Field(default="1.0", description="Configuration version")
    effective_date: Optional[datetime] = Field(default=None, description="Effective date")


class DataExchange(BaseModel):
    """Record of data exchange with external system."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    exchange_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"), description="Exchange identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Exchange timestamp")

    # Direction and type
    direction: str = Field(..., description="Direction: 'read' or 'write'")
    exchange_type: str = Field(default="single", description="Type: 'single', 'batch', 'subscription'")

    # Connection
    connection_id: str = Field(..., description="Connection used")
    protocol: ProtocolType = Field(..., description="Protocol used")

    # Data
    tags: List[str] = Field(default_factory=list, description="Tags involved")
    tag_count: int = Field(default=0, ge=0, description="Number of tags")
    values: Dict[str, Any] = Field(default_factory=dict, description="Values exchanged")

    # Result
    success: bool = Field(default=True, description="Whether exchange succeeded")
    error_message: Optional[str] = Field(default=None, max_length=500, description="Error message")
    partial_success: bool = Field(default=False, description="Whether partially successful")
    failed_tags: List[str] = Field(default_factory=list, description="Tags that failed")

    # Performance
    duration_ms: Optional[float] = Field(default=None, ge=0.0, description="Exchange duration")
    latency_ms: Optional[float] = Field(default=None, ge=0.0, description="Network latency")

    # Quality summary
    good_quality_count: int = Field(default=0, ge=0, description="Good quality values")
    bad_quality_count: int = Field(default=0, ge=0, description="Bad quality values")


__all__ = [
    "TagQuality",
    "ConnectionState",
    "WriteRequestStatus",
    "DataType",
    "ProtocolType",
    "TagValue",
    "TagValueBatch",
    "WriteRequest",
    "ConnectionStatus",
    "TagConfiguration",
    "IntegrationConfig",
    "DataExchange",
]
