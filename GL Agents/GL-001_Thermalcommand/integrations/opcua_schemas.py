# -*- coding: utf-8 -*-
"""
OPC-UA Schema Definitions for GL-001 ThermalCommand

This module defines all Pydantic models for OPC-UA operations including:
- Tag configuration and metadata
- Subscription management
- Write requests and responses
- Quality codes and timestamps
- Engineering units and safety boundaries

These schemas ensure type safety, validation, and complete audit trails
for all OPC-UA operations in industrial process heat systems.

Tag Naming Convention:
- Canonical: {system}.{equipment}.{measurement}
- Examples: steam.headerA.pressure, boiler.B1.fuel_flow

Author: GL-BackendDeveloper
Version: 1.0.0
Standard: OPC-UA Part 8 - Data Access
"""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import hashlib
import json
import uuid

from pydantic import BaseModel, Field, validator, root_validator


# =============================================================================
# ENUMS
# =============================================================================

class OPCUAQualityCode(str, Enum):
    """
    OPC-UA Quality Codes per OPC-UA Part 8.

    Quality codes indicate the reliability of the data value.
    """
    GOOD = "Good"
    GOOD_LOCAL_OVERRIDE = "Good_LocalOverride"
    GOOD_SUB_NORMAL = "Good_SubNormal"
    UNCERTAIN = "Uncertain"
    UNCERTAIN_LAST_USABLE = "Uncertain_LastUsableValue"
    UNCERTAIN_SENSOR_NOT_ACCURATE = "Uncertain_SensorNotAccurate"
    UNCERTAIN_ENGINEERING_UNITS_EXCEEDED = "Uncertain_EngineeringUnitsExceeded"
    UNCERTAIN_SUB_NORMAL = "Uncertain_SubNormal"
    BAD = "Bad"
    BAD_CONFIG_ERROR = "Bad_ConfigurationError"
    BAD_NOT_CONNECTED = "Bad_NotConnected"
    BAD_DEVICE_FAILURE = "Bad_DeviceFailure"
    BAD_SENSOR_FAILURE = "Bad_SensorFailure"
    BAD_LAST_KNOWN_VALUE = "Bad_LastKnownValue"
    BAD_COMMUNICATION_FAILURE = "Bad_CommunicationFailure"
    BAD_OUT_OF_SERVICE = "Bad_OutOfService"
    BAD_WAITING_FOR_INITIAL = "Bad_WaitingForInitialData"

    def is_good(self) -> bool:
        """Check if quality code indicates good data."""
        return self.value.startswith("Good")

    def is_uncertain(self) -> bool:
        """Check if quality code indicates uncertain data."""
        return self.value.startswith("Uncertain")

    def is_bad(self) -> bool:
        """Check if quality code indicates bad data."""
        return self.value.startswith("Bad")


class WriteConfirmationStatus(str, Enum):
    """Status of write confirmation in two-step process."""
    PENDING_RECOMMENDATION = "pending_recommendation"
    RECOMMENDATION_GENERATED = "recommendation_generated"
    PENDING_CONFIRMATION = "pending_confirmation"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    APPLIED = "applied"
    FAILED = "failed"
    EXPIRED = "expired"
    SAFETY_BLOCKED = "safety_blocked"


class TagAccessLevel(str, Enum):
    """Access level for OPC-UA tags."""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    WRITE_ONLY = "write_only"
    SUPERVISORY_WRITE = "supervisory_write"  # Requires two-step confirmation


class TagDataType(str, Enum):
    """Data types for OPC-UA tag values."""
    BOOLEAN = "Boolean"
    SBYTE = "SByte"
    BYTE = "Byte"
    INT16 = "Int16"
    UINT16 = "UInt16"
    INT32 = "Int32"
    UINT32 = "UInt32"
    INT64 = "Int64"
    UINT64 = "UInt64"
    FLOAT = "Float"
    DOUBLE = "Double"
    STRING = "String"
    DATETIME = "DateTime"
    BYTE_STRING = "ByteString"


class SecurityMode(str, Enum):
    """OPC-UA Security Mode."""
    NONE = "None"
    SIGN = "Sign"
    SIGN_AND_ENCRYPT = "SignAndEncrypt"


class SecurityPolicy(str, Enum):
    """OPC-UA Security Policy."""
    NONE = "None"
    BASIC128RSA15 = "Basic128Rsa15"
    BASIC256 = "Basic256"
    BASIC256SHA256 = "Basic256Sha256"
    AES128_SHA256_RSAOAEP = "Aes128_Sha256_RsaOaep"
    AES256_SHA256_RSAPSS = "Aes256_Sha256_RsaPss"


# =============================================================================
# ENGINEERING UNITS
# =============================================================================

class EngineeringUnit(BaseModel):
    """
    Engineering unit definition per OPC-UA Part 8.

    Attributes:
        namespace_uri: Unit namespace URI
        unit_id: Numeric unit identifier
        display_name: Human-readable unit name
        description: Unit description
    """
    namespace_uri: str = Field(
        default="http://www.opcfoundation.org/UA/units/un/cefact",
        description="Namespace URI for engineering units"
    )
    unit_id: int = Field(..., description="Numeric unit identifier per CEFACT")
    display_name: str = Field(..., description="Human-readable unit name")
    description: Optional[str] = Field(None, description="Unit description")

    class Config:
        frozen = True


# Common engineering units for process heat systems
ENGINEERING_UNITS = {
    "celsius": EngineeringUnit(unit_id=4408652, display_name="C", description="Degrees Celsius"),
    "kelvin": EngineeringUnit(unit_id=4932940, display_name="K", description="Kelvin"),
    "fahrenheit": EngineeringUnit(unit_id=4604232, display_name="F", description="Degrees Fahrenheit"),
    "bar": EngineeringUnit(unit_id=4342098, display_name="bar", description="Bar (pressure)"),
    "pascal": EngineeringUnit(unit_id=4932160, display_name="Pa", description="Pascal"),
    "psi": EngineeringUnit(unit_id=5264201, display_name="psi", description="Pounds per square inch"),
    "kg_per_s": EngineeringUnit(unit_id=4934739, display_name="kg/s", description="Kilograms per second"),
    "kg_per_h": EngineeringUnit(unit_id=4934736, display_name="kg/h", description="Kilograms per hour"),
    "m3_per_h": EngineeringUnit(unit_id=4607832, display_name="m3/h", description="Cubic meters per hour"),
    "kw": EngineeringUnit(unit_id=4937288, display_name="kW", description="Kilowatts"),
    "mw": EngineeringUnit(unit_id=4607575, display_name="MW", description="Megawatts"),
    "percent": EngineeringUnit(unit_id=4281139, display_name="%", description="Percent"),
    "rpm": EngineeringUnit(unit_id=5395765, display_name="rpm", description="Revolutions per minute"),
}


# =============================================================================
# SAFETY BOUNDARIES
# =============================================================================

class SafetyBoundary(BaseModel):
    """
    Safety boundary definition for tag values.

    Defines acceptable ranges for safe operation.
    Write operations outside these boundaries are blocked.
    """
    tag_id: str = Field(..., description="Tag identifier")
    min_value: Optional[float] = Field(None, description="Minimum safe value")
    max_value: Optional[float] = Field(None, description="Maximum safe value")
    rate_of_change_limit: Optional[float] = Field(
        None,
        description="Maximum allowed rate of change per second"
    )
    deadband: float = Field(
        default=0.0,
        ge=0.0,
        description="Deadband for change detection"
    )
    safety_interlock_tags: List[str] = Field(
        default_factory=list,
        description="Tags that must be in safe state before write"
    )
    requires_confirmation: bool = Field(
        default=True,
        description="Requires two-step confirmation"
    )
    sil_level: Optional[int] = Field(
        None,
        ge=1,
        le=4,
        description="Safety Integrity Level (1-4)"
    )

    @validator("max_value")
    def validate_max_greater_than_min(cls, v, values):
        """Ensure max_value is greater than min_value if both are set."""
        if v is not None and values.get("min_value") is not None:
            if v <= values["min_value"]:
                raise ValueError("max_value must be greater than min_value")
        return v

    def is_within_bounds(self, value: float) -> bool:
        """Check if value is within safety boundaries."""
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True

    def get_clamped_value(self, value: float) -> float:
        """Get value clamped to safety boundaries."""
        if self.min_value is not None:
            value = max(value, self.min_value)
        if self.max_value is not None:
            value = min(value, self.max_value)
        return value


# =============================================================================
# TAG METADATA
# =============================================================================

class TagMetadata(BaseModel):
    """
    Comprehensive metadata for an OPC-UA tag.

    Contains all information needed to interpret and validate tag data.
    """
    tag_id: str = Field(..., description="Unique tag identifier")
    node_id: str = Field(..., description="OPC-UA Node ID (e.g., ns=2;s=TagName)")
    canonical_name: str = Field(
        ...,
        description="Canonical name (e.g., steam.headerA.pressure)"
    )
    display_name: str = Field(..., description="Human-readable display name")
    description: Optional[str] = Field(None, description="Tag description")

    # Data characteristics
    data_type: TagDataType = Field(..., description="OPC-UA data type")
    engineering_unit: Optional[EngineeringUnit] = Field(
        None,
        description="Engineering unit"
    )
    eu_range_low: Optional[float] = Field(None, description="Engineering unit range low")
    eu_range_high: Optional[float] = Field(None, description="Engineering unit range high")

    # Access control
    access_level: TagAccessLevel = Field(
        default=TagAccessLevel.READ_ONLY,
        description="Tag access level"
    )
    is_supervisory: bool = Field(
        default=False,
        description="Is supervisory (setpoint) tag"
    )
    is_whitelisted_for_write: bool = Field(
        default=False,
        description="Whitelisted for write operations"
    )

    # Safety
    safety_boundary: Optional[SafetyBoundary] = Field(
        None,
        description="Safety boundary configuration"
    )

    # Scaling
    raw_low: Optional[float] = Field(None, description="Raw value low")
    raw_high: Optional[float] = Field(None, description="Raw value high")
    scaled_low: Optional[float] = Field(None, description="Scaled value low")
    scaled_high: Optional[float] = Field(None, description="Scaled value high")

    # Metadata
    equipment_id: Optional[str] = Field(None, description="Parent equipment ID")
    system_id: Optional[str] = Field(None, description="Parent system ID")
    area_id: Optional[str] = Field(None, description="Plant area ID")
    historian_tag: Optional[str] = Field(None, description="Historian tag name")

    # Versioning
    version: str = Field(default="1.0.0", description="Tag configuration version")
    last_modified: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last modification timestamp"
    )

    @validator("canonical_name")
    def validate_canonical_name(cls, v):
        """Validate canonical name format (system.equipment.measurement)."""
        parts = v.split(".")
        if len(parts) < 2:
            raise ValueError(
                "Canonical name must have at least 2 parts: system.measurement"
            )
        for part in parts:
            if not part or not part[0].isalpha():
                raise ValueError(
                    f"Each part of canonical name must start with a letter: {v}"
                )
        return v

    def apply_scaling(self, raw_value: float) -> float:
        """Apply scaling to convert raw value to engineering units."""
        if None in (self.raw_low, self.raw_high, self.scaled_low, self.scaled_high):
            return raw_value

        # Linear scaling
        raw_range = self.raw_high - self.raw_low
        if raw_range == 0:
            return self.scaled_low

        scaled_range = self.scaled_high - self.scaled_low
        return self.scaled_low + (
            (raw_value - self.raw_low) / raw_range * scaled_range
        )

    def reverse_scaling(self, scaled_value: float) -> float:
        """Reverse scaling to convert engineering units to raw value."""
        if None in (self.raw_low, self.raw_high, self.scaled_low, self.scaled_high):
            return scaled_value

        scaled_range = self.scaled_high - self.scaled_low
        if scaled_range == 0:
            return self.raw_low

        raw_range = self.raw_high - self.raw_low
        return self.raw_low + (
            (scaled_value - self.scaled_low) / scaled_range * raw_range
        )


# =============================================================================
# TAG CONFIGURATION
# =============================================================================

class OPCUATagConfig(BaseModel):
    """
    Complete configuration for an OPC-UA tag.

    Combines metadata with subscription and operational settings.
    """
    metadata: TagMetadata = Field(..., description="Tag metadata")

    # Subscription settings
    sampling_interval_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Sampling interval in milliseconds (100ms-60s)"
    )
    queue_size: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Monitored item queue size"
    )
    discard_oldest: bool = Field(
        default=True,
        description="Discard oldest when queue full"
    )

    # Data processing
    deadband_type: str = Field(
        default="absolute",
        description="Deadband type: 'absolute' or 'percent'"
    )
    deadband_value: float = Field(
        default=0.0,
        ge=0.0,
        description="Deadband value"
    )

    # Bad value handling
    substitute_bad_value: bool = Field(
        default=False,
        description="Substitute last good value for bad data"
    )
    max_bad_value_age_s: int = Field(
        default=60,
        ge=1,
        description="Max age of substituted value in seconds"
    )

    # Timestamp handling
    use_source_timestamp: bool = Field(
        default=True,
        description="Use source timestamp (vs server timestamp)"
    )
    max_timestamp_drift_s: float = Field(
        default=5.0,
        ge=0.0,
        description="Max allowed timestamp drift in seconds"
    )

    @property
    def tag_id(self) -> str:
        """Get tag ID from metadata."""
        return self.metadata.tag_id

    @property
    def node_id(self) -> str:
        """Get OPC-UA node ID from metadata."""
        return self.metadata.node_id


# =============================================================================
# DATA POINTS
# =============================================================================

class OPCUADataPoint(BaseModel):
    """
    Single data point from OPC-UA server.

    Represents a timestamped, quality-coded value with full provenance.
    """
    tag_id: str = Field(..., description="Tag identifier")
    node_id: str = Field(..., description="OPC-UA node ID")
    canonical_name: str = Field(..., description="Canonical tag name")

    # Value
    value: Any = Field(..., description="Tag value")
    data_type: TagDataType = Field(..., description="Value data type")

    # Timestamps
    source_timestamp: datetime = Field(
        ...,
        description="Source timestamp from device"
    )
    server_timestamp: datetime = Field(
        ...,
        description="Server timestamp"
    )
    received_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when received by client"
    )

    # Quality
    quality_code: OPCUAQualityCode = Field(
        default=OPCUAQualityCode.GOOD,
        description="OPC-UA quality code"
    )
    quality_substatus: Optional[int] = Field(
        None,
        description="Quality substatus bits"
    )

    # Engineering units
    engineering_unit: Optional[str] = Field(
        None,
        description="Engineering unit display name"
    )
    scaled_value: Optional[float] = Field(
        None,
        description="Value in engineering units"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail"
    )
    subscription_id: Optional[str] = Field(
        None,
        description="Source subscription ID"
    )
    sequence_number: Optional[int] = Field(
        None,
        description="Sequence number within subscription"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v),
        }

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 provenance hash for this data point."""
        data = {
            "tag_id": self.tag_id,
            "value": str(self.value),
            "source_timestamp": self.source_timestamp.isoformat(),
            "quality_code": self.quality_code.value,
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def is_good_quality(self) -> bool:
        """Check if data point has good quality."""
        return self.quality_code.is_good()

    def get_age_seconds(self) -> float:
        """Get age of data point in seconds."""
        now = datetime.now(timezone.utc)
        return (now - self.source_timestamp).total_seconds()


# =============================================================================
# SUBSCRIPTION CONFIGURATION
# =============================================================================

class OPCUASubscriptionConfig(BaseModel):
    """
    Configuration for an OPC-UA subscription.

    Defines publishing parameters and monitored items.
    """
    subscription_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique subscription identifier"
    )
    name: str = Field(..., description="Subscription name")
    description: Optional[str] = Field(None, description="Subscription description")

    # Publishing parameters
    publishing_interval_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Publishing interval in milliseconds"
    )
    lifetime_count: int = Field(
        default=10000,
        ge=3,
        description="Lifetime count (keep-alive multiplier)"
    )
    max_keep_alive_count: int = Field(
        default=10,
        ge=1,
        description="Max keep-alive count before timeout"
    )
    max_notifications_per_publish: int = Field(
        default=1000,
        ge=1,
        description="Max notifications per publish"
    )
    priority: int = Field(
        default=0,
        ge=0,
        le=255,
        description="Subscription priority (0-255)"
    )

    # Monitored items
    tag_configs: List[OPCUATagConfig] = Field(
        default_factory=list,
        description="Tag configurations in this subscription"
    )

    # State
    is_active: bool = Field(default=True, description="Subscription active state")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )

    @property
    def tag_count(self) -> int:
        """Get number of tags in subscription."""
        return len(self.tag_configs)


class OPCUASubscription(BaseModel):
    """
    Runtime state of an active OPC-UA subscription.

    Tracks subscription lifecycle and statistics.
    """
    config: OPCUASubscriptionConfig = Field(..., description="Subscription configuration")

    # Server-assigned IDs
    server_subscription_id: Optional[int] = Field(
        None,
        description="Server-assigned subscription ID"
    )
    revised_publishing_interval_ms: Optional[int] = Field(
        None,
        description="Server-revised publishing interval"
    )

    # State
    status: str = Field(default="created", description="Subscription status")
    is_connected: bool = Field(default=False, description="Connection state")
    last_publish_time: Optional[datetime] = Field(
        None,
        description="Last publish timestamp"
    )
    last_notification_time: Optional[datetime] = Field(
        None,
        description="Last notification timestamp"
    )

    # Statistics
    notification_count: int = Field(default=0, ge=0, description="Total notifications")
    publish_count: int = Field(default=0, ge=0, description="Total publishes")
    data_change_count: int = Field(default=0, ge=0, description="Data change count")
    error_count: int = Field(default=0, ge=0, description="Error count")

    # Monitored item handles
    monitored_item_handles: Dict[str, int] = Field(
        default_factory=dict,
        description="Mapping of tag_id to monitored item handle"
    )

    def record_notification(self) -> None:
        """Record a notification receipt."""
        self.notification_count += 1
        self.last_notification_time = datetime.now(timezone.utc)

    def record_error(self) -> None:
        """Record an error."""
        self.error_count += 1


# =============================================================================
# WRITE REQUESTS AND RESPONSES
# =============================================================================

class OPCUAWriteRequest(BaseModel):
    """
    Request to write a value to an OPC-UA tag.

    Implements two-step confirmation for supervisory writes.
    """
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier"
    )
    tag_id: str = Field(..., description="Target tag identifier")
    node_id: str = Field(..., description="OPC-UA node ID")
    canonical_name: str = Field(..., description="Canonical tag name")

    # Value
    value: Any = Field(..., description="Value to write")
    data_type: TagDataType = Field(..., description="Value data type")
    engineering_unit: Optional[str] = Field(None, description="Engineering unit")

    # Request metadata
    requested_by: str = Field(..., description="Requesting user/system ID")
    reason: str = Field(..., description="Reason for write")
    priority: int = Field(default=0, ge=0, le=10, description="Write priority")

    # Timestamps
    requested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Request timestamp"
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="Request expiration timestamp"
    )

    # Two-step confirmation
    confirmation_status: WriteConfirmationStatus = Field(
        default=WriteConfirmationStatus.PENDING_RECOMMENDATION,
        description="Confirmation status"
    )
    confirmation_token: Optional[str] = Field(
        None,
        description="Confirmation token for two-step process"
    )
    confirmed_by: Optional[str] = Field(None, description="Confirming user ID")
    confirmed_at: Optional[datetime] = Field(None, description="Confirmation timestamp")

    # Safety validation
    safety_check_passed: bool = Field(
        default=False,
        description="Safety validation passed"
    )
    safety_check_details: Optional[Dict[str, Any]] = Field(
        None,
        description="Safety check results"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash"
    )

    @validator("expires_at", pre=True, always=True)
    def set_default_expiration(cls, v, values):
        """Set default expiration to 5 minutes from request."""
        if v is None:
            return values.get("requested_at", datetime.now(timezone.utc)) + \
                   __import__("datetime").timedelta(minutes=5)
        return v

    def is_expired(self) -> bool:
        """Check if request has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 provenance hash for this request."""
        data = {
            "request_id": self.request_id,
            "tag_id": self.tag_id,
            "value": str(self.value),
            "requested_by": self.requested_by,
            "requested_at": self.requested_at.isoformat(),
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class OPCUAWriteResponse(BaseModel):
    """
    Response from an OPC-UA write operation.

    Contains complete audit trail for the write operation.
    """
    request_id: str = Field(..., description="Original request ID")
    response_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Response identifier"
    )

    # Result
    success: bool = Field(..., description="Write operation success")
    status_code: int = Field(..., description="OPC-UA status code")
    status_message: str = Field(..., description="Status message")

    # Value written
    tag_id: str = Field(..., description="Tag that was written")
    written_value: Any = Field(..., description="Value that was written")
    previous_value: Optional[Any] = Field(None, description="Previous tag value")

    # Timestamps
    requested_at: datetime = Field(..., description="Original request timestamp")
    written_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Write execution timestamp"
    )

    # Confirmation details
    confirmation_status: WriteConfirmationStatus = Field(
        ...,
        description="Final confirmation status"
    )
    confirmed_by: Optional[str] = Field(None, description="Confirming user ID")

    # Safety
    safety_checks: Dict[str, bool] = Field(
        default_factory=dict,
        description="Safety check results"
    )

    # Performance
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total processing time in milliseconds"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    audit_trail: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Complete audit trail"
    )

    def add_audit_entry(self, action: str, details: Dict[str, Any]) -> None:
        """Add entry to audit trail."""
        self.audit_trail.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "details": details,
        })


# =============================================================================
# CONNECTION CONFIGURATION
# =============================================================================

class OPCUASecurityConfig(BaseModel):
    """
    Security configuration for OPC-UA connection.

    Implements certificate-based mTLS authentication.
    """
    security_mode: SecurityMode = Field(
        default=SecurityMode.SIGN_AND_ENCRYPT,
        description="OPC-UA security mode"
    )
    security_policy: SecurityPolicy = Field(
        default=SecurityPolicy.BASIC256SHA256,
        description="OPC-UA security policy"
    )

    # Certificate paths
    client_certificate_path: Optional[str] = Field(
        None,
        description="Path to client certificate (PEM)"
    )
    client_private_key_path: Optional[str] = Field(
        None,
        description="Path to client private key (PEM)"
    )
    server_certificate_path: Optional[str] = Field(
        None,
        description="Path to server certificate (PEM)"
    )
    trusted_certificates_path: Optional[str] = Field(
        None,
        description="Path to trusted certificates directory"
    )

    # Certificate details
    application_uri: str = Field(
        default="urn:greenlang:gl001:thermalcommand",
        description="Application URI for certificate"
    )
    application_name: str = Field(
        default="GL-001 ThermalCommand",
        description="Application name"
    )

    # Authentication
    username: Optional[str] = Field(None, description="Username for authentication")
    # Note: Password should be retrieved from secure vault, not stored
    use_certificate_auth: bool = Field(
        default=True,
        description="Use certificate authentication"
    )

    @root_validator
    def validate_security_config(cls, values):
        """Validate security configuration consistency."""
        mode = values.get("security_mode")
        policy = values.get("security_policy")

        if mode != SecurityMode.NONE and policy == SecurityPolicy.NONE:
            raise ValueError(
                "Security policy cannot be None when security mode is not None"
            )

        if mode in (SecurityMode.SIGN, SecurityMode.SIGN_AND_ENCRYPT):
            if not values.get("client_certificate_path"):
                raise ValueError(
                    "Client certificate required for Sign/SignAndEncrypt mode"
                )

        return values


class OPCUAConnectionConfig(BaseModel):
    """
    Complete configuration for OPC-UA server connection.

    Includes endpoint, security, and operational parameters.
    """
    connection_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique connection identifier"
    )
    name: str = Field(..., description="Connection name")
    description: Optional[str] = Field(None, description="Connection description")

    # Endpoint
    endpoint_url: str = Field(
        ...,
        description="OPC-UA endpoint URL (e.g., opc.tcp://server:4840)"
    )
    server_name: Optional[str] = Field(None, description="Server name")

    # Security
    security: OPCUASecurityConfig = Field(
        default_factory=OPCUASecurityConfig,
        description="Security configuration"
    )

    # Network
    network_segment: str = Field(
        default="OT_DMZ",
        description="Network segment (OT_DMZ, OT_LEVEL2, etc.)"
    )

    # Connection parameters
    timeout_ms: int = Field(
        default=30000,
        ge=1000,
        le=300000,
        description="Connection timeout in milliseconds"
    )
    session_timeout_ms: int = Field(
        default=60000,
        ge=10000,
        description="Session timeout in milliseconds"
    )

    # Reconnection
    auto_reconnect: bool = Field(default=True, description="Auto-reconnect on failure")
    reconnect_interval_ms: int = Field(
        default=5000,
        ge=1000,
        description="Reconnect interval in milliseconds"
    )
    max_reconnect_attempts: int = Field(
        default=10,
        ge=0,
        description="Max reconnect attempts (0 = unlimited)"
    )

    # Health monitoring
    health_check_interval_ms: int = Field(
        default=30000,
        ge=5000,
        description="Health check interval in milliseconds"
    )

    # Version
    version: str = Field(default="1.0.0", description="Configuration version")

    @validator("endpoint_url")
    def validate_endpoint_url(cls, v):
        """Validate OPC-UA endpoint URL format."""
        if not v.startswith(("opc.tcp://", "opc.https://")):
            raise ValueError(
                "Endpoint URL must start with 'opc.tcp://' or 'opc.https://'"
            )
        return v


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "OPCUAQualityCode",
    "WriteConfirmationStatus",
    "TagAccessLevel",
    "TagDataType",
    "SecurityMode",
    "SecurityPolicy",
    # Engineering Units
    "EngineeringUnit",
    "ENGINEERING_UNITS",
    # Safety
    "SafetyBoundary",
    # Tag Configuration
    "TagMetadata",
    "OPCUATagConfig",
    # Data Points
    "OPCUADataPoint",
    # Subscriptions
    "OPCUASubscriptionConfig",
    "OPCUASubscription",
    # Write Operations
    "OPCUAWriteRequest",
    "OPCUAWriteResponse",
    # Connection
    "OPCUASecurityConfig",
    "OPCUAConnectionConfig",
]
