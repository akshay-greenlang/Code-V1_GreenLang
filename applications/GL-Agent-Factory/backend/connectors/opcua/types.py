"""
OPC-UA Data Types and Pydantic Models for GreenLang Process Heat Agents.

This module defines type-safe Pydantic models for OPC-UA operations including:
- Node values with quality codes and timestamps
- Node information for browsing
- Subscription configurations
- Historical data access (HDA) parameters
- Security configurations

All models support zero-hallucination principles with provenance tracking.

Usage:
    from connectors.opcua.types import NodeValue, NodeInfo, OPCUAQuality

    # Read a node value with full provenance
    value = NodeValue(
        node_id="ns=2;s=Temperature.PV",
        value=425.7,
        data_type="Double",
        quality=OPCUAQuality.GOOD,
        source_timestamp=datetime.utcnow(),
    )
"""

from datetime import datetime
from enum import IntEnum, Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
import uuid


# =============================================================================
# OPC-UA Quality Codes
# =============================================================================


class OPCUAQuality(IntEnum):
    """
    OPC-UA Status Code Quality Bits.

    Based on OPC-UA Part 4 - Services, Section 7.34.
    Quality codes indicate the reliability of the data value.
    """

    # Good quality (0x00000000 - 0x3FFFFFFF)
    GOOD = 0x00000000
    GOOD_LOCAL_OVERRIDE = 0x00000096
    GOOD_SUB_NORMAL = 0x00000054
    GOOD_CLAMPED = 0x00000300

    # Uncertain quality (0x40000000 - 0x7FFFFFFF)
    UNCERTAIN = 0x40000000
    UNCERTAIN_INITIAL_VALUE = 0x40000014
    UNCERTAIN_SENSOR_NOT_ACCURATE = 0x40000042
    UNCERTAIN_ENGINEERING_UNITS_EXCEEDED = 0x40000044
    UNCERTAIN_SUB_NORMAL = 0x40000054

    # Bad quality (0x80000000 - 0xFFFFFFFF)
    BAD = 0x80000000
    BAD_CONFIG_ERROR = 0x80000010
    BAD_NOT_CONNECTED = 0x80000012
    BAD_DEVICE_FAILURE = 0x80000014
    BAD_SENSOR_FAILURE = 0x80000016
    BAD_LAST_KNOWN_VALUE = 0x80000018
    BAD_COMM_FAILURE = 0x8001001A
    BAD_OUT_OF_SERVICE = 0x8001001C
    BAD_WAITING_FOR_INITIAL_DATA = 0x80000320


class QualityLevel(str, Enum):
    """Simplified quality level for easy filtering."""

    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"

    @classmethod
    def from_quality_code(cls, quality: OPCUAQuality) -> "QualityLevel":
        """Determine quality level from OPC-UA quality code."""
        code = int(quality)
        if code < 0x40000000:
            return cls.GOOD
        elif code < 0x80000000:
            return cls.UNCERTAIN
        else:
            return cls.BAD


# =============================================================================
# OPC-UA Data Types
# =============================================================================


class OPCUADataType(str, Enum):
    """OPC-UA built-in data types (Part 6 - Mappings, Section 5.1.2)."""

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
    GUID = "Guid"
    BYTESTRING = "ByteString"
    XMLELEMENT = "XmlElement"
    NODEID = "NodeId"
    EXPANDEDNODEID = "ExpandedNodeId"
    STATUSCODE = "StatusCode"
    QUALIFIEDNAME = "QualifiedName"
    LOCALIZEDTEXT = "LocalizedText"
    EXTENSIONOBJECT = "ExtensionObject"
    DATAVALUE = "DataValue"
    VARIANT = "Variant"
    DIAGNOSTICINFO = "DiagnosticInfo"


class NodeClass(str, Enum):
    """OPC-UA Node Classes (Part 3 - Address Space Model)."""

    UNSPECIFIED = "Unspecified"
    OBJECT = "Object"
    VARIABLE = "Variable"
    METHOD = "Method"
    OBJECT_TYPE = "ObjectType"
    VARIABLE_TYPE = "VariableType"
    REFERENCE_TYPE = "ReferenceType"
    DATA_TYPE = "DataType"
    VIEW = "View"


class AccessLevel(IntEnum):
    """OPC-UA Access Level Flags (Part 3 - Address Space Model)."""

    NONE = 0
    CURRENT_READ = 1
    CURRENT_WRITE = 2
    HISTORY_READ = 4
    HISTORY_WRITE = 8
    SEMANTIC_CHANGE = 16
    STATUS_WRITE = 32
    TIMESTAMP_WRITE = 64


# =============================================================================
# Node Value Models
# =============================================================================


class DataProvenance(BaseModel):
    """
    Data provenance tracking for zero-hallucination compliance.

    Records the complete lineage of data from source to consumption.
    """

    source_endpoint: str = Field(
        ...,
        description="OPC-UA endpoint URL where data was retrieved"
    )
    source_node_id: str = Field(
        ...,
        description="Original OPC-UA node ID"
    )
    retrieval_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this data was retrieved from OPC-UA server"
    )
    retrieval_method: str = Field(
        default="read",
        description="How data was obtained: read, subscription, historical"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="OPC-UA session ID used for retrieval"
    )
    trace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique trace ID for this data retrieval"
    )


class NodeValue(BaseModel):
    """
    OPC-UA Node Value with full quality and timestamp information.

    Represents a value read from or to be written to an OPC-UA node.
    Includes data quality, timestamps, and provenance for zero-hallucination.
    """

    node_id: str = Field(
        ...,
        description="OPC-UA Node ID (e.g., 'ns=2;s=Temperature.PV')"
    )
    value: Any = Field(
        ...,
        description="The actual value"
    )
    data_type: str = Field(
        default="Variant",
        description="OPC-UA data type name"
    )
    quality: OPCUAQuality = Field(
        default=OPCUAQuality.GOOD,
        description="OPC-UA quality/status code"
    )
    quality_level: QualityLevel = Field(
        default=QualityLevel.GOOD,
        description="Simplified quality level"
    )
    source_timestamp: Optional[datetime] = Field(
        default=None,
        description="Timestamp from the data source (sensor/device)"
    )
    server_timestamp: Optional[datetime] = Field(
        default=None,
        description="Timestamp when server received the value"
    )
    provenance: Optional[DataProvenance] = Field(
        default=None,
        description="Data provenance for traceability"
    )

    @field_validator("quality_level", mode="before")
    @classmethod
    def set_quality_level(cls, v, info):
        """Automatically set quality level from quality code."""
        if v is None and "quality" in info.data:
            return QualityLevel.from_quality_code(info.data["quality"])
        return v

    def is_good(self) -> bool:
        """Check if value has good quality."""
        return self.quality_level == QualityLevel.GOOD

    def is_usable(self) -> bool:
        """Check if value is usable (good or uncertain)."""
        return self.quality_level in (QualityLevel.GOOD, QualityLevel.UNCERTAIN)

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            OPCUAQuality: lambda v: v.value,
        }


class NodeValueBatch(BaseModel):
    """Batch of node values for efficient multi-node operations."""

    values: List[NodeValue] = Field(
        default_factory=list,
        description="List of node values"
    )
    batch_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this batch was created"
    )
    endpoint: str = Field(
        ...,
        description="Source endpoint for this batch"
    )

    @property
    def good_values(self) -> List[NodeValue]:
        """Get only good quality values."""
        return [v for v in self.values if v.is_good()]

    @property
    def usable_values(self) -> List[NodeValue]:
        """Get usable (good or uncertain) values."""
        return [v for v in self.values if v.is_usable()]

    def get_value(self, node_id: str) -> Optional[NodeValue]:
        """Get value for a specific node ID."""
        for v in self.values:
            if v.node_id == node_id:
                return v
        return None


# =============================================================================
# Node Information Models
# =============================================================================


class NodeInfo(BaseModel):
    """
    OPC-UA Node Information from browse operations.

    Contains metadata about a node in the OPC-UA address space.
    """

    node_id: str = Field(
        ...,
        description="OPC-UA Node ID"
    )
    browse_name: str = Field(
        ...,
        description="Browse name of the node"
    )
    display_name: str = Field(
        ...,
        description="Human-readable display name"
    )
    node_class: NodeClass = Field(
        default=NodeClass.UNSPECIFIED,
        description="Type of node"
    )
    description: Optional[str] = Field(
        default=None,
        description="Node description"
    )
    data_type: Optional[str] = Field(
        default=None,
        description="Data type for Variable nodes"
    )
    access_level: int = Field(
        default=AccessLevel.CURRENT_READ,
        description="Access level flags"
    )
    is_abstract: bool = Field(
        default=False,
        description="Whether this is an abstract type"
    )
    parent_node_id: Optional[str] = Field(
        default=None,
        description="Parent node ID"
    )
    children_count: int = Field(
        default=0,
        description="Number of child nodes"
    )

    def is_readable(self) -> bool:
        """Check if node is readable."""
        return bool(self.access_level & AccessLevel.CURRENT_READ)

    def is_writable(self) -> bool:
        """Check if node is writable."""
        return bool(self.access_level & AccessLevel.CURRENT_WRITE)

    def is_variable(self) -> bool:
        """Check if this is a variable node."""
        return self.node_class == NodeClass.VARIABLE


class BrowseResult(BaseModel):
    """Result of an OPC-UA browse operation."""

    parent_node_id: str = Field(
        ...,
        description="Node ID that was browsed"
    )
    nodes: List[NodeInfo] = Field(
        default_factory=list,
        description="Child nodes found"
    )
    continuation_point: Optional[str] = Field(
        default=None,
        description="Continuation point for paginated results"
    )
    has_more: bool = Field(
        default=False,
        description="Whether more results are available"
    )


# =============================================================================
# Subscription Models
# =============================================================================


class MonitoringMode(str, Enum):
    """OPC-UA Monitoring Modes."""

    DISABLED = "Disabled"
    SAMPLING = "Sampling"
    REPORTING = "Reporting"


class DataChangeTrigger(str, Enum):
    """OPC-UA Data Change Trigger conditions."""

    STATUS = "Status"
    STATUS_VALUE = "StatusValue"
    STATUS_VALUE_TIMESTAMP = "StatusValueTimestamp"


class MonitoredItemConfig(BaseModel):
    """Configuration for a monitored item in a subscription."""

    node_id: str = Field(
        ...,
        description="Node ID to monitor"
    )
    sampling_interval_ms: int = Field(
        default=1000,
        description="Sampling interval in milliseconds"
    )
    queue_size: int = Field(
        default=10,
        description="Queue size for buffering values"
    )
    discard_oldest: bool = Field(
        default=True,
        description="Discard oldest values when queue full"
    )
    monitoring_mode: MonitoringMode = Field(
        default=MonitoringMode.REPORTING,
        description="Monitoring mode"
    )
    data_change_trigger: DataChangeTrigger = Field(
        default=DataChangeTrigger.STATUS_VALUE,
        description="What triggers a data change notification"
    )
    deadband_type: Optional[str] = Field(
        default=None,
        description="Deadband type: None, Absolute, Percent"
    )
    deadband_value: float = Field(
        default=0.0,
        description="Deadband value"
    )


class SubscriptionConfig(BaseModel):
    """Configuration for an OPC-UA subscription."""

    subscription_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique subscription identifier"
    )
    publishing_interval_ms: int = Field(
        default=1000,
        description="Publishing interval in milliseconds"
    )
    lifetime_count: int = Field(
        default=10000,
        description="Lifetime count (multiples of publishing interval)"
    )
    max_keepalive_count: int = Field(
        default=10,
        description="Max keep-alive count"
    )
    max_notifications_per_publish: int = Field(
        default=0,
        description="Max notifications per publish (0 = unlimited)"
    )
    priority: int = Field(
        default=0,
        description="Subscription priority (0-255)"
    )
    publishing_enabled: bool = Field(
        default=True,
        description="Whether publishing is enabled"
    )
    monitored_items: List[MonitoredItemConfig] = Field(
        default_factory=list,
        description="Items to monitor"
    )


class DataChangeNotification(BaseModel):
    """Data change notification from a subscription."""

    subscription_id: str = Field(
        ...,
        description="Source subscription ID"
    )
    sequence_number: int = Field(
        ...,
        description="Notification sequence number"
    )
    publish_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="When notification was published"
    )
    values: List[NodeValue] = Field(
        default_factory=list,
        description="Changed values"
    )


# =============================================================================
# Historical Data Access (HDA) Models
# =============================================================================


class AggregateType(str, Enum):
    """OPC-UA Historical Aggregate Types."""

    AVERAGE = "Average"
    TIME_AVERAGE = "TimeAverage"
    TOTAL = "Total"
    MINIMUM = "Minimum"
    MAXIMUM = "Maximum"
    MINIMUM_ACTUAL_TIME = "MinimumActualTime"
    MAXIMUM_ACTUAL_TIME = "MaximumActualTime"
    RANGE = "Range"
    COUNT = "Count"
    DURATION_GOOD = "DurationGood"
    DURATION_BAD = "DurationBad"
    PERCENT_GOOD = "PercentGood"
    PERCENT_BAD = "PercentBad"
    START = "Start"
    END = "End"
    DELTA = "Delta"
    STANDARD_DEVIATION = "StandardDeviation"
    VARIANCE = "Variance"
    ANNOTATION_COUNT = "AnnotationCount"


class HistoricalReadConfig(BaseModel):
    """Configuration for historical data read operations."""

    node_ids: List[str] = Field(
        ...,
        description="Node IDs to read historical data from"
    )
    start_time: datetime = Field(
        ...,
        description="Start of time range"
    )
    end_time: datetime = Field(
        ...,
        description="End of time range"
    )
    aggregate_type: Optional[AggregateType] = Field(
        default=None,
        description="Aggregate type for processed data"
    )
    processing_interval_ms: Optional[int] = Field(
        default=None,
        description="Processing interval for aggregates in milliseconds"
    )
    max_values_per_node: int = Field(
        default=10000,
        description="Maximum values to return per node"
    )
    return_bounds: bool = Field(
        default=False,
        description="Whether to return bounding values"
    )


class HistoricalValue(BaseModel):
    """A single historical data point."""

    timestamp: datetime = Field(
        ...,
        description="Timestamp of the historical value"
    )
    value: Any = Field(
        ...,
        description="The historical value"
    )
    quality: OPCUAQuality = Field(
        default=OPCUAQuality.GOOD,
        description="Quality of the historical value"
    )


class HistoricalDataResult(BaseModel):
    """Result of a historical data read operation."""

    node_id: str = Field(
        ...,
        description="Node ID for this result"
    )
    values: List[HistoricalValue] = Field(
        default_factory=list,
        description="Historical values"
    )
    continuation_point: Optional[str] = Field(
        default=None,
        description="Continuation point for paginated results"
    )
    has_more: bool = Field(
        default=False,
        description="Whether more data is available"
    )
    aggregate_type: Optional[AggregateType] = Field(
        default=None,
        description="Aggregate type if processed data"
    )
    provenance: Optional[DataProvenance] = Field(
        default=None,
        description="Data provenance information"
    )


# =============================================================================
# Connection and Session Models
# =============================================================================


class SecurityPolicy(str, Enum):
    """OPC-UA Security Policies."""

    NONE = "None"
    BASIC128RSA15 = "Basic128Rsa15"
    BASIC256 = "Basic256"
    BASIC256SHA256 = "Basic256Sha256"
    AES128_SHA256_RSAOAEP = "Aes128_Sha256_RsaOaep"
    AES256_SHA256_RSAPSS = "Aes256_Sha256_RsaPss"


class MessageSecurityMode(str, Enum):
    """OPC-UA Message Security Modes."""

    NONE = "None"
    SIGN = "Sign"
    SIGN_AND_ENCRYPT = "SignAndEncrypt"


class AuthenticationType(str, Enum):
    """OPC-UA Authentication Types."""

    ANONYMOUS = "Anonymous"
    USERNAME_PASSWORD = "UsernamePassword"
    CERTIFICATE = "Certificate"
    ISSUED_TOKEN = "IssuedToken"


class ConnectionConfig(BaseModel):
    """OPC-UA connection configuration."""

    endpoint_url: str = Field(
        ...,
        description="OPC-UA server endpoint URL"
    )
    security_policy: SecurityPolicy = Field(
        default=SecurityPolicy.BASIC256SHA256,
        description="Security policy to use"
    )
    security_mode: MessageSecurityMode = Field(
        default=MessageSecurityMode.SIGN_AND_ENCRYPT,
        description="Message security mode"
    )
    authentication_type: AuthenticationType = Field(
        default=AuthenticationType.ANONYMOUS,
        description="Authentication method"
    )
    username: Optional[str] = Field(
        default=None,
        description="Username for authentication"
    )
    password: Optional[str] = Field(
        default=None,
        description="Password for authentication (use secrets manager)"
    )
    certificate_path: Optional[str] = Field(
        default=None,
        description="Path to client certificate"
    )
    private_key_path: Optional[str] = Field(
        default=None,
        description="Path to client private key"
    )
    server_certificate_path: Optional[str] = Field(
        default=None,
        description="Path to trusted server certificate"
    )
    application_name: str = Field(
        default="GreenLang-OPC-UA-Client",
        description="Application name for OPC-UA"
    )
    application_uri: str = Field(
        default="urn:greenlang:opcua:client",
        description="Application URI"
    )
    session_timeout_ms: int = Field(
        default=3600000,
        description="Session timeout in milliseconds"
    )
    secure_channel_lifetime_ms: int = Field(
        default=3600000,
        description="Secure channel lifetime in milliseconds"
    )
    request_timeout_ms: int = Field(
        default=30000,
        description="Request timeout in milliseconds"
    )
    max_message_size: int = Field(
        default=4194304,
        description="Maximum message size in bytes"
    )


class SessionInfo(BaseModel):
    """OPC-UA session information."""

    session_id: str = Field(
        ...,
        description="Session ID"
    )
    authentication_token: Optional[str] = Field(
        default=None,
        description="Authentication token"
    )
    session_timeout: int = Field(
        ...,
        description="Session timeout in milliseconds"
    )
    server_nonce: Optional[bytes] = Field(
        default=None,
        description="Server nonce"
    )
    server_certificate: Optional[bytes] = Field(
        default=None,
        description="Server certificate"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When session was created"
    )
    last_activity: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last activity timestamp"
    )


# =============================================================================
# Server Information Models
# =============================================================================


class ServerInfo(BaseModel):
    """OPC-UA server information from discovery."""

    application_uri: str = Field(
        ...,
        description="Server application URI"
    )
    product_uri: Optional[str] = Field(
        default=None,
        description="Product URI"
    )
    application_name: str = Field(
        ...,
        description="Human-readable server name"
    )
    application_type: str = Field(
        ...,
        description="Application type (Server, Client, etc.)"
    )
    gateway_server_uri: Optional[str] = Field(
        default=None,
        description="Gateway server URI if applicable"
    )
    discovery_profile_uri: Optional[str] = Field(
        default=None,
        description="Discovery profile URI"
    )
    discovery_urls: List[str] = Field(
        default_factory=list,
        description="Discovery endpoint URLs"
    )


class EndpointInfo(BaseModel):
    """OPC-UA endpoint information."""

    endpoint_url: str = Field(
        ...,
        description="Endpoint URL"
    )
    server: ServerInfo = Field(
        ...,
        description="Server information"
    )
    security_policy_uri: str = Field(
        ...,
        description="Security policy URI"
    )
    security_mode: MessageSecurityMode = Field(
        ...,
        description="Message security mode"
    )
    user_identity_tokens: List[str] = Field(
        default_factory=list,
        description="Supported authentication methods"
    )
    security_level: int = Field(
        default=0,
        description="Relative security level"
    )


# =============================================================================
# Error Models
# =============================================================================


class OPCUAError(BaseModel):
    """OPC-UA error information."""

    status_code: int = Field(
        ...,
        description="OPC-UA status code"
    )
    message: str = Field(
        ...,
        description="Error message"
    )
    node_id: Optional[str] = Field(
        default=None,
        description="Related node ID if applicable"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When error occurred"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Quality codes
    "OPCUAQuality",
    "QualityLevel",
    # Data types
    "OPCUADataType",
    "NodeClass",
    "AccessLevel",
    # Node values
    "DataProvenance",
    "NodeValue",
    "NodeValueBatch",
    # Node information
    "NodeInfo",
    "BrowseResult",
    # Subscriptions
    "MonitoringMode",
    "DataChangeTrigger",
    "MonitoredItemConfig",
    "SubscriptionConfig",
    "DataChangeNotification",
    # Historical data
    "AggregateType",
    "HistoricalReadConfig",
    "HistoricalValue",
    "HistoricalDataResult",
    # Connection
    "SecurityPolicy",
    "MessageSecurityMode",
    "AuthenticationType",
    "ConnectionConfig",
    "SessionInfo",
    # Server info
    "ServerInfo",
    "EndpointInfo",
    # Errors
    "OPCUAError",
]
