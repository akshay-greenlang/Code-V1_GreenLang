"""
OPC-UA Connector Module for GL-015 INSULSCAN (Insulation Inspection Agent).

Provides enterprise-grade OPC-UA integration for plant automation systems:
- Connect to OPC-UA servers (DCS, PLCs, SCADA)
- Subscribe to temperature sensor tags
- Read operating temperatures from control systems
- Async client with automatic reconnection logic
- Tag mapping and browsing capabilities
- Support for OPC-UA security policies

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import asyncio
import logging
import uuid
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict, field_validator

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class OPCUASecurityPolicy(str, Enum):
    """OPC-UA security policies."""

    NONE = "None"
    BASIC128RSA15 = "Basic128Rsa15"
    BASIC256 = "Basic256"
    BASIC256SHA256 = "Basic256Sha256"
    AES128_SHA256_RSAOAEP = "Aes128_Sha256_RsaOaep"
    AES256_SHA256_RSAPSS = "Aes256_Sha256_RsaPss"


class OPCUASecurityMode(str, Enum):
    """OPC-UA security modes."""

    NONE = "None"
    SIGN = "Sign"
    SIGN_AND_ENCRYPT = "SignAndEncrypt"


class OPCUAAuthenticationType(str, Enum):
    """OPC-UA authentication types."""

    ANONYMOUS = "anonymous"
    USERNAME_PASSWORD = "username_password"
    CERTIFICATE = "certificate"


class NodeClass(str, Enum):
    """OPC-UA node classes."""

    OBJECT = "Object"
    VARIABLE = "Variable"
    METHOD = "Method"
    OBJECT_TYPE = "ObjectType"
    VARIABLE_TYPE = "VariableType"
    REFERENCE_TYPE = "ReferenceType"
    DATA_TYPE = "DataType"
    VIEW = "View"


class DataQuality(str, Enum):
    """OPC-UA data quality status."""

    GOOD = "Good"
    GOOD_LOCAL_OVERRIDE = "GoodLocalOverride"
    UNCERTAIN = "Uncertain"
    UNCERTAIN_INITIAL = "UncertainInitial"
    BAD = "Bad"
    BAD_COMMUNICATION = "BadCommunication"
    BAD_SENSOR = "BadSensor"
    BAD_CONFIGURATION = "BadConfiguration"


class SubscriptionState(str, Enum):
    """Subscription state."""

    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"


class ConnectionState(str, Enum):
    """Connection state."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


# =============================================================================
# Custom Exceptions
# =============================================================================


class OPCUAError(Exception):
    """Base exception for OPC-UA errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class OPCUAConnectionError(OPCUAError):
    """OPC-UA connection error."""
    pass


class OPCUASubscriptionError(OPCUAError):
    """OPC-UA subscription error."""
    pass


class OPCUAReadError(OPCUAError):
    """OPC-UA read error."""
    pass


class OPCUAWriteError(OPCUAError):
    """OPC-UA write error."""
    pass


class OPCUABrowseError(OPCUAError):
    """OPC-UA browse error."""
    pass


class OPCUASecurityError(OPCUAError):
    """OPC-UA security error."""
    pass


# =============================================================================
# Pydantic Models - Configuration
# =============================================================================


class OPCUAServerConfig(BaseModel):
    """OPC-UA server connection configuration."""

    model_config = ConfigDict(extra="forbid")

    endpoint_url: str = Field(
        ...,
        description="OPC-UA server endpoint URL (e.g., opc.tcp://localhost:4840)"
    )
    server_name: str = Field(
        default="OPCUAServer",
        description="Friendly name for the server"
    )

    # Security settings
    security_policy: OPCUASecurityPolicy = Field(
        default=OPCUASecurityPolicy.BASIC256SHA256,
        description="Security policy"
    )
    security_mode: OPCUASecurityMode = Field(
        default=OPCUASecurityMode.SIGN_AND_ENCRYPT,
        description="Security mode"
    )

    # Authentication
    auth_type: OPCUAAuthenticationType = Field(
        default=OPCUAAuthenticationType.USERNAME_PASSWORD,
        description="Authentication type"
    )
    username: Optional[str] = Field(
        default=None,
        description="Username for authentication"
    )
    password: Optional[str] = Field(
        default=None,
        description="Password for authentication"
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

    # Application identity
    application_name: str = Field(
        default="GL-015-INSULSCAN",
        description="Application name for OPC-UA"
    )
    application_uri: str = Field(
        default="urn:greenlang:insulscan:client",
        description="Application URI"
    )
    product_uri: str = Field(
        default="urn:greenlang:insulscan",
        description="Product URI"
    )

    @field_validator('endpoint_url')
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        """Validate OPC-UA endpoint URL."""
        if not v.startswith(('opc.tcp://', 'opc.https://')):
            raise ValueError('Endpoint must start with opc.tcp:// or opc.https://')
        return v


class SubscriptionConfig(BaseModel):
    """Subscription configuration for OPC-UA."""

    model_config = ConfigDict(extra="forbid")

    subscription_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique subscription identifier"
    )
    subscription_name: str = Field(
        default="TemperatureSubscription",
        description="Friendly subscription name"
    )

    # Timing parameters
    publishing_interval_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Publishing interval in milliseconds"
    )
    lifetime_count: int = Field(
        default=10000,
        ge=100,
        description="Lifetime count"
    )
    max_keep_alive_count: int = Field(
        default=10,
        ge=1,
        description="Maximum keep alive count"
    )
    max_notifications_per_publish: int = Field(
        default=1000,
        ge=1,
        description="Maximum notifications per publish"
    )
    priority: int = Field(
        default=0,
        ge=0,
        le=255,
        description="Subscription priority"
    )

    # Monitored item defaults
    sampling_interval_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Default sampling interval for monitored items"
    )
    queue_size: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Default queue size for monitored items"
    )
    discard_oldest: bool = Field(
        default=True,
        description="Discard oldest when queue is full"
    )


class OPCUAConnectorConfig(BaseModel):
    """Configuration for OPC-UA connector."""

    model_config = ConfigDict(extra="forbid")

    connector_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique connector identifier"
    )
    connector_name: str = Field(
        default="OPCUA-Connector",
        description="Connector name"
    )

    # Server configuration
    server_config: OPCUAServerConfig = Field(
        ...,
        description="OPC-UA server configuration"
    )

    # Default subscription configuration
    default_subscription_config: SubscriptionConfig = Field(
        default_factory=SubscriptionConfig,
        description="Default subscription configuration"
    )

    # Connection settings
    connection_timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="Connection timeout"
    )
    session_timeout_ms: int = Field(
        default=60000,
        ge=10000,
        le=3600000,
        description="Session timeout in milliseconds"
    )

    # Reconnection settings
    reconnect_enabled: bool = Field(
        default=True,
        description="Enable automatic reconnection"
    )
    reconnect_initial_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Initial reconnection delay"
    )
    reconnect_max_delay_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=600.0,
        description="Maximum reconnection delay"
    )
    reconnect_max_attempts: int = Field(
        default=0,
        ge=0,
        description="Maximum reconnection attempts (0 = unlimited)"
    )

    # Data quality settings
    accept_uncertain_values: bool = Field(
        default=False,
        description="Accept uncertain quality values"
    )
    data_change_filter_enabled: bool = Field(
        default=True,
        description="Enable data change filtering"
    )
    deadband_value: float = Field(
        default=0.1,
        ge=0.0,
        description="Deadband value for data change filter"
    )

    # Buffering
    buffer_size: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Internal buffer size for received values"
    )

    # Health check
    health_check_enabled: bool = Field(
        default=True,
        description="Enable periodic health checks"
    )
    health_check_interval_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Health check interval"
    )


# =============================================================================
# Data Models - Tag Mapping
# =============================================================================


class OPCUATag(BaseModel):
    """OPC-UA tag definition for temperature sensors."""

    model_config = ConfigDict(frozen=False)

    tag_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique tag identifier"
    )
    tag_name: str = Field(
        ...,
        description="Human-readable tag name"
    )
    node_id: str = Field(
        ...,
        description="OPC-UA Node ID (e.g., ns=2;s=Temperature.Sensor1)"
    )

    # Tag metadata
    description: Optional[str] = Field(
        default=None,
        description="Tag description"
    )
    engineering_unit: str = Field(
        default="degC",
        description="Engineering unit"
    )
    data_type: str = Field(
        default="Double",
        description="Expected data type"
    )

    # Equipment mapping
    equipment_id: Optional[str] = Field(
        default=None,
        description="Associated equipment ID"
    )
    asset_id: Optional[str] = Field(
        default=None,
        description="Associated asset ID"
    )
    measurement_point: Optional[str] = Field(
        default=None,
        description="Measurement point identifier"
    )

    # Value ranges
    low_limit: Optional[float] = Field(
        default=None,
        description="Low engineering limit"
    )
    high_limit: Optional[float] = Field(
        default=None,
        description="High engineering limit"
    )
    alarm_low: Optional[float] = Field(
        default=None,
        description="Low alarm threshold"
    )
    alarm_high: Optional[float] = Field(
        default=None,
        description="High alarm threshold"
    )

    # Sampling
    sampling_interval_ms: Optional[int] = Field(
        default=None,
        description="Override sampling interval"
    )
    deadband: Optional[float] = Field(
        default=None,
        description="Override deadband value"
    )

    # Status
    enabled: bool = Field(
        default=True,
        description="Tag enabled for monitoring"
    )
    is_simulated: bool = Field(
        default=False,
        description="Tag is simulated/synthetic"
    )


class TagGroup(BaseModel):
    """Group of related OPC-UA tags."""

    model_config = ConfigDict(frozen=False)

    group_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique group identifier"
    )
    group_name: str = Field(
        ...,
        description="Group name"
    )
    description: Optional[str] = Field(
        default=None,
        description="Group description"
    )
    tags: List[OPCUATag] = Field(
        default_factory=list,
        description="Tags in this group"
    )
    equipment_id: Optional[str] = Field(
        default=None,
        description="Associated equipment ID"
    )
    functional_location: Optional[str] = Field(
        default=None,
        description="Functional location"
    )


# =============================================================================
# Data Models - Values and Events
# =============================================================================


class OPCUAValue(BaseModel):
    """OPC-UA value with quality and timestamp."""

    model_config = ConfigDict(frozen=True)

    tag_id: str = Field(..., description="Tag identifier")
    node_id: str = Field(..., description="OPC-UA node ID")
    value: Any = Field(..., description="Value")
    data_type: str = Field(..., description="Data type")
    quality: DataQuality = Field(..., description="Data quality")
    source_timestamp: datetime = Field(..., description="Source timestamp")
    server_timestamp: datetime = Field(..., description="Server timestamp")
    received_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Client receive timestamp"
    )


class TemperatureReading(BaseModel):
    """Temperature reading from OPC-UA."""

    model_config = ConfigDict(frozen=True)

    tag_id: str = Field(..., description="Tag identifier")
    tag_name: str = Field(..., description="Tag name")
    temperature_c: float = Field(..., description="Temperature in Celsius")
    quality: DataQuality = Field(..., description="Data quality")
    timestamp: datetime = Field(..., description="Reading timestamp")
    equipment_id: Optional[str] = Field(
        default=None,
        description="Associated equipment"
    )
    in_alarm: bool = Field(
        default=False,
        description="Reading is in alarm condition"
    )
    alarm_type: Optional[str] = Field(
        default=None,
        description="Alarm type if in alarm"
    )


class DataChangeEvent(BaseModel):
    """OPC-UA data change event."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Event identifier"
    )
    subscription_id: str = Field(..., description="Subscription identifier")
    values: List[OPCUAValue] = Field(..., description="Changed values")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event timestamp"
    )


class BrowseResult(BaseModel):
    """Result of browsing OPC-UA node."""

    model_config = ConfigDict(frozen=True)

    node_id: str = Field(..., description="Node ID")
    browse_name: str = Field(..., description="Browse name")
    display_name: str = Field(..., description="Display name")
    node_class: NodeClass = Field(..., description="Node class")
    type_definition: Optional[str] = Field(
        default=None,
        description="Type definition"
    )
    is_forward: bool = Field(
        default=True,
        description="Forward reference"
    )
    children: List["BrowseResult"] = Field(
        default_factory=list,
        description="Child nodes"
    )


BrowseResult.model_rebuild()


class ServerInfo(BaseModel):
    """OPC-UA server information."""

    model_config = ConfigDict(frozen=True)

    server_name: str = Field(..., description="Server name")
    product_uri: str = Field(..., description="Product URI")
    manufacturer_name: Optional[str] = Field(
        default=None,
        description="Manufacturer name"
    )
    product_name: Optional[str] = Field(
        default=None,
        description="Product name"
    )
    software_version: Optional[str] = Field(
        default=None,
        description="Software version"
    )
    build_number: Optional[str] = Field(
        default=None,
        description="Build number"
    )
    build_date: Optional[datetime] = Field(
        default=None,
        description="Build date"
    )


class HealthCheckResult(BaseModel):
    """Health check result."""

    model_config = ConfigDict(frozen=True)

    healthy: bool = Field(..., description="Is healthy")
    connection_state: ConnectionState = Field(..., description="Connection state")
    server_status: Optional[str] = Field(
        default=None,
        description="Server status"
    )
    active_subscriptions: int = Field(
        default=0,
        description="Active subscriptions"
    )
    monitored_items: int = Field(
        default=0,
        description="Monitored items count"
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Latency in milliseconds"
    )
    last_data_received: Optional[datetime] = Field(
        default=None,
        description="Last data received timestamp"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if unhealthy"
    )
    checked_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Check timestamp"
    )


# =============================================================================
# OPC-UA Connector
# =============================================================================


class OPCUAConnector:
    """
    OPC-UA Connector for GL-015 INSULSCAN.

    Provides async OPC-UA client with:
    - Secure connection to OPC-UA servers
    - Subscription-based data acquisition
    - Automatic reconnection with exponential backoff
    - Tag mapping and browsing
    - Temperature sensor monitoring
    - Data quality validation

    Note: This is an interface implementation. In production, use with
    asyncua (python-opcua-async) or opcua-asyncio library.
    """

    def __init__(self, config: OPCUAConnectorConfig) -> None:
        """
        Initialize OPC-UA connector.

        Args:
            config: Connector configuration
        """
        self._config = config
        self._logger = logging.getLogger(
            f"{__name__}.{config.connector_name}"
        )

        # Connection state
        self._state = ConnectionState.DISCONNECTED
        self._client: Optional[Any] = None  # OPC-UA client instance
        self._session: Optional[Any] = None

        # Subscriptions
        self._subscriptions: Dict[str, Any] = {}
        self._monitored_items: Dict[str, OPCUATag] = {}

        # Data buffers
        self._value_buffer: asyncio.Queue = asyncio.Queue(
            maxsize=config.buffer_size
        )
        self._last_values: Dict[str, OPCUAValue] = {}

        # Callbacks
        self._data_change_callbacks: List[Callable[[DataChangeEvent], None]] = []

        # Reconnection state
        self._reconnect_task: Optional[asyncio.Task] = None
        self._reconnect_attempts = 0

        # Health check
        self._health_check_task: Optional[asyncio.Task] = None
        self._last_data_received: Optional[datetime] = None

        # Server info
        self._server_info: Optional[ServerInfo] = None

    @property
    def config(self) -> OPCUAConnectorConfig:
        """Get connector configuration."""
        return self._config

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == ConnectionState.CONNECTED

    @property
    def server_info(self) -> Optional[ServerInfo]:
        """Get server information."""
        return self._server_info

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self) -> None:
        """
        Connect to OPC-UA server.

        Establishes secure session with authentication.

        Raises:
            OPCUAConnectionError: If connection fails
            OPCUASecurityError: If security handshake fails
        """
        if self._state == ConnectionState.CONNECTED:
            self._logger.warning("Already connected")
            return

        self._state = ConnectionState.CONNECTING
        server_config = self._config.server_config

        self._logger.info(
            f"Connecting to OPC-UA server: {server_config.endpoint_url}"
        )

        try:
            # In production, use asyncua library:
            # from asyncua import Client
            # self._client = Client(url=server_config.endpoint_url)

            # Configure security
            await self._configure_security()

            # Configure authentication
            await self._configure_authentication()

            # Connect
            # await self._client.connect()

            # Get server info
            self._server_info = await self._get_server_info()

            self._state = ConnectionState.CONNECTED
            self._reconnect_attempts = 0

            self._logger.info(
                f"Connected to OPC-UA server: {self._server_info.server_name if self._server_info else 'Unknown'}"
            )

            # Start health check if enabled
            if self._config.health_check_enabled:
                self._health_check_task = asyncio.create_task(
                    self._health_check_loop()
                )

        except Exception as e:
            self._state = ConnectionState.ERROR
            self._logger.error(f"Failed to connect to OPC-UA server: {e}")
            raise OPCUAConnectionError(
                f"Connection failed: {e}",
                details={
                    "endpoint": server_config.endpoint_url,
                    "security_policy": server_config.security_policy.value,
                }
            )

    async def disconnect(self) -> None:
        """Disconnect from OPC-UA server."""
        self._logger.info("Disconnecting from OPC-UA server")

        # Cancel health check
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Cancel reconnection
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        # Unsubscribe all
        for sub_id in list(self._subscriptions.keys()):
            await self.unsubscribe(sub_id)

        # Disconnect client
        if self._client:
            try:
                # await self._client.disconnect()
                pass
            except Exception as e:
                self._logger.warning(f"Error during disconnect: {e}")

        self._client = None
        self._session = None
        self._state = ConnectionState.DISCONNECTED

    async def reconnect(self) -> None:
        """
        Reconnect to OPC-UA server.

        Uses exponential backoff with configurable delays.
        """
        if not self._config.reconnect_enabled:
            self._logger.warning("Reconnection is disabled")
            return

        self._state = ConnectionState.RECONNECTING
        delay = self._config.reconnect_initial_delay_seconds

        while True:
            self._reconnect_attempts += 1

            if (self._config.reconnect_max_attempts > 0 and
                    self._reconnect_attempts > self._config.reconnect_max_attempts):
                self._logger.error(
                    f"Max reconnection attempts ({self._config.reconnect_max_attempts}) exceeded"
                )
                self._state = ConnectionState.ERROR
                raise OPCUAConnectionError("Max reconnection attempts exceeded")

            try:
                self._logger.info(
                    f"Reconnection attempt {self._reconnect_attempts} "
                    f"(delay: {delay:.1f}s)"
                )

                await self.disconnect()
                await asyncio.sleep(delay)
                await self.connect()

                # Re-subscribe to previous subscriptions
                await self._resubscribe_all()

                self._logger.info("Reconnection successful")
                return

            except Exception as e:
                self._logger.warning(f"Reconnection attempt failed: {e}")
                delay = min(
                    delay * 2,
                    self._config.reconnect_max_delay_seconds
                )

    async def _configure_security(self) -> None:
        """Configure OPC-UA security settings."""
        server_config = self._config.server_config

        if server_config.security_policy == OPCUASecurityPolicy.NONE:
            self._logger.warning("Using unsecured connection (no security policy)")
            return

        # In production:
        # await self._client.set_security(
        #     SecurityPolicyBasic256Sha256,
        #     server_config.certificate_path,
        #     server_config.private_key_path
        # )

        self._logger.info(
            f"Configured security: {server_config.security_policy.value} "
            f"({server_config.security_mode.value})"
        )

    async def _configure_authentication(self) -> None:
        """Configure OPC-UA authentication."""
        server_config = self._config.server_config

        if server_config.auth_type == OPCUAAuthenticationType.ANONYMOUS:
            self._logger.info("Using anonymous authentication")

        elif server_config.auth_type == OPCUAAuthenticationType.USERNAME_PASSWORD:
            if not server_config.username or not server_config.password:
                raise OPCUASecurityError("Username and password required")
            # self._client.set_user(server_config.username)
            # self._client.set_password(server_config.password)
            self._logger.info(f"Using username/password authentication: {server_config.username}")

        elif server_config.auth_type == OPCUAAuthenticationType.CERTIFICATE:
            if not server_config.certificate_path:
                raise OPCUASecurityError("Certificate path required")
            self._logger.info("Using certificate authentication")

    async def _get_server_info(self) -> ServerInfo:
        """Get server information."""
        # In production, read from server's ServerArray node
        return ServerInfo(
            server_name=self._config.server_config.server_name,
            product_uri=self._config.server_config.application_uri,
            manufacturer_name="Unknown",
            product_name="OPC-UA Server",
            software_version="1.0.0",
        )

    # =========================================================================
    # Subscription Management
    # =========================================================================

    async def create_subscription(
        self,
        tags: List[OPCUATag],
        config: Optional[SubscriptionConfig] = None,
        callback: Optional[Callable[[DataChangeEvent], None]] = None
    ) -> str:
        """
        Create subscription for tags.

        Args:
            tags: Tags to monitor
            config: Subscription configuration
            callback: Optional callback for data changes

        Returns:
            Subscription ID

        Raises:
            OPCUASubscriptionError: If subscription creation fails
        """
        if not self.is_connected:
            raise OPCUAConnectionError("Not connected to server")

        config = config or self._config.default_subscription_config
        subscription_id = config.subscription_id

        self._logger.info(
            f"Creating subscription {subscription_id} with {len(tags)} tags"
        )

        try:
            # In production:
            # subscription = await self._client.create_subscription(
            #     period=config.publishing_interval_ms,
            #     handler=self._subscription_handler
            # )

            # Add monitored items
            for tag in tags:
                if not tag.enabled:
                    continue

                # node = self._client.get_node(tag.node_id)
                # handle = await subscription.subscribe_data_change(node)

                self._monitored_items[tag.tag_id] = tag
                self._logger.debug(f"Added monitored item: {tag.tag_name} ({tag.node_id})")

            # Store subscription
            self._subscriptions[subscription_id] = {
                "config": config,
                "tags": tags,
                "callback": callback,
                # "subscription": subscription,
                "created_at": datetime.utcnow(),
            }

            if callback:
                self._data_change_callbacks.append(callback)

            self._logger.info(f"Created subscription: {subscription_id}")
            return subscription_id

        except Exception as e:
            self._logger.error(f"Failed to create subscription: {e}")
            raise OPCUASubscriptionError(
                f"Subscription creation failed: {e}",
                details={"tag_count": len(tags)}
            )

    async def unsubscribe(self, subscription_id: str) -> None:
        """
        Remove subscription.

        Args:
            subscription_id: Subscription to remove
        """
        if subscription_id not in self._subscriptions:
            self._logger.warning(f"Subscription not found: {subscription_id}")
            return

        try:
            sub_info = self._subscriptions[subscription_id]

            # Remove monitored items
            for tag in sub_info["tags"]:
                self._monitored_items.pop(tag.tag_id, None)

            # Remove callback
            if sub_info.get("callback"):
                try:
                    self._data_change_callbacks.remove(sub_info["callback"])
                except ValueError:
                    pass

            # Delete subscription
            # await sub_info["subscription"].delete()

            del self._subscriptions[subscription_id]
            self._logger.info(f"Removed subscription: {subscription_id}")

        except Exception as e:
            self._logger.error(f"Error removing subscription: {e}")

    async def _resubscribe_all(self) -> None:
        """Resubscribe to all previous subscriptions after reconnect."""
        previous_subs = list(self._subscriptions.items())
        self._subscriptions.clear()
        self._monitored_items.clear()

        for sub_id, sub_info in previous_subs:
            try:
                await self.create_subscription(
                    tags=sub_info["tags"],
                    config=sub_info["config"],
                    callback=sub_info.get("callback")
                )
            except Exception as e:
                self._logger.error(f"Failed to resubscribe {sub_id}: {e}")

    def _subscription_handler(self, node, value, data) -> None:
        """Handle subscription data changes."""
        # This would be called by asyncua library
        pass

    async def _process_data_change(
        self,
        node_id: str,
        value: Any,
        quality: DataQuality,
        source_timestamp: datetime,
        server_timestamp: datetime
    ) -> None:
        """Process incoming data change."""
        # Find tag
        tag = None
        for t in self._monitored_items.values():
            if t.node_id == node_id:
                tag = t
                break

        if not tag:
            self._logger.warning(f"Unknown node: {node_id}")
            return

        # Create value object
        opcua_value = OPCUAValue(
            tag_id=tag.tag_id,
            node_id=node_id,
            value=value,
            data_type=tag.data_type,
            quality=quality,
            source_timestamp=source_timestamp,
            server_timestamp=server_timestamp,
        )

        # Update last values
        self._last_values[tag.tag_id] = opcua_value
        self._last_data_received = datetime.utcnow()

        # Add to buffer
        try:
            self._value_buffer.put_nowait(opcua_value)
        except asyncio.QueueFull:
            # Drop oldest
            try:
                self._value_buffer.get_nowait()
                self._value_buffer.put_nowait(opcua_value)
            except asyncio.QueueEmpty:
                pass

        # Notify callbacks
        event = DataChangeEvent(
            subscription_id="",  # Would be set from actual subscription
            values=[opcua_value],
        )
        for callback in self._data_change_callbacks:
            try:
                callback(event)
            except Exception as e:
                self._logger.error(f"Callback error: {e}")

    # =========================================================================
    # Data Reading
    # =========================================================================

    async def read_value(self, node_id: str) -> OPCUAValue:
        """
        Read single value from OPC-UA node.

        Args:
            node_id: OPC-UA node ID

        Returns:
            Read value

        Raises:
            OPCUAReadError: If read fails
        """
        if not self.is_connected:
            raise OPCUAConnectionError("Not connected to server")

        try:
            # In production:
            # node = self._client.get_node(node_id)
            # value = await node.read_value()
            # data_value = await node.read_data_value()

            # Mock implementation
            return OPCUAValue(
                tag_id="",
                node_id=node_id,
                value=0.0,
                data_type="Double",
                quality=DataQuality.GOOD,
                source_timestamp=datetime.utcnow(),
                server_timestamp=datetime.utcnow(),
            )

        except Exception as e:
            self._logger.error(f"Read error for {node_id}: {e}")
            raise OPCUAReadError(f"Read failed: {e}", details={"node_id": node_id})

    async def read_values(self, node_ids: List[str]) -> List[OPCUAValue]:
        """
        Read multiple values.

        Args:
            node_ids: List of node IDs

        Returns:
            List of read values
        """
        values = []
        for node_id in node_ids:
            try:
                value = await self.read_value(node_id)
                values.append(value)
            except OPCUAReadError as e:
                self._logger.warning(f"Failed to read {node_id}: {e}")
        return values

    async def read_temperature(self, tag: OPCUATag) -> TemperatureReading:
        """
        Read temperature from a temperature sensor tag.

        Args:
            tag: Temperature sensor tag

        Returns:
            Temperature reading

        Raises:
            OPCUAReadError: If read fails
        """
        value = await self.read_value(tag.node_id)

        # Convert to temperature
        temp_c = float(value.value)

        # Check alarm conditions
        in_alarm = False
        alarm_type = None

        if tag.alarm_low is not None and temp_c < tag.alarm_low:
            in_alarm = True
            alarm_type = "LOW"
        elif tag.alarm_high is not None and temp_c > tag.alarm_high:
            in_alarm = True
            alarm_type = "HIGH"

        return TemperatureReading(
            tag_id=tag.tag_id,
            tag_name=tag.tag_name,
            temperature_c=temp_c,
            quality=value.quality,
            timestamp=value.source_timestamp,
            equipment_id=tag.equipment_id,
            in_alarm=in_alarm,
            alarm_type=alarm_type,
        )

    async def read_all_temperatures(self) -> List[TemperatureReading]:
        """
        Read all monitored temperature sensors.

        Returns:
            List of temperature readings
        """
        readings = []
        for tag in self._monitored_items.values():
            if tag.engineering_unit in ["degC", "C", "Celsius"]:
                try:
                    reading = await self.read_temperature(tag)
                    readings.append(reading)
                except Exception as e:
                    self._logger.warning(f"Failed to read {tag.tag_name}: {e}")
        return readings

    def get_last_value(self, tag_id: str) -> Optional[OPCUAValue]:
        """
        Get last received value for a tag.

        Args:
            tag_id: Tag identifier

        Returns:
            Last value or None
        """
        return self._last_values.get(tag_id)

    async def get_buffered_values(
        self,
        max_count: int = 100,
        timeout_seconds: float = 1.0
    ) -> List[OPCUAValue]:
        """
        Get buffered values from subscription.

        Args:
            max_count: Maximum values to return
            timeout_seconds: Wait timeout

        Returns:
            List of buffered values
        """
        values = []
        deadline = asyncio.get_event_loop().time() + timeout_seconds

        while len(values) < max_count:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break

            try:
                value = await asyncio.wait_for(
                    self._value_buffer.get(),
                    timeout=remaining
                )
                values.append(value)
            except asyncio.TimeoutError:
                break

        return values

    # =========================================================================
    # Node Browsing
    # =========================================================================

    async def browse(
        self,
        node_id: str = "i=85",  # Objects folder
        max_depth: int = 1
    ) -> BrowseResult:
        """
        Browse OPC-UA address space.

        Args:
            node_id: Starting node ID
            max_depth: Maximum browse depth

        Returns:
            Browse result with children

        Raises:
            OPCUABrowseError: If browse fails
        """
        if not self.is_connected:
            raise OPCUAConnectionError("Not connected to server")

        try:
            # In production:
            # node = self._client.get_node(node_id)
            # browse_name = await node.read_browse_name()
            # display_name = await node.read_display_name()
            # node_class = await node.read_node_class()
            # children = await node.get_children()

            return BrowseResult(
                node_id=node_id,
                browse_name="Objects",
                display_name="Objects",
                node_class=NodeClass.OBJECT,
                children=[],
            )

        except Exception as e:
            self._logger.error(f"Browse error: {e}")
            raise OPCUABrowseError(f"Browse failed: {e}", details={"node_id": node_id})

    async def find_temperature_sensors(
        self,
        root_node_id: str = "i=85"
    ) -> List[OPCUATag]:
        """
        Discover temperature sensor nodes.

        Args:
            root_node_id: Root node to start search

        Returns:
            List of discovered temperature sensor tags
        """
        discovered_tags = []

        # Browse and filter for temperature-related nodes
        # This would use heuristics based on browse names, types, etc.

        return discovered_tags

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> HealthCheckResult:
        """
        Perform health check.

        Returns:
            Health check result
        """
        start_time = datetime.utcnow()

        try:
            if not self.is_connected:
                return HealthCheckResult(
                    healthy=False,
                    connection_state=self._state,
                    error_message="Not connected",
                )

            # Read server status
            # status = await self._client.get_node("i=2259").read_value()

            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return HealthCheckResult(
                healthy=True,
                connection_state=self._state,
                server_status="Running",
                active_subscriptions=len(self._subscriptions),
                monitored_items=len(self._monitored_items),
                latency_ms=latency_ms,
                last_data_received=self._last_data_received,
            )

        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                connection_state=self._state,
                error_message=str(e),
            )

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while True:
            try:
                await asyncio.sleep(self._config.health_check_interval_seconds)

                result = await self.health_check()

                if not result.healthy:
                    self._logger.warning(f"Health check failed: {result.error_message}")

                    if self._config.reconnect_enabled:
                        asyncio.create_task(self.reconnect())

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Health check error: {e}")

    # =========================================================================
    # Streaming Interface
    # =========================================================================

    async def stream_values(self) -> AsyncIterator[OPCUAValue]:
        """
        Async iterator for streaming values.

        Yields:
            OPC-UA values as they arrive
        """
        while self.is_connected:
            try:
                value = await asyncio.wait_for(
                    self._value_buffer.get(),
                    timeout=5.0
                )
                yield value
            except asyncio.TimeoutError:
                continue

    async def stream_temperatures(self) -> AsyncIterator[TemperatureReading]:
        """
        Stream temperature readings.

        Yields:
            Temperature readings as they arrive
        """
        async for value in self.stream_values():
            tag = self._monitored_items.get(value.tag_id)
            if tag and tag.engineering_unit in ["degC", "C", "Celsius"]:
                in_alarm = False
                alarm_type = None

                temp_c = float(value.value)
                if tag.alarm_low is not None and temp_c < tag.alarm_low:
                    in_alarm = True
                    alarm_type = "LOW"
                elif tag.alarm_high is not None and temp_c > tag.alarm_high:
                    in_alarm = True
                    alarm_type = "HIGH"

                yield TemperatureReading(
                    tag_id=tag.tag_id,
                    tag_name=tag.tag_name,
                    temperature_c=temp_c,
                    quality=value.quality,
                    timestamp=value.source_timestamp,
                    equipment_id=tag.equipment_id,
                    in_alarm=in_alarm,
                    alarm_type=alarm_type,
                )


# =============================================================================
# Factory Functions
# =============================================================================


def create_opcua_connector(
    endpoint_url: str,
    connector_name: str = "OPCUA-Connector",
    username: Optional[str] = None,
    password: Optional[str] = None,
    security_policy: OPCUASecurityPolicy = OPCUASecurityPolicy.BASIC256SHA256,
    **kwargs
) -> OPCUAConnector:
    """
    Factory function to create OPC-UA connector.

    Args:
        endpoint_url: OPC-UA server endpoint
        connector_name: Connector name
        username: Username for authentication
        password: Password for authentication
        security_policy: Security policy
        **kwargs: Additional configuration

    Returns:
        Configured OPCUAConnector instance
    """
    server_config = OPCUAServerConfig(
        endpoint_url=endpoint_url,
        server_name=connector_name,
        security_policy=security_policy,
        auth_type=OPCUAAuthenticationType.USERNAME_PASSWORD if username else OPCUAAuthenticationType.ANONYMOUS,
        username=username,
        password=password,
    )

    config = OPCUAConnectorConfig(
        connector_name=connector_name,
        server_config=server_config,
        **kwargs
    )

    return OPCUAConnector(config)


def create_tag_from_node(
    node_id: str,
    tag_name: str,
    equipment_id: Optional[str] = None,
    **kwargs
) -> OPCUATag:
    """
    Create OPC-UA tag from node ID.

    Args:
        node_id: OPC-UA node ID
        tag_name: Tag name
        equipment_id: Associated equipment
        **kwargs: Additional tag properties

    Returns:
        Configured OPCUATag
    """
    return OPCUATag(
        tag_name=tag_name,
        node_id=node_id,
        equipment_id=equipment_id,
        **kwargs
    )
