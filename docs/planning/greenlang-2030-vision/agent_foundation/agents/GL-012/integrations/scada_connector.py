# -*- coding: utf-8 -*-
"""
SCADA Connector for GL-012 STEAMQUAL (SteamQualityController).

Provides integration with SCADA (Supervisory Control and Data Acquisition) systems
for steam quality monitoring and control including:
- Tag read/write operations
- Tag subscription for real-time updates
- Historical data retrieval
- Alarm management

Supports multiple protocols:
- OPC-DA (Data Access)
- OPC-UA (Unified Architecture)
- Modbus TCP/RTU

Features:
- Tag mapping and normalization
- Real-time subscriptions with callbacks
- Historical data queries
- Alarm generation and acknowledgment
- Connection pooling and retry logic

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

import asyncio
import logging
import time
from collections import deque
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
    Union,
)

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

from .base_connector import (
    BaseConnector,
    BaseConnectorConfig,
    ConnectionState,
    ConnectorType,
    ProtocolType,
    HealthStatus,
    HealthCheckResult,
    ConnectionError,
    ConfigurationError,
    CommunicationError,
    TimeoutError,
    with_retry,
    CircuitState,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class SCADAVendor(str, Enum):
    """Supported SCADA system vendors."""

    WONDERWARE = "wonderware"
    ROCKWELL = "rockwell"
    SIEMENS = "siemens"
    GE = "ge"
    HONEYWELL = "honeywell"
    ABB = "abb"
    SCHNEIDER = "schneider"
    YOKOGAWA = "yokogawa"
    GENERIC = "generic"


class TagDataType(str, Enum):
    """SCADA tag data types."""

    BOOL = "bool"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT16 = "uint16"
    UINT32 = "uint32"
    FLOAT = "float"
    DOUBLE = "double"
    STRING = "string"
    DATETIME = "datetime"


class TagQuality(str, Enum):
    """Tag quality codes."""

    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    NOT_CONNECTED = "not_connected"
    CONFIG_ERROR = "config_error"
    SENSOR_FAILURE = "sensor_failure"
    OUT_OF_SERVICE = "out_of_service"


class AlarmPriority(str, Enum):
    """Alarm priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlarmState(str, Enum):
    """Alarm states."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    CLEARED = "cleared"
    DISABLED = "disabled"


class SubscriptionMode(str, Enum):
    """Subscription update modes."""

    ON_CHANGE = "on_change"  # Only on value change
    PERIODIC = "periodic"  # At fixed intervals
    EXCEPTION = "exception"  # On deadband exception


# =============================================================================
# Data Models
# =============================================================================


class SCADAConnectorConfig(BaseConnectorConfig):
    """Configuration for SCADA connector."""

    model_config = ConfigDict(extra="forbid")

    connector_type: ConnectorType = Field(
        default=ConnectorType.SCADA,
        frozen=True
    )

    # SCADA system identification
    scada_id: str = Field(
        ...,
        description="Unique SCADA system identifier"
    )
    scada_name: str = Field(
        ...,
        description="SCADA system name"
    )
    scada_vendor: SCADAVendor = Field(
        default=SCADAVendor.GENERIC,
        description="SCADA vendor"
    )

    # OPC-DA settings
    opc_server_name: Optional[str] = Field(
        default=None,
        description="OPC-DA server name (e.g., 'Matrikon.OPC.Simulation')"
    )
    opc_server_host: Optional[str] = Field(
        default=None,
        description="OPC-DA server host (for DCOM)"
    )

    # OPC-UA settings
    opc_ua_endpoint: Optional[str] = Field(
        default=None,
        description="OPC-UA endpoint URL"
    )
    opc_ua_security_policy: str = Field(
        default="None",
        description="OPC-UA security policy"
    )
    opc_ua_security_mode: str = Field(
        default="None",
        description="OPC-UA security mode"
    )

    # Modbus settings
    modbus_slave_id: int = Field(
        default=1,
        ge=1,
        le=247,
        description="Modbus slave ID"
    )

    # Subscription settings
    default_update_rate_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Default subscription update rate"
    )
    default_deadband_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Default deadband for subscriptions"
    )
    max_subscriptions: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum number of subscriptions"
    )

    # Tag mapping
    tag_prefix: str = Field(
        default="",
        description="Prefix to add to all tag names"
    )
    tag_separator: str = Field(
        default=".",
        description="Tag hierarchy separator"
    )

    # Historian settings
    historian_enabled: bool = Field(
        default=True,
        description="Whether historian queries are enabled"
    )
    historian_max_points: int = Field(
        default=100000,
        ge=1000,
        description="Maximum points per historian query"
    )


class TagValue(BaseModel):
    """SCADA tag value with metadata."""

    model_config = ConfigDict(frozen=True)

    tag_name: str = Field(
        ...,
        description="Tag name"
    )
    value: Any = Field(
        ...,
        description="Tag value"
    )
    data_type: TagDataType = Field(
        default=TagDataType.DOUBLE,
        description="Data type"
    )
    quality: TagQuality = Field(
        default=TagQuality.GOOD,
        description="Value quality"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Value timestamp"
    )
    source_timestamp: Optional[datetime] = Field(
        default=None,
        description="Source timestamp from device"
    )
    engineering_units: Optional[str] = Field(
        default=None,
        description="Engineering units"
    )


class TagConfig(BaseModel):
    """Tag configuration for mapping."""

    model_config = ConfigDict(extra="forbid")

    tag_name: str = Field(
        ...,
        description="Tag name"
    )
    address: str = Field(
        ...,
        description="Tag address in SCADA system"
    )
    data_type: TagDataType = Field(
        default=TagDataType.DOUBLE,
        description="Data type"
    )
    description: str = Field(
        default="",
        description="Tag description"
    )
    engineering_units: str = Field(
        default="",
        description="Engineering units"
    )
    min_value: Optional[float] = Field(
        default=None,
        description="Minimum valid value"
    )
    max_value: Optional[float] = Field(
        default=None,
        description="Maximum valid value"
    )
    scale_factor: float = Field(
        default=1.0,
        description="Scale factor for raw value"
    )
    offset: float = Field(
        default=0.0,
        description="Offset for raw value"
    )
    deadband: float = Field(
        default=0.0,
        ge=0.0,
        description="Deadband for change detection"
    )
    read_only: bool = Field(
        default=False,
        description="Whether tag is read-only"
    )


class Subscription(BaseModel):
    """Tag subscription information."""

    model_config = ConfigDict(frozen=False)

    subscription_id: str = Field(
        ...,
        description="Unique subscription identifier"
    )
    tags: List[str] = Field(
        ...,
        description="List of subscribed tag names"
    )
    update_rate_ms: int = Field(
        default=1000,
        description="Update rate in milliseconds"
    )
    mode: SubscriptionMode = Field(
        default=SubscriptionMode.ON_CHANGE,
        description="Subscription mode"
    )
    deadband_percent: float = Field(
        default=0.0,
        description="Deadband percentage"
    )
    active: bool = Field(
        default=True,
        description="Whether subscription is active"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Subscription creation time"
    )
    callback_count: int = Field(
        default=0,
        description="Number of callbacks triggered"
    )


class AlarmData(BaseModel):
    """Alarm data for SCADA system."""

    model_config = ConfigDict(extra="forbid")

    alarm_id: str = Field(
        ...,
        description="Unique alarm identifier"
    )
    tag_name: str = Field(
        ...,
        description="Associated tag name"
    )
    message: str = Field(
        ...,
        description="Alarm message"
    )
    priority: AlarmPriority = Field(
        default=AlarmPriority.MEDIUM,
        description="Alarm priority"
    )
    value: Optional[float] = Field(
        default=None,
        description="Value that triggered alarm"
    )
    setpoint: Optional[float] = Field(
        default=None,
        description="Alarm setpoint"
    )
    state: AlarmState = Field(
        default=AlarmState.ACTIVE,
        description="Alarm state"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Alarm timestamp"
    )
    acknowledged_by: Optional[str] = Field(
        default=None,
        description="User who acknowledged alarm"
    )
    acknowledged_at: Optional[datetime] = Field(
        default=None,
        description="Acknowledgment timestamp"
    )


class AlarmSendResult(BaseModel):
    """Result of sending an alarm to SCADA."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(
        ...,
        description="Whether alarm was sent successfully"
    )
    alarm_id: str = Field(
        ...,
        description="Alarm identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Send timestamp"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )


class HistoricalDataRequest(BaseModel):
    """Request for historical data."""

    model_config = ConfigDict(extra="forbid")

    tags: List[str] = Field(
        ...,
        description="List of tag names"
    )
    start_time: datetime = Field(
        ...,
        description="Start time for query"
    )
    end_time: datetime = Field(
        ...,
        description="End time for query"
    )
    resolution_seconds: Optional[int] = Field(
        default=None,
        description="Data resolution in seconds"
    )
    aggregate_function: Optional[str] = Field(
        default=None,
        description="Aggregation function (avg, min, max, etc.)"
    )
    max_points: Optional[int] = Field(
        default=None,
        description="Maximum points to return"
    )


# =============================================================================
# SCADA Connector
# =============================================================================


class SCADAConnector(BaseConnector):
    """
    Connector for SCADA system integration.

    Provides comprehensive SCADA integration including tag read/write,
    subscriptions, historical data, and alarm management.

    Features:
    - Multi-protocol support (OPC-DA, OPC-UA, Modbus)
    - Real-time tag subscriptions with callbacks
    - Tag mapping and data normalization
    - Historical data queries
    - Alarm generation and management
    - Connection pooling and retry logic
    """

    def __init__(self, config: SCADAConnectorConfig) -> None:
        """
        Initialize SCADA connector.

        Args:
            config: SCADA configuration
        """
        super().__init__(config)
        self.scada_config: SCADAConnectorConfig = config

        # Tag management
        self._tag_configs: Dict[str, TagConfig] = {}
        self._tag_values: Dict[str, TagValue] = {}
        self._tag_history: Dict[str, deque] = {}

        # Subscriptions
        self._subscriptions: Dict[str, Subscription] = {}
        self._subscription_callbacks: Dict[str, Callable] = {}
        self._subscription_task: Optional[asyncio.Task] = None

        # Alarms
        self._active_alarms: Dict[str, AlarmData] = {}
        self._alarm_history: deque = deque(maxlen=10000)

        # Connection clients
        self._opc_da_client: Optional[Any] = None
        self._opc_ua_client: Optional[Any] = None
        self._modbus_client: Optional[Any] = None

        self._logger.info(
            f"Initialized SCADAConnector for {config.scada_name} "
            f"(vendor={config.scada_vendor.value})"
        )

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    async def connect(self, scada_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Establish connection to the SCADA system.

        Args:
            scada_config: Optional additional configuration

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails
        """
        if self._state == ConnectionState.CONNECTED:
            self._logger.warning(
                f"Already connected to SCADA system {self.scada_config.scada_name}"
            )
            return True

        self._state = ConnectionState.CONNECTING

        try:
            # Apply any additional config
            if scada_config:
                for key, value in scada_config.items():
                    if hasattr(self.scada_config, key):
                        setattr(self.scada_config, key, value)

            # Connect based on protocol
            if self.scada_config.protocol == ProtocolType.OPC_UA:
                await self._connect_opc_ua()
            elif self.scada_config.protocol == ProtocolType.OPC_DA:
                await self._connect_opc_da()
            elif self.scada_config.protocol in (ProtocolType.MODBUS_TCP, ProtocolType.MODBUS_RTU):
                await self._connect_modbus()
            else:
                raise ConfigurationError(
                    f"Unsupported protocol: {self.scada_config.protocol}",
                    connector_id=self._config.connector_id,
                )

            self._state = ConnectionState.CONNECTED

            # Start subscription processing task
            self._subscription_task = asyncio.create_task(self._subscription_loop())

            self._logger.info(
                f"Connected to SCADA system {self.scada_config.scada_name} via "
                f"{self.scada_config.protocol.value}"
            )

            await self._audit_logger.log_operation(
                operation="connect",
                status="success",
                response_summary=f"Connected to {self.scada_config.scada_name}",
            )

            return True

        except Exception as e:
            self._state = ConnectionState.ERROR
            self._logger.error(
                f"Failed to connect to SCADA system {self.scada_config.scada_name}: {e}"
            )

            await self._audit_logger.log_operation(
                operation="connect",
                status="failure",
                error_message=str(e),
            )

            raise ConnectionError(
                f"Failed to connect to SCADA: {e}",
                connector_id=self._config.connector_id,
                details={"scada_name": self.scada_config.scada_name},
            )

    async def _connect_opc_ua(self) -> None:
        """Establish OPC-UA connection."""
        # In production, would use asyncua library
        # from asyncua import Client as OPCUAClient
        # self._opc_ua_client = OPCUAClient(self.scada_config.opc_ua_endpoint)
        # await self._opc_ua_client.connect()

        self._opc_ua_client = {
            "type": "opc_ua",
            "endpoint": self.scada_config.opc_ua_endpoint or f"opc.tcp://{self.scada_config.host}:{self.scada_config.port}",
            "connected": True,
        }
        self._logger.debug(
            f"OPC-UA connection established to {self.scada_config.host}:"
            f"{self.scada_config.port}"
        )

    async def _connect_opc_da(self) -> None:
        """Establish OPC-DA connection."""
        # In production, would use OpenOPC library
        self._opc_da_client = {
            "type": "opc_da",
            "server": self.scada_config.opc_server_name,
            "host": self.scada_config.opc_server_host,
            "connected": True,
        }
        self._logger.debug(
            f"OPC-DA connection established to {self.scada_config.opc_server_name}"
        )

    async def _connect_modbus(self) -> None:
        """Establish Modbus connection."""
        self._modbus_client = {
            "type": "modbus",
            "host": self.scada_config.host,
            "port": self.scada_config.port,
            "slave_id": self.scada_config.modbus_slave_id,
            "connected": True,
        }
        self._logger.debug(
            f"Modbus connection established to {self.scada_config.host}:"
            f"{self.scada_config.port}"
        )

    async def disconnect(self) -> None:
        """Disconnect from the SCADA system."""
        self._logger.info(f"Disconnecting from SCADA system {self.scada_config.scada_name}")

        # Cancel subscription task
        if self._subscription_task:
            self._subscription_task.cancel()
            try:
                await self._subscription_task
            except asyncio.CancelledError:
                pass
            self._subscription_task = None

        # Clear subscriptions
        self._subscriptions.clear()
        self._subscription_callbacks.clear()

        # Close connections
        self._opc_ua_client = None
        self._opc_da_client = None
        self._modbus_client = None

        self._state = ConnectionState.DISCONNECTED

        await self._audit_logger.log_operation(
            operation="disconnect",
            status="success",
        )

    async def validate_configuration(self) -> bool:
        """Validate SCADA configuration."""
        # Validate protocol-specific settings
        if self.scada_config.protocol == ProtocolType.OPC_UA:
            if not self.scada_config.opc_ua_endpoint and not self.scada_config.host:
                raise ConfigurationError(
                    "OPC-UA endpoint or host must be specified",
                    connector_id=self._config.connector_id,
                )

        elif self.scada_config.protocol == ProtocolType.OPC_DA:
            if not self.scada_config.opc_server_name:
                raise ConfigurationError(
                    "OPC-DA server name must be specified",
                    connector_id=self._config.connector_id,
                )

        self._logger.debug("Configuration validated successfully")
        return True

    async def health_check(self) -> HealthCheckResult:
        """Perform health check on the SCADA connection."""
        start_time = time.time()

        try:
            if self._state != ConnectionState.CONNECTED:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message="SCADA not connected",
                    latency_ms=0.0,
                )

            # Try to read a simple tag
            # In production, would read a known good tag
            latency_ms = (time.time() - start_time) * 1000

            # Check subscription health
            active_subs = sum(1 for s in self._subscriptions.values() if s.active)
            total_subs = len(self._subscriptions)

            # Count recent alarms
            recent_alarms = sum(
                1 for a in self._active_alarms.values()
                if a.state == AlarmState.ACTIVE
            )

            if recent_alarms > 10:
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency_ms,
                    message=f"Multiple active alarms: {recent_alarms}",
                    details={
                        "active_subscriptions": active_subs,
                        "total_subscriptions": total_subs,
                        "active_alarms": recent_alarms,
                    },
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency_ms,
                    message="SCADA operational",
                    details={
                        "active_subscriptions": active_subs,
                        "total_subscriptions": total_subs,
                        "cached_tags": len(self._tag_values),
                        "active_alarms": recent_alarms,
                    },
                )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message=f"Health check failed: {e}",
            )

    # -------------------------------------------------------------------------
    # Tag Configuration
    # -------------------------------------------------------------------------

    def register_tag(self, tag_config: TagConfig) -> None:
        """
        Register a tag configuration.

        Args:
            tag_config: Tag configuration
        """
        full_tag_name = self._get_full_tag_name(tag_config.tag_name)
        self._tag_configs[full_tag_name] = tag_config
        self._tag_history[full_tag_name] = deque(maxlen=1000)

        self._logger.debug(f"Registered tag: {full_tag_name}")

    def register_tags(self, tag_configs: List[TagConfig]) -> None:
        """
        Register multiple tag configurations.

        Args:
            tag_configs: List of tag configurations
        """
        for config in tag_configs:
            self.register_tag(config)

    def _get_full_tag_name(self, tag_name: str) -> str:
        """Get full tag name with prefix."""
        if self.scada_config.tag_prefix:
            return f"{self.scada_config.tag_prefix}{self.scada_config.tag_separator}{tag_name}"
        return tag_name

    # -------------------------------------------------------------------------
    # Tag Read Operations
    # -------------------------------------------------------------------------

    @with_retry(max_retries=3, base_delay=0.5)
    async def read_tags(self, tag_list: List[str]) -> Dict[str, TagValue]:
        """
        Read multiple tags from SCADA.

        Args:
            tag_list: List of tag names to read

        Returns:
            Dictionary of tag name to TagValue

        Raises:
            CommunicationError: If read fails
        """
        if self._state != ConnectionState.CONNECTED:
            raise CommunicationError(
                "Not connected to SCADA",
                connector_id=self._config.connector_id,
            )

        results: Dict[str, TagValue] = {}
        import random

        for tag_name in tag_list:
            full_name = self._get_full_tag_name(tag_name)

            # Get tag config if available
            tag_config = self._tag_configs.get(full_name)

            # Simulate reading (in production, would read from actual SCADA)
            if tag_config:
                # Generate value within configured range
                if tag_config.min_value is not None and tag_config.max_value is not None:
                    raw_value = random.uniform(tag_config.min_value, tag_config.max_value)
                else:
                    raw_value = random.uniform(0, 100)

                # Apply scaling
                value = raw_value * tag_config.scale_factor + tag_config.offset
                data_type = tag_config.data_type
                eu = tag_config.engineering_units
            else:
                # Default behavior
                value = random.uniform(0, 100)
                data_type = TagDataType.DOUBLE
                eu = None

            tag_value = TagValue(
                tag_name=full_name,
                value=value,
                data_type=data_type,
                quality=TagQuality.GOOD,
                timestamp=datetime.utcnow(),
                engineering_units=eu,
            )

            results[full_name] = tag_value

            # Cache the value
            self._tag_values[full_name] = tag_value

            # Add to history
            if full_name in self._tag_history:
                self._tag_history[full_name].append({
                    "timestamp": tag_value.timestamp,
                    "value": value,
                    "quality": tag_value.quality.value,
                })

        return results

    async def read_tag(self, tag_name: str) -> TagValue:
        """
        Read a single tag from SCADA.

        Args:
            tag_name: Tag name to read

        Returns:
            Tag value
        """
        results = await self.read_tags([tag_name])
        full_name = self._get_full_tag_name(tag_name)
        return results[full_name]

    # -------------------------------------------------------------------------
    # Tag Write Operations
    # -------------------------------------------------------------------------

    @with_retry(max_retries=3, base_delay=0.5)
    async def write_tag(self, tag: str, value: Any) -> bool:
        """
        Write a value to a SCADA tag.

        Args:
            tag: Tag name
            value: Value to write

        Returns:
            True if write successful

        Raises:
            CommunicationError: If write fails
        """
        if self._state != ConnectionState.CONNECTED:
            raise CommunicationError(
                "Not connected to SCADA",
                connector_id=self._config.connector_id,
            )

        full_name = self._get_full_tag_name(tag)

        # Check if tag is read-only
        tag_config = self._tag_configs.get(full_name)
        if tag_config and tag_config.read_only:
            self._logger.error(f"Cannot write to read-only tag: {full_name}")
            return False

        # Validate value range
        if tag_config:
            if tag_config.min_value is not None and value < tag_config.min_value:
                self._logger.warning(
                    f"Value {value} below min {tag_config.min_value} for tag {full_name}"
                )
                return False
            if tag_config.max_value is not None and value > tag_config.max_value:
                self._logger.warning(
                    f"Value {value} above max {tag_config.max_value} for tag {full_name}"
                )
                return False

            # Apply reverse scaling for write
            raw_value = (value - tag_config.offset) / tag_config.scale_factor
        else:
            raw_value = value

        # In production, would write to actual SCADA
        self._logger.info(f"SCADA write: {full_name} = {value} (raw: {raw_value})")

        # Update cached value
        self._tag_values[full_name] = TagValue(
            tag_name=full_name,
            value=value,
            quality=TagQuality.GOOD,
            timestamp=datetime.utcnow(),
        )

        await self._audit_logger.log_operation(
            operation="write_tag",
            status="success",
            request_data={"tag": full_name, "value": value},
        )

        return True

    async def write_tags(self, tag_values: Dict[str, Any]) -> Dict[str, bool]:
        """
        Write multiple tag values.

        Args:
            tag_values: Dictionary of tag name to value

        Returns:
            Dictionary of tag name to success status
        """
        results: Dict[str, bool] = {}
        for tag, value in tag_values.items():
            results[tag] = await self.write_tag(tag, value)
        return results

    # -------------------------------------------------------------------------
    # Subscriptions
    # -------------------------------------------------------------------------

    async def subscribe_to_tags(
        self,
        tags: List[str],
        callback: Callable[[str, TagValue], None],
        update_rate_ms: int = None,
        mode: SubscriptionMode = SubscriptionMode.ON_CHANGE,
    ) -> Subscription:
        """
        Subscribe to tag updates with callback.

        Args:
            tags: List of tag names to subscribe to
            callback: Callback function (tag_name, tag_value)
            update_rate_ms: Update rate in milliseconds
            mode: Subscription mode

        Returns:
            Subscription object

        Raises:
            ConfigurationError: If max subscriptions exceeded
        """
        if len(self._subscriptions) >= self.scada_config.max_subscriptions:
            raise ConfigurationError(
                f"Maximum subscriptions ({self.scada_config.max_subscriptions}) exceeded",
                connector_id=self._config.connector_id,
            )

        import uuid
        subscription_id = str(uuid.uuid4())

        full_tags = [self._get_full_tag_name(t) for t in tags]

        subscription = Subscription(
            subscription_id=subscription_id,
            tags=full_tags,
            update_rate_ms=update_rate_ms or self.scada_config.default_update_rate_ms,
            mode=mode,
            deadband_percent=self.scada_config.default_deadband_percent,
        )

        self._subscriptions[subscription_id] = subscription
        self._subscription_callbacks[subscription_id] = callback

        self._logger.info(
            f"Created subscription {subscription_id} for {len(tags)} tags "
            f"(rate={subscription.update_rate_ms}ms, mode={mode.value})"
        )

        return subscription

    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from tag updates.

        Args:
            subscription_id: Subscription ID to cancel

        Returns:
            True if unsubscribed successfully
        """
        if subscription_id not in self._subscriptions:
            return False

        del self._subscriptions[subscription_id]
        if subscription_id in self._subscription_callbacks:
            del self._subscription_callbacks[subscription_id]

        self._logger.info(f"Cancelled subscription {subscription_id}")
        return True

    async def _subscription_loop(self) -> None:
        """Process subscriptions and trigger callbacks."""
        while self._state == ConnectionState.CONNECTED:
            try:
                for sub_id, subscription in list(self._subscriptions.items()):
                    if not subscription.active:
                        continue

                    callback = self._subscription_callbacks.get(sub_id)
                    if not callback:
                        continue

                    # Read subscribed tags
                    tag_values = await self.read_tags(subscription.tags)

                    # Trigger callbacks
                    for tag_name, tag_value in tag_values.items():
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(tag_name, tag_value)
                            else:
                                callback(tag_name, tag_value)

                            # Update callback count
                            subscription.callback_count += 1
                        except Exception as e:
                            self._logger.error(
                                f"Subscription callback error for {tag_name}: {e}"
                            )

                # Sleep for minimum update rate
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Subscription loop error: {e}")
                await asyncio.sleep(1.0)

    # -------------------------------------------------------------------------
    # Historical Data
    # -------------------------------------------------------------------------

    async def get_historical_data(
        self,
        tags: List[str],
        start_time: datetime,
        end_time: datetime,
        resolution_seconds: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get historical data for tags.

        Args:
            tags: List of tag names
            start_time: Start time
            end_time: End time
            resolution_seconds: Optional data resolution

        Returns:
            DataFrame with historical data
        """
        if not self.scada_config.historian_enabled:
            raise ConfigurationError(
                "Historian queries not enabled",
                connector_id=self._config.connector_id,
            )

        if self._state != ConnectionState.CONNECTED:
            raise CommunicationError(
                "Not connected to SCADA",
                connector_id=self._config.connector_id,
            )

        self._logger.info(
            f"Historical query: {len(tags)} tags, "
            f"{start_time.isoformat()} to {end_time.isoformat()}"
        )

        # In production, would query actual historian
        # Simulate historical data

        full_tags = [self._get_full_tag_name(t) for t in tags]

        # Generate time series
        total_seconds = (end_time - start_time).total_seconds()
        interval = resolution_seconds or max(1, int(total_seconds / 1000))
        num_points = min(
            int(total_seconds / interval),
            self.scada_config.historian_max_points
        )

        import random

        data = {"timestamp": []}
        for tag in full_tags:
            data[tag] = []

        current_time = start_time
        for i in range(num_points):
            data["timestamp"].append(current_time)

            for tag in full_tags:
                # Generate simulated historical value
                tag_config = self._tag_configs.get(tag)
                if tag_config and tag_config.min_value is not None:
                    base = (tag_config.min_value + tag_config.max_value) / 2
                    amplitude = (tag_config.max_value - tag_config.min_value) / 4
                else:
                    base = 50.0
                    amplitude = 20.0

                # Add some pattern (sinusoidal + noise)
                import math
                pattern = math.sin(i / 100.0 * 2 * math.pi)
                value = base + amplitude * pattern + random.uniform(-amplitude * 0.1, amplitude * 0.1)
                data[tag].append(value)

            current_time += timedelta(seconds=interval)

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)

        self._logger.info(f"Historical query returned {len(df)} points")

        return df

    # -------------------------------------------------------------------------
    # Alarm Management
    # -------------------------------------------------------------------------

    async def send_alarm(self, alarm_data: AlarmData) -> AlarmSendResult:
        """
        Send an alarm to the SCADA system.

        Args:
            alarm_data: Alarm data to send

        Returns:
            Alarm send result
        """
        if self._state != ConnectionState.CONNECTED:
            return AlarmSendResult(
                success=False,
                alarm_id=alarm_data.alarm_id,
                error_message="Not connected to SCADA",
            )

        try:
            # In production, would send to actual SCADA alarm system

            # Store in active alarms
            self._active_alarms[alarm_data.alarm_id] = alarm_data

            # Add to history
            self._alarm_history.append(alarm_data)

            self._logger.info(
                f"Alarm sent: {alarm_data.alarm_id} - {alarm_data.message} "
                f"(priority={alarm_data.priority.value})"
            )

            await self._audit_logger.log_operation(
                operation="send_alarm",
                status="success",
                request_data={
                    "alarm_id": alarm_data.alarm_id,
                    "tag_name": alarm_data.tag_name,
                    "message": alarm_data.message,
                    "priority": alarm_data.priority.value,
                },
            )

            return AlarmSendResult(
                success=True,
                alarm_id=alarm_data.alarm_id,
            )

        except Exception as e:
            self._logger.error(f"Failed to send alarm: {e}")
            return AlarmSendResult(
                success=False,
                alarm_id=alarm_data.alarm_id,
                error_message=str(e),
            )

    async def acknowledge_alarm(
        self,
        alarm_id: str,
        user_id: str,
    ) -> bool:
        """
        Acknowledge an alarm.

        Args:
            alarm_id: Alarm ID to acknowledge
            user_id: User acknowledging the alarm

        Returns:
            True if acknowledged successfully
        """
        if alarm_id not in self._active_alarms:
            return False

        alarm = self._active_alarms[alarm_id]

        # Create updated alarm with acknowledgment
        acknowledged_alarm = AlarmData(
            alarm_id=alarm.alarm_id,
            tag_name=alarm.tag_name,
            message=alarm.message,
            priority=alarm.priority,
            value=alarm.value,
            setpoint=alarm.setpoint,
            state=AlarmState.ACKNOWLEDGED,
            timestamp=alarm.timestamp,
            acknowledged_by=user_id,
            acknowledged_at=datetime.utcnow(),
        )

        self._active_alarms[alarm_id] = acknowledged_alarm

        self._logger.info(f"Alarm {alarm_id} acknowledged by {user_id}")

        await self._audit_logger.log_operation(
            operation="acknowledge_alarm",
            status="success",
            user_id=user_id,
            request_data={"alarm_id": alarm_id},
        )

        return True

    async def clear_alarm(self, alarm_id: str) -> bool:
        """
        Clear an alarm.

        Args:
            alarm_id: Alarm ID to clear

        Returns:
            True if cleared successfully
        """
        if alarm_id not in self._active_alarms:
            return False

        alarm = self._active_alarms[alarm_id]

        # Create cleared alarm for history
        cleared_alarm = AlarmData(
            alarm_id=alarm.alarm_id,
            tag_name=alarm.tag_name,
            message=alarm.message,
            priority=alarm.priority,
            value=alarm.value,
            setpoint=alarm.setpoint,
            state=AlarmState.CLEARED,
            timestamp=alarm.timestamp,
            acknowledged_by=alarm.acknowledged_by,
            acknowledged_at=alarm.acknowledged_at,
        )

        # Remove from active and add to history
        del self._active_alarms[alarm_id]
        self._alarm_history.append(cleared_alarm)

        self._logger.info(f"Alarm {alarm_id} cleared")

        return True

    def get_active_alarms(self) -> List[AlarmData]:
        """Get list of active alarms."""
        return list(self._active_alarms.values())

    def get_alarm_history(self, hours: int = 24) -> List[AlarmData]:
        """Get alarm history for specified hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [a for a in self._alarm_history if a.timestamp > cutoff]


# =============================================================================
# Factory Function
# =============================================================================


def create_scada_connector(
    scada_id: str,
    scada_name: str,
    host: str,
    port: int = 4840,
    protocol: ProtocolType = ProtocolType.OPC_UA,
    vendor: SCADAVendor = SCADAVendor.GENERIC,
    **kwargs,
) -> SCADAConnector:
    """
    Factory function to create a SCADA connector.

    Args:
        scada_id: Unique SCADA system identifier
        scada_name: SCADA system name
        host: SCADA server host
        port: Port number
        protocol: Communication protocol
        vendor: SCADA vendor
        **kwargs: Additional configuration options

    Returns:
        Configured SCADAConnector instance
    """
    config = SCADAConnectorConfig(
        connector_name=f"SCADA_{scada_name}",
        scada_id=scada_id,
        scada_name=scada_name,
        host=host,
        port=port,
        protocol=protocol,
        scada_vendor=vendor,
        **kwargs,
    )

    return SCADAConnector(config)
